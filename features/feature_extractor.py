"""
Feature Extractor Module
Pipeline position: IPM World Coords → [Feature Engineering] → LightGBM

Tại sao Feature Engineering là core của hệ thống?
──────────────────────────────────────────────────
Hybrid AI principle: thay vì để DL tự học từ pixel (3D-CNN style),
ta ENCODE prior knowledge vật lý vào feature vector:

  State vector per vehicle tại frame t:
  f_t^i = [v_t, a_t, θ_t, Δθ_t, d_min_t]

  Trong đó:
  - v_t    : vận tốc tức thời (m/s)     ← từ IPM + sai phân hữu hạn
  - a_t    : gia tốc tức thời (m/s²)    ← Δv/Δt
  - θ_t    : góc nghiêng cơ thể (độ)   ← từ OpenPose keypoints
  - Δθ_t   : độ biến thiên góc nghiêng  ← θ_t - θ_{t-1}
  - d_min  : khoảng cách đến xe gần nhất (m) ← từ IPM

  Sliding Window k=25 frames → Statistical Aggregation:
  [mean, std, max, min] × 5 features = 20-dim vector

  Đây là EXACTLY những gì vật lý của một vụ tai nạn thể hiện:
  - Gia tốc âm đột ngột (phanh gấp / va chạm)
  - Góc nghiêng lớn (mất thăng bằng / ngã)
  - Khoảng cách đến xe khác giảm đột ngột (va chạm)

Tham chiếu báo cáo: Section 3.4.1
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque

from geometry.ipm import IPMTransformer, WorldPoint
from config.settings import get_config
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VehicleState:
    """
    Trạng thái vật lý của một phương tiện tại frame t.

    Tất cả đơn vị đã được chuẩn hóa về đơn vị vật lý thực tế.
    """
    track_id: int
    frame_id: int
    timestamp: float           # giây

    # Tọa độ thực tế (mét)
    world_x: float
    world_y: float

    # Động học
    speed: float               # m/s — vận tốc tức thời
    acceleration: float        # m/s² — gia tốc tức thời
    heading_angle: float       # độ — hướng di chuyển (0=bắc, 90=đông)

    # Tư thế người lái (từ OpenPose)
    body_lean_angle: float     # độ — góc nghiêng cơ thể (0=thẳng đứng)
    body_lean_delta: float     # độ/frame — tốc độ thay đổi góc nghiêng

    # Tương tác
    min_dist_to_others: float  # mét — khoảng cách đến xe gần nhất

    def as_feature_vector(self) -> np.ndarray:
        """
        Chuyển state → 5-dim feature vector theo báo cáo:
        f_t^i = [v_t, a_t, θ_t, Δθ_t, d_min]
        """
        return np.array([
            self.speed,
            self.acceleration,
            self.body_lean_angle,
            self.body_lean_delta,
            self.min_dist_to_others,
        ], dtype=np.float32)

    def is_valid(self) -> bool:
        """Kiểm tra state có hợp lệ không (không có NaN/inf)."""
        vec = self.as_feature_vector()
        return bool(np.all(np.isfinite(vec)))


class VehicleStateBuffer:
    """
    Buffer lưu lịch sử VehicleState cho một xe cụ thể.

    Nhiệm vụ:
    1. Lưu sliding window 25 frame gần nhất
    2. Tính toán sai phân hữu hạn (v, a)
    3. Cung cấp window đủ lớn để Feature Aggregator xử lý

    Design: deque với maxlen → tự động loại bỏ state cũ nhất
    """

    def __init__(self, track_id: int, fps: float = 25.0, window_size: int = 25):
        self.track_id = track_id
        self.fps = fps
        self.dt = 1.0 / fps           # giây per frame
        self.window_size = window_size

        self._history: deque = deque(maxlen=window_size * 3)
        # × 3 để có buffer cho sliding stride
        self._prev_world: Optional[WorldPoint] = None
        self._prev_speed: Optional[float] = None
        self._prev_lean: Optional[float] = None

    def push(
        self,
        frame_id: int,
        timestamp: float,
        world_pos: WorldPoint,
        body_lean_angle: float = 0.0,
        all_world_positions: Optional[List[WorldPoint]] = None
    ) -> Optional[VehicleState]:
        """
        Thêm observation mới, tính toán derivative features.

        Args:
            frame_id: ID frame hiện tại
            timestamp: thời gian (giây)
            world_pos: tọa độ thực tế từ IPM
            body_lean_angle: góc nghiêng từ OpenPose (độ)
            all_world_positions: tọa độ tất cả xe trong frame (để tính d_min)

        Returns:
            VehicleState nếu đủ dữ liệu, None nếu frame đầu tiên
        """
        # Tính vận tốc (sai phân hữu hạn bậc 1)
        if self._prev_world is not None:
            dx = world_pos.x - self._prev_world.x
            dy = world_pos.y - self._prev_world.y
            dist = np.sqrt(dx**2 + dy**2)
            speed = dist / self.dt  # m/s

            # Hướng di chuyển (góc so với trục Y+)
            heading = np.degrees(np.arctan2(dx, dy)) % 360.0
        else:
            speed = 0.0
            heading = 0.0

        # Tính gia tốc (sai phân bậc 2)
        if self._prev_speed is not None:
            accel = (speed - self._prev_speed) / self.dt  # m/s²
        else:
            accel = 0.0

        # Tính tốc độ thay đổi góc nghiêng
        if self._prev_lean is not None:
            lean_delta = (body_lean_angle - self._prev_lean) / self.dt
        else:
            lean_delta = 0.0

        # Tính khoảng cách đến xe gần nhất
        min_dist = self._compute_min_distance(
            world_pos, all_world_positions
        )

        state = VehicleState(
            track_id=self.track_id,
            frame_id=frame_id,
            timestamp=timestamp,
            world_x=world_pos.x,
            world_y=world_pos.y,
            speed=float(np.clip(speed, 0, 50)),         # clip nhiễu
            acceleration=float(np.clip(accel, -15, 15)), # vật lý hợp lý
            heading_angle=heading,
            body_lean_angle=body_lean_angle,
            body_lean_delta=float(np.clip(lean_delta, -180, 180)),
            min_dist_to_others=min_dist,
        )

        # Smooth IPM jitter: apply EMA trên speed/accel
        if len(self._history) > 0:
            last = self._history[-1]
            alpha = 0.7  # EMA weight cho giá trị hiện tại
            state.speed = alpha * state.speed + (1 - alpha) * last.speed
            state.acceleration = (
                alpha * state.acceleration
                + (1 - alpha) * last.acceleration
            )

        self._history.append(state)
        self._prev_world = world_pos
        self._prev_speed = speed
        self._prev_lean = body_lean_angle

        return state if len(self._history) > 1 else None

    def _compute_min_distance(
        self,
        my_pos: WorldPoint,
        others: Optional[List[WorldPoint]]
    ) -> float:
        """Tính khoảng cách đến xe gần nhất trong không gian thực tế."""
        if not others:
            return 99.0  # Không có xe khác → distance lớn

        min_d = 99.0
        for pos in others:
            # Bỏ qua chính mình (cùng tọa độ)
            if abs(pos.x - my_pos.x) < 0.01 and abs(pos.y - my_pos.y) < 0.01:
                continue
            d = my_pos.distance_to(pos)
            if d < min_d:
                min_d = d

        return float(min_d)

    def get_window(self, size: int = 25) -> Optional[List[VehicleState]]:
        """
        Lấy window 'size' frame gần nhất.
        Trả về None nếu chưa đủ data.
        """
        if len(self._history) < size:
            return None
        return list(self._history)[-size:]

    @property
    def history_len(self) -> int:
        return len(self._history)


class FeatureAggregator:
    """
    Statistical Temporal Aggregation.

    Chuyển window (k=25 frames) × (5 features) → 20-dim flat vector
    bằng cách tính [mean, std, max, min] cho mỗi feature.

    Lý do dùng statistical aggregation thay vì raw sequence:
    - LightGBM không xử lý được 3D input (N, T, F)
    - Statistical features capture được pattern của sự cố:
      * mean(a) âm → phanh kéo dài
      * max(|θ|) lớn → ngã xe
      * min(d_min) nhỏ → va chạm gần
    - Không cần LSTM/RNN → inference nhanh hơn nhiều

    Tham chiếu báo cáo: Section 3.4.2 — "Statistical Temporal Aggregation"
    """

    FEATURE_NAMES = [
        "speed_mean", "speed_std", "speed_max", "speed_min",
        "accel_mean", "accel_std", "accel_max", "accel_min",
        "lean_mean", "lean_std", "lean_max", "lean_min",
        "lean_delta_mean", "lean_delta_std", "lean_delta_max", "lean_delta_min",
        "dist_mean", "dist_std", "dist_max", "dist_min",
    ]  # 5 features × 4 stats = 20 features

    FEATURE_DIM = 20

    @staticmethod
    def aggregate(window: List[VehicleState]) -> np.ndarray:
        """
        Args:
            window: List[VehicleState] với len = window_size

        Returns:
            np.ndarray shape (20,) — feature vector cho LightGBM
        """
        if not window:
            return np.zeros(FeatureAggregator.FEATURE_DIM, dtype=np.float32)

        # Stack thành matrix (T, 5)
        matrix = np.array(
            [s.as_feature_vector() for s in window],
            dtype=np.float32
        )  # shape (k, 5)

        if not np.all(np.isfinite(matrix)):
            # Replace NaN/inf với giá trị an toàn
            matrix = np.nan_to_num(matrix, nan=0.0, posinf=50.0, neginf=-15.0)

        # Tính 4 thống kê cho mỗi trong 5 features
        means = np.mean(matrix, axis=0)   # (5,)
        stds  = np.std(matrix, axis=0)    # (5,)
        maxs  = np.max(matrix, axis=0)    # (5,)
        mins  = np.min(matrix, axis=0)    # (5,)

        # Interleave: [mean0, std0, max0, min0, mean1, std1, ...]
        feature_vec = np.empty(20, dtype=np.float32)
        for i in range(5):
            feature_vec[i*4 + 0] = means[i]
            feature_vec[i*4 + 1] = stds[i]
            feature_vec[i*4 + 2] = maxs[i]
            feature_vec[i*4 + 3] = mins[i]

        return feature_vec

    @staticmethod
    def aggregate_batch(
        windows: List[List[VehicleState]]
    ) -> np.ndarray:
        """
        Batch aggregation cho training.

        Args:
            windows: List of windows, mỗi window là List[VehicleState]

        Returns:
            np.ndarray shape (N, 20)
        """
        return np.array(
            [FeatureAggregator.aggregate(w) for w in windows],
            dtype=np.float32
        )


class FeatureExtractor:
    """
    Orchestrator: quản lý state buffer cho tất cả active tracks.

    Nhiệm vụ:
    1. Nhận tracks từ DeepSORT + world positions từ IPM
    2. Cập nhật VehicleStateBuffer cho từng track
    3. Khi đủ window → trả về feature vector cho classifier
    4. Tự động cleanup buffer khi track expired
    """

    def __init__(self, config=None):
        self.cfg = config or get_config()
        feat_cfg = self.cfg.features

        self.window_size: int = feat_cfg.window_size      # 25
        self.window_stride: int = feat_cfg.window_stride  # 5
        self.fps: float = self.cfg.video.fps              # 25

        # track_id → VehicleStateBuffer
        self._buffers: Dict[int, VehicleStateBuffer] = {}

        # Frame counter (để trigger sliding window)
        self._frame_id: int = 0

        logger.info(
            f"FeatureExtractor init | "
            f"window={self.window_size} | stride={self.window_stride}"
        )

    def update(
        self,
        tracks: list,                           # List[Track] từ DeepSORT
        world_positions: Dict[int, WorldPoint],  # track_id → WorldPoint
        body_leans: Optional[Dict[int, float]] = None,  # track_id → angle
        frame_id: Optional[int] = None
    ) -> Dict[int, np.ndarray]:
        """
        Cập nhật state cho tất cả active tracks.

        Args:
            tracks: active tracks từ DeepSORT
            world_positions: dict track_id → WorldPoint (từ IPM)
            body_leans: dict track_id → góc nghiêng từ OpenPose (None = dùng 0)
            frame_id: ID frame hiện tại

        Returns:
            Dict[track_id → 20-dim feature vector]
            Chỉ trả về những track đã đủ window_size frames
        """
        if frame_id is not None:
            self._frame_id = frame_id
        else:
            self._frame_id += 1

        timestamp = self._frame_id / self.fps

        # Lấy tất cả world positions trong frame (để tính min_distance)
        all_positions = list(world_positions.values())

        # Đảm bảo buffer tồn tại cho mỗi active track
        active_ids = set()
        for track in tracks:
            tid = track.track_id
            active_ids.add(tid)

            if tid not in self._buffers:
                self._buffers[tid] = VehicleStateBuffer(
                    track_id=tid,
                    fps=self.fps,
                    window_size=self.window_size
                )

        # Cập nhật state
        for track in tracks:
            tid = track.track_id
            if tid not in world_positions:
                continue  # Không có IPM data cho track này

            buf = self._buffers[tid]
            lean = (body_leans or {}).get(tid, 0.0)

            # others = tất cả xe khác (không phải xe này)
            others = [
                pos for t_id, pos in world_positions.items()
                if t_id != tid
            ]

            buf.push(
                frame_id=self._frame_id,
                timestamp=timestamp,
                world_pos=world_positions[tid],
                body_lean_angle=lean,
                all_world_positions=others if others else None
            )

        # Cleanup expired tracks
        expired = set(self._buffers.keys()) - active_ids
        for tid in expired:
            del self._buffers[tid]

        # Trích xuất feature vectors nếu đủ window
        features: Dict[int, np.ndarray] = {}

        # Chỉ trigger tại stride frames (giảm tải CPU)
        should_extract = (self._frame_id % self.window_stride == 0)
        if not should_extract:
            return features

        for tid, buf in self._buffers.items():
            window = buf.get_window(self.window_size)
            if window is None:
                continue  # Chưa đủ data

            feat_vec = FeatureAggregator.aggregate(window)
            features[tid] = feat_vec

        return features

    @property
    def active_track_count(self) -> int:
        return len(self._buffers)

    def get_buffer(self, track_id: int) -> Optional[VehicleStateBuffer]:
        return self._buffers.get(track_id)

    def reset(self):
        """Reset tất cả buffers — dùng khi đổi video source."""
        self._buffers.clear()
        self._frame_id = 0
        logger.info("FeatureExtractor reset.")