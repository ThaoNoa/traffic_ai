"""
DeepSORT Tracker Module
Pipeline: YOLOv8 Detections → [DeepSORT] → Active Tracks

Customizations so với DeepSORT gốc (từ báo cáo Section 3.3.2):
  - max_age = 15  (gốc: 30) → chịu được occlusion 1.4 giây
  - n_init = 3              → xác nhận track sau 3 frame liên tiếp
  - λ = 0.7                 → ưu tiên Mahalanobis, giảm phụ thuộc Cosine
  - Tentative state tracking → recover ID sau occlusion
"""

from __future__ import annotations

import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from collections import deque

from deep_sort_realtime.deepsort_tracker import DeepSort

from detector.vehicle_detector import Detection
from config.settings import get_config
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Track:
    """
    Một phương tiện đang được theo dõi.

    track_id   : ID duy nhất, giữ xuyên suốt video kể cả khi bị khuất
    bbox_xyxy  : vị trí hiện tại [x1, y1, x2, y2]
    class_name : "motorcycle", "car", ...
    confidence : confidence của detection gần nhất
    age        : số frame track này tồn tại
    hits       : số lần được detect thành công
    time_since_update : số frame kể từ lần detect cuối
    is_confirmed : True nếu đã qua n_init frame
    trajectory : lịch sử bottom_center (để vẽ và feature extract)
    """
    track_id: int
    bbox_xyxy: np.ndarray
    class_name: str
    confidence: float
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    is_confirmed: bool = False
    trajectory: deque = field(default_factory=lambda: deque(maxlen=30))

    @property
    def bottom_center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, float(y2))

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def is_motorcycle(self) -> bool:
        return self.class_name == "motorcycle"

    @property
    def bbox_wh(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x2 - x1, y2 - y1)

    def update_trajectory(self):
        """Thêm vị trí hiện tại vào lịch sử quỹ đạo."""
        self.trajectory.append(self.bottom_center)

    def __repr__(self) -> str:
        cx, cy = self.center
        return (
            f"Track(id={self.track_id} | {self.class_name} | "
            f"center=({cx:.0f},{cy:.0f}) | "
            f"age={self.age} | hits={self.hits} | "
            f"confirmed={self.is_confirmed})"
        )


class DeepSORTTracker:
    """
    Wrapper quanh deep_sort_realtime với các tinh chỉnh
    cho môi trường xe máy Lĩnh Nam.

    Key design decisions:
    - max_cosine_distance = 0.4  điều khiển gián tiếp λ
      (thấp hơn = khắt khe hơn với Cosine → ưu tiên Mahalanobis)
    - max_age = 15             giữ track qua occlusion dài
    - embedder chạy trên GPU nếu có
    """

    def __init__(self, config=None):
        self.cfg = config or get_config()
        trk_cfg = self.cfg.tracker

        self.max_age: int = trk_cfg.max_age            # 60
        self.n_init: int = trk_cfg.n_init              # 3
        self.max_cosine_distance: float = trk_cfg.max_cosine_distance  # 0.4
        self.nn_budget: int = trk_cfg.nn_budget        # 100

        # λ=0.7: điều khiển qua max_cosine_distance
        # max_cosine_distance thấp → cosine bị reject nhiều hơn
        # → hệ thống phụ thuộc vào Mahalanobis nhiều hơn
        self.lambda_motion: float = trk_cfg.lambda_motion  # 0.7

        self._tracker: Optional[DeepSort] = None

        # State tracking nội bộ
        self._active_tracks: Dict[int, Track] = {}
        self._frame_count: int = 0

        # Thống kê
        self._total_id_switches: int = 0
        self._total_tracks_created: int = 0

        logger.info(
            f"DeepSORTTracker init | "
            f"max_age={self.max_age} | "
            f"n_init={self.n_init} | "
            f"max_cosine_dist={self.max_cosine_distance} | "
            f"λ_motion={self.lambda_motion}"
        )

    def initialize(self) -> "DeepSORTTracker":
        """
        Khởi tạo DeepSort engine.
        Gọi một lần sau khi VideoStream.start().
        """
        use_cuda = (self.cfg.system.device == "cuda")

        self._tracker = DeepSort(
            max_age=self.max_age,
            n_init=self.n_init,
            max_cosine_distance=self.max_cosine_distance,
            max_iou_distance=0.9,
            nn_budget=self.nn_budget,
            # embedder chạy trên GPU để tăng tốc ReID
            embedder="mobilenet",
            half=use_cuda,           # FP16 nếu có GPU → tiết kiệm VRAM
            bgr=True,                # OpenCV dùng BGR
            embedder_gpu=use_cuda,
        )

        logger.info(
            f"DeepSort engine ready | "
            f"GPU embedder: {use_cuda}"
        )
        return self

    def update(
        self,
        detections: List[Detection],
        frame: np.ndarray
    ) -> List[Track]:

        if self._tracker is None:
            raise RuntimeError("Tracker chưa được khởi tạo.")

        self._frame_count += 1

        # Chuyển Detection → format DeepSort [x1, y1, w, h]
        raw_detections = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            w = x2 - x1
            h = y2 - y1
            raw_detections.append((
                [float(x1), float(y1), float(w), float(h)],
                det.confidence,
                det.class_name
            ))

        ds_tracks = self._tracker.update_tracks(raw_detections, frame=frame)
        print("tracks:", len(ds_tracks))

        active_tracks: List[Track] = []
        current_ids = set()

        for ds_track in ds_tracks:
            if not ds_track.is_confirmed():
                continue

            # ✅ XÓA dòng: if ds_track.time_since_update > 1: continue
            # Dòng đó giết occlusion recovery

            track_id = int(ds_track.track_id)
            current_ids.add(track_id)

            ltrb = ds_track.to_ltrb()
            if ltrb is None:
                continue

            x1, y1, x2, y2 = ltrb
            if x2 <= x1 or y2 <= y1:
                continue
            if (x2 - x1) > frame.shape[1] * 0.8:
                continue
            if (y2 - y1) > frame.shape[0] * 0.8:
                continue

            bbox = np.array(ltrb, dtype=np.float32)
            class_name = ds_track.get_det_class() or "unknown"
            confidence = float(
                ds_track.get_det_conf()
                if ds_track.get_det_conf() is not None
                else 1.0
            )

            if track_id in self._active_tracks:
                track = self._active_tracks[track_id]
                track.bbox_xyxy = bbox
                track.class_name = class_name
                track.confidence = confidence
                track.age += 1
                track.hits += 1
                track.time_since_update = 0
                track.is_confirmed = True
            else:
                self._total_tracks_created += 1
                track = Track(
                    track_id=track_id,
                    bbox_xyxy=bbox,
                    class_name=class_name,
                    confidence=confidence,
                    age=0,
                    hits=1,
                    time_since_update=0,
                    is_confirmed=True,
                )
                self._active_tracks[track_id] = track
                logger.debug(f"New track: ID={track_id} | {class_name}")

            track.update_trajectory()
            active_tracks.append(track)

        # ✅ Fix bug: collect IDs to delete trước, xóa sau
        lost_ids = set(self._active_tracks.keys()) - current_ids
        ids_to_delete = []
        for tid in lost_ids:
            self._active_tracks[tid].time_since_update += 1
            if self._active_tracks[tid].time_since_update > self.max_age:
                ids_to_delete.append(tid)
                logger.debug(f"Track expired: ID={tid}")

        for tid in ids_to_delete:
            del self._active_tracks[tid]

        return active_tracks

    @property
    def active_count(self) -> int:
        return len(self._active_tracks)

    @property
    def stats(self) -> dict:
        return {
            "frames_processed": self._frame_count,
            "total_tracks_created": self._total_tracks_created,
            "currently_active": self.active_count,
        }

    def reset(self):
        """Reset tracker — dùng khi đổi video source."""
        if self._tracker:
            self._tracker = None
        self._active_tracks.clear()
        self._frame_count = 0
        self.initialize()
        logger.info("Tracker reset.")