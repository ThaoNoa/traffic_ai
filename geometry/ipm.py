"""
IPM — Inverse Perspective Mapping
Pipeline position: DeepSORT Tracks → [IPM] → Real-World Coords → Feature Engineering

Tại sao cần IPM?
─────────────────
Camera nhìn từ trên cao → ảnh bị méo phối cảnh (perspective distortion):
  • Xe ở xa trông nhỏ hơn xe ở gần dù kích thước thật giống nhau
  • Tốc độ tính bằng pixel/s bị sai lệch tùy vùng ảnh

IPM chiếu ảnh sang Bird's-Eye View (BEV) → mặt phẳng đường thực tế.
Sau IPM:
  • 1 pixel BEV = N cm thực tế (calibrated)
  • Vận tốc đo được bằng m/s — có ý nghĩa vật lý thực
  • Khoảng cách giữa xe đo được bằng mét

Toán học:
─────────
  s·[u, v, 1]ᵀ = H·[X, Y, 1]ᵀ

  H ∈ R(3×3): ma trận Homography, ước lượng từ 4 cặp điểm tương ứng
  (u, v): tọa độ pixel trên ảnh gốc
  (X, Y): tọa độ thực tế trên mặt phẳng đường (mét)

Tham chiếu báo cáo: Section 2.2.1, 3.4.1
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from config.settings import get_config
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class WorldPoint:
    """
    Tọa độ thực tế sau khi qua IPM.

    x, y: mét (gốc tọa độ = góc trên-trái của vùng quan sát)
    pixel_u, pixel_v: tọa độ gốc trên ảnh (để debug / visualize)
    """
    x: float   # mét, trục ngang
    y: float   # mét, trục dọc (hướng xa camera)
    pixel_u: float
    pixel_v: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)

    def distance_to(self, other: "WorldPoint") -> float:
        return float(np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2))

    def __repr__(self) -> str:
        return f"WorldPoint(x={self.x:.2f}m, y={self.y:.2f}m)"


class IPMTransformer:
    """
    Thực hiện phép biến đổi phối cảnh ngược (IPM).

    Flow:
        pixel (u,v) ──[H⁻¹]──→ BEV pixel (u_bev, v_bev) ──[scale]──→ world (X,Y) mét

    Key design decisions:
    - Dùng cv2.getPerspectiveTransform() — chính xác hơn DLT thuần túy
    - Lưu cả H và H_inv để có thể chiếu 2 chiều (dùng cho visualize BEV)
    - calibrate() tách biệt khỏi __init__ → có thể recalibrate khi đổi camera
    """

    def __init__(self, config=None):
        self.cfg = config or get_config()
        ipm_cfg = self.cfg.ipm

        # src_points: 4 điểm trên ảnh gốc (pixel)
        # Thứ tự: top-left, top-right, bottom-right, bottom-left
        self.src_points = np.array(
            ipm_cfg.src_points, dtype=np.float32
        )  # shape (4, 2)

        # dst_points: 4 điểm tương ứng trong không gian thực (mét)
        self.dst_points_m = np.array(
            ipm_cfg.dst_points, dtype=np.float32
        )  # shape (4, 2)

        # BEV image size (pixels)
        self.bev_w: int = ipm_cfg.bev_width
        self.bev_h: int = ipm_cfg.bev_height

        # Scale: pixels per meter trong BEV
        self.scale_x: float = ipm_cfg.scale_x   # px/m
        self.scale_y: float = ipm_cfg.scale_y   # px/m

        # Matrices (sẽ được tính trong calibrate())
        self.H: Optional[np.ndarray] = None      # pixel → BEV pixel
        self.H_inv: Optional[np.ndarray] = None  # BEV pixel → pixel

        # Pixel-to-meter scale (dùng để chuyển BEV pixel → mét)
        self._m_per_px_x = 1.0 / self.scale_x
        self._m_per_px_y = 1.0 / self.scale_y

        self._calibrated = False

        logger.info(
            f"IPMTransformer init | "
            f"BEV size: {self.bev_w}×{self.bev_h} | "
            f"scale: {self.scale_x:.1f}px/m (X), {self.scale_y:.1f}px/m (Y)"
        )

    def calibrate(self) -> "IPMTransformer":
        """
        Tính ma trận Homography H từ 4 cặp điểm.

        dst_points_px: chuyển dst_points_m (mét) → pixel trong BEV image
        bằng cách nhân với scale.
        """
        # Chuyển điểm thực tế (mét) → pixel trong BEV image
        dst_points_px = self.dst_points_m.copy()
        dst_points_px[:, 0] *= self.scale_x   # X mét → BEV pixel col
        dst_points_px[:, 1] *= self.scale_y   # Y mét → BEV pixel row

        self.H = cv2.getPerspectiveTransform(
            self.src_points.astype(np.float32),
            dst_points_px.astype(np.float32)
        )
        self.H_inv = np.linalg.inv(self.H)

        self._calibrated = True

        logger.info("IPM calibrated.")
        logger.debug(f"Homography H:\n{self.H}")

        return self

    def pixel_to_world(
        self,
        u: float,
        v: float
    ) -> WorldPoint:
        """
        Chiếu một điểm pixel (u, v) → WorldPoint (X, Y) tính bằng mét.

        Args:
            u: pixel column (x trên ảnh)
            v: pixel row (y trên ảnh)

        Returns:
            WorldPoint với tọa độ thực tế

        Lý do dùng bottom_center:
            Điểm giữa cạnh dưới bbox ≈ vị trí bánh xe chạm mặt đường.
            Đây là điểm duy nhất thực sự nằm trên mặt phẳng đường → IPM đúng.
            Nếu dùng tâm bbox, điểm chiếu sẽ bị sai (xe có chiều cao).
        """
        if not self._calibrated:
            raise RuntimeError("IPM chưa calibrate. Gọi calibrate() trước.")

        # Tạo homogeneous point
        pt = np.array([[[u, v]]], dtype=np.float32)  # shape (1,1,2)

        # Áp dụng perspective transform
        pt_bev = cv2.perspectiveTransform(pt, self.H)  # shape (1,1,2)
        u_bev, v_bev = pt_bev[0, 0]

        # Chuyển BEV pixel → mét
        x_m = float(u_bev) * self._m_per_px_x
        y_m = float(v_bev) * self._m_per_px_y

        return WorldPoint(x=x_m, y=y_m, pixel_u=u, pixel_v=v)

    def pixels_to_world_batch(
        self,
        points: List[Tuple[float, float]]
    ) -> List[WorldPoint]:
        """
        Batch version — hiệu quả hơn khi chiếu nhiều điểm cùng lúc.

        Args:
            points: List of (u, v) pixel coords

        Returns:
            List[WorldPoint]
        """
        if not points:
            return []

        if not self._calibrated:
            raise RuntimeError("IPM chưa calibrate.")

        # Chuẩn bị batch input cho perspectiveTransform
        pts_arr = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        pts_bev = cv2.perspectiveTransform(pts_arr, self.H)  # (N, 1, 2)

        results = []
        for i, (u, v) in enumerate(points):
            u_bev, v_bev = pts_bev[i, 0]
            x_m = float(u_bev) * self._m_per_px_x
            y_m = float(v_bev) * self._m_per_px_y
            results.append(WorldPoint(x=x_m, y=y_m, pixel_u=u, pixel_v=v))

        return results

    def world_to_pixel(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """
        Chiếu ngược: WorldPoint → pixel trên ảnh gốc.
        Dùng cho visualize (vẽ grid thực tế lên ảnh gốc).
        """
        if not self._calibrated:
            raise RuntimeError("IPM chưa calibrate.")

        # Mét → BEV pixel
        u_bev = x_m * self.scale_x
        v_bev = y_m * self.scale_y

        pt = np.array([[[u_bev, v_bev]]], dtype=np.float32)
        pt_img = cv2.perspectiveTransform(pt, self.H_inv)
        u_img, v_img = pt_img[0, 0]

        return (int(round(u_img)), int(round(v_img)))

    def get_bev_image(self, frame: np.ndarray) -> np.ndarray:
        """
        Tạo Bird's-Eye View image từ frame gốc.
        Dùng cho debug / visualization.
        """
        if not self._calibrated:
            raise RuntimeError("IPM chưa calibrate.")

        bev = cv2.warpPerspective(
            frame, self.H,
            (self.bev_w, self.bev_h),
            flags=cv2.INTER_LINEAR
        )
        return bev

    def draw_calibration_points(self, frame: np.ndarray) -> np.ndarray:
        """
        Vẽ 4 điểm calibration lên frame để verify.
        """
        vis = frame.copy()
        colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0)]
        labels = ["TL","TR","BR","BL"]

        for i, (pt, color, label) in enumerate(
            zip(self.src_points, colors, labels)
        ):
            u, v = int(pt[0]), int(pt[1])
            cv2.circle(vis, (u, v), 8, color, -1)
            cv2.putText(vis, label, (u+10, v),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return vis

    def draw_real_world_grid(
        self,
        frame: np.ndarray,
        grid_spacing_m: float = 5.0
    ) -> np.ndarray:
        """
        Vẽ lưới thực tế (đơn vị mét) lên ảnh gốc để verify calibration.
        Mỗi ô lưới = grid_spacing_m mét thực tế.
        """
        if not self._calibrated:
            return frame

        vis = frame.copy()
        max_x = self.dst_points_m[:, 0].max()
        max_y = self.dst_points_m[:, 1].max()

        # Vẽ đường dọc (X cố định)
        x = 0.0
        while x <= max_x:
            pts_world = [(x, y) for y in np.arange(0, max_y, 0.5)]
            pts_img = [self.world_to_pixel(wx, wy) for wx, wy in pts_world]
            for i in range(1, len(pts_img)):
                cv2.line(vis, pts_img[i-1], pts_img[i], (200, 200, 0), 1)
            if x > 0:
                mid = self.world_to_pixel(x, max_y / 2)
                cv2.putText(vis, f"{x:.0f}m", mid,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,0), 1)
            x += grid_spacing_m

        # Vẽ đường ngang (Y cố định)
        y = 0.0
        while y <= max_y:
            pts_world = [(x, y) for x in np.arange(0, max_x, 0.5)]
            pts_img = [self.world_to_pixel(wx, wy) for wx, wy in pts_world]
            for i in range(1, len(pts_img)):
                cv2.line(vis, pts_img[i-1], pts_img[i], (0, 200, 200), 1)
            if y > 0:
                mid = self.world_to_pixel(max_x / 2, y)
                cv2.putText(vis, f"{y:.0f}m", mid,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,200,200), 1)
            y += grid_spacing_m

        return vis

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated