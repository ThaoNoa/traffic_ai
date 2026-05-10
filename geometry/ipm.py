"""
IPM — Inverse Perspective Mapping (Production Grade)
Pipeline: Frame → ROI → Homography → BEV → World Coords

FIXES:
- Correct OpenCV point order: [TL, TR, BL, BR]
- H_adjusted using matrix multiplication H @ T
- Isotropic scaling (scale_x == scale_y)
- Horizon validation
- Lane-aligned BEV

Tham chieu bao cao: Section 2.2.1, 3.4.1
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum

from config.settings import get_config
from utils.logger import get_logger

logger = get_logger(__name__)


class CameraPosition(Enum):
    CENTERED = "centered"
    OFFSET_LEFT = "offset_left"
    OFFSET_RIGHT = "offset_right"
    HIGH_OBLIQUE = "high_oblique"


@dataclass
class WorldPoint:
    """Toa do thuc te (mét) sau IPM."""
    x: float
    y: float
    pixel_u: float = 0.0
    pixel_v: float = 0.0
    confidence: float = 1.0

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)

    def distance_to(self, other: "WorldPoint") -> float:
        return float(np.hypot(self.x - other.x, self.y - other.y))


@dataclass
class HomographyResult:
    """Ket qua validate homography."""
    is_valid: bool
    H: Optional[np.ndarray] = None
    H_inv: Optional[np.ndarray] = None
    error_msg: str = ""
    condition_number: float = 0.0
    area_ratio: float = 0.0


class IPMTransformer:
    """
    Inverse Perspective Mapping.
    
    OPENCV POINT ORDER (correct):
    cv2.getPerspectiveTransform expects: [TL, TR, BL, BR]
    - TL: top-left     (xa, trai)
    - TR: top-right    (xa, phai)
    - BL: bottom-left  (gan, trai)
    - BR: bottom-right (gan, phai)
    
    Khong reorder - giu nguyen thu tu nay xuyen suot.
    """

    def __init__(self, config=None):
        self.cfg = config or get_config()
        ipm_cfg = self.cfg.ipm

        # === CAMERA ===
        self.camera_position = CameraPosition(
            getattr(ipm_cfg, 'camera_position', 'offset_right')
        )

        # === SOURCE POINTS ===
        # Thu tu: TL, TR, BL, BR (OpenCV standard)
        raw_points = getattr(ipm_cfg, 'src_points', [
            [530, 220],    # TL
            [930, 220],    # TR
            [180, 760],    # BL
            [1260, 760],   # BR
        ])
        self.src_points = self._validate_and_order_points(raw_points)

        # === DESTINATION (world meters) ===
        # Thu tu: TL, TR, BL, BR
        self.dst_points_m = np.array([
            [0, 0],
            [10, 0],
            [0, 60],
            [10, 60],
        ], dtype=np.float32)

        # === ISOTROPIC SCALE (pixels per meter) ===
        self.scale = getattr(ipm_cfg, 'scale', 20.0)
        
        # BEV size tu dong tinh tu dst_points_m va scale
        max_x = self.dst_points_m[:, 0].max()
        max_y = self.dst_points_m[:, 1].max()
        self.bev_w = int(max_x * self.scale)
        self.bev_h = int(max_y * self.scale)

        # === ROI ===
        self.roi_enabled = getattr(ipm_cfg, 'roi_enabled', True)
        self.roi_margin = getattr(ipm_cfg, 'roi_margin', 50)

        # === State ===
        self.H: Optional[np.ndarray] = None
        self.H_inv: Optional[np.ndarray] = None
        self._calibrated = False
        self._last_validation: Optional[HomographyResult] = None
        self._lane_angle: float = 0.0  # Goc cua lane so voi truc doc anh
        self._calibration_frame_count = 0          # <-- THEM DONG NAY
        self._last_valid_points: Optional[np.ndarray] = None  # <-- THEM DONG NAY
        self.recalibrate_every = getattr(ipm_cfg, 'recalibrate_every', 300)  # <-- THEM DONG NAY

        logger.info(
            f"IPM init | cam={self.camera_position.value} | "
            f"BEV={self.bev_w}x{self.bev_h} | scale={self.scale} px/m (isotropic) | "
            f"coverage={max_x:.0f}x{max_y:.0f}m"
        )

    # ================================================================
    # POINT VALIDATION & ORDERING
    # ================================================================

    def auto_calibrate_from_frame(self, frame: np.ndarray) -> HomographyResult:
        """Tu dong tim 4 diem mat duong va calibrate IPM."""
        h, w = frame.shape[:2]
        is_portrait = (h > w)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # ROI: bo bau troi + chi giua anh (tranh via he)
        roi_mask = np.zeros_like(blur)
        roi_y_start = int(h * 0.25) if is_portrait else int(h * 0.20)
        roi_y_end = int(h * 0.95)
        margin_x = int(w * 0.12)
        roi_mask[roi_y_start:roi_y_end, margin_x:w-margin_x] = 255
        
        edges = cv2.Canny(blur, 40, 120)
        edges = cv2.bitwise_and(edges, roi_mask)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                                threshold=50, minLineLength=100, maxLineGap=50)
        
        if lines is None or len(lines) < 3:
            return HomographyResult(is_valid=False, error_msg="Khong tim thay duong thang")
        
        # Tim 2 duong bien
        mid_x = w // 2
        left_candidates, right_candidates = [], []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = abs((y2 - y1) / (x2 - x1))
            
            if slope > 0.4:
                avg_x = (x1 + x2) / 2
                avg_y = (y1 + y2) / 2
                
                if roi_y_start < avg_y < roi_y_end:
                    if avg_x < mid_x:
                        left_candidates.append((x1, y1, x2, y2, avg_x, avg_y))
                    else:
                        right_candidates.append((x1, y1, x2, y2, avg_x, avg_y))
        
        if len(left_candidates) < 1 or len(right_candidates) < 1:
            return HomographyResult(is_valid=False, error_msg="Thieu lane")
        
        # Chon line dai nhat, gan center
        def score(line):
            x1, y1, x2, y2, ax, ay = line
            length = np.hypot(x2-x1, y2-y1)
            dist = abs(ax - mid_x)
            return length * 2 - dist * 0.5
        
        left_edge = max(left_candidates, key=score)
        right_edge = max(right_candidates, key=score)
        
        def get_x_at(line, y):
            x1, y1, x2, y2, _, _ = line
            if abs(y2 - y1) < 1:
                return (x1 + x2) / 2
            return x1 + (x2 - x1) * (y - y1) / (y2 - y1)
        
        top_y = roi_y_start + int(h * 0.08)
        bottom_y = roi_y_end - int(h * 0.05)
        
        # Tinh 4 diem
        tl_x = get_x_at(left_edge, top_y)
        tr_x = get_x_at(right_edge, top_y)
        bl_x = get_x_at(left_edge, bottom_y)
        br_x = get_x_at(right_edge, bottom_y)
        
        # === DAM BAO HINH THANG DUNG ===
        # Top phai hep hon bottom
        top_w = tr_x - tl_x
        bot_w = br_x - bl_x
        
        if top_w > bot_w or top_w < 30:
            # Force trapezoid dung
            center = (bl_x + br_x) / 2
            top_w = bot_w * 0.45
            tl_x = center - top_w / 2
            tr_x = center + top_w / 2
        
        points = np.float32([
            [tl_x, top_y],     # TL
            [tr_x, top_y],     # TR
            [bl_x, bottom_y],  # BL
            [br_x, bottom_y],  # BR
        ])
        
        # Validate
        pts = self._validate_and_order_points(points)
        top_w = np.linalg.norm(pts[1] - pts[0])
        bot_w = np.linalg.norm(pts[3] - pts[2])
        
        if top_w < 40 or bot_w < 60:
            return HomographyResult(is_valid=False, error_msg="Road too narrow")
        
        self.src_points = pts
        self._last_valid_points = pts.copy()
        
        logger.info(f"Auto IPM: TL({pts[0,0]:.0f},{pts[0,1]:.0f}) "
                   f"TR({pts[1,0]:.0f},{pts[1,1]:.0f}) "
                   f"BL({pts[2,0]:.0f},{pts[2,1]:.0f}) "
                   f"BR({pts[3,0]:.0f},{pts[3,1]:.0f})")
        
        return self.calibrate()

        
    def update_calibration(self, frame: np.ndarray, force: bool = False) -> bool:
        """Cap nhat calibration - tu dong hoac manual."""
        self._calibration_frame_count += 1
        
        # Chi auto-calibrate moi N frame
        if not force and self._calibrated:

            # Khong recalibrate lien tuc
            if self._calibration_frame_count % self.recalibrate_every != 0:
                return True

            # === CHI RECALIBRATE KHI CHAT LUONG XAU ===

            need_recalibrate = False

            # 1. Homography condition xau
            if self._last_validation is not None:
                cond = self._last_validation.condition_number

                if cond > 5000:
                    logger.warning(f"Recalibrate: bad condition number {cond:.0f}")
                    need_recalibrate = True

            # 2. Lane angle bi drift
            current_angle = self._lane_angle

            if abs(current_angle) > 15:
                logger.warning(f"Recalibrate: lane angle drift {current_angle:.1f}")
                need_recalibrate = True

            # Neu moi thu on dinh -> skip
            if not need_recalibrate:
                return True
                
        # Thu auto-calibrate
        result = self.auto_calibrate_from_frame(frame)
        
        if result.is_valid:
            logger.info(f"Auto-calibration OK (frame {self._calibration_frame_count})")
            return True
        else:
            logger.warning(f"Auto-calibration failed: {result.error_msg}")
            
            # Fallback: dung diem cu neu co
            if self._last_valid_points is not None and self._calibrated:
                self.src_points = self._last_valid_points.copy()
                self.calibrate()
                return True
            
            # Fallback: dung diem config
            return self.calibrate().is_valid
    
    def auto_detect_road_plane(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Tu dong phat hien 4 diem mat duong.
        Hien tai chua implement - return None de dung fallback manual.
        """
        return None

    def _validate_and_order_points(self, points: List) -> np.ndarray:
        """
        Validate va sap xep 4 diem theo OpenCV order: TL, TR, BL, BR.
        
        TL = top-left   (u nho, v nho)
        TR = top-right  (u lon, v nho)
        BL = bottom-left (u nho, v lon)
        BR = bottom-right (u lon, v lon)
        """
        pts = np.array(points, dtype=np.float32)
        
        if len(pts) != 4:
            logger.warning(f"Can 4 diem, co {len(pts)}. Dung default.")
            return np.float32([[530, 220], [930, 220], [180, 760], [1260, 760]])

        # Sap xep theo v (top → bottom)
        sorted_by_v = pts[np.argsort(pts[:, 1])]
        
        # Top 2 → sap xep theo u (left → right)
        top_two = sorted_by_v[:2]
        top_two = top_two[np.argsort(top_two[:, 0])]
        
        # Bottom 2 → sap xep theo u (left → right)
        bottom_two = sorted_by_v[2:]
        bottom_two = bottom_two[np.argsort(bottom_two[:, 0])]
        
        # OpenCV order: TL, TR, BL, BR
        ordered = np.float32([
            top_two[0],     # TL
            top_two[1],     # TR
            bottom_two[0],  # BL
            bottom_two[1],  # BR
        ])
        
        # Validate cơ bản
        top_width = np.linalg.norm(ordered[1] - ordered[0])
        bottom_width = np.linalg.norm(ordered[3] - ordered[2])
        
        logger.debug(
            f"Points: TL({ordered[0,0]:.0f},{ordered[0,1]:.0f}) "
            f"TR({ordered[1,0]:.0f},{ordered[1,1]:.0f}) "
            f"BL({ordered[2,0]:.0f},{ordered[2,1]:.0f}) "
            f"BR({ordered[3,0]:.0f},{ordered[3,1]:.0f}) | "
            f"top_w={top_width:.0f}px bot_w={bottom_width:.0f}px"
        )
        
        return ordered

    # ================================================================
    # HOMOGRAPHY CALIBRATION & VALIDATION
    # ================================================================

    def calibrate(self) -> HomographyResult:
        """
        Tinh va validate ma tran Homography.
        
        Dung OpenCV order [TL, TR, BL, BR] cho ca src va dst.
        """
        src = self.src_points.copy().astype(np.float32)
        
        # Dst trong BEV pixel space: world (m) * scale
        dst_px = self.dst_points_m.copy()
        dst_px[:, 0] *= self.scale
        dst_px[:, 1] *= self.scale
        dst_px = dst_px.astype(np.float32)
        
        # === VALIDATION 1: Minimum area ===
        src_area = cv2.contourArea(src)
        dst_area = cv2.contourArea(dst_px)
        
        if src_area < 100:
            msg = f"Source area too small: {src_area:.0f} px²"
            logger.error(msg)
            return HomographyResult(is_valid=False, error_msg=msg)

        # === VALIDATION 2: Convexity ===
        poly_order = np.array([0, 1, 3, 2], dtype=int)
        poly_pts = src[poly_order].astype(np.int32)
        
        if not cv2.isContourConvex(poly_pts):
            msg = "Source points not convex"
            logger.error(msg)
            return HomographyResult(is_valid=False, error_msg=msg)

        # === VALIDATION 3: Trapezoid shape ===
        top_width = np.linalg.norm(src[1] - src[0])
        bottom_width = np.linalg.norm(src[3] - src[2])

        height_left = abs(src[2,1] - src[0,1])
        height_right = abs(src[3,1] - src[1,1])

        if height_left < 100 or height_right < 100:
            return HomographyResult(
                is_valid=False,
                error_msg="ROI height too small"
            )
        
        if top_width > bottom_width * 1.3:
            logger.warning(
                f"Top ({top_width:.0f}px) wider than bottom ({bottom_width:.0f}px). "
                f"Co the TL/TR bi dao voi BL/BR."
            )

        # === VALIDATION 4: Horizon check ===
        horizon_ratio = top_width / max(bottom_width, 1.0)
        if horizon_ratio < 0.02:
            msg = f"Top too narrow (ratio={horizon_ratio:.3f}) - near horizon"
            logger.error(msg)
            return HomographyResult(is_valid=False, error_msg=msg)

        # === COMPUTE HOMOGRAPHY ===
        try:
            H = cv2.getPerspectiveTransform(src, dst_px)
        except Exception as e:
            logger.exception("Perspective transform failed")
            return HomographyResult(
                is_valid=False,
                error_msg=str(e)
            )

        # === VALIDATION 5: Condition number ===
        try:
            cond = np.linalg.cond(H)
        except:
            cond = 1e6
        
        if cond > 1e5 or np.isnan(cond) or np.isinf(cond):
            msg = f"Poorly conditioned (cond={cond:.0f})"
            logger.error(msg)
            return HomographyResult(is_valid=False, error_msg=msg, condition_number=cond)

        # === VALIDATION 6: Warped corners inside canvas ===
        warped = self._warp_points(src, H)
        margin = 100
        for i, pt in enumerate(warped):
            if (pt[0] < -margin or pt[0] > self.bev_w + margin or
                pt[1] < -margin or pt[1] > self.bev_h + margin):
                msg = f"Corner {i} outside BEV: ({pt[0]:.0f},{pt[1]:.0f})"
                logger.error(msg)
                return HomographyResult(is_valid=False, error_msg=msg)

        # === SUCCESS ===
        self.H = H
        self.H_inv = np.linalg.inv(H)
        self._calibrated = True
        
        # Tinh lane angle
        self._lane_angle = self._compute_lane_angle()
        
        result = HomographyResult(
            is_valid=True,
            H=H.copy(),
            H_inv=self.H_inv.copy(),
            condition_number=cond,
            area_ratio=src_area / max(dst_area, 1.0)
        )
        self._last_validation = result
        
        logger.info(f"IPM OK | cond={cond:.0f} | lane_angle={self._lane_angle:.1f}°")
        return result

    def _warp_points(self, pts: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Warp points qua homography."""
        return cv2.perspectiveTransform(
            pts.reshape(-1, 1, 2).astype(np.float32), H
        ).reshape(-1, 2)

    def _compute_lane_angle(self) -> float:
        """Tinh goc cua lane so voi truc doc cua BEV."""
        if not self._calibrated:
            return 0.0
        
        # Chieu 2 diem giua lane len BEV
        mid_top = (self.src_points[0] + self.src_points[1]) / 2
        mid_bot = (self.src_points[2] + self.src_points[3]) / 2
        
        pts = np.float32([mid_top, mid_bot]).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)
        
        dx = warped[1, 0] - warped[0, 0]
        dy = warped[1, 1] - warped[0, 1]
        
        return float(np.degrees(np.arctan2(dx, dy)))

    # ================================================================
    # ROI & PREPROCESSING
    # ================================================================

    def compute_roi(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """Tinh ROI tu src_points + margin."""
        h, w = frame.shape[:2]
        x1 = max(0, int(self.src_points[:, 0].min()) - self.roi_margin)
        y1 = max(0, int(self.src_points[:, 1].min()) - self.roi_margin)
        x2 = min(w, int(self.src_points[:, 0].max()) + self.roi_margin)
        y2 = min(h, int(self.src_points[:, 1].max()) + self.roi_margin)
        return (x1, y1, x2, y2)

    # ================================================================
    # CORE TRANSFORMS
    # ================================================================

    def get_bev_image(self, frame: np.ndarray) -> np.ndarray:
        """
        Tao Bird's-Eye View image.
        
        Neu ROI enabled: crop truoc roi warp.
        Dung phep nhan ma tran H @ T (KHONG tru truc tiep).
        """
        if not self._calibrated:
            raise RuntimeError("IPM not calibrated.")

        if self.roi_enabled:
            x1, y1, x2, y2 = self.compute_roi(frame)
            cropped = frame[y1:y2, x1:x2]
            
            # Tao translation matrix T
            T = np.array([
                [1, 0, -x1],
                [0, 1, -y1],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Dung phep nhan ma tran: H_adjusted = H @ T
            H_adjusted = self.H @ T
        else:
            cropped = frame
            H_adjusted = self.H
        
        bev = cv2.warpPerspective(
            cropped, H_adjusted,
            (self.bev_w, self.bev_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        return bev

    def pixel_to_world(self, u: float, v: float) -> WorldPoint:
        """Pixel goc → WorldPoint (mét)."""
        if not self._calibrated:
            raise RuntimeError("IPM not calibrated.")
        
        pt = np.array([[[u, v]]], dtype=np.float32)
        pt_bev = cv2.perspectiveTransform(pt, self.H)
        u_bev, v_bev = pt_bev[0, 0]
        
        return WorldPoint(
            x=float(u_bev) / self.scale,
            y=float(v_bev) / self.scale,
            pixel_u=u, pixel_v=v
        )

    def world_to_pixel(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """World (mét) → Pixel goc."""
        if not self._calibrated:
            raise RuntimeError("IPM not calibrated.")
        
        u_bev = x_m * self.scale
        v_bev = y_m * self.scale
        pt = np.array([[[u_bev, v_bev]]], dtype=np.float32)
        pt_img = cv2.perspectiveTransform(pt, self.H_inv)
        return (int(pt_img[0, 0, 0]), int(pt_img[0, 0, 1]))

    # ================================================================
    # DEBUG & VISUALIZATION
    # ================================================================

    def create_debug_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Ve source polygon + grid + thong tin calibration."""
        result = frame.copy()
        h, w = result.shape[:2]
        
        # === REORDER CHO POLYGON: TL, TR, BR, BL (theo chu vi) ===
        # src_points luu: [TL, TR, BL, BR]
        # Can polygon order: [TL, TR, BR, BL]
        poly_order = np.array([0, 1, 3, 2], dtype=int)
        poly_pts = self.src_points[poly_order].astype(np.int32).reshape((-1, 1, 2))
        
        overlay = result.copy()
        cv2.fillPoly(overlay, [poly_pts], (0, 100, 0))
        cv2.addWeighted(overlay, 0.15, result, 0.85, 0, result)
        cv2.polylines(result, [poly_pts], True, (0, 255, 255), 2)
        
        # Ve 4 diem + labels
        labels = ["TL", "TR", "BL", "BR"]
        colors = [(0, 255, 0), (0, 255, 0), (0, 0, 255), (0, 0, 255)]
        
        for i, (pt, label, color) in enumerate(zip(self.src_points.astype(int), labels, colors)):
            cv2.circle(result, tuple(pt), 8, color, -1)
            cv2.circle(result, tuple(pt), 10, (255, 255, 255), 2)
            cv2.putText(result, label, (pt[0] + 12, pt[1] - 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Grid (neu calibrated)
        if self._calibrated:
            max_x = self.dst_points_m[:, 0].max()
            max_y = self.dst_points_m[:, 1].max()
            
            for x in np.arange(0, max_x + 0.1, 5):
                pts = []
                for y in np.linspace(0, max_y, 50):
                    px, py = self.world_to_pixel(x, y)
                    if 0 <= px < w and 0 <= py < h:
                        pts.append((px, py))
                for i in range(1, len(pts)):
                    p1 = pts[i - 1]
                    p2 = pts[i]

                    # Chi ve neu ca 2 diem nam trong ROI
                    if self.is_point_in_roi(*p1) and self.is_point_in_roi(*p2):
                        cv2.line(result, p1, p2, (200, 100, 0), 1)
            
            for y in np.arange(0, max_y + 0.1, 5):
                pts = []
                for x in np.linspace(0, max_x, 50):
                    px, py = self.world_to_pixel(x, y)
                    if 0 <= px < w and 0 <= py < h:
                        pts.append((px, py))
                for i in range(1, len(pts)):
                    p1 = pts[i - 1]
                    p2 = pts[i]

                    # Chi ve neu ca 2 diem nam trong ROI
                    if self.is_point_in_roi(*p1) and self.is_point_in_roi(*p2):
                        cv2.line(result, p1, p2, (200, 100, 0), 1)
        
        # ROI
        if self.roi_enabled:
            rx1, ry1, rx2, ry2 = self.compute_roi(frame)
            cv2.rectangle(result, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
        
        # Info box
        info_lines = [
            f"IPM | scale={self.scale}px/m",
            f"BEV={self.bev_w}x{self.bev_h}",
            f"lane_angle={self._lane_angle:.1f}deg",
        ]
        y_off = 30
        for line in info_lines:
            cv2.putText(result, line, (10, y_off),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            y_off += 22
        
        return result

    def create_bev_debug(self, frame: np.ndarray) -> np.ndarray:
        """BEV voi grid toa do."""
        if not self._calibrated:
            return np.zeros((self.bev_h, self.bev_w, 3), dtype=np.uint8)
        
        bev = self.get_bev_image(frame)
        max_x = self.dst_points_m[:, 0].max()
        max_y = self.dst_points_m[:, 1].max()
        
        for x in np.arange(0, max_x + 0.1, 5):
            px = int(x * self.scale)
            cv2.line(bev, (px, 0), (px, self.bev_h), (0, 100, 200), 1)
            cv2.putText(bev, f"{x:.0f}m", (px + 2, 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 200), 1)
        
        for y in np.arange(0, max_y + 0.1, 5):
            py = int(y * self.scale)
            cv2.line(bev, (0, py), (self.bev_w, py), (0, 100, 200), 1)
            cv2.putText(bev, f"{y:.0f}m", (3, py - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 200), 1)
        
        return bev

    def draw_calibration_points(self, frame: np.ndarray) -> np.ndarray:
        """Ve 4 diem calibration len frame."""
        vis = frame.copy()
        labels = ["TL", "TR", "BL", "BR"]
        colors = [(0, 255, 0), (0, 255, 0), (0, 0, 255), (0, 0, 255)]
        
        for i, (pt, label, color) in enumerate(zip(self.src_points.astype(int), labels, colors)):
            cv2.circle(vis, tuple(pt), 8, color, -1)
            cv2.circle(vis, tuple(pt), 10, (255, 255, 255), 2)
            cv2.putText(vis, f"{label}({pt[0]},{pt[1]})", 
                    (pt[0] + 12, pt[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        
        return vis

    def is_point_in_roi(self, u: float, v: float) -> bool:
        """
        Kiem tra pixel (u,v) co nam trong vung IPM (trapezoid) khong.
        """
        # Dung poly_order de co polygon lien tuc: TL, TR, BR, BL
        poly_order = np.array([0, 1, 3, 2], dtype=int)
        polygon = self.src_points[poly_order].astype(np.int32)
        return cv2.pointPolygonTest(polygon, (float(u), float(v)), False) >= 0
    
    def filter_detections_in_roi(self, detections) -> list:
        """
        Loc detection, chi giu lai nhung cai nam trong vung IPM.
        
        Args:
            detections: list cac Detection object (co bbox_xyxy)
        Returns:
            list cac Detection nam trong vung IPM
        """
        filtered = []
        for det in detections:
            # Kiem tra bottom_center (diem cham mat duong)
            if hasattr(det, 'bottom_center'):
                u, v = det.bottom_center
            elif hasattr(det, 'bbox_xyxy'):
                x1, y1, x2, y2 = det.bbox_xyxy
                u = (x1 + x2) / 2
                v = y2  # Bottom center
            else:
                continue
            
            if self.is_point_in_roi(u, v):
                filtered.append(det)
        
        return filtered
    
    def get_roi_area_m2(self) -> float:
        """
        Tinh dien tich vung IPM (m²).
        """
        # Dien tich hinh thang trong world space
        top_w = np.linalg.norm(self.dst_points_m[1] - self.dst_points_m[0])
        bottom_w = np.linalg.norm(self.dst_points_m[3] - self.dst_points_m[2])
        h = abs(self.dst_points_m[2, 1] - self.dst_points_m[0, 1])
        return (top_w + bottom_w) / 2 * h
    
    def get_congestion_level(self, vehicle_count: int) -> dict:
        """
        Danh gia muc do tac nghen dua tren so xe va dien tich.
        
        Returns:
            dict: {'level': 'LOW'|'MEDIUM'|'HIGH'|'JAM', 
                   'density': float (xe/100m²),
                   'color': (B,G,R)}
        """
        area = self.get_roi_area_m2()
        if area <= 0:
            return {'level': 'UNKNOWN', 'density': 0, 'color': (128, 128, 128)}
        
        # Mat do: xe / 100m²
        density = vehicle_count / area * 100
        
        # Nguong (dieu chinh theo thuc te)
        if density < 1.0:       # < 1 xe/100m²
            level = 'THONG THOANG'
            color = (0, 255, 0)     # Xanh
        elif density < 2.5:     # 1-2.5 xe/100m²
            level = 'DONG VUA'
            color = (0, 255, 255)   # Vang
        elif density < 4.0:     # 2.5-4 xe/100m²
            level = 'DONG DUC'
            color = (0, 165, 255)   # Cam
        else:                    # > 4 xe/100m²
            level = 'TAC NGHEN'
            color = (0, 0, 255)     # Do
        
        return {
            'level': level,
            'density': density,
            'color': color,
            'vehicle_count': vehicle_count,
            'area_m2': area
        }

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def get_stats(self) -> Dict:
        return {
            'calibrated': self._calibrated,
            'bev_size': f'{self.bev_w}x{self.bev_h}',
            'scale': f'{self.scale} px/m (isotropic)',
            'coverage': f'{self.dst_points_m[:,0].max():.0f}x{self.dst_points_m[:,1].max():.0f} m',
            'camera_position': self.camera_position.value,
            'lane_angle': f'{self._lane_angle:.1f}°',
        }