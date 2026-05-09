"""
Vehicle Detector Module
Pipeline position: VIDEO INPUT → [YOLOv8] → DeepSORT → ...

Input:  numpy frame (H, W, 3) BGR
Output: List[Detection] với Detection = (bbox_xyxy, confidence, class_id, class_name)
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path

from ultralytics import YOLO
from config.settings import get_config
from utils.logger import get_logger

logger = get_logger(__name__)

# ─── Class mapping từ COCO ─────────────────────────────────
COCO_VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class Detection:
    """
    Kết quả detect một phương tiện trong một frame.
    
    bbox_xyxy: [x1, y1, x2, y2] pixel coords
    confidence: float [0, 1]
    class_id: int (COCO class id)
    class_name: str ("motorcycle", "car", ...)
    """
    bbox_xyxy: np.ndarray          # shape (4,)
    confidence: float
    class_id: int
    class_name: str
    
    @property
    def bbox_xywh(self) -> np.ndarray:
        """Convert sang format [cx, cy, w, h] - dùng cho DeepSORT."""
        x1, y1, x2, y2 = self.bbox_xyxy
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return np.array([cx, cy, w, h], dtype=np.float32)
    
    @property
    def center(self) -> Tuple[float, float]:
        """Tâm của bounding box."""
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def bottom_center(self) -> Tuple[float, float]:
        """
        Điểm giữa cạnh dưới bbox — dùng để chiếu IPM.
        Lý do: Điểm này xấp xỉ vị trí bánh xe chạm đường.
        """
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2, y2)
    
    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x2 - x1) * (y2 - y1)
    
    def is_motorcycle(self) -> bool:
        return self.class_id == 3
    
    def __repr__(self) -> str:
        cx, cy = self.center
        return (
            f"Detection({self.class_name} | "
            f"conf={self.confidence:.2f} | "
            f"center=({cx:.0f},{cy:.0f}) | "
            f"area={self.area:.0f}px²)"
        )


class VehicleDetector:
    """
    YOLOv8-based vehicle detector.
    
    Architecture decisions (từ báo cáo):
    - Model: YOLOv8n (nano) cho demo, YOLOv8s cho production
    - conf=0.35: Giữ xe bị che khuất, để tracker quyết định lọc
    - iou=0.65: Tránh NMS gộp 2 xe đi sát nhau thành 1
    - Classes: [2,3,5,7] = car, motorcycle, bus, truck
    """
    
    def __init__(self, config=None):
        self.cfg = config or get_config()
        self.det_cfg = self.cfg.detector
        
        self.model: Optional[YOLO] = None
        self.device: str = self.cfg.system.device
        self.classes: List[int] = self.det_cfg.classes
        self.conf_thresh: float = self.det_cfg.confidence_threshold
        self.iou_thresh: float = self.det_cfg.iou_threshold
        self.input_size: int = self.det_cfg.input_size
        
        self._frame_count: int = 0
        self._total_detections: int = 0
        
        logger.info(
            f"VehicleDetector init | device={self.device} | "
            f"conf={self.conf_thresh} | iou={self.iou_thresh}"
        )

        logger.info(f"Classes filter: {self.classes}")
    
    def load_model(self) -> "VehicleDetector":
        """
        Load model YOLOv8. Hỗ trợ cả .pt và TensorRT .engine.
        Gọi một lần duy nhất khi khởi động hệ thống.
        """
        # Chọn model path
        if self.det_cfg.use_tensorrt:
            model_path = Path(self.det_cfg.tensorrt_model_path)
            logger.info(f"Loading TensorRT engine: {model_path}")
        else:
            model_path = Path(self.det_cfg.model_path)
            logger.info(f"Loading YOLOv8 model: {model_path}")
        
        if not model_path.exists():
            # Auto-download nếu chưa có
            logger.warning(
                f"Model không tồn tại tại {model_path}. "
                f"Auto-downloading yolov8n.pt..."
            )
            model_path = "yolov8n.pt"
        
        # Load
        self.model = YOLO(str(model_path))
        self.model.to(self.device)
        
        # Warmup — chạy 1 frame dummy để CUDA khởi tạo
        logger.info("Model warmup...")
        dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)
        logger.info("Model ready.")
        
        return self
    
    def detect(
        self,
        frame: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> List[Detection]:
        """
        Detect phương tiện trong một frame.
        
        Args:
            frame: BGR image (H, W, 3)
            roi: (x1, y1, x2, y2) vùng quan tâm, None = full frame
        
        Returns:
            List[Detection] đã được filter theo conf và class
        """
        if self.model is None:
            raise RuntimeError("Model chưa được load. Gọi load_model() trước.")
        
        self._frame_count += 1
        
        # Crop ROI nếu có
        frame_to_detect = frame
        roi_offset = (0, 0)
        
        if roi is not None:
            x1r, y1r, x2r, y2r = roi
            frame_to_detect = frame[y1r:y2r, x1r:x2r]
            roi_offset = (x1r, y1r)
        
        # Inference
        results = self.model(
            frame_to_detect,
            imgsz=self.input_size,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            classes=self.classes,
            verbose=False,
            device=self.device
        )
        
        # Parse results
        detections = self._parse_results(results[0], roi_offset)
        self._total_detections += len(detections)
        
        return detections
    
    def _parse_results(
        self,
        result,
        roi_offset: Tuple[int, int] = (0, 0)
    ) -> List[Detection]:
        """
        Convert YOLO result object thành List[Detection].
        
        Args:
            result: ultralytics YOLO result object
            roi_offset: (dx, dy) để shift coords về frame gốc
        """
        detections = []
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        boxes = result.boxes
        dx, dy = roi_offset
        
        for i in range(len(boxes)):
            # Lấy bbox
            xyxy = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())
            
            # Shift về coordinate của frame gốc (nếu dùng ROI)
            xyxy[0] += dx
            xyxy[1] += dy
            xyxy[2] += dx
            xyxy[3] += dy
            
            # Map class id
            class_name = COCO_VEHICLE_CLASSES.get(cls_id, f"class_{cls_id}")
            
            det = Detection(
                bbox_xyxy=xyxy.astype(np.float32),
                confidence=conf,
                class_id=cls_id,
                class_name=class_name
            )
            detections.append(det)
        
        return detections
    
    @property
    def stats(self) -> dict:
        """Thống kê để debug."""
        avg_det = (
            self._total_detections / self._frame_count
            if self._frame_count > 0 else 0
        )
        return {
            "frames_processed": self._frame_count,
            "total_detections": self._total_detections,
            "avg_detections_per_frame": round(avg_det, 2),
        }
    
    def __del__(self):
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()