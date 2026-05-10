"""YOLOv8-pose Estimator - thay thế Lightweight OpenPose"""

from ultralytics import YOLO
import numpy as np
import cv2

class PoseEstimator:
    def __init__(self, model_path='yolov8n-pose.pt', device='cuda', conf=0.5):
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf

    def extract_keypoints(self, frame):
        results = self.model(frame, device=self.device, conf=self.conf, verbose=False)
        persons = []
        if results[0].keypoints is not None and len(results[0].boxes) > 0:
            for i in range(len(results[0].boxes)):
                if results[0].boxes.cls[i] != 0:  # person class
                    continue
                bbox = results[0].boxes.xyxy[i].cpu().numpy()
                kpts = results[0].keypoints[i].data.cpu().numpy()
                # Đảm bảo keypoints có shape (17, 3)
                if kpts.ndim == 3:
                    kpts = kpts.squeeze(0)
                if kpts.shape == (3, 17):
                    kpts = kpts.T  # transpose về (17, 3)
                    kpts = kpts[0]
                lean_angle = self._calculate_lean_angle(kpts)
                persons.append({
                    'bbox': tuple(bbox),
                    'keypoints': kpts,
                    'lean_angle': lean_angle
                })
        return persons

    def _calculate_lean_angle(self, kpts):
        # Đảm bảo shape (17, 3)
        if kpts.shape[0] != 17:
            if kpts.shape[1] == 17:
                kpts = kpts.T
            else:
                return 0.0
        # Shoulders (5,6) and Hips (11,12)
        left_sho, right_sho = kpts[5], kpts[6]
        left_hip, right_hip = kpts[11], kpts[12]
        if max(left_sho[2], right_sho[2]) < 0.3 or max(left_hip[2], right_hip[2]) < 0.3:
            return 0.0
        mid_sho = (left_sho + right_sho) / 2 if min(left_sho[2], right_sho[2]) > 0.3 else (left_sho if left_sho[2] > right_sho[2] else right_sho)
        mid_hip = (left_hip + right_hip) / 2 if min(left_hip[2], right_hip[2]) > 0.3 else (left_hip if left_hip[2] > right_hip[2] else right_hip)
        dx, dy = mid_sho[0] - mid_hip[0], mid_sho[1] - mid_hip[1]
        return float(np.degrees(np.arctan2(dx, -dy)))

    def match_to_tracks(self, persons, tracks):
        matches = {}
        for p in persons:
            best_iou, best_id = 0.2, None
            for t in tracks:
                iou = self._iou(p['bbox'], t.bbox_xyxy)
                if iou > best_iou:
                    best_iou, best_id = iou, t.track_id
            if best_id is not None:
                matches[best_id] = {'lean_angle': p['lean_angle'], 'has_pose': True}
        return matches

    def draw_skeleton(self, frame, persons):
        """Vẽ keypoints + skeleton lên frame"""
        result = frame.copy()
        SKELETON = [
            (15,13),(13,11),(16,14),(14,12),(11,5),(12,6),
            (5,7),(7,9),(6,8),(8,10),(5,6),(11,12)
        ]
        for p in persons:
            kpts = p['keypoints']
            # Vẽ điểm
            for x, y, conf in kpts:
                if conf > 0.3:
                    cv2.circle(result, (int(x), int(y)), 3, (0,255,0), -1)
            # Vẽ đường nối
            for a, b in SKELETON:
                if kpts[a][2] > 0.3 and kpts[b][2] > 0.3:
                    pt1 = (int(kpts[a][0]), int(kpts[a][1]))
                    pt2 = (int(kpts[b][0]), int(kpts[b][1]))
                    cv2.line(result, pt1, pt2, (0,255,255), 1)
            # Góc nghiêng
            cv2.putText(result, f"Lean:{p['lean_angle']:.1f}",
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        return result

    @staticmethod
    def _iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        return inter / area_a if area_a > 0 else 0
    
    