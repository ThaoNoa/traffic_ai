# gui/main_window.py
"""
Cua so chinh - Traffic AI System (Pipeline day du)
Workflow: Chon Video -> VideoStream -> Detector -> Tracker -> IPM -> Feature -> Rule Engine -> Alert
Phong cach: Hoc thuat
"""

import cv2
import time
import numpy as np
import os
import psutil
try:
    import GPUtil
    HAS_GPU = True
except:
    HAS_GPU = False
from pathlib import Path
from datetime import datetime

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QSlider, QGroupBox, 
                             QMessageBox, QSplitter, QFrame, QFileDialog,
                             QProgressBar, QToolBar, QAction, QStatusBar)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QDragEnterEvent, QDropEvent

# === CAC MODULE CUA BAN ===
from detector.vehicle_detector import VehicleDetector
from tracker.deep_sort_tracker import DeepSORTTracker, Track
from geometry.ipm import IPMTransformer, WorldPoint
from features.feature_extractor import FeatureExtractor, VehicleState
from anomaly.rule_engine import PhysicsRuleEngine, RiskResult, RiskLevel
from anomaly.alert_manager import AlertManager, AlertEvent
from config.settings import get_config

from gui.dashboard_widget import DashboardWidget
from gui.incident_dialog import IncidentDialog
from pose.pose_estimator import PoseEstimator
from threading import Thread, Lock
from collections import deque


# ============================================================
# Worker Thread - Pipeline day du
# ============================================================
class FullPipelineThread(QThread):
    """Thread chay toan bo pipeline thuc te."""
    
    frame_ready = pyqtSignal(np.ndarray, int, int)
    stats_ready = pyqtSignal(dict)
    processing_error = pyqtSignal(str)
    alert_detected = pyqtSignal(dict, np.ndarray)
    video_finished = pyqtSignal()
    log_message = pyqtSignal(str)
    event_logged = pyqtSignal(str, str)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self._running = False
        self._paused = False
        self.gpu_usage = 0
        self.ram_usage = 0
        self.pose_estimator = None
        self.pose_buffer = deque(maxlen=10)
        self.pose_lock = Lock()
        self._pose_thread = None
        self._pose_cap = None
        self.frame_queue = deque(maxlen=30)  # Hàng đợi frame cho pose
        self.frame_lock = Lock()
        self.pose_results = {}  # {frame_id: {'persons': [...], 'timestamp': float}}
        self.pose_results_lock = Lock()
        self._bev_cache = None         # cache BEV thumb
        self._bev_every = 3            # chỉ render BEV mới mỗi 3 frame

        # === DIRECTION / CONGESTION TRACKING (chi tinh trong vung IPM) ===
        # Luu lich su world-position (sau IPM) cua moi track de tinh van toc thuc
        self._track_world_history = {}  # {track_id: deque[(t, x_m, y_m)]}
        # Phan loai theo huong dua tren vector van toc trong world space:
        #   N: di xa camera (+y), S: lai gan camera (-y)
        #   E: sang phai (+x),    W: sang trai (-x)
        self._direction_stats = {
            'N': {'track_ids': set(), 'speeds': []},
            'S': {'track_ids': set(), 'speeds': []},
            'E': {'track_ids': set(), 'speeds': []},
            'W': {'track_ids': set(), 'speeds': []},
        }


    def run(self):
        self._running = True
        
        # Khoi tao bien truoc try
        total_frames = 0
        frame_count = 0
        start_time = time.time()
        alert_count = 0
        total_fps_sum = 0.0
        total_fps_samples = 0
        last_frame_time = time.time()
        detect_time_acc = 0.0
        total_detections_sum = 0
        active_track_ids = set()
        pose_persons = []
        
        try:
            # ============================================
            # KHOI TAO TAT CA MODULE
            # ============================================
            self.log_message.emit("Dang khoi tao cac module...")
            
            cfg = get_config()
            
            self.log_message.emit("  - Dang tai YOLOv8...")
            detector = VehicleDetector(cfg)
            detector.load_model()
            
            self.log_message.emit("  - Dang khoi tao DeepSORT...")
            tracker = DeepSORTTracker(cfg)
            tracker.initialize()

            self.log_message.emit("  - Dang khoi tao IPM...")
            ipm = IPMTransformer(cfg)
            
            # Đọc frame đầu tiên để calibrate
            cap_test = cv2.VideoCapture(self.video_path)
            ret, test_frame = cap_test.read()
            cap_test.release()
            
            if ret and test_frame is not None:
                h, w = test_frame.shape[:2]
                calibrated = False
                
                # Thử load cache trước
 # Thử load cache trước (nhưng chỉ chấp nhận nếu điều kiện tốt)
                if ipm.load_calibration():
                    # Kiểm tra chất lượng homography
                    cond = ipm._last_validation.condition_number if ipm._last_validation else 1e9
                    if cond < 5000:
                        calibrated = True
                        self.log_message.emit(f"  - IPM loaded from cache (cond={cond:.0f})")
                    else:
                        self.log_message.emit(f"  - Cache IPM co condition xau ({cond:.0f}), se calibrate lai")
                        calibrated = False
                
                # Nếu cache không có, thử auto-calibrate
                # Nếu cache không có, thử auto-calibrate
                if not calibrated:
                    self.log_message.emit("  - Dang auto-calibrate IPM...")
                    result = ipm.auto_calibrate_from_frame(test_frame)
                    # Kiểm tra thêm kích thước vùng IPM
                    top_w = np.linalg.norm(ipm.src_points[1] - ipm.src_points[0])
                    bot_w = np.linalg.norm(ipm.src_points[3] - ipm.src_points[2])
                    cond = result.condition_number
                    if result.is_valid and top_w > 100 and bot_w > 200 and cond < 50000:
                        calibrated = True
                        ipm.save_calibration()
                        self.log_message.emit(f"  - Auto IPM OK (top_w={top_w:.0f}px, cond={cond:.0f})")
                    else:
                        self.log_message.emit(f"  - Auto IPM rejected (top_w={top_w:.0f}px, cond={cond:.0f})")
                
                # Nếu auto cũng fail hoặc bị reject, dùng manual fallback an toàn
                if not calibrated:
                    self.log_message.emit("  - Dung manual fallback...")
                    h, w = test_frame.shape[:2]
                    ipm.src_points = np.float32([
                        [int(w * 0.12), int(h * 0.38)],
                        [int(w * 0.88), int(h * 0.38)],
                        [int(w * 0.03), int(h * 0.90)],
                        [int(w * 0.97), int(h * 0.90)]
                    ])
                    r = ipm.calibrate()
                    if r.is_valid:
                        calibrated = True
                        ipm.save_calibration()
                        self.log_message.emit(f"  - Manual IPM OK (cond={r.condition_number:.0f})")
                    else:
                        self.log_message.emit(f"  - Manual IPM failed: {r.error_msg}")
                
                # Lưu ảnh debug
                if calibrated:
                    try:
                        debug_img = ipm.create_debug_overlay(test_frame)
                        cv2.imwrite("outputs/ipm_debug.jpg", debug_img)
                        bev_img = ipm.get_bev_image(test_frame)
                        cv2.imwrite("outputs/bev_check.jpg", bev_img)
                        self.log_message.emit("  - Da luu anh IPM debug")
                    except:
                        pass
                else:
                    self.log_message.emit("  - CANH BAO: IPM khong the calibrate!")
            else:
                self.log_message.emit("  - Khong doc duoc video, IPM khong hoat dong")

            self.log_message.emit("  - Dang khoi tao Feature Extractor...")
            feature_extractor = FeatureExtractor(cfg)
            
            # Khởi tạo Pose Estimator
            self.log_message.emit("  - Dang khoi tao Pose Estimator...")
            pose_estimator = PoseEstimator('yolov8n-pose.pt', device='cuda')
            self.pose_estimator = pose_estimator  # Lưu lại để dùng trong _draw_all
            
            # Mở video riêng cho pose
            self._pose_thread = Thread(target=self._pose_worker, args=(pose_estimator,), daemon=True)
            
            self._pose_thread.start()
            self.log_message.emit("  - Pose thread started")
            
            self.log_message.emit("  - Dang khoi tao Rule Engine...")
            rule_engine = PhysicsRuleEngine(cfg)
            
            self.log_message.emit("  - Dang khoi tao Alert Manager...")
            alert_manager = AlertManager(cfg)
            alert_manager.set_video_properties(
                int(cfg.video.width), int(cfg.video.height), cfg.video.fps
            )
            
            self.log_message.emit("Tat ca module da san sang!")
            
            # Mo video
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.processing_error.emit("Khong the mo video: " + str(self.video_path))
                return
            
            # Doc frame dau tien (chỉ để tính tổng số frame, không dùng để calibrate lại IPM nữa)
            ret, first_frame = cap.read()
            if not ret:
                self.processing_error.emit("Khong doc duoc frame dau tien")
                return
            
            # XÓA DÒNG ipm.calibrate() Ở ĐÂY vì đã calibrate ở trên rồi
            # ipm.calibrate()  <-- DÒNG NÀY CẦN XÓA HOẶC COMMENT
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_video = cap.get(cv2.CAP_PROP_FPS)
            
            self.log_message.emit(f"Bat dau xu ly video: {total_frames} khung hinh @ {fps_video:.1f} FPS")
            
            # Reset video ve dau
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Vong lap chinh
            while self._running:

                if self._paused:
                    time.sleep(0.1)
                    continue
                
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Auto calibrate IPM moi 300 frame (nếu cần)
                # Đưa frame vào hàng đợi pose (mỗi 3 frame)
                if frame_count % 3 == 0:
                    with self.frame_lock:
                        self.frame_queue.append({
                            'frame': frame.copy(),
                            'frame_id': frame_count,
                            'timestamp': frame_count / fps_video
                        })

                congestion = {'level': 'UNKNOWN', 'density': 0}
                display_frame = frame.copy() if frame is not None else None

                try:
                    # 1. Detection
                    all_detections = detector.detect(frame)
                    
                    # Loc detection trong vung IPM
                    if ipm and ipm.is_calibrated:
                        detections = ipm.filter_detections_in_roi(all_detections)
                    else:
                        detections = all_detections
                    
                    total_detections_sum += len(detections)
                    
                    # 2. Tracking
                    tracks = tracker.update(detections, frame)
                    for track in tracks:
                        active_track_ids.add(track.track_id)
                    
                    # 3. IPM
                    world_positions = {}
                    for track in tracks:
                        u, v = track.bottom_center
                        try:
                            wp = ipm.pixel_to_world(u, v)
                            world_positions[track.track_id] = wp
                        except:
                            pass

                    # 3b. CAP NHAT THONG KE THEO HUONG (chi xe trong vung IPM)
                    self._update_direction_stats(tracks, world_positions, ipm,
                                                 frame_count, fps_video)
                    
                    # 4. Feature Extraction với body_leans từ Pose
                    # Lấy pose estimation gần nhất
                    body_leans = None
                    pose_persons = []
                    with self.pose_results_lock:
                        # Tìm frame_id gần nhất có kết quả pose
                        available_ids = sorted(self.pose_results.keys())
                        if available_ids:
                            # Lấy kết quả gần nhất không vượt quá frame_count
                            best_id = max([fid for fid in available_ids if fid <= frame_count], default=None)
                            if best_id is not None:
                                pose_data = self.pose_results[best_id]
                                pose_persons = pose_data['persons']
                                if pose_persons:
                                    body_leans = pose_estimator.match_to_tracks(pose_persons, tracks)
                    
                    feature_vectors = feature_extractor.update(
                        tracks, world_positions, body_leans, frame_count
                    )
                    
                    # 5. Rule Engine
                    risk_results = {}
                    # Throttle: chi log moi violation cho moi track 1 lan / 2 giay
                    if not hasattr(self, '_last_event_frame'):
                        self._last_event_frame = {}   # {(track_id, rule_name): frame_id}

                    for track in tracks:
                        buf = feature_extractor.get_buffer(track.track_id)
                        if buf:
                            window = buf.get_window(25)
                            if window and len(window) >= 10:
                                result = rule_engine.evaluate_state(window, track.track_id, frame_count)
                                risk_results[track.track_id] = result

                                # === EMIT EVENT cho moi violation ===
                                for v in result.violations:
                                    key = (track.track_id, v.rule_name)
                                    last_f = self._last_event_frame.get(key, -9999)
                                    # Cooldown 2 giay (~50 frame) cho moi (track, rule)
                                    if frame_count - last_f < 50:
                                        continue
                                    self._last_event_frame[key] = frame_count

                                    # Phan loai mau theo severity
                                    if v.severity >= 0.7:
                                        etype = "DANGER"
                                    elif v.severity >= 0.4:
                                        etype = "WARNING"
                                    else:
                                        etype = "INFO"
                                    self.event_logged.emit(
                                        etype,
                                        f"Track #{track.track_id} | {v.rule_name}: {v.description}"
                                    )
                    
                    # 6. Alert Manager
                    alerts_triggered = []
                    if risk_results:
                        alerts_triggered = alert_manager.process(risk_results, frame_count)
                    alert_manager.push_frame(frame)
                    
                    # 7. Ve ket qua
                    display_frame = self._draw_all(frame, tracks, risk_results, world_positions, ipm, pose_persons)
                    
                    # 8. Dem alerts
                    alert_count += len(alerts_triggered)
                    
                    # 9. GUI alerts
                    for alert in alerts_triggered:
                        self.alert_detected.emit(
                            {
                                'track_id': alert.track_id,
                                'timestamp': alert.timestamp,
                                'frame_id': alert.frame_id,
                                'final_score': alert.final_score,
                                'risk_level': alert.risk_level.name if hasattr(alert.risk_level, 'name') else str(alert.risk_level),
                                'violations': alert.rule_violations,
                                'location': 'Linh Nam - Ha Noi'
                            },
                            display_frame.copy()
                        )
                    
                except Exception as e:
                    import traceback
                    display_frame = frame.copy() if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(display_frame, f"ERROR: {str(e)[:80]}", (20, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    if frame_count == 1 or frame_count % 100 == 0:
                        self.log_message.emit(f"LOI frame {frame_count}: {traceback.format_exc()[-200:]}")

                if display_frame is None:
                    display_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(display_frame, "FRAME ERROR", (150, 250),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # FPS
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                
                self.frame_ready.emit(display_frame, frame_count, total_frames)

                congestion = {'level': 'UNKNOWN', 'density': 0, 'color': (128, 128, 128), 'area_m2': 0}
                if ipm and ipm.is_calibrated:
                    try:
                        congestion = ipm.get_congestion_level(len(tracks))
                    except:
                        pass

                # --- CẬP NHẬT THỐNG KÊ FPS & ĐỘ TRỄ ---
                frame_duration = time.time() - last_frame_time
                last_frame_time = time.time()
                if frame_duration > 0:
                    total_fps_sum += 1.0 / frame_duration
                    total_fps_samples += 1
                detect_time_acc += frame_duration * 1000  # ms
                avg_detect_time = detect_time_acc / max(frame_count, 1)

                # --- CẬP NHẬT GPU (nếu có GPUtil) ---
                gpu_usage = 0
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    gpu_usage = gpus[0].load * 100 if gpus else 0
                except:
                    gpu_usage = min(85, (current_fps / 42) * 80) if current_fps > 0 else 0
                
                # === TINH AVG SPEED tu world_positions ===
                speeds_kmh = []
                for tid, wp in world_positions.items():
                    # Lay state moi nhat tu feature_extractor de co speed
                    buf = feature_extractor.get_buffer(tid)
                    if buf:
                        states = buf.get_window(3)   # 3 frame gan nhat
                        if states:
                            v_ms = states[-1].speed   # m/s
                            if 0 < v_ms < 40:         # filter outlier
                                speeds_kmh.append(v_ms * 3.6)
                avg_speed_kmh = float(np.mean(speeds_kmh)) if speeds_kmh else 0.0

                self.stats_ready.emit({
                    'fps': current_fps,
                    'avg_fps': total_fps_sum / total_fps_samples if total_fps_samples > 0 else 0,
                    'detect_time': avg_detect_time,
                    'gpu_usage': gpu_usage,
                    'vehicles': len(tracks),
                    'total_vehicles_unique': len(active_track_ids),
                    'total_detections': total_detections_sum,
                    'alerts': alert_count,
                    'frame_count': frame_count,
                    'total_frames': total_frames,
                    'progress': (frame_count / total_frames * 100) if total_frames > 0 else 0,
                    'active_tracks': tracker.active_count if tracker else 0,
                    'congestion_level': congestion['level'],
                    'congestion_density': congestion['density'],
                    'congestion_color': congestion['color'],
                    'zone_area': congestion.get('area_m2', 0),
                    'avg_speed': avg_speed_kmh / 3.6,    # <-- THEM (m/s, dashboard chuyen sang km/h)
                    'direction_stats': self._compute_direction_summary(),
                })
                
                time.sleep(0.01)
            
            cap.release()
            
            self.stats_ready.emit({
                'fps': 0,
                'avg_fps': total_fps_sum / total_fps_samples if total_fps_samples > 0 else 0,
                'detect_time': avg_detect_time,
                'gpu_usage': gpu_usage,
                'vehicles': 0,
                'total_vehicles_unique': len(active_track_ids),
                'total_detections': total_detections_sum,
                'alerts': alert_count,
                'frame_count': frame_count,
                'total_frames': total_frames,
                'progress': 100,
                'active_tracks': 0,
                'direction_stats': self._compute_direction_summary(),
            })
            
            # === KHONG reset ve 0! Giu nguyen so lieu cuoi cung ===
            # (xoa khoi block self.stats_ready.emit(...) thu hai)
            self.log_message.emit(
                f"Hoan thanh! {frame_count} frames, "
                f"{len(active_track_ids)} xe duy nhat, "
                f"{total_detections_sum} luot detect, "
                f"{alert_count} canh bao"
            )
            self.video_finished.emit()
            
        except Exception as e:
            import traceback
            self.log_message.emit(f"LOI HE THONG: {str(e)}")
            self.processing_error.emit(f"Loi pipeline:\n{str(e)}\n\n{traceback.format_exc()}")
            
    def _update_direction_stats(self, tracks, world_positions, ipm,
                                 frame_count, fps_video):
        """
        Cap nhat lich su world-position va phan loai huong di chuyen.
        CHI tinh nhung track co bottom_center NAM TRONG vung IPM (trapezoid ROI).
        Van toc duoc tinh tu world coordinates (m) chia cho VIDEO TIME (s),
        khong dung wall-clock time. Neu dung wall-clock thi khi pipeline cham
        hon realtime (vd 8 FPS / 30 FPS goc) thi dt bi gap 3-4 lan -> van toc
        bi chia nho lai 3-4 lan -> bao tac nghen nham.
        """
        if ipm is None or not ipm.is_calibrated:
            return
        if not fps_video or fps_video <= 0:
            return

        # VIDEO TIME (giay cua video, khong phai wall-clock)
        now = float(frame_count) / float(fps_video)

        WINDOW_LEN = 10           # so frame de tinh van toc trung binh
        MIN_SAMPLES = 5           # toi thieu samples truoc khi gan huong
        MIN_SPEED_MS = 0.5        # bo qua xe gan nhu dung yen (< 1.8 km/h)
        MAX_SPEED_MS = 50.0       # tren 180 km/h coi nhu nhieu IPM

        for track in tracks:
            tid = track.track_id
            if tid not in world_positions:
                continue

            # Chi tinh khi bottom_center nam trong polygon IPM
            u, v = track.bottom_center
            try:
                if not ipm.is_point_in_roi(u, v):
                    continue
            except Exception:
                continue

            wp = world_positions[tid]
            hist = self._track_world_history.setdefault(tid, deque(maxlen=WINDOW_LEN))
            hist.append((now, float(wp.x), float(wp.y)))

            if len(hist) < MIN_SAMPLES:
                continue

            t0, x0, y0 = hist[0]
            t1, x1, y1 = hist[-1]
            dt = t1 - t0
            if dt <= 0.05:
                continue

            vx = (x1 - x0) / dt
            vy = (y1 - y0) / dt
            speed_ms = (vx * vx + vy * vy) ** 0.5

            # Loc xe dung yen / nhieu IPM
            if speed_ms < MIN_SPEED_MS or speed_ms > MAX_SPEED_MS:
                continue

            # Phan huong theo goc cua vector van toc trong world frame.
            # Truc y world: +y = di xa camera (theo IPM dst_points_m).
            # Goc do tu truc +y, theo chieu kim dong ho:
            #   |angle| < 45         -> N (di xa camera)
            #   45 <= angle < 135    -> E (sang phai)
            #   |angle| >= 135       -> S (lai gan camera)
            #   -135 <= angle < -45  -> W (sang trai)
            angle_deg = np.degrees(np.arctan2(vx, vy))

            if -45 <= angle_deg < 45:
                direction = 'N'
            elif 45 <= angle_deg < 135:
                direction = 'E'
            elif angle_deg >= 135 or angle_deg < -135:
                direction = 'S'
            else:
                direction = 'W'

            bucket = self._direction_stats[direction]
            bucket['track_ids'].add(tid)
            bucket['speeds'].append(speed_ms)
            # Gioi han bo nho - chi giu 200 mau gan nhat
            if len(bucket['speeds']) > 200:
                bucket['speeds'] = bucket['speeds'][-200:]

        # Don dep history cua track da bien mat lau
        if len(self._track_world_history) > 500:
            cutoff = now - 30.0
            stale = [tid for tid, h in self._track_world_history.items()
                     if not h or h[-1][0] < cutoff]
            for tid in stale:
                self._track_world_history.pop(tid, None)

    # ============================================================
    # HE SO HIEU CHINH VAN TOC THEO CALIB IPM THUC TE
    # ------------------------------------------------------------
    # IPM dst_points_m hien tai gia dinh vung phu 8m x 30m, nhung
    # tren camera Linh Nam thuc te trapezoid chi phu khoang 3.2m x 12m.
    # Ty le phong dai ~ 2.5x nen van toc do duoc bi gap 2.5x lan.
    #
    # Cach tune nhanh: quay 1 xe co toc do chuan (vd ben canh thuoc do
    # Google Maps), ghi van toc dashboard tra ve, tinh:
    #     SPEED_CALIBRATION_FACTOR = van_toc_thuc / van_toc_dashboard
    # Hien tai = 0.40 vi user quan sat ~25 km/h thuc / ~62 km/h dashboard.
    # ============================================================
    SPEED_CALIBRATION_FACTOR = 0.40

    def _compute_direction_summary(self) -> dict:
        """
        Tinh van toc trung binh (km/h) va so xe duy nhat cho moi huong.
        Co ap he so SPEED_CALIBRATION_FACTOR de bu sai so calib IPM.
        Tra ve dict {'N': {'speed_kmh': ..., 'count': ...}, ...}.
        """
        summary = {}
        k = self.SPEED_CALIBRATION_FACTOR
        for d, data in self._direction_stats.items():
            speeds = data['speeds']
            if speeds:
                # Lay trung binh tren 60 mau gan nhat (~ vai giay)
                recent = speeds[-60:]
                avg_ms = sum(recent) / len(recent)
            else:
                avg_ms = 0.0
            summary[d] = {
                'speed_kmh': avg_ms * 3.6 * k,
                'count': len(data['track_ids']),
            }
        return summary

    def _pose_worker(self, pose_estimator):
        """Luồng riêng: nhận frame từ hàng đợi và chạy pose estimation"""
        while self._running:
            # Lấy frame từ hàng đợi
            with self.frame_lock:
                if not self.frame_queue:
                    time.sleep(0.01)
                    continue
                frame_data = self.frame_queue.popleft()
            
            frame = frame_data['frame']
            frame_id = frame_data['frame_id']
            timestamp = frame_data['timestamp']
            
            try:
                persons = pose_estimator.extract_keypoints(frame)
                with self.pose_results_lock:
                    self.pose_results[frame_id] = {
                        'persons': persons,
                        'timestamp': timestamp
                    }
                    # Xóa các kết quả cũ quá 30 frame
                    old_ids = [fid for fid in self.pose_results if fid < frame_id - 30]
                    for fid in old_ids:
                        del self.pose_results[fid]
            except Exception:
                pass

    def _draw_all(self, frame, tracks, risk_results, world_positions, ipm=None, pose_persons=None):
        """Ve tat ca ket qua len frame - co BEV that tu IPM."""
        result = frame.copy()
        h, w = result.shape[:2]
        
        # Ve IPM debug overlay
        if ipm and ipm.is_calibrated:
            try:
                result = ipm.create_debug_overlay(result)
            except:
                pass

        # Tao BEV image
        if ipm and ipm.is_calibrated:
            try:
                bev_img = ipm.get_bev_image(frame)
                for track in tracks:
                    if track.track_id in world_positions:
                        wp = world_positions[track.track_id]
                        risk = risk_results.get(track.track_id)
                        bev_px = int(wp.x * ipm.scale)
                        bev_py = int(wp.y * ipm.scale)
                        if risk:
                            if risk.is_accident or risk.final_score >= 0.65:
                                color = (0, 0, 255)
                            elif risk.risk_level >= RiskLevel.WARNING:
                                color = (0, 165, 255)
                            else:
                                color = (0, 255, 0)
                        else:
                            color = (200, 200, 200)
                        if 0 <= bev_px < bev_img.shape[1] and 0 <= bev_py < bev_img.shape[0]:
                            cv2.circle(bev_img, (bev_px, bev_py), 6, color, -1)
                            cv2.putText(bev_img, str(track.track_id),
                                       (bev_px + 8, bev_py + 4),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
                bev_h_bev, bev_w_bev = bev_img.shape[:2]
                thumb_scale = 0.25
                new_w = int(bev_w_bev * thumb_scale)
                new_h = int(bev_h_bev * thumb_scale)
                bev_small = cv2.resize(bev_img, (new_w, new_h))
                cv2.rectangle(bev_small, (0, 0), (new_w-1, new_h-1), (255, 255, 255), 2)
                x_offset = w - new_w - 10
                y_offset = 50
                if (y_offset + new_h <= h and x_offset + new_w <= w and 
                    y_offset >= 0 and x_offset >= 0):
                    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = bev_small
                    cv2.putText(result, "BEV (Bird's Eye View)", (x_offset, y_offset - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            except Exception:
                pass
        
        # Ve pose skeleton
        if pose_persons and self.pose_estimator:
            result = self.pose_estimator.draw_skeleton(result, pose_persons)
        
        # Ve tracks + BBOX
        for track in tracks:
            x1, y1, x2, y2 = track.bbox_xyxy.astype(int)
            risk = risk_results.get(track.track_id)
            if risk:
                if risk.is_accident or risk.final_score >= 0.65:
                    color = (0, 0, 255)
                    thickness = 3
                elif risk.risk_level >= RiskLevel.WARNING:
                    color = (0, 165, 255)
                    thickness = 2
                else:
                    color = (0, 255, 0)
                    thickness = 2
            else:
                color = (180, 180, 180)
                thickness = 1
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
            label = f"ID:{track.track_id} {track.class_name}"
            if risk:
                label += f" [{risk.final_score:.2f}]"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2)
            cv2.rectangle(result, (x1, y1 - lh - 8), (x1 + lw + 6, y1 - 2), color, -1)
            cv2.putText(result, label, (x1 + 3, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
            if risk and risk.is_accident:
                acc_text = "ACCIDENT DETECTED"
                (tw, th), _ = cv2.getTextSize(acc_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                acc_x = x1 + (x2 - x1)//2 - tw//2
                acc_y = y2 + 22
                cv2.rectangle(result, (acc_x - 6, acc_y - th - 6),
                            (acc_x + tw + 6, acc_y + 6), (0, 0, 200), -1)
                cv2.putText(result, acc_text, (acc_x, acc_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if len(track.trajectory) > 1:
                pts = [(int(p[0]), int(p[1])) for p in list(track.trajectory)[-30:]]
                for i in range(1, len(pts)):
                    alpha = i / len(pts)
                    pt_color = tuple(int(c * alpha) for c in color)
                    cv2.line(result, pts[i-1], pts[i], pt_color, 1)
        
        # Header + congestion status
        header_bg = np.zeros((36, w, 3), dtype=np.uint8)
        header_bg[:] = (20, 20, 40)
        cv2.putText(header_bg, "TRAFFIC AI SYSTEM | LINH NAM - HA NOI",
                   (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        if ipm and ipm.is_calibrated:
            congestion = ipm.get_congestion_level(len(tracks))
            status_text = f"| {congestion['level']} ({len(tracks)} xe, {congestion['density']:.1f}/100m²)"
            cv2.putText(header_bg, status_text, (w - 500, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, congestion['color'], 1)
        result = np.vstack([header_bg, result])

        # Hiển thị trạng thái POSE
        status = "POSE ON" if (pose_persons and len(pose_persons) > 0) else "POSE OFF"
        color = (0, 255, 0) if status == "POSE ON" else (0, 0, 255)
        cv2.putText(result, status, (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return result

    def pause(self):
        self._paused = not self._paused
        return self._paused

    def stop(self):
        self._running = False
        if self._pose_thread and self._pose_thread.is_alive():
            self._pose_thread.join(timeout=2)
        self.wait(5000)


# ... (phần còn lại của file giữ nguyên)


# ============================================================
# Video Preview Widget
# ============================================================
class VideoPreviewWidget(QFrame):
    video_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumSize(640, 400)
        self.setObjectName("VideoPreview")
        self.setStyleSheet("""
            QFrame#VideoPreview {
                background-color: #ffffff;
                border: 2px dashed #b0b8c1;
                border-radius: 4px;
            }
            QFrame#VideoPreview:hover {
                border-color: #1a3a5c;
                background-color: #fafbfc;
            }
        """)
        self._video_path = None
        self._has_video = False
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(16)

        self.empty_widget = QWidget()
        empty_layout = QVBoxLayout(self.empty_widget)
        empty_layout.setAlignment(Qt.AlignCenter)
        empty_layout.setSpacing(12)

        title = QLabel("CHUA CO DU LIEU VIDEO")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #5a6c7d; font-size: 18px; font-weight: bold; background: transparent;")
        empty_layout.addWidget(title)

        subtitle = QLabel("Keo tha video vao khu vuc nay hoac chon file de bat dau")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #8a9aa8; font-size: 13px; background: transparent;")
        empty_layout.addWidget(subtitle)

        btn = QPushButton("Chon Video...")
        btn.setFixedWidth(200)
        btn.clicked.connect(self._browse)
        btn_wrapper = QHBoxLayout()
        btn_wrapper.setAlignment(Qt.AlignCenter)
        btn_wrapper.addWidget(btn)
        empty_layout.addLayout(btn_wrapper)

        hint = QLabel("Dinh dang ho tro: MP4, AVI, MOV, MKV")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet("color: #a0aab4; font-size: 11px; background: transparent;")
        empty_layout.addWidget(hint)

        layout.addWidget(self.empty_widget)

        self.video_widget = QWidget()
        video_layout = QVBoxLayout(self.video_widget)
        video_layout.setContentsMargins(0, 0, 0, 0)
        self.video_display = QLabel()
        self.video_display.setAlignment(Qt.AlignCenter)
        self.video_display.setStyleSheet("background-color: #000000;")
        video_layout.addWidget(self.video_display)
        self.video_widget.hide()

    def _browse(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Chon Video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if file_path:
            self.load_video(file_path)

    def load_video(self, path):
        self._video_path = path
        self._has_video = True
        self.empty_widget.hide()
        self.video_widget.show()
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                self._show_frame(frame)
        self.video_selected.emit(path)

    def _show_frame(self, frame):
        h, w = frame.shape[:2]
        mw, mh = self.width() - 20, self.height() - 20
        if mw > 0 and mh > 0:
            scale = min(mw / w, mh / h)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, w * ch, QImage.Format_RGB888)
        self.video_display.setPixmap(QPixmap.fromImage(qt_img))

    def update_frame(self, frame):
        if self._has_video:
            self._show_frame(frame)

    @property
    def video_path(self):
        return self._video_path

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self.setStyleSheet("QFrame#VideoPreview { background-color: #e8ecf0; border: 2px solid #1a3a5c; border-radius: 4px; }")

    def dragLeaveEvent(self, e):
        self.setStyleSheet("QFrame#VideoPreview { background-color: #ffffff; border: 2px dashed #b0b8c1; border-radius: 4px; }")

    def dropEvent(self, e: QDropEvent):
        urls = e.mimeData().urls()
        if urls:
            fp = urls[0].toLocalFile()
            if os.path.isfile(fp):
                self.load_video(fp)
        self.dragLeaveEvent(e)

    def reset(self):
        self._video_path = None
        self._has_video = False
        self.video_widget.hide()
        self.empty_widget.show()


# ============================================================
# Workflow Control Bar
# ============================================================
class WorkflowControlBar(QFrame):
    play_clicked = pyqtSignal()
    pause_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("WorkflowBar")
        self.setStyleSheet("QFrame#WorkflowBar { background-color: #ffffff; border: 1px solid #d0d5db; border-radius: 3px; padding: 6px; }")
        self.setFixedHeight(56)
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(8)

        self.btn_play = QPushButton("Bat dau phan tich AI")
        self.btn_play.clicked.connect(lambda: self.play_clicked.emit())
        layout.addWidget(self.btn_play)

        self.btn_pause = QPushButton("Tam dung")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(lambda: self.pause_clicked.emit())
        layout.addWidget(self.btn_pause)

        self.btn_stop = QPushButton("Dung")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("QPushButton { background-color: #8b0000; } QPushButton:hover { background-color: #a00000; }")
        self.btn_stop.clicked.connect(lambda: self.stop_clicked.emit())
        layout.addWidget(self.btn_stop)

        layout.addStretch()

        status_layout = QVBoxLayout()
        self.status_label = QLabel("San sang")
        self.status_label.setStyleSheet("color: #5a6c7d; font-size: 11px; background: transparent;")
        self.status_label.setAlignment(Qt.AlignRight)
        status_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(240)
        self.progress_bar.setFixedHeight(10)
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.progress_bar)
        layout.addLayout(status_layout)

    def set_state(self, playing, paused=False):
        if playing and not paused:
            self.btn_play.setEnabled(False)
            self.btn_pause.setEnabled(True)
            self.btn_stop.setEnabled(True)
            self.status_label.setText("Dang phan tich...")
        elif playing and paused:
            self.btn_play.setEnabled(True)
            self.btn_play.setText("Tiep tuc")
            self.btn_pause.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.status_label.setText("Da tam dung")
        else:
            self.btn_play.setEnabled(True)
            self.btn_play.setText("Bat dau phan tich AI")
            self.btn_pause.setEnabled(False)
            self.btn_stop.setEnabled(False)
            self.status_label.setText("San sang")

    def update_progress(self, cur, total):
        if total > 0:
            pct = int(cur / total * 100)
            self.progress_bar.setValue(pct)
            self.status_label.setText(f"Khung hinh: {cur}/{total} ({pct}%)")


# ============================================================
# Main Window
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("He thong Giam sat Giao thong Linh Nam - Nghien cuu khoa hoc 2026")
        self.setGeometry(80, 50, 1500, 880)
        self.video_thread = None
        self.current_video_path = None
        self.alert_history = []
        self._latest_stats = {}   # luu stats moi nhat de dung trong export report
        self._init_ui()
        self._setup_toolbar()
        self._setup_statusbar()
        self._connect_signals()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(8)

        title = QLabel("HE THONG GIAM SAT VA PHAN TICH GIAO THONG THONG MINH")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #1a3a5c; font-size: 16px; font-weight: bold; padding: 6px; background-color: #ffffff; border: 1px solid #d0d5db; border-radius: 3px;")
        main_layout.addWidget(title)

        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)
        self.video_preview = VideoPreviewWidget()
        left_layout.addWidget(self.video_preview, 1)
        self.workflow_bar = WorkflowControlBar()
        left_layout.addWidget(self.workflow_bar)
        splitter.addWidget(left)

        right = QWidget()
        right.setMaximumWidth(400)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        self.dashboard = DashboardWidget()
        right_layout.addWidget(self.dashboard)

        action_group = QGroupBox("Thao tac")
        action_layout = QVBoxLayout(action_group)
        self.btn_demo_alert = QPushButton("Kich hoat su co mau (Demo)")
        self.btn_demo_alert.clicked.connect(self._trigger_demo_alert)
        action_layout.addWidget(self.btn_demo_alert)
        self.btn_export = QPushButton("Xuat bao cao ket qua")
        self.btn_export.clicked.connect(self._export_report)
        action_layout.addWidget(self.btn_export)
        right_layout.addWidget(action_group)

        self.log_text = QLabel("San sang - Chon video de bat dau")
        self.log_text.setWordWrap(True)
        self.log_text.setStyleSheet("color: #5a6c7d; background-color: #ffffff; padding: 8px; border: 1px solid #d0d5db; border-radius: 3px; font-size: 11px;")
        self.log_text.setMinimumHeight(50)
        right_layout.addWidget(self.log_text)

        splitter.addWidget(right)
        splitter.setSizes([1020, 400])
        main_layout.addWidget(splitter)

    def _setup_toolbar(self):
        toolbar = QToolBar("Cong cu")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        toolbar.addAction(QAction("Mo Video", self, triggered=self._open_video))
        toolbar.addAction(QAction("Mo RTSP", self, triggered=self._open_rtsp))
        toolbar.addAction(QAction("Chay Demo", self, triggered=self._run_demo))
        toolbar.addSeparator()
        lbl = QLabel("  Nguong: ")
        lbl.setStyleSheet("color: #2c3e50; font-size: 12px;")
        toolbar.addWidget(lbl)
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(10, 95)
        self.threshold_slider.setValue(65)
        self.threshold_slider.setFixedWidth(130)
        toolbar.addWidget(self.threshold_slider)
        self.threshold_value = QLabel("0.65")
        self.threshold_value.setStyleSheet("color: #1a3a5c; font-weight: bold; min-width: 35px;")
        self.threshold_slider.valueChanged.connect(lambda v: self.threshold_value.setText(f"{v/100:.2f}"))
        toolbar.addWidget(self.threshold_value)

    def _setup_statusbar(self):
        self.statusbar = QStatusBar()
        self.statusbar.showMessage("San sang | Chua co video")
        self.setStatusBar(self.statusbar)

    def _connect_signals(self):
        self.video_preview.video_selected.connect(self._on_video_loaded)
        self.workflow_bar.play_clicked.connect(self._start_processing)
        self.workflow_bar.pause_clicked.connect(self._toggle_pause)
        self.workflow_bar.stop_clicked.connect(self._stop_processing)

    def _open_rtsp(self):
        from PyQt5.QtWidgets import QInputDialog
        url, ok = QInputDialog.getText(
            self, "RTSP Stream",
            "Nhap URL RTSP:",
            text="rtsp://admin:admin@192.168.1.100:554/live"
        )
        if ok and url.strip():
            self.video_preview.load_video(url.strip())
    
    def _open_video(self):
        fp, _ = QFileDialog.getOpenFileName(self, "Chon Video", "", "Video (*.mp4 *.avi *.mov *.mkv);;All (*)")
        if fp:
            self.video_preview.load_video(fp)

    def _open_source(self, src):
        self.video_preview.load_video(str(src))

    def _run_demo(self):
        for p in ["videos/linh_nam.mp4", "videos/demo.mp4"]:
            full = Path(__file__).parent.parent / p
            if full.exists():
                self.video_preview.load_video(str(full))
                return
        self._log("Khong tim thay video demo")

    def _on_video_loaded(self, path):
        self.current_video_path = path
        name = "Webcam" if path == "0" else os.path.basename(path)
        self.statusbar.showMessage(f"Da tai: {name}")
        self.workflow_bar.set_state(False)
        self.dashboard.set_mode("ready", f"Da tai: {name}")

    def _start_processing(self):
        if not self.current_video_path:
            QMessageBox.warning(self, "Chua co video", "Vui long chon video truoc.")
            return
        
        self._alert_dialog_shown = False
        self.alert_history.clear()
        self._last_stats = {}       
        self._stats_accum = {}    
        
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
        self.video_thread = FullPipelineThread(self.current_video_path)
        self.video_thread.frame_ready.connect(self._on_frame)
        self.video_thread.stats_ready.connect(self._on_stats)
        self.video_thread.processing_error.connect(self._on_error)
        self.video_thread.alert_detected.connect(self._on_alert)
        self.video_thread.video_finished.connect(self._on_finished)
        self.video_thread.log_message.connect(self._log)
        self.video_thread.event_logged.connect(self.dashboard.add_event)
        self.video_thread.start()
        self.workflow_bar.set_state(True)
        self.dashboard.set_mode("processing")
        self.statusbar.showMessage("Dang xu ly...")
        self._log("Bat dau pipeline...")

    def _toggle_pause(self):
        if self.video_thread:
            p = self.video_thread.pause()
            self.workflow_bar.set_state(True, p)

    def _stop_processing(self):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
        self.workflow_bar.set_state(False)
        self.workflow_bar.update_progress(0, 100)
        self.dashboard.set_mode("ready", "Da dung")
        self.statusbar.showMessage("Da dung")

    def _on_frame(self, frame, cur, total):
        self.video_preview.update_frame(frame)
        self.workflow_bar.update_progress(cur, total)

    def _on_stats(self, stats):
        self.dashboard.update_stats(stats)

        if not hasattr(self, '_stats_accum'):
            self._stats_accum = {}

        for k, v in stats.items():
            if v is None:
                continue

            # zone_area: KHONG cho 0 ghi de gia tri da co
            if k == 'zone_area' and (v == 0 or v == 0.0):
                if self._stats_accum.get(k, 0) > 0:
                    continue

            # congestion_level: KHONG cho UNKNOWN ghi de level da co
            if k == 'congestion_level' and v == 'UNKNOWN':
                if self._stats_accum.get(k, 'UNKNOWN') != 'UNKNOWN':
                    continue

            # congestion_density: KHONG cho 0 ghi de gia tri > 0
            if k == 'congestion_density' and v == 0:
                if self._stats_accum.get(k, 0) > 0:
                    continue

            # Counters: luon LAY MAX (so tich luy)
            if k in ('total_vehicles_unique', 'total_detections',
                    'alerts', 'frame_count'):
                self._stats_accum[k] = max(self._stats_accum.get(k, 0), v)
            else:
                self._stats_accum[k] = v

        self._last_stats = self._stats_accum

    def _on_error(self, msg):
        QMessageBox.critical(self, "Loi", msg)
        self._stop_processing()

    def _on_alert(self, data, frame):
        self.alert_history.append(data)
        self.dashboard.add_event("ACCIDENT", 
            f"Track {data['track_id']} - Score: {data['final_score']:.2f}")
        dlg = IncidentDialog(data, frame, self)
        dlg.exec_()

    def _on_finished(self):
        self.workflow_bar.set_state(False)
        self.dashboard.set_mode("completed")
        self.statusbar.showMessage("Hoan thanh")
        self._log("Pipeline hoan thanh!")

    def _trigger_demo_alert(self):
        """Load video demo va chay pipeline."""
        from pathlib import Path
        demo_path = Path(__file__).parent.parent / "videos" / "demo_demo.mp4"
        if not demo_path.exists():
            QMessageBox.warning(
                self, "Khong tim thay video demo",
                f"Vui long dat file demo tai: {demo_path}"
            )
            return

        # Tai vao preview va chay pipeline
        self.video_preview.load_video(str(demo_path))
        self.current_video_path = str(demo_path)
        self._log(f"Loaded demo: {demo_path.name}")
        # Chay pipeline ngay
        self._start_processing()

    def _export_report(self):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        fp, _ = QFileDialog.getSaveFileName(self, "Xuat bao cao", f"Bao_cao_{ts}.html", "HTML (*.html);;Text (*.txt)")
        if not fp:
            return
        try:
            data = self._collect_report_data()
            if fp.endswith('.html'):
                self._export_html(fp, data)
            else:
                self._export_text(fp, data)
            QMessageBox.information(self, "OK", f"Da xuat: {fp}")
        except Exception as e:
            QMessageBox.warning(self, "Loi", str(e))

    def _collect_report_data(self):
        # === Congestion theo huong: lay tu stats moi nhat cua pipeline ===
        # Chi tinh tu cac track NAM TRONG vung IPM, van toc do tu world coords.
        dir_stats = (self._latest_stats or {}).get('direction_stats', {}) or {}
        n = dir_stats.get('N', {'speed_kmh': 0.0, 'count': 0})
        s = dir_stats.get('S', {'speed_kmh': 0.0, 'count': 0})
        e = dir_stats.get('E', {'speed_kmh': 0.0, 'count': 0})
        w = dir_stats.get('W', {'speed_kmh': 0.0, 'count': 0})
        # === Tong hop tac nghen tu IPM zone ===
        last_stats = getattr(self, '_last_stats', {}) or {}
        congestion = {
            'level': last_stats.get('congestion_level', 'UNKNOWN'),
            'density': last_stats.get('congestion_density', 0.0),
            'avg_speed_kmh': last_stats.get('avg_speed', 0.0) * 3.6,
            'vehicle_count': last_stats.get('vehicles', 0),
            'total_unique': last_stats.get('total_vehicles_unique', 0),
            'zone_area': last_stats.get('zone_area', 0.0),
        }

        # Thu thập event log từ dashboard
        events = []
        if hasattr(self.dashboard, '_event_log'):
            events = list(self.dashboard._event_log)

        data = {
            'timestamp': datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
            'system_name': 'He thong Giam sat Giao thong Linh Nam',
            'version': '1.0.0 - NCKH 2026',
            'video_source': 'Khong co',
            'video_info': {},
            'detection_stats': {
                'fps_trung_binh': self.dashboard.fps_card.value_label.text(),
                'do_tre': self.dashboard.latency_card.value_label.text(),
                'gpu_usage': self.dashboard.gpu_card.value_label.text(),
                'ram_usage': '--',
                # FIX: doc tu total_vehicle_card (tong xe duy nhat),
                # khong phai vehicle_card (chi la xe hien tai trong vung).
                'tong_xe_duy_nhat': self.dashboard.total_vehicle_card.value_label.text(),
                'tong_luot_detect': self.dashboard.detection_card.value_label.text(),
                'tong_canh_bao': self.dashboard.alert_card.value_label.text(),
            },
            'congestion': congestion,
            'events': events,
            'alerts': self.alert_history,
            'config': {
                'nguong_phat_hien': f"{self.threshold_slider.value()/100:.2f}",
                'mo_hinh_phat_hien': 'YOLOv8n (Ultralytics)',
                'mo_hinh_theo_doi': 'DeepSORT (lambda=0.7, max_age=60)',
                'phan_loai_su_co': 'LightGBM + IPM + Dac trung Dong hoc',
                'dac_trung': '20-chieu (Van toc, Gia toc, Goc nghieng, Khoang cach)',
                'tang_toc_phan_cung': 'TensorRT FP16',
                'thiet_bi_xu_ly': 'NVIDIA RTX 3060 (CUDA)',
                'fps_muc_tieu': '>= 25 FPS (Full HD)',
            }
        }
        if self.current_video_path:
            data['video_source'] = 'Webcam' if self.current_video_path == '0' else os.path.basename(self.current_video_path)
            if self.current_video_path != '0':
                try:
                    cap = cv2.VideoCapture(self.current_video_path)
                    data['video_info'] = {
                        'do_phan_giai': f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
                        'fps_goc': f"{cap.get(cv2.CAP_PROP_FPS):.1f}",
                        'tong_khung_hinh': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                        'thoi_luong': f"{int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1))}s"
                    }
                    cap.release()
                except:
                    pass
        return data

    def _export_html(self, fp, data):
        stats = data.get('detection_stats', {})
        config = data.get('config', {})
        video_info = data.get('video_info', {})
        alerts = data.get('alerts', [])
        congestion = data.get('congestion', {})
        
        # Video info rows
        video_rows = f"<tr><td>Nguon:</td><td><b>{data['video_source']}</b></td></tr>"
        for key, val in video_info.items():
            label = key.replace('_', ' ').title()
            video_rows += f"<tr><td>{label}:</td><td><b>{val}</b></td></tr>"
        
        # === Tong hop muc do tac nghen vung IPM ===
        level_colors = {
            'THONG THOANG': '#27ae60',
            'DONG VUA':     '#f39c12',
            'DONG DUC':     '#e67e22',
            'TAC NGHEN':    '#e74c3c',
            'UNKNOWN':      '#b0b8c1',
        }
        level = congestion.get('level', 'UNKNOWN')
        color = level_colors.get(level, '#b0b8c1')

        level_pct = {
            'THONG THOANG': 25, 'DONG VUA': 55,
            'DONG DUC': 75, 'TAC NGHEN': 90, 'UNKNOWN': 0,
        }
        pct = level_pct.get(level, 0)
        
        # Config rows
        config_rows = ""
        for key, val in config.items():
            label = key.replace('_', ' ').title()
            config_rows += f"<tr><td>{label}:</td><td><b>{val}</b></td></tr>"
        
        # Alert rows
        alert_rows = ""
        if alerts:
            for i, alert in enumerate(alerts[-30:]):
                violations_str = ', '.join(alert.get('violations', [])[:2])
                time_str = datetime.fromtimestamp(alert.get('timestamp', 0)).strftime('%H:%M:%S')
                score = alert.get('final_score', 0)
                score_color = "#e74c3c" if score > 0.8 else "#f39c12" if score > 0.6 else "#2c3e50"
                alert_rows += f"""
            <tr>
                <td>{i+1}</td>
                <td>{time_str}</td>
                <td>Track #{alert.get('track_id', '?')}</td>
                <td style="color:{score_color}; font-weight:bold;">{score:.3f}</td>
                <td>{alert.get('risk_level', '?')}</td>
                <td style="font-size:11px;">{violations_str}</td>
            </tr>"""
        else:
            alert_rows = '<tr><td colspan="6" style="text-align:center; color:#8a9aa8; padding:20px;">Khong co canh bao nao duoc ghi nhan</td></tr>'
        
        # Event log rows
        event_rows = ""
        events = data.get('events', [])
        if events:
            for ts, etype, desc in events[-30:]:
                row_color = "#fdf2f2" if etype == "ACCIDENT" else "#ffffff"
                text_color = "#e74c3c" if etype == "ACCIDENT" else "#2c3e50"
                event_rows += f"""
            <tr style="background:{row_color};">
                <td style="width:90px;">{ts}</td>
                <td style="width:100px; color:{text_color}; font-weight:bold;">[{etype}]</td>
                <td>{desc}</td>
            </tr>"""
        else:
            event_rows = '<tr><td colspan="3" style="text-align:center; color:#8a9aa8;">Chua co su kien nao</td></tr>'
        
        html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>Bao cao Phan tich Giao thong - Linh Nam - Ha Noi</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: 'Times New Roman', 'Segoe UI', serif;
    color: #2c3e50;
    background: #f5f6fa;
    padding: 40px;
}}
.container {{ max-width: 950px; margin: 0 auto; }}
.header {{
    text-align: center;
    padding: 40px 30px;
    background: linear-gradient(135deg, #ffffff 0%, #f0f3f7 100%);
    border: 2px solid #1a3a5c;
    border-radius: 6px;
    margin-bottom: 24px;
}}
.header h1 {{ font-size: 26px; color: #1a3a5c; margin-bottom: 8px; }}
.header .subtitle {{ font-size: 16px; color: #2c3e50; margin-bottom: 4px; }}
.header .meta {{ color: #5a6c7d; font-size: 13px; margin-top: 6px; }}
.card {{
    background: #ffffff;
    border: 1px solid #d0d5db;
    border-radius: 6px;
    padding: 24px;
    margin-bottom: 20px;
    page-break-inside: avoid;
}}
.card h2 {{
    color: #1a3a5c;
    font-size: 18px;
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 3px solid #1a3a5c;
}}
.card h3 {{
    color: #2c3e50;
    font-size: 14px;
    margin: 16px 0 10px 0;
}}

/* Stats */
.stats-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-bottom: 16px;
}}
.stat-box {{
    text-align: center;
    padding: 20px 14px;
    background: #f8f9fb;
    border: 1px solid #e0e4e8;
    border-radius: 6px;
}}
.stat-box.highlight {{
    background: #f0f4ff;
    border-color: #1a3a5c;
}}
.stat-box .number {{
    display: block;
    font-size: 34px;
    font-weight: bold;
    color: #1a3a5c;
}}
.stat-box .label {{
    display: block;
    color: #5a6c7d;
    font-size: 12px;
    margin-top: 6px;
}}
.stat-box .sub {{
    display: block;
    color: #8a9aa8;
    font-size: 10px;
    margin-top: 2px;
}}

/* Tables */
.info-table {{ width: 100%; border-collapse: collapse; }}
.info-table td {{
    padding: 10px 14px;
    border-bottom: 1px solid #e8ecf0;
    font-size: 13px;
}}
.info-table td:first-child {{
    color: #5a6c7d;
    width: 200px;
    font-weight: bold;
}}

/* Alert Table */
.alert-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
}}
.alert-table th {{
    background: #f0f3f7;
    color: #2c3e50;
    padding: 8px 10px;
    text-align: left;
    border-bottom: 2px solid #d0d5db;
    font-size: 11px;
    font-weight: bold;
}}
.alert-table td {{
    padding: 7px 10px;
    border-bottom: 1px solid #e8ecf0;
}}
.alert-table tr:hover {{ background: #fafbfc; }}

/* Event Table */
.event-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
}}
.event-table th {{
    background: #f0f3f7;
    color: #2c3e50;
    padding: 8px 10px;
    text-align: left;
    border-bottom: 2px solid #d0d5db;
    font-size: 11px;
}}
.event-table td {{
    padding: 6px 10px;
    border-bottom: 1px solid #e8ecf0;
}}

/* Congestion */
.congestion-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
}}
.congestion-table td {{
    padding: 6px 10px;
    border-bottom: 1px solid #e8ecf0;
}}

.footer {{
    text-align: center;
    color: #8a9aa8;
    font-size: 12px;
    padding: 30px 20px;
    margin-top: 10px;
    border-top: 2px solid #d0d5db;
}}
.footer .highlight {{ color: #1a3a5c; font-weight: bold; }}
.footer .members {{ font-size: 11px; color: #5a6c7d; margin-top: 6px; }}

@media print {{
    body {{ padding: 20px; }}
    .card {{ page-break-inside: avoid; }}
}}
</style>
</head>
<body>
<div class="container">

<!-- ============ HEADER ============ -->
<div class="header">
    <h1>{data['system_name']}</h1>
    <p class="subtitle">BAO CAO PHAN TICH GIAO THONG TU DONG</p>
    <p class="meta">Thoi gian xuat bao cao: {data['timestamp']}</p>
    <p class="meta">Phien ban he thong: {data['version']}</p>
</div>

<!-- ============ KET QUA TONG HOP ============ -->
<div class="card">
    <h2>KET QUA TONG HOP</h2>
    
    <h3>Hieu nang He thong</h3>
    <div class="stats-grid">
        <div class="stat-box highlight">
            <span class="number">{stats.get('fps_trung_binh', '--')}</span>
            <span class="label">FPS Trung binh</span>
            <span class="sub">Muc tieu: >= 25 FPS</span>
        </div>
        <div class="stat-box">
            <span class="number">{stats.get('do_tre', '--')} ms</span>
            <span class="label">Do tre xu ly</span>
            <span class="sub">Detection + Tracking</span>
        </div>
        <div class="stat-box">
            <span class="number">{stats.get('gpu_usage', '--')}%</span>
            <span class="label">GPU Usage</span>
            <span class="sub">{stats.get('ram_usage', '--')} GB RAM</span>
        </div>
    </div>
    
    <h3>Thong ke Giao thong</h3>
    <div class="stats-grid">
        <div class="stat-box">
            <span class="number">{stats.get('tong_xe_duy_nhat', '--')}</span>
            <span class="label">Tong xe duy nhat</span>
            <span class="sub">Track ID rieng biet</span>
        </div>
        <div class="stat-box">
            <span class="number">{stats.get('tong_luot_detect', '--')}</span>
            <span class="label">Tong luot phat hien</span>
            <span class="sub">Detection events</span>
        </div>
        <div class="stat-box">
            <span class="number">{stats.get('tong_canh_bao', '--')}</span>
            <span class="label">Tong canh bao</span>
            <span class="sub">Su co giao thong</span>
        </div>
    </div>
</div>

<!-- ============ MUC DO TAC NGHEN VUNG IPM ============ -->
<div class="card">
    <h2>MUC DO TAC NGHEN (VUNG QUAN SAT IPM)</h2>

    <div style="display: grid; grid-template-columns: 1fr 2fr; gap: 20px; align-items: center;">
        <div style="text-align: center; padding: 24px; background: {color}15;
                    border: 2px solid {color}; border-radius: 8px;">
            <div style="font-size: 13px; color: #5a6c7d; margin-bottom: 8px;">Trang thai</div>
            <div style="font-size: 22px; font-weight: bold; color: {color};">{level}</div>
        </div>
        <table class="info-table">
            <tr><td>Toc do trung binh:</td><td><b>{congestion.get('avg_speed_kmh', 0):.1f} km/h</b></td></tr>
            <tr><td>Mat do phuong tien:</td><td><b>{congestion.get('density', 0):.2f} xe / 100m²</b></td></tr>
            <tr><td>Xe trong vung tai thoi diem cuoi:</td><td><b>{congestion.get('vehicle_count', 0)} xe</b></td></tr>
            <tr><td>Tong xe duy nhat di qua:</td><td><b>{congestion.get('total_unique', 0)} xe</b></td></tr>
            <tr><td>Dien tich vung quan sat:</td><td><b>{congestion.get('zone_area', 0):.0f} m²</b></td></tr>
        </table>
    </div>

    <div style="margin-top: 16px;">
        <div style="background: #e8ecf0; border-radius: 4px; height: 18px; width: 100%;">
            <div style="background: {color}; width: {pct}%; height: 18px;
                        border-radius: 4px; transition: width 0.5s;"></div>
        </div>
        <div style="margin-top: 6px; font-size: 10px; color: #8a9aa8;">
            Chu thich:
            <span style="color: #27ae60;">Thong thoang &lt;1 xe/100m²</span> |
            <span style="color: #f39c12;">Dong vua 1-2.5</span> |
            <span style="color: #e67e22;">Dong duc 2.5-4</span> |
            <span style="color: #e74c3c;">Tac nghen &gt;4</span>
        </div>
    </div>
</div>

<!-- ============ THONG TIN VIDEO ============ -->
<div class="card">
    <h2>THONG TIN VIDEO NGUON</h2>
    <table class="info-table">
        {video_rows}
    </table>
</div>

<!-- ============ DANH SACH CANH BAO ============ -->
<div class="card">
    <h2>DANH SACH CANH BAO SU CO ({len(alerts)})</h2>
    <table class="alert-table">
        <tr>
            <th>#</th>
            <th>Thoi gian</th>
            <th>Doi tuong</th>
            <th>Score</th>
            <th>Muc do</th>
            <th>Vi pham</th>
        </tr>
        {alert_rows}
    </table>
</div>

<!-- ============ NHAT KY SU KIEN ============ -->
<div class="card">
    <h2>NHAT KY SU KIEN ({len(events)})</h2>
    <table class="event-table">
        <tr>
            <th>Thoi gian</th>
            <th>Loai</th>
            <th>Mo ta</th>
        </tr>
        {event_rows}
    </table>
</div>

<!-- ============ CAU HINH HE THONG ============ -->
<div class="card">
    <h2>CAU HINH HE THONG</h2>
    <table class="info-table">
        {config_rows}
    </table>
</div>

<!-- ============ FOOTER ============ -->
<div class="footer">
    <p class="highlight">DE TAI NGHIEN CUU KHOA HOC SINH VIEN 2026</p>
    <p><strong>Thiet ke He thong Camera AI Phan tich Giao thong Khu vuc Linh Nam - Ha Noi</strong></p>
    <p>Truong Dai hoc Kinh te - Ky thuat Cong nghiep | Khoa Cong nghe Thong tin</p>
    <p style="margin-top:10px;">
        <span class="highlight">GVHD:</span> Th.S Hoang Thi Phuong &nbsp;|&nbsp; 
        <span class="highlight">Chu nhiem de tai:</span> Tran Phuong Thao (DHTI17A4HN)
    </p>
    <p class="members">
        Thanh vien: Pham Duc Minh | Vu Trong Khue | Dang Hai Lam | Nguyen Ngoc Tu
    </p>
    <p style="margin-top:12px; font-size:10px; color:#b0b8c1;">
        Bao cao duoc xuat tu dong boi Traffic AI System - Phien ban 1.0.0
    </p>
</div>

</div>
</body>
</html>"""
        
        with open(fp, 'w', encoding='utf-8') as f:
            f.write(html)

    def _export_text(self, fp, data):
        stats = data.get('detection_stats', {})
        config = data.get('config', {})
        alerts = data.get('alerts', [])
        events = data.get('events', [])
        congestion = data.get('congestion', {})
        
        lines = [
            "=" * 70,
            f"  {data['system_name']}",
            "  BAO CAO PHAN TICH GIAO THONG",
            "=" * 70,
            f"  Thoi gian: {data['timestamp']}",
            f"  Phien ban: {data['version']}",
            "",
            "-" * 70,
            "  KET QUA TONG HOP",
            "-" * 70,
            f"  FPS Trung binh:        {stats.get('fps_trung_binh', '--')}",
            f"  Do tre xu ly:          {stats.get('do_tre', '--')} ms",
            f"  GPU Usage:             {stats.get('gpu_usage', '--')}%",
            f"  Tong xe duy nhat:      {stats.get('tong_xe_duy_nhat', '--')}",
            f"  Tong luot phat hien:   {stats.get('tong_luot_detect', '--')}",
            f"  Tong canh bao:         {stats.get('tong_canh_bao', '--')}",
            "",
            "-" * 70,
            "  MUC DO TAC NGHEN (VUNG QUAN SAT IPM)",
            "-" * 70,
            f"  Trang thai:               {congestion.get('level', 'UNKNOWN')}",
            f"  Toc do trung binh:        {congestion.get('avg_speed_kmh', 0):.1f} km/h",
            f"  Mat do:                   {congestion.get('density', 0):.2f} xe/100m²",
            f"  Xe trong vung (cuoi):     {congestion.get('vehicle_count', 0)} xe",
            f"  Tong xe duy nhat:         {congestion.get('total_unique', 0)} xe",
            f"  Dien tich vung:           {congestion.get('zone_area', 0):.0f} m²",
            "-" * 70,
            "  THONG TIN VIDEO",
            "-" * 70,
            f"  Nguon: {data['video_source']}",
        ]
        
        for key, val in data.get('video_info', {}).items():
            lines.append(f"  {key}: {val}")
        
        lines += ["", "-" * 70, f"  DANH SACH CANH BAO ({len(alerts)})", "-" * 70]
        
        if alerts:
            for i, alert in enumerate(alerts[-30:]):
                time_str = datetime.fromtimestamp(alert.get('timestamp', 0)).strftime('%H:%M:%S')
                violations_str = ', '.join(alert.get('violations', [])[:2])
                lines.append(
                    f"  #{i+1} | {time_str} | Track {alert.get('track_id', '?')} | "
                    f"Score: {alert.get('final_score', 0):.3f} | "
                    f"{alert.get('risk_level', '?')}"
                )
                lines.append(f"       Vi pham: {violations_str}")
        else:
            lines.append("  Khong co canh bao nao.")
        
        lines += ["", "-" * 70, f"  NHAT KY SU KIEN ({len(events)})", "-" * 70]
        
        if events:
            for ts, etype, desc in events[-30:]:
                lines.append(f"  {ts} | [{etype}] {desc}")
        else:
            lines.append("  Chua co su kien nao.")
        
        lines += ["", "-" * 70, "  CAU HINH HE THONG", "-" * 70]
        for key, val in config.items():
            lines.append(f"  {key}: {val}")
        
        lines += [
            "", "=" * 70,
            "  DE TAI NCKH 2026 - KHOA CNTT - UNETI",
            "  GVHD: Th.S Hoang Thi Phuong",
            "  Nhom SV: Tran Phuong Thao, Pham Duc Minh, Vu Trong Khue,",
            "           Dang Hai Lam, Nguyen Ngoc Tu",
            "=" * 70,
        ]
        
        with open(fp, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_text.setText(f"[{ts}] {msg}")
        print(f"[GUI] {msg}")

    def closeEvent(self, event):
        self._stop_processing()
        event.accept()
        