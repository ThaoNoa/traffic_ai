"""
ablation.py — Ablation study (chạy 4 cấu hình) trên 1-2 video có sự cố.

Cấu hình:
    #1: Pixel + Threshold       (tắt IPM, dùng pixel/s, ngưỡng cứng)
    #2: IPM + Threshold         (bật IPM, ngưỡng cứng)
    #3: IPM + Rule Engine       (bật IPM + Rule, không ML)
    #4: IPM + Rule + LightGBM   (full pipeline)

Cần 1 file JSON mô tả video eval (vd: ablation_videos.json):
{
  "incidents": [
    {"path": "videos/cadp_01.mp4", "t_start": 12.0, "t_end": 15.0},
    {"path": "videos/cadp_02.mp4", "t_start": 8.5, "t_end": 11.0}
  ],
  "normals": [
    {"path": "videos/linh_nam_normal_01.mp4"}
  ]
}

Chạy:
    cd traffic_ai
    python ablation.py --eval-list ablation_videos.json

Output:
    ablation_results.txt — Bảng F1, Recall, FPR cho từng cấu hình.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix

from config.settings import get_config
from detector.vehicle_detector import VehicleDetector
from tracker.deep_sort_tracker import DeepSORTTracker
from geometry.ipm import IPMTransformer
from features.feature_extractor import FeatureExtractor
from anomaly.rule_engine import PhysicsRuleEngine, RiskLevel
from anomaly.lgm_classifier import AccidentClassifier


WINDOW_SIZE = 25
STRIDE = 5


def sliding_windows(t_start: float, t_end: float, n_frames: int, fps: float):
    """Tạo danh sách (window_idx, label) cho từng sliding window."""
    windows = []
    for start_frame in range(0, n_frames - WINDOW_SIZE + 1, STRIDE):
        t_win_start = start_frame / fps
        t_win_end = (start_frame + WINDOW_SIZE) / fps
        # Cửa sổ là "sự cố" nếu overlap với [t_start, t_end] >= 50%
        overlap = max(0, min(t_win_end, t_end) - max(t_win_start, t_start))
        win_dur = t_win_end - t_win_start
        label = 1 if overlap >= 0.5 * win_dur else 0
        windows.append((start_frame, label))
    return windows


def run_ablation(video_path: str, ipm_enabled: bool, rule_enabled: bool, ml_enabled: bool,
                 gt_incident_times: list[tuple[float, float]]):
    """
    Chạy pipeline với cấu hình chỉ định.
    Trả về y_true, y_pred cho toàn bộ cửa sổ trượt.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Không mở được {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Tạo ground-truth windows từ tất cả các vùng sự cố
    all_windows = []
    for t_start, t_end in gt_incident_times:
        windows = sliding_windows(t_start, t_end, n_frames, fps)
        all_windows.extend(windows)
    # Merge: nếu 1 frame start xuất hiện nhiều lần, lấy max label (1 nếu có bất kỳ sự cố nào)
    window_labels = {}
    for start_frame, label in all_windows:
        window_labels[start_frame] = max(window_labels.get(start_frame, 0), label)

    # Init pipeline
    cfg = get_config()
    detector = VehicleDetector(cfg)
    _ = detector.load_model()
    tracker = DeepSORTTracker(cfg)
    tracker.initialize()
    ipm = IPMTransformer(cfg) if ipm_enabled else None
    fe = FeatureExtractor(cfg)
    re = PhysicsRuleEngine(cfg) if rule_enabled else None
    clf = AccidentClassifier(cfg) if ml_enabled else None
    if ml_enabled and clf is not None:
        try:
            clf.load_model()
        except Exception:
            pass  # fallback: không có model → chỉ rule

    # IPM calibration nếu bật
    if ipm is not None:
        ret, first = cap.read()
        if ret:
            try:
                ipm.auto_calibrate_from_frame(first)
            except Exception:
                ipm.calibrate()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    fe.reset()

    # Lưu trạng thái từng frame
    frame_scores: dict[int, float] = {}   # frame_id → final_score
    cooldown_until: dict[int, float] = {}  # track_id → t khả dụng

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t_now = frame_id / fps

        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)

        world_positions = {}
        for tk in tracks:
            u, v = tk.bottom_center
            if ipm is not None:
                try:
                    wp = ipm.pixel_to_world(u, v)
                    world_positions[tk.track_id] = wp
                except Exception:
                    continue
            else:
                # Pixel mode: dùng pixel làm đơn vị ảo
                world_positions[tk.track_id] = (float(u), float(v))

        body_leans = None
        features = fe.update(tracks, world_positions, body_leans=body_leans, frame_id=frame_id)
        
        for tid, feat_vec in features.items():
            buf = fe.get_buffer(tid)
            if buf is None:
                continue
            window = buf.get_window(WINDOW_SIZE)
            if window is None:
                continue

            # Đánh giá
            if rule_enabled and re is not None:
                risk = re.evaluate_state(window, tid, frame_id)
                if ml_enabled and clf is not None and clf.is_ready:
                    ml = clf.predict(feat_vec)
                    if ml >= 0:
                        risk = re.fuse_with_ml(risk, ml)
            else:
                # Chỉ dùng ngưỡng cứng trên vận tốc (pixel hoặc mét)
                from anomaly.rule_engine import RiskResult, RiskLevel
                vel = feat_vec[0]  # v_t
                threshold = 8.0 if ipm_enabled else 150.0  # m/s hoặc px/s
                is_acc = abs(vel) > threshold
                risk = RiskResult(
                    rule_score=min(abs(vel) / threshold, 1.0) if is_acc else 0.0,
                    ml_score=0.0,
                    final_score=min(abs(vel) / threshold, 1.0) if is_acc else 0.0,
                    risk_level=RiskLevel.ACCIDENT if is_acc else RiskLevel.NORMAL,
                    violations=[],
                )

            if t_now < cooldown_until.get(tid, -1):
                continue

            if risk.risk_level >= RiskLevel.DANGER:
                frame_scores[frame_id] = max(frame_scores.get(frame_id, 0), risk.final_score)
                cooldown_until[tid] = t_now + 5.0

        frame_id += 1

    cap.release()

    # Tạo y_true, y_pred từ các sliding windows
    y_true, y_pred = [], []
    for start_frame, label in sorted(window_labels.items()):
        if start_frame >= frame_id - WINDOW_SIZE:
            continue
        y_true.append(label)
        # Có alert trong window này không?
        has_alert = any(
            start_frame <= fid < start_frame + WINDOW_SIZE and score >= 0.5
            for fid, score in frame_scores.items()
        )
        y_pred.append(1 if has_alert else 0)

    return y_true, y_pred


def run_video_no_gt(video_path: str, ipm_enabled: bool, rule_enabled: bool, ml_enabled: bool):
    """
    Chạy pipeline trên video KHÔNG có sự cố (đo false alarm).
    Trả về: (duration_minutes, n_false_alerts)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Không mở được {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Init
    cfg = get_config()
    detector = VehicleDetector(cfg)
    _ = detector.load_model()
    tracker = DeepSORTTracker(cfg)
    tracker.initialize()
    ipm = IPMTransformer(cfg) if ipm_enabled else None
    fe = FeatureExtractor(cfg)
    re = PhysicsRuleEngine(cfg) if rule_enabled else None
    clf = AccidentClassifier(cfg) if ml_enabled else None
    if ml_enabled and clf is not None:
        try:
            clf.load_model()
        except Exception:
            pass

    if ipm is not None:
        ret, first = cap.read()
        if ret:
            try:
                ipm.auto_calibrate_from_frame(first)
            except Exception:
                ipm.calibrate()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    fe.reset()
    cooldown_until: dict[int, float] = {}
    n_alerts = 0
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t_now = frame_id / fps

        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)

        world_positions = {}
        for tk in tracks:
            u, v = tk.bottom_center
            if ipm is not None:
                try:
                    wp = ipm.pixel_to_world(u, v)
                    world_positions[tk.track_id] = wp
                except Exception:
                    continue
            else:
                world_positions[tk.track_id] = (float(u), float(v))

        features = fe.update(tracks, world_positions, body_leans=None, frame_id=frame_id)

        for tid, feat_vec in features.items():
            buf = fe.get_buffer(tid)
            if buf is None:
                continue
            window = buf.get_window(WINDOW_SIZE)
            if window is None:
                continue

            if rule_enabled and re is not None:
                risk = re.evaluate_state(window, tid, frame_id)
                if ml_enabled and clf is not None and clf.is_ready:
                    ml = clf.predict(feat_vec)
                    if ml >= 0:
                        risk = re.fuse_with_ml(risk, ml)
            else:
                from anomaly.rule_engine import RiskResult, RiskLevel
                vel = feat_vec[0]
                threshold = 8.0 if ipm_enabled else 150.0
                risk = RiskResult(
                    rule_score=min(abs(vel) / threshold, 1.0) if abs(vel) > threshold else 0.0,
                    ml_score=0.0,
                    final_score=min(abs(vel) / threshold, 1.0) if abs(vel) > threshold else 0.0,
                    risk_level=RiskLevel.ACCIDENT if abs(vel) > threshold else RiskLevel.NORMAL,
                    violations=[],
                )

            if t_now < cooldown_until.get(tid, -1):
                continue
            if risk.risk_level >= RiskLevel.DANGER:
                n_alerts += 1
                cooldown_until[tid] = t_now + 5.0

        frame_id += 1

    cap.release()
    duration = n_frames / fps / 60.0  # phút
    return duration, n_alerts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-list", required=True, help="JSON mô tả video eval")
    parser.add_argument("--out", default="ablation_results.txt")
    args = parser.parse_args()

    with open(args.eval_list, "r", encoding="utf-8") as f:
        eval_cfg = json.load(f)

    incidents = eval_cfg.get("incidents", [])
    normals = eval_cfg.get("normals", [])

    configs = [
        {"name": "#1: Pixel + Threshold",  "ipm": False, "rule": False, "ml": False},
        {"name": "#2: IPM + Threshold",    "ipm": True,  "rule": False, "ml": False},
        {"name": "#3: IPM + Rule Engine",  "ipm": True,  "rule": True,  "ml": False},
        {"name": "#4: IPM + Rule + LGBM",  "ipm": True,  "rule": True,  "ml": True},
    ]

    results = {}

    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Chạy {cfg['name']}")
        print(f"{'='*60}")

        # ── Video sự cố ──
        all_y_true, all_y_pred = [], []
        n_hit = 0
        for inc in incidents:
            try:
                tracker = DeepSORTTracker(get_config())
                tracker.initialize()
                gt_times = [(inc["t_start"], inc["t_end"])]
                y_true, y_pred = run_ablation(
                    inc["path"],
                    ipm_enabled=cfg["ipm"],
                    rule_enabled=cfg["rule"],
                    ml_enabled=cfg["ml"],
                    gt_incident_times=gt_times,
                )
                all_y_true.extend(y_true)
                all_y_pred.extend(y_pred)
                # Hit nếu có ít nhất 1 alert trong bất kỳ window sự cố nào
                windows_with_incident = [i for i, l in enumerate(y_true) if l == 1]
                hit = any(y_pred[i] == 1 for i in windows_with_incident)
                n_hit += int(hit)
                print(f"  {inc['path']}: {'HIT' if hit else 'MISS'} (n_windows={len(y_true)})")
            except Exception as e:
                print(f"  [ERROR] {inc['path']}: {e}")

        if all_y_true:
            prec, rec, f1, _ = precision_recall_fscore_support(all_y_true, all_y_pred, average='binary', zero_division=0)
            tn, fp, fn, tp = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1]).ravel() if len(set(all_y_true)) == 2 else (0, 0, 0, 0)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        else:
            prec = rec = f1 = fpr = 0

        # ── Video bình thường ──
        total_dur = 0.0
        total_fa = 0
        for nv in normals:
            try:
                dur, fa = run_video_no_gt(
                    nv["path"],
                    ipm_enabled=cfg["ipm"],
                    rule_enabled=cfg["rule"],
                    ml_enabled=cfg["ml"],
                )
                total_dur += dur
                total_fa += fa
                print(f"  NORM {nv['path']}: {dur:.1f}min, {fa} alerts")
            except Exception as e:
                print(f"  [ERROR] {nv['path']}: {e}")

        fa_per_hour = total_fa / (total_dur / 60.0) if total_dur > 0 else 0

        results[cfg["name"]] = {
            "F1": round(f1, 3),
            "Precision": round(prec, 3),
            "Recall": round(rec, 3),
            "FPR": round(fpr, 3),
            "recall@incident": f"{n_hit}/{len(incidents)}",
            "false_alerts_per_hour": round(fa_per_hour, 1),
        }

        print(f"  → F1={f1:.3f} | Precision={prec:.3f} | Recall={rec:.3f} | FPR={fpr:.3f}")

    # ── Lưu kết quả ──
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("ABLATION STUDY — Đóng góp của từng module\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Cấu hình':<30} {'F1':>8} {'Prec':>8} {'Rec':>8} {'FPR':>8} {'FA/h':>10} {'Rec@Inc':>12}\n")
        f.write("-" * 80 + "\n")
        for cfg in configs:
            r = results.get(cfg["name"], {})
            f.write(f"{cfg['name']:<30} "
                    f"{r.get('F1', 'N/A'):>8} "
                    f"{r.get('Precision', 'N/A'):>8} "
                    f"{r.get('Recall', 'N/A'):>8} "
                    f"{r.get('FPR', 'N/A'):>8} "
                    f"{r.get('false_alerts_per_hour', 'N/A'):>10} "
                    f"{r.get('recall@incident', 'N/A'):>12}\n")
        f.write("\n[JSON dump]\n")
        f.write(json.dumps(results, indent=2))

    print(f"\n[OK] Kết quả lưu vào {args.out}")
    print("     → Gửi file này, tôi cập nhật Bảng 4.2 trong báo cáo.")


if __name__ == "__main__":
    main()