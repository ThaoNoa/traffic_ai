"""
eval_incident.py — Đo Recall trên N video sự cố và FPR trên N video bình thường.

Chạy:
    cd traffic_ai
    python eval_incident.py --eval-list eval_videos.json

eval_videos.json có format:
{
  "incidents": [
    {"path": "videos/eval/cadp_01.mp4", "t_start": 12.0, "t_end": 16.0},
    {"path": "videos/eval/cadp_02.mp4", "t_start":  8.5, "t_end": 11.0},
    {"path": "videos/eval/linh_nam_incident.mp4", "t_start": 5.0, "t_end": 8.0}
  ],
  "normals": [
    {"path": "videos/eval/linh_nam_normal_01.mp4"},
    {"path": "videos/eval/linh_nam_normal_02.mp4"}
  ]
}

Output:
    eval_incident_results.txt — Recall (n đúng / n vụ), FPR, alert count.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import cv2

from config.settings import get_config
from detector.vehicle_detector import VehicleDetector
from tracker.deep_sort_tracker import DeepSORTTracker
from geometry.ipm import IPMTransformer
from features.feature_extractor import FeatureExtractor
from anomaly.rule_engine import PhysicsRuleEngine, RiskLevel
from anomaly.lgm_classifier import AccidentClassifier


def init_pipeline():
    cfg = get_config()
    detector = VehicleDetector(cfg).load_model()
    tracker = DeepSORTTracker(cfg).initialize()
    ipm = IPMTransformer(cfg)
    fe = FeatureExtractor(cfg)
    re = PhysicsRuleEngine(cfg)
    clf = AccidentClassifier(cfg).load_model()
    return cfg, detector, tracker, ipm, fe, re, clf


def run_video(video_path: str, detector, tracker, ipm, fe, re, clf,
              t_start: float = -1, t_end: float = -1) -> Dict:
    """
    Chạy hệ thống full trên 1 video.
    Trả về:
        - alerts: list các (frame_id, t, track_id, final_score)
        - n_frames, fps_video, runtime
        - hit_in_window: True nếu có alert nằm trong [t_start, t_end] (chỉ áp dụng video sự cố)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Không mở được {video_path}")

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Reset feature extractor cho video mới
    fe.reset()

    # Auto-calibrate IPM trên frame đầu
    ret, first = cap.read()
    if not ret:
        raise IOError("Video trống")
    if not ipm._calibrated:
        try:
            ipm.auto_calibrate_from_frame(first)
        except Exception:
            ipm.calibrate()

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    alerts: List[Dict] = []
    cooldown_until: Dict[int, float] = {}  # track_id → t khả dụng (chống spam)
    cooldown_s = 5.0

    frame_id = 0
    t_runtime_start = time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        t_now = frame_id / fps_video

        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)

        world_positions = {}
        for tk in tracks:
            u, v = tk.bottom_center
            try:
                wp = ipm.pixel_to_world(u, v)
                world_positions[tk.track_id] = wp
            except Exception:
                continue

        features = fe.update(tracks, world_positions, body_leans=None, frame_id=frame_id)
        if not features:
            continue

        for tid, feat_vec in features.items():
            buf = fe.get_buffer(tid)
            if buf is None:
                continue
            window = buf.get_window(fe.window_size)
            if window is None:
                continue
            risk = re.evaluate_state(window, tid, frame_id)
            if clf is not None and clf.is_ready:
                ml = clf.predict(feat_vec)
                if ml >= 0:
                    risk = re.fuse_with_ml(risk, ml)

            # Cooldown
            if t_now < cooldown_until.get(tid, -1):
                continue

            if risk.is_accident or risk.risk_level >= RiskLevel.DANGER:
                alerts.append({
                    "frame": frame_id, "t": t_now, "track_id": tid,
                    "final_score": risk.final_score, "rule": risk.rule_score,
                    "ml": risk.ml_score,
                    "violations": [v.rule_name for v in risk.violations],
                })
                cooldown_until[tid] = t_now + cooldown_s

    cap.release()
    runtime = time.perf_counter() - t_runtime_start

    hit_in_window = False
    if t_start >= 0 and t_end >= 0:
        # Cho phép alert nằm trong [t_start - 1, t_end + 1] (sai số gán nhãn ±1s)
        margin = 1.0
        for a in alerts:
            if t_start - margin <= a["t"] <= t_end + margin:
                hit_in_window = True
                break

    return {
        "video": video_path,
        "n_frames": frame_id,
        "fps_video": fps_video,
        "runtime_s": runtime,
        "system_fps": frame_id / runtime if runtime > 0 else 0,
        "alerts": alerts,
        "n_alerts": len(alerts),
        "hit_in_window": hit_in_window,
        "gt_window": (t_start, t_end) if t_start >= 0 else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-list", required=True, help="JSON file chứa danh sách video")
    parser.add_argument("--out", default="eval_incident_results.txt")
    args = parser.parse_args()

    with open(args.eval_list, "r", encoding="utf-8") as f:
        eval_cfg = json.load(f)

    incidents = eval_cfg.get("incidents", [])
    normals = eval_cfg.get("normals", [])

    print(f"[INFO] Eval set: {len(incidents)} video sự cố | {len(normals)} video bình thường")

    cfg, detector, tracker, ipm, fe, re, clf = init_pipeline()
    if not clf.is_ready:
        print("[WARN] LightGBM model chưa train → chạy rule-only.")

    results = {"incidents": [], "normals": []}

    # ── Video sự cố: đo Recall ──
    print("\n" + "=" * 60)
    print("EVAL TRÊN VIDEO SỰ CỐ (đo Recall)")
    print("=" * 60)
    n_hit = 0
    for inc in incidents:
        # Reset tracker cho video mới
        try:
            tracker.initialize()
        except Exception:
            pass
        try:
            r = run_video(
                inc["path"], detector, tracker, ipm, fe, re, clf,
                t_start=inc.get("t_start", -1), t_end=inc.get("t_end", -1),
            )
            results["incidents"].append(r)
            tag = "✓ HIT" if r["hit_in_window"] else "✗ MISS"
            n_hit += int(r["hit_in_window"])
            print(f"  [{tag}] {inc['path']} | alerts={r['n_alerts']} "
                  f"| sys_fps={r['system_fps']:.1f}")
        except Exception as e:
            print(f"  [ERROR] {inc['path']}: {e}")
            results["incidents"].append({"video": inc["path"], "error": str(e)})

    recall = n_hit / max(len(incidents), 1)
    print(f"\n  → Recall = {n_hit}/{len(incidents)} = {recall:.2%}")

    # ── Video bình thường: đo FPR thô (số alert/giờ) ──
    print("\n" + "=" * 60)
    print("EVAL TRÊN VIDEO BÌNH THƯỜNG (đo False Alarm)")
    print("=" * 60)
    total_minutes = 0.0
    total_false_alerts = 0
    for nv in normals:
        try:
            tracker.initialize()
        except Exception:
            pass
        try:
            r = run_video(nv["path"], detector, tracker, ipm, fe, re, clf)
            results["normals"].append(r)
            duration_min = r["n_frames"] / r["fps_video"] / 60.0
            total_minutes += duration_min
            total_false_alerts += r["n_alerts"]
            print(f"  {nv['path']} | duration={duration_min:.1f} min "
                  f"| false_alerts={r['n_alerts']} | sys_fps={r['system_fps']:.1f}")
        except Exception as e:
            print(f"  [ERROR] {nv['path']}: {e}")
            results["normals"].append({"video": nv["path"], "error": str(e)})

    if total_minutes > 0:
        fa_per_hour = total_false_alerts / (total_minutes / 60.0)
        print(f"\n  → False alerts: {total_false_alerts} / {total_minutes:.1f} phút "
              f"= {fa_per_hour:.1f} alerts/giờ")
    else:
        fa_per_hour = 0

    # ── Lưu kết quả ──
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("EVAL — Phát hiện sự cố giao thông\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Video sự cố: {len(incidents)}\n")
        f.write(f"  → Recall = {n_hit}/{len(incidents)} = {recall:.2%}\n\n")
        f.write(f"Video bình thường: {len(normals)}\n")
        f.write(f"  → Tổng thời lượng: {total_minutes:.1f} phút\n")
        f.write(f"  → False alerts: {total_false_alerts}\n")
        f.write(f"  → False-alert rate: {fa_per_hour:.1f} /giờ\n\n")
        f.write("Chi tiết từng video:\n")
        f.write("-" * 70 + "\n")
        for r in results["incidents"]:
            if "error" in r:
                f.write(f"  [INC]  {r['video']}: ERROR {r['error']}\n")
            else:
                tag = "HIT" if r["hit_in_window"] else "MISS"
                f.write(f"  [INC]  {tag} | {r['video']} | alerts={r['n_alerts']} "
                        f"| sys_fps={r['system_fps']:.1f} "
                        f"| gt={r['gt_window']}\n")
        for r in results["normals"]:
            if "error" in r:
                f.write(f"  [NORM] {r['video']}: ERROR {r['error']}\n")
            else:
                f.write(f"  [NORM] {r['video']} | alerts={r['n_alerts']} "
                        f"| sys_fps={r['system_fps']:.1f}\n")
        f.write("\n[JSON dump]\n")
        f.write(json.dumps(results, indent=2, default=str))

    print(f"\n[OK] Đã lưu kết quả vào: {args.out}")


if __name__ == "__main__":
    main()