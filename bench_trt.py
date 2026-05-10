"""
bench_trt.py — Benchmark PyTorch FP32 vs TensorRT FP16 cho YOLOv8.

Chạy:
    cd traffic_ai
    python bench_trt.py

Output:
    bench_trt_results.txt — kết quả latency (ms) và FPS cho từng cấu hình.

Yêu cầu:
    pip install ultralytics torch
    GPU NVIDIA + driver CUDA. Nếu không có GPU script sẽ exit sớm.

Lưu ý: bước export sang .engine (TensorRT) chỉ thành công nếu có TensorRT.
Nếu không có TensorRT trên máy, script vẫn cho ra số PyTorch FP32 — đủ để
báo cáo trung thực rằng "phiên bản hiện tại chưa benchmark được TRT".
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


N_WARMUP = 30
N_BENCH = 200
IMGSZ = 640
MODEL_PT = "yolov8s.pt"               # nếu bạn dùng yolov8s.pt thì đổi tên
MODEL_ENGINE = "yolov8s_fp16.engine"  # khớp với config.yaml
OUTPUT_FILE = "bench_trt_results.txt"


def bench(model: YOLO, device: str, n_warmup: int = N_WARMUP, n_bench: int = N_BENCH) -> dict:
    """Đo latency trung bình (ms/frame) và FPS trên ảnh dummy 640x640."""
    img = np.random.randint(0, 255, (IMGSZ, IMGSZ, 3), dtype=np.uint8)

    # Warmup
    for _ in range(n_warmup):
        _ = model(img, imgsz=IMGSZ, device=device, verbose=False)

    # Đảm bảo GPU sync trước khi đo
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_bench):
        _ = model(img, imgsz=IMGSZ, device=device, verbose=False)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    latency_ms = (elapsed / n_bench) * 1000.0
    fps = n_bench / elapsed
    return {"latency_ms": latency_ms, "fps": fps, "n_runs": n_bench}


def main():
    if not torch.cuda.is_available():
        print("[ERROR] Không có GPU NVIDIA / CUDA. Script bench TRT cần GPU.")
        print("        Có thể chạy bench PyTorch CPU nhưng số sẽ không có ý nghĩa.")
        return

    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[INFO] GPU: {gpu_name}")
    print(f"[INFO] Warmup: {N_WARMUP} | Bench: {N_BENCH} | Input: {IMGSZ}x{IMGSZ}\n")

    results = {"gpu": gpu_name, "imgsz": IMGSZ, "n_bench": N_BENCH}

    # ── 1. PyTorch FP32 ────────────────────────────────────────
    print(f"[1/3] PyTorch FP32 ({MODEL_PT})...")
    if not Path(MODEL_PT).exists():
        print(f"      Tải tự động {MODEL_PT}...")
    model_pt = YOLO(MODEL_PT)
    r_fp32 = bench(model_pt, device)
    print(f"      Latency: {r_fp32['latency_ms']:.2f} ms | FPS: {r_fp32['fps']:.1f}")
    results["pytorch_fp32"] = r_fp32

    # ── 2. PyTorch FP16 (half) — không cần TRT, vẫn cho số tham chiếu ──
    print(f"\n[2/3] PyTorch FP16 (half precision)...")
    try:
        model_pt_half = YOLO(MODEL_PT)
        # Ultralytics hỗ trợ half=True ở predict
        img = np.random.randint(0, 255, (IMGSZ, IMGSZ, 3), dtype=np.uint8)
        for _ in range(N_WARMUP):
            _ = model_pt_half(img, imgsz=IMGSZ, device=device, half=True, verbose=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_BENCH):
            _ = model_pt_half(img, imgsz=IMGSZ, device=device, half=True, verbose=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        r_fp16 = {"latency_ms": (elapsed / N_BENCH) * 1000, "fps": N_BENCH / elapsed, "n_runs": N_BENCH}
        print(f"      Latency: {r_fp16['latency_ms']:.2f} ms | FPS: {r_fp16['fps']:.1f}")
        results["pytorch_fp16"] = r_fp16
    except Exception as e:
        print(f"      [WARN] PyTorch FP16 thất bại: {e}")
        results["pytorch_fp16"] = None

    # ── 3. TensorRT FP16 ───────────────────────────────────────
    print(f"\n[3/3] TensorRT FP16 ({MODEL_ENGINE})...")
    engine_path = Path(MODEL_ENGINE)
    if not engine_path.exists():
        print(f"      Engine chưa có. Đang export {MODEL_PT} → {MODEL_ENGINE}...")
        try:
            export_model = YOLO(MODEL_PT)
            exported = export_model.export(format="engine", half=True, imgsz=IMGSZ, device=0)
            print(f"      Export OK: {exported}")
            # Tên file thực có thể là yolov8n.engine — copy/rename
            real_engine = Path(exported)
            if real_engine.exists() and real_engine.name != MODEL_ENGINE:
                import shutil
                shutil.copy(real_engine, MODEL_ENGINE)
        except Exception as e:
            print(f"      [ERROR] Export TRT thất bại: {e}")
            print("      → Giữ nguyên kết quả PyTorch, bỏ qua TRT.")
            results["tensorrt_fp16"] = None
            _save(results)
            return

    try:
        model_trt = YOLO(MODEL_ENGINE)
        r_trt = bench(model_trt, device)
        print(f"      Latency: {r_trt['latency_ms']:.2f} ms | FPS: {r_trt['fps']:.1f}")
        results["tensorrt_fp16"] = r_trt
    except Exception as e:
        print(f"      [ERROR] Load engine thất bại: {e}")
        results["tensorrt_fp16"] = None

    _save(results)


def _save(results: dict) -> None:
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 64 + "\n")
        f.write("BENCHMARK YOLOv8 — PyTorch vs TensorRT\n")
        f.write("=" * 64 + "\n")
        f.write(f"GPU: {results.get('gpu', 'N/A')}\n")
        f.write(f"Input size: {results.get('imgsz')}x{results.get('imgsz')}\n")
        f.write(f"N runs: {results.get('n_bench')}\n\n")
        f.write(f"{'Cấu hình':<20} {'Latency (ms)':>14} {'FPS':>10}\n")
        f.write("-" * 46 + "\n")

        for key, label in [
            ("pytorch_fp32", "PyTorch FP32"),
            ("pytorch_fp16", "PyTorch FP16"),
            ("tensorrt_fp16", "TensorRT FP16"),
        ]:
            r = results.get(key)
            if r is None:
                f.write(f"{label:<20} {'N/A':>14} {'N/A':>10}\n")
            else:
                f.write(f"{label:<20} {r['latency_ms']:>14.2f} {r['fps']:>10.1f}\n")

        f.write("\n[JSON dump]\n")
        f.write(json.dumps(results, indent=2))

    print(f"\n[OK] Đã lưu kết quả vào: {OUTPUT_FILE}")
    print(f"     → Gửi lại file này, tôi sẽ cập nhật vào báo cáo.")


if __name__ == "__main__":
    main()
