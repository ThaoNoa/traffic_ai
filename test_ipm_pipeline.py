# test_ipm_pipeline.py
"""Test IPM pipeline - Luu anh debug."""
import cv2
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geometry.ipm import IPMTransformer
from config.settings import get_config


def main():
    video_path = "videos/demo2.mp4"
    if len(sys.argv) > 1:
        video_path = sys.argv[1]

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Khong doc duoc frame!")
        return

    h, w = frame.shape[:2]
    scale = min(1400 / w, 900 / h)
    frame_small = cv2.resize(frame, (int(w * scale), int(h * scale)))

    print(f"Frame original: {w}x{h}")
    print(f"Frame display: {frame_small.shape[1]}x{frame_small.shape[0]}")
    print("=" * 50)

    # Init IPM
    cfg = get_config()
    ipm = IPMTransformer(cfg)

    # Calibrate
    result = ipm.calibrate()
    print(f"\nCalibration: {'OK' if result.is_valid else 'FAILED'}")
    if not result.is_valid:
        print(f"  Error: {result.error_msg}")
        print("  FIX: Chinh lai src_points trong config.yaml")
    print(f"  Condition number: {result.condition_number:.0f}")
    print(f"  Area ratio: {result.area_ratio:.3f}")
    print(f"\nStats: {ipm.get_stats()}")

    # Output dir
    os.makedirs("outputs/ipm_test", exist_ok=True)

    # 1. Original
    cv2.imwrite("outputs/ipm_test/1_original.jpg", frame_small)

    # 2. Debug overlay
    debug = ipm.create_debug_overlay(frame_small)
    cv2.imwrite("outputs/ipm_test/2_debug_overlay.jpg", debug)

    # 3. BEV debug
    if ipm.is_calibrated:
        bev_debug = ipm.create_bev_debug(frame_small)
        cv2.imwrite("outputs/ipm_test/3_bev_debug.jpg", bev_debug)

        # 4. BEV raw
        bev = ipm.get_bev_image(frame_small)
        cv2.imwrite("outputs/ipm_test/4_bev_raw.jpg", bev)

    print("\nDone! Check outputs/ipm_test/")
    print("  1_original.jpg    - Anh goc")
    print("  2_debug_overlay.jpg - Source points + grid + ROI")
    print("  3_bev_debug.jpg   - BEV voi grid toa do")
    print("  4_bev_raw.jpg     - BEV thuan tuy")


if __name__ == '__main__':
    main()