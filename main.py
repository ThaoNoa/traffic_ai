"""
TRAFFIC AI SYSTEM
YOLOv8 + DeepSORT + Live Preview
Optimized for RTX 3050 4GB
"""

from __future__ import annotations

import cv2
import time
import argparse
import numpy as np
from pathlib import Path

from config.settings import get_config
from utils.logger import get_logger

from detector.vehicle_detector import VehicleDetector
from tracker.deep_sort_tracker import DeepSORTTracker

logger = get_logger(__name__)

# ============================================================
# TRACK COLORS
# ============================================================

np.random.seed(42)
_COLORS = np.random.randint(50, 255, size=(500, 3))


def get_color(track_id: int):
    c = _COLORS[track_id % 500]
    return (int(c[0]), int(c[1]), int(c[2]))


# ============================================================
# DRAW
# ============================================================

def draw_tracks(
    frame: np.ndarray,
    tracks: list,
    fps: float
) -> np.ndarray:

    vis = frame.copy()

    for track in tracks:

        x1, y1, x2, y2 = track.bbox_xyxy.astype(int)

        color = get_color(track.track_id)

        # ----------------------------------------------------
        # BBOX
        # ----------------------------------------------------

        cv2.rectangle(
            vis,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        label = (
            f"ID {track.track_id} | "
            f"{track.class_name} | "
            f"{track.confidence:.2f}"
        )

        (tw, th), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2
        )

        cv2.rectangle(
            vis,
            (x1, y1 - th - 10),
            (x1 + tw + 8, y1),
            color,
            -1
        )

        cv2.putText(
            vis,
            label,
            (x1 + 4, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2
        )

        # ----------------------------------------------------
        # TRAJECTORY
        # ----------------------------------------------------

        traj = list(track.trajectory)

        if len(traj) >= 2:

            for i in range(1, len(traj)):

                p1 = (
                    int(traj[i - 1][0]),
                    int(traj[i - 1][1])
                )

                p2 = (
                    int(traj[i][0]),
                    int(traj[i][1])
                )

                cv2.line(
                    vis,
                    p1,
                    p2,
                    color,
                    2
                )

    # ========================================================
    # HUD
    # ========================================================

    cv2.rectangle(vis, (10, 10), (320, 120), (0, 0, 0), -1)

    cv2.putText(
        vis,
        f"FPS: {fps:.1f}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )

    cv2.putText(
        vis,
        f"Tracks: {len(tracks)}",
        (20, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )

    return vis


# ============================================================
# MAIN
# ============================================================

def run(video_source: str | None = None):

    cfg = get_config()

    source = video_source or cfg.video.source

    logger.info("=" * 60)
    logger.info("TRAFFIC AI SYSTEM")
    logger.info("=" * 60)

    logger.info(f"Source : {source}")
    logger.info(f"Device : {cfg.system.device}")

    # ========================================================
    # DETECTOR
    # ========================================================

    detector = VehicleDetector(cfg)
    detector.load_model()

    # ========================================================
    # TRACKER
    # ========================================================

    tracker = DeepSORTTracker(cfg)
    tracker.initialize()

    # ========================================================
    # VIDEO
    # ========================================================

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    input_fps = cap.get(cv2.CAP_PROP_FPS)

    if input_fps <= 0:
        input_fps = 25

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(
        f"Video: {W}x{H} @ {input_fps:.1f} FPS"
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "tracking_output.mp4"

    # --------------------------------------------------------
    # IMPORTANT:
    # Save at 10 FPS to avoid fast-forward video
    # --------------------------------------------------------

    W = 1280
    H = 720

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        20,
        (W, H)
    )

    logger.info(f"Saving output: {output_path}")

    # ========================================================
    # DISPLAY
    # ========================================================

    DISPLAY_W = 1280
    DISPLAY_H = 720

    # ========================================================
    # FPS
    # ========================================================

    fps_timer = time.time()

    fps_counter = 0
    current_fps = 0.0

    frame_id = 0

    paused = False

    logger.info(
        "[Q] Quit | [P] Pause | [S] Screenshot"
    )

    # ========================================================
    # LOOP
    # ========================================================

    try:

        while True:

            # ------------------------------------------------
            # PAUSE MODE
            # ------------------------------------------------

            if paused:

                key = cv2.waitKey(30) & 0xFF

                if key == ord("p"):
                    paused = False

                elif key == ord("q"):
                    break

                continue

            # ------------------------------------------------
            # READ
            # ------------------------------------------------

            ret, frame = cap.read()

            if not ret:
                break

            TARGET_W = 1280
            TARGET_H = 720
            frame = cv2.resize(frame, (TARGET_W, TARGET_H))

            frame_id += 1
            if frame_id % 2 != 0:
                continue

            # ------------------------------------------------
            # DETECT
            # ------------------------------------------------

            t0 = time.time()

            detections = detector.detect(frame)
            print("detections:", len(detections))

            detect_ms = (time.time() - t0) * 1000

            # ------------------------------------------------
            # TRACK
            # ------------------------------------------------

            t1 = time.time()

            tracks = tracker.update(
                detections,
                frame
            )

            track_ms = (time.time() - t1) * 1000

            # ------------------------------------------------
            # FPS
            # ------------------------------------------------

            fps_counter += 1

            elapsed = time.time() - fps_timer

            if elapsed >= 1.0:

                current_fps = fps_counter / elapsed

                fps_counter = 0
                fps_timer = time.time()

                progress = (
                    frame_id / total_frames * 100
                )

                logger.info(
                    f"Frame {frame_id:5d}/{total_frames} | "
                    f"DET {detect_ms:.1f}ms | "
                    f"TRK {track_ms:.1f}ms | "
                    f"FPS {current_fps:.1f} | "
                    f"Tracks {len(tracks)} | "
                    f"{progress:.1f}%"
                )

            # ------------------------------------------------
            # DRAW
            # ------------------------------------------------

            vis = draw_tracks(
                frame,
                tracks,
                current_fps
            )

            # ------------------------------------------------
            # SAVE
            # ------------------------------------------------

            writer.write(vis)

            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            display = cv2.resize(
                vis,
                (DISPLAY_W, DISPLAY_H)
            )

            # cv2.imshow(
            #     "Traffic AI - Tracking",
            #     display
            # )

            # ------------------------------------------------
            # KEYBOARD
            # ------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            # Quit
            if key == ord("q"):
                logger.info("Quit.")
                break

            # Pause
            elif key == ord("p"):

                paused = True

                logger.info("Paused.")

            # Screenshot
            elif key == ord("s"):

                screenshot_path = (
                    output_dir /
                    f"screenshot_{frame_id:05d}.jpg"
                )

                cv2.imwrite(
                    str(screenshot_path),
                    vis
                )

                logger.info(
                    f"Saved: {screenshot_path}"
                )

    except KeyboardInterrupt:

        logger.info("Interrupted.")

    finally:

        cap.release()

        writer.release()

        cv2.destroyAllWindows()

        logger.info("=" * 60)
        logger.info("STATS")
        logger.info("=" * 60)

        all_stats = {
            **detector.stats,
            **tracker.stats
        }

        for k, v in all_stats.items():
            logger.info(f"{k}: {v}")

        logger.info(f"Output: {output_path}")


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Video path"
    )

    args = parser.parse_args()

    run(args.source)