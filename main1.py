"""
MAIN1.PY
Pipeline hoàn chỉnh:
VIDEO -> YOLOv8 -> DeepSORT -> IPM -> Feature Extraction
      -> Rule Engine + LightGBM -> Alert -> Visualization
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

from geometry.ipm import IPMTransformer
from features.feature_extractor import FeatureExtractor

from anomaly.rule_engine import PhysicsRuleEngine
from anomaly.lgm_classifier import AccidentClassifier
from anomaly.alert_manager import AlertManager

logger = get_logger("MAIN1")


class TrafficAIApp:

    def __init__(self, config_path: str = "config/config.yaml"):

        self.cfg = get_config(config_path)

        # =========================================================
        # INIT MODULES
        # =========================================================

        logger.info("Initializing detector...")
        self.detector = VehicleDetector().load_model()

        logger.info("Initializing tracker...")
        self.tracker = DeepSORTTracker().initialize()

        logger.info("Initializing IPM...")
        self.ipm = IPMTransformer().calibrate()

        logger.info("Initializing feature extractor...")
        self.feature_extractor = FeatureExtractor()

        logger.info("Initializing rule engine...")
        self.rule_engine = PhysicsRuleEngine()

        logger.info("Initializing classifier...")
        self.classifier = AccidentClassifier().load_model()

        logger.info("Initializing alert manager...")
        self.alert_manager = AlertManager()

        self.frame_id = 0

    # =========================================================
    # MAIN LOOP
    # =========================================================

    def run(self):

        source = self.cfg.video.source

        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise RuntimeError(f"Không mở được video source: {source}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))

        logger.info(
            f"Video opened | {width}x{height} | FPS={fps:.2f}"
        )

        # =========================================================
        # VIDEO WRITER
        # =========================================================

        output_dir = Path(self.cfg.alert.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "demo_result.mp4"

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        # =========================================================
        # LOOP
        # =========================================================

        try:

            while True:

                ret, frame = cap.read()

                if not ret:
                    logger.info("End of video.")
                    break

                t0 = time.perf_counter()

                self.frame_id += 1

                # =================================================
                # STEP 1 — DETECTION
                # =================================================

                detections = self.detector.detect(frame)

                # =================================================
                # STEP 2 — TRACKING
                # =================================================

                tracks = self.tracker.update(
                    detections=detections,
                    frame=frame
                )

                # =================================================
                # STEP 3 — WORLD COORDS (IPM)
                # =================================================

                world_positions = {}

                for track in tracks:

                    x, y = track.bottom_center

                    world_pos = self.ipm.pixel_to_world(
                        int(x),
                        int(y)
                    )

                    world_positions[track.track_id] = world_pos

                # =================================================
                # STEP 4 — FEATURE EXTRACTION
                # =================================================

                features = self.feature_extractor.update(
                    tracks=tracks,
                    world_positions=world_positions,
                    frame_id=self.frame_id
                )

                # =================================================
                # STEP 5 — HYBRID AI
                # =================================================

                risk_results = {}

                for track in tracks:

                    tid = track.track_id

                    if tid not in features:
                        continue

                    feat_vec = features[tid]

                    buf = self.feature_extractor.get_buffer(tid)

                    if buf is None:
                        continue

                    window = buf.get_window()

                    if window is None:
                        continue

                    # ---------------------------------------------
                    # RULE ENGINE
                    # ---------------------------------------------

                    rule_result = self.rule_engine.evaluate_state(
                        window=window,
                        track_id=tid,
                        frame_id=self.frame_id
                    )

                    # ---------------------------------------------
                    # LIGHTGBM
                    # ---------------------------------------------

                    ml_score = self.classifier.predict(feat_vec)

                    # ---------------------------------------------
                    # FUSION
                    # ---------------------------------------------

                    fused_result = self.rule_engine.fuse_with_ml(
                        rule_result,
                        ml_score
                    )

                    risk_results[tid] = fused_result

                # =================================================
                # STEP 6 — ALERT
                # =================================================

                active_alerts = self.alert_manager.process(
                    risk_results,
                    self.frame_id
                )

                self.alert_manager.push_frame(frame)

                # =================================================
                # STEP 7 — VISUALIZATION
                # =================================================

                vis = self.visualize(
                    frame=frame,
                    tracks=tracks,
                    risk_results=risk_results,
                    alerts=active_alerts
                )

                # FPS
                dt = time.perf_counter() - t0
                fps_now = 1.0 / max(dt, 1e-6)

                cv2.putText(
                    vis,
                    f"FPS: {fps_now:.1f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )

                # =================================================
                # SHOW + SAVE
                # =================================================

                writer.write(vis)

                # cv2.imshow("Traffic AI", vis)

                # key = cv2.waitKey(1)

                # if key == ord("q"):
                #     break

        finally:

            cap.release()
            writer.release()

            cv2.destroyAllWindows()

            logger.info(f"Saved output: {output_path}")

    # =========================================================
    # VISUALIZATION
    # =========================================================

    def visualize(
        self,
        frame,
        tracks,
        risk_results,
        alerts
    ):

        canvas = frame.copy()

        h, w = canvas.shape[:2]

        # =====================================================
        # DRAW TRACKS
        # =====================================================

        for track in tracks:

            tid = track.track_id

            x1, y1, x2, y2 = track.bbox_xyxy.astype(int)

            color = (0, 255, 0)

            label = f"ID {tid} | {track.class_name}"

            if tid in risk_results:

                result = risk_results[tid]

                score = float(result.final_score)

                label += f" | risk={score:.2f}"

                if score > 0.8:
                    color = (0, 0, 255)

                elif score > 0.5:
                    color = (0, 165, 255)

            # BOX

            cv2.rectangle(
                canvas,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # TEXT

            cv2.putText(
                canvas,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            # TRAJECTORY

            traj = list(track.trajectory)

            for i in range(1, len(traj)):

                p1 = tuple(np.int32(traj[i - 1]))
                p2 = tuple(np.int32(traj[i]))

                cv2.line(
                    canvas,
                    p1,
                    p2,
                    (255, 255, 0),
                    2
                )

        # =====================================================
        # ALERT PANEL
        # =====================================================

        if alerts:

            overlay = canvas.copy()

            cv2.rectangle(
                overlay,
                (w - 350, 0),
                (w, 180),
                (0, 0, 0),
                -1
            )

            cv2.addWeighted(
                overlay,
                0.5,
                canvas,
                0.5,
                0,
                canvas
            )

            cv2.putText(
                canvas,
                "ACTIVE ALERTS",
                (w - 330, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            for i, alert in enumerate(alerts[:5]):

                y = 80 + i * 28

                text = (
                    f"Track {alert.track_id} "
                    f"| {alert.risk_level.name}"
                )

                cv2.putText(
                    canvas,
                    text,
                    (w - 330, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1
                )

        return canvas


# =============================================================
# ENTRY
# =============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml"
    )

    parser.add_argument(
        "--source",
        type=str,
        default=None
    )

    args = parser.parse_args()

    app = TrafficAIApp(args.config)

    # override video source

    if args.source is not None:
        app.cfg.video.source = args.source

    app.run()