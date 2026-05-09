import cv2
import numpy as np

from geometry.ipm import IPMTransformer
from features.feature_extractor import FeatureExtractor
from anomaly.rule_engine import PhysicsRuleEngine
from anomaly.lgm_classifier import AccidentClassifier
from anomaly.alert_manager import AlertManager


# =========================
# INIT
# =========================

ipm = IPMTransformer().calibrate()

feature_extractor = FeatureExtractor()

rule_engine = PhysicsRuleEngine()

classifier = AccidentClassifier().load_model()

alert_manager = AlertManager()


# =========================
# VIDEO
# =========================

cap = cv2.VideoCapture("demo.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

alert_manager.set_video_properties(w, h, fps)

frame_id = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_id += 1

    # =====================================================
    # FAKE TRACKS (sau này thay bằng DeepSORT)
    # =====================================================

    class FakeTrack:
        def __init__(self, track_id):
            self.track_id = track_id

    tracks = [FakeTrack(1)]

    # =====================================================
    # GIẢ lập bbox bottom-center
    # =====================================================

    u = 1000 + frame_id * 2
    v = 700

    # Pixel -> World
    world_pos = ipm.pixel_to_world(u, v)

    world_positions = {
        1: world_pos
    }

    # Lean angle giả lập
    body_leans = {
        1: np.sin(frame_id * 0.1) * 10
    }

    # =====================================================
    # FEATURE EXTRACTION
    # =====================================================

    features = feature_extractor.update(
        tracks=tracks,
        world_positions=world_positions,
        body_leans=body_leans,
        frame_id=frame_id
    )

    risk_results = {}

    for track_id, feat_vec in features.items():

        # lấy window
        window = feature_extractor.get_buffer(track_id).get_window()

        # =================================================
        # RULE ENGINE
        # =================================================

        result = rule_engine.evaluate_state(
            window=window,
            track_id=track_id,
            frame_id=frame_id
        )

        # =================================================
        # ML
        # =================================================

        ml_score = classifier.predict(feat_vec)

        if ml_score >= 0:
            result = rule_engine.fuse_with_ml(
                result,
                ml_score
            )

        risk_results[track_id] = result

        print(result.explain())

    # =====================================================
    # ALERT
    # =====================================================

    alerts = alert_manager.process(
        risk_results,
        frame_id
    )

    for alert in alerts:
        print("🚨 ALERT:", alert)

    alert_manager.push_frame(frame)

    # =====================================================
    # VISUALIZE
    # =====================================================

    cv2.putText(
        frame,
        f"Frame: {frame_id}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()