# train_on_tudat.py (phiên bản dùng annotation CSV)
import sys
import csv
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import get_config
from detector.vehicle_detector import VehicleDetector
from tracker.deep_sort_tracker import DeepSORTTracker
from geometry.ipm import IPMTransformer
from features.feature_extractor import FeatureExtractor
from anomaly.lgm_classifier import AccidentClassifier, DatasetBuilder
from utils.logger import get_logger

logger = get_logger("TRAIN_TU_DAT")

# ========== CẤU HÌNH ==========
TU_DAT_ROOT = Path("D:\Downloads\datasets\TU-DAT")                # SỬA THEO MÁY BẠN
ANNOTATION_PATH = Path("E:/NCKH/annotations.csv")   # SỬA THEO ĐƯỜNG DẪN FILE CSV

# Danh sách thư mục video và cách gán nhãn mặc định
# (nếu có annotation thì dùng annotation, nếu không sẽ gán mặc định)
POSITIVE_VIDEO_DIRS = [
    TU_DAT_ROOT / "Final_videos/Positive_Vidoes",
    TU_DAT_ROOT / "Rash-Driving/beaNG",
]
NEGATIVE_VIDEO_DIRS = [
    TU_DAT_ROOT / "Final_videos/Negative_Videos",
    TU_DAT_ROOT / "Final_videos/challenging-environment",  # thêm challenging
]

WINDOW_SIZE = 25
STRIDE = 5
SKIP_INITIAL_FRAMES = 20
SKIP_FACTOR = 2
MAX_SAMPLES_PER_VIDEO = 300   # giảm để đa dạng
# ===============================

def load_annotations(csv_path):
    """
    Đọc file CSV chứa thông tin khoảng frame tai nạn.
    Tự động xử lý BOM (nếu có) và nhận diện các biến thể tên cột.
    """
    annotations = {}
    if not csv_path.exists():
        logger.warning(f"Annotation file không tồn tại: {csv_path}. Dùng nhãn mặc định.")
        return annotations

    with open(csv_path, newline='', encoding='utf-8-sig') as f:  # utf-8-sig tự bỏ BOM
        # Đọc dòng đầu để kiểm tra delimiter và tiêu đề
        first_line = f.readline()
        f.seek(0)

        # Tự phát hiện delimiter (dấu phẩy hoặc tab)
        delimiter = ',' if ',' in first_line else '\t'
        reader = csv.DictReader(f, delimiter=delimiter)

        if reader.fieldnames is None:
            logger.error("File CSV không có dòng tiêu đề.")
            return annotations

        # Chuẩn hóa tên cột: bỏ khoảng trắng, chuyển thường
        fieldnames = [name.strip().lower() for name in reader.fieldnames]
        col_video = next((name for name in fieldnames if name in ['video_name', 'video', 'file', 'filename']), None)
        col_start = next((name for name in fieldnames if name in ['start_frame', 'start', 'frame_start']), None)
        col_end = next((name for name in fieldnames if name in ['end_frame', 'end', 'frame_end']), None)

        if not all([col_video, col_start, col_end]):
            logger.error(f"Không tìm thấy đủ cột cần thiết. Các cột hiện có: {fieldnames}")
            return annotations

        for row in reader:
            video = row[col_video].strip()
            try:
                start = int(row[col_start])
                end = int(row[col_end])
            except ValueError:
                logger.warning(f"Dòng không hợp lệ: {row}")
                continue
            annot_list = annotations.setdefault(video, [])
            annot_list.append((start, end))

    logger.info(f"Đã tải annotation: {len(annotations)} video.")
    return annotations

def is_accident_frame(annotations, video_name, frame_num):
    """Kiểm tra frame có nằm trong vùng tai nạn không"""
    if video_name not in annotations:
        return False
    for start, end in annotations[video_name]:
        if start <= frame_num <= end:
            return True
    return False

def process_video(video_path, annotations, default_label, detector, tracker, ipm, feature_extractor, builder):
    """
    default_label: label được gán khi không có annotation.
    Nếu có annotation, sẽ dùng annotation; ngược lại dùng default_label.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Không mở được video: {video_path}")
        return

    video_name = video_path.name
    feature_extractor.reset()
    frame_count = 0
    processed_frames = 0
    samples_added = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Skip frame theo SKIP_FACTOR
        if (frame_count - 1) % SKIP_FACTOR != 0:
            continue

        if processed_frames < SKIP_INITIAL_FRAMES:
            processed_frames += 1
            continue

        frame = cv2.resize(frame, (960, 540))

        # Dùng annotation nếu có, nếu không dùng default_label
        if video_name in annotations:
            label = 1 if is_accident_frame(annotations, video_name, frame_count) else 0
        else:
            label = default_label

        # Detection
        detections = detector.detect(frame)

        # Tracking
        tracks = tracker.update(detections, frame)

        # IPM
        world_positions = {}
        for track in tracks:
            x, y = track.bottom_center
            try:
                wp = ipm.pixel_to_world(int(x), int(y))
                world_positions[track.track_id] = wp
            except:
                pass

        # Feature extraction (frame_id dùng processed_frames)
        features = feature_extractor.update(
            tracks=tracks,
            world_positions=world_positions,
            body_leans=None,
            frame_id=processed_frames
        )

        for track_id, feat_vec in features.items():
            buf = feature_extractor.get_buffer(track_id)
            if buf is None:
                continue
            window = buf.get_window(WINDOW_SIZE)
            if window is None or len(window) != WINDOW_SIZE:
                continue
            if np.any(np.isnan(feat_vec)) or np.any(np.isinf(feat_vec)):
                continue

            builder.add_sample(feat_vec, label)
            samples_added += 1

            if samples_added >= MAX_SAMPLES_PER_VIDEO:
                cap.release()
                logger.info(f"Đủ {MAX_SAMPLES_PER_VIDEO} sample cho {video_name}, dừng sớm.")
                return

        processed_frames += 1

    cap.release()
    logger.info(f"{video_name}: {samples_added} samples mới, tổng {len(builder)}")

def main():
    cfg = get_config()
    detector = VehicleDetector(cfg)
    detector.load_model()

    ipm = IPMTransformer(cfg)
    ipm.calibrate()

    annotations = load_annotations(ANNOTATION_PATH)

    builder = DatasetBuilder(window_size=WINDOW_SIZE, stride=STRIDE)

    # ---- Video Positive (có annotation sẽ ghi đè label) ----
    logger.info("=== XỬ LÝ VIDEO TÍCH CỰC ===")
    for pos_dir in POSITIVE_VIDEO_DIRS:
        if not pos_dir.exists():
            logger.warning(f"Thư mục không tồn tại: {pos_dir}")
            continue
        video_files = list(pos_dir.glob("*.mp4")) + list(pos_dir.glob("*.avi")) + list(pos_dir.glob("*.mov"))
        for vf in tqdm(video_files, desc=f"Pos {pos_dir.name}"):
            tracker = DeepSORTTracker(cfg)
            tracker.initialize()
            feature_extractor = FeatureExtractor(cfg)
            # Default label = 1 cho thư mục này (nếu không có annotation)
            process_video(vf, annotations, default_label=1, detector=detector, tracker=tracker,
                          ipm=ipm, feature_extractor=feature_extractor, builder=builder)

    # ---- Video Negative (và challenging) ----
    logger.info("=== XỬ LÝ VIDEO TIÊU CỰC ===")
    for neg_dir in NEGATIVE_VIDEO_DIRS:
        if not neg_dir.exists():
            logger.warning(f"Thư mục không tồn tại: {neg_dir}")
            continue
        video_files = list(neg_dir.glob("*.mp4")) + list(neg_dir.glob("*.avi")) + list(neg_dir.glob("*.mov"))
        for vf in tqdm(video_files, desc=f"Neg {neg_dir.name}"):
            tracker = DeepSORTTracker(cfg)
            tracker.initialize()
            feature_extractor = FeatureExtractor(cfg)
            # Default label = 0 cho thư mục negative và challenging
            process_video(vf, annotations, default_label=0, detector=detector, tracker=tracker,
                          ipm=ipm, feature_extractor=feature_extractor, builder=builder)

    if len(builder) == 0:
        logger.error("Không có sample nào.")
        return

    X, y = builder.build()
    logger.info(f"Dataset: {X.shape[0]} samples, positive={y.sum()}")

    classifier = AccidentClassifier(cfg)
    metrics = classifier.train(X, y, save=True)
    print("\n===== KẾT QUẢ HUẤN LUYỆN =====")
    for k, v in metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()