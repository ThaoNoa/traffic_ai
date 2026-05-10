"""
DEMO: Trích xuất features từ video bất kỳ
Chạy ngay được với video hiện có, KHÔNG cần dataset.
Output: in ra console hoặc lưu vào file CSV.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Thêm đường dẫn để import các module của project
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import get_config
from detector.vehicle_detector import VehicleDetector
from tracker.deep_sort_tracker import DeepSORTTracker
from geometry.ipm import IPMTransformer
from features.feature_extractor import FeatureExtractor, FeatureAggregator


def main():
    print("=" * 60)
    print("FEATURE EXTRACTION DEMO")
    print("=" * 60)
    
    # 1. Khởi tạo các module
    print("\n[1] Loading models...")
    detector = VehicleDetector().load_model()
    tracker = DeepSORTTracker().initialize()
    ipm = IPMTransformer().calibrate()
    feature_extractor = FeatureExtractor()
    
    # 2. Mở video
    video_path = "videos/linh_nam.mp4"
    print(f"\n[2] Opening video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"    FPS: {fps:.1f}, Total frames: {total_frames}")
    
    # 3. Xử lý từng frame
    frame_id = 0
    all_features = []  # Lưu tất cả feature vectors đã extract
    
    print("\n[3] Processing frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_id += 1
        
        # Bỏ qua frame đầu để tránh warmup
        if frame_id < 30:
            continue
        
        # Chỉ xử lý mỗi 5 frame (tiết kiệm thời gian)
        if frame_id % 5 != 0:
            continue
        
        # Detection
        detections = detector.detect(frame)
        
        # Tracking
        tracks = tracker.update(detections, frame)
        
        # IPM: pixel → world coordinates
        world_positions = {}
        for track in tracks:
            x, y = track.bottom_center
            world_pos = ipm.pixel_to_world(int(x), int(y))
            world_positions[track.track_id] = world_pos
        
        # Feature extraction
        features = feature_extractor.update(
            tracks=tracks,
            world_positions=world_positions,
            frame_id=frame_id
        )
        
        # Lưu features nếu có
        for track_id, feat_vec in features.items():
            all_features.append({
                "frame_id": frame_id,
                "track_id": track_id,
                "features": feat_vec.tolist()
            })
        
        # In ra màn hình mỗi 100 frame
        if frame_id % 100 == 0:
            pct = frame_id / total_frames * 100
            print(f"    Frame {frame_id}/{total_frames} ({pct:.1f}%) | "
                  f"Tracks: {len(tracks)} | Features: {len(features)}")
    
    cap.release()
    
    # 4. Kết quả
    print("\n[4] Results")
    print(f"    Total frames processed: {frame_id}")
    print(f"    Total feature vectors extracted: {len(all_features)}")
    
    if all_features:
        # In 3 mẫu đầu tiên
        print("\n[5] Sample feature vectors (first 3):")
        for i, sample in enumerate(all_features[:3]):
            print(f"\n    Sample {i+1}:")
            print(f"      Frame: {sample['frame_id']}")
            print(f"      Track ID: {sample['track_id']}")
            print(f"      Features (20-dim): {sample['features'][:5]}... (and 15 more)")
        
        # Lưu ra file nếu muốn
        import json
        output_file = "extracted_features.json"
        with open(output_file, "w") as f:
            json.dump(all_features, f, indent=2)
        print(f"\n[6] Saved all features to: {output_file}")
        
        # Cũng có thể lưu dạng numpy
        X = np.array([s["features"] for s in all_features])
        np.save("extracted_features.npy", X)
        print(f"    Saved numpy array to: extracted_features.npy (shape: {X.shape})")
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()