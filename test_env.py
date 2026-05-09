# test_env.py — đặt ở thư mục gốc traffic_ai/
import sys
print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

import cv2
print(f"OpenCV: {cv2.__version__}")

import ultralytics
print(f"Ultralytics: {ultralytics.__version__}")

import lightgbm as lgb
print(f"LightGBM: {lgb.__version__}")

import numpy as np
print(f"NumPy: {np.__version__}")

print("\n✅ Môi trường OK — sẵn sàng chạy hệ thống.")