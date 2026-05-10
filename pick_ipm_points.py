# pick_ipm_points.py
"""Dung chuot de chon 4 diem IPM tren mat duong - Dung matplotlib."""
import cv2
import numpy as np
import sys
import matplotlib
matplotlib.use('TkAgg')  # Hoac 'Qt5Agg'
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.lines import Line2D

video_path = sys.argv[1] if len(sys.argv) > 1 else "videos/linh_nam.mp4"

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Khong doc duoc video!")
    exit()

# BGR -> RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
h, w = frame_rgb.shape[:2]

print("=" * 60)
print("  CHON 4 DIEM TREN MAT DUONG")
print("=" * 60)
print(f"Anh: {w}x{h}")
print()
print("Bam theo thu tu:")
print("  1. TOP-LEFT    (xa nhat, ben trai)   - BAM DAU TIEN")
print("  2. TOP-RIGHT   (xa nhat, ben phai)   - BAM THU HAI")
print("  3. BOTTOM-LEFT (gan nhat, ben trai)  - BAM THU BA")
print("  4. BOTTOM-RIGHT(gan nhat, ben phai)  - BAM CUOI CUNG")
print()
print("QUY TAC QUAN TRONG:")
print("  - BAM TREN MAT DUONG, khong bam len via he")
print("  - Canh tren PHAI hep hon canh duoi")
print("  - Dong cua so de xem ket qua")
print("=" * 60)

points = []
labels = ["1. TOP-LEFT", "2. TOP-RIGHT", "3. BOTTOM-LEFT", "4. BOTTOM-RIGHT"]
colors = ['lime', 'lime', 'red', 'red']

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.imshow(frame_rgb)
ax.set_title("BAM 4 DIEM TREN MAT DUONG THEO THU TU\nTL(1) -> TR(2) -> BL(3) -> BR(4)", 
             fontsize=14, fontweight='bold')
ax.axis('off')

# Tight layout
plt.tight_layout()

def onclick(event):
    if event.xdata is None or event.ydata is None:
        return
    if len(points) >= 4:
        return
    
    x, y = int(event.xdata), int(event.ydata)
    points.append((x, y))
    
    # Ve diem
    color = colors[len(points)-1]
    ax.plot(x, y, 'o', color=color, markersize=12, markeredgecolor='white', markeredgewidth=2)
    ax.annotate(labels[len(points)-1], (x + 15, y - 15), 
                color=color, fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    # Neu du 4 diem
    if len(points) == 4:
        # Ve hinh thang
        poly_pts = [points[0], points[1], points[3], points[2]]
        polygon = Polygon(poly_pts, fill=True, alpha=0.2, color='yellow', edgecolor='yellow', linewidth=2)
        ax.add_patch(polygon)
        
        # Tinh kich thuoc
        top_w = np.linalg.norm(np.array(points[1]) - np.array(points[0]))
        bot_w = np.linalg.norm(np.array(points[3]) - np.array(points[2]))
        
        mid_top = ((points[0][0] + points[1][0])//2, (points[0][1] + points[1][1])//2)
        mid_bot = ((points[2][0] + points[3][0])//2, (points[2][1] + points[3][1])//2)
        
        ax.annotate(f"top={top_w:.0f}px", mid_top, color='yellow', fontsize=10)
        ax.annotate(f"bottom={bot_w:.0f}px", mid_bot, color='yellow', fontsize=10)
        
        # In config
        print("\n" + "=" * 50)
        print("  COPY VAO config.yaml:")
        print("=" * 50)
        print("ipm:")
        print("  src_points:")
        for i, (px, py) in enumerate(points):
            label_names = ["TL", "TR", "BL", "BR"]
            print(f"    - [{px}, {py}]    # {label_names[i]}")
        print(f"\n  # Top width: {top_w:.0f}px, Bottom width: {bot_w:.0f}px")
        print(f"  # Anh goc: {w}x{h}")
        
        # Uoc luong dst
        ratio = bot_w / max(top_w, 1)
        est_width = 15 if ratio < 2 else 12
        est_length = int(est_width * 2.5)
        print(f"\n  # Uoc luong dst_points:")
        print(f"  dst_points:")
        print(f"    - [0, 0]")
        print(f"    - [{est_width}, 0]")
        print(f"    - [0, {est_length}]")
        print(f"    - [{est_width}, {est_length}]")
        print(f"\n  # Suggested scale:")
        print(f"  bev_width: {est_width * 40}")
        print(f"  bev_height: {est_length * 20}")
        print(f"  scale_x: 40.0")
        print(f"  scale_y: 20.0")
        print("=" * 50)
    
    fig.canvas.draw()

# Connect event
fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
print("\nDone!")