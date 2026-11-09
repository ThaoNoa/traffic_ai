# Traffic AI
Hệ thống camera AI cho giao thông đô thị tại khu vực Lĩnh Nam. Pipeline đọc video hoặc luồng camera, phát hiện phương tiện, giữ ID qua các frame, đổi toạ độ pixel sang mét, rồi cảnh báo sự cố bằng luật vật lý kết hợp LightGBM.
**AI traffic surveillance for urban video.** Detect vehicles, track them through occlusion, map pixels to metres, and raise incident alerts from physics rules fused with a LightGBM classifier.
Stack: Python, PyTorch, Ultralytics YOLOv8, OpenCV, DeepSORT, LightGBM, PyQt5. Bản demo mặc định chạy PyTorch. TensorRT là tuỳ chọn khi benchmark detector.
## Pipeline
1. **Nguồn video** — file, webcam (`0`), hoặc RTSP. Xem `utils/video_stream.py`.
2. **Phát hiện** — YOLOv8s, đầu vào 640 px, confidence 0.25, IoU NMS 0.5. Giữ bốn lớp COCO: car (2), motorcycle (3), bus (5), truck (7). Điểm dùng cho hình học là tâm cạnh dưới bbox (vị trí bánh chạm đường).
3. **Theo dõi** — DeepSORT (`deep-sort-realtime`), embedder MobileNet. `max_age = 20`, `n_init = 2`, `max_cosine_distance = 0.4`, `nn_budget = 50`. Tham số ưu tiên chuyển động khi gán ID.
4. **IPM** — ma trận đồng dạng đưa pixel sang mét. Vùng fallback trong `config/config.yaml` là 8 m × 30 m, tỉ lệ 20 px/m. Có tự hiệu chuẩn từ vạch làn (Canny + Hough) và hiệu chuẩn lại khi ma trận kém.
5. **Đặc trưng** — mỗi xe một trạng thái 5 chiều: vận tốc, gia tốc, góc nghiêng, biến thiên góc nghiêng, khoảng cách tới xe gần nhất. Cửa sổ 25 frame, bước 5, gộp thành vector 20 chiều (mean, std, max, min). Vận tốc và gia tốc được làm trơn EMA.
6. **Cảnh báo** — năm luật vật lý cộng điểm LightGBM. Ngưỡng rủi ro: NORMAL &lt; 0.30, WARNING &lt; 0.55, DANGER &lt; 0.65, ACCIDENT ≥ 0.65. Cooldown 5 giây theo từng track. Clip sự cố ghi vào `outputs/`.
7. **Giao diện** — cửa sổ PyQt5: xem video, dashboard và hộp thoại sự cố.
Năm luật trong `anomaly/rule_engine.py`:
| Luật | Ý nghĩa | Trạng thái |
| --- | --- | --- |
| `SUDDEN_BRAKE` | Gia tốc giảm mạnh | Đang dùng |
| `SUDDEN_STOP` | Đang chạy rồi dừng trong vài frame | Đang dùng |
| `COLLISION_RISK` | Khoảng cách nhỏ và vẫn đang chạy | Đang dùng |
| `HIGH_LEAN_ANGLE` | Người lái nghiêng quá ngưỡng | Chờ pose được nối vào pipeline |
| `LEAN_RAPID_CHANGE` | Góc nghiêng đổi quá nhanh | Chờ pose được nối vào pipeline |
Điểm cuối cùng không phải một trọng số cố định:
- `rule_score ≥ 0.85`: lấy `max(rule, ml)` để không bỏ qua vi phạm vật lý rất rõ.
- `rule_score < 0.2` và `ml_score < 0.25`: cũng lấy `max`, để hạ cảnh báo khi cả hai nhánh đều thấp.
- Các trường hợp còn lại: `0.3 × rule + 0.7 × ml`.
`pose/pose_estimator.py` ước lượng góc nghiêng bằng YOLOv8-pose. Script đánh giá và huấn luyện vẫn truyền `body_leans=None`, nên hai luật nghiêng chưa sinh cảnh báo.
## Yêu cầu
- Python 3.10 trở lên
- GPU NVIDIA nếu giữ `system.device: cuda` trong config. Đổi thành `cpu` để chạy không có CUDA; detector và tracker sẽ chậm hơn nhiều
- Bộ nhớ GPU khoảng 4 GB là đủ cho YOLOv8s ở 640 px. Số đo trong repo chạy trên RTX 3050 Laptop
## Cài đặt
```bash
git clone https://github.com/ThaoNoa/traffic_ai.git
cd traffic_ai
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install PyQt5           # chỉ cần cho run_gui.py
```
Chọn bản wheel PyTorch đúng CUDA của máy tại [pytorch.org](https://pytorch.org). `requirements.txt` ghim `numpy<2`.
Tải trọng số detector vào đúng đường dẫn config:
```bash
mkdir -p models
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"
mv yolov8s.pt models/yolov8s.pt
```
File `models/*.pt`, `*.engine`, `*.pkl` nằm trong `.gitignore`. Repo có sẵn `yolov8n.onnx` và `yolov8s.onnx` ở thư mục gốc; config mặc định không trỏ tới hai file đó.
## Cấu hình
Sửa `config/config.yaml` trước khi chạy.
```yaml
system:
  device: "cuda"          # "cpu" nếu không có GPU
video:
  source: "videos/demo.mp4"   # file, "0" (webcam), hoặc rtsp://...
detector:
  model_path: "models/yolov8s.pt"
  use_tensorrt: false
```
Đặt mật khẩu camera trong file local, không commit URL RTSP có user và password. File config đang public trên GitHub cần được đổi nguồn video và xoá thông tin đăng nhập camera.
Bốn điểm IPM fallback (`ipm.src_points`, thứ tự TL, TR, BL, BR) chỉ dùng khi tự hiệu chuẩn thất bại. Chọn lại trên một frame:
```bash
python pick_ipm_points.py videos/demo.mp4
```
Bấm đúng thứ tự: trên-trái, trên-phải, dưới-trái, dưới-phải, trên mặt đường. Cạnh trên hẹp hơn cạnh dưới. Rồi điền toạ độ in ra vào `src_points`.
## Chạy
Theo dõi và ghi video (mỗi hai frame xử lý một frame, xuất `outputs/tracking_output.mp4`):
```bash
python main.py --source videos/demo.mp4
```
Bỏ `--source` thì chương trình lấy `video.source` trong config.
Giao diện desktop:
```bash
python run_gui.py
```
`main.py` chỉ gồm detect, track và vẽ quỹ đạo. Nhánh IPM, luật và LightGBM nằm ở GUI, `eval_incident.py` và `train_on_tudat.py`.
## Huấn luyện LightGBM
`train_on_tudat.py` đọc video TU-DAT và CSV khoảng frame tai nạn. Sửa hai đường dẫn ở đầu file cho đúng máy:
```python
TU_DAT_ROOT = Path(r"D:\Downloads\datasets\TU-DAT")
ANNOTATION_PATH = Path(r"E:\NCKH\annotations.csv")
```
CSV cần cột video, frame bắt đầu và frame kết thúc (`video_name` / `start_frame` / `end_frame`, hoặc tên tương đương). Video không có trong CSV nhận nhãn theo thư mục: positive = 1, negative và challenging = 0. Cửa sổ 25 frame được gán positive khi ít nhất một nửa frame nằm trong khoảng sự cố.
```bash
python train_on_tudat.py
```
Model ghi theo `classifier.model_path` (`models/lgbm_accident.pkl`). Chưa có file này thì đánh giá chạy rule-only.
## Đánh giá
`eval_videos.json`:
```json
{
  "incidents": [
    {"path": "videos/eval/incident_01.mp4", "t_start": 4.0, "t_end": 10.0}
  ],
  "normals": [
    {"path": "videos/eval/normal_01.mp4"}
  ]
}
```
```bash
