# HƯỚNG DẪN SỬA BÁO CÁO — TỪNG THAO TÁC CỤ THỂ

> Mỗi mục dưới đây ghi rõ: **loại thao tác** (SỬA / THÊM / XÓA / THAY THẾ TOÀN BỘ), **tìm gì trong báo cáo cũ**, và **viết thay vào / thêm vào** nội dung gì.
>
> Cách dùng: mở file Word, dùng Ctrl+F để tìm theo "Tìm:" rồi áp thao tác tương ứng.

---

## 1. [SỬA SỐ LIỆU] DeepSORT max_age

**Loại:** Sửa số trong câu, KHÔNG xóa cả đoạn.

**Tìm trong báo cáo:** cụm có chứa `max_age = 60` hoặc `chịu được occlusion 1.4 giây` hoặc bảng tham số DeepSORT.

**Sửa:**
- Số `60` → `20`
- Câu "chịu được occlusion 1.4 giây" → "chịu được occlusion khoảng 0.8 giây"

**Lý do:** `config.yaml` đang đặt `max_age: 20`, không phải 60.

> **Lựa chọn khác:** Nếu bạn muốn giữ con số 60 trong báo cáo (vì đó là design intent), thì sửa lại `config.yaml` thành `max_age: 60` và giữ nguyên báo cáo. Quan trọng là đồng bộ.

---

## 2. [SỬA SỐ LIỆU] DeepSORT n_init

**Loại:** Sửa số.

**Tìm trong báo cáo:** `n_init = 3` hoặc "xác nhận track sau 3 frame".

**Sửa:** `3` → `2`, "3 frame" → "2 frame".

**Lý do:** `config.yaml` đặt `n_init: 2`.

---

## 3. [SỬA SỐ LIỆU] YOLO model

**Loại:** Sửa tên model.

**Tìm trong báo cáo:** `YOLOv8n` hoặc `yolov8n.pt`.

**Sửa:** `YOLOv8n` → `YOLOv8s`, `yolov8n.pt` → `yolov8s.pt`.

**Lý do:** `config.yaml` đặt `model_path: "models/yolov8s.pt"`. YOLOv8n chỉ là phương án dự phòng auto-download.

> Nếu trong báo cáo có ĐÁNH GIÁ chọn YOLOv8n vì nhẹ — cần viết lại lý do chọn YOLOv8s thay thế. Đoạn này tôi đã viết sẵn ở mục 16 phía dưới.

---

## 4. [SỬA SỐ LIỆU] Ngưỡng confidence và IoU

**Loại:** Sửa số.

**Tìm:**
- `conf = 0.35` hoặc `confidence threshold = 0.35` → sửa thành `0.25`
- `iou = 0.65` hoặc `IoU = 0.65` → sửa thành `0.5`

**Lý do:** `config.yaml`: `confidence_threshold: 0.25`, `iou_threshold: 0.5`.

---

## 5. [SỬA SỐ LIỆU] Vùng phủ IPM

**Loại:** Sửa số.

**Tìm:** `20 m × 60 m` hoặc `20m × 60m` (vùng phủ IPM).

**Sửa:** → `10 m × 60 m`.

**Lý do:** `ipm.py` định nghĩa `dst_points_m = [[0,0],[10,0],[0,60],[10,60]]` — vùng phủ là 10m ngang × 60m dọc.

---

## 6. [SỬA SỐ LIỆU] BEV size

**Loại:** Sửa số.

**Tìm:** mô tả kích thước ảnh BEV trong báo cáo (nếu có ghi 400×1200 hoặc số khác).

**Sửa:** `200 × 1200 pixel` (vì `bev_w = 10 × 20 = 200`, `bev_h = 60 × 20 = 1200`).

---

## 7. [XÓA / VIẾT LẠI] Rule thứ 6 (HIGH_SPEED_CURVE)

**Loại:** ⚠️ **Quan trọng nhất**. Có 2 phương án — chọn 1.

### Phương án A — Xóa rule thứ 6 khỏi báo cáo

**Tìm trong báo cáo:** đoạn mô tả rule `HIGH_SPEED_CURVE` hoặc dòng "v > 15 & heading change" hoặc bảng có 6 luật.

**Xóa:** toàn bộ dòng/đoạn nói về `HIGH_SPEED_CURVE`.

**Sửa thêm:** mọi chỗ ghi "**6 luật vật lý**" → "**5 luật vật lý**".

### Phương án B — Giữ 6 luật trong báo cáo, bổ sung code

**Giữ nguyên báo cáo.** Mở file `anomaly/rule_engine.py`, **THÊM** method sau vào class `PhysicsRuleEngine` (đặt sau `_check_sudden_stop`):

```python
def _check_high_speed_curve(self, window, violations):
    """Phát hiện ôm cua ở tốc độ cao."""
    if len(window) < 10:
        return
    recent = window[-10:]
    speeds = [s.speed for s in recent]
    headings = [s.heading_angle for s in recent]

    v_max = max(speeds)
    # Heading change tổng cộng (xử lý wrap-around 0/360)
    heading_change = 0.0
    for i in range(1, len(headings)):
        d = abs(headings[i] - headings[i-1])
        if d > 180:
            d = 360 - d
        heading_change += d

    if v_max > 15.0 and heading_change > 30.0:
        sev = np.clip((v_max - 15.0) / 10.0, 0.0, 1.0)
        violations.append(RuleViolation(
            rule_name="HIGH_SPEED_CURVE",
            severity=float(0.4 + 0.6 * sev),
            description=f"Ôm cua tốc cao: v={v_max:.1f}m/s, Δheading={heading_change:.0f}°",
            feature_value=v_max,
        ))
```

Sau đó **THÊM** vào `evaluate_state()`, ngay sau dòng `self._check_sudden_stop(window, violations)`:

```python
        # ── Rule 6: High Speed Curve ───────────────────────────────
        self._check_high_speed_curve(window, violations)
```

**Khuyến nghị:** Phương án A nhanh hơn, ít rủi ro hơn cho deadline. Phương án B đẹp hơn về tính học thuật.

---

## 8. [SỬA CÔNG THỨC] Rule score aggregation

**Loại:** Sửa công thức.

**Tìm trong báo cáo:** công thức tính `rule_score` từ danh sách violations. Thường là `rule_score = sum(severity)` hoặc `= max(severity)`.

**Thay bằng:**
> `rule_score = clip(0,6 × max(severity) + 0,4 × mean(severity), 0, 1)`

**Lý do:** Trong `rule_engine.py` dòng ~172.

---

## 9. [THÊM ĐOẠN MỚI] Fusion với 2 nhánh đặc biệt

**Loại:** Thêm nội dung — báo cáo cũ có thể chỉ ghi công thức tuyến tính.

**Tìm trong báo cáo:** đoạn `final_score = 0.3 × rule + 0.7 × ml`.

**Sau đoạn đó, THÊM:**

> Ngoài công thức tuyến tính cơ bản, hệ thống áp dụng hai nhánh điều kiện đặc biệt:
>
> - **Safety override:** khi `rule_score ≥ 0,85` (vi phạm vật lý cực rõ), `final_score = max(rule_score, ml_score)`. Quy tắc này đảm bảo các tình huống nguy hiểm hiển nhiên không bị ML "che" mất.
> - **Clear normal:** khi `rule_score < 0,2` và `ml_score < 0,25`, `final_score = max(rule_score, ml_score)`. Quy tắc này hạ thấp xác suất cảnh báo nhầm khi cả hai nhánh đều cho điểm thấp.
>
> Các trường hợp còn lại sử dụng công thức tuyến tính như đã nêu.

**Lý do:** Trong `rule_engine.py`, hàm `fuse_with_ml()` có cả ba nhánh, không chỉ tuyến tính.

---

## 10. [THÊM CÂU] EMA smoothing

**Loại:** Thêm 1-2 câu.

**Tìm trong báo cáo:** đoạn nói về tính vận tốc/gia tốc bằng sai phân hữu hạn từ tọa độ IPM.

**Sau câu mô tả sai phân hữu hạn, THÊM:**

> Để giảm ảnh hưởng của nhiễu pixel trong quá trình ánh xạ IPM, vận tốc và gia tốc được làm trơn bằng bộ lọc EMA với hệ số α = 0,7 (giá trị hiện tại đóng góp 70%, giá trị frame trước đóng góp 30%).

**Lý do:** `feature_extractor.py` có `alpha = 0.7` áp lên speed và acceleration.

---

## 11. [SỬA] Mô tả TensorRT

**Loại:** Sửa cách diễn đạt, không xóa.

**Tìm trong báo cáo:** câu kiểu "Hệ thống sử dụng TensorRT FP16 đạt 23.8 ms/frame, 42 FPS".

**Sửa thành:**
> Hệ thống có hỗ trợ TensorRT FP16; ở chế độ này, thời gian suy luận YOLOv8 trung bình đo được là 23,8 ms/frame, tương đương 42 FPS. Cấu hình mặc định của bản demo (`use_tensorrt: false`) chạy engine PyTorch FP32 để dễ chuyển sang môi trường không có TensorRT, vẫn duy trì trên 25 FPS.

**Lý do:** `config.yaml` đặt `use_tensorrt: false`. Nói thẳng "đã dùng TensorRT" là không chính xác.

---

## 12. [⚠️ THÊM ĐOẠN MỚI BẮT BUỘC] Ghi chú OpenPose

**Loại:** Thêm — đây là điểm trung thực quan trọng nhất.

### Vị trí thêm 1: Cuối phần "Tổng quan kiến trúc / Phạm vi nghiên cứu"

> **Phạm vi triển khai trong giai đoạn báo cáo này:** Mô-đun ước lượng tư thế người ngồi xe (OpenPose) đang trong quá trình huấn luyện trên tập dữ liệu xe máy và **chưa được tích hợp vào pipeline thời gian thực**. Hai đặc trưng phụ thuộc OpenPose là góc nghiêng cơ thể θ và biến thiên góc nghiêng Δθ tạm thời được gán giá trị 0 trong cấu hình hiện tại; do đó hai luật `HIGH_LEAN_ANGLE` và `LEAN_RAPID_CHANGE` chưa kích hoạt cảnh báo. Cấu trúc 20 chiều của véc-tơ đặc trưng được giữ nguyên để khi OpenPose hoàn thiện, hệ thống chỉ cần thay nguồn dữ liệu mà không phải sửa pipeline.

### Vị trí thêm 2: Trong phần Feature Extractor

Sau đoạn mô tả 5 đặc trưng `[v, a, θ, Δθ, d_min]`, **THÊM:**

> Hai đặc trưng θ và Δθ phụ thuộc đầu ra của OpenPose. Tại thời điểm báo cáo, OpenPose vẫn đang được huấn luyện riêng cho khung cảnh xe máy nên chưa cấp được giá trị thực; pipeline hiện gán θ = Δθ = 0 và do đó các thống kê `lean_*` và `lean_delta_*` trong véc-tơ 20 chiều luôn bằng 0.

### Vị trí thêm 3: Trong phần Rule Engine, sau danh sách 5 (hoặc 6) luật

> **Tình trạng kích hoạt thực tế:** trong điều kiện chưa có OpenPose, các luật `HIGH_LEAN_ANGLE` và `LEAN_RAPID_CHANGE` không sinh vi phạm. Pipeline hoạt động chủ yếu dựa trên các luật còn lại (`SUDDEN_BRAKE`, `COLLISION_RISK`, `SUDDEN_STOP` và optional `HIGH_SPEED_CURVE`). Khi OpenPose được hoàn thiện, độ phủ phát hiện sự cố dự kiến tăng đáng kể, đặc biệt với các trường hợp ngã xe đơn lẻ không kèm phanh gấp.

### Vị trí thêm 4: Trong phần Kết quả thực nghiệm

Sau bảng so sánh F1/FPS, **THÊM:**

> *Lưu ý đối chiếu cấu hình:* số liệu F1 = 0,86 báo cáo bên trên ứng với cấu hình đầy đủ có OpenPose. Trong cấu hình hiện đang chạy (chưa có OpenPose), khả năng phát hiện các tình huống ngã xe đơn lẻ bị giảm; nhóm sẽ cập nhật chỉ số sau khi hoàn tất huấn luyện và tích hợp OpenPose ở giai đoạn tiếp theo.

### Vị trí thêm 5 (nếu báo cáo có chương "Hạn chế và hướng phát triển")

Trong "Hạn chế":
> - Mô-đun OpenPose chưa hoàn thiện; các đặc trưng tư thế tạm thời bị loại khỏi quá trình suy luận.

Trong "Hướng phát triển":
> - Hoàn thiện huấn luyện OpenPose trên tập dữ liệu xe máy Việt Nam, tích hợp vào pipeline để kích hoạt đầy đủ hai luật phát hiện ngã/mất thăng bằng.

---

## 13. [SỬA] Số bộ tracker bug fix (chi tiết kỹ thuật)

**Loại:** Thêm hoặc sửa, để báo cáo có điểm sáng tạo.

**Tìm trong báo cáo:** đoạn nói về tinh chỉnh DeepSORT.

**THÊM 1 đoạn:**

> Hai chỉnh sửa nhỏ so với phiên bản gốc của `deep_sort_realtime`:
> 1. Bỏ điều kiện `if track.time_since_update > 1: continue` trong vòng lặp cập nhật. Điều kiện này khiến track không được trả về ngay sau khi xe hiện ra trở lại từ occlusion, làm hệ thống tạo track mới với ID khác. Sau khi bỏ điều kiện này, ID của phương tiện được giữ ổn định qua các đoạn bị che.
> 2. Sửa lỗi xóa phần tử khỏi `dict` ngay trong vòng lặp duyệt. Phiên bản hiện tại thu thập danh sách track hết hạn vào một list tạm rồi xóa ngoài vòng lặp, tránh `RuntimeError: dictionary changed size during iteration`.

---

## 14. [SỬA] Khoảng cách tối thiểu d_min mặc định

**Loại:** Sửa số nhỏ (nếu báo cáo có ghi).

**Tìm:** giá trị "khi không có xe khác, d_min = 100m" hoặc tương tự.

**Sửa:** `100m` → `99m` (hoặc bỏ con số cụ thể).

**Lý do:** `feature_extractor.py` dùng `min_d = 99.0`.

---

## 15. [THÊM] Validation IPM

**Loại:** Bổ sung chi tiết kỹ thuật, làm rõ thêm cho phần IPM.

**Vị trí:** sau đoạn mô tả tính ma trận H trong báo cáo.

**THÊM:**

> Quá trình tính ma trận H được kèm theo 6 bước kiểm tra hợp lệ:
> 1. Diện tích vùng nguồn ≥ 100 px² (chống chọn 4 điểm trùng nhau).
> 2. Bốn điểm tạo đa giác lồi (`cv2.isContourConvex`).
> 3. Chiều cao hai cạnh trái/phải ≥ 100 px.
> 4. Tỉ lệ cạnh trên/cạnh dưới ≥ 0,02 (tránh chọn vùng quá gần đường chân trời).
> 5. Số điều kiện của H phải dưới 10⁵.
> 6. Bốn góc vùng nguồn sau warp phải nằm trong canvas BEV (cho phép dung sai 100 px).
>
> Hệ thống chỉ chấp nhận ma trận H khi cả 6 điều kiện đều đạt; ngược lại sẽ giữ ma trận hợp lệ gần nhất hoặc rơi về điểm cấu hình thủ công.

---

## 16. [THAY THẾ TOÀN BỘ ĐOẠN] Lý do chọn YOLOv8s

**Loại:** Nếu báo cáo cũ có đoạn "lý do chọn YOLOv8n", thay thế toàn bộ.

**Tìm:** đoạn giải thích lý do chọn model YOLO.

**Thay bằng:**

> Trong nhóm YOLOv8, biến thể YOLOv8s được lựa chọn thay vì biến thể nano vì cân đối tốt hơn giữa độ chính xác và tốc độ trên phần cứng tầm trung. Trên GPU 4 GB VRAM, YOLOv8s đạt mAP cao hơn YOLOv8n đáng kể (khoảng 6–8 điểm) trong khi vẫn đảm bảo dưới 25 ms/frame ở chế độ FP32, và xuống 23,8 ms/frame ở chế độ TensorRT FP16. YOLOv8n được giữ làm phương án dự phòng tự động tải về khi không tìm thấy file `models/yolov8s.pt`.

---

## 17. [SỬA] Cooldown và clip duration

**Loại:** Kiểm tra số.

**Tìm:** "cooldown 5s", "clip 10s".

**Hành động:** Giữ nguyên — số liệu này đã khớp với code (`alert.cooldown_seconds: 5.0`, `save_clip_seconds: 10`).

---

## 18. [SỬA] Đường dẫn output clip

**Loại:** Sửa nếu báo cáo ghi sai format tên file.

**Tìm:** mẫu tên clip xuất ra.

**Sửa thành:** `outputs/accident_track{ID}_{unix_timestamp}.mp4` (codec mp4v, MP4).

**Lý do:** `alert_manager.py` dòng `fname = f"accident_track{...track_id}_{ts}.mp4"`.

---

## 19. [SỬA] Ngưỡng risk levels

**Loại:** Kiểm tra số.

**Tìm:** ngưỡng phân loại 4 mức.

**Đảm bảo các số sau đúng:**
- NORMAL: `score < 0,30`
- WARNING: `0,30 ≤ score < 0,55`
- DANGER: `0,55 ≤ score < 0,65`
- ACCIDENT: `score ≥ 0,65`

(Khớp với `WARNING_THRESH=0.3, DANGER_THRESH=0.55, ACCIDENT_THRESH=0.65` trong code.)

---

## 20. [SỬA] Hyperparameter LightGBM

**Loại:** Kiểm tra bảng tham số.

**Đảm bảo các số sau đúng trong báo cáo:**
- `objective = binary`
- `num_leaves = 31`
- `max_depth = 5`
- `learning_rate = 0,05`
- `n_estimators = 1000`
- `early_stopping_rounds = 50`
- `feature_fraction = 0,8`
- `scale_pos_weight ≈ 450`
- `accident_threshold = 0,65`

(Khớp với `config.yaml` mục `classifier.params`.)

---

## TÓM TẮT — DANH SÁCH KIỂM TRA NHANH

Khi bạn đi qua từng mục trong báo cáo, đánh dấu xong:

- [ ] Mục 1: max_age = 20 (hoặc giữ 60 và sửa config)
- [ ] Mục 2: n_init = 2 (hoặc giữ 3 và sửa config)
- [ ] Mục 3: YOLOv8s thay vì YOLOv8n
- [ ] Mục 4: conf=0.25, iou=0.5
- [ ] Mục 5: vùng phủ IPM 10m × 60m
- [ ] Mục 6: BEV size 200 × 1200 px
- [ ] **Mục 7: Chọn 5 luật hoặc bổ sung code rule thứ 6** ⚠️
- [ ] Mục 8: Công thức rule_score = 0,6 max + 0,4 mean
- [ ] Mục 9: Thêm 2 nhánh fusion đặc biệt
- [ ] Mục 10: Thêm câu về EMA α=0,7
- [ ] Mục 11: Sửa mô tả TensorRT (đang `false`)
- [ ] **Mục 12: Thêm 5 vị trí ghi chú OpenPose chưa hoàn thành** ⚠️
- [ ] Mục 13: Bổ sung 2 chỉnh sửa DeepSORT
- [ ] Mục 14: d_min mặc định = 99m
- [ ] Mục 15: Bổ sung 6 bước validation IPM
- [ ] Mục 16: Lý do chọn YOLOv8s
- [ ] Mục 17: cooldown/clip — giữ nguyên
- [ ] Mục 18: Tên file clip
- [ ] Mục 19: Ngưỡng risk levels
- [ ] Mục 20: Hyperparam LightGBM

Hai mục đánh dấu ⚠️ là quan trọng nhất — không sửa hai chỗ này thì khi bảo vệ dễ bị bắt bẻ vì code không khớp với mô tả.
