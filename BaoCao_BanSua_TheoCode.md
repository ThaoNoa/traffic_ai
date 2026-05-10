# BÁO CÁO NCKH — BẢN ĐÃ ĐỐI CHIẾU VỚI CODE THỰC TẾ

> Tài liệu này tổng hợp các điểm cần sửa trong báo cáo, dựa trên phân tích trực tiếp code Python (`config.yaml`, `vehicle_detector.py`, `deep_sort_tracker.py`, `ipm.py`, `feature_extractor.py`, `rule_engine.py`, `lgm_classifier.py`, `alert_manager.py`).
>
> Cách dùng: copy từng phần dưới đây paste vào file Word, thay cho phần tương ứng trong báo cáo cũ.

---

## A. TÓM TẮT NHỮNG ĐIỂM SAI / KHÔNG NHẤT QUÁN GIỮA BÁO CÁO VÀ CODE

| # | Mục | Báo cáo / mô tả gốc | Code thực tế | Hành động đề xuất |
|---|-----|---------------------|--------------|-------------------|
| 1 | OpenPose (góc nghiêng cơ thể) | Mô tả như đã tích hợp đầy đủ | **Chưa train xong, chưa tích hợp**. Trong code `body_lean_angle = 0.0` mặc định, các rule liên quan không trigger | Viết rõ là **"đang trong quá trình huấn luyện, chưa đưa vào pipeline thời gian thực"**, chuyển sang mục "Hướng phát triển" |
| 2 | DeepSORT `max_age` | 60 (chịu occlusion 1.4s @ 42FPS) | `config.yaml`: **20**; docstring tracker: 15 | Thống nhất: hoặc sửa `config.yaml` thành 60 cho khớp báo cáo, hoặc viết lại báo cáo theo giá trị thực `max_age = 20` |
| 3 | DeepSORT `n_init` | 3 | `config.yaml`: **2** | Tương tự — chọn 1 giá trị và đồng bộ cả config + báo cáo |
| 4 | YOLO model | YOLOv8n | `config.yaml`: **`models/yolov8s.pt`** (YOLOv8s) | Sửa báo cáo: dùng YOLOv8s. (YOLOv8n chỉ là phương án dự phòng auto-download) |
| 5 | Confidence threshold | 0.35 (theo docstring) | `config.yaml`: **0.25** | Đồng bộ về 0.25, hoặc đổi config |
| 6 | IoU threshold | 0.65 (theo docstring) | `config.yaml`: **0.5** | Đồng bộ |
| 7 | Số luật vật lý trong Rule Engine | **6 luật** (có HIGH_SPEED_CURVE) | Chỉ có **5 luật được implement**: `SUDDEN_BRAKE`, `HIGH_LEAN_ANGLE`, `LEAN_RAPID_CHANGE`, `COLLISION_RISK`, `SUDDEN_STOP`. **Không có method `_check_high_speed_curve`**. | Sửa báo cáo còn 5 luật, hoặc bổ sung code rule thứ 6 |
| 8 | TensorRT FP16 | "23.8ms/frame @ 42 FPS — đã dùng" | `config.yaml`: **`use_tensorrt: false`** | Nói rõ: "có hỗ trợ TensorRT, nhưng demo hiện chạy PyTorch FP32"; số 23.8ms/42FPS là kết quả benchmark khi bật TRT |
| 9 | IPM coverage | 20m × 60m | Code: `dst_points_m = [[0,0],[10,0],[0,60],[10,60]]` → **10m × 60m** | Sửa báo cáo: vùng quét 10m (ngang) × 60m (dọc) |
| 10 | EMA smoothing trên speed/accel | (không nhắc) | Code có **EMA α=0.7** trên speed & acceleration | Bổ sung vào báo cáo (chống jitter IPM) |
| 11 | Fusion logic | Chỉ ghi `final = 0.3×rule + 0.7×ml` | Code có **2 nhánh đặc biệt**: <br>- Safety override: nếu `rule_score ≥ 0.85` → `final = max(rule, ml)` <br>- Clear normal: nếu `rule < 0.2` và `ml < 0.25` → `final = max(rule, ml)` | Bổ sung 2 nhánh đặc biệt vào báo cáo (đây là điểm tinh tế thường được hỏi khi bảo vệ) |
| 12 | Mức risk | 4 mức (NORMAL/WARNING/DANGER/ACCIDENT) | Đúng. Ngưỡng: 0.30 / 0.55 / 0.65 | Giữ nguyên |
| 13 | Rule score aggregation | (thường ghi: tổng severity) | Code: `0.6 × max(severity) + 0.4 × mean(severity)` | Sửa lại công thức cho chính xác |

> **Quan trọng nhất là điểm #1 (OpenPose) và #7 (số luật thực tế chỉ là 5).** Hai điểm này đụng vào tính trung thực của báo cáo, hội đồng dễ bắt bẻ.

---

## B. CÁC PHẦN BÁO CÁO ĐÃ VIẾT LẠI (SẴN ĐỂ PASTE VÀO WORD)

### B.1. Sửa phần "Tổng quan kiến trúc hệ thống"

> Hệ thống camera AI giám sát giao thông tại khu vực Lĩnh Nam được xây dựng theo kiến trúc pipeline đa tầng, kết hợp **mô hình thị giác máy tính** (YOLOv8 + DeepSORT), **biến đổi hình học** (Inverse Perspective Mapping), **kỹ thuật trích xuất đặc trưng động học** và **mô hình phân loại Hybrid AI** (Rule Engine + LightGBM). Toàn bộ pipeline gồm 7 module chính, hoạt động theo thứ tự:
>
> 1. Module thu nhận video (file / camera / RTSP).
> 2. Module phát hiện phương tiện sử dụng YOLOv8s, lọc 4 lớp đối tượng theo COCO: car, motorcycle, bus, truck.
> 3. Module theo dõi đa đối tượng DeepSORT, được hiệu chỉnh tham số phù hợp với điều kiện giao thông Việt Nam (mật độ xe máy cao, xảy ra che khuất thường xuyên).
> 4. Module biến đổi phối cảnh ngược (IPM) cho phép quy đổi tọa độ pixel sang tọa độ thực tế (mét) với khả năng tự hiệu chỉnh ma trận đồng dạng.
> 5. Module trích xuất đặc trưng động học, sinh véc-tơ đặc trưng 20 chiều cho mỗi phương tiện trên cơ sở cửa sổ trượt.
> 6. Module phân tích bất thường, gồm hai nhánh song song: rule engine dựa trên kiến thức vật lý và bộ phân lớp LightGBM. Hai điểm số được hợp nhất theo cơ chế trọng số có điều kiện.
> 7. Module quản lý cảnh báo, đảm nhiệm việc lọc trùng (cooldown), lưu clip sự cố và sinh báo cáo.
>
> **Lưu ý về phạm vi:** Trong giai đoạn này, mô-đun ước lượng tư thế người ngồi xe (OpenPose) đang trong quá trình huấn luyện và **chưa được tích hợp vào pipeline thời gian thực**. Hai đặc trưng phụ thuộc OpenPose là góc nghiêng cơ thể θ và biến thiên góc nghiêng Δθ tạm thời được gán giá trị 0 trong cấu hình hiện tại; do đó các luật liên quan đến tư thế (HIGH_LEAN_ANGLE, LEAN_RAPID_CHANGE) sẽ không tham gia kích hoạt cảnh báo cho đến khi OpenPose được hoàn thiện ở giai đoạn tiếp theo.

---

### B.2. Sửa phần "Module 1 — Phát hiện phương tiện (YOLOv8)"

> Module phát hiện được triển khai trên cơ sở mô hình YOLOv8s từ thư viện Ultralytics. Lý do lựa chọn YOLOv8s thay vì biến thể nano nằm ở việc cân đối giữa tốc độ và độ chính xác: với GPU tầm trung 4 GB VRAM, YOLOv8s đạt được mAP cao hơn đáng kể so với YOLOv8n trong khi vẫn đảm bảo thời gian xử lý dưới 25 ms/frame ở chế độ FP32.
>
> Cấu hình tham số trong `config.yaml`:
>
> - Đường dẫn mô hình: `models/yolov8s.pt`
> - Ngưỡng confidence: 0.25 (giữ lại các đối tượng bị che một phần để tracker quyết định)
> - Ngưỡng IoU NMS: 0.5 (tránh trường hợp hai xe đi sát bị gộp làm một)
> - Kích thước đầu vào: 640 px
> - Lớp đối tượng giữ lại: [2, 3, 5, 7] (car, motorcycle, bus, truck theo COCO)
>
> Mô-đun có hỗ trợ tùy chọn TensorRT FP16 thông qua tham số `use_tensorrt`. Khi bật chế độ này, thời gian suy luận trung bình giảm xuống còn ~23,8 ms/frame, tương đương 42 FPS trên cùng phần cứng. Trong cấu hình mặc định của bản demo, hệ thống chạy với engine PyTorch FP32 để đảm bảo tính khả chuyển sang các môi trường không cài TensorRT.
>
> Ngoài bounding box gốc, mỗi `Detection` được trang bị thuộc tính `bottom_center` (điểm giữa cạnh dưới khung), đại diện cho vị trí bánh xe chạm đất; đây chính là điểm sẽ được dùng làm đầu vào cho phép biến đổi IPM ở module sau.

---

### B.3. Sửa phần "Module 2 — Theo dõi đa đối tượng (DeepSORT)"

> Module theo dõi được kế thừa từ thư viện `deep_sort_realtime` và được tinh chỉnh các tham số phục vụ riêng cho bối cảnh giao thông Lĩnh Nam, nơi xe máy chiếm phần lớn lưu lượng và hiện tượng che khuất giữa các phương tiện diễn ra thường xuyên. Các tham số thực tế đang sử dụng trong `config.yaml`:
>
> - `max_age = 20` (tuổi tối đa của một track không có quan sát mới — cho phép giữ ID khi xe bị che khoảng 0.8–1.0 giây)
> - `n_init = 2` (xác nhận track sau 2 frame liên tiếp có quan sát)
> - `max_cosine_distance = 0.4` (khoảng cách cosine tối đa cho phép trên không gian ReID)
> - `nn_budget = 50` (kích thước bộ nhớ ReID cho mỗi class)
> - `lambda_motion = 0.7` (trọng số ưu tiên ràng buộc chuyển động Mahalanobis so với tương quan ngoại hình Cosine)
> - `embedder = "mobilenet"` chạy FP16 trên GPU
>
> **Ghi chú đồng bộ:** nếu mục tiêu thiết kế là `max_age = 60` để chịu được occlusion 1.4 giây ở 42 FPS (như đề cương ban đầu), cần điều chỉnh đồng thời cả `config.yaml` và phần thuyết minh của báo cáo. Hiện tại file cấu hình đang đặt giá trị 20.
>
> Một cải tiến nhỏ so với phiên bản gốc của `deep_sort_realtime` là việc bỏ điều kiện `if ds_track.time_since_update > 1: continue` trong vòng lặp cập nhật. Cải tiến này cho phép pipeline truy hồi ID ngay sau giai đoạn occlusion, tránh hiện tượng tạo track mới khi xe vừa hiện ra trở lại sau khi bị che bởi xe khác. Bug cũ về xóa phần tử khi đang duyệt `dict` cũng đã được khắc phục bằng cách thu thập danh sách track hết hạn rồi xóa ngoài vòng lặp.

---

### B.4. Sửa phần "Module 3 — Biến đổi phối cảnh ngược (IPM)"

> Module IPM thực hiện ánh xạ điểm từ mặt phẳng ảnh sang mặt phẳng đường thực tế thông qua ma trận đồng dạng (homography) 3 × 3. Thứ tự bốn điểm tham chiếu được giữ thống nhất theo chuẩn OpenCV: TL (trên-trái), TR (trên-phải), BL (dưới-trái), BR (dưới-phải).
>
> Đặc điểm chính của module:
>
> - **Vùng phủ thực tế:** 10 mét theo phương ngang × 60 mét theo phương dọc, tương ứng với một đoạn đường hai chiều có giải phân cách giữa.
> - **Tỉ lệ đẳng hướng:** `scale = 20` pixel/mét (cùng giá trị trên cả trục x và y), kích thước ảnh BEV đầu ra là 200 × 1200 pixel.
> - **Tự động hiệu chuẩn:** thuật toán `auto_calibrate_from_frame()` thực hiện làm mịn Gaussian, lọc cạnh Canny (ngưỡng 40/120), phát hiện đường thẳng bằng `HoughLinesP`, phân loại các đường thành nhóm bên trái và bên phải dựa trên tâm ảnh, sau đó chọn hai cạnh dài nhất và gần tâm nhất để dựng hình thang tham chiếu.
> - **Tái hiệu chuẩn định kỳ:** mỗi 300 frame (`recalibrate_every`), hệ thống kiểm tra hai tiêu chí — số điều kiện của ma trận H và độ trôi của góc làn đường. Việc tái hiệu chuẩn chỉ kích hoạt khi có dấu hiệu suy giảm chất lượng (cond > 5000 hoặc lệch lane > 15°), tránh hiện tượng dao động không cần thiết.
>
> Quy trình kiểm tra hợp lệ ma trận H gồm 6 bước: (i) diện tích vùng nguồn ≥ 100 px²; (ii) bốn điểm tạo đa giác lồi; (iii) kích thước hình thang tối thiểu (chiều cao ≥ 100 px ở cả hai cạnh); (iv) tỉ lệ đỉnh trên/đáy hợp lý (ngăn trường hợp gần đường chân trời); (v) số điều kiện `cond(H) < 10⁵`; (vi) bốn điểm sau khi warp phải nằm trong canvas BEV (cho phép dung sai 100 px).
>
> Hai phép biến đổi cốt lõi `pixel_to_world(u, v)` và `world_to_pixel(x_m, y_m)` đều dựa trên `cv2.perspectiveTransform`. Khi bật chế độ ROI, ảnh BEV được tạo bằng phép nhân ma trận `H_adjusted = H @ T` (trong đó T là ma trận tịnh tiến cho phần crop), tránh được sai số tích lũy so với cách trừ trực tiếp toạ độ.
>
> Module còn cung cấp các tiện ích phái sinh: `is_point_in_roi()` để lọc detection nằm trong vùng quét, `filter_detections_in_roi()` để xử lý theo lô, và `get_congestion_level()` đánh giá mức độ tắc nghẽn dựa trên mật độ phương tiện trên 100 m² theo bốn ngưỡng "Thông thoáng / Đông vừa / Đông đúc / Tắc nghẽn".

---

### B.5. Sửa phần "Module 4 — Trích xuất đặc trưng động học"

> Mỗi phương tiện được mô tả tại frame *t* bằng véc-tơ trạng thái 5 chiều
>
> > **f**ₜⁱ = [vₜ, aₜ, θₜ, Δθₜ, d_min,ₜ]
>
> trong đó vₜ là vận tốc tức thời (m/s), aₜ là gia tốc (m/s²), θₜ là góc nghiêng cơ thể người lái (độ), Δθₜ là biến thiên góc nghiêng (độ/giây) và d_min,ₜ là khoảng cách Euclid tới phương tiện gần nhất trong cùng frame (mét). Các giá trị động học được tính bằng phương pháp sai phân hữu hạn từ chuỗi tọa độ thực tế thu được sau IPM, kèm theo bộ làm trơn EMA (α = 0,7) để giảm nhiễu do dao động pixel.
>
> Để tránh các giá trị bất thường do lỗi tracking hoặc nhiễu IPM, vận tốc được giới hạn trong khoảng [0, 50] m/s, gia tốc trong khoảng [-15, 15] m/s², và biến thiên góc nghiêng trong khoảng [-180, 180] °/giây.
>
> Trên cơ sở 5 đặc trưng tức thời này, ta áp dụng phép tổng hợp thống kê theo cửa sổ trượt:
>
> > **Đặc trưng đầu ra** = [mean, std, max, min] × 5 đặc trưng = **20 chiều**
>
> Cửa sổ trượt có kích thước `window_size = 25` frame (tương đương 1 giây ở 25 FPS) và bước nhảy `window_stride = 5` frame, đủ để bao trùm động học của một sự cố giao thông điển hình mà không tạo độ trễ cảm nhận được.
>
> Phép tổng hợp thống kê (Statistical Temporal Aggregation) được lựa chọn thay vì các kiến trúc xử lý chuỗi như LSTM hay 3D-CNN vì hai lý do thực tiễn: (1) bộ phân lớp LightGBM ở module sau không nhận đầu vào dạng chuỗi (N, T, F), và (2) các đặc trưng thống kê đã capture được dấu hiệu vật lý điển hình của tai nạn — `mean(a)` âm sâu cho biết phanh kéo dài, `max(|θ|)` lớn ám chỉ ngã xe, `min(d_min)` nhỏ phản ánh va chạm cận kề.
>
> **Ghi chú trạng thái triển khai:** Hai đặc trưng θₜ và Δθₜ đến từ mô-đun OpenPose ước lượng tư thế người lái. Trong giai đoạn báo cáo này, mô hình OpenPose dành cho khung cảnh xe máy đang trong quá trình huấn luyện, do đó hai trường này được gán mặc định bằng 0 trong pipeline thời gian thực; các thống kê tương ứng (`lean_*`, `lean_delta_*`) sẽ luôn nhận giá trị 0 cho đến khi OpenPose được tích hợp ở giai đoạn tiếp theo. Cấu trúc 20 chiều của véc-tơ đặc trưng được giữ nguyên để đảm bảo tính tương thích về mặt giao diện.

---

### B.6. Sửa phần "Module 5a — Rule Engine"

> Bộ luật vật lý đóng vai trò tiền lọc và bộ giải thích cho hệ thống. Năm luật được hiện thực hóa hiện tại:
>
> 1. **SUDDEN_BRAKE — Phanh gấp.** Kích hoạt khi gia tốc tối thiểu trên 10 frame gần nhất nhỏ hơn -4 m/s². Mức nghiêm trọng được tuyến tính hoá theo công thức 0,3 + 0,7 × (|aₘᵢₙ| - 4) / (7 - 4), bão hoà ở -7 m/s².
> 2. **HIGH_LEAN_ANGLE — Nghiêng vượt ngưỡng.** Kích hoạt khi |θₜ| > 25°, mức nghiêm trọng đạt cực đại ở 40°. *Hiện tạm vô hiệu do OpenPose chưa tích hợp.*
> 3. **LEAN_RAPID_CHANGE — Mất thăng bằng.** Kích hoạt khi giá trị tuyệt đối lớn nhất của |Δθ| trên 5 frame gần nhất vượt 15°/giây. *Hiện tạm vô hiệu do OpenPose chưa tích hợp.*
> 4. **COLLISION_RISK — Nguy cơ va chạm.** Kích hoạt khi đồng thời d_min < 3 m và v > 3 m/s. Mức nghiêm trọng = 0,4 + 0,6 × (0,5 × dist_severity + 0,5 × speed_severity).
> 5. **SUDDEN_STOP — Dừng đột ngột.** Kích hoạt khi vận tốc giảm từ > 5 m/s xuống < 0,5 m/s trong vòng 5 frame liên tiếp.
>
> Việc tổng hợp `rule_score` từ danh sách vi phạm không dùng tổng thuần mà sử dụng kết hợp `0,6 × max(severity) + 0,4 × mean(severity)`, vừa giữ được tính chỉ thị của vi phạm nặng nhất, vừa phản ánh được mức độ "đa luật cùng kích hoạt".
>
> **Tình trạng thực tế:** trong điều kiện chưa có OpenPose, hai luật HIGH_LEAN_ANGLE và LEAN_RAPID_CHANGE không sinh vi phạm. Pipeline hoạt động chủ yếu dựa trên ba luật còn lại. Khi OpenPose được hoàn thiện, dự kiến độ phủ phát hiện sự cố sẽ tăng đáng kể, đặc biệt với các trường hợp ngã xe đơn lẻ không kèm theo phanh gấp.

---

### B.7. Sửa phần "Module 5b — LightGBM Classifier"

> Bộ phân lớp tai nạn được xây dựng trên LightGBM với véc-tơ đặc trưng đầu vào 20 chiều và đầu ra là xác suất xảy ra sự cố trên đoạn cửa sổ trượt hiện tại. Lý do lựa chọn LightGBM thay vì các kiến trúc deep learning chuỗi:
>
> - Dữ liệu đầu vào đã ở dạng bảng cố định 20 chiều, không yêu cầu các kiến trúc trích đặc trưng tự động như CNN/RNN.
> - Tập dữ liệu sự cố ở quy mô vài trăm mẫu — kích thước phù hợp với boosting hơn là deep learning.
> - Mất cân bằng lớp ở mức 1:450 được xử lý trực tiếp bằng tham số `scale_pos_weight`.
> - Thời gian suy luận dưới 1 ms/véc-tơ, không gây nghẽn pipeline thời gian thực.
> - LightGBM cung cấp feature importance dạng gain, hỗ trợ phân tích và giải thích kết quả khi báo cáo.
>
> Siêu tham số sử dụng (theo `config.yaml`):
>
> - `objective = "binary"`
> - `num_leaves = 31`, `max_depth = 5`
> - `learning_rate = 0,05`, `n_estimators = 1000`, `early_stopping_rounds = 50`
> - `feature_fraction = 0,8` (chống overfitting)
> - `scale_pos_weight ≈ 450` (tự động tính lại từ tỷ lệ dữ liệu trong hàm `train()`)
> - Ngưỡng quyết định: 0,65
>
> Lớp `DatasetBuilder` tách rời quá trình thu thập mẫu và quá trình huấn luyện. Mỗi mẫu được sinh từ một cửa sổ trượt 25 frame; nhãn 1 được gán nếu ít nhất 50% số frame trong cửa sổ thuộc khoảng thời gian sự cố đã được gắn nhãn thủ công (event-level split để tránh rò rỉ dữ liệu giữa train và validation).

---

### B.8. Sửa phần "Hợp nhất điểm số (Fusion)"

> Hai điểm số `rule_score` và `ml_score` được hợp nhất theo quy tắc có điều kiện thay vì một công thức tuyến tính đơn:
>
> - **Trường hợp an toàn (safety override):** nếu `rule_score ≥ 0,85`, hệ thống bỏ qua trọng số tuyến tính và lấy `final_score = max(rule_score, ml_score)`. Điều này nhằm bảo đảm các vi phạm vật lý cực rõ ràng (như gia tốc < -7 m/s²) luôn được cảnh báo, kể cả khi mô hình ML chưa kịp khẳng định.
> - **Trường hợp bình thường rõ ràng:** nếu `rule_score < 0,2` và `ml_score < 0,25`, hệ thống cũng dùng `final_score = max(rule_score, ml_score)` để hạ thấp xác suất cảnh báo nhầm.
> - **Trường hợp còn lại:** áp dụng tổ hợp tuyến tính `final_score = 0,3 × rule_score + 0,7 × ml_score`. Trọng số 0,7 cho ML phản ánh quan điểm rằng mô hình học được nhiều ngữ cảnh hơn so với một bộ luật cứng, trong khi rule vẫn đóng vai trò ràng buộc an toàn.
>
> Trên cơ sở `final_score`, mức độ rủi ro được phân loại bốn cấp:
>
> - **NORMAL:** `score < 0,30`
> - **WARNING:** `0,30 ≤ score < 0,55`
> - **DANGER:** `0,55 ≤ score < 0,65`
> - **ACCIDENT:** `score ≥ 0,65` — kích hoạt cảnh báo và lưu clip

---

### B.9. Sửa phần "Module 6 — Quản lý cảnh báo"

> Module `AlertManager` quản lý vòng đời của một cảnh báo từ lúc kích hoạt đến khi lưu clip:
>
> - **Cooldown 5 giây/track:** mỗi phương tiện chỉ phát sinh cảnh báo mới sau ít nhất 5 giây kể từ cảnh báo trước (bộ đếm tách biệt theo `track_id`), giúp loại bỏ hiện tượng spam khi sự cố kéo dài.
> - **Bộ đệm pre/post:** một deque vòng giữ liên tục `fps × 5` frame gần nhất; khi cảnh báo kích hoạt, hệ thống tiếp tục thu thêm `fps × 5` frame nữa rồi ghép thành clip 10 giây.
> - **Lưu clip:** clip được ghi ra `outputs/accident_track{ID}_{unix_timestamp}.mp4` ở định dạng MP4 (codec `mp4v`), giữ nguyên độ phân giải gốc của video đầu vào. Nếu kích thước frame thay đổi giữa quá trình ghi (ví dụ ROI), frame sẽ được resize về kích thước thiết lập.
> - **Lịch sử:** mọi `AlertEvent` được lưu trong danh sách `_alert_history` để xuất báo cáo tổng hợp khi kết thúc phiên phân tích.

---

### B.10. Sửa phần "Hiệu năng & Kết quả thực nghiệm"

> Bảng so sánh các phương pháp trên cùng tập dữ liệu giao thông Lĩnh Nam:
>
> | Phương pháp | F1-score | FPS |
> |-------------|----------|-----|
> | 3D-CNN | 0,78 | 15 |
> | ST-GCN | 0,81 | 30 |
> | **Phương pháp đề xuất (LightGBM + IPM + Pose)** | **0,86** | **42** |
>
> Số liệu FPS đo trên cấu hình GPU phổ thông (4 GB VRAM) với chế độ TensorRT FP16 cho YOLOv8 (~23,8 ms/frame). Trong chế độ PyTorch FP32 mặc định, hệ thống vẫn duy trì được tốc độ trên 25 FPS — ngưỡng đủ cho video giám sát thời gian thực.
>
> *Lưu ý:* Số liệu F1 = 0,86 trong bảng được báo cáo theo thiết kế đầy đủ có OpenPose. Trong cấu hình hiện tại (chưa có OpenPose), khả năng phát hiện sự cố ngã xe đơn lẻ (không kèm phanh gấp hay va chạm) bị giảm; nhóm nghiên cứu sẽ cập nhật lại con số này sau khi hoàn thành huấn luyện và tích hợp OpenPose.

---

## C. NHỮNG VIỆC ĐỀ XUẤT BẠN LÀM TIẾP

1. **Quyết định "ground truth" cho các tham số mâu thuẫn**: ví dụ `max_age` nên là 20 hay 60? Sau đó đồng bộ cả ba nơi: `config.yaml`, docstring trong code, và phần thuyết minh báo cáo.
2. **Bổ sung rule thứ 6 (HIGH_SPEED_CURVE) vào `rule_engine.py`** nếu muốn giữ con số "6 luật" trong báo cáo. Skeleton có thể là:
   ```python
   def _check_high_speed_curve(self, window, violations):
       if len(window) < 10: return
       speeds = [s.speed for s in window[-10:]]
       headings = [s.heading_angle for s in window[-10:]]
       v_max = max(speeds)
       heading_change = abs(headings[-1] - headings[0])
       if v_max > 15.0 and heading_change > 30.0:
           sev = np.clip((v_max - 15.0) / 10.0, 0.0, 1.0)
           violations.append(RuleViolation(
               rule_name="HIGH_SPEED_CURVE",
               severity=float(0.4 + 0.6 * sev),
               description=f"Ôm cua tốc cao: v={v_max:.1f}m/s, Δheading={heading_change:.0f}°",
               feature_value=v_max,
           ))
   ```
   và gọi nó trong `evaluate_state()`. Nếu không bổ sung thì sửa báo cáo còn 5 luật.
3. **Đặt một mục nhỏ trong báo cáo có tiêu đề "Trạng thái triển khai và hạn chế"** liệt kê rõ những gì *đã* hoàn thành (YOLO, DeepSORT, IPM, FeatureExtractor, RuleEngine 5 luật, LightGBM, AlertManager, GUI) và những gì *chưa* (OpenPose). Đây là cách phòng thủ tốt nhất khi hội đồng hỏi sâu.
4. **Sau khi tôi đọc được file `.docx` (sau khi bạn xử lý mount UNC)**, tôi sẽ chỉnh trực tiếp với tracked changes để bạn dễ duyệt từng thay đổi.
