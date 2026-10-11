# Báo cáo Tuần 7 — Nhận diện Out-of-Distribution (OOD) & Tối ưu hóa Toàn diện Hệ thống

**Đồ án:** Hệ thống AI nhận diện đa nhãn loại hoa và gợi ý chăm sóc (VBxIris)  
**Tuần:** 7 — Nghiên cứu & Thực nghiệm Phát hiện Ngoài phân phối (OOD Detection) + Tích hợp Backend & Web  

---

## 1. Bối cảnh & Vấn đề giải quyết trong Tuần 7

Ở các tuần trước (Tuần 5–6), hệ thống đã hoàn thiện mô hình phân loại đơn nhãn DenseNet121 kết hợp cơ chế quét cửa sổ trượt đa tỉ lệ (21 sliding windows) để nhận diện từ 1 đến 3 bông hoa trong ảnh. Tuy nhiên, qua thử nghiệm thực tế phát sinh một vấn đề nghiêm trọng: **Hiện tượng nhận diện nhầm ảnh ngoài phân phối (Out-of-Distribution - OOD)**.

* **Vấn đề 1 (Hoa lạ - Near-OOD):** Người dùng chụp một loài hoa lạ không nằm trong 10 lớp được huấn luyện (ví dụ: hoa cẩm tú cầu, hoa loa kèn, hoa bìm bìm...), mô hình cũ dùng softmax gượng ép phân loại vào 1 trong 10 lớp quen thuộc với độ tin cậy giả tạo rất cao (Overconfidence).
* **Vấn đề 2 (Mảng nền & Vật thể khác - Far-OOD):** Khi trượt 21 cửa sổ, các mảng cắt chỉ chứa đất, thảm cỏ, lá cây hoặc các vật thể/con vật vô tình bị mô hình gán nhầm thành hoa (False Positives), dẫn đến việc xuất hiện nhãn rác.
* **Hạn chế của giải pháp cũ (MSP >= 0.85):** Cơ chế lọc ngưỡng xác suất softmax Maximum Softmax Probability (MSP >= 0.85) tuy đơn giản nhưng làm **bỏ sót oan uổng 15.2% ảnh hoa thật (ID)**, trong khi vẫn để **lọt qua tới 88.98% ảnh hoa lạ** và **60.00% mảng nền**.

Mục tiêu trọng tâm của Tuần 7 là nghiên cứu, thực nghiệm và đưa ra giải pháp giải quyết triệt để vấn đề OOD mà không làm suy giảm độ chính xác của 10 loài hoa ban đầu.

---

## 2. Công việc đã hoàn thành

### 2.1. Xây dựng bộ dữ liệu đánh giá OOD đa tầng (`data/ood/`)

Đã thiết lập bộ dữ liệu kiểm thử OOD phong phú, phân tầng rõ ràng theo độ khó và nguồn gốc:
1. **Hoa lạ (Near-OOD):** Thu thập từ tập chuẩn **Oxford 102 Flowers**. Loại trừ 19 loài có hình thái quá tương đồng với 10 loài ID. Đặc biệt, qua phân tích cosine similarity và kiểm tra bằng dHash, đã phát hiện và loại trừ lớp **#11 (Coltsfoot)** do thực chất cùng chi/loài với Bồ Công Anh. Tách thành tập Test riêng (30 loài chưa từng thấy lúc tinh chỉnh) và tập Val (53 loài).
2. **Họa tiết & Bề mặt (Far-OOD - `far_texture`):** 47 lớp từ tập **DTD (Describable Textures Dataset)** (đã loại bỏ lớp `flowery`).
3. **Động vật (Far-OOD - `far_animal`):** 37 giống chó mèo từ tập **Oxford-IIIT Pet**.
4. **Vật thể & Cảnh tự nhiên tự thu thập (`far_own`):** Ảnh chụp lá cây, cành khô, đồ vật ngoài thiên nhiên.
5. **Mảng nền cắt tự nhiên (`bg_crop`):** 126 mảng nền kích thước đa dạng cắt ra từ ảnh phong cảnh thiên nhiên không chứa hoa.
6. **Mảng nền tự động từ Places365:** Bổ sung 1.072 mảng nền đa dạng từ dataset **Places365** (rừng, bãi cỏ, vách đá, bầu trời, hàng rào...) cho tập huấn luyện bổ trợ.

### 2.2. Thực nghiệm & Đối chuẩn các giải pháp OOD (`notebooks/07_ood_detection.ipynb`)

Đã tiến hành thực nghiệm so sánh hai hướng tiếp cận chính trên môi trường Google Colab (GPU T4):

#### Hướng tiếp cận 1: Hậu xử lý điểm số (Post-hoc Scoring Methods)
* **Maximum Softmax Probability (MSP)** kết hợp **Temperature Scaling ($T_{cal} = 1.166$)**: Xác nhận việc căn chỉnh nhiệt độ giúp giảm sai số tự tin (ECE) trên tập ID nhưng hầu như không cải thiện khả năng phát hiện OOD.
* **Energy Score ($T=1.0$)**: Khắc phục nhược điểm bão hòa của softmax, đạt AUROC macro 0.935 (so với MSP 0.909).
* **Mahalanobis Distance (MD) & Relative Mahalanobis Distance (RMD)**: Trích xuất vector đặc trưng 1.024 chiều từ lớp `GlobalAveragePooling2D`. Sử dụng hiệp phương sai co rút Ledoit-Wolf. MD rất mạnh ở Far-OOD (AUROC texture 0.983) trong khi RMD vượt trội ở Near-OOD (AUROC hoa lạ 0.898).
* **Tổ hợp đa điểm số ($Energy \vee MD \vee RMD$)**: Đạt AUROC macro 0.938 trên validation. Tuy nhiên, khi áp dụng vào cấp ảnh (21 cửa sổ trượt), tỷ lệ hoa lạ lọt qua vẫn còn ở mức cao (~55.3%).

#### Hướng tiếp cận 2: Kiến trúc 11 đầu ra với lớp từ chối (11-Class Classifier with Reject Class)
* Thay vì chỉ dựa vào ngưỡng điểm hậu xử lý, thiết kế lại đầu phân loại (Classifier Head) gồm **11 lớp** (10 lớp hoa ID quen thuộc + lớp thứ 11 là **Reject Class** dành cho hoa lạ và mảng nền).
* Tận dụng trọng số DenseNet121 đã đóng băng (warm-start), huấn luyện nhanh đầu phân loại trên đặc trưng 1.024-d với kỹ thuật cân bằng trọng số phạt mẫu (`rho = 0.1, 0.2, 0.3`).
* Quét tìm cấu hình tối ưu trên tập Validation với 21 cửa sổ trượt: Cấu hình thắng cuộc đạt chuẩn là **`nền=False | rho=0.3 | confidence_gate=0.7`**.

### 2.3. Đánh giá khách quan trên tập Test Set độc lập (21 Sliding Windows)

Toàn bộ đánh giá cuối cùng được đo lường một lần duy nhất trên tập Test (752 ảnh ID + 405 ảnh OOD + 150 ảnh ghép đa nhãn tạo từ tập Test):

```
                                  1hoa_đúng  1hoa_thừa  near_flower  far_texture  far_animal  far_own  bg_crop  far_TB  2hoa_đủ  2hoa_thừa  3hoa_đủ  3hoa_thừa
HIỆN TẠI (MSP >= 0.85)                0.955      0.116        0.890        0.400       0.846    0.533    0.600   0.595    0.707      0.053    0.413      0.107
ĐẦU 11 LỚP (đã chọn)                  0.961      0.165        0.284        0.088       0.282    0.133    0.114   0.154    0.707      0.240    0.373      0.240
```

* **Khả năng giữ lại hoa thật (`1hoa_đúng`):** Tăng từ 95.48% lên **96.14%** (khoảng tin cậy Bootstrap 95%: [94.7%, 97.5%]).
* **Khả năng chặn hoa lạ (`near_flower`):** Tỷ lệ lọt qua giảm mạnh từ 88.98% xuống chỉ còn **28.39%** (chặn được hơn 71.6% ảnh hoa lạ).
* **Khả năng triệt tiêu mảng nền (`bg_crop`):** Tỷ lệ lọt giảm từ 60.00% xuống còn **11.43%**.
* **Trung bình vật thể không phải hoa (`far_TB`):** Tỷ lệ lọt giảm từ 59.49% xuống còn **15.43%**.
* **Khả năng nhận 2 hoa:** Duy trì ổn định ở mức **70.67%**, đạt đúng chỉ tiêu thiết kế.

### 2.4. Tích hợp Backend & Khắc phục lỗi cục bộ trên Chrome

1. **Xuất mô hình hoàn thiện:**
   * Lưu mô hình Keras 11 lớp hoàn chỉnh tại `models/final/flower_singlelabel_ood11.keras` (dung lượng ~30.7 MB).
   * Tạo file cấu hình tham số OOD tại `backend/ood_config.json`.
2. **Cập nhật mã nguồn Backend:**
   * `backend/model_loader.py`: Tự động nhận diện và nạp ưu tiên mô hình 11 lớp mới.
   * `backend/predict.py`: Bổ sung bộ lọc triệt tiêu các cửa sổ rơi vào lớp Reject (`class_idx == 10`), áp dụng ngưỡng tin cậy tối ưu `gate: 0.7`.
   * `backend/app.py`: Định tuyến trực tiếp phục vụ giao diện Frontend tĩnh (`index.html`, `style.css`, `script.js`) ngay trên `http://localhost:5000`.
3. **Khắc phục triệt để lỗi chạy Web trên Chrome:**
   * Loại bỏ hoàn toàn lỗi CORS / bảo mật `file:///` khi mở file HTML cục bộ trong Chrome. Trình duyệt giờ đây truy cập trực tiếp web app tại cùng origin `http://localhost:5000`.
   * Cập nhật `frontend/script.js` tự động nhận diện domain linh hoạt giữa Flask và Live Server.
   * Tạo tiện ích khởi động 1-click `run_app.bat` tại thư mục gốc của project: tự động nạp môi trường ảo, khởi chạy Flask server và bật trình duyệt Chrome truy cập web.

---

## 3. Cấu trúc các file mới và thay đổi trong tuần

| Đường dẫn file | Trạng thái | Chức năng |
|---|---|---|
| `notebooks/07_ood_detection.ipynb` | Cập nhật | Hoàn thiện toàn bộ các cell phân tích, huấn luyện và đánh giá test OOD |
| `models/final/flower_singlelabel_ood11.keras` | Mới | Mô hình DenseNet121 hoàn chỉnh 11 lớp (10 lớp hoa ID + Reject Class) |
| `models/ood/ood_test_results_image_level.csv` | Mới | Bảng số liệu chi tiết đánh giá cấp ảnh trên tập Test độc lập |
| `models/ood/step2_win_test.pkl` | Mới | Cache đặc trưng 21 cửa sổ trượt tập Test |
| `backend/ood_config.json` | Mới | Cấu hình tham số ngưỡng gate, chỉ số lớp reject và metric test cho backend |
| `backend/model_loader.py` | Cập nhật | Nạp tự động model 11 lớp và đọc cấu hình OOD |
| `backend/predict.py` | Cập nhật | Tích hợp lọc bỏ cửa sổ reject class và áp ngưỡng gate 0.7 |
| `backend/app.py` | Cập nhật | Phục vụ trực tiếp giao diện Frontend từ Flask server, sửa lỗi CORS |
| `frontend/script.js` | Cập nhật | Nhận diện linh hoạt URL backend trên localhost |
| `run_app.bat` | Mới | File script khởi động 1-click toàn bộ hệ thống cho người dùng |

---

## 4. Đánh giá & Bài học kinh nghiệm

1. **Hiệu quả của việc đưa Reject Class vào quá trình huấn luyện:**
   * So với các kỹ thuật hậu xử lý thụ động (post-hoc score thresholding), việc huấn luyện một lớp từ chối chủ động (active rejection) trên không gian đặc trưng của mạng nơ-ron giúp mô hình học được ranh giới quyết định (decision boundary) rõ ràng hơn hẳn, giải quyết được cả trường hợp ảnh chụp góc cận cảnh chỉ thấy lá/nền đất.
2. **Tầm quan trọng của việc đánh giá ở cấp ảnh (Image-Level) thay vì cấp vùng (Crop-Level):**
   * Một phương pháp lọc OOD có thể cho kết quả ấn tượng ở cấp độ từng ảnh cắt đơn lẻ (FPR ~15–20%), nhưng khi ghép vào cơ chế quét 21 cửa sổ trượt (sliding window aggregation), chỉ cần 1 trong 21 cửa sổ bị phân loại nhầm là toàn bộ ảnh sẽ bị báo nhầm. Do đó, việc đo lường trực tiếp trên 21 cửa sổ là bước bắt buộc để phản ánh đúng năng lực hệ thống khi chạy thực tế.
3. **Phát hiện dữ liệu nhầm lẫn trong dataset gốc:**
   * Nhờ công cụ phân tích tương đồng cosine ở Cell 12c-noise, nhóm đã phát hiện loài Coltsfoot (#11 trong Oxford) thực chất cùng chi và đặc điểm hình thái với Bồ Công Anh, giúp kịp thời loại bỏ và làm sạch dữ liệu huấn luyện.

---

## 5. Kế hoạch tiếp theo (Tuần 8)

1. Tối ưu hóa thời gian suy luận (Inference Latency): Áp dụng kỹ thuật Batching tối ưu hơn khi người dùng upload ảnh kích thước lớn.
2. Thu thập thêm ảnh kiểm thử thực tế đa dạng (ảnh chụp hoa ngoài vườn, ảnh chụp trong nhà thiếu sáng) để đánh giá độ bền vững (Robustness).
3. Hoàn thiện tài liệu báo cáo tổng kết đồ án cuối kỳ và chuẩn bị Slide thuyết trình.
