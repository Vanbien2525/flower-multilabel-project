# 🔹 TUẦN 3 — Xây dựng & Huấn luyện mô hình (Benchmark 5 kiến trúc)

> Lưu ý: Tuần 3 đã mở rộng theo góp ý giáo viên (xem mục 6b blueprint v2) — thay vì chỉ train 1 model (MobileNetV2), tuần này benchmark 5 kiến trúc cùng thời để chọn ra mô hình tốt nhất, đồng thời giữ lại kết quả cả 5 để hiển thị lên web.

---

## 1. Việc đã làm

1. Kiểm tra lại dữ liệu Tuần 2 (train/val/test = 4200/900/900 ảnh), xác nhận `classes.json` đúng thứ tự 10 lớp cố định
2. Tạo cấu trúc thư mục mới cho Tuần 3: `models/checkpoints/<5 kiến trúc>/`, `models/comparison/`
3. Xây dựng data pipeline (`tf.data.Dataset`) đọc từ `data/meta/full_labels.csv` (6000 dòng, gồm cả ảnh composite và augmented) — sửa lỗi ban đầu do đọc nhầm file `data/composite/labels.csv` (chỉ có 3000 dòng, thiếu phần augmented)
4. **Sửa lỗi chuẩn hóa ảnh:** bỏ bước `/255.0` cố định trong pipeline, chuyển sang dùng `preprocess_input` riêng của từng kiến trúc (đặt ngay trong model) — đảm bảo so sánh 5 model công bằng, vì mỗi kiến trúc pretrained ImageNet với chuẩn hóa khác nhau
5. Viết hàm `build_model()` tổng quát, tham số hóa theo tên kiến trúc, dùng chung cho cả 5 model:
   - MobileNetV2, MobileNetV3Large, EfficientNetB0, ResNet50V2, DenseNet121
6. Train **Giai đoạn 1 — Feature Extraction** (đóng băng base model, chỉ train Dense head) cho cả 5 model, tối đa 20 epoch, `EarlyStopping` (patience 4) + `ModelCheckpoint`
7. Train **Giai đoạn 2 — Fine-tuning** (mở khóa 30% lớp cuối base model theo tỷ lệ, learning rate giảm 10 lần còn 1e-5) cho cả 5 model, tối đa 15 epoch, `EarlyStopping` (patience 3)
8. Đánh giá cả 5 model trên tập test (460 ảnh): AUC, Precision, Recall, Hamming Accuracy, Subset Accuracy
9. Vẽ biểu đồ loss (train/val) của cả 5 model để xác minh không có dấu hiệu overfitting/hội tụ bất thường
10. Chọn model tốt nhất theo AUC, lưu vào `models/final/`, lưu bảng so sánh đầy đủ vào `models/comparison/model_comparison.json`

---

## 2. Kết quả so sánh 5 mô hình (đánh giá trên tập test)

| Model | Test AUC | Precision | Recall | Hamming Acc | Subset Acc | Epochs (S1+S2) |
|---|---|---|---|---|---|---|
| **DenseNet121 (được chọn 🏆)** | **0.9889** | 0.9468 | 0.8454 | 0.9593 | 0.6856 | 20+15 |
| EfficientNetB0 | 0.9871 | 0.9465 | 0.8305 | 0.9564 | 0.6500 | 20+15 |
| MobileNetV3Large | 0.9822 | 0.9350 | 0.8261 | 0.9534 | 0.6511 | 20+15 |
| ResNet50V2 | 0.9796 | 0.9441 | 0.8200 | 0.9540 | 0.6500 | 20+15 |
| MobileNetV2 (baseline gốc) | 0.9780 | 0.9206 | 0.8200 | 0.9496 | 0.6211 | 20+15 |

**Tiêu chí chọn model:** Test AUC cao nhất — vì AUC không phụ thuộc threshold 0.5 tạm thời, phản ánh đúng khả năng phân biệt thật của model. Threshold tối ưu riêng từng lớp sẽ được tinh chỉnh ở Tuần 4.

**Xác minh không có lỗi huấn luyện:** biểu đồ loss của cả 5 model cho thấy `val_loss` giảm liên tục đến hết epoch (không nằm ngang/tăng lên), nên việc `EarlyStopping` không kích hoạt sớm là hợp lý — không phải bug. Điểm gai nhọn nhỏ ở epoch chuyển Giai đoạn 2 (do đổi learning rate + mở khóa lớp) rồi giảm trở lại ngay là hiện tượng bình thường, cho thấy fine-tuning có tác dụng thật.

---

## 3. Deliverable

- `notebooks/03_train_model.ipynb` — chạy hoàn chỉnh, đầy đủ 2 giai đoạn cho cả 5 model
- `models/checkpoints/<5 kiến trúc>/` — weight tốt nhất + history từng giai đoạn của mỗi model
- `models/comparison/model_comparison.json` — bảng so sánh đầy đủ 5 model (dùng cho endpoint `/models/comparison` ở Tuần 4)
- `models/final/flower_multilabel_model.keras` — model được chọn (DenseNet121)
- `reports/weekly_progress/tuan3_loss_curves.png` — biểu đồ xác minh hội tụ
- `reports/weekly_progress/tuan3.md` — file này

---

## 4. Kiến thức Tuần 3 cần nắm vững (vấn đáp)

### 4.1. Transfer Learning
- **Là gì:** tận dụng lại trọng số (weights) của một mạng đã được huấn luyện trước trên tập dữ liệu lớn (ở đây là ImageNet, ~1.4 triệu ảnh, 1000 lớp), thay vì huấn luyện một mạng mới từ đầu (from scratch)
- **Vì sao dùng:** dữ liệu của đồ án (6000 ảnh) rất nhỏ so với ImageNet — nếu train from scratch, mạng CNN sâu sẽ dễ overfitting và cần rất nhiều dữ liệu + thời gian để học được các đặc trưng cơ bản (cạnh, góc, kết cấu...). Base model pretrained đã học sẵn các đặc trưng tổng quát này, ta chỉ cần "dạy thêm" phần phân loại cuối cho đúng bài toán hoa

### 4.2. Vì sao Sigmoid + Binary Crossentropy, KHÔNG dùng Softmax + Categorical Crossentropy
- Bài toán là **multi-label** (1 ảnh có thể chứa nhiều loại hoa cùng lúc, ví dụ vừa có hoa hồng vừa có hướng dương) — các lớp **không loại trừ nhau**
- Softmax bắt buộc tổng xác suất của tất cả lớp = 1 → nếu ảnh có 2 loại hoa, softmax sẽ ép 2 xác suất "chia nhau" tổng = 1, làm sai lệch bản chất bài toán (là 2 xác suất đó đáng lẽ phải độc lập, mỗi cái có thể gần 1)
- Sigmoid tính xác suất **độc lập cho từng lớp** (mỗi lớp ra 1 giá trị 0-1 riêng biệt, không ràng buộc tổng), phù hợp đúng bản chất multi-label
- Binary Crossentropy tính loss riêng cho từng lớp rồi cộng lại — khớp với cách sigmoid hoạt động, khác với Categorical Crossentropy (giả định chỉ 1 lớp đúng duy nhất)

### 4.3. Feature Extraction vs Fine-tuning
| | Feature Extraction (Giai đoạn 1) | Fine-tuning (Giai đoạn 2) |
|---|---|---|
| Base model | Đóng băng hoàn toàn (`trainable=False`) | Mở khóa một phần cuối (ở đây 30% lớp cuối) |
| Phần được học | Chỉ Dense head phía sau | Dense head + phần base đã mở khóa |
| Learning rate | Cao hơn (1e-4) | Thấp hơn nhiều (1e-5) — tránh phá vỡ trọng số ImageNet đã học tốt |
| Mục đích | Học nhanh phần phân loại mới, tận dụng đặc trưng có sẵn | Tinh chỉnh các đặc trưng cấp cao cho khớp hơn với đặc điểm riêng của hoa |

- Trên biểu đồ loss, ranh giới 2 giai đoạn thể hiện rõ bằng điểm gai nhọn nhỏ (do đổi learning rate + mở khóa lớp làm loss tăng đột ngột rồi giảm trở lại)

### 4.4. Vì sao cần benchmark nhiều kiến trúc thay vì chỉ dùng 1 model
- Không có kiến trúc nào tốt nhất cho mọi bài toán — hiệu quả phụ thuộc vào đặc điểm dữ liệu cụ thể
- Bằng chứng thực tế từ đồ án: cùng dữ liệu, cùng epoch budget, cùng quy trình 2 giai đoạn, nhưng AUC của 5 model chênh nhau từ 0.978 (MobileNetV2) đến 0.989 (DenseNet121) — chứng minh lựa chọn kiến trúc ảnh hưởng thật đến kết quả, không chỉ là lý thuyết suông
- Việc benchmark còn thể hiện tư duy thực nghiệm khoa học (thử nghiệm có kiểm soát), tránh chọn model một cách chủ quan

### 4.5. Vì sao phải cho các model cùng "epoch budget" khi so sánh
- Nếu chỉ train mỗi model vài epoch, kết quả so sánh sẽ phản ánh mô hình nào **hội tụ nhanh** trong giai đoạn đầu (thường là các model nhỏ, nông), chứ không phản ánh khả năng thật sự của kiến trúc khi được học đủ
- Giải pháp: dùng `EarlyStopping` (patience 3-5) + `ModelCheckpoint` (lưu bản tốt nhất) để mỗi model tự dừng khi val_loss không cải thiện thêm, đảm bảo mỗi kiến trúc đều được học đến gần điểm hội tụ trước khi so sánh

---

## 5. Kiến thức cần chuẩn bị trước khi bước sang Tuần 4

### 5.1. Vì sao multi-label không dùng Accuracy đơn thuần
- Bằng chứng trực tiếp từ kết quả Tuần 3: **Hamming Accuracy** (~0.95-0.96, tính trung bình đúng/sai từng nhãn riêng lẻ) và **Subset Accuracy** (~0.62-0.69, chỉ tính đúng khi TẤT CẢ 10 nhãn của 1 ảnh đều khớp) chênh lệch rất lớn
- Một con số "Accuracy" duy nhất dễ gây hiểu lầm — cần hiểu và phân biệt rõ Precision, Recall, F1-score tính riêng cho từng lớp (per-class), vì mỗi loại hoa có độ khó nhận diện và số lượng ảnh training khác nhau

### 5.2. Threshold trong multi-label
- Threshold là ngưỡng xác suất để quyết định 1 lớp "có" hay "không" xuất hiện trong ảnh (ví dụ xác suất sigmoid ≥ 0.5 → coi là "có")
- Ở Tuần 3 đang dùng threshold chung 0.5 cho tất cả 10 lớp (tạm thời)
- Vì sao nên có threshold riêng từng lớp: các lớp có ít ảnh training (như `hoa_oai_huong`, `hoa_mai`, `hoa_dao` — nhóm tự thu thập) thường có xác suất dự đoán thấp hơn hệ thống dù đúng, nên cần threshold thấp hơn 0.5; ngược lại lớp có nhiều dữ liệu có thể cần threshold cao hơn để giảm dự đoán sai (false positive)
- Tuần 4 sẽ tìm threshold tối ưu riêng từng lớp dựa trên đường cong Precision-Recall trên tập validation

### 5.3. Nhận biết Overfitting/Underfitting qua biểu đồ loss
- **Overfitting:** train loss tiếp tục giảm nhưng val loss ngừng giảm rồi tăng trở lại — model học thuộc dữ liệu train, không tổng quát hóa tốt cho dữ liệu mới
- **Underfitting:** cả train loss và val loss đều cao, không giảm được nhiều — model chưa học đủ, có thể do quá đơn giản hoặc chưa train đủ
- Từ biểu đồ Tuần 3: chưa thấy dấu hiệu overfitting rõ rệt ở cả 5 model (val loss vẫn giảm đến cuối), nhưng cần tiếp tục theo dõi khi đánh giá per-class ở Tuần 4

### 5.4. Kiến thức mới cho Tuần 4 — Flask cơ bản
- Khái niệm route (`@app.route()`), phương thức HTTP (GET/POST), xử lý `multipart/form-data` (nhận file ảnh upload)
- Cách trả về response dạng JSON (`jsonify()`), cấu trúc endpoint `/predict` và `/models/comparison` theo đúng thiết kế mục 5 blueprint
- Cách load model `.keras` đã lưu và chạy `predict()` trong môi trường Flask (khác với Colab — không có GPU mặc định, cần load model 1 lần khi khởi động server, không load lại mỗi request)
