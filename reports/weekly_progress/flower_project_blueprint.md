# 🌸 BLUEPRINT CỐ ĐỊNH — ĐỒ ÁN AI NHẬN DIỆN & CHĂM SÓC HOA
> File này là khung tham chiếu DUY NHẤT cho toàn bộ đồ án. Không thay đổi cấu trúc giữa chừng.
> Đầu mỗi buổi làm việc với Claude, upload lại file này để tiếp tục đúng mạch.
>
> **📌 v2 — cập nhật sau góp ý giáo viên (sau Tuần 2):** bổ sung SRS, workflow diagram, và yêu cầu train-so sánh 5 mô hình. Xem mục 6b và mục 8 để biết chi tiết thay đổi so với bản gốc.

---

## 1. TỔNG QUAN ĐỒ ÁN

| Mục | Nội dung |
|---|---|
| Tên đồ án | Hệ thống AI nhận diện đa nhãn loại hoa và gợi ý chăm sóc |
| Bài toán AI | Multi-label Image Classification (Computer Vision) |
| Số lớp (classes) | 10 loại hoa |
| Input | Ảnh upload (có thể chứa 1-3 loại hoa) |
| Output | Danh sách loại hoa phát hiện được + bảng hướng dẫn chăm sóc từng loại + bảng so sánh 5 mô hình đã thử |
| Trình độ | Mới bắt đầu code Deep Learning |
| Stack | Python, TensorFlow/Keras, Google Colab, Flask, HTML/CSS/JS |
| Kiến trúc mô hình chính thức (deploy) | Transfer Learning — mô hình tốt nhất trong 5 mô hình được benchmark ở Tuần 3 (xem mục 6b) |
| Activation lớp cuối | Sigmoid (KHÔNG dùng Softmax — vì multi-label) |
| Loss function | Binary Crossentropy |

---

## 2. DANH SÁCH 10 LOẠI HOA (CỐ ĐỊNH)

| ID (dùng trong code) | Tên hiển thị | Nguồn dữ liệu gợi ý |
|---|---|---|
| `hoa_hong` | Hoa Hồng (Rose) | Kaggle "Flowers Recognition" |
| `hoa_huong_duong` | Hoa Hướng Dương (Sunflower) | Kaggle "Flowers Recognition" |
| `hoa_cuc` | Hoa Cúc (Daisy) | Kaggle "Flowers Recognition" |
| `hoa_tulip` | Hoa Tulip | Kaggle "Flowers Recognition" |
| `hoa_bo_cong_anh` | Bồ Công Anh (Dandelion) | Kaggle "Flowers Recognition" |
| `hoa_sen` | Hoa Sen (Lotus) | Oxford 102 Flowers / tự thu thập Google Images |
| `hoa_lan` | Hoa Lan (Orchid) | Oxford 102 Flowers |
| `hoa_mai` | Hoa Mai (Apricot Blossom) | Tự thu thập (đặc trưng VN, cần crawl riêng) |
| `hoa_dao` | Hoa Đào (Peach Blossom) | Tự thu thập (đặc trưng VN, cần crawl riêng) |
| `hoa_oai_huong` | Oải Hương (Lavender) | Tự thu thập (không có sẵn trong Oxford 102 — đã đổi nguồn thực tế ở Tuần 1) |

> ⚠️ Lưu ý quan trọng để vấn đáp: 5 loại đầu có dataset công khai sẵn (dễ), 5 loại sau cần ghép từ nhiều nguồn — đây là điểm cộng thể hiện công sức thu thập dữ liệu, nên chuẩn bị sẵn câu trả lời "vì sao chọn dataset này/nguồn này".

---

## 3. CẤU TRÚC THƯ MỤC CỐ ĐỊNH (v2 — đã cập nhật)

```
flower-multilabel-project/
├── data/
│   ├── raw/                          # Ảnh gốc, single-label, theo từng lớp
│   │   ├── hoa_hong/
│   │   ├── hoa_huong_duong/
│   │   ├── hoa_cuc/
│   │   ├── hoa_tulip/
│   │   ├── hoa_bo_cong_anh/
│   │   ├── hoa_sen/
│   │   ├── hoa_lan/
│   │   ├── hoa_mai/
│   │   ├── hoa_dao/
│   │   └── hoa_oai_huong/
│   ├── composite/                    # Ảnh ghép multi-label tự tạo (Tuần 2)
│   │   ├── images/
│   │   └── labels.csv                # cột: filename, hoa_hong, hoa_huong_duong, ... (multi-hot 0/1)
│   ├── augmented/                    # Ảnh sau augmentation
│   ├── splits/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── metadata/
│       └── classes.json              # danh sách 10 class + tên hiển thị, index cố định
│
├── notebooks/                        # Toàn bộ code chạy trên Google Colab
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_composite_augment.ipynb
│   ├── 03_train_model.ipynb          # v2: giờ train + benchmark 5 kiến trúc, không chỉ 1
│   └── 04_evaluate_model.ipynb
│
├── models/
│   ├── checkpoints/                  # v2: tách theo từng kiến trúc
│   │   ├── mobilenetv2/
│   │   ├── mobilenetv3large/
│   │   ├── efficientnetb0/
│   │   ├── resnet50v2/
│   │   └── densenet121/
│   ├── comparison/                   # v2 — MỚI
│   │   └── model_comparison.json     # accuracy/AUC của cả 5 model, dùng để hiển thị lên web
│   └── final/
│       └── flower_multilabel_model.h5   # chỉ giữ lại model THẮNG CUỘC trong 5 model
│
├── backend/                          # Flask API (Tuần 4)
│   ├── app.py                        # v2: thêm endpoint GET /models/comparison
│   ├── model_loader.py
│   ├── predict.py
│   ├── care_guide.json               # dữ liệu hướng dẫn chăm sóc 10 loại hoa
│   ├── model_comparison.json         # v2 — MỚI, copy từ models/comparison/
│   ├── requirements.txt
│   └── static/uploads/               # ảnh user upload tạm thời
│
├── frontend/                         # Giao diện web (Tuần 5)
│   ├── index.html                    # v2: thêm khu vực hiển thị bảng so sánh 5 model
│   ├── style.css
│   └── script.js
│
├── docs/
│   ├── de_cuong.docx
│   ├── srs.docx                      # v2 — MỚI: tài liệu đặc tả yêu cầu (SRS)
│   ├── workflow_diagrams/            # v2 — MỚI
│   │   ├── functional_flow.png       # luồng chức năng
│   │   └── nonfunctional_flow.png    # luồng phi chức năng
│   ├── bao_cao_cuoi_ky.docx
│   └── slide_thuyet_trinh.pptx
│
├── reports/weekly_progress/          # Báo cáo tiến độ nộp từng tuần
│   ├── tuan1.md
│   ├── tuan2.md
│   ├── tuan3.md
│   ├── tuan4.md
│   └── tuan5.md
│
└── README.md
```

**Quy tắc đặt tên cố định:**
- ID lớp hoa: `snake_case` tiếng Việt không dấu (vd `hoa_hong`), dùng xuyên suốt code, JSON, tên thư mục.
- Tên hiển thị (UI/báo cáo): tiếng Việt có dấu, viết hoa chữ đầu (vd "Hoa Hồng").
- Thứ tự 10 class trong `classes.json` = thứ tự cố định của vector multi-hot, KHÔNG được đổi thứ tự sau khi train.
- ID mô hình trong so sánh: `snake_case` không dấu, đúng tên kiến trúc (vd `mobilenetv2`, `efficientnetb0`).

> 💾 Ghi chú dung lượng: không bắt buộc giữ nguyên cả 5 file `.h5` đầy đủ mãi mãi. Chỉ cần giữ `model_comparison.json` (kết quả) + model thắng cuộc ở `models/final/`. 4 checkpoint còn lại có thể xóa sau khi lấy xong metric, hoặc giữ đến cuối đồ án rồi dọn.

---

## 4. KIẾN TRÚC MÔ HÌNH

### 4a. Kiến trúc chung áp dụng cho cả 5 mô hình (đầu/tail giống nhau, chỉ đổi base model)

```
Input (224x224x3)
   → [BASE MODEL] (include_top=False, weights='imagenet', frozen ban đầu)
   → GlobalAveragePooling2D
   → Dense(256, activation='relu')
   → Dropout(0.3)
   → Dense(10, activation='sigmoid')   # 10 = số class, sigmoid vì multi-label

Loss: binary_crossentropy
Optimizer: Adam (lr=1e-4 giai đoạn feature extraction, lr=1e-5 giai đoạn fine-tuning)
Metrics: AUC, Precision, Recall (per-class threshold sau đánh giá Tuần 4)
```

### 4b. 5 mô hình BASE cần benchmark (Tuần 3) — CỐ ĐỊNH

| Model | Năm | Ghi chú |
|---|---|---|
| MobileNetV2 | 2018 | baseline ban đầu của đồ án |
| MobileNetV3Large | 2019 | bản nâng cấp của MobileNetV2 |
| EfficientNetB0 | 2019 | cân bằng tốt giữa accuracy và tốc độ |
| ResNet50V2 | 2016/2018 | kiến trúc kinh điển, residual connection |
| DenseNet121 | 2017 | dense connection, thường cho accuracy cao |

Cả 5 đều có sẵn trong `tensorflow.keras.applications`, đều pretrained ImageNet, đều đủ nhẹ để train trên Colab free tier.

---

## 5. THIẾT KẾ API BACKEND (v2 — đã cập nhật)

```
POST /predict
Input: multipart/form-data, field "image"
Output JSON:
{
  "detected_flowers": [
    {
      "id": "hoa_hong",
      "name": "Hoa Hồng",
      "confidence": 0.91,
      "care_guide": {
        "anh_sang": "...",
        "nuoc_tuoi": "...",
        "dat_trong": "...",
        "nhiet_do": "...",
        "luu_y": "..."
      }
    }
  ]
}
```

```
GET /models/comparison          # v2 — MỚI
Output JSON:
{
  "models": [
    { "id": "mobilenetv2", "name": "MobileNetV2", "val_accuracy": 0.88, "val_auc": 0.94, "is_selected": false },
    { "id": "efficientnetb0", "name": "EfficientNetB0", "val_accuracy": 0.91, "val_auc": 0.96, "is_selected": true },
    ...
  ]
}
```

---

## 6. LỘ TRÌNH CHI TIẾT 5 TUẦN

### 🔹 TUẦN 1 — Nền tảng + Đề cương + Thu thập dữ liệu ✅ ĐÃ XONG
**Việc làm:**
1. Chốt danh sách 10 loại hoa (mục 2) — không đổi sau tuần này
2. Tải dataset Kaggle "Flowers Recognition" (5 lớp có sẵn)
3. Thu thập thêm ảnh cho 5 lớp còn lại (Oxford 102 + crawl thủ công + thu thập tay cho hoa_oai_huong)
4. Tạo cấu trúc thư mục `data/raw/` đúng khung mục 3
5. Viết đề cương: mục tiêu, phạm vi, phương pháp, công cụ
6. Setup Google Colab, làm quen numpy, matplotlib, PIL

**Deliverable đã nộp:** `reports/weekly_progress/tuan1.md` + đề cương + 5,015 ảnh raw đã sắp xếp đúng thư mục (imbalance đã ghi nhận, xử lý ở Tuần 4)

---

### 🔹 TUẦN 2 — Xử lý & Xây dựng bộ dữ liệu Multi-label ✅ ĐÃ XONG
**Việc làm:**
1. Viết script ghép ảnh (compositing): chọn ngẫu nhiên 1-3 ảnh từ các lớp khác nhau, ghép thành 1 ảnh
2. Gán nhãn multi-hot vào `data/composite/labels.csv`
3. Data augmentation: xoay, lật, đổi độ sáng, zoom
4. Chia tập `train/val/test` theo tỉ lệ 70/15/15, lưu vào `data/splits/`

**Deliverable đã nộp:** 3,000 ảnh composite + 3,000 ảnh augmented (6,000 tổng) + split 4,200/900/900 + `tuan2.md`

---

### 🔹 TUẦN 3 — Xây dựng & Huấn luyện mô hình (v2 — ĐÃ MỞ RỘNG, bắt đầu thay đổi)
**Việc làm:**
1. Viết 1 pipeline train tổng quát, nhận vào base model làm tham số (thay vì hardcode 1 kiến trúc)
2. Áp dụng đúng quy trình 2 giai đoạn cho **cả 5 kiến trúc** ở mục 4b:
   - Giai đoạn 1: Feature extraction (đóng băng base model, chỉ train lớp Dense)
   - Giai đoạn 2: Fine-tuning (mở khóa vài lớp cuối base model, train tiếp với lr nhỏ hơn)
3. Dùng cùng "epoch budget" cho cả 5 model để so sánh công bằng — dùng `EarlyStopping` (monitor `val_loss`, patience 3-5) + `ModelCheckpoint` để mỗi model tự dừng khi hội tụ, tránh so sánh lệch do train chưa đủ hoặc train quá tay
4. Theo dõi loss/accuracy qua biểu đồ cho từng model, lưu checkpoint tốt nhất của mỗi kiến trúc vào `models/checkpoints/<ten_model>/`
5. Tổng hợp kết quả cả 5 model (val accuracy, val AUC) vào `models/comparison/model_comparison.json`
6. Chọn model có kết quả tốt nhất → copy vào `models/final/flower_multilabel_model.h5`

**Deliverable:** `notebooks/03_train_model.ipynb` chạy hoàn chỉnh cho cả 5 model + `model_comparison.json` + model thắng cuộc + `tuan3.md`

**Kiến thức vấn đáp:**
- Transfer Learning là gì, vì sao dùng thay vì train from scratch
- **Vì sao multi-label dùng sigmoid + binary crossentropy, KHÔNG dùng softmax + categorical crossentropy** (vì các lớp không loại trừ nhau — 1 ảnh có thể vừa là hoa hồng vừa là hướng dương, softmax buộc tổng xác suất = 1 nên sẽ triệt tiêu lẫn nhau)
- Fine-tuning vs Feature extraction khác nhau ra sao
- **[MỚI]** Vì sao cần benchmark nhiều kiến trúc thay vì chỉ dùng 1 model — thể hiện tư duy thực nghiệm, tránh chọn model chủ quan
- **[MỚI]** Vì sao phải cho các model cùng "epoch budget"/dùng EarlyStopping khi so sánh — nếu train không đủ epoch, model hội tụ chậm hơn (vd ResNet, DenseNet) sẽ bị đánh giá thấp một cách không công bằng so với model hội tụ nhanh (MobileNet)

---

### 🔹 TUẦN 4 — Đánh giá mô hình + Xây dựng Backend (v2 — điều chỉnh nhẹ)
**Việc làm:**
1. Đánh giá **model thắng cuộc** (không phải cả 5) trên tập test: Precision, Recall, F1-score theo từng lớp (per-class)
2. Chọn threshold tối ưu riêng cho từng loại hoa (không dùng chung 0.5 cho tất cả)
3. Xuất model cuối cùng vào `models/final/flower_multilabel_model.h5`
4. Xây `backend/app.py` (Flask) — endpoint `/predict` theo thiết kế mục 5
5. **[MỚI]** Thêm endpoint `GET /models/comparison` trả về nội dung `model_comparison.json`
6. Tạo `backend/care_guide.json` chứa hướng dẫn chăm sóc 10 loại hoa

**Deliverable:** Backend chạy được cục bộ, test bằng Postman/curl (bao gồm cả endpoint so sánh model) + `tuan4.md`

**Kiến thức vấn đáp:**
- Vì sao multi-label không dùng Accuracy đơn thuần, cần Precision/Recall/F1 riêng từng lớp
- Threshold trong multi-label nghĩa là gì, vì sao mỗi lớp có thể có threshold khác nhau
- Nhận biết Overfitting/Underfitting qua biểu đồ loss (train giảm, val tăng = overfitting)

---

### 🔹 TUẦN 5 — Frontend + Tích hợp + Báo cáo cuối kỳ (v2 — điều chỉnh nhẹ)
**Việc làm:**
1. Xây `frontend/index.html` + `style.css` + `script.js`: upload ảnh → gọi API `/predict` → hiển thị kết quả + bảng chăm sóc
2. **[MỚI]** Thêm khu vực/trang hiển thị bảng "So sánh 5 mô hình" (đọc từ `GET /models/comparison`)
3. Test toàn hệ thống end-to-end (upload ảnh thật có nhiều hoa)
4. Viết báo cáo cuối kỳ (`docs/bao_cao_cuoi_ky.docx`) + slide thuyết trình — **bao gồm SRS, workflow diagram, và bảng so sánh 5 model như điểm nhấn**
5. Ôn tập toàn bộ câu hỏi vấn đáp từ Tuần 1-4

**Deliverable:** Sản phẩm hoàn chỉnh chạy end-to-end + báo cáo (kèm SRS + workflow) + slide

**Kiến thức vấn đáp:**
- Luồng hoạt động toàn hệ thống: Client (upload ảnh) → Server Flask → Model dự đoán → Trả JSON → Frontend hiển thị
- Giải thích được TỪNG lựa chọn kỹ thuật đã dùng xuyên suốt đồ án (vì sao các base model đã chọn, vì sao sigmoid, vì sao threshold riêng từng lớp, vì sao chia thư mục như vậy, vì sao benchmark 5 model...)

---

## 6b. TÓM TẮT THAY ĐỔI SO VỚI BẢN GỐC (v1 → v2)

> Ghi lại ở đây để không đi lạc hướng ở các buổi sau. Nguồn gốc: góp ý của giáo viên sau khi báo cáo tiến độ Tuần 2.

1. **Bổ sung tài liệu (không ảnh hưởng code/pipeline):**
   - Tài liệu đặc tả yêu cầu (SRS) → `docs/srs.docx`
   - Sơ đồ luồng chức năng + luồng phi chức năng (workflow diagram) → `docs/workflow_diagrams/`
   - Có thể làm song song bất cứ lúc nào (không phụ thuộc tiến độ code), hoàn thiện bản cuối ở Tuần 5

2. **Bổ sung vào pipeline kỹ thuật (bắt đầu từ Tuần 3):**
   - Không chỉ train 1 model (MobileNetV2) mà train + benchmark **5 kiến trúc cùng thời**: MobileNetV2, MobileNetV3Large, EfficientNetB0, ResNet50V2, DenseNet121
   - Mỗi model train đủ epoch (dùng EarlyStopping để tự hội tụ, đảm bảo so sánh công bằng)
   - Chọn model có kết quả tốt nhất làm model chính thức để deploy
   - Giữ lại kết quả (accuracy/AUC) của cả 5 model, hiển thị công khai trên web qua endpoint `GET /models/comparison`

3. **Thư mục mới:** `models/comparison/`, `backend/model_comparison.json`, `docs/srs.docx`, `docs/workflow_diagrams/`

4. **API mới:** `GET /models/comparison`

5. **Điểm bắt đầu áp dụng:** Tuần 3 (huấn luyện model) là mốc thay đổi kỹ thuật đầu tiên. Tuần 1–2 giữ nguyên, không cần làm lại.

---

## 7. QUY TẮC LÀM VIỆC CÙNG CLAUDE (GHI NHỚ)
- Luôn bám đúng cấu trúc thư mục ở mục 3 (bản v2) — không tạo thêm/bớt thư mục tùy tiện
- Luôn dùng đúng 10 ID class ở mục 2 — không đổi tên giữa chừng
- Luôn dùng đúng 5 ID model ở mục 4b khi làm phần benchmark — không đổi kiến trúc giữa chừng trừ khi có lý do kỹ thuật rõ ràng
- Mỗi tuần làm xong phải có file `tuan{N}.md` trong `reports/weekly_progress/`
- Đầu phiên chat mới: upload lại file `flower_project_blueprint.md` (bản v2) này để Claude bám đúng khung
- Nếu có thay đổi/góp ý mới từ giáo viên ở các tuần sau, luôn cập nhật lại mục 6b + toàn bộ file này ngay, không chỉ nói miệng
