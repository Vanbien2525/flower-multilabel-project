# Báo cáo Tuần 2 — Xử lý & Xây dựng bộ dữ liệu Multi-label

**Đồ án:** Hệ thống AI nhận diện đa nhãn loại hoa và gợi ý chăm sóc
**Tuần:** 2/5 — Xử lý & Xây dựng bộ dữ liệu Multi-label

---

## 1. Công việc đã hoàn thành

### 1.1. Tạo `classes.json`
Cố định thứ tự 10 lớp hoa (index 0-9), làm chuẩn xuyên suốt cho vector multi-hot trong toàn bộ dự án (không đổi sau khi train theo đúng blueprint mục 3):

| Index | ID | Tên hiển thị |
|---|---|---|
| 0 | hoa_hong | Hoa Hồng |
| 1 | hoa_huong_duong | Hoa Hướng Dương |
| 2 | hoa_cuc | Hoa Cúc |
| 3 | hoa_tulip | Hoa Tulip |
| 4 | hoa_bo_cong_anh | Bồ Công Anh |
| 5 | hoa_sen | Hoa Sen |
| 6 | hoa_lan | Hoa Lan |
| 7 | hoa_mai | Hoa Mai |
| 8 | hoa_dao | Hoa Đào |
| 9 | hoa_oai_huong | Oải Hương |

### 1.2. Ghép ảnh composite (`data/composite/`)
- Viết script ghép ngẫu nhiên 1-3 ảnh từ các lớp khác nhau vào 1 canvas 224x224 (bố cục thay đổi theo số lượng hoa: full canvas / trái-phải / trên-dưới-dưới).
- Tạo **3000 ảnh composite**, gán nhãn multi-hot tương ứng lưu vào `data/composite/labels.csv`.
- Phát hiện 1 ảnh lỗi (`composite_02597.jpg`) do ảnh nguồn `hoa_hong_0043.jpg` đã bị xóa thủ công trước đó (ảnh rác), khiến 1 vùng trong composite bị trống. Đã viết hàm kiểm tra độ lệch chuẩn pixel theo từng vùng dán (std < 2.0 → vùng trống bất thường) để phát hiện chính xác, loại bỏ ảnh lỗi và tạo bù 1 ảnh thay thế để giữ đủ 3000 ảnh sạch.

### 1.3. Data augmentation (`data/augmented/`)
Áp dụng các phép biến đổi lên toàn bộ ảnh composite để tạo thêm biến thể:
- Lật ngang (xác suất 50%)
- Xoay nhẹ (-15° đến 15°)
- Điều chỉnh độ sáng (0.7x - 1.3x) và tương phản (0.8x - 1.2x)
- Zoom nhẹ (crop + resize, xác suất 50%)

Kết quả: **3000 ảnh augmented**, nhãn giữ nguyên từ ảnh gốc (augmentation không làm thay đổi loại hoa xuất hiện trong ảnh). Tổng dataset sau bước này: **6000 ảnh**.

### 1.4. Chia train/val/test (`data/splits/`)
Chia ngẫu nhiên theo tỉ lệ 70/15/15 (dùng `train_test_split` 2 lần do bài toán multi-label không stratify được theo cách thông thường), copy ảnh vật lý vào đúng cấu trúc thư mục và lưu `labels.csv` riêng cho mỗi tập.

| Tập | Số ảnh | Tỉ lệ |
|---|---|---|
| Train | 4200 | 70.0% |
| Val | 900 | 15.0% |
| Test | 900 | 15.0% |
| **Tổng** | **6000** | **100%** |

---

## 2. Kiểm tra chất lượng dữ liệu

### Phân bố nhãn theo từng lớp trong bộ composite gốc (3000 ảnh)
Tương đối đồng đều, dao động 573-646 lần xuất hiện/lớp — không bị lệch theo nhóm dữ liệu (dù dữ liệu raw mất cân bằng giữa nhóm Kaggle và nhóm tự thu thập).

### Phân bố nhãn giữa 3 tập train/val/test (sau augmentation + split)

| Lớp | train_% | val_% | test_% |
|---|---|---|---|
| hoa_hong | 9.7 | 8.8 | 9.8 |
| hoa_huong_duong | 10.7 | 9.7 | 11.5 |
| hoa_cuc | 10.1 | 10.5 | 8.9 |
| hoa_tulip | 10.0 | 10.3 | 10.0 |
| hoa_bo_cong_anh | 9.5 | 10.8 | 10.6 |
| hoa_sen | 9.7 | 9.8 | 8.8 |
| hoa_lan | 10.5 | 10.4 | 10.3 |
| hoa_mai | 10.5 | 9.7 | 11.0 |
| hoa_dao | 9.4 | 9.7 | 9.6 |
| hoa_oai_huong | 10.1 | 10.3 | 9.5 |

→ Tất cả các lớp đều dao động trong khoảng 8.8%-11.5% ở cả 3 tập — không có lớp nào bị lệch nghiêm trọng giữa train/val/test.

---

## 3. Vấn đề gặp phải & cách xử lý

| Vấn đề | Nguyên nhân | Cách xử lý |
|---|---|---|
| 1 ảnh composite bị vùng trống | Ảnh nguồn `hoa_hong_0043.jpg` bị xóa thủ công sau khi danh sách ảnh đã load vào bộ nhớ (danh sách lỗi thời so với thư mục thực tế) | Viết hàm phát hiện vùng trống dựa trên std pixel theo từng vùng dán, xác nhận trực quan bằng mắt, loại bỏ ảnh lỗi và tạo bù 1 ảnh thay thế |
| Dữ liệu raw mất cân bằng | 5 lớp Kaggle có 700-1000+ ảnh/lớp, 5 lớp tự thu thập chỉ có 100-167 ảnh/lớp | Không ảnh hưởng đến bước ghép composite vì chọn lớp ngẫu nhiên đều (không theo số lượng ảnh gốc); cần lưu ý khi đánh giá per-class ở Tuần 4 |

---

## 4. Kiến thức nắm được (chuẩn bị vấn đáp)

- **Multi-hot encoding**: khác one-hot ở chỗ cho phép nhiều vị trí "1" đồng thời trong 1 vector (vd `[1,0,1,0,...]`), vì các lớp hoa không loại trừ lẫn nhau — 1 ảnh có thể chứa nhiều loại hoa cùng lúc.
- **Data augmentation**: tạo biến thể ảnh (xoay, lật, đổi sáng, zoom...) từ ảnh gốc để chống overfitting, buộc model học đặc trưng bản chất của từng loại hoa (hình dạng cánh, màu sắc, texture) thay vì ghi nhớ pixel cụ thể của từng ảnh.
- **Train/val/test split**: train dùng để model học trực tiếp; val dùng để theo dõi và điều chỉnh trong lúc train (không cho model học trực tiếp từ tập này); test chỉ dùng để đánh giá cuối cùng, không đụng đến cho tới khi có model hoàn chỉnh — tránh việc "lộ đề" khiến đánh giá không khách quan.

---

## 5. Tiếp theo (Tuần 3)

- Xây dựng model theo kiến trúc cố định: MobileNetV2 (frozen) → GlobalAveragePooling2D → Dense(256, relu) → Dropout(0.3) → Dense(10, sigmoid)
- Giai đoạn 1: Feature extraction (đóng băng MobileNetV2, chỉ train lớp Dense)
- Giai đoạn 2: Fine-tuning (mở khóa vài lớp cuối MobileNetV2, train tiếp với learning rate nhỏ hơn)
- Theo dõi loss/accuracy qua biểu đồ, lưu checkpoint tốt nhất
