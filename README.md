# 🌸 VBxIris — Hệ thống AI Nhận diện Hoa & Hướng dẫn Chăm sóc

> **Đồ án môn học / Dự án AI Computer Vision:** Hệ thống AI nhận diện đa loài hoa trong một khung hình kết hợp cơ chế phát hiện ảnh ngoài phân phối (Out-of-Distribution - OOD) và cẩm nang chăm sóc chi tiết.

---

## 📌 1. Giới thiệu dự án

**VBxIris** là giải pháp ứng dụng học sâu (Deep Learning / Transfer Learning) giải quyết bài toán nhận diện và hỗ trợ chăm sóc cây hoa cho người dùng thực tế:
- **Nhận diện đa loài hoa (Multi-flower Detection):** Cho phép người dùng tải lên ảnh chứa từ 1 đến 3 loài hoa khác nhau. Hệ thống sử dụng cơ chế lưới cửa sổ trượt đa tỉ lệ (21 Multi-scale Sliding Windows) kết hợp thuật toán NMS / Containment Suppression để xác định chính xác vị trí và tên từng loài.
- **Cơ chế chống nhận nhầm hoa lạ & mảng nền (OOD Detection):** Tích hợp đầu mạng phân loại 11 lớp với **Reject Class** (lớp từ chối), triệt tiêu hiện tượng báo nhầm khi người dùng chụp hoa lạ không thuộc 10 loài hoặc khi khung hình chỉ chứa đất, thảm cỏ, cành khô.
- **Cẩm nang chăm sóc chuyên sâu:** Tự động cung cấp hướng dẫn chăm sóc tương ứng theo 5 tiêu chí: ☀️ Ánh sáng, 💧 Nước tưới, 🌱 Đất trồng, 🌡️ Nhiệt độ, 📌 Lưu ý sâu bệnh & cắt tỉa.
- **Đối chuẩn 5 mô hình Transfer Learning:** Cung cấp bảng so sánh hiệu năng thực tế giữa 5 kiến trúc: `MobileNetV2`, `MobileNetV3Large`, `EfficientNetB0`, `ResNet50V2` và `DenseNet121`.

---

## 🌺 2. Danh mục 10 loài hoa hỗ trợ

| STT | Mã định danh (`id`) | Tên tiếng Việt | Tên tiếng Anh | Nguồn dữ liệu |
| :---: | :--- | :--- | :--- | :--- |
| 1 | `hoa_hong` | Hoa Hồng | Rose | Kaggle Flowers Recognition |
| 2 | `hoa_huong_duong` | Hoa Hướng Dương | Sunflower | Kaggle Flowers Recognition |
| 3 | `hoa_cuc` | Hoa Cúc | Daisy | Kaggle Flowers Recognition |
| 4 | `hoa_tulip` | Hoa Tulip | Tulip | Kaggle Flowers Recognition |
| 5 | `hoa_bo_cong_anh` | Bồ Công Anh | Dandelion | Kaggle Flowers Recognition |
| 6 | `hoa_sen` | Hoa Sen | Lotus | Oxford 102 / Crawl thực tế |
| 7 | `hoa_lan` | Hoa Lan | Orchid | Oxford 102 Flowers |
| 8 | `hoa_mai` | Hoa Mai | Apricot Blossom | Dữ liệu đặc trưng Việt Nam |
| 9 | `hoa_dao` | Hoa Đào | Peach Blossom | Dữ liệu đặc trưng Việt Nam |
| 10 | `hoa_oai_huong` | Oải Hương | Lavender | Thu thập & gán nhãn thực tế |

---

## 🏗️ 3. Cấu trúc thư mục dự án

```text
flower-multilabel-project/
├── backend/                       # Máy chủ Flask API
│   ├── app.py                     # Khởi tạo Server, CORS, định tuyến API & phục vụ Web
│   ├── model_loader.py            # Nạp và cache mô hình AI, cấu hình OOD
│   ├── predict.py                 # Pipeline trượt cửa sổ, lọc NMS & suy luận
│   ├── care_guide.json            # CSDL hướng dẫn chăm sóc 10 loài hoa
│   ├── ood_config.json            # Cấu hình tham số mô hình OOD (ngưỡng gate, reject class)
│   ├── model_comparison.json      # Bảng số liệu Benchmark 5 mô hình
│   └── requirements.txt           # Thư viện Python yêu cầu
│
├── frontend/                      # Giao diện Web SPA
│   ├── index.html                 # Giao diện tải ảnh, kết quả Tabs, bảng Benchmark
│   ├── style.css                  # Thiết kế Glassmorphism hiện đại, responsive
│   └── script.js                  # Xử lý tương tác client, gọi API Backend
│
├── notebooks/                     # Mã nguồn thực nghiệm trên Google Colab
│   ├── 01_data_exploration.ipynb          # Khám phá và thống kê phân phối dữ liệu
│   ├── 02_data_composite_augment.ipynb    # Ghép ảnh đa loài hoa & Data Augmentation
│   ├── 03_train_model.ipynb               # Huấn luyện đa nhãn (sigmoid) ban đầu
│   ├── 04_evaluate_model.ipynb            # Đánh giá ma trận nhầm lẫn & quét threshold
│   ├── 05_train_singlelabel_model.ipynb   # Huấn luyện mô hình đơn nhãn DenseNet121
│   ├── 06_benchmark_singlelabel.ipynb     # Benchmark 5 kiến trúc Transfer Learning
│   └── 07_ood_detection.ipynb            # Nghiên cứu & hoàn thiện OOD Detection
│
├── reports/                       # Báo cáo tiến độ đồ án
│   └── weekly_progress/
│       ├── flower_project_blueprint.md    # Tài liệu kiến trúc chuẩn của dự án
│       ├── tuan1.md - tuan7.md            # Báo cáo chi tiết từ Tuần 1 đến Tuần 7
│       └── *.png, *.csv                   # Biểu đồ và dữ liệu đánh giá thực nghiệm
│
├── run_app.bat                    # Script khởi động 1-click cả Backend & Web trên Windows
└── README.md                      # Tài liệu dự án
```

---

## 🚀 4. Hướng dẫn cài đặt & Khởi chạy

### Cách 1: Khởi động 1-click (Dành cho Windows)
Chỉ cần **click đúp vào file `run_app.bat`** tại thư mục gốc:
* Script sẽ tự động kích hoạt môi trường ảo Python.
* Khởi chạy Flask Server tại `http://localhost:5000`.
* Tự động bật trình duyệt Google Chrome truy cập ứng dụng.

### Cách 2: Khởi chạy thủ công từ Terminal

1. **Clone repository:**
   ```bash
   git clone https://github.com/Vanbien2525/flower-multilabel-project.git
   cd flower-multilabel-project
   ```

2. **Cài đặt môi trường:**
   ```bash
   cd backend
   python -m venv venv
   # Trên Windows:
   venv\Scripts\activate
   # Cài đặt thư viện:
   pip install -r requirements.txt
   ```

3. **Chạy máy chủ Backend:**
   ```bash
   python app.py
   ```

4. **Truy cập ứng dụng:**
   Mở trình duyệt web và truy cập: `http://localhost:5000`

---

## 📊 5. Kết quả thực nghiệm nổi bật (Test Set)

* **Hiệu năng phân loại:** Mô hình **DenseNet121** đạt F1-macro **94.73%**, Test Accuracy **93.75%**, AUC **0.997**.
* **Độ chính xác nhận diện hoa thật:** Đạt **96.14%** ở cấp ảnh (21 cửa sổ trượt).
* **Khả năng chống nhận nhầm ngoài phân phối (OOD):**
  * Tỷ lệ lọt hoa lạ (`near_flower` FPR): Giảm mạnh từ 88.98% xuống còn **28.39%** (chặn được hơn 71.6% hoa lạ).
  * Tỷ lệ lọt mảng nền (`bg_crop` FPR): Giảm từ 60.00% xuống còn **11.43%**.
  * Tỷ lệ lọt vật thể không phải hoa trung bình: Chỉ còn **15.43%**.

---

## 📜 6. Giấy phép (License)
Dự án được phát triển phục vụ mục đích học tập và nghiên cứu mã nguồn mở.
