# Báo cáo tiến độ - Tuần 1

**Đồ án:** Hệ thống AI nhận diện đa nhãn loại hoa và gợi ý chăm sóc
**Giai đoạn:** Nền tảng + Đề cương + Thu thập dữ liệu

## 1. Công việc đã hoàn thành

- [x] Chốt danh sách 10 loại hoa cố định (mục 2 blueprint)
- [x] Setup Google Colab, kết nối Google Drive, làm quen numpy/matplotlib/PIL
- [x] Tạo cấu trúc thư mục dự án theo đúng khung cố định (mục 3 blueprint)
- [x] Tải dataset Kaggle "Flowers Recognition" cho 5 lớp có sẵn
- [x] Thu thập dữ liệu Oxford 102 Flowers cho hoa sen, hoa lan
- [x] Thu thập thủ công hoa mai, hoa đào, hoa oải hương (không có sẵn dataset công khai phù hợp)

## 2. Số liệu dữ liệu thu thập được

| Lớp hoa | Số ảnh | Nguồn |
|---|---|---|
| hoa_hong | 784 | Kaggle Flowers Recognition |
| hoa_huong_duong | 733 | Kaggle Flowers Recognition |
| hoa_cuc | 764 | Kaggle Flowers Recognition |
| hoa_tulip | 984 | Kaggle Flowers Recognition |
| hoa_bo_cong_anh | 1052 | Kaggle Flowers Recognition |
| hoa_sen | 137 | Oxford 102 Flowers |
| hoa_lan | 100 | Oxford 102 Flowers |
| hoa_mai | 159 | Thu thập thủ công |
| hoa_dao | 168 | Thu thập thủ công |
| hoa_oai_huong | 134 | Thu thập thủ công |
| **Tổng cộng** | **5,015** | |

## 3. Khó khăn gặp phải & cách xử lý

- **Hoa mai, hoa đào:** không có trong Kaggle "Flowers Recognition" hay Oxford 102 Flowers vì là đặc trưng Việt Nam. Đã thử crawl tự động (bing-image-downloader, icrawler) nhưng kết quả toàn ảnh rác/lỗi → chuyển sang thu thập thủ công có kiểm soát chất lượng.
- **Hoa oải hương:** ban đầu dự kiến lấy từ Oxford 102 Flowers theo blueprint, nhưng sau khi kiểm tra danh sách 102 loài chính thức thì không có lớp "lavender" trong dataset này → chuyển sang thu thập thủ công như hoa mai, hoa đào.
- Số lượng ảnh giữa các lớp không đồng đều (dao động từ 100 đến hơn 1000 ảnh/lớp) — cần lưu ý khi huấn luyện mô hình ở Tuần 3 (có thể ảnh hưởng đến class imbalance).

## 4. Kế hoạch tuần tiếp theo (Tuần 2)

- Viết script ghép ảnh composite (1-3 hoa/ảnh) để tạo bộ dữ liệu multi-label
- Gán nhãn multi-hot vào `data/composite/labels.csv`
- Thực hiện data augmentation (xoay, lật, đổi độ sáng, zoom)
- Chia tập train/val/test theo tỉ lệ 70/15/15
