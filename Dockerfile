# ==============================================================================
# Dockerfile cho Flower Multi-label Classification & OOD Detection
# Tối ưu hóa cho Google Cloud Run & Docker Container
# ==============================================================================

# Sử dụng Python 3.11 slim để tối ưu kích thước image
FROM python:3.11-slim

# Thiết lập biến môi trường
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

# Cài đặt các gói hệ thống cơ bản
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục gốc trong container
WORKDIR /app

# Copy file requirements và cài đặt các thư viện Python
COPY backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy các thư mục và file cần thiết vào container
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/
COPY data/metadata/ /app/data/metadata/
COPY models/final/flower_singlelabel_ood11.keras /app/models/final/flower_singlelabel_ood11.keras

# Chuyển thư mục làm việc vào backend để chạy ứng dụng
WORKDIR /app/backend

# Cổng mặc định của Cloud Run
EXPOSE 8080

# Chạy server với Gunicorn (1 worker, 4 threads để tối ưu RAM cho TensorFlow)
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 0 app:app
