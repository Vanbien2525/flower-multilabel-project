import os
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge

from predict import validate_image, detect_flowers, get_model_predictions
from model_loader import load_singlelabel_model, get_model_comparison

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
UPLOAD_DIR = os.path.join(BACKEND_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)  # cho phép frontend (chạy port khác, vd Live Server 5500) gọi API

# ---- NFR-07: giới hạn kích thước file tải lên. Chốt 8MB (trong khoảng 5-10MB SRS đề xuất) ----
MAX_FILE_SIZE_MB = 8
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


def allowed_extension(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    # EF-02: file vượt quá giới hạn kích thước -> từ chối, không gọi model
    return jsonify({"error": f"File vượt quá giới hạn {MAX_FILE_SIZE_MB}MB."}), 413


@app.route("/predict", methods=["POST"])
def predict():
    # FR-02 / EF-03: chưa chọn ảnh hoặc thiếu field 'image'
    if "image" not in request.files:
        return jsonify({"error": "Thiếu file ảnh. Vui lòng chọn một ảnh để tải lên."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Chưa chọn file ảnh nào."}), 400

    # FR-02 / EF-01: kiểm tra định dạng theo phần mở rộng trước (nhanh, rẻ)
    if not allowed_extension(file.filename):
        return jsonify({"error": "Định dạng file không được hỗ trợ. Chỉ chấp nhận JPEG hoặc PNG."}), 400

    file_bytes = file.read()

    # FR-02 / EF-01: kiểm tra thực sự là ảnh hợp lệ (không chỉ dựa vào đuôi file)
    is_valid, error_msg = validate_image(file_bytes)
    if not is_valid:
        return jsonify({"error": error_msg}), 400

    # 8.1: lưu tạm vào backend/static/uploads/, không lưu trữ lâu dài
    temp_filename = f"{uuid.uuid4().hex}_{file.filename}"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)
    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    try:
        detected_flowers, boxes_by_id, img = detect_flowers(file_bytes)
    except FileNotFoundError as e:
        # model hoặc classes.json bị thiếu -> lỗi cấu hình hệ thống
        return jsonify({"error": f"Lỗi cấu hình hệ thống: {str(e)}"}), 500
    except ValueError as e:
        # EF-05: care_guide.json thiếu dữ liệu cho một lớp được phát hiện
        return jsonify({"error": f"Lỗi dữ liệu chăm sóc: {str(e)}"}), 500
    finally:
        # dọn file tạm ngay sau khi xử lý xong, không lưu lâu dài
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # FR-08: không nhãn nào vượt ngưỡng -> danh sách rỗng kèm thông điệp
    if len(detected_flowers) == 0:
        return jsonify({
            "detected_flowers": [],
            "message": "Không nhận diện được loại hoa nào trong ảnh này, vui lòng thử ảnh khác."
        }), 200

    # Bảng phụ: 5 mô hình benchmark dự đoán bao nhiêu % cho các loại hoa vừa phát hiện được,
    # dùng đúng vùng ảnh (box) mà model chính đã dùng để nhận diện từng loại hoa
    model_predictions = get_model_predictions(img, boxes_by_id)

    return jsonify({
        "detected_flowers": detected_flowers,
        "model_predictions": model_predictions,
    }), 200


@app.route("/models/comparison", methods=["GET"])
def models_comparison():
    try:
        data = get_model_comparison()
    except FileNotFoundError as e:
        # EF-06: model_comparison.json chưa tồn tại hoặc không đọc được
        return jsonify({"error": f"Chưa có dữ liệu so sánh mô hình: {str(e)}"}), 500

    return jsonify(data), 200


@app.route("/", methods=["GET"])
def index():
    """Phục vụ file index.html cho trình duyệt khi truy cập http://localhost:5000/."""
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_file):
        return send_from_directory(FRONTEND_DIR, "index.html")
    return jsonify({"message": "VBxIris Backend đang chạy."}), 200


@app.route("/<path:filename>", methods=["GET"])
def serve_frontend_static(filename):
    """Phục vụ style.css, script.js và các tài nguyên frontend khác."""
    target = os.path.join(FRONTEND_DIR, filename)
    if os.path.exists(target) and os.path.isfile(target):
        return send_from_directory(FRONTEND_DIR, filename)
    return jsonify({"error": "Không tìm thấy tài nguyên."}), 404


if __name__ == "__main__":
    print("Đang nạp model...")
    load_singlelabel_model()  # nạp model 1 lần khi khởi động server (NFR-05)
    print("Model đã sẵn sàng. Khởi động server tại http://localhost:5000")
    print("👉 Mở trình duyệt Chrome và truy cập: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)