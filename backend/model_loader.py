import os
import json
import threading
import tensorflow as tf

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

SINGLELABEL_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "final", "flower_singlelabel_model.keras")
OOD_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "final", "flower_singlelabel_ood11.keras")
OOD_CONFIG_PATH = os.path.join(BACKEND_DIR, "ood_config.json")
CLASSES_PATH = os.path.join(PROJECT_ROOT, "data", "metadata", "classes.json")
THRESHOLDS_PATH = os.path.join(BACKEND_DIR, "thresholds.json")
CARE_GUIDE_PATH = os.path.join(BACKEND_DIR, "care_guide.json")
COMPARISON_PATH = os.path.join(BACKEND_DIR, "model_comparison.json")
COMPARISON_MODEL_IDS = ["mobilenetv2", "mobilenetv3large", "efficientnetb0", "resnet50v2", "densenet121"]

_model_lock = threading.Lock()
_singlelabel_model = None
_ood_config = None
_classes = None
_thresholds = None
_care_guide = None
_comparison_models = {}  # model_id -> model đã nạp, cache lại (lazy load)


def get_ood_config():
    """Trả về cấu hình OOD (nếu có) từ ood_config.json."""
    global _ood_config
    if _ood_config is None and os.path.exists(OOD_CONFIG_PATH):
        try:
            with open(OOD_CONFIG_PATH, "r", encoding="utf-8") as f:
                _ood_config = json.load(f)
        except Exception as e:
            print(f"[model_loader] Cảnh báo đọc ood_config.json: {e}")
    return _ood_config


def load_singlelabel_model():
    """Model đơn nhãn dùng để phân loại từng vùng ảnh cắt ra (sliding window).
    Ưu tiên nạp mô hình 11 lớp (kèm reject class cho OOD) nếu đã huấn luyện xong (Tuần 7).
    Là model chính cho luồng nhận diện (FR-04), nạp 1 lần khi khởi động server (NFR-05)."""
    global _singlelabel_model
    with _model_lock:
        if _singlelabel_model is None:
            cfg = get_ood_config()
            target_path = None
            if cfg and cfg.get("model_path"):
                p = os.path.join(PROJECT_ROOT, *cfg["model_path"].split("/"))
                if os.path.exists(p):
                    target_path = p
            if target_path is None and os.path.exists(OOD_MODEL_PATH):
                target_path = OOD_MODEL_PATH
            if target_path is None:
                target_path = SINGLELABEL_MODEL_PATH

            if not os.path.exists(target_path):
                raise FileNotFoundError(f"Không tìm thấy model tại: {target_path}")
            _singlelabel_model = tf.keras.models.load_model(target_path)
            print(f"[model_loader] Đã nạp model nhận diện từ: {target_path}")
    return _singlelabel_model



def get_classes():
    """Trả về danh sách 10 lớp theo đúng thứ tự cố định trong classes.json (CON-05)."""
    global _classes
    if _classes is None:
        if not os.path.exists(CLASSES_PATH):
            raise FileNotFoundError(f"Không tìm thấy classes.json tại: {CLASSES_PATH}")
        with open(CLASSES_PATH, "r", encoding="utf-8") as f:
            _classes = json.load(f)
    return _classes


def get_thresholds():
    """Ngưỡng riêng từng lớp của model đa nhãn cũ (Tuần 3-4). Không dùng cho luồng nhận diện
    chính (sliding window dùng REGION_CONFIDENCE_THRESHOLD trong predict.py), chỉ giữ lại
    để tham khảo/đối chiếu khi viết báo cáo."""
    global _thresholds
    if _thresholds is None:
        if not os.path.exists(THRESHOLDS_PATH):
            raise FileNotFoundError(f"Không tìm thấy thresholds.json tại: {THRESHOLDS_PATH}")
        with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
            _thresholds = json.load(f)
    return _thresholds


def get_care_guide():
    """Trả về dữ liệu hướng dẫn chăm sóc, bắt buộc đủ 10 loại x 5 trường (CON-11).
    Thiếu dữ liệu là lỗi cấu hình (EF-05), không được trả care_guide rỗng/giả."""
    global _care_guide
    if _care_guide is None:
        if not os.path.exists(CARE_GUIDE_PATH):
            raise FileNotFoundError(f"Không tìm thấy care_guide.json tại: {CARE_GUIDE_PATH}")
        with open(CARE_GUIDE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        classes = get_classes()
        class_ids = [c["id"] for c in classes]
        required_fields = {"anh_sang", "nuoc_tuoi", "dat_trong", "nhiet_do", "luu_y"}
        for cls_id in class_ids:
            if cls_id not in data:
                raise ValueError(f"[EF-05] care_guide.json thiếu dữ liệu cho lớp: {cls_id}")
            fields = data[cls_id]
            if set(fields.keys()) != required_fields or not all(str(v).strip() for v in fields.values()):
                raise ValueError(f"[EF-05] care_guide.json thiếu/rỗng trường dữ liệu cho lớp: {cls_id}")

        _care_guide = data
    return _care_guide


def get_model_comparison():
    """Trả về nội dung model_comparison.json (FR-10). Thiếu file là lỗi cấu hình (EF-06)."""
    if not os.path.exists(COMPARISON_PATH):
        raise FileNotFoundError(f"[EF-06] Không tìm thấy model_comparison.json tại: {COMPARISON_PATH}")
    with open(COMPARISON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_comparison_model(model_id):
    """Nạp 1 trong 5 model benchmark ĐƠN NHÃN (softmax) theo yêu cầu (lazy load).
    Đường dẫn checkpoint lấy từ trường "checkpoint" của model_comparison.json, vì mỗi kiến trúc
    dùng checkpoint có val_loss thấp nhất giữa giai đoạn 1 và 2 (stage1_best hoặc stage2_best).
    Cache lại sau lần nạp đầu."""
    if model_id not in COMPARISON_MODEL_IDS:
        raise ValueError(f"model_id không hợp lệ: {model_id}")
    with _model_lock:
        if model_id not in _comparison_models:
            meta = next((m for m in get_model_comparison()["models"] if m["id"] == model_id), None)
            if meta is None or not meta.get("checkpoint"):
                raise FileNotFoundError(f"model_comparison.json thiếu trường 'checkpoint' cho {model_id}")
            path = os.path.join(PROJECT_ROOT, *meta["checkpoint"].split("/"))
            if not os.path.exists(path):
                raise FileNotFoundError(f"Không tìm thấy checkpoint cho {model_id} tại: {path}")
            print(f"[model_loader] Đang nạp model so sánh: {model_id} ...")
            _comparison_models[model_id] = tf.keras.models.load_model(path)
            print(f"[model_loader] Đã nạp xong: {model_id}")
        return _comparison_models[model_id]