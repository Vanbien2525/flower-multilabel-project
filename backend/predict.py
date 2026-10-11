import io
import numpy as np
from PIL import Image

from model_loader import (
    load_singlelabel_model, get_classes, get_care_guide,
    get_comparison_model, get_model_comparison, get_ood_config,
)

IMG_SIZE = (224, 224)

# Định dạng ảnh được phép (FR-02)
ALLOWED_FORMATS = {"JPEG", "PNG"}

# Độ tin cậy tối thiểu để 1 vùng được coi là "có hoa rõ ràng" (softmax, model đơn nhãn)
REGION_CONFIDENCE_THRESHOLD = 0.85
# Ngưỡng chồng lấp (IoU) để coi 2 vùng CÙNG loại là trùng nhau, gộp lại
IOU_MERGE_THRESHOLD = 0.3
# Ngưỡng "vùng này nằm trong vùng khác" -> nhiều khả năng chỉ là 1 mảnh cắt của cùng 1 bông hoa,
# không phải hoa thứ 2 (VD: cận cảnh vài cánh hoa sen bị nhận nhầm thành tulip)
CONTAINMENT_THRESHOLD = 0.75

COMPARISON_MODEL_ORDER = ["mobilenetv2", "mobilenetv3large", "efficientnetb0", "resnet50v2", "densenet121"]


def validate_image(file_bytes):
    """Kiểm tra ảnh có đúng định dạng JPEG/PNG không (FR-02, EF-01).
    Trả về (is_valid, error_message)."""
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()  # kiểm tra file có bị hỏng không
    except Exception:
        return False, "File không phải là ảnh hợp lệ hoặc đã bị hỏng."

    # Mở lại vì verify() làm "hỏng" object ảnh, không dùng lại được
    img = Image.open(io.BytesIO(file_bytes))
    if img.format not in ALLOWED_FORMATS:
        return False, f"Định dạng ảnh không được hỗ trợ ({img.format}). Chỉ chấp nhận JPEG hoặc PNG."

    return True, None


def generate_windows(width, height):
    """Sinh danh sách vùng ứng viên (x1, y1, x2, y2) theo lưới đa tỉ lệ,
    không ràng buộc đúng 1 bố cục cố định — phủ được ảnh 1, 2 hoặc 3 hoa ở nhiều vị trí."""
    windows = []

    windows.append((0, 0, width, height))

    windows.append((0, 0, width // 2, height))
    windows.append((width // 2, 0, width, height))
    windows.append((0, 0, width, height // 2))
    windows.append((0, height // 2, width, height))

    for i in range(3):
        x1, x2 = i * width // 3, (i + 1) * width // 3
        windows.append((x1, 0, x2, height))
    for i in range(2):
        y1, y2 = i * height // 2, (i + 1) * height // 2
        windows.append((0, y1, width // 2, y2))
        windows.append((width // 2, y1, width, y2))

    step_x, step_y = width // 4, height // 4
    win_w, win_h = width // 2, height // 2
    for gy in range(3):
        for gx in range(3):
            x1, y1 = gx * step_x, gy * step_y
            x2, y2 = min(x1 + win_w, width), min(y1 + win_h, height)
            if x2 - x1 > 20 and y2 - y1 > 20:
                windows.append((x1, y1, x2, y2))

    return windows


def crop_to_square(img, box):
    """Cắt vùng box rồi cắt vuông giữa vùng đó, resize về IMG_SIZE (không làm méo tỉ lệ)."""
    region = img.crop(box)
    w, h = region.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return region.crop((left, top, left + s, top + s)).resize(IMG_SIZE)


def iou(box1, box2):
    """Độ chồng lấp giữa 2 vùng, dùng để gộp các vùng cùng loại trùng nhau."""
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def containment_ratio(inner_box, outer_box):
    """Tỉ lệ diện tích inner_box nằm trong outer_box. 1.0 nghĩa là inner nằm trọn trong outer."""
    x1 = max(inner_box[0], outer_box[0]); y1 = max(inner_box[1], outer_box[1])
    x2 = min(inner_box[2], outer_box[2]); y2 = min(inner_box[3], outer_box[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    inner_area = (inner_box[2] - inner_box[0]) * (inner_box[3] - inner_box[1])
    return inter / inner_area if inner_area > 0 else 0


def detect_flowers(file_bytes):
    """FR-04: chia ảnh thành nhiều vùng, phân loại từng vùng bằng model đơn nhãn,
    gộp kết quả trùng lặp -> danh sách hoa duy nhất kèm care_guide.
    Trả về thêm (boxes_by_id, img): vùng ảnh đã dùng để nhận diện mỗi loại hoa và ảnh gốc
    đã mở sẵn, để get_model_predictions() dùng ĐÚNG vùng này khi so sánh 5 model
    (đảm bảo so sánh công bằng, không cho model khác xem ảnh khác)."""
    model = load_singlelabel_model()
    classes = get_classes()
    care_guide_data = get_care_guide()
    ood_cfg = get_ood_config()

    # Áp dụng ngưỡng gate và chỉ số lớp Reject (nếu dùng model 11 lớp OOD)
    conf_threshold = float(ood_cfg.get("confidence_gate", REGION_CONFIDENCE_THRESHOLD)) if ood_cfg else REGION_CONFIDENCE_THRESHOLD
    reject_idx = int(ood_cfg.get("reject_class_index", -1)) if ood_cfg and "reject_class_index" in ood_cfg else -1

    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    width, height = img.size
    windows = generate_windows(width, height)

    batch = np.stack([
        np.array(crop_to_square(img, box), dtype=np.float32) for box in windows
    ])
    probs = model.predict(batch, verbose=0)

    candidates = []
    for box, p in zip(windows, probs):
        class_idx = int(np.argmax(p))
        confidence = float(p[class_idx])

        # Bỏ qua nếu cửa sổ được dự đoán là lớp Reject (OOD/hoa lạ/ảnh nền)
        if reject_idx >= 0 and class_idx == reject_idx:
            continue

        # Chỉ chấp nhận nếu thuộc 10 loài hoa ID và độ tin cậy vượt ngưỡng gate
        if class_idx < len(classes) and confidence >= conf_threshold:
            candidates.append((class_idx, confidence, box))

    candidates.sort(key=lambda c: c[1], reverse=True)
    kept = []
    for class_idx, confidence, box in candidates:
        is_fragment_of_other = any(
            k_class != class_idx and containment_ratio(box, k_box) >= CONTAINMENT_THRESHOLD
            for k_class, _, k_box in kept
        )
        if is_fragment_of_other:
            continue

        is_duplicate = any(
            k_class == class_idx and iou(box, k_box) >= IOU_MERGE_THRESHOLD
            for k_class, _, k_box in kept
        )
        if not is_duplicate:
            kept.append((class_idx, confidence, box))

    best_per_class = {}
    for class_idx, confidence, box in kept:
        if class_idx not in best_per_class or confidence > best_per_class[class_idx][0]:
            best_per_class[class_idx] = (confidence, box)

    detected = []
    boxes_by_id = {}
    for class_idx, (confidence, box) in best_per_class.items():
        cls = classes[class_idx]
        detected.append({
            "id": cls["id"],
            "name": cls["name"],
            "confidence": round(confidence, 4),
            "care_guide": care_guide_data[cls["id"]],
        })
        boxes_by_id[cls["id"]] = box

    detected.sort(key=lambda x: x["confidence"], reverse=True)
    return detected, boxes_by_id, img


def get_model_predictions(img, boxes_by_id):
    """Chạy cả 5 mô hình benchmark trên ĐÚNG vùng ảnh mà hệ thống đã dùng để nhận diện
    mỗi loại hoa (thay vì nguyên ảnh gốc) — đảm bảo mọi mô hình được so sánh trên cùng
    1 bằng chứng. Chênh lệch còn lại giữa các mô hình chỉ phản ánh khác biệt thật giữa
    các kiến trúc, không phải do bị cho xem ảnh khác nhau."""
    classes = get_classes()
    class_index = {c["id"]: i for i, c in enumerate(classes)}
    meta_by_id = {m["id"]: m for m in get_model_comparison()["models"]}

    flower_ids = list(boxes_by_id.keys())
    batch = np.stack([
        np.array(crop_to_square(img, boxes_by_id[fid]), dtype=np.float32) for fid in flower_ids
    ])

    results = []
    for model_id in COMPARISON_MODEL_ORDER:
        try:
            model = get_comparison_model(model_id)
        except FileNotFoundError:
            continue  # thiếu checkpoint model này -> bỏ qua, không làm hỏng cả bảng

        probs = model.predict(batch, verbose=0)  # shape (số hoa, 10)
        predictions = {
            fid: round(float(probs[i][class_index[fid]]), 4)
            for i, fid in enumerate(flower_ids)
        }
        results.append({
            "model_id": model_id,
            "model_name": meta_by_id.get(model_id, {}).get("name", model_id),
            "is_selected": meta_by_id.get(model_id, {}).get("is_selected", False),
            "predictions": predictions,
        })
    return results