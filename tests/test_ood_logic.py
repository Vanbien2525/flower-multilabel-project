import os
import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

REJ_CLS_IDX = 10
IOU_MERGE_THRESHOLD = 0.3
CONTAINMENT_THRESHOLD = 0.75

def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0

def containment_ratio(inner, outer):
    x1, y1, x2, y2 = max(inner[0], outer[0]), max(inner[1], outer[1]), min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return inter / area if area > 0 else 0

def detect_head(P_win, boxes, gate):
    """Mô phỏng pipeline detect_flowers(): bỏ qua các cửa sổ mà model dự đoán là lớp Reject (argmax == 10)"""
    cls_preds = P_win.argmax(axis=1)
    conf_preds = P_win.max(axis=1)

    # Chỉ giữ cửa sổ có argmax != REJ_CLS_IDX và conf >= gate
    cands = [(int(c), float(m), b) for c, m, b in zip(cls_preds, conf_preds, boxes)
             if c != REJ_CLS_IDX and m >= gate]
    cands.sort(key=lambda t: t[1], reverse=True)

    kept = []
    for c, m, b in cands:
        if any(kc != c and containment_ratio(b, kb) >= CONTAINMENT_THRESHOLD for kc, _, kb in kept): continue
        if any(kc == c and iou(b, kb) >= IOU_MERGE_THRESHOLD for kc, _, kb in kept): continue
        kept.append((c, m, b))
    return {c for c, _, _ in kept}

def test_detect_head():
    # Giả lập 21 cửa sổ với shape (21, 11)
    np.random.seed(42)
    P_win = np.zeros((21, 11), dtype=np.float32)

    # 1. Các cửa sổ 0 -> 9: argmax = 10 (Reject class), confidence = 0.95
    P_win[:10, 10] = 0.95
    P_win[:10, 0] = 0.05

    # 2. Cửa sổ 10: argmax = 2 (ID class), confidence = 0.85
    P_win[10, 2] = 0.85
    P_win[10, 10] = 0.15

    # 3. Cửa sổ 11: argmax = 5 (ID class), confidence = 0.40 (dưới gate 0.70)
    P_win[11, 5] = 0.40
    P_win[11, 10] = 0.60

    # 4. Các cửa sổ còn lại 12 -> 20: argmax = 10 (Reject class)
    P_win[12:, 10] = 0.90

    # Dummy boxes cho 21 cửa sổ
    boxes = [(0, 0, 100, 100) for _ in range(21)]

    gate = 0.70
    detected_classes = detect_head(P_win, boxes, gate)

    # Khẳng định index 10 bị bỏ qua
    assert 10 not in detected_classes, "Index 10 (reject class) must be ignored!"
    # Khẳng định index 5 bị bỏ qua vì conf (0.40) < gate (0.70)
    assert 5 not in detected_classes, "Class 5 must be ignored due to low confidence!"
    # Khẳng định index 2 được giữ lại vì conf (0.85) >= gate (0.70)
    assert 2 in detected_classes, "Class 2 should be detected!"
    assert detected_classes == {2}


def test_model_stitching():
    num_classes = 10
    input_shape = (224, 224, 3)

    # 1. Khởi tạo dummy backbone 10 lớp Keras có layer 'global_average_pooling2d'
    inp = keras.Input(shape=input_shape, name="input_layer")
    x = layers.Conv2D(16, (3, 3), padding="same")(inp)
    gap = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    dense_old = layers.Dense(256, activation="relu", name="dense")(gap)
    out_old = layers.Dense(num_classes, activation="softmax", name="dense_1")(dense_old)
    backbone_model = keras.Model(inputs=inp, outputs=out_old)

    # 2. Khởi tạo dummy head 11 lớp
    head_inp = keras.Input(shape=(16,), name="head_input") # GAP output shape is 16
    h1 = layers.Dense(256, activation="relu", name="h1")(head_inp)
    drop = layers.Dropout(0.3)(h1)
    h2 = layers.Dense(num_classes + 1, name="h2")(drop)
    head_model = keras.Model(inputs=head_inp, outputs=h2)

    # 3. Chạy logic ghép model (như Cell 13)
    gap_out = backbone_model.get_layer("global_average_pooling2d").output
    dense_h1 = layers.Dense(256, activation="relu", name="dense_ood")(gap_out)
    drop_ood = layers.Dropout(0.3, name="dropout_ood")(dense_h1)
    logits_11 = layers.Dense(num_classes + 1, name="dense_11_logits")(drop_ood)
    out_softmax = layers.Softmax(name="softmax_11")(logits_11)

    full11_model = keras.Model(inputs=backbone_model.input, outputs=out_softmax)

    # Gán trọng số từ head_model sang full11_model
    full11_model.get_layer("dense_ood").set_weights(head_model.get_layer("h1").get_weights())
    full11_model.get_layer("dense_11_logits").set_weights(head_model.get_layer("h2").get_weights())

    # 4. Truyền dummy tensor (2, 224, 224, 3) vào full11_model
    dummy_input = np.random.randn(2, 224, 224, 3).astype(np.float32)
    output = full11_model.predict(dummy_input, verbose=0)

    # Assert output shape là (2, 11)
    assert output.shape == (2, 11), f"Expected shape (2, 11), got {output.shape}"
