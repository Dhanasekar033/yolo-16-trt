from email.mime import text

import cv2
import numpy as np


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize + pad image to new_shape while keeping aspect ratio.
    Returns padded image, scale ratio, and (dw, dh) padding."""
    h0, w0 = img.shape[:2]
    new_h, new_w = new_shape

    r = min(new_h / h0, new_w / w0)
    resized_w, resized_h = int(round(w0 * r)), int(round(h0 * r))

    dw, dh = new_w - resized_w, new_h - resized_h
    dw /= 2
    dh /= 2

    if (w0, h0) != (resized_w, resized_h):
        img = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def preprocess(img_bgr, input_size=(640, 640)):
    """BGR np.uint8 HWC -> normalized float32 NCHW, plus letterbox metadata."""
    img, ratio, (dw, dh) = letterbox(img_bgr, input_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, 0)  # add batch dim
    return img, ratio, (dw, dh)


def scale_boxes(boxes, ratio, pad):
    """Map boxes from letterboxed-input coords back to original image coords."""
    dw, dh = pad
    boxes = boxes.copy()
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes[:, :4] /= ratio
    return boxes


def postprocess(raw_output, ratio, pad, orig_shape, conf_thres=0.25):
    """raw_output: (1, max_det, 6) = [x1,y1,x2,y2,conf,cls] from an
    end2end YOLO26 engine (NMS already applied by the model)."""
    dets = raw_output[0]
    dets = dets[dets[:, 4] >= conf_thres]
    if dets.shape[0] == 0:
        return np.empty((0, 6), dtype=np.float32)

    boxes = scale_boxes(dets[:, :4], ratio, pad)
    h0, w0 = orig_shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w0)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h0)

    return np.concatenate([boxes, dets[:, 4:6]], axis=1)  # x1,y1,x2,y2,conf,cls


def draw_detections(img, dets, class_names=None):
    for x1, y1, x2, y2, conf, cls_id in dets:
        cls_id = int(cls_id)
        label = class_names[cls_id] if class_names and cls_id < len(class_names) else str(cls_id)
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
        text = f"{label} {conf:.2f}"
        # print(text)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (p1[0], p1[1] - th - 4), (p1[0] + tw, p1[1]), (0, 255, 0), -1)
        cv2.putText(img, text, (p1[0], p1[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img
