#!/usr/bin/env python3
"""Check how many of the QRs sitting on the vertical center line actually decode.

Runs the same detector + center-line logic as run.py, but on a single frame
(from an image file, or grabbed live from the camera) and prints a per-label
table: which labels cross the line, which QR was paired with each, and whether
zxing-cpp read it. Use it to tune --qr-margin and --conf-thres before running
the live pipeline.

Usage:
    python3 check_qr.py --engine best.engine --classes classes.txt --image shot.png
    python3 check_qr.py --engine best.engine --classes classes.txt          # live grab
    python3 check_qr.py --engine best.engine --classes classes.txt --qr-margin 0.25
"""

import argparse
import os
import time

import cv2
import numpy as np

import run as cam
from utils.qr import box_center, decode_qr, pick_qr_for_label
from utils.trt_engine import YOLO26TRT
from utils.utils import preprocess, postprocess, draw_detections


def grab_frame(args):
    """One frame from an image file, or from the camera (after warm-up)."""
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise RuntimeError(f"could not read {args.image}")
        return frame

    cam_index = args.index if args.index is not None else cam.find_camera_index()
    pipeline = cam.gstreamer_pipeline(cam_index, args.width, args.height,
                                      args.fps, args.format)
    print(f"[camera] using /dev/video{cam_index}")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera via GStreamer pipeline")
    try:
        frame = None
        # Throw away the first frames — the sensor's auto-exposure needs a
        # moment to settle, and a dark frame decodes badly.
        for _ in range(args.warmup):
            ok, f = cap.read()
            if ok:
                frame = f
        if frame is None:
            raise RuntimeError("no frame captured")
        return cam.rotate_frame(frame, args.rotate)
    finally:
        cap.release()


def main():
    ap = argparse.ArgumentParser(description="Check QR decoding on the center line for one frame.")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--classes", default=None)
    ap.add_argument("--image", default=None, help="check this image instead of the camera")
    ap.add_argument("--conf-thres", type=float, default=cam.DEFAULT_CONF_THRES)
    ap.add_argument("--imgsz", type=int, default=cam.DEFAULT_IMGSZ)
    ap.add_argument("--label-class", default=cam.DEFAULT_LABEL_CLASS)
    ap.add_argument("--qr-class", default=cam.DEFAULT_QR_CLASS)
    ap.add_argument("--line-pos", type=float, default=cam.DEFAULT_LINE_POS)
    ap.add_argument("--qr-margin", type=float, default=cam.DEFAULT_QR_MARGIN)
    ap.add_argument("--qr-margin-min", type=int, default=8)
    ap.add_argument("--all", action="store_true",
                     help="check every detected label, not just the ones on the line.")
    ap.add_argument("--out", default="qr_check.png", help="annotated output image")
    ap.add_argument("--dump-crops", default=None,
                     help="directory to write each crop that was handed to zxing")
    # camera args (ignored with --image)
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--width", type=int, default=cam.DEFAULT_WIDTH)
    ap.add_argument("--height", type=int, default=cam.DEFAULT_HEIGHT)
    ap.add_argument("--fps", type=int, default=cam.DEFAULT_FPS)
    ap.add_argument("--format", default=cam.DEFAULT_FORMAT, choices=["MJPG", "YUYV"])
    ap.add_argument("--rotate", type=int, default=cam.DEFAULT_ROTATE, choices=[0, 90, 180, 270])
    ap.add_argument("--warmup", type=int, default=15, help="frames to discard before grabbing")
    args = ap.parse_args()

    frame = grab_frame(args)
    class_names = cam.load_class_names(args.classes)
    label_cls = cam.class_index(class_names, args.label_class, 0)
    qr_cls = cam.class_index(class_names, args.qr_class, 1)

    model = YOLO26TRT(args.engine, input_size=(args.imgsz, args.imgsz))
    inp, ratio, pad = preprocess(frame, model.input_size)
    t0 = time.time()
    dets = postprocess(model.infer(inp), ratio, pad, frame.shape, args.conf_thres)
    print(f"[model] {len(dets)} detections in {(time.time() - t0) * 1e3:.0f} ms "
          f"on a {frame.shape[1]}x{frame.shape[0]} frame")

    line_x = int(frame.shape[1] * args.line_pos)
    labels = [d for d in dets if int(d[5]) == label_cls]
    qr_dets = [d for d in dets if int(d[5]) == qr_cls]
    on_line = [d for d in labels if d[0] <= line_x <= d[2]]
    checked = sorted(labels if args.all else on_line, key=lambda d: box_center(d)[1])

    print(f"[line] x={line_x}  labels={len(labels)}  qr_boxes={len(qr_dets)}  "
          f"labels crossing the line={len(on_line)}"
          f"{'  (checking all labels)' if args.all else ''}")

    if args.dump_crops:
        os.makedirs(args.dump_crops, exist_ok=True)

    annotated = draw_detections(frame.copy(), dets, class_names)
    cv2.line(annotated, (line_x, 0), (line_x, frame.shape[0]), (0, 128, 255), 3)

    ok_count = 0
    print(f"\n{'#':>2}  {'label y':>10}  {'qr box':>26}  {'crop':>11}  result")
    print("-" * 88)
    for i, label in enumerate(checked, 1):
        cy = box_center(label)[1]
        qr = pick_qr_for_label(label[:4], qr_dets)
        if qr is None:
            print(f"{i:>2}  {cy:>10.0f}  {'-- no qr paired --':>26}  {'-':>11}  NO QR BOX")
            continue

        text, crop = decode_qr(frame, qr[:4], args.qr_margin, args.qr_margin_min)
        qx1, qy1, qx2, qy2 = (int(v) for v in qr[:4])
        crop_wh = f"{crop[2] - crop[0]}x{crop[3] - crop[1]}"
        status = f"OK  {text}" if text else "FAIL (not decoded)"
        if text:
            ok_count += 1
        print(f"{i:>2}  {cy:>10.0f}  {f'{qx1},{qy1},{qx2},{qy2}':>26}  {crop_wh:>11}  {status}")

        if args.dump_crops:
            tag = "ok" if text else "fail"
            cv2.imwrite(os.path.join(args.dump_crops, f"{i:02d}_{tag}.png"),
                        frame[crop[1]:crop[3], crop[0]:crop[2]])

        color = (255, 0, 255) if text else (0, 0, 255)
        cv2.rectangle(annotated, (crop[0], crop[1]), (crop[2], crop[3]), color, 3)
        cv2.putText(annotated, f"{i}:{'OK' if text else 'FAIL'}", (crop[0], crop[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    total = len(checked)
    print("-" * 88)
    print(f"decoded {ok_count}/{total} "
          f"({'all good' if ok_count == total and total else 'see FAIL rows above'})")

    cv2.imwrite(args.out, annotated)
    print(f"[out] annotated frame written to {args.out}")


if __name__ == "__main__":
    main()
