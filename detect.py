"""
Standalone YOLO26 TensorRT inference. No ultralytics import.

Usage:
    python detect.py --engine best.engine --source img.jpg --classes classes.txt
    python detect.py --engine best.engine --source video.mp4 --save out.mp4
    python detect.py --engine best.engine --source 0                     # webcam
"""
import argparse
import time

import cv2

from utils.trt_engine import YOLO26TRT
from utils.utils import preprocess, postprocess, draw_detections


def load_class_names(path):
    if not path:
        return None
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def run_image(model, img_path, class_names, conf_thres, save_path=None, show=False):
    img = cv2.imread(img_path)
    inp, ratio, pad = preprocess(img, model.input_size)
    t0 = time.time()
    raw = model.infer(inp)
    dt = time.time() - t0
    dets = postprocess(raw, ratio, pad, img.shape, conf_thres)
    print(f"{img_path}: {len(dets)} detections in {dt * 1000:.1f} ms")
    out = draw_detections(img.copy(), dets, class_names)
    if save_path:
        cv2.imwrite(save_path, out)
        print(f"Saved: {save_path}")
    if show:
        cv2.imshow("YOLO26-TRT", out)
        cv2.waitKey(0)


def run_video(model, source, class_names, conf_thres, save_path=None, show=False):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    writer = None
    if save_path:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        inp, ratio, pad = preprocess(frame, model.input_size)
        t0 = time.time()
        raw = model.infer(inp)
        dt = time.time() - t0
        dets = postprocess(raw, ratio, pad, frame.shape, conf_thres)
        out = draw_detections(frame, dets, class_names)
        cv2.putText(out, f"{1 / max(dt, 1e-6):.1f} FPS", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if writer:
            writer.write(out)
        if show:
            cv2.imshow("YOLO26-TRT", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="path to .engine file")
    ap.add_argument("--source", required=True, help="image/video path or webcam index")
    ap.add_argument("--classes", default=None, help="txt file, one class name per line")
    ap.add_argument("--conf-thres", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--save", default=None, help="output path")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    class_names = load_class_names(args.classes)
    model = YOLO26TRT(args.engine, input_size=(args.imgsz, args.imgsz))

    is_image = args.source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    if is_image:
        run_image(model, args.source, class_names, args.conf_thres, args.save, args.show)
    else:
        run_video(model, args.source, class_names, args.conf_thres, args.save, args.show)


if __name__ == "__main__":
    main()
