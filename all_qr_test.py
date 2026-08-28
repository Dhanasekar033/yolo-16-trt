#!/usr/bin/env python3
"""
Global Shutter Camera viewer + live YOLO26 TensorRT inference.

Same GStreamer/OpenCV capture pipeline and staleness-guard setup as
cam_view.py, with per-frame YOLO26 detection + box drawing added in.
No ultralytics import anywhere — inference goes through trt_engine.py.

QR handling in this version:
  Every detection box classified as --qr-class is cropped (with a margin)
  and decoded EVERY FRAME. There is no "trigger line" / "label crossing"
  logic — if a QR box is visible and readable, it gets decoded. A small
  dedup cache avoids reprinting the same code on every consecutive frame
  while it sits in view.

Usage:
    python3 run.py --engine best.engine
    python3 run.py --engine best.engine --classes classes.txt --conf-thres 0.35
    python3 run.py --engine best.engine --fps 15 --width 1280 --height 972
    python3 run.py --engine best.engine --index 0        # skip auto-detect
    python3 run.py --engine best.engine --no-display     # headless, prints detections
    python3 run.py --engine best.engine --save out.mp4   # also record annotated video
"""

import argparse
import os
import time

import cv2

from utils.trt_engine import YOLO26TRT
from utils.utils import preprocess, postprocess, draw_detections

# ── Stream config (same defaults as cam_view.py) ────────────────────────────
DEFAULT_CAM_INDEX = 0
DEFAULT_WIDTH     = 2592
DEFAULT_HEIGHT    = 1944
DEFAULT_FPS       = 60       # MJPG supports 60fps at full 2592x1944; YUYV only
                              # goes to 35fps at that size (see --list-formats-ext)
DEFAULT_FORMAT    = "MJPG"
DISPLAY_MAX_W     = 1280     # imshow window is capped to this width so a full
DISPLAY_MAX_H     = 960      # -res frame doesn't overflow the screen
DEFAULT_ROTATE    = 270      # fixed rotation applied to every frame: 0/90/180/270

# ── Inference config ─────────────────────────────────────────────────────────
DEFAULT_IMGSZ      = 640
DEFAULT_CONF_THRES = 0.25

# ── QR decode config ─────────────────────────────────────────────────────────
DEFAULT_QR_CLASS  = "qr_code"    # class that is cropped and decoded
DEFAULT_QR_MARGIN = 0.15         # quiet zone added around the qr box
DEDUP_TTL_SEC     = 2.0          # suppress reprinting the same string for this long

ROTATE_MAP = {
    0:   None,
    90:  cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def rotate_frame(frame, degrees):
    """Rotate a frame by a fixed angle (0/90/180/270). No-op for 0."""
    code = ROTATE_MAP[degrees]
    return frame if code is None else cv2.rotate(frame, code)


GS_CAMERA_NAME = "Global Shutter Camera"


def list_cameras():
    """List (index, name) for every /dev/videoN device via its v4l2 sysfs name."""
    cams = []
    for d in sorted(os.listdir("/dev")):
        if not d.startswith("video"):
            continue
        sys_path = f"/sys/class/video4linux/{d}/name"
        if not os.path.exists(sys_path):
            continue
        with open(sys_path) as f:
            name = f.read().strip()
        cams.append((int(d.replace("video", "")), name))
    return cams


def find_camera_index(name_substring=GS_CAMERA_NAME, default=DEFAULT_CAM_INDEX):
    """Find the /dev/videoN index whose v4l2 name contains name_substring."""
    cams = list_cameras()
    for index, name in cams:
        if name_substring.lower() in name.lower():
            return index
    print(f"[camera] '{name_substring}' not found among {cams} — "
          f"falling back to index {default}")
    return default


def gstreamer_pipeline(cam_index=DEFAULT_CAM_INDEX, width=DEFAULT_WIDTH,
                        height=DEFAULT_HEIGHT, fps=DEFAULT_FPS, format=DEFAULT_FORMAT):
    """Build a GStreamer pipeline string for v4l2src (MJPG or YUYV)."""
    QUEUE = "queue leaky=downstream max-size-buffers=1"
    SINK  = ("videoconvert ! video/x-raw, format=BGR ! "
             "appsink drop=true max-buffers=1 sync=false")

    if format.upper() == "MJPG":
        return (
            f"v4l2src device=/dev/video{cam_index} ! "
            f"image/jpeg, width={width}, height={height}, framerate={fps}/1 ! "
            f"{QUEUE} ! jpegdec ! {SINK}"
        )
    else:
        return (
            f"v4l2src device=/dev/video{cam_index} ! "
            f"video/x-raw, width={width}, height={height}, framerate={fps}/1 ! "
            f"{QUEUE} ! {SINK}"
        )


def load_class_names(path):
    if not path:
        return None
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def class_index(class_names, name, fallback):
    """Resolve a class name to its id; fall back to a fixed index when no
    classes.txt was supplied (or the name isn't in it)."""
    if class_names and name in class_names:
        return class_names.index(name)
    print(f"[qr] class '{name}' not found in --classes, using index {fallback}")
    return fallback


class AllBoxQRDecoder:
    """
    Decodes every detection box of a given class, every frame — no trigger
    line, no label-crossing state machine.

    ASSUMPTION (adjust `_unpack` below if wrong): each item in `dets` is a
    row/sequence [x1, y1, x2, y2, conf, cls_id]. If your postprocess()
    returns objects/dicts instead, only `_unpack` needs to change.
    """

    def __init__(self, qr_cls, margin=DEFAULT_QR_MARGIN, min_px=8,
                 on_decode=None, dump_dir=None, dedup_ttl=DEDUP_TTL_SEC):
        self.qr_cls = qr_cls
        self.margin = margin
        self.min_px = min_px
        self.on_decode = on_decode or (lambda t: None)
        self.dump_dir = dump_dir
        self.dedup_ttl = dedup_ttl
        self.detector = cv2.QRCodeDetector()
        self.count = 0
        self._last_boxes = []          # boxes decoded this frame, for draw()
        self._recent = {}              # decoded text -> last-seen timestamp
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)

    @staticmethod
    def _unpack(det):
        x1, y1, x2, y2, conf, cls_id = det[0], det[1], det[2], det[3], det[4], det[5]
        return float(x1), float(y1), float(x2), float(y2), float(conf), int(cls_id)

    def _expand_box(self, x1, y1, x2, y2, w, h):
        bw, bh = x2 - x1, y2 - y1
        mx = max(bw * self.margin, self.min_px)
        my = max(bh * self.margin, self.min_px)
        nx1 = max(0, int(x1 - mx))
        ny1 = max(0, int(y1 - my))
        nx2 = min(w, int(x2 + mx))
        ny2 = min(h, int(y2 + my))
        return nx1, ny1, nx2, ny2

    def update(self, frame, dets):
        """Crop + decode every qr_cls box in this frame. Call once per frame."""
        h, w = frame.shape[:2]
        now = time.time()
        self._last_boxes = []

        for det in dets:
            x1, y1, x2, y2, conf, cls_id = self._unpack(det)
            if cls_id != self.qr_cls:
                continue

            nx1, ny1, nx2, ny2 = self._expand_box(x1, y1, x2, y2, w, h)
            if nx2 <= nx1 or ny2 <= ny1:
                continue

            crop = frame[ny1:ny2, nx1:nx2]
            self._last_boxes.append((nx1, ny1, nx2, ny2))

            text, points, _ = self.detector.detectAndDecode(crop)

            if text:
                last_seen = self._recent.get(text, 0.0)
                if now - last_seen > self.dedup_ttl:
                    self.count += 1
                    self.on_decode(text)
                self._recent[text] = now
            elif self.dump_dir:
                fname = os.path.join(self.dump_dir, f"qr_fail_{int(now * 1000)}.png")
                cv2.imwrite(fname, crop)

        # prune old dedup entries
        stale = [t for t, ts in self._recent.items() if now - ts > self.dedup_ttl]
        for t in stale:
            del self._recent[t]

    def draw(self, frame):
        for (x1, y1, x2, y2) in self._last_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(frame, f"QR decoded: {self.count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="Stream the Global Shutter Camera + run live YOLO26 TensorRT inference.")
    # camera args
    ap.add_argument("--index", type=int, default=None,
                     help="Force a /dev/videoN index (skips auto-detect).")
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--format", default=DEFAULT_FORMAT, choices=["MJPG", "YUYV"])
    ap.add_argument("--rotate", type=int, default=DEFAULT_ROTATE, choices=[0, 90, 180, 270],
                     help="Rotate every frame by a fixed angle (clockwise).")
    ap.add_argument("--no-display", action="store_true",
                     help="Just print FPS/detections instead of opening a window (headless).")
    # inference args
    ap.add_argument("--engine", default="best.engine", help="path to .engine file")
    ap.add_argument("--classes", default=None, help="txt file, one class name per line")
    ap.add_argument("--conf-thres", type=float, default=DEFAULT_CONF_THRES)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--save", default=None, help="optional path to record annotated video")
    # qr decode args
    ap.add_argument("--no-qr", action="store_true",
                     help="disable QR decoding.")
    ap.add_argument("--qr-class", default=DEFAULT_QR_CLASS,
                     help="class name of the QR box that gets cropped and decoded.")
    ap.add_argument("--qr-margin", type=float, default=DEFAULT_QR_MARGIN,
                     help="quiet zone around the qr box, as a fraction of its size.")
    ap.add_argument("--qr-margin-min", type=int, default=8,
                     help="minimum quiet zone in pixels.")
    ap.add_argument("--qr-dedup-sec", type=float, default=DEDUP_TTL_SEC,
                     help="suppress reprinting the same decoded string for this many seconds.")
    ap.add_argument("--dump-crops", default=None,
                     help="directory to save qr crops that failed to decode.")
    args = ap.parse_args()

    cam_index = args.index if args.index is not None else find_camera_index()
    pipeline = gstreamer_pipeline(cam_index, args.width, args.height, args.fps, args.format)
    print(f"[camera] using /dev/video{cam_index}")
    print(f"[camera] pipeline: {pipeline}")

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera via GStreamer pipeline")

    class_names = load_class_names(args.classes)
    model = YOLO26TRT(args.engine, input_size=(args.imgsz, args.imgsz))
    print(f"[model] loaded {args.engine}")

    # 90/270 rotation swaps the effective width/height for sizing the window/writer.
    disp_w, disp_h = (args.height, args.width) if args.rotate in (90, 270) else (args.width, args.height)

    decoder = None
    if not args.no_qr:
        decoder = AllBoxQRDecoder(
            qr_cls=class_index(class_names, args.qr_class, 1),
            margin=args.qr_margin,
            min_px=args.qr_margin_min,
            on_decode=lambda t: print(f"\n[qr] decoded: {t}"),
            dump_dir=args.dump_crops,
            dedup_ttl=args.qr_dedup_sec,
        )
        print(f"[qr] decoding every '{args.qr_class}' box every frame "
              f"(dedup window {args.qr_dedup_sec}s)")

    writer = None
    if args.save:
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                  args.fps, (disp_w, disp_h))

    if not args.no_display:
        # WINDOW_NORMAL makes the window resizable; without it, imshow opens
        # at the frame's native resolution which overflows most screens. We
        # set an initial on-screen size, capped to DISPLAY_MAX_*, while the
        # capture/inference itself still runs at full resolution.
        win_name = "Global Shutter Camera - YOLO26 TRT"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        scale = min(DISPLAY_MAX_W / disp_w, DISPLAY_MAX_H / disp_h, 1.0)
        cv2.resizeWindow(win_name, int(disp_w * scale), int(disp_h * scale))

    try:
        prev_t = time.time()
        fps = 0.0
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[camera] frame grab failed, retrying...")
                continue

            frame = rotate_frame(frame, args.rotate)

            # ── inference ────────────────────────────────────────────────
            inp, ratio, pad = preprocess(frame, model.input_size)
            raw = model.infer(inp)
            dets = postprocess(raw, ratio, pad, frame.shape, args.conf_thres)

            # ── decode every QR box in this frame ───────────────────────
            if decoder is not None:
                decoder.update(frame, dets)

            frame = draw_detections(frame, dets, class_names)
            if decoder is not None:
                decoder.draw(frame)

            # Simple running FPS: time between consecutive frames, smoothed
            # with an exponential moving average so the readout doesn't
            # jitter frame-to-frame.
            now = time.time()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                inst_fps = 1.0 / dt
                fps = inst_fps if fps == 0.0 else (0.9 * fps + 0.1 * inst_fps)

            if args.no_display:
                qr_status = f"  qr={decoder.count}" if decoder is not None else ""
                print(f"[camera] frame {frame.shape}  fps={fps:.1f}  "
                      f"dets={len(dets)}{qr_status}", end="\r")
            else:
                cv2.putText(frame, f"FPS: {fps:.1f}  dets: {len(dets)}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow(win_name, frame)

            if writer:
                writer.write(frame)

            if not args.no_display and (cv2.waitKey(1) & 0xFF == ord("q")):
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
