#!/usr/bin/env python3
"""
Global Shutter Camera viewer + live YOLO26 TensorRT inference.

Same GStreamer/OpenCV capture pipeline and staleness-guard setup as
cam_view.py, with per-frame YOLO26 detection + box drawing added in.
No ultralytics import anywhere — inference goes through trt_engine.py.

Usage:
    python3 run.py --engine best.engine
    python3 run.py --engine best.engine --classes classes.txt --conf-thres 0.35
    python3 run.py --conf-label 0.35 --conf-qr 0.5       # per-class thresholds
    python3 run.py --engine best.engine --fps 15 --width 1280 --height 972
    python3 run.py --engine best.engine --index 0        # skip auto-detect
    python3 run.py --engine best.engine --no-display     # headless, prints detections
    python3 run.py --engine best.engine --save out.mp4   # also record annotated video
    python3 run.py --xlsx validation.xlsx                # validate against the sheet
    python3 run.py --no-relay --no-validate              # vision only, no machine

On start-up relay 0 is switched ON to run the winding machine, and every
label crossing the trigger line is decoded and checked, top to bottom, against
the QR DATA1..4 columns of validation.xlsx.
"""

import argparse
import os
import time

import cv2

from utils.qr import CenterLineQRDecoder
from utils.relay import RelayController
from utils.trt_engine import YOLO26TRT
from utils.utils import preprocess, postprocess, draw_detections
from utils.validation import BOTTOM_UP, TOP_DOWN, SequenceValidator, ValidationSheet

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
DEFAULT_CONF_THRES = 0.25   # applies to any class without its own threshold
DEFAULT_CONF_LABEL = None   # None -> fall back to DEFAULT_CONF_THRES
DEFAULT_CONF_QR    = None

# ── QR decode config ─────────────────────────────────────────────────────────
DEFAULT_LABEL_CLASS = "label"      # class whose crossing triggers a decode
DEFAULT_QR_CLASS    = "qr_code"    # class that is cropped and decoded
DEFAULT_LINE_POS    = 0.5          # vertical trigger line, fraction of width
DEFAULT_LINE_WIDTH  = 0            # widen the line into a band, in pixels
DEFAULT_QR_MARGIN   = 0.15         # quiet zone added around the qr box

# ── Validation / machine config ──────────────────────────────────────────────
DEFAULT_XLSX        = "validation_x300.xlsx"   # expected QR sequence
DEFAULT_START_RELAY = 0                   # relay 0 = winding machine start

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
    ap.add_argument("--conf-thres", type=float, default=DEFAULT_CONF_THRES,
                     help="confidence threshold for classes without their own.")
    ap.add_argument("--conf-label", type=float, default=DEFAULT_CONF_LABEL,
                     help="confidence threshold for the label class — this is "
                          "what gates a trigger-line crossing.")
    ap.add_argument("--conf-qr", type=float, default=DEFAULT_CONF_QR,
                     help="confidence threshold for the qr class — this is what "
                          "gates which box gets cropped and decoded.")
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--save", default=None, help="optional path to record annotated video")
    # qr decode args
    ap.add_argument("--no-qr", action="store_true",
                     help="disable the center-line QR decoding.")
    ap.add_argument("--label-class", default=DEFAULT_LABEL_CLASS,
                     help="class name that triggers a decode when it crosses the line.")
    ap.add_argument("--qr-class", default=DEFAULT_QR_CLASS,
                     help="class name of the QR box that gets cropped and decoded.")
    ap.add_argument("--line-pos", type=float, default=DEFAULT_LINE_POS,
                     help="vertical trigger line as a fraction of frame width (0-1).")
    ap.add_argument("--line-width", type=int, default=DEFAULT_LINE_WIDTH,
                     help="widen the trigger line into a band, this many pixels "
                          "either side — keeps an angled row of labels together.")
    ap.add_argument("--qr-margin", type=float, default=DEFAULT_QR_MARGIN,
                     help="quiet zone around the qr box, as a fraction of its size.")
    ap.add_argument("--qr-margin-min", type=int, default=8,
                     help="minimum quiet zone in pixels.")
    ap.add_argument("--dump-crops", default=None,
                     help="directory to save qr crops that failed to decode.")
    # validation args
    ap.add_argument("--xlsx", default=DEFAULT_XLSX,
                     help="xlsx holding the expected QR DATA1..N sequence.")
    ap.add_argument("--sheet", default=None,
                     help="worksheet name inside --xlsx (default: the first one).")
    ap.add_argument("--no-validate", action="store_true",
                     help="decode only, don't check against the xlsx.")
    ap.add_argument("--labels-per-row", type=int, default=None,
                     help="labels per crossing (default: the number of QR DATA "
                          "columns found in the sheet).")
    ap.add_argument("--label-order", default=TOP_DOWN, choices=[TOP_DOWN, BOTTOM_UP],
                     help="does the topmost label on screen hold QR DATA1 "
                          "(top-down) or the last column (bottom-up)?")
    ap.add_argument("--no-resync", action="store_true",
                     help="don't jump the cursor to a crossing that arrives out "
                          "of sequence (every later row then fails too).")
    ap.add_argument("--stop-on-mismatch", action="store_true",
                     help="switch the machine relay off as soon as a crossing "
                          "fails validation.")
    # relay args
    ap.add_argument("--no-relay", action="store_true",
                     help="don't touch the relay board (vision only).")
    ap.add_argument("--relay-port", default=None,
                     help="serial port of the relay board (default: auto-detect).")
    ap.add_argument("--start-relay", type=int, default=DEFAULT_START_RELAY,
                     help="relay that starts the winding machine.")
    ap.add_argument("--relay-verbose", action="store_true",
                     help="print every modbus frame sent to the relay board.")
    args = ap.parse_args()

    # ── expected sequence + machine start ────────────────────────────────
    validator = None
    if not args.no_validate:
        sheet = ValidationSheet(args.xlsx, args.sheet)
        validator = SequenceValidator(sheet, per_row=args.labels_per_row,
                                      order=args.label_order,
                                      resync=not args.no_resync)

    relay = RelayController(port=args.relay_port, enabled=not args.no_relay,
                            verbose=args.relay_verbose)
    machine_running = False

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

    # Per-class thresholds: the label one gates line crossings, the qr one
    # gates which box gets cropped and decoded, and --conf-thres is what every
    # other class is held to. Class ids are only resolved when something
    # actually needs them, so --no-qr runs stay quiet.
    label_cls = qr_cls = None
    conf_per_class = {}
    if not args.no_qr or args.conf_label is not None or args.conf_qr is not None:
        label_cls = class_index(class_names, args.label_class, 0)
        qr_cls = class_index(class_names, args.qr_class, 1)
        if args.conf_label is not None:
            conf_per_class[label_cls] = args.conf_label
        if args.conf_qr is not None:
            conf_per_class[qr_cls] = args.conf_qr
    print(f"[model] conf thresholds: default={args.conf_thres}"
          + "".join(f"  {args.label_class if c == label_cls else args.qr_class}={v}"
                    for c, v in conf_per_class.items()))

    def on_decode(text, index):
        """Per label, the moment it reads. Only the payload's own identity is
        reported here — which row and column of the sheet it is. Whether it is
        in the *right* place can't be known until the whole crossing is in, so
        that is left to the batch verdict."""
        print(f"\n[qr] read: {text}")
        if validator is None:
            return None
        where = validator.peek(text)
        if where is None:
            print("[qr]   not in the sheet")
            return "not in sheet", False
        row, col = where
        print(f"[qr]   sheet row {row}, QR DATA{col}")
        return f"row {row} D{col}", True

    def on_batch(texts):
        """Once per crossing, with the payloads in top-to-bottom slot order."""
        nonlocal machine_running
        if validator is None:
            return
        result = validator.check_batch(texts)
        validator.report(result)
        if (result is not None and args.stop_on_mismatch
                and result.anchored and not result.ok):
            print("[relay] mismatch — stopping the machine")
            relay.off(args.start_relay)
            machine_running = False

    decoder = None
    if not args.no_qr:
        decoder = CenterLineQRDecoder(
            line_x=int(disp_w * args.line_pos),
            label_cls=label_cls,
            qr_cls=qr_cls,
            margin=args.qr_margin,
            min_px=args.qr_margin_min,
            on_decode=on_decode,
            on_batch=on_batch,
            dump_dir=args.dump_crops,
            half_width=args.line_width,
            expect=validator.per_row if validator is not None else args.labels_per_row,
        )
        print(f"[qr] trigger line at x={decoder.line_x} "
              f"({args.label_class} crossing -> decode {args.qr_class})")

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
        # The machine starts once everything above is ready, so the very first
        # labels off the winder are already being watched.
        relay.on(args.start_relay)
        machine_running = True
        print(f"[relay] winding machine started (relay {args.start_relay} ON)")

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
            dets = postprocess(raw, ratio, pad, frame.shape, args.conf_thres,
                               conf_per_class)

            # ── center-line QR decode ────────────────────────────────────
            # Every label crossing the line is decoded once — labels are kept
            # apart by the y-centre of their box, and the set is cleared when
            # the line goes clear again (no tracker involved).
            if decoder is not None:
                decoder.update(frame, dets)

            frame = draw_detections(frame, dets, class_names)
            if decoder is not None:
                decoder.draw(frame)
            if validator is not None:
                validator.draw(frame)

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
                val_status = (f"  ok={validator.batches_ok} fail={validator.batches_bad}"
                              if validator is not None else "")
                print(f"[camera] frame {frame.shape}  fps={fps:.1f}  "
                      f"dets={len(dets)}{qr_status}{val_status}", end="\r")
            else:
                cv2.putText(frame, f"FPS: {fps:.1f}  dets: {len(dets)}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow(win_name, frame)

            if writer:
                writer.write(frame)

            if not args.no_display and (cv2.waitKey(1) & 0xFF == ord("q")):
                break
    finally:
        if machine_running:
            print(f"\n[relay] stopping the machine (relay {args.start_relay} OFF)")
            relay.off(args.start_relay)
        relay.close()
        cap.release()
        if writer:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        if validator is not None:
            print(f"[validate] done: {validator.batches_ok} rows passed, "
                  f"{validator.batches_bad} failed "
                  f"({validator.labels_ok} codes ok, {validator.labels_bad} bad)")


if __name__ == "__main__":
    main()
