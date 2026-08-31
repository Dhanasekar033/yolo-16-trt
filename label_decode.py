#!/usr/bin/env python3
"""
Read every label in the frame: zxing-cpp first, zbar only on what it drops.

One decoder is not enough for this line. zxing-cpp reads about 98% of ordinary
crops and is the fastest of the libraries available, so it stays the first
pass — but there are codes on this reel it cannot read at all, at any scale,
rotated or inverted. libzbar reads those. Measured over 79 such crops, zbar
read 79, with no false decodes.

So: every label goes to zxing. Only the ones zxing comes back empty on are
handed to zbar, and only those. On a normal frame the fallback never runs, and
the pair costs about the same as zxing alone.

Every label visible in the frame is decoded on every frame, and the payload is
drawn next to the box it came from. Boxes are coloured by which library read
them, so the rescues are visible at a glance:

    green   zxing read it
    amber   zxing failed, zbar rescued it
    red     neither could read it

No trigger line, no rolling window, no validation, no relay — this reads and
shows, nothing else.

Usage:
    python3 label_decode.py
    python3 label_decode.py --no-fallback        # zxing alone, to see the gap
    python3 label_decode.py --no-display         # headless, prints payloads
    python3 label_decode.py --dump-crops bad/    # save what neither could read
"""

import argparse
import os
import time

import cv2

from utils.qr import decode_qr, decode_qr_pyzbar, expand_box
from utils.trt_engine import YOLO26TRT
from utils.utils import preprocess, postprocess

# ── Stream config ────────────────────────────────────────────────────────────
DEFAULT_CAM_INDEX = 0
DEFAULT_WIDTH     = 2592
DEFAULT_HEIGHT    = 1944
DEFAULT_FPS       = 60
DEFAULT_FORMAT    = "MJPG"
DISPLAY_MAX_W     = 1280
DISPLAY_MAX_H     = 960
DEFAULT_ROTATE    = 270

# ── Inference config ─────────────────────────────────────────────────────────
DEFAULT_IMGSZ      = 640
DEFAULT_CONF_THRES = 0.25

# ── Decode config ────────────────────────────────────────────────────────────
DEFAULT_LABEL_CLASS   = "label"   # the class whose box gets cropped and decoded
DEFAULT_QR_MARGIN     = 0.0       # the label box already carries the quiet zone
DEFAULT_QR_MARGIN_MIN = 0

ZXING  = (80, 220, 80)      # green
ZBAR   = (60, 200, 255)     # amber
FAIL   = (60, 60, 235)      # red
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)

ROTATE_MAP = {
    0:   None,
    90:  cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}
VIDEOFLIP_MAP = {0: None, 90: "clockwise", 180: "rotate-180",
                 270: "counterclockwise"}

GS_CAMERA_NAME = "Global Shutter Camera"


def rotate_frame(frame, degrees):
    """Only used when the rotation could not be pushed into the pipeline."""
    code = ROTATE_MAP[degrees]
    return frame if code is None else cv2.rotate(frame, code)


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
            cams.append((int(d.replace("video", "")), f.read().strip()))
    return cams


def find_camera_index(name_substring=GS_CAMERA_NAME, default=DEFAULT_CAM_INDEX):
    for index, name in list_cameras():
        if name_substring.lower() in name.lower():
            return index
    print(f"[camera] '{name_substring}' not found — falling back to {default}")
    return default


def gstreamer_pipeline(cam_index=DEFAULT_CAM_INDEX, width=DEFAULT_WIDTH,
                       height=DEFAULT_HEIGHT, fps=DEFAULT_FPS,
                       format=DEFAULT_FORMAT, rotate=0):
    """v4l2src pipeline. videoflip rotates on the GStreamer thread, which
    keeps a full 2592x1944 rotation off the capture loop."""
    QUEUE = "queue leaky=downstream max-size-buffers=1"
    method = VIDEOFLIP_MAP.get(rotate)
    FLIP = f"videoflip method={method} ! " if method else ""
    SINK = (f"{FLIP}videoconvert ! video/x-raw, format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false")
    if format.upper() == "MJPG":
        return (f"v4l2src device=/dev/video{cam_index} ! "
                f"image/jpeg, width={width}, height={height}, framerate={fps}/1 ! "
                f"{QUEUE} ! jpegdec ! {SINK}")
    return (f"v4l2src device=/dev/video{cam_index} ! "
            f"video/x-raw, width={width}, height={height}, framerate={fps}/1 ! "
            f"{QUEUE} ! {SINK}")


def load_class_names(path):
    if not path:
        return None
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def class_index(class_names, name, fallback):
    if class_names and name in class_names:
        return class_names.index(name)
    print(f"[decode] class '{name}' not in --classes, using index {fallback}")
    return fallback


def tail(text, n=16):
    """Payloads on this reel differ only at the end, so show the end."""
    text = str(text or "")
    return text if len(text) <= n else text[-n:]


def draw_result(frame, box, text, who, font_scale):
    """Outline one label and write what it decoded to underneath it."""
    x1, y1, x2, y2 = (int(v) for v in box[:4])
    colour = ZXING if who == "zxing" else ZBAR if who == "zbar" else FAIL
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)

    label = tail(text) if text else "NO READ"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                  font_scale, 2)
    ty = y2 + th + 12
    if ty > frame.shape[0] - 4:            # runs off the bottom: put it inside
        ty = y1 + th + 12
    cv2.rectangle(frame, (x1, ty - th - 8), (x1 + tw + 10, ty + 6), colour, -1)
    cv2.putText(frame, label, (x1 + 5, ty), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, BLACK, 2, cv2.LINE_AA)


def draw_panel(frame, rows, header, stats, font_scale):
    """Counts at the top, then what each label in this frame said."""
    cv2.putText(frame, header, (20, 100), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 1.3, WHITE, 3, cv2.LINE_AA)
    step = int(38 * font_scale / 0.8)
    y = 150
    for name, colour in (("zxing", ZXING), ("zbar rescued", ZBAR),
                         ("no read", FAIL)):
        key = name.split()[0]
        cv2.putText(frame, f"{name:<14}{stats.get(key, 0):>7}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, colour, 2,
                    cv2.LINE_AA)
        y += step
    y += step // 2
    for text, who in rows:
        colour = ZXING if who == "zxing" else ZBAR if who == "zbar" else FAIL
        cv2.putText(frame, f"{who[:5]:<6} {tail(text, 24) if text else 'NO READ'}",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, colour,
                    2, cv2.LINE_AA)
        y += step


def main():
    ap = argparse.ArgumentParser(
        description="Decode every label crop with zxing-cpp, falling back to "
                    "zbar only on the labels zxing could not read.")
    # camera
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--format", default=DEFAULT_FORMAT, choices=["MJPG", "YUYV"])
    ap.add_argument("--rotate", type=int, default=DEFAULT_ROTATE,
                    choices=[0, 90, 180, 270])
    ap.add_argument("--no-display", action="store_true",
                    help="print payloads instead of opening a window.")
    # model
    ap.add_argument("--engine", default="best.engine")
    ap.add_argument("--classes", default=None,
                    help="txt file, one class per line (default: classes.txt "
                         "beside this script, if present).")
    ap.add_argument("--conf-thres", type=float, default=DEFAULT_CONF_THRES)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    # decode
    ap.add_argument("--label-class", default=DEFAULT_LABEL_CLASS,
                    help="class whose box gets cropped and decoded.")
    ap.add_argument("--no-fallback", action="store_true",
                    help="zxing only, no zbar. Useful for seeing what the "
                         "fallback is actually rescuing.")
    ap.add_argument("--qr-margin", type=float, default=DEFAULT_QR_MARGIN,
                    help="quiet zone added round the box, as a fraction.")
    ap.add_argument("--qr-margin-min", type=int, default=DEFAULT_QR_MARGIN_MIN,
                    help="minimum quiet zone in pixels.")
    ap.add_argument("--dump-crops", default=None,
                    help="directory to save the crops neither library read.")
    ap.add_argument("--text-scale", type=float, default=0.8,
                    help="font size for the overlay.")
    args = ap.parse_args()

    cam_index = args.index if args.index is not None else find_camera_index()
    print(f"[camera] using /dev/video{cam_index}")
    pipeline = gstreamer_pipeline(cam_index, args.width, args.height, args.fps,
                                  args.format, rotate=args.rotate)
    print(f"[camera] pipeline: {pipeline}")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    rotate_in_loop = 0
    if not cap.isOpened() and args.rotate:
        print("[camera] videoflip pipeline would not open — rotating in the loop")
        pipeline = gstreamer_pipeline(cam_index, args.width, args.height,
                                      args.fps, args.format)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        rotate_in_loop = args.rotate
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera via GStreamer pipeline")

    classes_path = args.classes
    if classes_path is None:
        beside = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "classes.txt")
        if os.path.exists(beside):
            classes_path = beside
            print(f"[model] using {beside}")
    class_names = load_class_names(classes_path)
    label_cls = class_index(class_names, args.label_class, 0)
    model = YOLO26TRT(args.engine, input_size=(args.imgsz, args.imgsz))
    print(f"[model] loaded {args.engine}")
    print("[decode] zxing-cpp on every '%s'%s" %
          (args.label_class,
           "" if args.no_fallback else ", zbar on the ones it drops"))

    if args.dump_crops:
        os.makedirs(args.dump_crops, exist_ok=True)

    disp_w, disp_h = ((args.height, args.width) if args.rotate in (90, 270)
                      else (args.width, args.height))
    win_name = "Label decode - zxing + zbar fallback"
    scale = 1.0
    if not args.no_display:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        scale = min(DISPLAY_MAX_W / disp_w, DISPLAY_MAX_H / disp_h, 1.0)
        cv2.resizeWindow(win_name, int(disp_w * scale), int(disp_h * scale))

    seen = set()                       # every distinct payload read since start
    rescued = set()                    # payloads only zbar ever managed
    stats = {"zxing": 0, "zbar": 0, "no": 0}
    t_zxing = t_zbar = 0.0
    n_zbar_calls = 0
    prev_t, fps = time.time(), 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[camera] frame grab failed, retrying...")
                continue
            if rotate_in_loop:
                frame = rotate_frame(frame, rotate_in_loop)

            inp, ratio, pad = preprocess(frame, model.input_size)
            dets = postprocess(model.infer(inp), ratio, pad, frame.shape,
                               args.conf_thres)
            labels = [d for d in dets if int(d[5]) == label_cls]

            rows = []
            for det in labels:
                who = "fail"
                t0 = time.perf_counter()
                text, _ = decode_qr(frame, det[:4], args.qr_margin,
                                    args.qr_margin_min)
                t_zxing += (time.perf_counter() - t0) * 1000
                if text:
                    who = "zxing"
                    stats["zxing"] += 1
                elif not args.no_fallback:
                    # Only this label, and only because zxing came back empty.
                    t0 = time.perf_counter()
                    text, _ = decode_qr_pyzbar(frame, det[:4], args.qr_margin,
                                               args.qr_margin_min)
                    t_zbar += (time.perf_counter() - t0) * 1000
                    n_zbar_calls += 1
                    if text:
                        who = "zbar"
                        stats["zbar"] += 1
                        rescued.add(text)

                if text:
                    seen.add(text)
                else:
                    stats["no"] += 1
                    if args.dump_crops:
                        x1, y1, x2, y2 = expand_box(det[:4], frame.shape,
                                                    args.qr_margin,
                                                    args.qr_margin_min)
                        if x2 - x1 > 4 and y2 - y1 > 4:
                            cv2.imwrite(os.path.join(
                                args.dump_crops,
                                f"noread-{time.strftime('%H%M%S')}-"
                                f"{int(time.time() * 1000) % 1000:03d}.jpg"),
                                frame[y1:y2, x1:x2])
                rows.append((text, who))
                if not args.no_display:
                    draw_result(frame, det, text, who, args.text_scale)

            now = time.time()
            dt, prev_t = now - prev_t, now
            if dt > 0:
                inst = 1.0 / dt
                fps = inst if fps == 0.0 else 0.9 * fps + 0.1 * inst

            read = sum(1 for t, _ in rows if t)
            header = (f"FPS {fps:.0f}   labels {len(labels)}   "
                      f"read {read}/{len(labels)}   unique {len(seen)}")
            if args.no_display:
                print(f"[decode] {header}   zxing={stats['zxing']} "
                      f"zbar={stats['zbar']} fail={stats['no']}   ", end="\r")
            else:
                draw_panel(frame, rows, header, stats, args.text_scale)
                small = (frame if scale >= 1.0 else
                         cv2.resize(frame, None, fx=scale, fy=scale,
                                    interpolation=cv2.INTER_NEAREST))
                cv2.imshow(win_name, small)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        total = stats["zxing"] + stats["zbar"] + stats["no"]
        print(f"\n[decode] {total} label crops seen")
        print(f"           zxing read     {stats['zxing']:>7}"
              f"   {t_zxing / max(total, 1):6.2f} ms per crop")
        if not args.no_fallback:
            print(f"           zbar rescued   {stats['zbar']:>7}"
                  f"   {t_zbar / max(n_zbar_calls, 1):6.2f} ms per call, "
                  f"{n_zbar_calls} calls")
        print(f"           neither read   {stats['no']:>7}")
        print(f"\n[decode] {len(seen)} distinct payloads read")
        for t in sorted(seen):
            print(f"           {t}")
        if rescued:
            print(f"\n[decode] {len(rescued)} payload(s) zbar rescued at least once:")
            for t in sorted(rescued):
                print(f"           {t}")


if __name__ == "__main__":
    main()
