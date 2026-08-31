#!/usr/bin/env python3
"""
Decode every LABEL crop in the frame and show what each one says.

The sibling of all_qr_test.py, with two things changed:

  * it crops the `label` box, not the `qr_code` box — the whole label carries
    the QR's printed quiet zone with it, and a label still reads when the
    tighter qr_code box was never detected;
  * the decoder is a chain you choose on the command line, so several
    libraries can be compared against the same labels, live.

No trigger line, no window, no validation, no relay. Every label visible in
the frame is cropped and decoded on every frame, and the payload is drawn
next to the box it came from — so you can point the camera at the web and see
directly which labels read, which do not, and which library read them.

--decoder takes a comma-separated chain, tried left to right until one reads:

    zxing        zxing-cpp, the three-pass ladder in utils/qr.py
    pyzbar       libzbar, via pyzbar
    wechat       cv2.wechat_qrcode, classical detector (no model files)
    wechat-cnn   the same with the CNN + super-resolution models (--wechat-models)
    opencv       cv2.QRCodeDetector

Measured on 79 crops off this line that zxing-cpp could not read at all, and
60 that it reads fine:

    strategy          hard    easy    ms/crop
    zxing               0%     98%      1.23
    pyzbar            100%     95%      2.35
    wechat             97%    100%      1.76
    wechat-cnn         96%    100%      4.88
    opencv             53%     68%     13.16
    zxing,pyzbar      100%    100%      0.47   <- the default

The chain is cheap because the first decoder handles nearly everything and
the rest only run on the crops it drops.

Usage:
    python3 label_qr_test.py                          # zxing,pyzbar
    python3 label_qr_test.py --decoder zxing          # one library alone
    python3 label_qr_test.py --decoder wechat,pyzbar
    python3 label_qr_test.py --decoder zxing,wechat,pyzbar,opencv
    python3 label_qr_test.py --list-decoders          # what is installed
    python3 label_qr_test.py --no-display             # headless, prints payloads
    python3 label_qr_test.py --dump-crops bad/        # save what would not read
"""

import argparse
import os
import time

import cv2

from utils.qr import decode_qr, decode_qr_opencv, expand_box
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
DEFAULT_LABEL_CLASS = "label"  # the class whose box gets cropped and decoded
DEFAULT_QR_MARGIN   = 0.0    # the label box already carries the quiet zone
DEFAULT_QR_MARGIN_MIN = 0
DEFAULT_CHAIN = "zxing,pyzbar"
# Where the WeChat CNN models live, if they are anywhere. Only needed for
# the 'wechat-cnn' decoder; the classical one takes no files.
DEFAULT_WECHAT_MODELS = None

WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
RED    = (60, 60, 235)
# One colour per decoder, so a glance at the boxes says which library won.
DECODER_COLOUR = {
    "zxing":      (80, 220, 80),      # green
    "pyzbar":     (60, 200, 255),     # amber
    "wechat":     (235, 130, 220),    # violet
    "wechat-cnn": (235, 180, 90),     # blue
    "opencv":     (235, 170, 60),     # steel
}

ROTATE_MAP = {
    0:   None,
    90:  cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}
VIDEOFLIP_MAP = {0: None, 90: "clockwise", 180: "rotate-180",
                 270: "counterclockwise"}


def rotate_frame(frame, degrees):
    """Only used when the rotation could not be pushed into the pipeline."""
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


# ── decoders ─────────────────────────────────────────────────────────────────
# Each is built lazily and returns text_or_None for one box, so asking for a
# library you have not installed only fails if you actually named it.

def _crop(frame, box, margin, min_px):
    x1, y1, x2, y2 = expand_box(box, frame.shape, margin, min_px)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return frame[y1:y2, x1:x2]


def _build_zxing(_):
    return lambda frame, box, m, p: decode_qr(frame, box, m, p)[0]


def _build_opencv(_):
    return lambda frame, box, m, p: decode_qr_opencv(frame, box, m, p)[0]


def _build_pyzbar(_):
    from pyzbar import pyzbar          # needs libzbar0 on the system

    def read(frame, box, m, p):
        crop = _crop(frame, box, m, p)
        if crop is None:
            return None
        found = pyzbar.decode(crop, symbols=[pyzbar.ZBarSymbol.QRCODE])
        return found[0].data.decode("utf-8", "replace") if found else None
    return read


def _wechat(models):
    """The classical detector takes no files; the CNN one takes four."""
    if models:
        need = ["detect.prototxt", "detect.caffemodel",
                "sr.prototxt", "sr.caffemodel"]
        paths = [os.path.join(models, n) for n in need]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "missing WeChat model files: " + ", ".join(missing))
        det = cv2.wechat_qrcode_WeChatQRCode(*paths)
    else:
        det = cv2.wechat_qrcode_WeChatQRCode()

    def read(frame, box, m, p):
        crop = _crop(frame, box, m, p)
        if crop is None:
            return None
        found, _pts = det.detectAndDecode(crop)
        return found[0] if found else None
    return read


BUILDERS = {
    "zxing":      _build_zxing,
    "pyzbar":     _build_pyzbar,
    "wechat":     lambda models: _wechat(None),
    "wechat-cnn": lambda models: _wechat(models or DEFAULT_WECHAT_MODELS),
    "opencv":     _build_opencv,
}


def build_chain(spec, wechat_models):
    """Turn 'zxing,pyzbar' into [(name, callable), ...]."""
    chain = []
    for name in [n.strip() for n in spec.split(",") if n.strip()]:
        if name not in BUILDERS:
            raise SystemExit(f"unknown decoder '{name}'. "
                             f"choose from: {', '.join(BUILDERS)}")
        try:
            chain.append((name, BUILDERS[name](wechat_models)))
        except Exception as exc:
            raise SystemExit(f"decoder '{name}' is not usable here: {exc}")
    if not chain:
        raise SystemExit("--decoder needs at least one library")
    return chain


def report_available(wechat_models):
    print("decoder      status")
    for name, build in BUILDERS.items():
        try:
            build(wechat_models)
            print(f"  {name:<11} available")
        except Exception as exc:
            print(f"  {name:<11} NOT usable — {str(exc)[:60]}")


# ── overlay ──────────────────────────────────────────────────────────────────
def tail(text, n=16):
    """Payloads on this reel differ only at the end, so show the end."""
    text = str(text or "")
    return text if len(text) <= n else text[-n:]


def draw_result(frame, box, text, who, font_scale):
    """Outline one label and write what it decoded to underneath it."""
    x1, y1, x2, y2 = (int(v) for v in box[:4])
    colour = DECODER_COLOUR.get(who, RED)
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


def draw_panel(frame, rows, header, chain, reads, ms, font_scale):
    """What was decoded this frame, and how each library in the chain is doing."""
    cv2.putText(frame, header, (20, 100), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 1.3, WHITE, 3, cv2.LINE_AA)
    y = 150
    step = int(38 * font_scale / 0.8)
    for name, _ in chain:
        n = reads.get(name, 0)
        avg = (ms.get(name, 0.0) / n) if n else 0.0
        cv2.putText(frame, f"{name:<11} {n:>6}   {avg:5.2f} ms",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    DECODER_COLOUR.get(name, WHITE), 2, cv2.LINE_AA)
        y += step
    y += step // 2
    for text, who in rows:
        cv2.putText(frame,
                    f"{who[:6]:<7} {tail(text, 24) if text else 'NO READ'}",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    DECODER_COLOUR.get(who, RED), 2, cv2.LINE_AA)
        y += step


def main():
    ap = argparse.ArgumentParser(
        description="Decode every label crop in the frame and show what each "
                    "one reads, with a decoder chain you pick.")
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
    ap.add_argument("--decoder", default=DEFAULT_CHAIN,
                    help="comma-separated chain, tried left to right until one "
                         "reads: " + ", ".join(BUILDERS) +
                         f". Default: {DEFAULT_CHAIN}.")
    ap.add_argument("--wechat-models", default=DEFAULT_WECHAT_MODELS,
                    help="directory holding detect.prototxt, detect.caffemodel, "
                         "sr.prototxt and sr.caffemodel, for 'wechat-cnn'.")
    ap.add_argument("--list-decoders", action="store_true",
                    help="print which libraries are usable here, and exit.")
    ap.add_argument("--qr-margin", type=float, default=DEFAULT_QR_MARGIN,
                    help="quiet zone added round the box, as a fraction.")
    ap.add_argument("--qr-margin-min", type=int, default=DEFAULT_QR_MARGIN_MIN,
                    help="minimum quiet zone in pixels.")
    ap.add_argument("--dump-crops", default=None,
                    help="directory to save the crops nothing could decode.")
    ap.add_argument("--text-scale", type=float, default=0.8,
                    help="font size for the overlay.")
    args = ap.parse_args()

    if args.list_decoders:
        report_available(args.wechat_models)
        return

    chain = build_chain(args.decoder, args.wechat_models)

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
    print(f"[decode] every '{args.label_class}' crop, every frame, chain: "
          + " -> ".join(n for n, _ in chain))

    if args.dump_crops:
        os.makedirs(args.dump_crops, exist_ok=True)

    disp_w, disp_h = ((args.height, args.width) if args.rotate in (90, 270)
                      else (args.width, args.height))
    win_name = "Label QR test - " + args.decoder
    scale = 1.0
    if not args.no_display:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        scale = min(DISPLAY_MAX_W / disp_w, DISPLAY_MAX_H / disp_h, 1.0)
        cv2.resizeWindow(win_name, int(disp_w * scale), int(disp_h * scale))

    seen = set()                 # every distinct payload read since start
    reads, ms, tries = {}, {}, {}    # per decoder: reads, total ms, attempts
    n_fail = 0
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
                text, who = None, "fail"
                for name, fn in chain:
                    t0 = time.perf_counter()
                    try:
                        text = fn(frame, det[:4], args.qr_margin,
                                  args.qr_margin_min)
                    except Exception:
                        text = None
                    ms[name] = ms.get(name, 0.0) + \
                        (time.perf_counter() - t0) * 1000
                    tries[name] = tries.get(name, 0) + 1
                    if text:
                        who = name
                        reads[name] = reads.get(name, 0) + 1
                        break

                if text:
                    seen.add(text)
                else:
                    n_fail += 1
                    if args.dump_crops:
                        crop = _crop(frame, det[:4], args.qr_margin,
                                     args.qr_margin_min)
                        if crop is not None and crop.size:
                            cv2.imwrite(os.path.join(
                                args.dump_crops,
                                f"noread-{time.strftime('%H%M%S')}-"
                                f"{int(time.time() * 1000) % 1000:03d}.jpg"),
                                crop)
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
                per = "  ".join(f"{n}={reads.get(n, 0)}" for n, _ in chain)
                print(f"[decode] {header}   {per}  fail={n_fail}   ", end="\r")
            else:
                # the panel counts attempts, not reads: what a library costs
                # you is what it costs on every crop it is handed
                draw_panel(frame, rows, header, chain,
                           {n: tries.get(n, 0) for n, _ in chain},
                           {n: ms.get(n, 0.0) for n, _ in chain},
                           args.text_scale)
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
        print(f"\n[decode] chain: " + " -> ".join(n for n, _ in chain))
        print(f"{'decoder':<12}{'attempts':>10}{'reads':>8}{'hit rate':>10}"
              f"{'ms/attempt':>12}")
        for name, _ in chain:
            a = tries.get(name, 0)
            r = reads.get(name, 0)
            avg = ms.get(name, 0.0) / a if a else 0.0
            print(f"{name:<12}{a:>10}{r:>8}{(100*r/a if a else 0):>9.0f}%"
                  f"{avg:>12.2f}")
        print(f"\n[decode] {len(seen)} distinct payloads read, "
              f"{n_fail} crops nothing could read")
        for t in sorted(seen):
            print(f"           {t}")


if __name__ == "__main__":
    main()
