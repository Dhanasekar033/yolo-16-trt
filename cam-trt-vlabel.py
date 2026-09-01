#!/usr/bin/env python3
"""
Global Shutter Camera viewer + live YOLO26 TensorRT inference, narrowed to
the vertical labels and the red card each one sits on.

Same GStreamer/OpenCV capture pipeline and staleness-guard setup as
cam-trt.py, with three changes:

  * only the "label" class is kept — qr_code and logo boxes are dropped
    before anything is drawn;
  * of those labels only the vertical ones survive. The horizontal
    "DO NOT ACCEPT IF TAMPERED" strip at the top of every card is the same
    class as the tall label below it and is told apart purely by shape:
    a box is vertical when height/width >= --min-aspect;
  * for each vertical label the red card around it is measured and boxed,
    off a red mask in HSV (both hue wrap-around ranges). The card's red
    frame is the one red blob that *encloses* the label, which is what
    picks it out from the red tamper band printed above the label and from
    a neighbouring card's frame alike — neither of those wraps this label.
    When glare or JPEG noise breaks the frame into pieces so that nothing
    encloses the label, it falls back to walking outward from the label
    box until a column/row is mostly red.

No ultralytics import anywhere — inference goes through trt_engine.py.

Usage:
    python3 cam-trt-vlabel.py --engine best.engine
    python3 cam-trt-vlabel.py --engine best.engine --min-aspect 1.3
    python3 cam-trt-vlabel.py --engine best.engine --show-mask   # tune the red HSV range
    python3 cam-trt-vlabel.py --engine best.engine --no-red      # labels only
    python3 cam-trt-vlabel.py --engine best.engine --red-method scan   # force the fallback
    python3 cam-trt-vlabel.py --engine best.engine --no-display  # headless, prints counts
    python3 cam-trt-vlabel.py --engine best.engine --save out.mp4
"""

import argparse
import os
import time

import cv2
import numpy as np

from utils.trt_engine import YOLO26TRT
from utils.utils import preprocess, postprocess

# ── Stream config (same defaults as cam-trt.py) ─────────────────────────────
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
DEFAULT_LABEL_CLS  = 0       # "label" is the first line of classes.txt
DEFAULT_MIN_ASPECT = 1.20    # height/width at or above which a label is vertical

# ── Red card config ──────────────────────────────────────────────────────────
# Red straddles the hue wrap-around, so the mask is two inRange calls OR'd.
RED_HUE_LOW_MAX  = 10
RED_HUE_HIGH_MIN = 170
DEFAULT_SAT_MIN  = 80        # keeps grey card stock and shadows out of the mask
DEFAULT_VAL_MIN  = 50        # keeps near-black print out of the mask
# How far out of the label box to look for the card's red frame, as a fraction
# of the label's own width/height. The whole card has to fit inside this window
# for the enclosing-blob method to see it, so these are deliberately loose —
# a neighbouring card falling inside the window is harmless, it does not
# enclose this label.
DEFAULT_MARGIN_X = 1.00
DEFAULT_MARGIN_Y = 0.90
# A blob has to reach past every edge of the label to count as its card frame.
# The slack absorbs a card sitting slightly skewed and a label box drawn a
# pixel or two proud of the print.
ENCLOSE_SLACK_FRAC = 0.03
ENCLOSE_SLACK_MIN  = 3
MIN_BLOB_AREA_FRAC = 0.01   # of the label's area; drops speckle from the mask
# Two cards printed hard against each other merge into one red blob, which
# would enclose both labels. A blob wider or taller than this many label
# widths/heights is that merge, not a card, and is handed to the fallback scan.
DEFAULT_MAX_CARD_SCALE = 2.2
# Fallback scan only: a scan line counts as "on the red frame" at this red
# fraction, and the run is followed outward until the fraction drops below
# EDGE_FALLOFF times it.
DEFAULT_EDGE_FRAC = 0.40
EDGE_FALLOFF      = 0.5

LABEL_COLOR = (0, 255, 0)     # vertical label box
RED_COLOR   = (0, 0, 255)     # measured red card box

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


# ── vertical label filtering ────────────────────────────────────────────────

def vertical_labels(dets, label_cls=DEFAULT_LABEL_CLS, min_aspect=DEFAULT_MIN_ASPECT):
    """Keep only label-class boxes that are taller than they are wide.

    dets is the (N, 6) x1,y1,x2,y2,conf,cls array from postprocess(). The
    horizontal tamper strip and the tall label carry the same class id, so
    aspect ratio is the only thing separating them."""
    if dets.shape[0] == 0:
        return dets

    keep = dets[dets[:, 5].astype(np.int32) == label_cls]
    if keep.shape[0] == 0:
        return keep

    widths  = np.maximum(keep[:, 2] - keep[:, 0], 1e-6)
    heights = keep[:, 3] - keep[:, 1]
    return keep[(heights / widths) >= min_aspect]


# ── red card measurement ────────────────────────────────────────────────────

def red_mask(bgr, sat_min=DEFAULT_SAT_MIN, val_min=DEFAULT_VAL_MIN):
    """Binary mask of red pixels, covering both ends of the hue circle."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    low  = cv2.inRange(hsv, (0, sat_min, val_min), (RED_HUE_LOW_MAX, 255, 255))
    high = cv2.inRange(hsv, (RED_HUE_HIGH_MIN, sat_min, val_min), (180, 255, 255))
    mask = cv2.bitwise_or(low, high)
    # Close over print texture and the gaps JPEG artefacts leave in the frame.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _enclosing_blob(mask, label_box, max_scale=DEFAULT_MAX_CARD_SCALE):
    """Bounding box of the smallest red blob that wraps the whole label box.

    The card's red frame is a closed ring around the label, so it is the only
    red in the ROI that reaches past all four label edges: the tamper band
    printed above the label sits inside it, and a neighbouring card's frame
    is off to one side. Where several blobs qualify — a frame plus, say, a red
    border printed further out — the tightest one is the card itself.

    mask is 0/255; label_box is (x1, y1, x2, y2) in mask coords. Returns
    (x1, y1, x2, y2) or None."""
    lx1, ly1, lx2, ly2 = label_box
    lw, lh = max(lx2 - lx1, 1), max(ly2 - ly1, 1)
    slack = max(ENCLOSE_SLACK_MIN, int(ENCLOSE_SLACK_FRAC * max(lw, lh)))
    min_area = MIN_BLOB_AREA_FRAC * lw * lh
    max_w, max_h = lw * max_scale, lh * max_scale

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    best, best_area = None, None
    for i in range(1, count):          # 0 is the background
        x, y, w, h, area = stats[i]
        if area < min_area or w > max_w or h > max_h:
            continue
        if not (x <= lx1 + slack and y <= ly1 + slack
                and x + w >= lx2 - slack and y + h >= ly2 - slack):
            continue
        if best_area is None or w * h < best_area:
            best, best_area = (x, y, x + w - 1, y + h - 1), w * h
    return best


def _scan_out(profile, start, step, edge_frac):
    """Walk a 1-D red-fraction profile outward from `start` and return the far
    side of the first red run met, or None if the scan leaves the ROI first.

    `step` is -1 to walk toward index 0 and +1 to walk toward the end."""
    n = len(profile)
    i = start
    while 0 <= i < n and profile[i] < edge_frac:      # gap between label and frame
        i += step
    if not (0 <= i < n):
        return None
    falloff = edge_frac * EDGE_FALLOFF
    while 0 <= i + step < n and profile[i + step] >= falloff:   # across the frame
        i += step
    return i


def red_box_for_label(frame, box, margin_x=DEFAULT_MARGIN_X, margin_y=DEFAULT_MARGIN_Y,
                       edge_frac=DEFAULT_EDGE_FRAC, sat_min=DEFAULT_SAT_MIN,
                       val_min=DEFAULT_VAL_MIN, method="auto",
                       max_scale=DEFAULT_MAX_CARD_SCALE):
    """Box the red card carrying one vertical label.

    Takes the smallest red blob enclosing the label; with method="scan", or
    when no blob encloses it, walks outward from the label edges instead.
    That fallback stops at the first red it meets on each side, so red printed
    between the label and the card edge — the tamper band above the label —
    cuts the box short there.

    Returns (x1, y1, x2, y2) in frame coords, or None when no red was found on
    any side of the label. Under the fallback a side that finds no red keeps
    the label's own edge, so a card clipped by the frame border still gets a
    box off the sides that are visible."""
    h0, w0 = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box[:4])
    bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)

    rx1 = max(0,  x1 - int(bw * margin_x))
    ry1 = max(0,  y1 - int(bh * margin_y))
    rx2 = min(w0, x2 + int(bw * margin_x))
    ry2 = min(h0, y2 + int(bh * margin_y))
    if rx2 - rx1 < 2 or ry2 - ry1 < 2:
        return None

    mask = red_mask(frame[ry1:ry2, rx1:rx2], sat_min, val_min)

    # Label box in ROI coords, clamped so the profiles below are non-empty.
    lx1, ly1 = np.clip([x1 - rx1, y1 - ry1], 0, [mask.shape[1] - 1, mask.shape[0] - 1])
    lx2, ly2 = np.clip([x2 - rx1, y2 - ry1], 1, [mask.shape[1], mask.shape[0]])

    if method != "scan":
        blob = _enclosing_blob(mask, (lx1, ly1, lx2, ly2), max_scale)
        if blob is not None:
            return (rx1 + blob[0], ry1 + blob[1], rx1 + blob[2], ry1 + blob[3])
        if method == "blob":
            return None

    mask = mask > 0
    # Sideways scans read columns over the label's rows; vertical scans read
    # rows over the label's columns. Either way the card's frame band spans the
    # whole slice, so it shows up as a near-1.0 fraction.
    col_frac = mask[ly1:ly2, :].mean(axis=0)
    row_frac = mask[:, lx1:lx2].mean(axis=1)

    left   = _scan_out(col_frac, lx1, -1, edge_frac)
    right  = _scan_out(col_frac, lx2 - 1, +1, edge_frac)
    top    = _scan_out(row_frac, ly1, -1, edge_frac)
    bottom = _scan_out(row_frac, ly2 - 1, +1, edge_frac)
    if left is None and right is None and top is None and bottom is None:
        return None

    return (rx1 + (left if left is not None else lx1),
            ry1 + (top if top is not None else ly1),
            rx1 + (right if right is not None else lx2 - 1),
            ry1 + (bottom if bottom is not None else ly2 - 1))


def draw_label_and_card(img, box, conf, card):
    """Draw one vertical label (green) and its red card box (red)."""
    x1, y1, x2, y2 = (int(v) for v in box[:4])
    if card is not None:
        cv2.rectangle(img, (card[0], card[1]), (card[2], card[3]), RED_COLOR, 3)
    cv2.rectangle(img, (x1, y1), (x2, y2), LABEL_COLOR, 2)

    text = f"label {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), LABEL_COLOR, -1)
    cv2.putText(img, text, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img


def main():
    ap = argparse.ArgumentParser(
        description="Stream the Global Shutter Camera and box vertical labels + their red cards.")
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
                     help="Just print FPS/counts instead of opening a window (headless).")
    # inference args
    ap.add_argument("--engine", default="best.engine", help="path to .engine file")
    ap.add_argument("--conf-thres", type=float, default=DEFAULT_CONF_THRES)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--label-class", type=int, default=DEFAULT_LABEL_CLS,
                     help="class id of 'label' in classes.txt")
    ap.add_argument("--min-aspect", type=float, default=DEFAULT_MIN_ASPECT,
                     help="height/width at or above which a label counts as vertical")
    ap.add_argument("--save", default=None, help="optional path to record annotated video")
    # red card args
    ap.add_argument("--no-red", action="store_true",
                     help="skip the red card boxes, draw the vertical labels only")
    ap.add_argument("--red-method", choices=["auto", "blob", "scan"], default="auto",
                     help="auto: enclosing red blob, falling back to the outward scan; "
                          "blob/scan: force one of the two")
    ap.add_argument("--max-card-scale", type=float, default=DEFAULT_MAX_CARD_SCALE,
                     help="largest card, in label widths/heights, the enclosing blob may be")
    ap.add_argument("--margin-x", type=float, default=DEFAULT_MARGIN_X,
                     help="sideways red search range, in label widths")
    ap.add_argument("--margin-y", type=float, default=DEFAULT_MARGIN_Y,
                     help="vertical red search range, in label heights")
    ap.add_argument("--edge-frac", type=float, default=DEFAULT_EDGE_FRAC,
                     help="fallback scan only: red fraction of a scan line that counts "
                          "as the card's frame")
    ap.add_argument("--sat-min", type=int, default=DEFAULT_SAT_MIN,
                     help="minimum HSV saturation for a pixel to count as red")
    ap.add_argument("--val-min", type=int, default=DEFAULT_VAL_MIN,
                     help="minimum HSV value for a pixel to count as red")
    ap.add_argument("--show-mask", action="store_true",
                     help="open a second window with the whole-frame red mask (tuning aid)")
    args = ap.parse_args()

    cam_index = args.index if args.index is not None else find_camera_index()
    pipeline = gstreamer_pipeline(cam_index, args.width, args.height, args.fps, args.format)
    print(f"[camera] using /dev/video{cam_index}")
    print(f"[camera] pipeline: {pipeline}")

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera via GStreamer pipeline")

    model = YOLO26TRT(args.engine, input_size=(args.imgsz, args.imgsz))
    print(f"[model] loaded {args.engine}")

    # 90/270 rotation swaps the effective width/height for sizing the window/writer.
    disp_w, disp_h = (args.height, args.width) if args.rotate in (90, 270) else (args.width, args.height)

    writer = None
    if args.save:
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                  args.fps, (disp_w, disp_h))

    win_name = "Global Shutter Camera - vertical labels + red cards"
    mask_win = "red mask"
    if not args.no_display:
        # WINDOW_NORMAL makes the window resizable; without it, imshow opens
        # at the frame's native resolution which overflows most screens. We
        # set an initial on-screen size, capped to DISPLAY_MAX_*, while the
        # capture/inference itself still runs at full resolution.
        scale = min(DISPLAY_MAX_W / disp_w, DISPLAY_MAX_H / disp_h, 1.0)
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, int(disp_w * scale), int(disp_h * scale))
        if args.show_mask:
            cv2.namedWindow(mask_win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(mask_win, int(disp_w * scale), int(disp_h * scale))

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
            labels = vertical_labels(dets, args.label_class, args.min_aspect)

            # The mask window is built off the untouched frame, so it has to be
            # made before any box is drawn on top.
            mask_view = None
            if args.show_mask and not args.no_display:
                mask_view = red_mask(frame, args.sat_min, args.val_min)

            cards = 0
            for det in labels:
                card = None
                if not args.no_red:
                    card = red_box_for_label(frame, det, args.margin_x, args.margin_y,
                                              args.edge_frac, args.sat_min, args.val_min,
                                              args.red_method, args.max_card_scale)
                    cards += card is not None
                draw_label_and_card(frame, det, det[4], card)

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
                print(f"[camera] frame {frame.shape}  fps={fps:.1f}  "
                      f"vertical labels={len(labels)}  red cards={cards}", end="\r")
            else:
                cv2.putText(frame, f"FPS: {fps:.1f}  labels: {len(labels)}  cards: {cards}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow(win_name, frame)
                if mask_view is not None:
                    cv2.imshow(mask_win, mask_view)

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
