#!/usr/bin/env python3
"""
One box round the whole label, built from the two parts the model finds.

best-2.engine was trained on the dataset with the `label` class stripped out,
so it finds the code and the artifact and nothing else — two small boxes per
label where what you want to cut out is the label they both sit on. This puts
that box back: the parts are paired up, one code to one artifact, and the box
is drawn round the pair.

WHICH TWO PARTS BELONG TOGETHER is the whole problem, and it is not solved by
taking the nearest artifact to each code. On this reel the labels sit in a
grid about 640px apart with the artifact ~350px below its own code, so a code
whose artifact was missed has a neighbour's artifact well within reach, and
pairing it there draws a box across two labels. So pairs are made one-to-one,
cheapest first, and then held against the median offset of the frame's own
pairs -- a pair that does not sit like the others is not a pair.

A part left over is still a label: its partner is put where the other pairs
say it would be and the box is drawn round that, in a different colour so an
inferred box is never mistaken for a measured one. --no-singles drops them.

The box itself is the union of the two parts plus --pad, which takes in the
serial number printed alongside them and nothing else -- the shape you get
when you crop what the model actually found. --gutter is the other answer:
widen instead into the gap between the labels, half of whatever room is
measured there, which reaches the edges of the card without ever taking in
the card next door.

Usage:
    python3 cam-trt-labelbox.py --engine best-2.engine --source clip.mp4
    python3 cam-trt-labelbox.py --engine best-2.engine            # live camera
    python3 cam-trt-labelbox.py --engine best-2.engine --source clip.mp4 --show-parts
    python3 cam-trt-labelbox.py --engine best-2.engine --source clip.mp4 --save out.mp4
    python3 cam-trt-labelbox.py --engine best-2.engine --source clip.mp4 --save-crops crops/
    python3 cam-trt-labelbox.py --engine best-2.engine --pad 0.1  # fixed margin instead
    python3 cam-trt-labelbox.py --engine best-2.engine --no-display --frames 30

Keys (window mode):
    q / Esc   quit          space  pause / resume
    n         step one frame while paused
    s         save this frame and its label crops, losslessly

`s` writes the picture the camera gave, not the one on the screen: the boxes
are drawn onto a copy, so nothing that was drawn can end up in a saved crop.
And it writes it in a format that keeps the pixels exactly as they were --
PNG by default, BMP for literally no compression at all. A crop saved as JPEG
has had its detail smoothed by the encoder, which is the one thing you cannot
undo later and the whole reason for the key.
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
DEFAULT_WIDTH     = 1920
DEFAULT_HEIGHT    = 1200     # the 2MP global shutter's full frame
DEFAULT_FPS       = 60
DEFAULT_FORMAT    = "MJPG"
DISPLAY_MAX_W     = 1280
DISPLAY_MAX_H     = 960
DEFAULT_ROTATE    = 270      # camera only; a recorded file is already rotated

# ── Inference config ─────────────────────────────────────────────────────────
DEFAULT_IMGSZ      = 640
DEFAULT_CONF_THRES = 0.25

# The two parts, by the names the stripped dataset uses and the names it used
# before the rename. Looked up in classes.txt when there is one, so an engine
# built either side of that rename works without a flag.
CODE_NAMES     = ("code", "qr_code", "qr")
ARTIFACT_NAMES = ("artifact", "logo")

# Pairing. Both are fractions, so nothing here is tied to this camera or this
# working distance -- a lens change or a different reel re-measures itself.
MIN_SIDE_OVERLAP = 0.35   # of the narrower part, across the pairing axis
MAX_OFFSET_RATIO = 1.8    # how far a pair's offset may stray from the median

LABEL_COLOR    = (0, 200, 255)    # measured pair: amber
INFERRED_COLOR = (255, 120, 0)    # partner inferred: blue
CODE_COLOR     = (0, 255, 0)
ARTIFACT_COLOR = (255, 255, 0)

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg")

ROTATE_MAP = {
    0:   None,
    90:  cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

GS_CAMERA_NAME = "Global Shutter Camera"


def rotate_frame(frame, degrees):
    code = ROTATE_MAP[degrees]
    return frame if code is None else cv2.rotate(frame, code)


def list_cameras():
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
    print(f"[camera] '{name_substring}' not found among {list_cameras()} — "
          f"falling back to index {default}")
    return default


def gstreamer_pipeline(cam_index, width, height, fps, fmt):
    QUEUE = "queue leaky=downstream max-size-buffers=1"
    SINK  = ("videoconvert ! video/x-raw, format=BGR ! "
             "appsink drop=true max-buffers=1 sync=false")
    if fmt.upper() == "MJPG":
        return (f"v4l2src device=/dev/video{cam_index} ! "
                f"image/jpeg, width={width}, height={height}, framerate={fps}/1 ! "
                f"{QUEUE} ! jpegdec ! {SINK}")
    return (f"v4l2src device=/dev/video{cam_index} ! "
            f"video/x-raw, width={width}, height={height}, framerate={fps}/1 ! "
            f"{QUEUE} ! {SINK}")


def open_source(args):
    """(cap, label, kind) — a video file, a plain camera index, or the rig."""
    if args.source:
        if args.source.isdigit():
            return (cv2.VideoCapture(int(args.source)),
                    f"camera index {args.source}", "camera")
        if not os.path.exists(args.source):
            raise SystemExit(f"[source] {args.source} does not exist")
        if not args.source.lower().endswith(VIDEO_EXT):
            print(f"[source] {args.source} is not a known video extension — "
                  f"trying to open it anyway")
        return cv2.VideoCapture(args.source), args.source, "file"

    index = args.index if args.index is not None else find_camera_index()
    pipeline = gstreamer_pipeline(index, args.width, args.height,
                                  args.fps, args.format)
    print(f"[camera] using /dev/video{index}")
    print(f"[camera] pipeline: {pipeline}")
    return (cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER),
            f"/dev/video{index}", "camera")


def load_class_names(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def class_index(names, wanted, fallback):
    """The index of the first of `wanted` that classes.txt has.

    `wanted` carries a class's spellings either side of the dataset rename, so
    an engine that says qr_code and one that says code both resolve. With no
    classes.txt there is nothing to match on and the fixed index stands."""
    if names:
        for name in wanted:
            if name in names:
                return names.index(name)
        raise SystemExit(f"[model] none of {wanted} is in classes.txt "
                         f"({', '.join(names)}) — pass --classes for this "
                         f"engine, or drop it to use the fixed indices")
    return fallback


# ── pairing ──────────────────────────────────────────────────────────────────

def _overlap_1d(a1, a2, b1, b2):
    return max(0.0, min(a2, b2) - max(a1, b1))


def _centre(box):
    return (float(box[0]) + float(box[2])) / 2, (float(box[1]) + float(box[3])) / 2


def pair_parts(codes, artifacts, axis="auto"):
    """[(code, artifact), ...] and whatever was left over.

    One code to one artifact, cheapest first: a greedy pass over every
    candidate sorted by centre distance, each part taken once. On six labels
    a frame this is exact and costs nothing, and it cannot do what
    nearest-neighbour does -- give two codes the same artifact.

    Candidates must overlap across the pairing axis. The parts of one label
    are stacked along the web, so they share a lane; a neighbour's part sits
    beside it and shares nothing.
    """
    if not codes or not artifacts:
        return [], list(codes), list(artifacts)

    # Which way the labels are stacked, measured rather than assumed: the axis
    # a part's own partner lies along is the one with the LARGER spread of
    # centre offsets. A camera turned 90 degrees swaps them.
    if axis == "auto":
        cx = np.median([_centre(c)[0] for c in codes]) - \
             np.median([_centre(a)[0] for a in artifacts])
        cy = np.median([_centre(c)[1] for c in codes]) - \
             np.median([_centre(a)[1] for a in artifacts])
        axis = "y" if abs(cy) >= abs(cx) else "x"

    cands = []
    for i, c in enumerate(codes):
        for j, a in enumerate(artifacts):
            if axis == "y":
                over = _overlap_1d(c[0], c[2], a[0], a[2])
                span = min(float(c[2]) - float(c[0]), float(a[2]) - float(a[0]))
            else:
                over = _overlap_1d(c[1], c[3], a[1], a[3])
                span = min(float(c[3]) - float(c[1]), float(a[3]) - float(a[1]))
            if span <= 0 or over / span < MIN_SIDE_OVERLAP:
                continue
            (ccx, ccy), (acx, acy) = _centre(c), _centre(a)
            cands.append((abs(acx - ccx) + abs(acy - ccy), i, j))

    cands.sort()
    took_c, took_a, pairs = set(), set(), []
    for _cost, i, j in cands:
        if i in took_c or j in took_a:
            continue
        took_c.add(i)
        took_a.add(j)
        pairs.append((i, j))

    # A pair that does not sit like the rest of them is not a pair: it is a
    # code that lost its artifact reaching across to the next label's. The
    # frame's own median offset is the yardstick, so nothing here is a number
    # anybody had to measure at the machine.
    if len(pairs) >= 3:
        offs = []
        for i, j in pairs:
            (ccx, ccy), (acx, acy) = _centre(codes[i]), _centre(artifacts[j])
            offs.append((acx - ccx, acy - ccy))
        med = (float(np.median([o[0] for o in offs])),
               float(np.median([o[1] for o in offs])))
        reach = max(np.hypot(*med), 1.0) * MAX_OFFSET_RATIO
        kept = []
        for (i, j), off in zip(pairs, offs):
            if np.hypot(off[0] - med[0], off[1] - med[1]) <= reach:
                kept.append((i, j))
            else:
                took_c.discard(i)
                took_a.discard(j)
        pairs = kept

    return ([(codes[i], artifacts[j]) for i, j in pairs],
            [c for i, c in enumerate(codes) if i not in took_c],
            [a for j, a in enumerate(artifacts) if j not in took_a])


def median_offset(pairs):
    """Where an artifact sits relative to its code, on this frame."""
    if not pairs:
        return None
    dx, dy = [], []
    for c, a in pairs:
        (ccx, ccy), (acx, acy) = _centre(c), _centre(a)
        dx.append(acx - ccx)
        dy.append(acy - ccy)
    return float(np.median(dx)), float(np.median(dy))


def union(*boxes):
    return (min(float(b[0]) for b in boxes), min(float(b[1]) for b in boxes),
            max(float(b[2]) for b in boxes), max(float(b[3]) for b in boxes))


def shifted(box, dx, dy):
    return (float(box[0]) + dx, float(box[1]) + dy,
            float(box[2]) + dx, float(box[3]) + dy)


# ── the box round the label ──────────────────────────────────────────────────

def gutter_pad(box, others, share, cap):
    """Room to grow on each side: half the clear gap to the nearest label.

    The same rule the inspector cuts its crops by. The gap is measured off the
    other label boxes rather than assumed, so a reel with a different pitch, a
    lens moved, a different camera, all measure themselves -- and a crop can
    never take in the neighbour it measured against, because it only ever
    takes half of what is between them.
    """
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    gaps = {"left": None, "right": None, "up": None, "down": None}

    def keep(side, gap):
        if gap >= 0 and (gaps[side] is None or gap < gaps[side]):
            gaps[side] = gap

    for other in others:
        ox1, oy1, ox2, oy2 = (float(v) for v in other[:4])
        if (ox1, oy1, ox2, oy2) == (x1, y1, x2, y2):
            continue
        if _overlap_1d(y1, y2, oy1, oy2) > 0:
            if ox2 <= x1:
                keep("left", x1 - ox2)
            elif ox1 >= x2:
                keep("right", ox1 - x2)
        if _overlap_1d(x1, x2, ox1, ox2) > 0:
            if oy2 <= y1:
                keep("up", y1 - oy2)
            elif oy1 >= y2:
                keep("down", oy1 - y2)

    # A side with nothing standing on it borrows its opposite's measure: the
    # label at the end of a row is the same label as the rest of them.
    for a, b in (("left", "right"), ("up", "down")):
        if gaps[a] is None:
            gaps[a] = gaps[b]
        if gaps[b] is None:
            gaps[b] = gaps[a]

    w, h = x2 - x1, y2 - y1
    pad = {}
    for side in ("left", "right"):
        pad[side] = 0 if gaps[side] is None else min(gaps[side] * share, w * cap)
    for side in ("up", "down"):
        pad[side] = 0 if gaps[side] is None else min(gaps[side] * share, h * cap)
    return pad


def label_box(parts_box, others, shape, args):
    """The final box: the union of the parts, widened, clipped to the frame."""
    x1, y1, x2, y2 = (float(v) for v in parts_box)
    h, w = shape[:2]
    if args.gutter:
        pad = gutter_pad(parts_box, others, args.gutter_share, args.gutter_cap)
    else:
        px = max((x2 - x1) * args.pad, args.pad_px)
        py = max((y2 - y1) * args.pad, args.pad_px)
        pad = {"left": px, "right": px, "up": py, "down": py}
    return (int(max(x1 - pad["left"], 0)), int(max(y1 - pad["up"], 0)),
            int(min(x2 + pad["right"], w)), int(min(y2 + pad["down"], h)))


def find_labels(dets, code_cls, art_cls, shape, args):
    """[(box, measured), ...] — one entry per label found in this frame."""
    codes = [d[:4] for d in dets if int(d[5]) == code_cls]
    arts  = [d[:4] for d in dets if int(d[5]) == art_cls]
    pairs, lone_codes, lone_arts = pair_parts(codes, arts, args.axis)

    unions = [(union(c, a), True) for c, a in pairs]
    if not args.no_singles:
        off = median_offset(pairs)
        if off is not None:
            dx, dy = off
            # The missing partner goes where the frame's own pairs say it is.
            unions += [(union(c, shifted(c, dx, dy)), False) for c in lone_codes]
            unions += [(union(a, shifted(a, -dx, -dy)), False) for a in lone_arts]

    others = [u for u, _ in unions]
    return [(label_box(u, others, shape, args), measured) for u, measured in unions]


def draw(frame, labels, dets, code_cls, art_cls, args):
    if args.show_parts:
        for d in dets:
            colour = CODE_COLOR if int(d[5]) == code_cls else ARTIFACT_COLOR
            cv2.rectangle(frame, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])),
                          colour, 1)
    for n, (box, measured) in enumerate(labels, 1):
        colour = LABEL_COLOR if measured else INFERRED_COLOR
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), colour, 3)
        text = f"label {n}" if measured else f"label {n} (part missing)"
        cv2.putText(frame, text, (box[0], max(18, box[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
    return frame


# Params per format. PNG at compression 1 rather than the default 3: both are
# bit-exact, and on a full frame the cheap one costs a few ms instead of tens.
# BMP takes no params because there is nothing to tune -- it is the raw bytes.
SHOT_PARAMS = {
    "png":  [cv2.IMWRITE_PNG_COMPRESSION, 1],
    "bmp":  [],
    "tiff": [cv2.IMWRITE_TIFF_COMPRESSION, 1],      # 1 = none
    "jpg":  [cv2.IMWRITE_JPEG_QUALITY, 100,
             cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444],
}


def save_shot(frame, labels, out_dir, ext, n):
    """Write the clean frame and one file per label box. Returns (paths, bytes).

    `frame` must be the frame BEFORE anything was drawn on it -- a crop with a
    box outline burnt into it is no longer a picture of the label."""
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() % 1 * 1000):03d}"
    params = SHOT_PARAMS.get(ext, [])
    written, size = [], 0

    path = os.path.join(out_dir, f"shot_{stamp}_f{n:06d}_frame.{ext}")
    if cv2.imwrite(path, frame, params):
        written.append(path)
        size += os.path.getsize(path)

    for i, (box, _measured) in enumerate(labels, 1):
        crop = frame[box[1]:box[3], box[0]:box[2]]
        if crop.size == 0:
            continue
        path = os.path.join(out_dir, f"shot_{stamp}_f{n:06d}_label{i:02d}.{ext}")
        if cv2.imwrite(path, crop, params):
            written.append(path)
            size += os.path.getsize(path)
    return written, size


def main():
    ap = argparse.ArgumentParser(
        description="Box the whole label from the code and artifact the model finds.")
    # source
    ap.add_argument("--source", default=None,
                     help="video file to run on instead of the camera "
                          "(or a bare digit for a plain webcam index)")
    ap.add_argument("--loop", action="store_true", help="restart the file at the end")
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames")
    ap.add_argument("--index", type=int, default=None, help="force /dev/videoN")
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--format", default=DEFAULT_FORMAT, choices=["MJPG", "YUYV"])
    ap.add_argument("--rotate", type=int, default=None, choices=[0, 90, 180, 270],
                     help=f"default {DEFAULT_ROTATE} for the camera, 0 for --source")
    ap.add_argument("--no-display", action="store_true")
    # model
    ap.add_argument("--engine", default="best-2.engine", help="path to .engine")
    ap.add_argument("--classes", default=None,
                     help="class names, one per line, to find the two parts by "
                          "name rather than at the fixed indices 0 and 1. It "
                          "must be the list THIS engine was built against: "
                          "the repo's own classes.txt still has three classes "
                          "and would put the parts at 1 and 2, which is right "
                          "for best.engine and wrong for best-2.engine")
    ap.add_argument("--conf-thres", type=float, default=DEFAULT_CONF_THRES)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    # the box
    ap.add_argument("--axis", choices=["auto", "x", "y"], default="auto",
                     help="which way a label's two parts are stacked. Measured "
                          "off the frame by default")
    ap.add_argument("--pad", type=float, default=0.06,
                     help="margin round the union of the two parts, as a "
                          "fraction of its size. This is the default shape of "
                          "the box: the two parts and the serial number "
                          "between them, and none of the card around it")
    ap.add_argument("--pad-px", type=int, default=0,
                     help="least margin in pixels, whatever --pad works out to")
    ap.add_argument("--gutter", action="store_true",
                     help="take the whole card instead: widen into the gap "
                          "between the labels by --gutter-share of whatever "
                          "room is measured there, rather than by --pad")
    ap.add_argument("--gutter-share", type=float, default=0.5,
                     help="how much of the gap to the next label a box may "
                          "take. Half is the most that can never overlap")
    ap.add_argument("--gutter-cap", type=float, default=0.6,
                     help="ceiling on that, as a fraction of the box's own "
                          "size, for the label with nothing beside it")
    ap.add_argument("--no-singles", action="store_true",
                     help="drop a label whose second part was not detected "
                          "instead of inferring where it would be")
    ap.add_argument("--show-parts", action="store_true",
                     help="also draw the code and artifact boxes")
    # output
    ap.add_argument("--save", default=None, help="record the annotated video")
    ap.add_argument("--save-crops", default=None,
                     help="write every label box to this folder as an image")
    ap.add_argument("--crop-format", default="png", choices=["png", "jpg"],
                     help="png keeps the pixels exactly as the frame had them")
    ap.add_argument("--shot-dir", default="shots",
                     help="where the s key writes the frame and its crops")
    ap.add_argument("--shot-format", default="png",
                     choices=["png", "bmp", "tiff", "jpg"],
                     help="png and tiff and bmp are all lossless -- png is "
                          "compressed and smallest, bmp is the raw bytes and "
                          "fastest to write. jpg is here for when size beats "
                          "fidelity, at quality 100 and 4:4:4 chroma")
    args = ap.parse_args()

    cap, label, kind = open_source(args)
    if not cap.isOpened():
        raise SystemExit(
            f"[source] could not open {label}"
            + (" — is another process holding the camera?" if kind == "camera"
               else " — unsupported codec, or the file is unreadable"))

    if kind == "file":
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
        src_fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        src_w, src_h, src_fps, total = args.width, args.height, float(args.fps), 0
    rotate = args.rotate if args.rotate is not None else (
        0 if kind == "file" else DEFAULT_ROTATE)
    print(f"[source] {label} — {src_w}x{src_h} @ {src_fps:.0f} fps"
          + (f", {total} frames" if total > 0 else "") + f", rotate {rotate}")

    names = load_class_names(args.classes)
    code_cls = class_index(names, CODE_NAMES, 0)
    art_cls = class_index(names, ARTIFACT_NAMES, 1)
    print(f"[model] parts: {code_cls}={names[code_cls] if names else 'code'}, "
          f"{art_cls}={names[art_cls] if names else 'artifact'}"
          + (f"  (from {args.classes})" if names else "  (fixed indices)"))
    model = YOLO26TRT(args.engine, input_size=(args.imgsz, args.imgsz))
    print(f"[model] loaded {args.engine}")

    disp_w, disp_h = (src_h, src_w) if rotate in (90, 270) else (src_w, src_h)

    writer = None
    if args.save:
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                  src_fps, (disp_w, disp_h))
        if not writer.isOpened():
            raise SystemExit(f"[save] could not open {args.save} for writing")
    if args.save_crops:
        os.makedirs(args.save_crops, exist_ok=True)
        print(f"[crops] label crops -> {args.save_crops}/")

    win_name = "label boxes — " + label
    if not args.no_display:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        scale = min(DISPLAY_MAX_W / disp_w, DISPLAY_MAX_H / disp_h, 1.0)
        cv2.resizeWindow(win_name, int(disp_w * scale), int(disp_h * scale))

    prev_t = time.time()
    fps = 0.0
    n = found = inferred = crops = misses = shots = 0
    paused = step = False
    shown = None
    labels = []
    note = ["", 0.0]         # what the s key just did, and when
    was_visible = False
    try:
        while True:
            if not paused or step:
                step = False
                ok, frame = cap.read()
                if not ok:
                    if kind == "file":
                        if args.loop and n:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        break
                    misses += 1
                    if misses > 50:
                        print("\n[camera] 50 failed grabs in a row — giving up")
                        break
                    continue
                misses = 0
                n += 1

                frame = rotate_frame(frame, rotate)
                inp, ratio, pad = preprocess(frame, model.input_size)
                dets = postprocess(model.infer(inp), ratio, pad, frame.shape,
                                   args.conf_thres)
                labels = find_labels(dets, code_cls, art_cls, frame.shape, args)
                found += len(labels)
                inferred += sum(1 for _, measured in labels if not measured)

                if args.save_crops:
                    for i, (box, _measured) in enumerate(labels, 1):
                        crop = frame[box[1]:box[3], box[0]:box[2]]
                        if crop.size == 0:
                            continue
                        path = os.path.join(
                            args.save_crops,
                            f"frame{n:06d}_label{i:02d}.{args.crop_format}")
                        if cv2.imwrite(path, crop):
                            crops += 1

                # The boxes go onto a COPY. `frame` stays as the camera gave
                # it, which is what the crops are cut from and what the s key
                # writes -- an outline burnt into a saved crop cannot be taken
                # out again. Nothing is copied or drawn at all when no one is
                # going to look at it.
                annotated = frame
                if not args.no_display or writer:
                    annotated = draw(frame.copy(), labels, dets,
                                     code_cls, art_cls, args)

                now = time.time()
                dt, prev_t = now - prev_t, now
                if dt > 0:
                    inst = 1.0 / dt
                    fps = inst if fps == 0.0 else (0.9 * fps + 0.1 * inst)

                progress = f"{n}/{total}" if total > 0 else str(n)
                if args.no_display:
                    print(f"[run] frame {progress}  fps={fps:.1f}  "
                          f"labels={len(labels)}", end="\r")
                else:
                    cv2.putText(annotated, f"{progress}  FPS: {fps:.1f}  "
                                f"labels: {len(labels)}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (0, 255, 0), 2, cv2.LINE_AA)
                    if time.time() - note[1] < 2.0:
                        cv2.putText(annotated, note[0], (20, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                    LABEL_COLOR, 2, cv2.LINE_AA)
                shown = annotated
                if writer:
                    writer.write(annotated)
                if args.frames and n >= args.frames:
                    break

            if not args.no_display:
                if shown is not None:
                    cv2.imshow(win_name, shown)
                key = cv2.waitKey(30 if paused else 1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    paused = not paused
                    print(f"\n[run] {'paused' if paused else 'resumed'} at frame {n}")
                elif key == ord("n"):
                    step = True
                elif key == ord("s"):
                    # Works paused too: `frame` and `labels` are the last ones
                    # read, which is exactly the picture on the screen.
                    paths, size = save_shot(frame, labels, args.shot_dir,
                                            args.shot_format, n)
                    shots += len(paths)
                    note[0] = (f"saved {len(paths)} {args.shot_format} file(s), "
                               f"{size / 1e6:.1f} MB")
                    note[1] = time.time()
                    print(f"\n[shot] frame {n}: {len(paths)} file(s) "
                          f"({size / 1e6:.1f} MB) -> {args.shot_dir}/")
                visible = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) >= 1
                was_visible = was_visible or visible
                if was_visible and not visible:
                    break
    finally:
        cap.release()
        if writer:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    if n:
        print(f"\n[run] {n} frame(s), {found} label box(es), "
              f"{found / n:.1f} per frame"
              + (f", {inferred} with a part inferred" if inferred else ""))
        if args.save:
            print(f"[save] wrote {args.save}")
        if args.save_crops:
            print(f"[crops] wrote {crops} crop(s) to {args.save_crops}/")
        if shots:
            print(f"[shot] wrote {shots} file(s) to {args.shot_dir}/")
    else:
        print("\n[run] no frames read")


if __name__ == "__main__":
    main()
