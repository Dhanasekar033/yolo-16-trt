#!/usr/bin/env python3
"""
Capture a red-label detection dataset off the Global Shutter Camera.

Runs the same pipeline as cam-trt-vlabel.py — YOLO26 TensorRT finds the
vertical labels, the red card around each one is measured from a red mask —
but instead of only drawing the result it lets you bank it. Press **s** and
the frame is written to disk with three annotated classes:

    label     the RED CARD box, not the model's own label box. The old model
              boxes only the tall white label; the red card measured around it
              is what the next model should learn, so the red geometry is what
              gets written under the label class.
    qr_code   the box zxing-cpp reports for each symbol it actually DECODES —
              not the model's qr_code box. zxing returns the four corners of
              the symbol it read, and their extent is the annotation. Because
              the formats mask covers DataMatrix as well as QR, the Data
              Matrix codes on the card are picked up by the same pass and
              annotated under this class too. The decoded text and the symbol
              format ride along in the COCO annotation.
    logo      the model's logo boxes, passed through as detected

Only symbols zxing can read become annotations, so a code that is blurred or
glared out is left unboxed rather than guessed at. zxing is the slow part of
the frame, so the preview rescans only every --qr-every frames; a save always
rescans the exact frame being written, so the annotations match its pixels.

Every press of s writes the frame twice by default: once straight, once turned
180 degrees, with the same classes on both. The detector only finds these
labels the right way up — inferring on an upside-down frame returns nothing —
so the upside-down copy cannot be captured directly. Its boxes are the
straight frame's, mapped through the image centre, which is exact for 180
degrees. That is what puts upside-down cards in the training set. --variants
picks one or the other if you do not want both.

The horizontal tamper strip is still dropped — only vertical labels get a red
card, and only red cards become label annotations.

The detection half is imported from cam-trt-vlabel.py rather than copied, so
whatever you tune there (--sat-min, --margin-y, --max-card-scale …) is exactly
what gets baked into the dataset.

Keys in the preview window:
    s   save the current frame + its boxes (straight and upside-down)
    u   undo the last save (drops every image it wrote, and its annotations)
    q   quit

Output, under --out (default dataset/):
    images/frame_000001.jpg        the frame as shot
    images/frame_000001_r180.jpg   the same frame upside down
    annotations.json          COCO: one file, all images and boxes. Each image
                              carries `capture` (which press of s wrote it) and
                              `variant` (straight / upside_down).
    labels/frame_000001.txt   YOLO: one file per image (--format yolo/both)
    classes.txt               label / qr_code / logo, in class-index order

Re-running appends to an existing dataset instead of overwriting it.

Usage:
    python3 make_dataset.py --engine best.engine
    python3 make_dataset.py --engine best.engine --out dataset/run2
    python3 make_dataset.py --engine best.engine --format both
    python3 make_dataset.py --engine best.engine --format yolo --image-format png
    python3 make_dataset.py --engine best.engine --allow-empty   # keep negatives too
    python3 make_dataset.py --engine best.engine --class-names card,qr,mark
    python3 make_dataset.py --engine best.engine --qr-scan cards   # scan inside each card
    python3 make_dataset.py --engine best.engine --barcode-formats QRCode,DataMatrix
    python3 make_dataset.py --engine best.engine --variants straight   # no rotated copy
"""

import argparse
import datetime as dt
import importlib.util
import json
import os
import sys
import time
from collections import namedtuple
from pathlib import Path

import cv2
import zxingcpp

# cam-trt-vlabel.py has hyphens in its name, so it cannot be imported by the
# normal statement — load it by path out of this script's own directory.
def _load_sibling(filename):
    path = Path(__file__).resolve().parent / filename
    if not path.exists():
        sys.exit(f"[dataset] {filename} not found next to {Path(__file__).name}")
    spec = importlib.util.spec_from_file_location("vlabel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V = _load_sibling("cam-trt-vlabel.py")

DEFAULT_OUT     = "dataset"
DEFAULT_QUALITY = 95
FLASH_SECONDS   = 0.6        # how long the on-screen save confirmation lingers

# Class indices as written to the dataset. Index 0 is the red card, so a model
# trained on this set keeps the same class order as the old classes.txt.
LABEL_IDX, QR_IDX, LOGO_IDX = 0, 1, 2
DEFAULT_CLASS_NAMES = ["label", "qr_code", "logo"]

# Preview colours, BGR. The label/card pair is drawn by cam-trt-vlabel.py in
# green and red; these two have to stay clear of both.
QR_COLOR   = (255, 200, 0)   # cyan-blue
LOGO_COLOR = (0, 255, 255)   # yellow
BOX_COLORS = {QR_IDX: QR_COLOR, LOGO_IDX: LOGO_COLOR}

# DataMatrix sits alongside QR here because the cards carry both and zxing
# reads them in the same pass.
DEFAULT_BARCODE_FORMATS = "QRCode,MicroQRCode,DataMatrix"
DEFAULT_QR_EVERY = 5         # preview rescan interval; a save always rescans
DEFAULT_QR_PAD   =  0.1      # grow each symbol box by this fraction of itself

# One annotation on its way to disk. `meta` is merged into the COCO annotation
# and ignored by YOLO, which has nowhere to put it.
Box = namedtuple("Box", "cls xyxy meta")

ROT_SUFFIX = "_r180"         # marks the upside-down copy of a capture


# ── zxing symbol detection ──────────────────────────────────────────────────

def barcode_formats(names):
    """Turn 'QRCode,DataMatrix' into the list zxing wants for `formats`."""
    formats = []
    for name in names:
        fmt = getattr(zxingcpp.BarcodeFormat, name, None)
        if fmt is None:
            sys.exit(f"[zxing] unknown barcode format {name!r}. Pick from: "
                     f"QRCode, MicroQRCode, RMQRCode, DataMatrix, Aztec, PDF417, Code128 …")
        formats.append(fmt)
    return formats


def scan_symbols(image, formats, offset=(0, 0), pad=DEFAULT_QR_PAD):
    """Decode every symbol in `image` and return one Box per successful read.

    The annotation is the extent of the four corners zxing reports for the
    symbol it decoded, which is the symbol proper — the quiet zone around it
    is not included. `offset` shifts the boxes back into full-frame coords
    when `image` is a crop."""
    try:
        results = zxingcpp.read_barcodes(image, formats=formats, try_rotate=True,
                                          try_downscale=True, try_invert=True)
    except Exception as exc:                      # a bad frame must not kill the run
        print(f"\n[zxing] read failed: {exc}")
        return []

    ox, oy = offset
    h, w = image.shape[:2]
    boxes = []
    for r in results:
        if not r.valid or not r.text:
            continue
        p = r.position
        corners = (p.top_left, p.top_right, p.bottom_right, p.bottom_left)
        xs = [c.x for c in corners]
        ys = [c.y for c in corners]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        if pad:
            gx, gy = int((x2 - x1) * pad), int((y2 - y1) * pad)
            x1, y1, x2, y2 = x1 - gx, y1 - gy, x2 + gx, y2 + gy
        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        boxes.append(Box(QR_IDX, (x1 + ox, y1 + oy, x2 + ox, y2 + oy),
                          {"text": r.text, "symbol_format": r.format.name}))
    return boxes


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


def dedupe_symbols(boxes, iou_thres=0.5):
    """Drop repeats of the same symbol.

    Card crops overlap where two cards sit close, so a symbol in the overlap
    is decoded once per crop. Same text and a box on the same spot means the
    same physical symbol."""
    kept = []
    for box in boxes:
        if any(other.meta.get("text") == box.meta.get("text")
               and _iou(other.xyxy, box.xyxy) >= iou_thres for other in kept):
            continue
        kept.append(box)
    return kept


def scan_frame_symbols(frame, formats, mode, cards, pad, card_margin):
    """All decoded symbols in a frame, either in one whole-frame pass or one
    pass per red card. Per-card passes hand zxing a small image with the
    symbol large in it, which reads better, but only cover symbols that sit on
    a card whose red box was measured."""
    if mode == "frame":
        return scan_symbols(frame, formats, pad=pad)

    h, w = frame.shape[:2]
    found = []
    for x1, y1, x2, y2 in cards:
        mx = int((x2 - x1) * card_margin)
        my = int((y2 - y1) * card_margin)
        cx1, cy1 = max(0, x1 - mx), max(0, y1 - my)
        cx2, cy2 = min(w, x2 + mx), min(h, y2 + my)
        if cx2 - cx1 < 8 or cy2 - cy1 < 8:
            continue
        found += scan_symbols(frame[cy1:cy2, cx1:cx2], formats, (cx1, cy1), pad)
    return dedupe_symbols(found)


# ── upside-down variant ─────────────────────────────────────────────────────

def rotate180(frame, boxes):
    """Turn a frame upside down and carry its boxes across, classes unchanged.

    This is why the rotation happens here and not at the camera: the detector
    only finds these labels the right way up, so inferring on an upside-down
    frame returns nothing. The boxes have to be measured on the straight frame
    and then mapped, which is exact for 180 degrees — every box keeps its size
    and its corners swap through the image centre:

        (x1, y1, x2, y2) -> (w - x2, h - y2, w - x1, h - y1)

    A decoded QR payload rides along untouched; it is the same physical symbol,
    just photographed the other way up."""
    h, w = frame.shape[:2]
    flipped = [Box(b.cls, (w - b.xyxy[2], h - b.xyxy[3], w - b.xyxy[0], h - b.xyxy[1]),
                    b.meta) for b in boxes]
    return cv2.rotate(frame, cv2.ROTATE_180), flipped


def build_variants(frame, boxes, mode):
    """The (frame, boxes, suffix) list one press of s should write."""
    out = []
    if mode in ("both", "straight"):
        out.append((frame, boxes, ""))
    if mode in ("both", "upside"):
        rframe, rboxes = rotate180(frame, boxes)
        out.append((rframe, rboxes, ROT_SUFFIX))
    return out


# ── durable writes ──────────────────────────────────────────────────────────
# Everything below writes through a temp file, fsync, then rename. A dataset is
# built over hours of camera time and a hard power-off must not be able to
# leave a half-written annotations.json — the same failure that empties git
# object files also truncates plain writes.

def _write_bytes(path, payload):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _write_text(path, text):
    _write_bytes(path, text.encode("utf-8"))


def _fsync_dir(directory):
    fd = os.open(directory, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


# ── dataset on disk ─────────────────────────────────────────────────────────

class DatasetWriter:
    """Images plus label/qr_code/logo boxes, in COCO and/or YOLO layout.

    Boxes arrive as Box(cls, (x1, y1, x2, y2), meta) records. COCO category
    ids are the class index plus one, since COCO numbers categories from 1
    while YOLO numbers classes from 0. Anything in `meta` — the decoded text
    and symbol format for a zxing box — is merged into the COCO annotation;
    YOLO's five numbers have nowhere to carry it.

    COCO is one annotations.json for the whole set — that is the format's
    actual shape, there is no per-image COCO text file. YOLO is one .txt per
    image, class then normalised centre/size. Both are rewritten after every
    save so an interrupted session leaves a loadable dataset."""

    def __init__(self, root, class_names=None, fmt="coco",
                 image_ext="jpg", quality=DEFAULT_QUALITY):
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.labels_dir = self.root / "labels"
        self.json_path = self.root / "annotations.json"
        self.class_names = list(class_names or DEFAULT_CLASS_NAMES)
        self.fmt = fmt
        self.image_ext = image_ext
        self.quality = quality

        self.images_dir.mkdir(parents=True, exist_ok=True)
        if fmt in ("yolo", "both"):
            self.labels_dir.mkdir(parents=True, exist_ok=True)

        self.images, self.annotations = self._resume()
        self.saved_this_run = 0

    def _resume(self):
        """Pick up an existing dataset so a second session appends to it."""
        if not self.json_path.exists():
            return [], []
        try:
            with open(self.json_path) as f:
                doc = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            sys.exit(f"[dataset] {self.json_path} exists but is unreadable ({exc}). "
                     f"Move it aside or point --out somewhere else.")
        images, annotations = doc.get("images", []), doc.get("annotations", [])
        print(f"[dataset] resuming {self.json_path}: "
              f"{len(images)} images, {len(annotations)} boxes")
        return images, annotations

    @property
    def count(self):
        return len(self.images)

    def _next_capture(self):
        """Next capture number, skipping any already on disk.

        One press of s is one capture, which may write more than one image —
        the straight frame and its upside-down copy share a capture number and
        are undone together."""
        n = max((img.get("capture", img["id"]) for img in self.images), default=0) + 1
        while (self.images_dir / f"frame_{n:06d}.{self.image_ext}").exists():
            n += 1
        return n

    def save(self, variants):
        """Write one capture: a list of (frame, boxes, suffix) to store together.

        Each frame must be a clean frame — anything drawn on it would be baked
        into the training image. Returns the file names written, or None if
        nothing could be encoded."""
        capture = self._next_capture()
        next_id = max((i["id"] for i in self.images), default=0) + 1
        next_ann = max((a["id"] for a in self.annotations), default=0) + 1

        params = ([cv2.IMWRITE_JPEG_QUALITY, self.quality]
                  if self.image_ext in ("jpg", "jpeg") else [])
        staged = []
        for frame, boxes, suffix in variants:
            stem = f"frame_{capture:06d}{suffix}"
            name = f"{stem}.{self.image_ext}"
            ok, buf = cv2.imencode(f".{self.image_ext}", frame, params)
            if not ok:
                print(f"\n[dataset] could not encode {name}, capture dropped")
                return None
            staged.append((stem, name, buf.tobytes(), boxes, frame.shape[:2]))

        # Nothing is written until every variant has encoded, so a capture
        # cannot land half in the set.
        written = []
        for k, (stem, name, payload, boxes, (h, w)) in enumerate(staged):
            _write_bytes(self.images_dir / name, payload)
            image_id = next_id + k
            self.images.append({
                "id": image_id,
                "file_name": name,
                "width": w,
                "height": h,
                "capture": capture,
                "variant": "upside_down" if stem.endswith(ROT_SUFFIX) else "straight",
                "date_captured": dt.datetime.now().isoformat(timespec="seconds"),
            })
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy
                bw, bh = x2 - x1, y2 - y1
                self.annotations.append({
                    "id": next_ann,
                    "image_id": image_id,
                    "category_id": box.cls + 1,
                    # COCO bbox is [x, y, width, height] in absolute pixels, from
                    # the top-left corner — not the [x1,y1,x2,y2] the detector uses.
                    "bbox": [int(x1), int(y1), int(bw), int(bh)],
                    "area": int(bw * bh),
                    "iscrowd": 0,
                    "segmentation": [],
                    **(box.meta or {}),
                })
                next_ann += 1
            written.append((stem, boxes, w, h))

        self._flush(written)
        self.saved_this_run += 1
        return [name for _, name, _, _, _ in staged]

    def undo(self):
        """Drop the most recent capture — every image it wrote — on disk and in
        the annotations."""
        if not self.images:
            return None
        capture = max(img.get("capture", img["id"]) for img in self.images)
        dropped = [i for i in self.images if i.get("capture", i["id"]) == capture]
        ids = {i["id"] for i in dropped}
        self.images = [i for i in self.images if i["id"] not in ids]
        self.annotations = [a for a in self.annotations if a["image_id"] not in ids]
        for image in dropped:
            (self.images_dir / image["file_name"]).unlink(missing_ok=True)
            (self.labels_dir / f"{Path(image['file_name']).stem}.txt").unlink(missing_ok=True)
        self._flush([])
        self.saved_this_run = max(0, self.saved_this_run - 1)
        return [i["file_name"] for i in dropped]

    def _flush(self, written):
        """Rewrite annotations.json, plus a YOLO .txt per image just written."""
        if self.fmt in ("coco", "both"):
            _write_text(self.json_path, json.dumps({
                "info": {
                    "description": "label boxes are the red card measured around each "
                                   "vertical label; qr_code boxes are the symbols "
                                   "zxing-cpp decoded (QR and DataMatrix), carrying "
                                   "their text; logo boxes are the detector's own. "
                                   "Images marked variant=upside_down are the straight "
                                   "frame turned 180 degrees, boxes carried across. "
                                   "Written by make_dataset.py",
                    "date_created": dt.datetime.now().isoformat(timespec="seconds"),
                },
                "licenses": [],
                "images": self.images,
                "annotations": self.annotations,
                "categories": [{"id": i + 1, "name": name, "supercategory": "label"}
                                for i, name in enumerate(self.class_names)],
            }, indent=2))
        if self.fmt in ("yolo", "both"):
            for stem, boxes, w, h in written:
                lines = []
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy
                    # YOLO wants class then centre and size, each divided through
                    # by the image dimensions.
                    lines.append("%d %.6f %.6f %.6f %.6f" % (
                        box.cls,
                        ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h,
                        (x2 - x1) / w, (y2 - y1) / h))
                _write_text(self.labels_dir / f"{stem}.txt", "\n".join(lines) + "\n")

    def counts(self):
        """Boxes banked per class name, for the closing summary."""
        tally = {name: 0 for name in self.class_names}
        for a in self.annotations:
            i = a["category_id"] - 1
            if 0 <= i < len(self.class_names):
                tally[self.class_names[i]] += 1
        return tally

    def write_classes(self):
        """classes.txt beside the labels, so the YOLO set is self-describing."""
        if self.fmt in ("yolo", "both"):
            _write_text(self.root / "classes.txt", "\n".join(self.class_names) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Bank frames + red card boxes as a COCO/YOLO dataset.")
    # camera args — same meaning as cam-trt-vlabel.py
    ap.add_argument("--index", type=int, default=None,
                     help="Force a /dev/videoN index (skips auto-detect).")
    ap.add_argument("--width", type=int, default=V.DEFAULT_WIDTH)
    ap.add_argument("--height", type=int, default=V.DEFAULT_HEIGHT)
    ap.add_argument("--fps", type=int, default=V.DEFAULT_FPS)
    ap.add_argument("--format-v4l2", dest="v4l2_format", default=V.DEFAULT_FORMAT,
                     choices=["MJPG", "YUYV"])
    ap.add_argument("--rotate", type=int, default=V.DEFAULT_ROTATE, choices=[0, 90, 180, 270])
    # inference args
    ap.add_argument("--engine", default="best.engine", help="path to .engine file")
    ap.add_argument("--conf-thres", type=float, default=V.DEFAULT_CONF_THRES)
    ap.add_argument("--imgsz", type=int, default=V.DEFAULT_IMGSZ)
    ap.add_argument("--label-class", type=int, default=V.DEFAULT_LABEL_CLS,
                     help="class id of 'label' in the ENGINE's classes.txt")
    ap.add_argument("--logo-class", type=int, default=2,
                     help="class id of 'logo' in the ENGINE's classes.txt")
    ap.add_argument("--min-aspect", type=float, default=V.DEFAULT_MIN_ASPECT)
    # red card args
    ap.add_argument("--red-method", choices=["auto", "blob", "scan"], default="auto")
    ap.add_argument("--max-card-scale", type=float, default=V.DEFAULT_MAX_CARD_SCALE)
    ap.add_argument("--margin-x", type=float, default=V.DEFAULT_MARGIN_X)
    ap.add_argument("--margin-y", type=float, default=V.DEFAULT_MARGIN_Y)
    ap.add_argument("--edge-frac", type=float, default=V.DEFAULT_EDGE_FRAC)
    ap.add_argument("--sat-min", type=int, default=V.DEFAULT_SAT_MIN)
    ap.add_argument("--val-min", type=int, default=V.DEFAULT_VAL_MIN)
    # dataset args
    ap.add_argument("--out", default=DEFAULT_OUT, help="dataset directory")
    ap.add_argument("--format", choices=["coco", "yolo", "both"], default="coco",
                     help="coco: one annotations.json; yolo: one .txt per image")
    ap.add_argument("--class-names", default=",".join(DEFAULT_CLASS_NAMES),
                     help="comma-separated names written to the dataset, in class-index "
                          "order: the red card class first, then qr, then logo")
    ap.add_argument("--image-format", choices=["jpg", "png"], default="jpg")
    ap.add_argument("--jpeg-quality", type=int, default=DEFAULT_QUALITY)
    ap.add_argument("--variants", choices=["both", "straight", "upside"], default="both",
                     help="what each press of s writes: both the straight frame and its "
                          "180-degree copy (default), or only one of the two")
    ap.add_argument("--allow-empty", action="store_true",
                     help="also save frames where no red box was found (negatives)")
    ap.add_argument("--no-qr", action="store_true", help="do not annotate qr_code boxes")
    # zxing args
    ap.add_argument("--barcode-formats", default=DEFAULT_BARCODE_FORMATS,
                     help="comma-separated zxing formats to decode")
    ap.add_argument("--qr-scan", choices=["frame", "cards"], default="frame",
                     help="frame: one zxing pass over the whole frame; cards: one pass "
                          "inside each red card, which reads better but misses symbols "
                          "off a card")
    ap.add_argument("--qr-every", type=int, default=DEFAULT_QR_EVERY,
                     help="rescan for symbols every N preview frames (a save always "
                          "rescans the frame it writes)")
    ap.add_argument("--qr-pad", type=float, default=DEFAULT_QR_PAD,
                     help="grow each symbol box by this fraction of its own size")
    ap.add_argument("--qr-card-margin", type=float, default=0.05,
                     help="--qr-scan cards: grow each card crop by this fraction")
    ap.add_argument("--no-logo", action="store_true", help="do not annotate logo boxes")
    args = ap.parse_args()

    # Validated before anything touches the camera or loads the engine, so a
    # typo here fails instantly instead of after the engine warm-up.
    class_names = [n.strip() for n in args.class_names.split(",") if n.strip()]
    if len(class_names) != 3:
        sys.exit(f"[dataset] --class-names needs 3 names (label,qr,logo), "
                 f"got {len(class_names)}: {class_names}")

    formats = barcode_formats([f.strip() for f in args.barcode_formats.split(",") if f.strip()])
    if args.qr_every < 1:
        sys.exit("[dataset] --qr-every must be at least 1")

    cam_index = args.index if args.index is not None else V.find_camera_index()
    pipeline = V.gstreamer_pipeline(cam_index, args.width, args.height,
                                     args.fps, args.v4l2_format)
    print(f"[camera] using /dev/video{cam_index}")

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera via GStreamer pipeline")

    model = V.YOLO26TRT(args.engine, input_size=(args.imgsz, args.imgsz))
    print(f"[model] loaded {args.engine}")

    writer = DatasetWriter(args.out, class_names, args.format,
                            args.image_format, args.jpeg_quality)
    writer.write_classes()
    print(f"[dataset] writing to {Path(args.out).resolve()}  (format: {args.format})")
    if not args.no_qr:
        print(f"[zxing] decoding {args.barcode_formats} — {args.qr_scan} scan, "
              f"preview every {args.qr_every} frames")
    print(f"[dataset] variants per capture: {args.variants}"
          + ("  (straight + upside-down, same classes)" if args.variants == "both" else ""))
    print("[keys] s = save frame   u = undo last save   q = quit")

    disp_w, disp_h = ((args.height, args.width) if args.rotate in (90, 270)
                      else (args.width, args.height))
    win_name = "make_dataset - s: save   u: undo   q: quit"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    scale = min(V.DISPLAY_MAX_W / disp_w, V.DISPLAY_MAX_H / disp_h, 1.0)
    cv2.resizeWindow(win_name, int(disp_w * scale), int(disp_h * scale))

    flash_text, flash_until = "", 0.0
    frame_no = 0
    symbols = []           # last zxing result, reused between rescans
    try:
        while True:
            frame_no += 1
            ok, frame = cap.read()
            if not ok:
                print("[camera] frame grab failed, retrying...")
                continue

            frame = V.rotate_frame(frame, args.rotate)
            clean = frame.copy()   # what gets saved; the preview is drawn on a copy

            inp, ratio, pad = V.preprocess(frame, model.input_size)
            raw = model.infer(inp)
            dets = V.postprocess(raw, ratio, pad, frame.shape, args.conf_thres)
            labels = V.vertical_labels(dets, args.label_class, args.min_aspect)

            # label boxes are the red card, not the model's own label box
            cards = []
            for det in labels:
                card = V.red_box_for_label(frame, det, args.margin_x, args.margin_y,
                                            args.edge_frac, args.sat_min, args.val_min,
                                            args.red_method, args.max_card_scale)
                if card is not None:
                    cards.append(card)
                V.draw_label_and_card(frame, det, det[4], card)
            boxes = [Box(LABEL_IDX, card, None) for card in cards]
            n_cards = len(cards)
            # a vertical label whose red card could not be measured would go
            # into the set unannotated, which trains the model against itself
            missing = len(labels) - n_cards

            # qr boxes come from zxing, not the model: only symbols that
            # actually decoded, and their own reported corners. Too slow to run
            # on every frame, so the preview reuses the last result in between.
            if not args.no_qr and frame_no % args.qr_every == 1 % args.qr_every:
                symbols = scan_frame_symbols(clean, formats, args.qr_scan, cards,
                                              args.qr_pad, args.qr_card_margin)
            if not args.no_qr:
                boxes += symbols

            # logo still passes straight through as the detector found it
            if not args.no_logo:
                for x1, y1, x2, y2, conf, cls_id in dets:
                    if int(cls_id) != args.logo_class:
                        continue
                    boxes.append(Box(LOGO_IDX, (int(x1), int(y1), int(x2), int(y2)), None))

            for box in boxes:
                if box.cls == LABEL_IDX:
                    continue                      # already drawn with its label
                x1, y1, x2, y2 = box.xyxy
                cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLORS[box.cls], 2)
                caption = class_names[box.cls]
                if box.meta:
                    caption = f"{box.meta['symbol_format']}: {box.meta['text'][:16]}"
                cv2.putText(frame, caption, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, BOX_COLORS[box.cls], 1, cv2.LINE_AA)

            now = time.time()
            n_qr = sum(1 for b in boxes if b.cls == QR_IDX)
            n_logo = sum(1 for b in boxes if b.cls == LOGO_IDX)
            cv2.putText(frame, f"{class_names[0]}: {n_cards}  {class_names[1]}: {n_qr}  "
                               f"{class_names[2]}: {n_logo}   images: {writer.count} "
                               f"(+{writer.saved_this_run} captures this run)",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            if missing:
                cv2.putText(frame, f"WARNING: {missing} vertical label(s) with no red box",
                            (20, disp_h - 30 if disp_h < 1200 else 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 140, 255), 2, cv2.LINE_AA)
            if now < flash_until:
                cv2.putText(frame, flash_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 200, 255), 2, cv2.LINE_AA)
            cv2.imshow(win_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                if not n_cards and not args.allow_empty:
                    flash_text = "no red box in frame - not saved (--allow-empty to keep)"
                else:
                    # The preview's symbols may be a few frames stale. Rescan the
                    # frame actually being written so its annotations describe its
                    # own pixels, not a neighbouring frame's.
                    saved_boxes = [b for b in boxes if b.cls != QR_IDX]
                    if not args.no_qr:
                        symbols = scan_frame_symbols(clean, formats, args.qr_scan, cards,
                                                      args.qr_pad, args.qr_card_margin)
                        saved_boxes += symbols
                        n_qr = len(symbols)
                    names = writer.save(build_variants(clean, saved_boxes, args.variants))
                    flash_text = (f"saved {' + '.join(names)}  ({n_cards} {class_names[0]}, "
                                  f"{n_qr} {class_names[1]}, {n_logo} {class_names[2]} each)"
                                  if names else "save failed")
                    if names and missing:
                        flash_text += f"  [WARNING: {missing} label(s) had no red box]"
                flash_until = now + FLASH_SECONDS
                print(f"[dataset] {flash_text}")
            elif key == ord("u"):
                dropped = writer.undo()
                flash_text = f"undid {' + '.join(dropped)}" if dropped else "nothing to undo"
                flash_until = now + FLASH_SECONDS
                print(f"[dataset] {flash_text}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tally = ", ".join(f"{n} {name}" for name, n in writer.counts().items())
        captures = len({i.get("capture", i["id"]) for i in writer.images})
        print(f"\n[dataset] {captures} captures -> {writer.count} images, "
              f"{len(writer.annotations)} boxes ({tally}) in {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
