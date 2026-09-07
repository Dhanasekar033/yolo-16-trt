#!/usr/bin/env python3
"""
Capture a red-label detection dataset off the Global Shutter Camera.

Runs the same pipeline as cam-trt-vlabel.py — YOLO26 TensorRT finds the
vertical labels, the red card around each one is measured from a red mask —
but instead of only drawing the result it lets you bank it. Press **s** and
the frame is written to disk with two annotated classes:

    qr_code   the box zxing-cpp reports for each symbol it actually DECODES —
              not the model's qr_code box. zxing returns the four corners of
              the symbol it read, and their extent is the annotation. Because
              the formats mask covers DataMatrix as well as QR, the Data
              Matrix codes on the card are picked up by the same pass and
              annotated under this class too. The decoded text and the symbol
              format ride along in the COCO annotation.
    logo      the model's logo boxes, passed through as detected

The red card is NOT a class here. It is still measured on every frame, because
--qr-scan cards crops to it and because a frame with no card is a frame with
nothing worth banking, but it is a locator and never an annotation. The card
still draws in the preview so you can see what the qr pass is working from;
what leaves the preview for the dataset is qr_code and logo alone. Class
indices follow that: qr_code is 0 and logo is 1.

Only symbols zxing can read become annotations, so a code that is blurred or
glared out is left unboxed rather than guessed at. zxing is the slow part of
the frame, so the preview rescans only every --qr-every frames; a save always
rescans the exact frame being written, so the annotations match its pixels.

Duplicates of one object are collapsed before anything is drawn or written. The
engine's own NMS lets near-copies through — two logo boxes on one logo, two
vertical label detections resolving to the same red card — and each would land
as its own annotation, training the next model to fire twice on one object.
Same class plus the same piece of the frame means one box: the highest
confidence wins, ties go to the bigger box. Two symbols whose decoded text
differs are never merged however much their boxes overlap, because that is two
symbols and not one. --no-dedupe turns this off; --dedupe-iou and
--dedupe-contain tune what counts as the same box.

Every press of s writes the frame twice by default: once straight, once turned
180 degrees, with the same classes on both. The detector only finds these
labels the right way up — inferring on an upside-down frame returns nothing —
so the upside-down copy cannot be captured directly. Its boxes are the
straight frame's, mapped through the image centre, which is exact for 180
degrees. That is what puts upside-down cards in the training set. --variants
picks one or the other if you do not want both.

The horizontal tamper strip is still dropped — only vertical labels get a red
card, and only cards steer the symbol scan.

The detection half is imported from cam-trt-vlabel.py rather than copied, so
whatever you tune there (--sat-min, --margin-y, --max-card-scale …) is exactly
what gets baked into the dataset.

Keys in the preview window:
    s   save the current frame + its boxes (straight and upside-down)
    u   undo the last save (drops every image it wrote, and its annotations)
    q   quit

Output, under --out (default dataset/) — the flat layout split_dataset.py takes:
    images/20260907_113314_123.jpg       the frame as shot
    images/20260907_113314_123_r180.jpg  the same frame upside down
    labels/20260907_113314_123.txt  YOLO: one file per image, class then
                              normalised centre/size (--format yolo/both —
                              both is the default, so every capture gets a .txt)
    annotations.json          COCO: one file, all images and boxes. Each image
                              carries `capture` (the stem shared by the images
                              one press of s wrote) and `variant` (straight /
                              upside_down).

A capture is named for the instant it was shot — YYYYmmdd_HHMMSS_mmm, local
time, to the millisecond — and both variants of it share that stem. Nothing
counts frames, so two sessions or two machines can be poured into one set
without a rename, and ls still lists them in the order they were taken.
    classes.txt               qr_code / logo, in class-index order

Re-running appends to an existing dataset instead of overwriting it, and
annotations.json is what it appends to: emptying images/ by hand does not empty
that, so the set still remembers every frame it ever wrote. Deleting the whole
--out directory is what starts one over. A resume that finds records with no
image file says so and writes no .txt for them; --prune-missing drops those
records for good.

A set banked earlier under --format coco has no labels/*.txt at all; resuming it
writes the missing ones straight out of the annotations already on disk, which
is exact — nothing is re-measured off the images. --rewrite-only does only that
and exits, so an existing dataset can be given its .txt files (and, with
--dedupe-existing, have duplicate boxes already banked collapsed) without a
camera or an engine.

A set banked before the card stopped being a class carries three categories,
where 2 meant qr_code and now means logo. Appending to it as-is would leave two
numberings in one file, so resuming one is refused until --remap-classes says
to renumber it by name; that drops its label boxes and keeps everything else,
images included.

Usage:
    python3 make_dataset.py --engine best.engine
    python3 make_dataset.py --engine best.engine --out dataset/run2
    python3 make_dataset.py --engine best.engine --format both
    python3 make_dataset.py --engine best.engine --format yolo --image-format png
    python3 make_dataset.py --engine best.engine --allow-empty   # keep negatives too
    python3 make_dataset.py --engine best.engine --class-names qr,mark
    python3 make_dataset.py --engine best.engine --qr-scan cards   # scan inside each card
    python3 make_dataset.py --engine best.engine --barcode-formats QRCode,DataMatrix
    python3 make_dataset.py --engine best.engine --variants straight   # no rotated copy
    python3 make_dataset.py --out dataset --rewrite-only   # write the missing .txt files
    python3 make_dataset.py --out dataset --rewrite-only --dedupe-existing
    python3 make_dataset.py --out dataset --rewrite-only --remap-classes  # drop label
    python3 make_dataset.py --out dataset --rewrite-only --prune-missing  # forget deleted images
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

# Class indices as written to the dataset. The red card is NOT one of them —
# it is still measured every frame, because the qr pass crops to it and a frame
# without one has nothing worth banking, but it is a locator now and never an
# annotation. CARD_CLS is negative so a card box can ride through dedupe with
# the rest and still be impossible to confuse with a written class.
QR_IDX, LOGO_IDX = 0, 1
CARD_CLS = -1
DEFAULT_CLASS_NAMES = ["qr_code", "logo"]

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

# Two boxes of one class are the same object when they overlap this much. The
# duplicate logo boxes the engine emits sit at IoU ~0.85 with the smaller box
# ~0.95 inside the larger, so both thresholds are set well under that: loose
# enough to catch a near-copy, tight enough to leave two neighbouring cards —
# which barely touch — as two annotations.
DEFAULT_DEDUPE_IOU     = 0.45
DEFAULT_DEDUPE_CONTAIN = 0.85   # intersection over the SMALLER box's area

# One annotation on its way to disk. `meta` is merged into the COCO annotation
# and ignored by YOLO, which has nowhere to put it. `score` is the detector's
# confidence where there is one — it decides which of two duplicates survives,
# and is never written to either format.
Box = namedtuple("Box", "cls xyxy meta score")
Box.__new__.__defaults__ = (None,)          # score: only detections carry one

ROT_SUFFIX = "_r180"         # marks the upside-down copy of a capture
STAMP_FORMAT = "%Y%m%d_%H%M%S"   # + _mmm milliseconds, the stem of every capture


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


def _area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _intersection(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def _iou(a, b):
    inter = _intersection(a, b)
    if inter == 0:
        return 0.0
    return inter / float(_area(a) + _area(b) - inter)


def _containment(a, b):
    """How much of the SMALLER box lies inside the larger.

    IoU alone is blind to a box sitting wholly inside a bigger one — a tight
    logo box inside a loose one can score below any sane IoU threshold and
    still be the same logo twice."""
    smaller = min(_area(a), _area(b))
    if smaller == 0:
        return 0.0
    return _intersection(a, b) / float(smaller)


def _same_object(a, b, iou_thres, contain_frac):
    """Whether two boxes are two annotations of one thing."""
    if a.cls != b.cls:
        return False
    # A decoded payload is identity: two symbols that read differently are two
    # symbols, no matter how their padded boxes overlap. Boxes without text
    # (cards, logos) fall through to geometry alone.
    if (a.meta or {}).get("text") != (b.meta or {}).get("text"):
        return False
    return (_iou(a.xyxy, b.xyxy) >= iou_thres
            or _containment(a.xyxy, b.xyxy) >= contain_frac)


def dedupe_boxes(boxes, iou_thres=DEFAULT_DEDUPE_IOU,
                 contain_frac=DEFAULT_DEDUPE_CONTAIN):
    """One box per object per class, keeping the best of each duplicate group.

    Duplicates reach here from three directions: the engine's end2end NMS
    passing two boxes on one logo, two vertical label detections measuring out
    to the same red card, and — under --qr-scan cards — overlapping crops
    decoding one symbol once per crop. All three write the same object twice.

    Ranked by confidence first so the detector's own preference decides, then
    by area so an untouched pair (cards and symbols carry no score) keeps the
    box that covers the whole object rather than a clipped one. Input order is
    restored on the way out, which keeps annotations grouped by class."""
    order = {id(box): i for i, box in enumerate(boxes)}
    ranked = sorted(boxes, reverse=True,
                    key=lambda b: (b.score if b.score is not None else 0.0,
                                   _area(b.xyxy)))
    kept = []
    for box in ranked:
        if any(_same_object(box, other, iou_thres, contain_frac) for other in kept):
            continue
        kept.append(box)
    return sorted(kept, key=lambda b: order[id(b)])


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
    # Crop overlap is mechanical, not a judgement call: where two cards sit
    # close a symbol in the overlap is decoded once per crop, so those repeats
    # are collapsed here whatever the caller asked for.
    return dedupe_boxes(found)


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
    """Images plus qr_code/logo boxes, in COCO and/or YOLO layout.

    Boxes arrive as Box(cls, (x1, y1, x2, y2), meta) records. COCO category
    ids are the class index plus one, since COCO numbers categories from 1
    while YOLO numbers classes from 0. Anything in `meta` — the decoded text
    and symbol format for a zxing box — is merged into the COCO annotation;
    YOLO's five numbers have nowhere to carry it.

    COCO is one annotations.json for the whole set — that is the format's
    actual shape, there is no per-image COCO text file. YOLO is one .txt per
    image, class then normalised centre/size, and those .txt files are written
    from the annotations rather than alongside them, so a file backfilled into
    an old dataset and one written at capture time come out identical. Both are
    rewritten after every save so an interrupted session leaves a loadable
    dataset."""

    def __init__(self, root, class_names=None, fmt="both",
                 image_ext="jpg", quality=DEFAULT_QUALITY, remap=False):
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

        self.saved_this_run = 0
        self.needs_rewrite = False
        self.images, self.annotations, stored = self._resume()
        if stored and stored != self.class_names:
            self._reconcile(stored, remap)
        self._warn_missing_images()

    def missing_images(self):
        """Image records whose .jpg is no longer on disk.

        annotations.json is the dataset — emptying images/ by hand does not
        empty it, and a resume picks the records straight back up. Left
        unnoticed that reads as the capture having written labels and no
        images, when what actually happened is that the images were deleted
        under a set that still remembers them."""
        return [i for i in self.images
                if not (self.images_dir / i["file_name"]).exists()]

    def _warn_missing_images(self):
        gone = self.missing_images()
        if not gone:
            return
        print(f"[dataset] WARNING: {len(gone)} of {len(self.images)} image record(s) "
              f"in {self.json_path} have no file in {self.images_dir}.")
        print(f"          Their boxes stay in the annotations and no .txt is written "
              f"for them. Put the images back, or --prune-missing to drop the "
              f"records, or delete {self.json_path} to start the set over.")

    def _resume(self):
        """Pick up an existing dataset so a second session appends to it."""
        if not self.json_path.exists():
            return [], [], None
        try:
            with open(self.json_path) as f:
                doc = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            sys.exit(f"[dataset] {self.json_path} exists but is unreadable ({exc}). "
                     f"Move it aside or point --out somewhere else.")
        images, annotations = doc.get("images", []), doc.get("annotations", [])
        print(f"[dataset] resuming {self.json_path}: "
              f"{len(images)} images, {len(annotations)} boxes")
        stored = [c["name"] for c in sorted(doc.get("categories", []),
                                             key=lambda c: c["id"])]
        return images, annotations, stored

    def _reconcile(self, stored, remap):
        """Bring a set banked under a different class list onto this one.

        A category id means nothing on its own — 2 was qr_code when the card
        was still class 0, and is logo now. Appending new boxes to a set
        numbered the other way silently mislabels half of it, so this refuses
        to run rather than mix the two, and --remap-classes is how you say to
        renumber by name instead. Classes the run no longer writes lose their
        annotations; the images stay, as negatives or as carriers of the
        classes that remain."""
        keep = {name: i for i, name in enumerate(self.class_names)}
        gone = [n for n in stored if n not in keep]
        if not remap:
            sys.exit(
                f"[dataset] {self.json_path} was banked with classes {stored}, but "
                f"this run writes {self.class_names}. Saving into it would leave two "
                f"numberings in one file.\n"
                f"          --remap-classes renumbers it by name"
                + (f" and DROPS every {'/'.join(gone)} box in it" if gone else "")
                + f", or point --out at a new directory.")

        kept, dropped = [], {}
        for a in self.annotations:
            i = a["category_id"] - 1
            name = stored[i] if 0 <= i < len(stored) else None
            if name in keep:
                a["category_id"] = keep[name] + 1
                kept.append(a)
            else:
                dropped[name] = dropped.get(name, 0) + 1
        self.annotations = kept
        self.needs_rewrite = True          # every file on disk is now stale
        summary = ", ".join(f"{n} {name}" for name, n in dropped.items()) or "none"
        print(f"[dataset] remapped {stored} -> {self.class_names}: "
              f"dropped {summary}, {len(kept)} box(es) kept")

    @property
    def count(self):
        return len(self.images)

    def _next_capture(self):
        """Name the capture about to be written: when it was shot, to the ms.

        One press of s is one capture, which may write more than one image —
        the straight frame and its upside-down copy share the stem, differing
        only by ROT_SUFFIX, and are undone together.

        A clock beats a counter here because the name then means something
        outside its own directory: two sessions, two cameras or two machines
        can be poured into one set without renaming a thing, and the frames
        still sort into the order they were shot. Two captures inside one
        millisecond would collide, so the stamp is walked forward until the
        name is free — no capture can quietly overwrite another.

        Returns (datetime, stem) so the record's date_captured is the very
        instant its file is named after, not a second reading of the clock."""
        taken = dt.datetime.now()
        used = {Path(i["file_name"]).stem for i in self.images}
        while True:
            stem = taken.strftime(STAMP_FORMAT) + f"_{taken.microsecond // 1000:03d}"
            names = [f"{stem}{suffix}" for suffix in ("", ROT_SUFFIX)]
            if not (used.intersection(names)
                    or any((self.images_dir / f"{n}.{self.image_ext}").exists()
                           for n in names)):
                return taken, stem
            taken += dt.timedelta(milliseconds=1)

    def save(self, variants):
        """Write one capture: a list of (frame, boxes, suffix) to store together.

        Each frame must be a clean frame — anything drawn on it would be baked
        into the training image. Returns the file names written, or None if
        nothing could be encoded."""
        taken, capture = self._next_capture()
        next_id = max((i["id"] for i in self.images), default=0) + 1
        next_ann = max((a["id"] for a in self.annotations), default=0) + 1

        params = ([cv2.IMWRITE_JPEG_QUALITY, self.quality]
                  if self.image_ext in ("jpg", "jpeg") else [])
        staged = []
        for frame, boxes, suffix in variants:
            stem = f"{capture}{suffix}"
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
            path = self.images_dir / name
            _write_bytes(path, payload)
            # An annotation record for an image that is not on disk is worse
            # than a lost frame: the set looks complete and trains on nothing.
            if not path.exists() or path.stat().st_size != len(payload):
                print(f"\n[dataset] {path} did not land on disk, capture dropped")
                return None
            image_id = next_id + k
            image = {
                "id": image_id,
                "file_name": name,
                "width": w,
                "height": h,
                "capture": capture,
                "variant": "upside_down" if stem.endswith(ROT_SUFFIX) else "straight",
                "date_captured": taken.isoformat(timespec="milliseconds"),
            }
            self.images.append(image)
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
            written.append(image)

        self._flush(written)
        self.saved_this_run += 1
        return [name for _, name, _, _, _ in staged]

    def undo(self):
        """Drop the most recent capture — every image it wrote — on disk and in
        the annotations."""
        if not self.images:
            return None
        # The newest capture is the last one appended. Taking the largest
        # `capture` instead would break the moment a set holds both the old
        # numbered captures and the timestamped ones, which do not compare.
        capture = self.images[-1].get("capture", self.images[-1]["id"])
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

    def _flush(self, images):
        """Rewrite annotations.json, plus a YOLO .txt for each image given."""
        if self.fmt in ("coco", "both"):
            _write_text(self.json_path, self._coco_json())
        if self.fmt in ("yolo", "both"):
            self._write_yolo(images)

    def _coco_json(self):
        return json.dumps({
            "info": {
                "description": "qr_code boxes are the symbols zxing-cpp decoded "
                               "(QR and DataMatrix), carrying their text; logo "
                               "boxes are the detector's own. The red card around "
                               "each vertical label is measured to find them but is "
                               "not annotated. "
                               "Duplicate boxes of one object are collapsed to one. "
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
        }, indent=2)

    def _write_yolo(self, images):
        """One .txt per image, taken from that image's COCO annotations.

        An image that is not on disk gets no label file. A .txt beside a
        missing .jpg trains nothing and reads as though the capture half
        worked, which is worse than the file simply being absent."""
        images = [i for i in images
                  if (self.images_dir / i["file_name"]).exists()]
        by_image = {}
        for a in self.annotations:
            by_image.setdefault(a["image_id"], []).append(a)

        for image in images:
            w, h = image["width"], image["height"]
            lines = []
            for a in by_image.get(image["id"], []):
                x, y, bw, bh = a["bbox"]
                # YOLO wants class then centre and size, each divided through by
                # the image dimensions; its classes start at 0, COCO's ids at 1.
                lines.append("%d %.6f %.6f %.6f %.6f" % (
                    a["category_id"] - 1,
                    (x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h))
            # A negative frame gets a genuinely empty file, not a blank line.
            _write_text(self.labels_dir / f"{Path(image['file_name']).stem}.txt",
                        "\n".join(lines) + "\n" if lines else "")

    def backfill_labels(self):
        """Write the YOLO .txt files an existing dataset is missing.

        A set banked under --format coco has every box on disk and no labels/
        directory at all. The boxes were never lost, only never written in
        YOLO's layout, so this reads annotations.json and writes them out —
        nothing is re-measured off the images."""
        if self.fmt not in ("yolo", "both") or not self.images:
            return 0
        missing = [i for i in self.images
                   if (self.images_dir / i["file_name"]).exists()
                   and not (self.labels_dir / f"{Path(i['file_name']).stem}.txt").exists()]
        self._write_yolo(missing)
        if missing:
            print(f"[dataset] wrote {len(missing)} missing label file(s) "
                  f"into {self.labels_dir}")
        return len(missing)

    def dedupe_existing(self, iou_thres=DEFAULT_DEDUPE_IOU,
                        contain_frac=DEFAULT_DEDUPE_CONTAIN):
        """Collapse duplicate same-class boxes already banked, image by image.

        For frames captured before duplicates were being suppressed. No
        confidence survives in a written annotation, so of a duplicate pair the
        larger box is what is kept."""
        by_image = {}
        for a in self.annotations:
            by_image.setdefault(a["image_id"], []).append(a)

        keep = set()
        for anns in by_image.values():
            boxes, ann_id = [], {}
            for a in anns:
                x, y, bw, bh = a["bbox"]
                box = Box(a["category_id"] - 1, (x, y, x + bw, y + bh),
                          {"text": a["text"]} if "text" in a else None)
                ann_id[id(box)] = a["id"]
                boxes.append(box)
            keep.update(ann_id[id(b)] for b in
                        dedupe_boxes(boxes, iou_thres, contain_frac))

        dropped = len(self.annotations) - len(keep)
        self.annotations = [a for a in self.annotations if a["id"] in keep]
        return dropped

    def prune_missing(self):
        """Drop the records of images that are no longer on disk, with their
        boxes and any label file left stranded beside them."""
        gone = self.missing_images()
        if not gone:
            return 0, 0, 0
        ids = {i["id"] for i in gone}
        n_boxes = sum(1 for a in self.annotations if a["image_id"] in ids)
        self.images = [i for i in self.images if i["id"] not in ids]
        self.annotations = [a for a in self.annotations if a["image_id"] not in ids]

        n_txt = 0
        for image in gone:
            txt = self.labels_dir / f"{Path(image['file_name']).stem}.txt"
            if txt.exists():
                txt.unlink()
                n_txt += 1
        self.needs_rewrite = True
        return len(gone), n_boxes, n_txt

    def rewrite(self):
        """Rewrite annotations.json and every label file from what is in memory."""
        self._flush(self.images)

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
    ap.add_argument("--format", choices=["coco", "yolo", "both"], default="both",
                     help="both (default): annotations.json AND one .txt per image; "
                          "coco: json only; yolo: .txt only")
    ap.add_argument("--class-names", default=",".join(DEFAULT_CLASS_NAMES),
                     help="comma-separated names written to the dataset, in class-index "
                          "order: the qr class first, then logo")
    ap.add_argument("--image-format", choices=["jpg", "png"], default="jpg")
    ap.add_argument("--jpeg-quality", type=int, default=DEFAULT_QUALITY)
    ap.add_argument("--variants", choices=["both", "straight", "upside"], default="both",
                     help="what each press of s writes: both the straight frame and its "
                          "180-degree copy (default), or only one of the two")
    ap.add_argument("--allow-empty", action="store_true",
                     help="also save frames with nothing to annotate (negatives)")
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
    # duplicate suppression
    ap.add_argument("--dedupe-iou", type=float, default=DEFAULT_DEDUPE_IOU,
                     help="two boxes of one class overlapping this much are one object")
    ap.add_argument("--dedupe-contain", type=float, default=DEFAULT_DEDUPE_CONTAIN,
                     help="or this much of the smaller box lying inside the larger")
    ap.add_argument("--no-dedupe", action="store_true",
                     help="keep every box, duplicates included")
    # maintenance of a dataset already on disk
    ap.add_argument("--dedupe-existing", action="store_true",
                     help="also collapse duplicate boxes already in --out's annotations")
    ap.add_argument("--remap-classes", action="store_true",
                     help="renumber an existing --out banked under other class names, "
                          "dropping the boxes of classes this run no longer writes")
    ap.add_argument("--prune-missing", action="store_true",
                     help="drop the records of images no longer in --out's images/, "
                          "with their boxes and any stranded label file")
    ap.add_argument("--rewrite-only", action="store_true",
                     help="rewrite --out's annotations and label files from what is "
                          "already banked, then exit (no camera, no engine)")
    args = ap.parse_args()

    # Validated before anything touches the camera or loads the engine, so a
    # typo here fails instantly instead of after the engine warm-up.
    class_names = [n.strip() for n in args.class_names.split(",") if n.strip()]
    if len(class_names) != 2:
        sys.exit(f"[dataset] --class-names needs 2 names (qr,logo), "
                 f"got {len(class_names)}: {class_names}")

    formats = barcode_formats([f.strip() for f in args.barcode_formats.split(",") if f.strip()])
    if args.qr_every < 1:
        sys.exit("[dataset] --qr-every must be at least 1")

    writer = DatasetWriter(args.out, class_names, args.format, args.image_format,
                            args.jpeg_quality, args.remap_classes)
    writer.write_classes()
    if args.prune_missing:
        n_imgs, n_boxes, n_txt = writer.prune_missing()
        print(f"[dataset] pruned {n_imgs} image record(s), {n_boxes} box(es) and "
              f"{n_txt} stranded label file(s)")
    if args.dedupe_existing:
        dropped = writer.dedupe_existing(args.dedupe_iou, args.dedupe_contain)
        print(f"[dataset] dropped {dropped} duplicate box(es) from the annotations "
              f"already on disk")
        writer.needs_rewrite = True
    if writer.needs_rewrite:
        writer.rewrite()               # a remap leaves every file on disk stale
    else:
        writer.backfill_labels()
    print(f"[dataset] writing to {Path(args.out).resolve()}  (format: {args.format})")
    if args.rewrite_only:
        return

    cam_index = args.index if args.index is not None else V.find_camera_index()
    pipeline = V.gstreamer_pipeline(cam_index, args.width, args.height,
                                     args.fps, args.v4l2_format)
    print(f"[camera] using /dev/video{cam_index}")

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera via GStreamer pipeline")

    model = V.YOLO26TRT(args.engine, input_size=(args.imgsz, args.imgsz))
    print(f"[model] loaded {args.engine}")

    if not args.no_qr:
        print(f"[zxing] decoding {args.barcode_formats} — {args.qr_scan} scan, "
              f"preview every {args.qr_every} frames")
    if args.no_dedupe:
        print("[dataset] duplicate suppression OFF")
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

            # The red card is measured for the qr crop and the save gate; it
            # is not one of the classes written any more.
            measured = [(det, V.red_box_for_label(
                            frame, det, args.margin_x, args.margin_y,
                            args.edge_frac, args.sat_min, args.val_min,
                            args.red_method, args.max_card_scale))
                        for det in labels]
            # Two detections on one label measure out to the same card, so the
            # cards are deduped before the qr pass — otherwise --qr-scan cards
            # would crop and decode that card twice as well.
            card_boxes, from_det = [], {}
            for i, (det, card) in enumerate(measured):
                if card is None:
                    continue
                box = Box(CARD_CLS, tuple(int(v) for v in card), None, float(det[4]))
                from_det[id(box)] = i
                card_boxes.append(box)
            n_dupes = len(card_boxes)
            if not args.no_dedupe:
                card_boxes = dedupe_boxes(card_boxes, args.dedupe_iou, args.dedupe_contain)
            n_dupes -= len(card_boxes)

            # only the surviving card is drawn; the detection whose duplicate
            # was dropped still shows its own green label box
            kept_dets = {from_det[id(b)] for b in card_boxes}
            for i, (det, card) in enumerate(measured):
                V.draw_label_and_card(frame, det, det[4], card if i in kept_dets else None)

            cards = [b.xyxy for b in card_boxes]
            boxes = []
            n_cards = len(cards)
            # under --qr-scan cards a label with no red card is never cropped,
            # so its symbols are never decoded and never annotated
            missing = sum(1 for _, card in measured if card is None)

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
                    boxes.append(Box(LOGO_IDX, (int(x1), int(y1), int(x2), int(y2)),
                                      None, float(conf)))

            # The engine's own NMS lets near-copies of one logo through, and a
            # whole-frame zxing pass can report one symbol twice. Both would be
            # written as two annotations on one object.
            n_boxes = len(boxes)
            if not args.no_dedupe:
                boxes = dedupe_boxes(boxes, args.dedupe_iou, args.dedupe_contain)
            n_dupes += n_boxes - len(boxes)

            for box in boxes:
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
            cv2.putText(frame, f"{class_names[0]}: {n_qr}  {class_names[1]}: {n_logo}  "
                               f"(cards seen: {n_cards})   images: {writer.count} "
                               f"(+{writer.saved_this_run} captures this run)"
                               + (f"   [-{n_dupes} dup]" if n_dupes else ""),
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
                # The preview's symbols may be a few frames stale. Rescan the
                # frame actually being written so its annotations describe its
                # own pixels, not a neighbouring frame's — and rescan before
                # deciding whether the frame is worth keeping, so a symbol the
                # preview had not got to yet still counts.
                saved_boxes = [b for b in boxes if b.cls != QR_IDX]
                if not args.no_qr:
                    symbols = scan_frame_symbols(clean, formats, args.qr_scan, cards,
                                                  args.qr_pad, args.qr_card_margin)
                    saved_boxes += symbols
                if not args.no_dedupe:
                    saved_boxes = dedupe_boxes(saved_boxes, args.dedupe_iou,
                                                args.dedupe_contain)
                n_qr = sum(1 for b in saved_boxes if b.cls == QR_IDX)
                n_logo = sum(1 for b in saved_boxes if b.cls == LOGO_IDX)

                # The gate is the annotations, not the card: the card is no
                # longer written, so a frame holding one but no readable symbol
                # and no logo would go in as an unlabelled negative.
                if not saved_boxes and not args.allow_empty:
                    flash_text = (f"no {class_names[0]} / {class_names[1]} box in frame "
                                  f"- not saved (--allow-empty to keep)")
                else:
                    names = writer.save(build_variants(clean, saved_boxes, args.variants))
                    flash_text = (f"saved {Path(names[0]).stem} x{len(names)}  "
                                  f"({n_qr} {class_names[0]}, {n_logo} {class_names[1]} each)"
                                  if names else "save failed")
                    if names and missing:
                        flash_text += f"  [WARNING: {missing} label(s) had no red box]"
                flash_until = now + FLASH_SECONDS
                print(f"[dataset] {flash_text}")
            elif key == ord("u"):
                dropped = writer.undo()
                flash_text = (f"undid {Path(dropped[0]).stem} x{len(dropped)}"
                              if dropped else "nothing to undo")
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
