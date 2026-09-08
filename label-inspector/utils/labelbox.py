"""Turning the two parts a stripped-dataset engine finds back into a label.

best-2.engine knows `code` and `artifact` and nothing else -- two small boxes
per label, where what anyone wants to cut out is the label they both sit on.
This is the part that puts that box back, kept here rather than in one of the
scripts so the one-camera and two-camera viewers cannot drift apart: they pair
the parts by the same rule and draw the same box.

See cam-trt-labelbox.py for what the rule is and why it is not simply the
nearest artifact to each code.

The inspector uses it to put the `label` class back. Every part of this app
downstream of the detector -- the crops, the missing-part check, which code
belongs to which label, what a line crossing is -- is written in terms of a
label box, and an engine trained without one still has to feed all of it. So
the pairs are made here and handed on as label detections, and nothing
downstream needs to know they were not the model's own.
"""

import os
import time

import cv2
import numpy as np


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


def save_shot(frame, labels, out_dir, ext, n, tag=""):
    """Write the clean frame and one file per label box. Returns (paths, bytes).

    `frame` must be the frame BEFORE anything was drawn on it -- a crop with a
    box outline burnt into it is no longer a picture of the label."""
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() % 1 * 1000):03d}"
    # A tag keeps two cameras' files apart while a shared stamp keeps the pair
    # together: shot_<stamp>_camA_... and shot_<stamp>_camB_... are the same
    # instant, which is the whole point of saving both at once.
    tag = f"_{tag}" if tag else ""
    params = SHOT_PARAMS.get(ext, [])
    written, size = [], 0

    path = os.path.join(out_dir, f"shot_{stamp}{tag}_f{n:06d}_frame.{ext}")
    if cv2.imwrite(path, frame, params):
        written.append(path)
        size += os.path.getsize(path)

    for i, (box, _measured) in enumerate(labels, 1):
        crop = frame[box[1]:box[3], box[0]:box[2]]
        if crop.size == 0:
            continue
        path = os.path.join(out_dir,
                            f"shot_{stamp}{tag}_f{n:06d}_label{i:02d}.{ext}")
        if cv2.imwrite(path, crop, params):
            written.append(path)
            size += os.path.getsize(path)
    return written, size


