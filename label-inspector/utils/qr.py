"""QR decoding helpers: crop a detected qr_code box and read it with zxing-cpp.

The detector gives tight boxes around the QR symbol, but a QR needs a quiet
zone (blank margin) around it to be decodable, so every crop is expanded by a
margin before it is handed to zxing.
"""

import bisect

import cv2
import numpy as np
import zxingcpp

QR_FORMATS = zxingcpp.BarcodeFormat.QRCode | zxingcpp.BarcodeFormat.MicroQRCode

LABEL = "label"    # hand zxing the whole label crop
QR    = "qr"       # hand zxing just the detected qr box, plus a quiet zone


def expand_box(box, frame_shape, margin=0.15, min_px=8):
    """Grow a box by `margin` (fraction of its own size, at least min_px),
    clipped to the frame. Returns int (x1, y1, x2, y2)."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    mx = max(int(round((x2 - x1) * margin)), min_px)
    my = max(int(round((y2 - y1) * margin)), min_px)
    return (max(int(x1) - mx, 0), max(int(y1) - my, 0),
            min(int(x2) + mx, w), min(int(y2) + my, h))


def _read(img):
    res = zxingcpp.read_barcode(img, formats=QR_FORMATS, try_rotate=True,
                                try_downscale=True, try_invert=True)
    if res is None or not res.valid or not res.text:
        return None
    return res.text


# Built once and reused: constructing a QRCodeDetector per call costs more
# than the decode it performs.
_CV_QR = None


def decode_qr_opencv(frame, box, margin=0.15, min_px=8, min_side=160):
    """Same contract as decode_qr, but through OpenCV's detector.

    zxing and OpenCV locate a symbol differently and do not fail on the same
    codes, so this is a genuinely different attempt rather than a retry.
    Measured over 79 crops off this line that zxing could not read at any
    scale, rotated or inverted, OpenCV recovered 53% of them — including a
    payload on this reel that zxing has never once managed.

    It costs roughly 10ms against a ~17ms frame budget, so it is worth
    spending only on a crop that stands a chance: after zxing has failed, and
    not on one running off the frame edge, where part of the symbol was never
    on the sensor at all.

    Doubling the grayscale crop is what makes it work — at native size the
    same crops read 39%, upscaled 53%.

    Returns (text_or_None, expanded_box)."""
    global _CV_QR
    x1, y1, x2, y2 = expand_box(box, frame.shape, margin, min_px)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None, (x1, y1, x2, y2)

    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    side = min(gray.shape[:2])
    if side < min_side:
        scale = float(min_side) / max(side, 1)
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    if _CV_QR is None:
        _CV_QR = cv2.QRCodeDetector()
    try:
        text, _pts, _ = _CV_QR.detectAndDecode(gray)
    except cv2.error:
        return None, (x1, y1, x2, y2)
    return (text or None), (x1, y1, x2, y2)


# libzbar, imported on first use so this module still loads without it.
_ZBAR = None


def decode_qr_pyzbar(frame, box, margin=0.15, min_px=8, min_side=160):
    """Same contract as decode_qr, but through libzbar.

    zxing and zbar do not fail on the same codes, and on this line the gap is
    not marginal: over 79 crops that zxing could not read at any scale,
    rotated or inverted, zbar read 79 — every one, with no false decodes.
    zxing is still the better first pass (98% vs 95% on ordinary crops, and
    about twice as fast), so the pair belongs in that order.

    Costs roughly 2.3ms, against 13ms for the OpenCV detector, which is why
    this is the fallback worth having.

    Returns (text_or_None, expanded_box)."""
    global _ZBAR
    if _ZBAR is None:
        from pyzbar import pyzbar          # needs libzbar0 on the system
        _ZBAR = pyzbar

    x1, y1, x2, y2 = expand_box(box, frame.shape, margin, min_px)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None, (x1, y1, x2, y2)

    crop = frame[y1:y2, x1:x2]
    side = min(crop.shape[:2])
    if side < min_side:
        scale = float(min_side) / max(side, 1)
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    try:
        found = _ZBAR.decode(crop, symbols=[_ZBAR.ZBarSymbol.QRCODE])
    except Exception:
        return None, (x1, y1, x2, y2)
    if not found:
        return None, (x1, y1, x2, y2)
    return found[0].data.decode("utf-8", "replace"), (x1, y1, x2, y2)


def decode_qr(frame, box, margin=0.15, min_px=8, min_side=160):
    """Crop `box` (+margin) out of `frame` and try to decode a QR from it.

    Tries the plain BGR crop first, then a grayscale copy upscaled to at least
    `min_side` px, then an Otsu-binarized version — small/low-contrast crops
    off a moving line often only read after one of the fallbacks.

    All three passes are zxing. When they all fail, decode_qr_opencv below is
    the thing to reach for — but the caller decides that, because it costs
    real time and is only worth spending on a crop that stands a chance.
    Returns (text_or_None, expanded_box)."""
    x1, y1, x2, y2 = expand_box(box, frame.shape, margin, min_px)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None, (x1, y1, x2, y2)

    crop = frame[y1:y2, x1:x2]

    text = _read(crop)
    if text:
        return text, (x1, y1, x2, y2)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    side = min(gray.shape[:2])
    if side < min_side:
        scale = float(min_side) / max(side, 1)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    text = _read(gray)
    if text:
        return text, (x1, y1, x2, y2)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _read(binary), (x1, y1, x2, y2)


def box_center(box):
    x1, y1, x2, y2 = box[:4]
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def contains_center(outer, inner):
    """True if inner's center falls inside outer."""
    cx, cy = box_center(inner)
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def intersection_area(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return iw * ih


def pick_qr_for_label(label_box, qr_dets):
    """Choose the qr_code detection belonging to `label_box`: prefer a QR whose
    center sits inside the label, else the one overlapping it most. Returns the
    detection row or None. `qr_dets` rows are [x1,y1,x2,y2,conf,cls]."""
    if len(qr_dets) == 0:
        return None

    inside = [d for d in qr_dets if contains_center(label_box, d)]
    if inside:
        return max(inside, key=lambda d: d[4])

    best = max(qr_dets, key=lambda d: intersection_area(label_box, d))
    return best if intersection_area(label_box, best) > 0 else None


class Column:
    """One physical column of labels being read as it crosses the zone.

    A column accumulates the labels that belong to one spreadsheet row. Its
    `key` is the identity resolved from any payload it has read — that is what
    makes the reading robust: two fragments of the same column, or the same
    column picked up twice by the position tracker, resolve to the same key
    and are merged rather than judged as two separate rows.
    """

    def __init__(self, seq, x, frame_no):
        self.seq = seq            # creation order
        self.x = x                # mean x-centre, updated every frame
        self.key = None           # identity of the sheet row, once known
        self.slots_y = []         # y-centres of its labels, sorted, top first
        self.reads = {}           # {slot index: (text, status)}
        self.tol = 20.0           # how close two y-centres count as one label
        self.judged = False
        self.frames = 0
        self.first_frame = frame_no
        self.last_frame = frame_no

    # ── slots ─────────────────────────────────────────────────────────────
    def _slot_for_y(self, cy):
        """Slot holding this y-centre, registering it if it is new. Slots are
        registered for every label seen, decoded or not, so a label that never
        reads still holds its place and the ones below keep their column."""
        for i, sy in enumerate(self.slots_y):
            if abs(cy - sy) <= self.tol:
                return i

        at = bisect.bisect(self.slots_y, cy)
        self.slots_y.insert(at, cy)
        if at < len(self.slots_y) - 1:      # inserted above known labels: the
            self.reads = {                  # slots below it all shift down one
                (k + 1 if k >= at else k): v for k, v in self.reads.items()
            }
        return at

    def slot_index(self, label):
        self.tol = max(self.tol, (label[3] - label[1]) * 0.5, 20.0)
        return self._slot_for_y(box_center(label)[1])

    def merge(self, other):
        """Fold another record of the same column into this one."""
        self.tol = max(self.tol, other.tol)
        for i, cy in enumerate(other.slots_y):
            slot = self._slot_for_y(cy)
            if i in other.reads and slot not in self.reads:
                self.reads[slot] = other.reads[i]
        self.first_frame = min(self.first_frame, other.first_frame)
        self.seq = min(self.seq, other.seq)

    # ── state ─────────────────────────────────────────────────────────────
    def texts(self):
        """Payloads in slot order, None where a label never decoded — so each
        text keeps the QR DATA column it belongs to."""
        return [self.reads[i][0] if i in self.reads else None
                for i in range(len(self.slots_y))]

    def complete(self, expect, check=None):
        """Has this column read everything that is going to be asked of it?

        With only some positions being checked, the rest are never decoded, so
        waiting for them would hold the verdict back until the column left the
        zone."""
        if len(self.slots_y) < (expect or 0):
            return False
        wanted = (range(len(self.slots_y)) if check is None
                  else [i for i in check if i < len(self.slots_y)])
        return all(i in self.reads for i in wanted)


class CenterLineQRDecoder:
    """Read each column of labels once as it travels through a decode zone.

    The zone is a vertical band `zone` px either side of the trigger line. A
    label is retried for every frame it spends anywhere in that band, so a code
    that is blurred or badly lit on one frame gets another chance on the next;
    widening the zone buys more decode time.

    A wide zone holds several columns at once and every column has labels at
    the same heights, so labels are first grouped into columns by x-centre and
    each column is followed across frames by its position. Position tracking
    alone is not trusted, though: it breaks when the web moves further between
    frames than the tracker expects, and a broken track used to mean the same
    column was read and judged twice. Instead, as soon as any label in a
    column decodes, `identify` resolves the payload to the row it belongs to
    and that becomes the column's identity. Columns sharing an identity are
    merged, and an identity that has already been judged is never judged
    again.
    """

    def __init__(self, line_x, label_cls, qr_cls, margin=0.15, min_px=8,
                 on_decode=None, on_batch=None, on_label=None, dump_dir=None,
                 half_width=0, zone=None, column_gap=None, expect=None,
                 source=LABEL, identify=None, memory=900, min_dwell=3,
                 check=None):
        self.line_x = line_x
        self.half_width = half_width      # trigger band, drawn on the overlay
        self.zone = half_width if zone is None else zone
        self.column_gap = column_gap      # x spread within one column; None =
                                          # derive it from the label width
        self.expect = expect              # labels per column
        self.check = check                # slot indices to read; None = all
        self.label_cls = label_cls
        self.qr_cls = qr_cls
        self.margin = margin
        self.min_px = min_px
        self.source = source              # what goes to zxing: LABEL or QR
        self.identify = identify          # text -> hashable row key, or None
        self.memory = memory              # frames to remember a judged column

        self.on_decode = on_decode        # (text, index) -> status for overlay
        self.on_batch = on_batch          # (texts top-to-bottom) per column
        self.on_label = on_label          # (frame, label box, text) per decode
        self.dump_dir = dump_dir

        self.tracked = []          # columns currently in the zone
        self.judged_keys = {}      # identity -> frame it was judged on
        self.next_seq = 0
        self.frame_no = 0
        self.drift = 0.0           # mean x movement per frame; its sign is the
                                   # web's travel direction
        self.frames_steady = 0     # consecutive frames the speed agreed
        self.settled = False       # is the measured speed trustworthy yet?
        self.results = []          # [(crop_box, text_or_None)] for drawing
        self.pending = 0           # labels in the zone still undecoded
        self.count = 0             # successful decodes so far
        self.duplicates = 0        # re-tracked columns suppressed
        self.min_dwell = min_dwell # frames a column should spend in the zone
        self.thin_run = 0          # consecutive columns that fell short
        self.unsettled_run = 0     # consecutive columns judged without a lock
        self.dwells = []           # frames the last few columns were tracked
        self.label_w = 0.0         # median label width seen in the zone
        self.frame_w = 0           # frame width, for the zone advice

    @property
    def occupied(self):
        return bool(self.tracked)

    @property
    def step(self):
        """How far the web moves between frames, in pixels."""
        return abs(self.drift)

    @property
    def frames_in_zone(self):
        """How many frames a column is inside the zone at the speed measured.

        This is the number that matters for tuning: it is how many decode
        attempts each label gets. It is worked out from the geometry rather
        than observed, because a column that reads on its first attempt is
        judged and released immediately — a short observed life means the
        codes read easily, not that there was no time to read them.
        """
        if self.step < 0.5:
            return 0.0
        return (2 * self.zone + self.label_w) / self.step

    @property
    def frames_per_column(self):
        """Frames the recent columns were actually tracked for before being
        judged — how long reading them took, not how long there was."""
        return sum(self.dwells) / len(self.dwells) if self.dwells else 0.0

    def zone_for(self, frames):
        """Zone half-width that would give a column `frames` frames at the
        speed being measured now."""
        return int(max((frames * self.step - self.label_w) / 2.0, 0))

    @property
    def head(self):
        """The column next in line to be judged — or, once every column in the
        zone is judged, the most recent, so the overlay keeps showing it."""
        ordered = self._travel_order(self.tracked)
        for column in ordered:
            if not column.judged:
                return column
        return ordered[-1] if ordered else None

    # ── grouping ──────────────────────────────────────────────────────────
    def _zone_labels(self, dets):
        lo, hi = self.line_x - self.zone, self.line_x + self.zone
        return [d for d in dets
                if int(d[5]) == self.label_cls and d[0] <= hi and d[2] >= lo]

    def _gap(self, labels):
        """How far apart two labels' x-centres may be and still be one column.
        Derived from the label width, so it scales with the camera setup."""
        widths = sorted(d[2] - d[0] for d in labels)
        self.label_w = widths[len(widths) // 2]
        if self.column_gap:
            return self.column_gap
        return max(self.label_w * 0.7, 20.0)

    def _columns(self, labels, gap):
        """Split the labels in the zone into columns by x-centre."""
        ordered = sorted(labels, key=lambda d: box_center(d)[0])
        groups, current = [], [ordered[0]]
        for label in ordered[1:]:
            if box_center(label)[0] - box_center(current[-1])[0] > gap:
                groups.append(current)
                current = [label]
            else:
                current.append(label)
        groups.append(current)
        return groups

    def _travel_order(self, columns):
        """Columns ordered by how far along they are, most advanced first, so
        verdicts come out in the order the labels pass the camera. Creation
        order is not enough: at start-up several columns are already in view
        and are picked up in the same frame."""
        if self.drift > 0.5:
            return sorted(columns, key=lambda c: -c.x)
        if self.drift < -0.5:
            return sorted(columns, key=lambda c: c.x)
        return sorted(columns, key=lambda c: c.seq)

    def _shift(self, xs, gap):
        """How far everything moved since the last frame.

        Measured from the whole frame at once rather than per column: every
        offset between a tracked column and a column seen now is a candidate,
        and the one the most pairs agree on wins. Estimating it this way is
        what lets a fast web be tracked at all — deriving the step from
        matches alone cannot start, because matching needs the step first.

        Once a speed has been established it is also used to throw candidates
        out. With a single column in view at a time the vote is degenerate —
        the one offset on offer always wins, so the next column arriving looks
        exactly like the current one having moved. A web does not reverse or
        change speed from one frame to the next, so an offset that disagrees
        with the measured speed is the wrong pairing, and dead reckoning is
        the better guess.
        """
        if not self.tracked or not xs:
            return self.drift

        offsets = [x - c.x for c in self.tracked for x in xs]
        window = max(gap * 0.4, 30.0)

        if self.settled:
            allowed = max(abs(self.drift) * 0.6, window)
            offsets = [o for o in offsets if abs(o - self.drift) <= allowed]
            if not offsets:
                return self.drift          # nothing plausible: dead reckon

        best, votes = offsets[0], 0
        for candidate in offsets:
            agree = sum(1 for o in offsets if abs(o - candidate) <= window)
            if agree > votes:
                best, votes = candidate, agree
        return best

    def _associate(self, clusters, gap):
        """Tie this frame's columns to the ones already tracked, by predicted
        position. Identity merging downstream repairs whatever this misses."""
        xs = [sum(box_center(d)[0] for d in cluster) / len(cluster)
              for cluster in clusters]
        shift = self._shift(xs, gap)
        if self.tracked:
            steady = abs(shift - self.drift) <= max(abs(self.drift) * 0.5, gap * 0.4)
            self.frames_steady = self.frames_steady + 1 if steady else 0
            self.settled = self.frames_steady >= 2
            self.drift = 0.5 * self.drift + 0.5 * shift
        else:
            self.drift = shift

        unmatched = list(self.tracked)
        tol = max(gap * 0.6, 40.0)
        pairs, alive = [], []

        for cluster, x in zip(clusters, xs):
            best, best_d = None, None
            for column in unmatched:
                d = abs((column.x + shift) - x)
                if best_d is None or d < best_d:
                    best, best_d = column, d

            # A column that has already read everything has nothing to gain
            # from being followed further, so when the fit is loose prefer to
            # treat this as the next column arriving. If that guess is wrong,
            # identity merging catches it.
            if (best is not None and best.complete(self.expect, self.check)
                    and best_d > gap * 0.4):
                best = None

            if best is not None and best_d <= tol:
                unmatched.remove(best)
            else:
                best = Column(self.next_seq, x, self.frame_no)
                self.next_seq += 1

            best.x = x
            best.frames += 1
            best.last_frame = self.frame_no
            pairs.append((best, cluster))
            alive.append(best)

        for gone in self._travel_order(unmatched):
            self._judge(gone)          # tracked last frame, gone now: it left
        self.tracked = alive
        return pairs

    # ── identity ──────────────────────────────────────────────────────────
    def _reconcile(self):
        """Give every column an identity where one can be resolved, merge the
        ones that turn out to be the same column, and silence any that repeat
        a column already judged."""
        if self.identify is None:
            return

        for column in list(self.tracked):
            if column.key is not None:
                continue
            for slot in sorted(column.reads):
                key = self.identify(column.reads[slot][0])
                if key is not None:
                    column.key = key
                    break

        seen = {}
        keep = []
        for column in sorted(self.tracked, key=lambda c: c.seq):
            key = column.key
            if key is None:
                keep.append(column)
                continue
            if key in seen:                      # two records, one column
                seen[key].merge(column)
                continue
            if key in self.judged_keys and not column.judged:
                column.judged = True             # a re-track of a judged column
                self.duplicates += 1
            seen[key] = column
            keep.append(column)
        self.tracked = keep

    # ── judging ───────────────────────────────────────────────────────────
    def _judge(self, column):
        """Hand a finished column over for validation, once and only once."""
        if column.judged:
            return
        column.judged = True
        self._check_dwell(column)
        if column.key is not None:
            if column.key in self.judged_keys:
                self.duplicates += 1
                return
            self.judged_keys[column.key] = self.frame_no
        if self.on_batch and column.slots_y:
            self.on_batch(column.texts())

    def _judge_ready(self):
        """Judge completed columns, most advanced first."""
        if abs(self.drift) < 0.5 and len(self.tracked) > 1:
            # Which column leads depends on which way the web runs, and that is
            # only known once something has moved. With more than one column in
            # the zone, hold for a frame rather than risk judging back to front.
            return
        for column in self._travel_order(self.tracked):
            if column.judged:
                continue
            if not column.complete(self.expect, self.check):
                break         # wait for it, so verdicts stay in arrival order
            self._judge(column)

    def _check_dwell(self, column):
        """Warn when the zone is too short for the speed the web is running.

        Every label needs several attempts to read reliably, and a column that
        is only in the zone for a frame or two can be missed altogether — which
        shows up downstream as an out-of-sequence fault rather than anything
        that names the real cause. Say the real cause.
        """
        self.dwells.append(column.last_frame - column.first_frame + 1)
        del self.dwells[:-10]

        if not self.settled:
            # No stable speed was ever measured for this column, so the
            # figures below cannot be trusted either. Columns are passing
            # faster than consecutive frames can be tied together, which
            # loses whole columns rather than merely reading them badly.
            self.unsettled_run += 1
            # Not at the first few: the tracker always takes a column or two
            # to find the speed, and a start-up wobble is not a fault.
            if self.unsettled_run in (8, 40) or self.unsettled_run % 400 == 0:
                print("[qr] cannot lock onto the web's motion — columns are "
                      "crossing faster than the frame rate can follow, so some "
                      "will be missed. Raise FPS (smaller --imgsz or capture "
                      "size), widen --zone, or slow the web.")
            return
        self.unsettled_run = 0

        available = self.frames_in_zone
        if not available or available >= self.min_dwell:
            self.thin_run = 0
            return
        self.thin_run += 1
        if self.thin_run not in (3, 30) and self.thin_run % 300:
            return

        want = self.zone_for(self.min_dwell)
        room = self.frame_w // 2 if self.frame_w else 0
        advice = (f"try --zone {want}" if room and want <= room else
                  "no zone width fits that on this frame — raise FPS "
                  "(smaller --imgsz or capture size) or slow the web")
        print(f"[qr] a column is only in the zone for {available:.1f} frame(s) "
              f"at {self.step:.0f} px/frame — too few attempts per label: "
              f"{advice}")

    def _forget(self):
        """Drop judged identities once they are far enough in the past that
        the same column cannot still be in front of the camera."""
        if len(self.judged_keys) < 64:
            return
        cutoff = self.frame_no - self.memory
        self.judged_keys = {k: f for k, f in self.judged_keys.items()
                            if f >= cutoff}

    # ── main entry point ──────────────────────────────────────────────────
    def update(self, frame, dets):
        """Run one frame's detections through the zone.
        Returns the list of texts decoded on THIS frame (may be empty)."""
        self.frame_no += 1
        self.frame_w = frame.shape[1] if hasattr(frame, "shape") else self.frame_w
        self._forget()

        labels = self._zone_labels(dets)
        if not labels:
            for column in self._travel_order(self.tracked):
                self._judge(column)
            self.tracked = []
            self.results = []
            self.pending = 0
            return []

        gap = self._gap(labels)
        pairs = self._associate(self._columns(labels, gap), gap)

        qr_dets = [d for d in dets if int(d[5]) == self.qr_cls]
        new_texts = []
        self.results = []
        self.pending = 0

        for column, cluster in pairs:
            if column.judged:
                continue
            for label in sorted(cluster, key=lambda d: box_center(d)[1]):
                index = column.slot_index(label)
                if index in column.reads:
                    continue
                if self.check is not None and index not in self.check:
                    continue          # this position is not being checked

                qr = pick_qr_for_label(label[:4], qr_dets)
                text, crop_box = self._decode(frame, label, qr)
                if crop_box is None:
                    self.pending += 1      # nothing to hand to zxing at all
                    continue
                self.results.append((crop_box, text))

                if not text:
                    self.pending += 1      # retry next frame, still in the zone
                    if self.dump_dir:
                        self._dump(frame, crop_box)
                    continue

                # Saved from the frame handed to update(), which is still clean
                # — the detection boxes are drawn onto it after this returns.
                if self.on_label:
                    self.on_label(frame, label[:4], text)

                status = self.on_decode(text, index) if self.on_decode else None
                column.reads[index] = (text, status)
                self.count += 1
                new_texts.append(text)

        self._reconcile()
        self._judge_ready()
        return new_texts

    def reset(self):
        """Forget what is in the zone without judging it — for a restart,
        where the web has moved while the camera was not watching."""
        self.tracked = []
        self.results = []
        self.pending = 0
        self.drift = 0.0
        self.frames_steady = 0
        self.settled = False

    def _decode(self, frame, label, qr):
        """Hand a crop to zxing and return (text, crop_box).

        With LABEL the whole detected label goes in — the QR's printed quiet
        zone comes along with it, and a label still decodes when the qr_code
        detector misses its box entirely. The label box is used exactly as
        detected, with no margin: growing it can pull a neighbouring label's
        code into the crop, which would decode as the wrong payload. A tight
        qr crop is tried afterwards if the label crop came up empty.
        """
        if self.source == QR:
            if qr is None:
                return None, None
            return decode_qr(frame, qr[:4], self.margin, self.min_px)

        text, box = decode_qr(frame, label[:4], margin=0.0, min_px=0)
        if text or qr is None:
            return text, box
        return decode_qr(frame, qr[:4], self.margin, self.min_px)

    def _dump(self, frame, box):
        import os
        import time
        os.makedirs(self.dump_dir, exist_ok=True)
        x1, y1, x2, y2 = box
        path = os.path.join(self.dump_dir, f"miss_{time.time():.3f}.png")
        cv2.imwrite(path, frame[y1:y2, x1:x2])

    # ── overlay ───────────────────────────────────────────────────────────
    def draw(self, frame):
        """Draw the zone, the crops in use and the head column's results."""
        h = frame.shape[0]
        head = self.head
        if not self.tracked:
            color = (0, 0, 255)         # red    – zone empty
        elif self.pending:
            color = (0, 255, 255)       # yellow – still decoding
        else:
            color = (0, 255, 0)         # green  – everything in the zone read

        if self.zone > 0:               # zone edges, then the trigger line
            for x in (self.line_x - self.zone, self.line_x + self.zone):
                cv2.line(frame, (int(x), 0), (int(x), h), color, 1)
        cv2.line(frame, (self.line_x, 0), (self.line_x, h), color, 3)

        for (x1, y1, x2, y2), text in self.results:
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (255, 0, 255) if text else (0, 0, 255), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        read = len(head.reads) if head else 0
        cv2.putText(frame, f"QR total: {self.count}  in zone: "
                           f"{len(self.tracked)} col / {read} read"
                           f" / {self.pending} pending", (20, 80),
                    font, 1.0, (255, 0, 255), 2, cv2.LINE_AA)

        # What the tuning actually depends on: how far the web moves per frame
        # and how many frames that leaves each column to be read in.
        available = self.frames_in_zone
        thin = bool(available) and available < self.min_dwell
        stats = (f"web {self.step:.0f} px/frame   zone {available:.1f} frames"
                 f"   read in {self.frames_per_column:.1f}   dup {self.duplicates}")
        if thin:
            stats += f"   -> --zone {self.zone_for(self.min_dwell)}"
        cv2.putText(frame, stats, (20, 116), font, 0.7,
                    (0, 0, 255) if thin else (255, 0, 255), 2, cv2.LINE_AA)

        # One line per column, always D1..Dn top to bottom, whether or not that
        # label has read yet — the list keeps its shape instead of renumbering
        # as labels arrive in the zone.
        slots = len(head.slots_y) if head else 0
        for i in range(min(max(slots, self.expect or 0), 8)):
            if head is None or i not in head.reads:
                off = self.check is not None and i not in self.check
                cv2.putText(frame, f"D{i + 1}. {'not checked' if off else '...'}",
                            (20, 156 + i * 34), font, 0.8,
                            (110, 110, 110) if off else (160, 160, 160), 2,
                            cv2.LINE_AA)
                continue
            text, status = head.reads[i]
            # The sheet position goes first so it is never the part that runs
            # off the edge — reading "D2 [row 7 D3]" is the whole point.
            if status is None:
                lead, tcolor = "", (255, 0, 255)
            else:
                note, good = status
                lead = f"[{note}] "
                tcolor = (0, 255, 0) if good else (0, 0, 255)
            room = max(48 - len(lead), 12)
            shown = text if len(text) <= room else text[:room - 3] + "..."
            cv2.putText(frame, f"D{i + 1}. {lead}{shown}", (20, 156 + i * 34),
                        font, 0.8, tcolor, 2, cv2.LINE_AA)
        return frame
