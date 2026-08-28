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


def decode_qr(frame, box, margin=0.15, min_px=8, min_side=160):
    """Crop `box` (+margin) out of `frame` and try to decode a QR from it.

    Tries the plain BGR crop first, then a grayscale copy upscaled to at least
    `min_side` px, then an Otsu-binarized version — small/low-contrast crops
    off a moving line often only read after one of the fallbacks.
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


class CenterLineQRDecoder:
    """Decode every label that crosses a vertical line — once each, no tracker.

    Several labels can sit on the line at the same time (a whole column of a
    tray, for instance), so the latch is per label rather than per crossing:
    labels are told apart by the y-centre of their box, which barely moves
    while they travel across a vertical line. A label is retried every frame
    until it decodes, then skipped for the rest of the crossing. The whole set
    -- including the decoded texts shown in the overlay -- is cleared once the
    line goes clear again, ready for the next batch.
    """

    def __init__(self, line_x, label_cls, qr_cls, margin=0.15, min_px=8,
                 on_decode=None, on_batch=None, dump_dir=None,
                 half_width=0, expect=None):
        self.line_x = line_x
        self.half_width = half_width  # widen the line into a band, in pixels
        self.expect = expect          # labels per crossing; once that many have
                                      # read, the crossing is judged right away
        self.label_cls = label_cls
        self.qr_cls = qr_cls
        self.margin = margin
        self.min_px = min_px
        self.on_decode = on_decode   # (text, index) -> status str for overlay
        self.on_batch = on_batch     # (texts top-to-bottom) once per crossing
        self.dump_dir = dump_dir

        self.occupied = False      # is any label on the line right now?
        self.finalized = False     # crossing already judged (all labels read)
        self.slots_y = []          # y-centres of this crossing's labels, sorted
        self.done = {}             # {slot index: (text, status)} this crossing
        self.results = []          # [(crop_box, text_or_None)] for drawing
        self.pending = 0           # labels on the line still undecoded
        self.count = 0             # successful decodes so far

    def _crossing_labels(self, dets):
        """Labels overlapping the trigger band, ordered top to bottom.

        The band is `half_width` px either side of the line. Labels printed at
        a slight angle reach a zero-width line one at a time and can leave a
        gap where none is touching it, which would split one physical row into
        two crossings; widening the band keeps the row together."""
        lo, hi = self.line_x - self.half_width, self.line_x + self.half_width
        labels = [d for d in dets
                  if int(d[5]) == self.label_cls and d[0] <= hi and d[2] >= lo]
        return sorted(labels, key=lambda d: box_center(d)[1])   # top to bottom

    def _slot_index(self, label):
        """Position of this label within the crossing, counted top to bottom.

        Slots are keyed by y-centre (which barely moves as a label travels
        across a vertical line) and registered the first time a label is seen,
        decoded or not — so a label that never decodes still holds its place
        and the ones below it keep their QR DATA column.
        """
        cy = box_center(label)[1]
        tol = max((label[3] - label[1]) * 0.5, 20.0)
        for i, sy in enumerate(self.slots_y):
            if abs(cy - sy) <= tol:
                return i

        at = bisect.bisect(self.slots_y, cy)
        self.slots_y.insert(at, cy)
        if at < len(self.slots_y) - 1:      # inserted above known labels: the
            self.done = {                   # slots below it all shift down one
                (k + 1 if k >= at else k): v for k, v in self.done.items()
            }
        return at

    def update(self, frame, dets):
        """Run one frame's detections through the line logic.
        Returns the list of texts decoded on THIS frame (may be empty)."""
        labels = self._crossing_labels(dets)

        if not labels:
            # Line is clear: this batch has passed. Hand the whole crossing
            # over (top to bottom, None where a label never decoded) for
            # validation, then wipe the decoded texts so the overlay goes
            # blank until the next label reaches the line.
            if self.occupied and not self.finalized and self.on_batch:
                self.on_batch(self.batch_texts())
            self.occupied = False
            self.finalized = False
            self.slots_y = []
            self.done = {}
            self.results = []
            self.pending = 0
            return []

        self.occupied = True
        if self.finalized:
            return []              # judged already; wait for the line to clear

        qr_dets = [d for d in dets if int(d[5]) == self.qr_cls]
        new_texts = []
        self.results = []
        self.pending = 0

        for label in labels:
            index = self._slot_index(label)
            if index in self.done:
                continue

            qr = pick_qr_for_label(label[:4], qr_dets)
            if qr is None:
                self.pending += 1
                continue

            text, crop_box = decode_qr(frame, qr[:4], self.margin, self.min_px)
            self.results.append((crop_box, text))

            if not text:
                self.pending += 1        # retry next frame, still on the line
                if self.dump_dir:
                    self._dump(frame, crop_box)
                continue

            status = self.on_decode(text, index) if self.on_decode else None
            self.done[index] = (text, status)
            self.count += 1
            new_texts.append(text)

        # Judge the crossing the moment every label has read, rather than
        # waiting for it to clear the line — the verdict lands while the row
        # is still in front of the camera.
        if (self.expect and not self.finalized
                and len(self.slots_y) >= self.expect
                and len(self.done) >= len(self.slots_y)):
            self.finalized = True
            self.pending = 0
            if self.on_batch:
                self.on_batch(self.batch_texts())

        return new_texts

    def batch_texts(self):
        """This crossing's payloads in slot order, None for labels that never
        decoded — so each text keeps the QR DATA column it belongs to."""
        return [self.done[i][0] if i in self.done else None
                for i in range(len(self.slots_y))]

    def _dump(self, frame, box):
        import os
        import time
        os.makedirs(self.dump_dir, exist_ok=True)
        x1, y1, x2, y2 = box
        path = os.path.join(self.dump_dir, f"miss_{time.time():.3f}.png")
        cv2.imwrite(path, frame[y1:y2, x1:x2])

    def draw(self, frame):
        """Overlay the center line, the crop boxes in use and the results."""
        h = frame.shape[0]
        if not self.occupied:
            color = (0, 0, 255)         # red    – waiting for labels
        elif self.finalized:
            color = (255, 255, 0)       # cyan   – crossing judged
        elif self.pending:
            color = (0, 255, 255)       # yellow – still decoding this batch
        else:
            color = (0, 255, 0)         # green  – every label on the line done
        cv2.line(frame, (self.line_x, 0), (self.line_x, h), color, 3)

        for (x1, y1, x2, y2), text in self.results:
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (255, 0, 255) if text else (0, 0, 255), 2)

        cv2.putText(frame, f"QR total: {self.count}  on line: {len(self.done)}"
                           f" ok / {self.pending} pending", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2, cv2.LINE_AA)
        for i in range(min(len(self.slots_y), 8)):
            if i not in self.done:
                cv2.putText(frame, f"{i + 1}. ...", (20, 120 + i * 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160, 160, 160), 2,
                            cv2.LINE_AA)
                continue
            text, status = self.done[i]
            shown = text if len(text) <= 40 else text[:37] + "..."
            if status is None:
                mark, tcolor = "", (255, 0, 255)
            else:
                note, good = status
                mark = f"  [{note}]"
                tcolor = (0, 255, 0) if good else (0, 0, 255)
            cv2.putText(frame, f"{i + 1}. {shown}{mark}", (20, 120 + i * 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, tcolor, 2, cv2.LINE_AA)
        return frame
