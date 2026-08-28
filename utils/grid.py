"""Spot label detections the model missed, using the geometry of the sheet.

The labels arrive as a regular grid that flows horizontally, so every row of
labels sits on a 1-D lattice: the x-centres of one row are (near enough)
`x0 + k * pitch`. That makes a miss visible without any tracker — if the model
drops a label at high speed, the row it belonged to shows a gap of ~2x (or 3x…)
the normal spacing, i.e. an empty lattice slot.

Two checks run on every frame:

  interior  a gap inside a row that is a near-integer multiple of the pitch
            -> that many empty slots between the two neighbours.
  end       a row that is short at one end compared with the other rows,
            while the missing slot would still sit comfortably inside the
            frame (labels leaving/entering the frame are not misses).

The pitch is not hard-coded: it is measured from the smallest gaps seen in
recent frames (the 10th-percentile band), so it survives frames that are full
of misses and adapts if the camera height or lens changes.
"""

from collections import deque

import cv2
import numpy as np


def centers(dets):
    """(cx, cy) arrays for [x1,y1,x2,y2,...] rows."""
    return (dets[:, 0] + dets[:, 2]) * 0.5, (dets[:, 1] + dets[:, 3]) * 0.5


def cluster_1d(values, tol):
    """Group 1-D values into runs separated by more than `tol`.
    Returns a list of index arrays, ordered by ascending value."""
    order = np.argsort(values)
    groups, cur = [], [order[0]]
    for i in order[1:]:
        if values[i] - values[cur[-1]] <= tol:
            cur.append(i)
        else:
            groups.append(np.array(cur))
            cur = [i]
    groups.append(np.array(cur))
    return groups


class MissingLabelDetector:
    """Flag lattice slots in the label grid that carry no detection.

    Usage per frame::

        misses = checker.update(dets, frame.shape)   # list of dicts
        checker.draw(frame)

    Each miss is ``{"x", "y", "w", "h", "row", "kind"}`` in frame pixels;
    ``kind`` is ``"interior"`` or ``"end"``.
    """

    def __init__(self, label_cls=0, pitch=None, history=120, row_tol=0.6,
                 lattice_tol=0.30, edge_px=8, min_row_labels=2, end_check=True):
        self.label_cls    = label_cls
        self.fixed_pitch  = pitch is not None
        self.pitch        = pitch          # px between neighbouring labels in a row
        self.row_tol      = row_tol        # row grouping tol, x label height
        self.lattice_tol  = lattice_tol    # how far off an integer a gap may be
        self.edge_px      = edge_px        # boxes this close to a border are clipped
        self.min_row_labels = min_row_labels
        self.end_check    = end_check

        self._pitch_samples = deque(maxlen=history)

        # per-frame state, kept for draw()
        self.misses   = []
        self.rows     = 0
        self.n_labels = 0

        # cumulative stats
        self.frames         = 0     # frames seen
        self.frames_checked = 0     # frames the geometry check could run on
        self.frames_with_miss = 0
        self.total_missing  = 0     # sum of empty slots over all frames
        self.max_missing    = 0     # worst single frame

    # ── pitch ────────────────────────────────────────────────────────────
    def _update_pitch(self, gaps):
        """Feed this frame's neighbour gaps into the running pitch estimate.

        The true pitch is the *smallest* recurring gap: a gap of 2x means a
        label was dropped, never that the sheet stretched. So the estimate is
        the median of the band around the 10th percentile, which ignores the
        doubled/tripled gaps entirely."""
        if self.fixed_pitch or len(gaps) == 0:
            return
        gaps = np.asarray(gaps, dtype=np.float64)
        base = np.percentile(gaps, 10)
        band = gaps[(gaps >= base * 0.70) & (gaps <= base * 1.45)]
        if band.size:
            self._pitch_samples.append(float(np.median(band)))
            self.pitch = float(np.median(self._pitch_samples))

    # ── main entry ───────────────────────────────────────────────────────
    def update(self, dets, frame_shape):
        """Check one frame's detections. Returns this frame's miss list."""
        self.frames += 1
        self.misses = []

        labels = dets[dets[:, 5].astype(int) == self.label_cls] if len(dets) else dets
        self.n_labels = len(labels)
        if self.n_labels < 2:
            self.rows = 0
            return self.misses

        h, w = frame_shape[:2]
        cx, cy = centers(labels)
        bw = np.median(labels[:, 2] - labels[:, 0])
        bh = np.median(labels[:, 3] - labels[:, 1])

        # A box touching a vertical border is clipped, so its centre is pulled
        # inwards and would fake a gap. Keep it out of the lattice maths.
        clipped = (labels[:, 0] <= self.edge_px) | (labels[:, 2] >= w - self.edge_px)

        row_groups = cluster_1d(cy, bh * self.row_tol)
        self.rows = len(row_groups)

        # First pass: neighbour gaps feed the pitch estimate.
        rows = []
        gaps = []
        for g in row_groups:
            keep = g[~clipped[g]]
            xs = np.sort(cx[keep])
            rows.append((float(np.median(cy[g])), xs))
            if xs.size >= 2:
                gaps.extend(np.diff(xs).tolist())
        self._update_pitch(gaps)

        if not self.pitch or self.pitch <= 1:
            return self.misses

        self.frames_checked += 1

        # Second pass: place each row on the lattice and look for empty slots.
        lattices = []      # (row_y, x0, set_of_k, k_min, k_max)
        for row_y, xs in rows:
            if xs.size < self.min_row_labels:
                lattices.append((row_y, None, set(), 0, -1))
                continue

            x0 = float(xs[0])
            ks = np.rint((xs - x0) / self.pitch).astype(int)
            present = set(ks.tolist())

            for i in range(xs.size - 1):
                # A gap that is not a clean multiple of the pitch is a skewed
                # or mis-sized box, not a dropped label — don't cry wolf.
                a, b = int(ks[i]), int(ks[i + 1])
                span = b - a
                if span < 2:
                    continue
                if abs((xs[i + 1] - xs[i]) / self.pitch - span) > self.lattice_tol * span:
                    continue
                for k in range(a + 1, b):
                    self._add_miss(x0 + k * self.pitch, row_y, bw, bh,
                                   len(lattices), "interior")
                    present.add(k)

            lattices.append((row_y, x0, present, int(ks.min()), int(ks.max())))

        if self.end_check:
            self._check_ends(lattices, bw, bh, w)

        n = len(self.misses)
        if n:
            self.frames_with_miss += 1
            self.total_missing += n
            self.max_missing = max(self.max_missing, n)
        return self.misses

    def _add_miss(self, x, y, bw, bh, row, kind):
        self.misses.append({"x": float(x), "y": float(y), "w": float(bw),
                            "h": float(bh), "row": int(row), "kind": kind})

    def _check_ends(self, lattices, bw, bh, frame_w):
        """A row that is short at one end while its neighbours are not.

        Rows all cross the frame together, so they should carry the same
        number of labels. A slot is only called a miss when it would land
        well inside the frame — a label half-way out of view is not a miss.
        """
        usable = [l for l in lattices if l[1] is not None]
        if len(usable) < 2:
            return

        lead  = np.median([l[1] + l[3] * self.pitch for l in usable])   # left edge
        trail = np.median([l[1] + l[4] * self.pitch for l in usable])   # right edge
        inset = bw * 0.60          # a slot must sit this far inside the frame

        for row, (row_y, x0, present, k_min, k_max) in enumerate(lattices):
            if x0 is None:
                continue
            x_lo = x0 + k_min * self.pitch
            while x_lo - lead > self.pitch * (1 - self.lattice_tol):
                x_lo -= self.pitch
                k_min -= 1
                if x_lo - bw * 0.5 < inset:
                    break
                self._add_miss(x_lo, row_y, bw, bh, row, "end")

            x_hi = x0 + k_max * self.pitch
            while trail - x_hi > self.pitch * (1 - self.lattice_tol):
                x_hi += self.pitch
                k_max += 1
                if x_hi + bw * 0.5 > frame_w - inset:
                    break
                self._add_miss(x_hi, row_y, bw, bh, row, "end")

    # ── reporting ────────────────────────────────────────────────────────
    def summary(self):
        rate = (100.0 * self.frames_with_miss / self.frames_checked
                if self.frames_checked else 0.0)
        pitch = f"{self.pitch:.1f}px" if self.pitch else "n/a"
        return (f"frames={self.frames} checked={self.frames_checked} pitch={pitch} "
                f"| frames with a miss: {self.frames_with_miss} ({rate:.1f}%) "
                f"| empty slots seen: {self.total_missing} "
                f"| worst frame: {self.max_missing}")

    def draw(self, frame):
        """Mark every empty slot with a red box + banner."""
        for m in self.misses:
            x1 = int(m["x"] - m["w"] * 0.5)
            y1 = int(m["y"] - m["h"] * 0.5)
            x2 = int(m["x"] + m["w"] * 0.5)
            y2 = int(m["y"] + m["h"] * 0.5)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, f"MISSING ({m['kind']})", (x1, max(y1 - 8, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        color = (0, 0, 255) if self.misses else (0, 255, 0)
        pitch = f"{self.pitch:.0f}px" if self.pitch else "…"
        cv2.putText(frame,
                    f"labels: {self.n_labels}  rows: {self.rows}  pitch: {pitch}"
                    f"  missing: {len(self.misses)}  (total {self.total_missing})",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        return frame
