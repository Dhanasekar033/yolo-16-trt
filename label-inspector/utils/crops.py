"""Save the label crop behind every decode, as a record of what was read.

Files land in

    <root>/<xlsx name>/<decoded payload>_<timestamp>.jpg

so a run against validation.xlsx fills labels/validation/. `root` is the
folder the operator chose from the console and holds nothing but crops — the
record that goes with them is written into the project instead. The payload is
slugified — a QR carries a URL, and '/' and ':' can't go in a filename — but
stays readable enough to match a file back to the code it holds.
"""

import os
import re
import time

import cv2

UNSAFE = re.compile(r"[^A-Za-z0-9]+")


def slugify(text, max_len=80):
    """Filename-safe form of a payload, e.g.
    'https://scan.smartqr.io/LS5/7016' -> 'https_scan_smartqr_io_LS5_7016'."""
    slug = UNSAFE.sub("_", (text or "").strip()).strip("_")
    if len(slug) > max_len:                  # keep the tail: that's the part
        slug = slug[-max_len:].lstrip("_")   # that differs between codes
    return slug or "unknown"


def timestamp():
    """Sortable stamp with milliseconds, e.g. 20260828-114530-472."""
    now = time.time()
    return time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + \
        f"-{int((now % 1) * 1000):03d}"


class LabelSaver:
    """Writes one image per decoded label into a per-sheet folder."""

    def __init__(self, root="labels", name="run", subdir=None, ext="jpg",
                 pad=0.0, min_pad=0, quality=95):
        self.dir = os.path.join(root, name, subdir) if subdir \
            else os.path.join(root, name)
        self.ext = ext.lstrip(".")
        self.pad = pad
        self.min_pad = min_pad
        self.params = ([cv2.IMWRITE_JPEG_QUALITY, quality]
                       if self.ext in ("jpg", "jpeg") else [])
        self.count = 0
        # Made on the first write, not here: loading a sheet and then pointing
        # the crops somewhere else is two operations, and the folder for the
        # in-between combination should not be left behind empty.
        self._made = False
        print(f"[crops] saving label crops to {self.dir}/")

    def _box(self, box, shape):
        h, w = shape[:2]
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        px = max(int(round((x2 - x1) * self.pad)), self.min_pad)
        py = max(int(round((y2 - y1) * self.pad)), self.min_pad)
        return (max(int(x1) - px, 0), max(int(y1) - py, 0),
                min(int(x2) + px, w), min(int(y2) + py, h))

    def save(self, frame, box, text):
        """Crop `box` out of `frame` and write it. Returns the path, or None
        if the box was degenerate."""
        x1, y1, x2, y2 = self._box(box, frame.shape)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None

        if not self._made:
            os.makedirs(self.dir, exist_ok=True)
            self._made = True

        name = f"{slugify(text)}_{timestamp()}.{self.ext}"
        path = os.path.join(self.dir, name)
        n = 1
        while os.path.exists(path):          # same code twice inside a ms
            path = os.path.join(self.dir, f"{name[:-len(self.ext) - 1]}"
                                          f"_{n}.{self.ext}")
            n += 1

        if not cv2.imwrite(path, frame[y1:y2, x1:x2], self.params):
            print(f"[crops] failed to write {path}")
            return None
        self.count += 1
        return path
