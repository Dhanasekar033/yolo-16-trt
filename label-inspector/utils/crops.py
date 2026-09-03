"""Save the label crop behind every decode, as a record of what was read.

Files land in

    <root>/<xlsx name>/<decoded payload>_<timestamp>.jpg

so a run against validation.xlsx fills labels/validation/. `root` is the
folder the operator chose from the console and holds nothing but crops — the
record that goes with them is written into the project instead.

One file per decode, and only per decode: a label with no code on it is not
a record of anything read, and a folder of blanks is a folder nobody can
search. What stopped the line is said on the screen and in the run log.

The crop is the detection box as it came off the model, widened left and
right into the gutter between the lanes -- half of whatever room is actually
there, measured off the labels either side, so a code the box clipped comes
out whole without any of the neighbour coming with it. --label-pad and
--label-pad-px override that with a fixed margin on every side.

The payload is slugified — a QR carries a URL, and '/' and ':' can't go in a
filename — but stays readable enough to match a file back to the code it
holds.
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

    @staticmethod
    def _gaps(box, neighbours):
        """Clear space on each side of this label, in pixels.

        The web carries the labels in lanes with a gutter between them, and
        the detector's box is drawn to the label, sometimes a shade inside
        it -- which is how a crop comes out with the code sliced down one
        edge. The gutter is the room available to put that back, and the
        labels themselves are what measure it: the nearest label on each
        side is as far as a crop may spread.

        Nothing is assumed about how wide a label is, how far apart they
        sit, or what the picture is scaled to -- a reel with a different
        pitch, a lens moved, a different camera, all measure themselves.
        A side with nothing standing on it comes back None.
        """
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        gaps = {"left": None, "right": None, "up": None, "down": None}

        def keep(side, gap):
            if gaps[side] is None or gap < gaps[side]:
                gaps[side] = gap

        for other in neighbours:
            ox1, oy1, ox2, oy2 = (float(v) for v in other[:4])
            if (ox1, oy1, ox2, oy2) == (x1, y1, x2, y2):
                continue
            # Abreast of this one -- same lane, so its gutter is this one's.
            if min(y2, oy2) - max(y1, oy1) > 0:
                if ox2 <= x1:
                    keep("left", x1 - ox2)
                elif ox1 >= x2:
                    keep("right", ox1 - x2)
            # In line with it -- the label before or after it down the web.
            if min(x2, ox2) - max(x1, ox1) > 0:
                if oy2 <= y1:
                    keep("up", y1 - oy2)
                elif oy1 >= y2:
                    keep("down", oy1 - y2)
        return gaps

    def _box(self, box, shape, neighbours=(), motion=(0.0, 0.0)):
        """The crop rectangle: the detection box, clipped to the frame.

        Widened sideways into the gutter either side, by half of it, so a
        code the box clipped comes out whole and two neighbouring crops
        still meet without ever overlapping.

        Then `motion` is spent on the trailing edge. A box on a web standing
        still is a box on the label; on a moving one it lands late, and how
        late is how far the label travelled while the frame was being taken
        and the model was looking at it -- so the edge the crop cuts into is
        always the one the label is coming *from*. That distance is not a
        number anyone can write down: it is the speed of the winder, and the
        winder is turned up and down all shift. It is measured instead, off
        the labels themselves, frame against frame -- so the faster the web
        runs the further back the crop reaches, and on a coil that is
        standing still nothing is added at all.

        The trailing edge may take the whole gutter, and no more: past that
        is the next label, and a crop with two labels in it is worse than a
        crop with a clipped one.

        An explicit --label-pad / --label-pad-px overrides all of that and
        pads every side by what it says, because that is what asking for it
        means.
        """
        h, w = shape[:2]
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        if self.pad or self.min_pad:
            px = max(int(round((x2 - x1) * self.pad)), self.min_pad)
            py = max(int(round((y2 - y1) * self.pad)), self.min_pad)
            return (max(int(x1) - px, 0), max(int(y1) - py, 0),
                    min(int(x2) + px, w), min(int(y2) + py, h))

        gaps = self._gaps(box, neighbours)
        # A side with nothing beside it borrows its opposite's measure -- the
        # label at the end of a row is the same label as the rest of them.
        for a, b in (("left", "right"), ("up", "down")):
            if gaps[a] is None:
                gaps[a] = gaps[b]
            if gaps[b] is None:
                gaps[b] = gaps[a]

        # Left and right only. What a label is short of at top and bottom is
        # the next label down the web, and there is nothing to be gained by
        # cropping part of it in -- only the axis the reel travels on has a
        # box edge that needs help. The other two sides are measured all the
        # same, because the motion allowance below may land on one of them
        # if this camera is mounted the other way round.
        pad = {"left": 0, "right": 0, "up": 0, "down": 0}
        for side in ("left", "right"):
            if gaps[side] is not None:
                pad[side] = max(int(gaps[side] // 2), 0)

        # The web runs one way at a time, so only the axis it actually
        # travels on gets the allowance; the other is however the labels
        # happen to sit and has nothing to do with speed.
        dx, dy = (float(motion[0]), float(motion[1])) if motion else (0.0, 0.0)
        if abs(dx) >= abs(dy):
            behind, travelled = ("left" if dx > 0 else "right"), abs(dx)
        else:
            behind, travelled = ("up" if dy > 0 else "down"), abs(dy)
        if travelled >= 1 and gaps[behind] is not None:
            pad[behind] = min(pad[behind] + int(round(travelled)),
                              int(gaps[behind]))

        return (max(int(x1) - pad["left"], 0), max(int(y1) - pad["up"], 0),
                min(int(x2) + pad["right"], w),
                min(int(y2) + pad["down"], h))

    def save(self, frame, box, text, neighbours=(), motion=(0.0, 0.0)):
        """Crop `box` out of `frame` and write it. Returns the path, or None
        if the box was degenerate.

        `neighbours` are the other label boxes in the same frame, which is
        what the crop measures its own margins from. `motion` is how far the
        web moved since the last frame, in pixels, which is what decides how
        much of the trailing edge to reach back for.
        """
        x1, y1, x2, y2 = self._box(box, frame.shape, neighbours, motion)
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
