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

A file is named by the ID THE CODE CARRIES and by nothing else. The QR on
these labels holds a URL -- HTTPS://SCAN.SMARTQR.IO/LS5/7016 -- of which only
the last segment differs from one label to the next, so a folder named by the
whole payload sorts by the part that never changes and has to be read to the
end to tell two files apart. The datamatrix holds the value itself, and that
is kept exactly as it was read.

There is no timestamp in the name: the name is the code, so it says which
label the picture is of, which is the only question anyone asks a folder of
these. When the same code is photographed twice -- a re-inspection pass over
the same coil -- the second file takes a _1, so nothing is overwritten and the
first read stays the one with the plain name.

PNG, and stored rather than deflated: the crop is evidence of what was
printed, so nothing about it should be a re-encode of the picture the camera
gave. JPEG would smooth exactly the detail that decides whether a code was
printed badly, and that cannot be undone afterwards.
"""

import os
import re
import time

import cv2

# Only what a filesystem actually refuses, so a value is altered as little as
# it can be: a name that has been tidied up is a name that no longer matches
# the sheet the value came from.
UNSAFE = re.compile(r'[\x00-\x1f/\\:*?"<>|]+')
URL_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


def code_id(text, max_len=120):
    """What to call the file for this payload.

    A URL is cut to its last path segment, which is the id and the only part
    of it that identifies the label:

        HTTPS://SCAN.SMARTQR.IO/LS5/7016              -> 7016
        HTTPS://NONCLONE.PHARMASECURE.US/22/ZNR2PS..  -> ZNR2PSTVGXT8Z

    Anything else -- a datamatrix, which carries the value and not a link --
    is kept as it stands. Query and fragment go first so that a URL with
    either still ends at its id rather than at the parameters after it.
    """
    text = (text or "").strip()
    if URL_SCHEME.match(text):
        parts = [p for p in text.split("?")[0].split("#")[0].split("/") if p]
        if parts:
            text = parts[-1]
    text = UNSAFE.sub("_", text).strip(". ")
    if len(text) > max_len:                  # keep the tail: that's the part
        text = text[-max_len:]               # that differs between codes
    return text or "unknown"


def timestamp():
    """Sortable stamp with milliseconds, e.g. 20260828-114530-472."""
    now = time.time()
    return time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + \
        f"-{int((now % 1) * 1000):03d}"


STAMP = re.compile(r"_(\d{8}-\d{6})(?:-(\d{3}))?(?:_\d+)?$")

IMAGE_EXT = (".jpg", ".jpeg", ".png")


def read_stamp(name, path=None):
    """When the label in this file was read, as epoch seconds.

    Off the filename, because that stamp is the moment the code was decoded
    -- which is what the console is reporting -- while the file's mtime is
    only when the write finished, and survives being copied about even less
    well. A name that was not written by this class falls back to the mtime,
    and then to nothing.
    """
    hit = STAMP.search(os.path.splitext(name)[0])
    if hit:
        try:
            secs = time.mktime(time.strptime(hit.group(1), "%Y%m%d-%H%M%S"))
            return secs + (int(hit.group(2)) / 1000 if hit.group(2) else 0)
        except ValueError:
            pass
    try:
        return os.path.getmtime(path) if path else None
    except OSError:
        return None


class LabelSaver:
    """Writes one image per decoded label into a per-sheet folder."""

    def __init__(self, root="labels", name="run", subdir=None, ext="png",
                 pad=0.0, min_pad=0, quality=95, png_compress=0):
        self.dir = os.path.join(root, name, subdir) if subdir \
            else os.path.join(root, name)
        self.ext = ext.lstrip(".")
        self.pad = pad
        self.min_pad = min_pad
        # PNG at compression 0 stores the pixels instead of deflating them.
        # Every PNG level is lossless -- the level only trades file size for
        # the time spent packing -- and on a crop this size the packing costs
        # more than the disk does, on a thread that is also running the
        # camera. JPEG's quality is a different thing entirely: it throws
        # detail away, which is why it is not the default here.
        if self.ext in ("jpg", "jpeg"):
            self.params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif self.ext == "png":
            self.params = [cv2.IMWRITE_PNG_COMPRESSION, int(png_compress)]
        else:
            self.params = []
        # Made on the first write, not here: loading a sheet and then pointing
        # the crops somewhere else is two operations, and the folder for the
        # in-between combination should not be left behind empty. A folder
        # that is already there is another matter -- it is this sheet's own
        # crops from an earlier run, and the console counts them in rather
        # than reporting nothing saved while a folder of them sits on disk.
        self.count, self.last = 0, None
        self._made = os.path.isdir(self.dir)
        if self._made:
            self.count, self.last = self._survey()
        where = f"{self.dir}/"
        if self.count:
            print(f"[crops] {self.count} crop(s) already in {where} — "
                  f"adding to them")
        else:
            print(f"[crops] saving label crops to {where}")

    def _survey(self):
        """What is in the folder already: how many crops, and the last one.

        The last is the newest by the stamp in its own name, not by the order
        the directory happens to list them in -- a folder is not sorted, and
        the one written last is the one the console has to name.
        """
        newest, count = None, 0
        try:
            names = os.listdir(self.dir)
        except OSError:
            return 0, None
        for name in names:
            if not name.lower().endswith(IMAGE_EXT):
                continue
            count += 1
            when = read_stamp(name, os.path.join(self.dir, name))
            if when is not None and (newest is None or when > newest[1]):
                newest = (name, when)
        return count, newest

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

    def save(self, frame, box, text, neighbours=(), motion=(0.0, 0.0),
             exact=False):
        """Crop `box` out of `frame` and write it. Returns the path, or None
        if the box was degenerate.

        `neighbours` are the other label boxes in the same frame, which is
        what the crop measures its own margins from. `motion` is how far the
        web moved since the last frame, in pixels, which is what decides how
        much of the trailing edge to reach back for.

        `exact` takes the box as it stands, clipped to the picture and
        nothing else. That is for a crop measured out from the code itself
        rather than drawn round a label by the detector: the four margins
        were set by the operator against this camera and this reel, and
        widening them here by something measured off the neighbours would
        be the app quietly overruling what it was told.
        """
        if exact:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = (int(round(float(v))) for v in box[:4])
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, w), min(y2, h)
        else:
            x1, y1, x2, y2 = self._box(box, frame.shape, neighbours, motion)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None

        if not self._made:
            os.makedirs(self.dir, exist_ok=True)
            self._made = True

        name = f"{code_id(text)}.{self.ext}"
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
        self.last = (os.path.basename(path), time.time())
        return path
