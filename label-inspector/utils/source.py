"""A recording standing in for the camera, for testing off the machine.

The inspector normally reads a live camera through GStreamer. There is no
camera on a desk, and the parts worth testing away from the winder -- the
window, the sheet, the faults, the crops, the log -- do not care where a
frame came from. So this reads a video file, a single image, or a folder of
images, and does it through the same three methods cv2.VideoCapture offers:

    isOpened()   read()   release()

Nothing else in the app then needs to know. `cap` is one or the other, the
capture loop calls read() either way, and the camera path is left exactly as
it was -- which matters more than the convenience, because that path is the
one that runs the line.

IT IS PACED, and that is not decoration. A file read flat out arrives at
several hundred frames a second, and the machine reasons about time: how far
the web moved between frames decides how much of the trailing edge a crop
reaches back for, and how long nothing has read decides when the line stops
for a camera that has stopped seeing. Handed frames faster than a camera could
ever deliver them, both measure nonsense. So frames come at the file's own
rate, or whatever --source-fps says, and --source-speed winds that up or down
deliberately rather than by accident.

At the end it holds the last frame rather than failing: a dead source would
have the loop printing "frame grab failed" forever, and a still frame on the
screen is a truthful picture of a recording that has finished.
"""

import glob
import os
import time

import cv2

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg")


class FrameSource:
    """A video, an image, or a folder of images, read like a camera."""

    def __init__(self, path, fps=None, speed=1.0, loop=False):
        self.path = path
        self.loop = loop
        self.speed = max(0.01, float(speed or 1.0))
        self.kind = None
        self.frames = []          # image paths, when this is a folder
        self.index = 0
        self.total = 0
        self.ended = False        # ran out, and is now holding the last frame
        self._cap = None
        self._last = None
        self._due = None
        self._said = False

        if os.path.isdir(path):
            self.frames = sorted(
                p for p in glob.glob(os.path.join(path, "*"))
                if p.lower().endswith(IMAGE_EXT))
            if not self.frames:
                raise SystemExit(f"[source] no images in {path}")
            self.kind = "images"
            self.total = len(self.frames)
        elif path.lower().endswith(IMAGE_EXT):
            self.frames = [path]
            self.kind = "image"
            self.total = 1
        else:
            self._cap = cv2.VideoCapture(path)
            if not self._cap.isOpened():
                raise SystemExit(
                    f"[source] could not open {path} — unsupported codec, or "
                    f"the file is unreadable")
            self.kind = "video"
            self.total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        native = (self._cap.get(cv2.CAP_PROP_FPS) if self._cap else 0) or 0
        # A folder of stills has no rate of its own, so one is chosen: 10/s is
        # slow enough to watch and fast enough that a hundred frames is not a
        # coffee break.
        self.fps = float(fps or native or (25.0 if self.kind == "video" else 10.0))
        self.interval = 1.0 / (self.fps * self.speed)

    # ── the VideoCapture face ────────────────────────────────────────────
    def isOpened(self):
        return self.kind is not None

    def read(self):
        """(ok, frame), paced to the source's rate. Never returns ok=False.

        The capture loop treats a failed grab as a camera glitch and retries
        forever, which is right for a camera and wrong for a file that has
        simply finished -- so the end of the recording is a held frame, and
        `ended` is what says so.
        """
        self._wait()
        frame = self._next()
        if frame is None:
            frame = self._last
            if frame is None:
                raise SystemExit(f"[source] {self.path} gave no frames at all")
            if not self._said:
                self._said = True
                print(f"[source] end of {os.path.basename(self.path)} — "
                      f"holding the last frame")
            self.ended = True
            return True, frame.copy()
        self._last = frame
        return True, frame

    def release(self):
        if self._cap is not None:
            self._cap.release()

    # ── inside ───────────────────────────────────────────────────────────
    def _wait(self):
        """Hold the frame back until it is due. See the note at the top."""
        now = time.monotonic()
        if self._due is None:
            self._due = now
        if self._due > now:
            time.sleep(self._due - now)
        # From the due time, not from now: a slow frame is caught up on
        # rather than added to every frame after it.
        self._due = max(self._due + self.interval, now)

    def _next(self):
        """The next frame, or None at the end."""
        if self.kind == "video":
            ok, frame = self._cap.read()
            if ok:
                self.index += 1
                return frame
            if self.loop and self.index:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.index = 0
                ok, frame = self._cap.read()
                if ok:
                    self.index += 1
                    return frame
            return None

        if self.index >= len(self.frames):
            if not (self.loop and self.frames):
                return None
            self.index = 0
        # A single image is the only case that keeps giving the same frame
        # back: there is nothing to advance to, and holding it is the point.
        frame = cv2.imread(self.frames[self.index])
        self.index += 1
        if frame is None:
            print(f"[source] {self.frames[self.index - 1]} would not read")
            return self._next()
        return frame

    def describe(self):
        what = {"video": "video", "images": "folder of images",
                "image": "image"}[self.kind]
        size = ""
        if self._cap is not None:
            size = (f", {int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                    f"{int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        return (f"{what} {self.path}{size}"
                + (f", {self.total} frames" if self.total else "")
                + f", played at {self.fps * self.speed:.0f}/s"
                + (" on a loop" if self.loop else ""))
