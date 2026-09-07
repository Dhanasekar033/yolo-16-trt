#!/usr/bin/env python3
"""
Two cameras at once, each with the label boxes drawn on it.

Same pairing and the same box as cam-trt-labelbox.py -- both scripts call
utils/labelbox.py, so a change to the rule reaches both -- with two streams
side by side in one window instead of one.

TELLING THE TWO CAMERAS APART is the first problem and not the obvious one.
A single USB camera presents several /dev/videoN nodes: this rig's Global
Shutter Camera holds /dev/video0 for pictures and /dev/video1 for metadata,
so a script that counts video nodes finds two cameras where there is one, and
opens the metadata node as camera B. The nodes are grouped here by the USB
device they belong to, one camera per device, and only the capture node of
each is offered -- so --list-cameras says what is actually there.

Both streams are read on the one loop rather than on a thread each. The
GStreamer pipeline is built with `drop=true max-buffers=1`, so a read always
returns the newest frame and never a queued stale one; there is no backlog for
a thread to drain. The cost is that the loop runs at the slower of the two
cameras, which is what a synchronised pair of views is worth having anyway.

Inference is one engine, run over each frame in turn. A TensorRT context is
not safe to execute from two threads at once, and a second context would cost
a second copy of the weights on the card for no gain -- at ~117 fps a frame on
this GPU, two frames is still faster than either camera delivers.

Usage:
    python3 cam-trt-2cam.py --list-cameras            # what is plugged in
    python3 cam-trt-2cam.py --engine best-2.engine    # the first two cameras
    python3 cam-trt-2cam.py --engine best-2.engine --source-a 0 --source-b 2
    python3 cam-trt-2cam.py --engine best-2.engine --source-b clip.mp4
    python3 cam-trt-2cam.py --engine best-2.engine --source-a left.mp4 --source-b right.mp4
    python3 cam-trt-2cam.py --engine best-2.engine --layout stack --save out.mp4
    python3 cam-trt-2cam.py --engine best-2.engine --no-display --frames 30

Keys (window mode):
    q / Esc   quit          space  pause / resume
    n         step one frame while paused
    s         save BOTH cameras' frames and label crops, losslessly, under one
              timestamp so the pair can be matched up afterwards
"""

import argparse
import os
import subprocess
import shutil
import time

import cv2
import numpy as np

from utils.trt_engine import YOLO26TRT
from utils.utils import preprocess, postprocess
from utils.labelbox import (ARTIFACT_NAMES, CODE_NAMES, LABEL_COLOR,
                            class_index, draw, find_labels, load_class_names,
                            save_shot)

# ── Stream config ────────────────────────────────────────────────────────────
DEFAULT_WIDTH     = 2592
DEFAULT_HEIGHT    = 1944 
DEFAULT_FPS     = 60
DEFAULT_FORMAT  = "MJPG"
DISPLAY_MAX_W   = 1600       # two panes, so wider than the one-camera viewer
DISPLAY_MAX_H   = 900
DEFAULT_ROTATE  = 270        # cameras only; a recorded file is already rotated

DEFAULT_IMGSZ      = 640
DEFAULT_CONF_THRES = 0.25

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg")

ROTATE_MAP = {
    0:   None,
    90:  cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def rotate_frame(frame, degrees):
    code = ROTATE_MAP[degrees]
    return frame if code is None else cv2.rotate(frame, code)


# ── which cameras are actually there ─────────────────────────────────────────

def _is_capture_node(index):
    """Does /dev/videoN deliver pictures, or only metadata?

    Asked of the driver rather than guessed from the number. Without v4l2-ctl
    there is nothing to ask, and the caller falls back to the lowest-numbered
    node of each camera, which is the capture one on every UVC device seen
    here -- a guess, but a documented one."""
    if not shutil.which("v4l2-ctl"):
        return None
    try:
        out = subprocess.run(["v4l2-ctl", "-d", f"/dev/video{index}", "--info"],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    caps = out.split("Device Caps", 1)
    return "Video Capture" in caps[1] if len(caps) > 1 else None


def list_cameras():
    """[(index, name), ...] — one entry per physical camera, in node order.

    Grouped by the USB device each node hangs off, which is what stops one
    camera's picture node and metadata node from being counted as two
    cameras."""
    by_device, order = {}, []
    for entry in sorted(os.listdir("/dev")):
        if not entry.startswith("video") or not entry[5:].isdigit():
            continue
        index = int(entry[5:])
        name_path = f"/sys/class/video4linux/{entry}/name"
        if not os.path.exists(name_path):
            continue
        with open(name_path) as f:
            name = f.read().strip()
        try:
            device = os.path.realpath(f"/sys/class/video4linux/{entry}/device")
        except OSError:
            device = entry
        if device not in by_device:
            by_device[device] = []
            order.append(device)
        by_device[device].append((index, name))

    cameras = []
    for device in order:
        nodes = by_device[device]
        pick = next((n for n in nodes if _is_capture_node(n[0])), None)
        cameras.append(pick or nodes[0])
    return cameras


def gstreamer_pipeline(index, width, height, fps, fmt):
    QUEUE = "queue leaky=downstream max-size-buffers=1"
    SINK  = ("videoconvert ! video/x-raw, format=BGR ! "
             "appsink drop=true max-buffers=1 sync=false")
    if fmt.upper() == "MJPG":
        return (f"v4l2src device=/dev/video{index} ! "
                f"image/jpeg, width={width}, height={height}, framerate={fps}/1 ! "
                f"{QUEUE} ! jpegdec ! {SINK}")
    return (f"v4l2src device=/dev/video{index} ! "
            f"video/x-raw, width={width}, height={height}, framerate={fps}/1 ! "
            f"{QUEUE} ! {SINK}")


class Stream:
    """One camera or one file, and everything the loop knows about it."""

    def __init__(self, spec, tag, args, rotate):
        self.tag = tag
        self.frame = None        # the last frame, as the camera gave it
        self.shown = None        # the same frame with the boxes on it
        self.labels = []
        self.n = 0
        self.found = 0
        self.inferred = 0
        self.misses = 0
        self.ended = False

        self.kind, self.name, self.cap = self._open(spec, args)
        if not self.cap.isOpened():
            raise SystemExit(
                f"[{tag}] could not open {self.name}"
                + (" — another process may hold it, or the mode is not one "
                   "this camera has" if self.kind == "camera"
                   else " — unsupported codec, or the file is unreadable"))

        if self.kind == "file":
            self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
            self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or args.fps
            self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            self.w, self.h = args.width, args.height
            self.fps, self.total = float(args.fps), 0
        # A file was rotated when it was recorded; a camera is rotated here.
        self.rotate = rotate if rotate is not None else (
            0 if self.kind == "file" else DEFAULT_ROTATE)

    def _open(self, spec, args):
        """A device index, a file, or a camera matched by name."""
        if spec is not None and str(spec).isdigit():
            index = int(spec)
            pipeline = gstreamer_pipeline(index, args.width, args.height,
                                          args.fps, args.format)
            return ("camera", f"/dev/video{index}",
                    cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER))
        if spec is not None and os.path.exists(spec):
            if not str(spec).lower().endswith(VIDEO_EXT):
                print(f"[{self.tag}] {spec} is not a known video extension — "
                      f"trying to open it anyway")
            return "file", spec, cv2.VideoCapture(spec)
        if spec is not None:
            # A spec that was plainly meant as a path is reported as a missing
            # file, not as an unmatched camera name -- the second reads like
            # the camera is at fault when the path is simply wrong.
            if os.sep in str(spec) or str(spec).lower().endswith(VIDEO_EXT):
                raise SystemExit(f"[{self.tag}] {spec} does not exist")
            hits = [i for i, name in list_cameras() if spec.lower() in name.lower()]
            if not hits:
                raise SystemExit(f"[{self.tag}] no file and no camera matching "
                                 f"'{spec}' — try --list-cameras")
            pipeline = gstreamer_pipeline(hits[0], args.width, args.height,
                                          args.fps, args.format)
            return ("camera", f"/dev/video{hits[0]}",
                    cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER))
        raise SystemExit(f"[{self.tag}] nothing to open")

    def read(self, loop):
        """The next frame, or None. Sets .ended when a file runs out."""
        if self.ended:
            return None
        ok, frame = self.cap.read()
        if not ok:
            if self.kind == "file":
                if loop and self.n:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    return self.read(False)
                self.ended = True          # a file ends; a camera does not
                return None
            self.misses += 1
            if self.misses > 50:
                print(f"\n[{self.tag}] 50 failed grabs in a row — dropping it")
                self.ended = True
            return None
        self.misses = 0
        self.n += 1
        self.frame = rotate_frame(frame, self.rotate)
        return self.frame

    @property
    def size(self):
        return (self.h, self.w) if self.rotate in (90, 270) else (self.w, self.h)

    def release(self):
        self.cap.release()


def pane(stream, height, fps):
    """One camera's picture, captioned and scaled to the common height."""
    view = stream.shown
    if view is None:
        w, h = stream.size
        view = np.zeros((h, w, 3), np.uint8)
        cv2.putText(view, "no picture yet", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (60, 60, 60), 2, cv2.LINE_AA)
    scale = height / view.shape[0]
    view = cv2.resize(view, (int(view.shape[1] * scale), height))
    progress = f"{stream.n}/{stream.total}" if stream.total else str(stream.n)
    caption = (f"{stream.tag}  {os.path.basename(stream.name)}  {progress}  "
               f"labels: {len(stream.labels)}"
               + ("  ENDED" if stream.ended else ""))
    cv2.rectangle(view, (0, 0), (view.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(view, caption, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0) if not stream.ended else (80, 80, 200), 2, cv2.LINE_AA)
    return view


def main():
    ap = argparse.ArgumentParser(
        description="Two cameras at once, with a box round every label on both.")
    ap.add_argument("--list-cameras", action="store_true",
                     help="print the cameras this machine has and exit")
    ap.add_argument("--source-a", default=None,
                     help="camera A: a /dev/videoN index, a video file, or part "
                          "of a camera's name. Default: the first camera found")
    ap.add_argument("--source-b", default=None,
                     help="camera B, same forms. Default: the second camera")
    ap.add_argument("--rotate-a", type=int, default=None, choices=[0, 90, 180, 270])
    ap.add_argument("--rotate-b", type=int, default=None, choices=[0, 90, 180, 270],
                     help=f"default {DEFAULT_ROTATE} for a camera, 0 for a file")
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--format", default=DEFAULT_FORMAT, choices=["MJPG", "YUYV"])
    ap.add_argument("--loop", action="store_true", help="restart a file at its end")
    ap.add_argument("--frames", type=int, default=0, help="stop after N passes")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--layout", choices=["side", "stack"], default="side",
                     help="side by side, or one above the other")
    # model
    ap.add_argument("--engine", default="best-2.engine")
    ap.add_argument("--classes", default=None,
                     help="class names for THIS engine. Left out, the parts are "
                          "taken to be at indices 0 and 1")
    ap.add_argument("--conf-thres", type=float, default=DEFAULT_CONF_THRES)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    # the box — the same knobs as cam-trt-labelbox.py
    ap.add_argument("--axis", choices=["auto", "x", "y"], default="auto")
    ap.add_argument("--pad", type=float, default=0.06)
    ap.add_argument("--pad-px", type=int, default=0)
    ap.add_argument("--gutter", action="store_true",
                     help="take the whole card: widen into the gap between the "
                          "labels rather than by --pad")
    ap.add_argument("--gutter-share", type=float, default=0.5)
    ap.add_argument("--gutter-cap", type=float, default=0.6)
    ap.add_argument("--no-singles", action="store_true")
    ap.add_argument("--show-parts", action="store_true")
    # output
    ap.add_argument("--save", default=None, help="record the combined view")
    ap.add_argument("--save-crops", default=None,
                     help="write every label box from both cameras here")
    ap.add_argument("--crop-format", default="png", choices=["png", "jpg"])
    ap.add_argument("--shot-dir", default="shots")
    ap.add_argument("--shot-format", default="png",
                     choices=["png", "bmp", "tiff", "jpg"],
                     help="what the s key writes. png, tiff and bmp are all "
                          "lossless; bmp is the raw bytes and the fastest, "
                          "which is what to use while the web is running")
    args = ap.parse_args()

    cameras = list_cameras()
    if args.list_cameras:
        if not cameras:
            print("[cameras] none found")
        for i, (index, name) in enumerate(cameras):
            print(f"  {'AB'[i] if i < 2 else ' '}  /dev/video{index}  {name}")
        return

    # Defaults only fill in what was not asked for, so --source-b clip.mp4
    # still puts the real camera on A.
    spec_a = args.source_a
    spec_b = args.source_b
    free = [str(i) for i, _ in cameras]
    for used in (spec_a, spec_b):
        if used is not None and str(used).isdigit() and str(used) in free:
            free.remove(str(used))
    if spec_a is None:
        spec_a = free.pop(0) if free else None
    if spec_b is None:
        spec_b = free.pop(0) if free else None
    if spec_a is None or spec_b is None:
        raise SystemExit(
            f"[cameras] this machine has {len(cameras)} camera(s) "
            f"({', '.join(f'/dev/video{i}' for i, _ in cameras) or 'none'}) and "
            f"two sources are needed. Plug the second one in, or give it as a "
            f"video file: --source-b clip.mp4")

    streams = [Stream(spec_a, "camA", args, args.rotate_a),
               Stream(spec_b, "camB", args, args.rotate_b)]
    for s in streams:
        print(f"[{s.tag}] {s.name} — {s.w}x{s.h} @ {s.fps:.0f} fps"
              + (f", {s.total} frames" if s.total else "")
              + f", rotate {s.rotate}")

    names = load_class_names(args.classes)
    code_cls = class_index(names, CODE_NAMES, 0)
    art_cls = class_index(names, ARTIFACT_NAMES, 1)
    print(f"[model] parts: {code_cls}={names[code_cls] if names else 'code'}, "
          f"{art_cls}={names[art_cls] if names else 'artifact'}"
          + (f"  (from {args.classes})" if names else "  (fixed indices)"))
    model = YOLO26TRT(args.engine, input_size=(args.imgsz, args.imgsz))
    print(f"[model] loaded {args.engine}")

    if args.save_crops:
        os.makedirs(args.save_crops, exist_ok=True)
        print(f"[crops] label crops -> {args.save_crops}/")

    win_name = "label boxes — two cameras"
    writer = None
    combined = None
    prev_t = time.time()
    fps = 0.0
    n = shots = crops = 0
    paused = step = False
    note = ["", 0.0]
    was_visible = False

    try:
        while True:
            if not paused or step:
                step = False
                got = False
                for s in streams:
                    frame = s.read(args.loop)
                    if frame is None:
                        continue
                    got = True
                    inp, ratio, pad = preprocess(frame, model.input_size)
                    dets = postprocess(model.infer(inp), ratio, pad,
                                       frame.shape, args.conf_thres)
                    s.labels = find_labels(dets, code_cls, art_cls,
                                           frame.shape, args)
                    s.found += len(s.labels)
                    s.inferred += sum(1 for _, m in s.labels if not m)

                    if args.save_crops:
                        for i, (box, _m) in enumerate(s.labels, 1):
                            crop = frame[box[1]:box[3], box[0]:box[2]]
                            if crop.size == 0:
                                continue
                            path = os.path.join(
                                args.save_crops,
                                f"{s.tag}_frame{s.n:06d}_label{i:02d}."
                                f"{args.crop_format}")
                            if cv2.imwrite(path, crop):
                                crops += 1

                    # Boxes onto a copy: s.frame stays as the camera gave it,
                    # which is what the s key writes and the crops come from.
                    s.shown = (draw(frame.copy(), s.labels, dets, code_cls,
                                    art_cls, args)
                               if (not args.no_display or args.save) else frame)

                if all(s.ended for s in streams):
                    break
                if not got:
                    continue
                n += 1

                now = time.time()
                dt, prev_t = now - prev_t, now
                if dt > 0:
                    inst = 1.0 / dt
                    fps = inst if fps == 0.0 else (0.9 * fps + 0.1 * inst)

                if not args.no_display or args.save:
                    height = min(s.size[1] for s in streams)
                    panes = [pane(s, height, fps) for s in streams]
                    if args.layout == "stack":
                        width = min(p.shape[1] for p in panes)
                        panes = [cv2.resize(p, (width, int(p.shape[0] * width
                                                           / p.shape[1])))
                                 for p in panes]
                        combined = cv2.vconcat(panes)
                    else:
                        combined = cv2.hconcat(panes)
                    total_labels = sum(len(s.labels) for s in streams)
                    cv2.putText(combined, f"pass {n}  FPS: {fps:.1f}  "
                                f"labels: {total_labels}",
                                (10, combined.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                                cv2.LINE_AA)
                    if time.time() - note[1] < 2.0:
                        cv2.putText(combined, note[0],
                                    (10, combined.shape[0] - 55),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, LABEL_COLOR,
                                    2, cv2.LINE_AA)

                if args.save and combined is not None:
                    if writer is None:
                        # Opened on the first combined frame: its size is not
                        # known until both panes have been laid out.
                        writer = cv2.VideoWriter(
                            args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                            min(s.fps for s in streams),
                            (combined.shape[1], combined.shape[0]))
                        if not writer.isOpened():
                            raise SystemExit(f"[save] could not open {args.save}")
                    writer.write(combined)

                if args.no_display:
                    print(f"[run] pass {n}  fps={fps:.1f}  "
                          + "  ".join(f"{s.tag}={len(s.labels)}" for s in streams),
                          end="\r")
                if args.frames and n >= args.frames:
                    break

            if not args.no_display:
                if combined is not None:
                    if not was_visible:
                        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                        scale = min(DISPLAY_MAX_W / combined.shape[1],
                                    DISPLAY_MAX_H / combined.shape[0], 1.0)
                        cv2.resizeWindow(win_name,
                                         int(combined.shape[1] * scale),
                                         int(combined.shape[0] * scale))
                    cv2.imshow(win_name, combined)
                key = cv2.waitKey(30 if paused else 1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    paused = not paused
                    print(f"\n[run] {'paused' if paused else 'resumed'} at pass {n}")
                elif key == ord("n"):
                    step = True
                elif key == ord("s"):
                    # One timestamp for both cameras, so the two sets of files
                    # are recognisable afterwards as the same moment.
                    stamp_files, stamp_bytes = 0, 0
                    for s in streams:
                        if s.frame is None:
                            continue
                        paths, size = save_shot(s.frame, s.labels, args.shot_dir,
                                                args.shot_format, s.n, s.tag)
                        stamp_files += len(paths)
                        stamp_bytes += size
                    shots += stamp_files
                    note[0] = (f"saved {stamp_files} {args.shot_format} file(s), "
                               f"{stamp_bytes / 1e6:.1f} MB")
                    note[1] = time.time()
                    print(f"\n[shot] pass {n}: {stamp_files} file(s) "
                          f"({stamp_bytes / 1e6:.1f} MB) -> {args.shot_dir}/")
                visible = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) >= 1
                was_visible = was_visible or visible
                if was_visible and not visible:
                    break
    finally:
        for s in streams:
            s.release()
        if writer:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    print(f"\n[run] {n} pass(es)")
    for s in streams:
        if s.n:
            print(f"  {s.tag}: {s.n} frame(s), {s.found} label box(es), "
                  f"{s.found / s.n:.1f} per frame"
                  + (f", {s.inferred} with a part inferred" if s.inferred else ""))
    if args.save and writer:
        print(f"[save] wrote {args.save}")
    if crops:
        print(f"[crops] wrote {crops} crop(s) to {args.save_crops}/")
    if shots:
        print(f"[shot] wrote {shots} file(s) to {args.shot_dir}/")


if __name__ == "__main__":
    main()
