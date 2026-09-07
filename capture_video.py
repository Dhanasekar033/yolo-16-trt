#!/usr/bin/env python3
"""
Record video off the Global Shutter Camera.

A live preview, and SPACE starts and stops a take. Each take lands as one mp4
under saved_videos/, named for the instant it started —
20260907_120102_123.mp4, the same stamp make_dataset.py gives its frames, so a
video and the stills pulled from it sort together and never collide.

    SPACE   start recording / stop and save
    q       quit (a take still running is stopped and saved, not lost)

No model, no annotations — this is the camera and a file. The frames are the
same frames make_dataset.py banks: the camera half of cam-trt-vlabel.py is
imported rather than copied, so auto-detected index, GStreamer pipeline,
--width/--height/--fps/--format-v4l2 and the --rotate applied before anything
else all behave exactly as they do there. What you record is what the detector
would have seen.

THE FRAME RATE IN THE FILE IS MEASURED, NOT ASSUMED, and that is the one thing
here worth explaining. An mp4 carries a single frame rate in its header and
players trust it completely: write 30 into a file whose frames were actually
grabbed at 12 and the take plays back at two and a half times speed, silently.
The camera's requested --fps is a request, not a promise — MJPG at 5MP, a busy
USB bus or a slow encode all pull the real rate below it. So the preview keeps
the last FPS_WINDOW frame times, and the rate measured over them is what goes
into the header when SPACE starts a take. If the rate then drifts while
recording (encoding 5MP frames is not free), the closing line says by how much
and what to pass to --write-fps to pin it.

A take in progress is written to <stamp>.part.mp4 and renamed only once the
writer has been released and the file finalised. An mp4 that was never released
has no index and no player will open it, so a crash or a power cut leaves an
obvious .part.mp4 rather than something that looks like a finished recording.

Usage:
    python3 capture_video.py
    python3 capture_video.py --out takes --rotate 180
    python3 capture_video.py --width 1280 --height 720 --fps 60
    python3 capture_video.py --write-fps 30      # pin the header rate
    python3 capture_video.py --fourcc avc1       # H.264, where the build has it
"""

import argparse
import datetime as dt
import importlib.util
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2


# cam-trt-vlabel.py has hyphens in its name, so it cannot be imported by the
# normal statement — load it by path out of this script's own directory.
def _load_sibling(filename):
    path = Path(__file__).resolve().parent / filename
    if not path.exists():
        sys.exit(f"[video] {filename} not found next to {Path(__file__).name}")
    spec = importlib.util.spec_from_file_location("vlabel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V = _load_sibling("cam-trt-vlabel.py")

DEFAULT_OUT    = "saved_videos"
DEFAULT_FOURCC = "mp4v"      # in every OpenCV build; avc1 is smaller where present
STAMP_FORMAT   = "%Y%m%d_%H%M%S"    # + _mmm, matching make_dataset.py
PART_SUFFIX    = ".part"     # still ends .mp4 so OpenCV picks the container
FPS_WINDOW     = 90          # frame times kept to measure the real capture rate
FPS_LIMITS     = (1.0, 240.0)
DRIFT_WARN     = 0.05        # report a header/actual gap wider than this

REC_COLOR  = (0, 0, 255)
IDLE_COLOR = (0, 255, 0)
KEY_SPACE  = 32


def stamp_name(taken):
    """20260907_120102_123 — local time, to the millisecond."""
    return taken.strftime(STAMP_FORMAT) + f"_{taken.microsecond // 1000:03d}"


def measured_fps(ticks, fallback):
    """Frames per second over the times in `ticks`, or `fallback` if too few.

    Measured across the whole window rather than from the last interval: one
    slow grab is normal and must not move the number the file is stamped with.
    """
    if len(ticks) < 5:
        return fallback
    span = ticks[-1] - ticks[0]
    if span <= 0:
        return fallback
    return min(max((len(ticks) - 1) / span, FPS_LIMITS[0]), FPS_LIMITS[1])


def hhmmss(seconds):
    return f"{int(seconds) // 60:02d}:{seconds % 60:04.1f}"


class Take:
    """One space-to-space recording, on its way to one mp4."""

    def __init__(self, out_dir, fps, size, fourcc):
        self.started = time.time()
        self.stem = stamp_name(dt.datetime.now())
        self.final = Path(out_dir) / f"{self.stem}.mp4"
        self.part = Path(out_dir) / f"{self.stem}{PART_SUFFIX}.mp4"
        self.fps = fps
        self.size = size
        self.frames = 0
        self.writer = cv2.VideoWriter(str(self.part),
                                      cv2.VideoWriter_fourcc(*fourcc), fps, size)

    @property
    def opened(self):
        return self.writer.isOpened()

    @property
    def elapsed(self):
        return time.time() - self.started

    def add(self, frame):
        """Write one frame. Must be the clean frame — anything drawn on it for
        the preview would be burned into the recording."""
        if (frame.shape[1], frame.shape[0]) != self.size:
            return False                  # a resize mid-take would corrupt the file
        self.writer.write(frame)
        self.frames += 1
        return True

    def stop(self):
        """Finalise the mp4 and give it its real name.

        Returns (path, frames, seconds, actual_fps, header_fps), or None when
        the take held no frames — an empty mp4 is not worth keeping and some
        players choke on one."""
        seconds = self.elapsed
        self.writer.release()
        if not self.frames:
            self.part.unlink(missing_ok=True)
            return None
        os.replace(self.part, self.final)
        actual = self.frames / seconds if seconds > 0 else self.fps
        return self.final, self.frames, seconds, actual, self.fps


def main():
    ap = argparse.ArgumentParser(description="Record mp4 takes off the camera.")
    # camera args — same meaning as cam-trt-vlabel.py
    ap.add_argument("--index", type=int, default=None,
                     help="Force a /dev/videoN index (skips auto-detect).")
    ap.add_argument("--width", type=int, default=V.DEFAULT_WIDTH)
    ap.add_argument("--height", type=int, default=V.DEFAULT_HEIGHT)
    ap.add_argument("--fps", type=int, default=V.DEFAULT_FPS,
                     help="rate requested of the camera")
    ap.add_argument("--format-v4l2", dest="v4l2_format", default=V.DEFAULT_FORMAT,
                     choices=["MJPG", "YUYV"])
    ap.add_argument("--rotate", type=int, default=V.DEFAULT_ROTATE,
                     choices=[0, 90, 180, 270])
    # recording args
    ap.add_argument("--out", default=DEFAULT_OUT, help="directory for the mp4 files")
    ap.add_argument("--fourcc", default=DEFAULT_FOURCC,
                     help="codec fourcc: mp4v (default, always available) or avc1")
    ap.add_argument("--write-fps", type=float, default=0.0,
                     help="frame rate stamped into the file; 0 measures the real one")
    args = ap.parse_args()

    if len(args.fourcc) != 4:
        sys.exit(f"[video] --fourcc must be 4 characters, got {args.fourcc!r}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cam_index = args.index if args.index is not None else V.find_camera_index()
    pipeline = V.gstreamer_pipeline(cam_index, args.width, args.height,
                                     args.fps, args.v4l2_format)
    print(f"[camera] using /dev/video{cam_index}")

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera via GStreamer pipeline")

    print(f"[video] takes go to {out_dir.resolve()}  (codec {args.fourcc})")
    print(f"[video] header rate: "
          + (f"pinned at {args.write_fps:g}" if args.write_fps
             else "measured from the preview when recording starts"))
    print("[keys] SPACE = start / stop   q = quit")

    disp_w, disp_h = ((args.height, args.width) if args.rotate in (90, 270)
                      else (args.width, args.height))
    win_name = "capture_video - SPACE: record   q: quit"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    scale = min(V.DISPLAY_MAX_W / disp_w, V.DISPLAY_MAX_H / disp_h, 1.0)
    cv2.resizeWindow(win_name, int(disp_w * scale), int(disp_h * scale))

    ticks = deque(maxlen=FPS_WINDOW)
    take = None
    saved = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[camera] frame grab failed, retrying...")
                continue

            frame = V.rotate_frame(frame, args.rotate)
            ticks.append(time.time())
            live_fps = measured_fps(ticks, float(args.fps))

            # The recording gets the clean frame. Everything below draws on it
            # afterwards, so the preview's overlay is never in the file.
            if take is not None and not take.add(frame):
                print(f"\n[video] frame size changed mid-take, stopping")
                result = take.stop()
                take = None
                if result:
                    saved.append(result)

            h, w = frame.shape[:2]
            if take is None:
                cv2.putText(frame, f"SPACE to record   {w}x{h}  {live_fps:.1f} fps"
                                   f"   takes: {len(saved)}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            IDLE_COLOR, 2, cv2.LINE_AA)
            else:
                # A blinking dot, because a still one reads as part of the UI
                # and this is the difference between recording and not.
                if int(take.elapsed * 2) % 2 == 0:
                    cv2.circle(frame, (40, 32), 14, REC_COLOR, -1)
                cv2.putText(frame, f"REC {hhmmss(take.elapsed)}  {take.frames} frames"
                                   f"  @{take.fps:.1f} fps -> {take.final.name}",
                            (70, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            REC_COLOR, 2, cv2.LINE_AA)
            cv2.imshow(win_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == KEY_SPACE:
                if take is None:
                    fps = args.write_fps or live_fps
                    take = Take(out_dir, fps, (w, h), args.fourcc)
                    if not take.opened:
                        take = None
                        print(f"\n[video] could not open a writer for {args.fourcc!r} "
                              f"at {w}x{h}. Try --fourcc mp4v.")
                    else:
                        print(f"[video] recording {take.final.name} "
                              f"at {w}x{h}, {fps:.1f} fps")
                else:
                    result = take.stop()
                    take = None
                    if result is None:
                        print("[video] take held no frames, nothing written")
                    else:
                        saved.append(result)
                        path, frames, seconds, actual, _ = result
                        print(f"[video] saved {path.name}  {frames} frames, "
                              f"{seconds:.1f}s, {actual:.1f} fps actual")
    finally:
        # A take still running when the loop ends is finished and kept — the
        # frames are already in the file, only the index is missing.
        if take is not None:
            result = take.stop()
            if result:
                saved.append(result)
                print(f"[video] saved {result[0].name} on exit")
        cap.release()
        cv2.destroyAllWindows()

        total = sum(r[2] for r in saved)
        print(f"\n[video] {len(saved)} take(s), {hhmmss(total)} total, "
              f"in {out_dir.resolve()}")
        for path, frames, seconds, actual, header in saved:
            line = f"  {path.name}  {frames} frames  {seconds:.1f}s  {actual:.1f} fps"
            # The header is what players believe. Say so when the two parted
            # company, rather than leaving a take that quietly plays too fast.
            if header and abs(actual - header) / header > DRIFT_WARN:
                line += (f"   [header {header:.1f} fps, so it plays back at "
                         f"{header / actual:.2f}x real time — "
                         f"--write-fps {actual:.1f} pins the next one]")
            print(line)


if __name__ == "__main__":
    main()
