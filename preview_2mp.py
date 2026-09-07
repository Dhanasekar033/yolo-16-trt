#!/usr/bin/env python3
"""
Minimal viewer for the 2MP Global Shutter Camera (OV2311-class, 1920x1200).

Same shape as preview_5mp.py, with the defaults this sensor actually has:

    MJPG  1920x1200  up to 90 fps        <- the full frame, 2.3MP
    YUYV  1920x1200  5 fps               <- uncompressed, and that is the cap

preview_5mp.py asks for 2592x1944, which THIS camera does not have at any
frame rate. v4l2src cannot negotiate caps the device never advertised, so the
pipeline dies before the first frame with

    Internal data stream error. / unable to start pipeline

and OpenCV then reports only that the capture would not open — the size that
caused it is nowhere in the message. That is the whole failure, and it is why
this script checks the mode against what the device enumerates BEFORE opening
anything, and prints the supported list when the answer is no.

Usage:
    python3 preview_2mp.py                       # 1920x1200 MJPG @ 60
    python3 preview_2mp.py --list                # what this camera supports
    python3 preview_2mp.py --fps 90              # the sensor's ceiling
    python3 preview_2mp.py --width 1280 --height 960
    python3 preview_2mp.py --format YUYV --fps 5 # uncompressed, slow
    python3 preview_2mp.py --no-display --frames 60
"""

import argparse
import os
import re
import shutil
import subprocess
import time

import cv2

# ── Stream config ────────────────────────────────────────────────────────────
DEFAULT_CAM_INDEX = 0
DEFAULT_WIDTH     = 1920
DEFAULT_HEIGHT    = 1200     # 1920x1200 is the full sensor: 2.3MP, 16:10
DEFAULT_FPS       = 60       # MJPG reaches 90 here; 60 is the steadier default
DEFAULT_FORMAT    = "MJPG"   # YUYV at this size is capped at 5fps by USB bandwidth
DISPLAY_MAX_W     = 1280
DISPLAY_MAX_H     = 960
DEFAULT_ROTATE    = 270      # matches run.py, which the capture scripts expect

GS_CAMERA_NAME    = "Global Shutter Camera"

ROTATE_MAP = {
    0:   None,
    90:  cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def rotate_frame(frame, degrees):
    """Rotate a frame by a fixed angle (0/90/180/270). No-op for 0."""
    code = ROTATE_MAP[degrees]
    return frame if code is None else cv2.rotate(frame, code)


def list_cameras():
    """(index, name) for every /dev/videoN, from its v4l2 sysfs name."""
    cams = []
    for d in sorted(os.listdir("/dev")):
        if not d.startswith("video"):
            continue
        sys_path = f"/sys/class/video4linux/{d}/name"
        if not os.path.exists(sys_path):
            continue
        with open(sys_path) as f:
            cams.append((int(d.replace("video", "")), f.read().strip()))
    return cams


def find_camera_index(name_substring=GS_CAMERA_NAME, default=DEFAULT_CAM_INDEX):
    for index, name in list_cameras():
        if name_substring.lower() in name.lower():
            return index
    print(f"[camera] '{name_substring}' not found among {list_cameras()} — "
          f"falling back to index {default}")
    return default


def probe_modes(cam_index):
    """{("MJPG", w, h): [fps, ...]} straight from the driver, or None.

    A USB camera advertises exactly what it can do and nothing more, so this
    is the authority on whether a mode exists — guessing from the model name is
    how a 5MP pipeline ends up pointed at a 2MP sensor. Returns None when
    v4l2-ctl is not installed, and the caller then skips the check rather than
    refusing to run."""
    if not shutil.which("v4l2-ctl"):
        return None
    try:
        out = subprocess.run(
            ["v4l2-ctl", "-d", f"/dev/video{cam_index}", "--list-formats-ext"],
            capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None

    modes, fmt, size = {}, None, None
    for line in out.splitlines():
        m = re.search(r"\]:\s*'(\w+)'", line)
        if m:
            fmt = m.group(1)
            continue
        m = re.search(r"Size:\s*\w+\s*(\d+)x(\d+)", line)
        if m:
            size = (int(m.group(1)), int(m.group(2)))
            modes.setdefault((fmt, *size), [])
            continue
        m = re.search(r"\(([\d.]+)\s*fps\)", line)
        if m and fmt and size:
            modes[(fmt, *size)].append(float(m.group(1)))
    return modes or None


def get_control(cam_index, name):
    """One v4l2 control's current value, or None if it cannot be read."""
    if not shutil.which("v4l2-ctl"):
        return None
    try:
        out = subprocess.run(["v4l2-ctl", "-d", f"/dev/video{cam_index}",
                              "-C", name], capture_output=True, text=True,
                             timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r":\s*(-?\d+)", out)
    return int(m.group(1)) if m else None


def set_control(cam_index, name, value):
    if not shutil.which("v4l2-ctl"):
        print(f"[camera] v4l2-ctl not installed — cannot set {name}")
        return False
    r = subprocess.run(["v4l2-ctl", "-d", f"/dev/video{cam_index}",
                        "-c", f"{name}={value}"], capture_output=True, text=True)
    if r.returncode:
        print(f"[camera] could not set {name}={value}: {r.stderr.strip()}")
        return False
    print(f"[camera] {name} = {value}")
    return True


def report_exposure(cam_index, fps):
    """Print the exposure, and say so when it — not the mode — caps the rate.

    Only a MANUAL exposure binds. In the auto modes the driver keeps steering
    the sensor and exposure_time_absolute goes stale: this camera still reads
    10000 there while streaming at 68 fps, so reading the number without the
    mode beside it is how you end up chasing a limit that is not there."""
    auto = get_control(cam_index, "auto_exposure")
    exp = get_control(cam_index, "exposure_time_absolute")
    if exp is None:
        return
    exp_ms = exp / 10.0        # UVC exposure_time_absolute is in 100us units
    mode = {0: "auto", 1: "manual", 2: "shutter priority",
            3: "aperture priority"}.get(auto, str(auto))
    if auto != 1:
        print(f"[camera] exposure: {mode} — the camera is choosing "
              f"(the manual value, {exp}, does not apply)")
        return
    print(f"[camera] exposure: {exp} ({exp_ms:g} ms/frame), manual")
    if exp_ms > 1000.0 / max(fps, 1):
        print(f"[camera] WARNING: {exp_ms:g} ms of exposure caps this stream at "
              f"~{1000.0 / exp_ms:.1f} fps, not the {fps} asked for.")


def fit_exposure(cam_index, fps):
    """Shorten a manual exposure that is longer than one frame. Returns what it
    was, to be put back on the way out.

    The sensor cannot finish a frame faster than it exposes one, so a manual
    exposure left at the maximum silently caps a 90 fps camera at 1 fps — the
    stream still runs, it just crawls, and nothing in the pipeline says why.
    This is the single most common reason a preview looks broken, so the
    preview fixes it rather than printing advice and running slowly anyway.

    It is put back on exit because the value is the camera's, not this
    script's: run.py and the capture tools inherit whatever is left behind, and
    a preview that quietly re-exposed the rig would change what they record."""
    auto = get_control(cam_index, "auto_exposure")
    exp = get_control(cam_index, "exposure_time_absolute")
    if auto != 1 or exp is None:
        return None                       # the camera is choosing; leave it
    per_frame = max(1, int(10000 / max(fps, 1)))   # 100us units in one frame
    if exp <= per_frame:
        return None
    print(f"[camera] exposure {exp} ({exp / 10:g} ms) is longer than one frame "
          f"at {fps} fps — lowering to {per_frame} ({per_frame / 10:g} ms) for "
          f"this preview only")
    print(f"[camera] (the original is restored on exit; --keep-exposure leaves "
          f"it alone, --exposure N sets it for good)")
    if not set_control(cam_index, "exposure_time_absolute", per_frame):
        return None
    return exp


def print_modes(modes, cam_index):
    print(f"[camera] /dev/video{cam_index} supports:")
    for (fmt, w, h), rates in sorted(modes.items(),
                                     key=lambda kv: (kv[0][0], -kv[0][1] * kv[0][2])):
        top = max(rates) if rates else 0
        print(f"    {fmt}  {w}x{h:<6} up to {top:g} fps"
              + (f"   ({', '.join(f'{r:g}' for r in sorted(rates, reverse=True))})"
                 if len(rates) > 1 else ""))


def nearest_mode(modes, fmt, width, height, fps):
    """The supported mode closest to what was asked for, same format if it can.

    Closest by pixel count, then by frame rate — a request for 2592x1944 lands
    on 1920x1200 rather than on 320x240, which is what makes the fallback worth
    having instead of just an error."""
    same_fmt = [k for k in modes if k[0] == fmt.upper()] or list(modes)
    if not same_fmt:
        return None
    want = width * height
    best = min(same_fmt, key=lambda k: (abs(k[1] * k[2] - want), -k[1] * k[2]))
    rates = modes[best] or [fps]
    return best[0], best[1], best[2], min(rates, key=lambda r: abs(r - fps))


def gstreamer_pipeline(cam_index, width, height, fps, fmt):
    """v4l2src -> BGR appsink. The queue and appsink flags keep the reader on
    the newest frame: without them a slow consumer drifts behind live."""
    QUEUE = "queue leaky=downstream max-size-buffers=1"
    SINK  = ("videoconvert ! video/x-raw, format=BGR ! "
             "appsink drop=true max-buffers=1 sync=false")
    if fmt.upper() == "MJPG":
        return (f"v4l2src device=/dev/video{cam_index} ! "
                f"image/jpeg, width={width}, height={height}, framerate={fps}/1 ! "
                f"{QUEUE} ! jpegdec ! {SINK}")
    return (f"v4l2src device=/dev/video{cam_index} ! "
            f"video/x-raw, width={width}, height={height}, framerate={fps}/1 ! "
            f"{QUEUE} ! {SINK}")


def open_v4l2_direct(cam_index, width, height, fps, fmt):
    """Plain V4L2 fallback, no GStreamer.

    Worth having because the two backends fail differently: if this one opens
    and streams, the camera and the mode are fine and the problem is the
    GStreamer install, which is a much shorter thing to go and fix."""
    cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    if fmt.upper() == "MJPG":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main():
    ap = argparse.ArgumentParser(
        description="Stream the 2MP Global Shutter Camera (1920x1200) via GStreamer.")
    ap.add_argument("--index", type=int, default=None,
                     help="force a /dev/videoN index (skips auto-detect)")
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--format", default=DEFAULT_FORMAT, choices=["MJPG", "YUYV"])
    ap.add_argument("--rotate", type=int, default=DEFAULT_ROTATE,
                     choices=[0, 90, 180, 270],
                     help="rotate every frame by a fixed angle (clockwise)")
    ap.add_argument("--list", action="store_true",
                     help="print the modes this camera supports and exit")
    ap.add_argument("--strict", action="store_true",
                     help="fail on an unsupported mode instead of falling back "
                          "to the nearest one the camera has")
    ap.add_argument("--exposure", type=int, default=None,
                     help="set exposure_time_absolute, in units of 100us "
                          "(156 is the driver default, 10000 = 1 s = 1 fps). "
                          "Switches the camera to manual exposure")
    ap.add_argument("--auto-exposure", action="store_true",
                     help="hand exposure back to the camera (aperture priority)")
    ap.add_argument("--keep-exposure", action="store_true",
                     help="do not touch the exposure, even if it caps the frame "
                          "rate below --fps")
    ap.add_argument("--v4l2", action="store_true",
                     help="skip GStreamer and open the device directly")
    ap.add_argument("--no-display", action="store_true",
                     help="print FPS instead of opening a window (headless)")
    ap.add_argument("--frames", type=int, default=0,
                     help="stop after N frames (0 = run until q)")
    args = ap.parse_args()

    cam_index = args.index if args.index is not None else find_camera_index()
    modes = probe_modes(cam_index)

    if args.list:
        if modes:
            print_modes(modes, cam_index)
        else:
            print(f"[camera] could not probe /dev/video{cam_index} "
                  f"(v4l2-ctl not installed?)")
        return

    width, height, fps, fmt = args.width, args.height, args.fps, args.format
    if modes is None:
        print("[camera] v4l2-ctl not available — skipping the mode check")
    elif (fmt.upper(), width, height) not in modes:
        print(f"[camera] {fmt} {width}x{height} is NOT a mode this camera has.")
        print_modes(modes, cam_index)
        if args.strict:
            raise SystemExit("[camera] --strict: refusing to guess")
        picked = nearest_mode(modes, fmt, width, height, fps)
        if not picked:
            raise SystemExit("[camera] no usable mode found")
        fmt, width, height, fps = picked[0], picked[1], picked[2], int(picked[3])
        print(f"[camera] using the nearest supported mode instead: "
              f"{fmt} {width}x{height} @ {fps}")
    else:
        rates = modes[(fmt.upper(), width, height)]
        if rates and fps not in [int(r) for r in rates]:
            best = int(min(rates, key=lambda r: abs(r - fps)))
            print(f"[camera] {fps} fps not offered at {width}x{height} "
                  f"({', '.join(f'{r:g}' for r in sorted(rates, reverse=True))}) "
                  f"— using {best}")
            fps = best

    # Before opening the stream: the control survives across processes, so
    # whatever the last program left it at is what this one inherits.
    restore_exposure = None
    if args.auto_exposure:
        set_control(cam_index, "auto_exposure", 3)
    elif args.exposure is not None:
        set_control(cam_index, "auto_exposure", 1)
        set_control(cam_index, "exposure_time_absolute", args.exposure)
    elif not args.keep_exposure:
        # Only this branch is undone on exit: an explicit --exposure is the
        # user setting the camera, not the preview borrowing it.
        restore_exposure = fit_exposure(cam_index, fps)
    report_exposure(cam_index, fps)

    print(f"[camera] using /dev/video{cam_index}")
    if args.v4l2:
        cap = open_v4l2_direct(cam_index, width, height, fps, fmt)
    else:
        pipeline = gstreamer_pipeline(cam_index, width, height, fps, fmt)
        print(f"[camera] pipeline: {pipeline}")
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            # The mode is known good by here, so the pipeline is the suspect —
            # try the other backend before giving up, and say which one worked.
            print("[camera] GStreamer would not start — trying plain V4L2")
            cap = open_v4l2_direct(cam_index, width, height, fps, fmt)
            if cap is not None and cap.isOpened():
                print("[camera] V4L2 opened it, so the camera and mode are fine "
                      "and the GStreamer install is the problem")
    if cap is None or not cap.isOpened():
        raise SystemExit(
            f"[camera] could not open /dev/video{cam_index} at {fmt} "
            f"{width}x{height}@{fps}. Another process may hold it — check with "
            f"`fuser -v /dev/video{cam_index}`.")

    disp_w, disp_h = (height, width) if args.rotate in (90, 270) else (width, height)
    win_name = "Global Shutter Camera 2MP"
    if not args.no_display:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        scale = min(DISPLAY_MAX_W / disp_w, DISPLAY_MAX_H / disp_h, 1.0)
        cv2.resizeWindow(win_name, int(disp_w * scale), int(disp_h * scale))

    try:
        prev_t, fps_ema, n, misses = time.time(), 0.0, 0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                misses += 1
                if misses > 50:
                    raise SystemExit("[camera] 50 failed grabs in a row — the "
                                     "stream stopped; unplug/replug the camera")
                print("[camera] frame grab failed, retrying...")
                continue
            misses = 0
            n += 1

            frame = rotate_frame(frame, args.rotate)

            now = time.time()
            dt, prev_t = now - prev_t, now
            if dt > 0:
                inst = 1.0 / dt
                fps_ema = inst if fps_ema == 0.0 else (0.9 * fps_ema + 0.1 * inst)

            if args.no_display:
                print(f"[camera] frame {n} {frame.shape}  fps={fps_ema:.1f}",
                      end="\r")
            else:
                cv2.putText(frame, f"{width}x{height} {fmt}  FPS: {fps_ema:.1f}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow(win_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.frames and n >= args.frames:
                break
        print(f"\n[camera] {n} frame(s), {fps_ema:.1f} fps")
    finally:
        cap.release()
        if restore_exposure is not None:
            set_control(cam_index, "exposure_time_absolute", restore_exposure)
            print(f"[camera] exposure restored to {restore_exposure}")
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
