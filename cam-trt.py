#!/usr/bin/env python3
"""
Global Shutter Camera viewer + live YOLO26 TensorRT inference.

Same GStreamer/OpenCV capture pipeline and staleness-guard setup as
cam_view.py, with per-frame YOLO26 detection + box drawing added in.
No ultralytics import anywhere — inference goes through trt_engine.py.

Runs on the live camera, or on a recorded video with --source. A file is not
just another camera: it ends, it carries its own frame rate and size, and one
written by capture_video.py already has the --rotate baked in — so --source
changes those defaults rather than making you remember to.

Usage:
    python3 cam-trt.py --engine best.engine
    python3 cam-trt.py --engine best.engine --classes classes.txt --conf-thres 0.35
    python3 cam-trt.py --engine best.engine --source clip.mp4       # a recorded file
    python3 cam-trt.py --engine best.engine --source clip.mp4 --save out.mp4
    python3 cam-trt.py --engine best.engine --source clip.mp4 --loop
    python3 cam-trt.py --engine best.engine --source 0              # plain webcam
    python3 cam-trt.py --engine best.engine --fps 15 --width 1280 --height 972
    python3 cam-trt.py --engine best.engine --index 0        # skip auto-detect
    python3 cam-trt.py --engine best.engine --no-display     # headless, prints detections

Keys (window mode):
    q / Esc   quit          space  pause / resume
    n         step one frame while paused
"""

import argparse
import os
import time

import cv2

from utils.trt_engine import YOLO26TRT
from utils.utils import preprocess, postprocess, draw_detections

# ── Stream config (same defaults as cam_view.py) ────────────────────────────
DEFAULT_CAM_INDEX = 0
DEFAULT_WIDTH     = 1920 #2592
DEFAULT_HEIGHT    = 1200 #1944
DEFAULT_FPS       = 60       # MJPG supports 60fps at full 2592x1944; YUYV only
                              # goes to 35fps at that size (see --list-formats-ext)
DEFAULT_FORMAT    = "MJPG"
DISPLAY_MAX_W     = 1280     # imshow window is capped to this width so a full
DISPLAY_MAX_H     = 960      # -res frame doesn't overflow the screen
DEFAULT_ROTATE    = 270      # fixed rotation applied to every frame: 0/90/180/270

# ── Inference config ─────────────────────────────────────────────────────────
DEFAULT_IMGSZ      = 640
DEFAULT_CONF_THRES = 0.25

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


GS_CAMERA_NAME = "Global Shutter Camera"


def list_cameras():
    """List (index, name) for every /dev/videoN device via its v4l2 sysfs name."""
    cams = []
    for d in sorted(os.listdir("/dev")):
        if not d.startswith("video"):
            continue
        sys_path = f"/sys/class/video4linux/{d}/name"
        if not os.path.exists(sys_path):
            continue
        with open(sys_path) as f:
            name = f.read().strip()
        cams.append((int(d.replace("video", "")), name))
    return cams


def find_camera_index(name_substring=GS_CAMERA_NAME, default=DEFAULT_CAM_INDEX):
    """Find the /dev/videoN index whose v4l2 name contains name_substring."""
    cams = list_cameras()
    for index, name in cams:
        if name_substring.lower() in name.lower():
            return index
    print(f"[camera] '{name_substring}' not found among {cams} — "
          f"falling back to index {default}")
    return default


def gstreamer_pipeline(cam_index=DEFAULT_CAM_INDEX, width=DEFAULT_WIDTH,
                        height=DEFAULT_HEIGHT, fps=DEFAULT_FPS, format=DEFAULT_FORMAT):
    """Build a GStreamer pipeline string for v4l2src (MJPG or YUYV)."""
    QUEUE = "queue leaky=downstream max-size-buffers=1"
    SINK  = ("videoconvert ! video/x-raw, format=BGR ! "
             "appsink drop=true max-buffers=1 sync=false")

    if format.upper() == "MJPG":
        return (
            f"v4l2src device=/dev/video{cam_index} ! "
            f"image/jpeg, width={width}, height={height}, framerate={fps}/1 ! "
            f"{QUEUE} ! jpegdec ! {SINK}"
        )
    else:
        return (
            f"v4l2src device=/dev/video{cam_index} ! "
            f"video/x-raw, width={width}, height={height}, framerate={fps}/1 ! "
            f"{QUEUE} ! {SINK}"
        )


VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg")


def open_source(args):
    """(cap, label, kind) — a video file, a plain camera index, or the rig.

    kind is "file" or "camera" and decides more than where the frames come
    from. A file ends, and the read loop has to stop there; a camera does not,
    and a failed grab is worth retrying. Getting that backwards is the
    difference between a clean finish and a console filling with `frame grab
    failed` forever at the end of every clip."""
    if args.source:
        if args.source.isdigit():
            return (cv2.VideoCapture(int(args.source)),
                    f"camera index {args.source}", "camera")
        if not os.path.exists(args.source):
            raise SystemExit(f"[source] {args.source} does not exist")
        if not args.source.lower().endswith(VIDEO_EXT):
            print(f"[source] {args.source} is not a known video extension — "
                  f"trying to open it anyway")
        return cv2.VideoCapture(args.source), args.source, "file"

    index = args.index if args.index is not None else find_camera_index()
    pipeline = gstreamer_pipeline(index, args.width, args.height,
                                  args.fps, args.format)
    print(f"[camera] using /dev/video{index}")
    print(f"[camera] pipeline: {pipeline}")
    return (cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER),
            f"/dev/video{index}", "camera")


def load_class_names(path):
    if not path:
        return None
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser(description="Stream the Global Shutter Camera + run live YOLO26 TensorRT inference.")
    # source args
    ap.add_argument("--source", default=None,
                     help="video file to run on instead of the camera "
                          "(or a bare digit for a plain webcam index)")
    ap.add_argument("--loop", action="store_true",
                     help="with --source, start the file again at the end")
    ap.add_argument("--frames", type=int, default=0,
                     help="stop after N frames (0 = until the end, or until q)")
    # camera args
    ap.add_argument("--index", type=int, default=None,
                     help="Force a /dev/videoN index (skips auto-detect).")
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--format", default=DEFAULT_FORMAT, choices=["MJPG", "YUYV"])
    ap.add_argument("--rotate", type=int, default=None, choices=[0, 90, 180, 270],
                     help=f"rotate every frame by a fixed angle (clockwise). "
                          f"Default {DEFAULT_ROTATE} for the camera and 0 for "
                          f"--source, because capture_video.py rotates before "
                          f"it writes and a file is already the right way up")
    ap.add_argument("--no-display", action="store_true",
                     help="Just print FPS/detections instead of opening a window (headless).")
    # inference args
    ap.add_argument("--engine", default="best.engine", help="path to .engine file")
    ap.add_argument("--classes", default="classes.txt", help="txt file, one class name per line")
    ap.add_argument("--conf-thres", type=float, default=DEFAULT_CONF_THRES)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--save", default=None, help="optional path to record annotated video")
    args = ap.parse_args()

    cap, label, kind = open_source(args)
    if not cap.isOpened():
        raise SystemExit(
            f"[source] could not open {label}"
            + (" — is another process holding the camera?" if kind == "camera"
               else " — unsupported codec, or the file is unreadable"))

    # A recorded file knows its own size and frame rate; --width/--height/--fps
    # describe a capture that already happened and cannot change it now.
    if kind == "file":
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
        src_fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        src_w, src_h, src_fps, total = args.width, args.height, float(args.fps), 0

    rotate = args.rotate if args.rotate is not None else (
        0 if kind == "file" else DEFAULT_ROTATE)
    print(f"[source] {label} — {src_w}x{src_h} @ {src_fps:.0f} fps"
          + (f", {total} frames" if total > 0 else "")
          + f", rotate {rotate}")

    class_names = load_class_names(args.classes)
    model = YOLO26TRT(args.engine, input_size=(args.imgsz, args.imgsz))
    print(f"[model] loaded {args.engine}")

    # 90/270 rotation swaps the effective width/height for sizing the window/writer.
    disp_w, disp_h = (src_h, src_w) if rotate in (90, 270) else (src_w, src_h)

    writer = None
    if args.save:
        # The output is written at the SOURCE's frame rate, not at whatever the
        # inference loop manages — a 25 fps clip written at the 8 fps the loop
        # ran would play back three times too slow and read as dropped frames.
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                  src_fps, (disp_w, disp_h))
        if not writer.isOpened():
            raise SystemExit(f"[save] could not open {args.save} for writing")

    win_name = "YOLO26 TRT — " + label
    if not args.no_display:
        # WINDOW_NORMAL makes the window resizable; without it, imshow opens
        # at the frame's native resolution which overflows most screens. We
        # set an initial on-screen size, capped to DISPLAY_MAX_*, while the
        # capture/inference itself still runs at full resolution.
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        scale = min(DISPLAY_MAX_W / disp_w, DISPLAY_MAX_H / disp_h, 1.0)
        cv2.resizeWindow(win_name, int(disp_w * scale), int(disp_h * scale))

    prev_t = time.time()
    fps = 0.0
    n = total_dets = misses = 0
    paused = step = False
    shown = None
    # WND_PROP_VISIBLE reads 0 until the window manager has actually mapped the
    # window, which is normally a frame or two after the first imshow. Treating
    # that as "the user closed it" ends the run after one frame — so the close
    # check only counts once the window has been seen open at least once, and a
    # backend that never reports visible simply never arms it.
    was_visible = False
    try:
        while True:
            if not paused or step:
                step = False
                ok, frame = cap.read()
                if not ok:
                    if kind == "file":
                        if args.loop and n:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        break                      # end of the file: done
                    misses += 1
                    if misses > 50:
                        print("\n[camera] 50 failed grabs in a row — giving up")
                        break
                    print("[camera] frame grab failed, retrying...")
                    continue
                misses = 0
                n += 1

                frame = rotate_frame(frame, rotate)

                # ── inference ────────────────────────────────────────────
                inp, ratio, pad = preprocess(frame, model.input_size)
                raw = model.infer(inp)
                dets = postprocess(raw, ratio, pad, frame.shape, args.conf_thres)
                frame = draw_detections(frame, dets, class_names)
                total_dets += len(dets)

                # Simple running FPS: time between consecutive frames, smoothed
                # with an exponential moving average so the readout doesn't
                # jitter frame-to-frame.
                now = time.time()
                dt = now - prev_t
                prev_t = now
                if dt > 0:
                    inst_fps = 1.0 / dt
                    fps = inst_fps if fps == 0.0 else (0.9 * fps + 0.1 * inst_fps)

                progress = f"{n}/{total}" if total > 0 else str(n)
                if args.no_display:
                    print(f"[run] frame {progress}  fps={fps:.1f}  "
                          f"dets={len(dets)}", end="\r")
                else:
                    cv2.putText(frame, f"{progress}  FPS: {fps:.1f}  "
                                f"dets: {len(dets)}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (0, 255, 0), 2, cv2.LINE_AA)
                shown = frame

                # Only newly read frames are recorded — pausing would otherwise
                # write the same frame into the output over and over.
                if writer:
                    writer.write(frame)

                if args.frames and n >= args.frames:
                    break

            if not args.no_display:
                if shown is not None:
                    cv2.imshow(win_name, shown)
                # A longer wait while paused keeps a still frame from spinning
                # a core on a thousand polls a second.
                key = cv2.waitKey(30 if paused else 1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    paused = not paused
                    print(f"\n[run] {'paused' if paused else 'resumed'} at frame {n}")
                elif key == ord("n"):
                    step = True
                visible = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) >= 1
                was_visible = was_visible or visible
                if was_visible and not visible:
                    break                          # window closed with the X
    finally:
        cap.release()
        if writer:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    if n:
        print(f"\n[run] {n} frame(s), {total_dets} detection(s), "
              f"{total_dets / n:.1f} per frame")
        if args.save:
            print(f"[save] wrote {args.save} "
                  f"({disp_w}x{disp_h} @ {src_fps:.0f} fps)")
    else:
        print("\n[run] no frames read")


if __name__ == "__main__":
    main()
