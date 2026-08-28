#!/usr/bin/env python3
"""
Global Shutter Camera + live YOLO26 TensorRT inference, with a geometric
check for label detections the model MISSED at high speed.

The labels flow horizontally in a regular grid, so each row of labels lies on
a 1-D lattice of equally spaced x-centres. The average distance between
neighbouring labels (the pitch) is measured live from the smallest recurring
gaps; any row gap that comes out as ~2x / ~3x that pitch means the detector
dropped one/two labels there, and any row that is short at one end while its
neighbour rows are not means a label was dropped at the edge. Missed slots are
drawn in red and can be logged to CSV / dumped as frames for review.

Usage:
    python3 lable_missing_test.py --engine best.engine
    python3 lable_missing_test.py --engine best.engine --miss-log misses.csv
    python3 lable_missing_test.py --engine best.engine --save-miss-frames misses/
    python3 lable_missing_test.py --engine best.engine --pitch 190   # fix the pitch
    python3 lable_missing_test.py --engine best.engine --no-miss-check
    python3 lable_missing_test.py --engine best.engine --no-display  # headless
"""

import argparse
import os
import time

import cv2

from utils.grid import MissingLabelDetector
from utils.trt_engine import YOLO26TRT
from utils.utils import preprocess, postprocess, draw_detections

# ── Stream config (same defaults as cam_view.py) ────────────────────────────
DEFAULT_CAM_INDEX = 0
DEFAULT_WIDTH     = 2592
DEFAULT_HEIGHT    = 1944
DEFAULT_FPS       = 60       # MJPG supports 60fps at full 2592x1944; YUYV only
                              # goes to 35fps at that size (see --list-formats-ext)
DEFAULT_FORMAT    = "MJPG"
DISPLAY_MAX_W     = 1280     # imshow window is capped to this width so a full
DISPLAY_MAX_H     = 960      # -res frame doesn't overflow the screen
DEFAULT_ROTATE    = 270      # fixed rotation applied to every frame: 0/90/180/270

# ── Inference config ─────────────────────────────────────────────────────────
DEFAULT_IMGSZ      = 640
DEFAULT_CONF_THRES = 0.25

# ── Missing-label check config ───────────────────────────────────────────────
DEFAULT_LABEL_CLASS = "label"   # class laid out on the grid
MISS_PRINT_EVERY    = 0.5       # s, rate-limit for repeating the same complaint

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


def load_class_names(path):
    if not path:
        return None
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def class_index(class_names, name, fallback):
    """Resolve a class name to its id; fall back to a fixed index when no
    classes.txt was supplied (or the name isn't in it)."""
    if class_names and name in class_names:
        return class_names.index(name)
    print(f"[miss] class '{name}' not found in --classes, using index {fallback}")
    return fallback


def main():
    ap = argparse.ArgumentParser(description="Stream the Global Shutter Camera + run live YOLO26 TensorRT inference.")
    # camera args
    ap.add_argument("--index", type=int, default=None,
                     help="Force a /dev/videoN index (skips auto-detect).")
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--format", default=DEFAULT_FORMAT, choices=["MJPG", "YUYV"])
    ap.add_argument("--rotate", type=int, default=DEFAULT_ROTATE, choices=[0, 90, 180, 270],
                     help="Rotate every frame by a fixed angle (clockwise).")
    ap.add_argument("--no-display", action="store_true",
                     help="Just print FPS/detections instead of opening a window (headless).")
    # inference args
    ap.add_argument("--engine", default="best.engine", help="path to .engine file")
    ap.add_argument("--classes", default="classes.txt", help="txt file, one class name per line")
    ap.add_argument("--conf-thres", type=float, default=DEFAULT_CONF_THRES)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--save", default=None, help="optional path to record annotated video")
    # missing-label check args
    ap.add_argument("--no-miss-check", action="store_true",
                     help="disable the grid-based missing-label check.")
    ap.add_argument("--label-class", default=DEFAULT_LABEL_CLASS,
                     help="class name laid out on the grid.")
    ap.add_argument("--pitch", type=float, default=None,
                     help="fix the label-to-label distance in px instead of measuring it.")
    ap.add_argument("--lattice-tol", type=float, default=0.30,
                     help="how far a gap may sit off a whole pitch multiple (0-1).")
    ap.add_argument("--no-end-check", action="store_true",
                     help="only flag gaps inside a row, never a short row end.")
    ap.add_argument("--miss-log", default=None,
                     help="CSV file to append every missed slot to.")
    ap.add_argument("--save-miss-frames", default=None,
                     help="directory to dump annotated frames that contain a miss.")
    args = ap.parse_args()

    cam_index = args.index if args.index is not None else find_camera_index()
    pipeline = gstreamer_pipeline(cam_index, args.width, args.height, args.fps, args.format)
    print(f"[camera] using /dev/video{cam_index}")
    print(f"[camera] pipeline: {pipeline}")

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera via GStreamer pipeline")

    class_names = load_class_names(args.classes)
    model = YOLO26TRT(args.engine, input_size=(args.imgsz, args.imgsz))
    print(f"[model] loaded {args.engine}")

    # 90/270 rotation swaps the effective width/height for sizing the window/writer.
    disp_w, disp_h = (args.height, args.width) if args.rotate in (90, 270) else (args.width, args.height)

    checker = None
    if not args.no_miss_check:
        checker = MissingLabelDetector(
            label_cls=class_index(class_names, args.label_class, 0),
            pitch=args.pitch,
            lattice_tol=args.lattice_tol,
            end_check=not args.no_end_check,
        )
        print(f"[miss] grid check on '{args.label_class}'"
              f"{f' with a fixed pitch of {args.pitch:.0f}px' if args.pitch else ''}")

    miss_log = None
    if args.miss_log:
        new_file = not os.path.exists(args.miss_log)
        miss_log = open(args.miss_log, "a", buffering=1)
        if new_file:
            miss_log.write("frame,time,row,kind,x,y,pitch,n_labels\n")
        print(f"[miss] logging missed slots to {args.miss_log}")

    if args.save_miss_frames:
        os.makedirs(args.save_miss_frames, exist_ok=True)

    writer = None
    if args.save:
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                  args.fps, (disp_w, disp_h))

    if not args.no_display:
        # WINDOW_NORMAL makes the window resizable; without it, imshow opens
        # at the frame's native resolution which overflows most screens. We
        # set an initial on-screen size, capped to DISPLAY_MAX_*, while the
        # capture/inference itself still runs at full resolution.
        win_name = "Global Shutter Camera - YOLO26 TRT"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        scale = min(DISPLAY_MAX_W / disp_w, DISPLAY_MAX_H / disp_h, 1.0)
        cv2.resizeWindow(win_name, int(disp_w * scale), int(disp_h * scale))

    try:
        prev_t = time.time()
        fps = 0.0
        frame_idx = 0
        last_miss_print = 0.0
        last_miss_count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[camera] frame grab failed, retrying...")
                continue

            frame = rotate_frame(frame, args.rotate)

            # ── inference ────────────────────────────────────────────────
            inp, ratio, pad = preprocess(frame, model.input_size)
            raw = model.infer(inp)
            dets = postprocess(raw, ratio, pad, frame.shape, args.conf_thres)
            frame_idx += 1

            # ── missing-label check ──────────────────────────────────────
            # Rows of labels sit on an even lattice, so an empty slot in a row
            # is a detection the model dropped — no tracker needed.
            misses = checker.update(dets, frame.shape) if checker else []

            frame = draw_detections(frame, dets, class_names)
            if checker:
                checker.draw(frame)

            if misses:
                now_t = time.time()
                # The same physical label stays missing for many frames as it
                # travels, so only shout when the count changes or twice a second.
                if (len(misses) != last_miss_count
                        or now_t - last_miss_print >= MISS_PRINT_EVERY):
                    where = ", ".join(f"row {m['row']} @ x={m['x']:.0f} ({m['kind']})"
                                      for m in misses[:4])
                    more = f" +{len(misses) - 4} more" if len(misses) > 4 else ""
                    print(f"\n[miss] frame {frame_idx}: {len(misses)} label(s) "
                          f"not detected — {where}{more}")
                    last_miss_print = now_t
                    last_miss_count = len(misses)
                if miss_log:
                    for m in misses:
                        miss_log.write(f"{frame_idx},{now_t:.3f},{m['row']},{m['kind']},"
                                       f"{m['x']:.1f},{m['y']:.1f},"
                                       f"{checker.pitch:.1f},{checker.n_labels}\n")
                if args.save_miss_frames:
                    cv2.imwrite(os.path.join(args.save_miss_frames,
                                             f"miss_{frame_idx:06d}.jpg"), frame)
            else:
                last_miss_count = 0

            # Simple running FPS: time between consecutive frames, smoothed
            # with an exponential moving average so the readout doesn't
            # jitter frame-to-frame.
            now = time.time()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                inst_fps = 1.0 / dt
                fps = inst_fps if fps == 0.0 else (0.9 * fps + 0.1 * inst_fps)

            if args.no_display:
                miss_status = (f"  labels={checker.n_labels}  missing={len(misses)}"
                               if checker else "")
                print(f"[camera] frame {frame.shape}  fps={fps:.1f}  "
                      f"dets={len(dets)}{miss_status}   ", end="\r")
            else:
                cv2.putText(frame, f"FPS: {fps:.1f}  dets: {len(dets)}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow(win_name, frame)

            if writer:
                writer.write(frame)

            if not args.no_display and (cv2.waitKey(1) & 0xFF == ord("q")):
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        if miss_log:
            miss_log.close()
        if checker:
            print(f"\n[miss] {checker.summary()}")
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
