#!/usr/bin/env python3
"""
Look through a dataset built by make_dataset.py and check its boxes by eye.

Reads either annotations.json (COCO) or the per-image labels/*.txt (YOLO) and
draws them over the image, one frame at a time, in the same colours the
capture preview used: label red, qr_code cyan-blue, logo yellow. Both readers
end up in the same shape, so you can flip --source to confirm the two views of
the dataset actually agree.

Keys:
    n / space / ->    next image          p / <-    previous image
    N / P             jump 10 forward / back
    g                 go to the first image        G   go to the last
    1 2 3             toggle a class on/off
    t                 cycle captions: off -> class names -> names + QR payload
    s                 write the current annotated view to <dataset>/preview/
    h                 toggle the key help
    q / Esc           quit

Usage:
    python3 view_dataset.py                              # dataset/, COCO
    python3 view_dataset.py --dataset dataset/run2
    python3 view_dataset.py --source yolo                # read labels/*.txt
    python3 view_dataset.py --start 120                  # open at an index
    python3 view_dataset.py --only upside_down           # just the rotated copies
    python3 view_dataset.py --stats                      # print a summary, no window
    python3 view_dataset.py --export out/                # write every frame, headless
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

# Same colours as the capture preview, so a frame looks the same here as it did
# when it was banked.
CLASS_COLORS = [
    (0, 0, 255),      # 0 label    – red, the card box
    (255, 200, 0),    # 1 qr_code  – cyan-blue, the zxing symbol
    (0, 255, 255),    # 2 logo     – yellow
]
FALLBACK_COLOR = (200, 200, 200)

# Captions have three settings because a sheet holds ~20 QR codes and their
# payloads are long URLs — drawn all at once they overlap into a smear, so the
# payload is a deliberate third step rather than the default.
CAPTION_MODES = ["off", "names", "text"]
CAPTION_TEXT_CHARS = 18

DISPLAY_MAX_W = 1100
DISPLAY_MAX_H = 900
HUD_BG = (30, 30, 30)

HELP_LINES = [
    "n/space/->  next        p/<-  prev",
    "N/P  jump 10            g/G  first/last",
    "1 2 3  toggle class     t  captions (3 modes)",
    "s  save view            h  this help",
    "q/Esc  quit",
]

# GTK builds of OpenCV return the keycode with modifier bits set above the low
# byte — 'n' arrives as 1048686 (0x10006E), not 110 — so every key is masked to
# its low byte before being compared. That also folds the arrow keys down to a
# single byte: left 65361 -> 81, right 65363 -> 83. Those collide with upper-
# case Q and S, which is why nothing is bound to them.
LEFT_KEYS  = {81}
RIGHT_KEYS = {83}


def load_class_names(root):
    path = root / "classes.txt"
    if path.exists():
        names = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
        if names:
            return names
    return ["label", "qr_code", "logo"]


def load_coco(root):
    """[(file_name, variant, [(cls, x1, y1, x2, y2, text), ...]), ...] from COCO."""
    path = root / "annotations.json"
    if not path.exists():
        sys.exit(f"[view] {path} not found — use --source yolo, or point --dataset elsewhere")
    with open(path) as f:
        doc = json.load(f)

    by_image = defaultdict(list)
    for a in doc["annotations"]:
        x, y, w, h = a["bbox"]
        by_image[a["image_id"]].append(
            (a["category_id"] - 1, int(x), int(y), int(x + w), int(y + h), a.get("text")))

    names = [c["name"] for c in sorted(doc["categories"], key=lambda c: c["id"])]
    frames = [(img["file_name"], img.get("variant", "?"), by_image.get(img["id"], []))
              for img in doc["images"]]
    return frames, names


def load_yolo(root):
    """Same shape, read back from labels/*.txt. Needs each image to size the
    normalised numbers, so it reads dimensions as it goes."""
    images_dir, labels_dir = root / "images", root / "labels"
    if not labels_dir.exists():
        sys.exit(f"[view] {labels_dir} not found — this dataset has no YOLO labels")

    frames = []
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        boxes = []
        if label_path.exists():
            img = cv2.imread(str(image_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            for line in label_path.read_text().split("\n"):
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
                boxes.append((cls,
                              int((cx - bw / 2) * w), int((cy - bh / 2) * h),
                              int((cx + bw / 2) * w), int((cy + bh / 2) * h), None))
        variant = "upside_down" if image_path.stem.endswith("_r180") else "straight"
        frames.append((image_path.name, variant, boxes))
    return frames, load_class_names(root)


def print_stats(frames, names):
    per_class = Counter()
    per_variant = Counter()
    empty = 0
    for _, variant, boxes in frames:
        per_variant[variant] += 1
        if not boxes:
            empty += 1
        for b in boxes:
            per_class[b[0]] += 1

    print(f"images      : {len(frames)}")
    print(f"annotations : {sum(per_class.values())}")
    for i, name in enumerate(names):
        print(f"  {i} {name:<10}: {per_class.get(i, 0)}")
    print("variants    : " + ", ".join(f"{k}={v}" for k, v in sorted(per_variant.items())))
    if empty:
        print(f"images with no boxes: {empty}")
    per_image = [len(b) for _, _, b in frames]
    if per_image:
        print(f"boxes per image: min {min(per_image)}, "
              f"mean {sum(per_image)/len(per_image):.1f}, max {max(per_image)}")


def render(image, boxes, names, scale, show, caption_mode):
    """Resize first, then draw, so box outlines stay one pixel wide on screen
    instead of vanishing into the downscale."""
    view = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    drawn = Counter()
    for cls, x1, y1, x2, y2, text in boxes:
        if not show.get(cls, True):
            continue
        drawn[cls] += 1
        color = CLASS_COLORS[cls] if 0 <= cls < len(CLASS_COLORS) else FALLBACK_COLOR
        p1 = (int(x1 * scale), int(y1 * scale))
        p2 = (int(x2 * scale), int(y2 * scale))
        cv2.rectangle(view, p1, p2, color, 2)
        if caption_mode != "off":
            name = names[cls] if cls < len(names) else str(cls)
            caption = (f"{name}: {text[:CAPTION_TEXT_CHARS]}"
                       if caption_mode == "text" and text else name)
            cv2.putText(view, caption, (p1[0], max(12, p1[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return view, drawn


def draw_hud(view, lines, corner=(0, 0)):
    """A dark plate behind the text so it stays readable over any frame."""
    pad = 6
    sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0] for t in lines]
    box_w = max(w for w, _ in sizes) + 2 * pad
    box_h = sum(h for _, h in sizes) + pad * (len(lines) + 1)
    x0, y0 = corner
    overlay = view.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), HUD_BG, -1)
    cv2.addWeighted(overlay, 0.65, view, 0.35, 0, view)
    y = y0 + pad
    for text, (_, th) in zip(lines, sizes):
        y += th
        cv2.putText(view, text, (x0 + pad, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += pad
    return view


def main():
    ap = argparse.ArgumentParser(description="View the boxes in a make_dataset.py dataset.")
    ap.add_argument("--dataset", default="dataset", help="dataset directory")
    ap.add_argument("--source", choices=["coco", "yolo"], default="coco",
                     help="read annotations.json or the per-image labels/*.txt")
    ap.add_argument("--start", type=int, default=0, help="index to open at")
    ap.add_argument("--only", choices=["all", "straight", "upside_down"], default="all",
                     help="show only one variant")
    ap.add_argument("--captions", choices=CAPTION_MODES, default="names",
                     help="off: boxes only; names: class name (default); "
                          "text: class name plus the decoded QR payload")
    ap.add_argument("--stats", action="store_true", help="print a summary and exit")
    ap.add_argument("--debug-keys", action="store_true",
                     help="print the code of every key pressed, to diagnose bindings")
    ap.add_argument("--export", default=None,
                     help="write every annotated frame to this directory and exit "
                          "(no window needed)")
    args = ap.parse_args()

    root = Path(args.dataset)
    if not root.is_dir():
        sys.exit(f"[view] {root} is not a directory")

    frames, names = load_coco(root) if args.source == "coco" else load_yolo(root)
    if args.only != "all":
        frames = [f for f in frames if f[1] == args.only]
    if not frames:
        sys.exit(f"[view] nothing to show in {root} (--only {args.only})")

    print(f"[view] {root.resolve()} — {len(frames)} images, "
          f"{sum(len(b) for _, _, b in frames)} boxes, source={args.source}")
    print(f"[view] classes: " + ", ".join(f"{i}={n}" for i, n in enumerate(names)))

    if args.stats:
        print_stats(frames, names)
        return

    images_dir = root / "images"
    show = {i: True for i in range(max(len(names), 3))}
    caption_mode = args.captions

    if args.export:
        out_dir = Path(args.export)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, (name, _, boxes) in enumerate(frames, 1):
            image = cv2.imread(str(images_dir / name))
            if image is None:
                print(f"[view] missing {name}, skipped")
                continue
            h, w = image.shape[:2]
            scale = min(DISPLAY_MAX_W / w, DISPLAY_MAX_H / h, 1.0)
            view, _ = render(image, boxes, names, scale, show, caption_mode)
            cv2.imwrite(str(out_dir / name), view)
            print(f"\r[view] exported {i}/{len(frames)}", end="")
        print(f"\n[view] wrote {len(frames)} frames to {out_dir.resolve()}")
        return

    win = "view_dataset"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, DISPLAY_MAX_W, DISPLAY_MAX_H)

    index = max(0, min(args.start, len(frames) - 1))
    help_on = False
    preview_dir = root / "preview"

    while True:
        name, variant, boxes = frames[index]
        image = cv2.imread(str(images_dir / name))
        if image is None:
            # A missing or unreadable file should not end the session — show a
            # blank plate saying so, and let the keys keep working.
            view = np.zeros((DISPLAY_MAX_H, DISPLAY_MAX_W, 3), np.uint8)
            lines = [f"[{index + 1}/{len(frames)}]  {name}",
                     "MISSING OR UNREADABLE in images/"]
        else:
            h, w = image.shape[:2]
            scale = min(DISPLAY_MAX_W / w, DISPLAY_MAX_H / h, 1.0)
            view, drawn = render(image, boxes, names, scale, show, caption_mode)
            tally = "  ".join(
                f"{'' if show.get(i, True) else '('}{n}: {drawn.get(i, 0)}"
                f"{'' if show.get(i, True) else ' off)'}"
                for i, n in enumerate(names))
            lines = [f"[{index + 1}/{len(frames)}]  {name}  ({variant})",
                     f"{w}x{h}   {tally}   captions: {caption_mode}"]
        if help_on:
            lines = lines + HELP_LINES
        else:
            lines = lines + ["h for keys"]
        draw_hud(view, lines)
        cv2.imshow(win, view)

        raw = cv2.waitKeyEx(0)
        # Closing the window with its X button leaves waitKey returning -1 on a
        # dead window; without this the loop would spin forever.
        if raw < 0:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            continue
        key = raw & 0xFF
        if args.debug_keys:
            print(f"[view] key raw={raw} masked={key} "
                  f"({chr(key) if 32 <= key < 127 else '-'})")

        if key in (ord("q"), 27):
            break
        elif key in RIGHT_KEYS or key in (ord("n"), ord(" ")):
            index = (index + 1) % len(frames)
        elif key in LEFT_KEYS or key == ord("p"):
            index = (index - 1) % len(frames)
        elif key == ord("N"):
            index = min(index + 10, len(frames) - 1)
        elif key == ord("P"):
            index = max(index - 10, 0)
        elif key == ord("g"):
            index = 0
        elif key == ord("G"):
            index = len(frames) - 1
        elif key in (ord("1"), ord("2"), ord("3")):
            cls = key - ord("1")
            show[cls] = not show.get(cls, True)
        elif key == ord("t"):
            caption_mode = CAPTION_MODES[(CAPTION_MODES.index(caption_mode) + 1)
                                          % len(CAPTION_MODES)]
        elif key == ord("h"):
            help_on = not help_on
        elif key == ord("s"):
            preview_dir.mkdir(parents=True, exist_ok=True)
            out = preview_dir / name
            cv2.imwrite(str(out), view)
            print(f"[view] wrote {out}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
