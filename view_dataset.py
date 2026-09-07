#!/usr/bin/env python3
"""
Look through a dataset built by make_dataset.py and check its boxes by eye.

Reads either annotations.json (COCO) or the per-image labels/*.txt (YOLO) and
draws them over the image, one frame at a time, in the same colours the
capture preview used: label red, qr_code cyan-blue, logo yellow. Both readers
end up in the same shape, so you can flip --source to confirm the two views of
the dataset actually agree.

It opens a folder in whichever shape it is in — flat out of make_dataset.py, or
already dealt into train/val by split_dataset.py — and takes the class names
from data.yaml when there is one, because that is the file the trainer reads.

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
    python3 view_dataset.py                              # dataset/, any layout
    python3 view_dataset.py --dataset dataset/data.yaml  # point at the yaml
    python3 view_dataset.py --split val                  # just the val side
    python3 view_dataset.py --source coco                # force annotations.json
    python3 view_dataset.py --start 120                  # open at an index
    python3 view_dataset.py --only upside_down           # just the rotated copies
    python3 view_dataset.py --stats                      # print a summary, no window
    python3 view_dataset.py --export out/                # write every frame, headless
"""

import argparse
import json
import sys
from collections import Counter, defaultdict, namedtuple
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

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

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

# path is what cv2 opens, name is what the HUD shows (relative to the dataset),
# split is "" for a flat folder and "train"/"val" once it has been dealt.
Frame = namedtuple("Frame", "path name variant split boxes")


def read_yaml(path):
    """data.yaml as a dict. PyYAML if it is installed, a two-key reader if not.

    The fallback only has to survive what split_dataset.py and Ultralytics
    write — `key: value` lines plus a `names:` block in either the mapping or
    the list form — and a viewer is not worth a hard dependency."""
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        pass
    except Exception as exc:
        print(f"[view] {path.name} unreadable ({exc}) — ignoring it")
        return {}

    cfg, names, in_names = {}, {}, False
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indented = line[:1].isspace()
        line = line.strip()
        if in_names and (indented or line.startswith("-")):
            if line.startswith("-"):
                names[len(names)] = line[1:].strip()
            elif ":" in line:
                k, v = line.split(":", 1)
                names[int(k.strip())] = v.strip()
            continue
        in_names = False
        if line.rstrip(":") == "names":
            in_names = True
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            cfg[k.strip()] = v.strip()
    if names:
        cfg["names"] = [names[k] for k in sorted(names)]
    return cfg


def find_yaml(root):
    for name in ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml"):
        if (root / name).exists():
            return root / name
    hits = sorted(root.glob("*.yaml")) + sorted(root.glob("*.yml"))
    return hits[0] if hits else None


def resolve_root(arg):
    """--dataset takes a folder or a data.yaml. Returns the folder to read.

    A yaml written by split_dataset.py carries an ABSOLUTE `path:`, which goes
    stale the moment the folder is copied or moved. So the yaml's own folder
    wins whenever it holds the images, and `path:` is only followed when it
    does not — otherwise pointing at a yaml on a second machine would quietly
    open the dataset it was exported from, or nothing at all."""
    p = Path(arg)
    if p.is_file() and p.suffix.lower() in (".yaml", ".yml"):
        here = p.parent
        if (here / "images").is_dir() or any(here.glob(f"*{e}") for e in IMG_EXT):
            return here
        stated = read_yaml(p).get("path")
        if stated and Path(stated).is_dir():
            return Path(stated)
        return here
    return p


def load_class_names(root):
    """data.yaml first: it is what the trainer reads, so where the two disagree
    the yaml is the version a trained model's class indices actually mean."""
    yml = find_yaml(root)
    if yml:
        names = read_yaml(yml).get("names")
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names, key=lambda k: int(k))]
        if isinstance(names, list) and names:
            return [str(n) for n in names]

    path = root / "classes.txt"
    if path.exists():
        names = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
        if names:
            return names
    return ["label", "qr_code", "logo"]


def find_images(root):
    """Every image in the dataset, with the split it sits in, across the four
    layouts these scripts and Ultralytics produce:

        <ds>/images/*.jpg              flat, straight out of make_dataset.py
        <ds>/images/train/*.jpg        after split_dataset.py
        <ds>/train/images/*.jpg        the other common Ultralytics shape
        <ds>/*.jpg beside <ds>/*.txt   a folder labelled by hand

    Scanning for all four rather than asking for a --layout flag matters
    because the folder changes shape under you: split_dataset.py moves the
    files, and a viewer that only knew the flat form would come up empty on a
    dataset that is merely one command further along."""
    found, seen = [], set()

    def add(path, split):
        if path.suffix.lower() not in IMG_EXT or not path.is_file():
            return
        key = path.resolve()
        if key in seen:
            return
        seen.add(key)
        found.append((path, split))

    images_dir = root / "images"
    if images_dir.is_dir():
        for p in sorted(images_dir.iterdir()):
            add(p, "")
        for sub in sorted(d for d in images_dir.iterdir() if d.is_dir()):
            for p in sorted(sub.rglob("*")):
                add(p, sub.name)

    for sub in sorted(d for d in root.iterdir() if d.is_dir()):
        if sub.name in ("images", "preview"):
            continue
        if (sub / "images").is_dir():
            for p in sorted((sub / "images").rglob("*")):
                add(p, sub.name)

    for p in sorted(root.iterdir()):
        add(p, "")
    return found


def label_for(image_path):
    """The .txt for an image: under labels/ at the mirrored path, or beside it.

    Ultralytics finds labels by swapping the last `images` component of the
    path for `labels`, so that is tried first and on the same component it
    would use; a hand-labelled folder keeps the txt next to the jpg instead."""
    parts = list(image_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            swapped = Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")
            return swapped if swapped.exists() else None
    beside = image_path.with_suffix(".txt")
    return beside if beside.exists() else None


def rel_name(path, root):
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def variant_of(path):
    return "upside_down" if path.stem.endswith("_r180") else "straight"


def coco_docs(root):
    """The COCO files to read, newest layout first.

    split_dataset.py writes annotations_train.json and annotations_val.json but
    leaves the flat annotations.json in place, so reading every match would
    list each frame twice. The flat file, when present, is the whole set."""
    flat = root / "annotations.json"
    if flat.exists():
        return [flat]
    return [p for p in (root / "annotations_train.json",
                        root / "annotations_val.json") if p.exists()]


def load_coco(root):
    """[Frame, ...] from the COCO file(s)."""
    docs = coco_docs(root)
    if not docs:
        sys.exit(f"[view] no annotations.json under {root} — "
                 f"try --source yolo to read labels/*.txt")

    # file_name in COCO is a bare name; where it lives on disk depends on
    # whether the folder has been split since, so it is looked up, not joined.
    index = {}
    for path, split in find_images(root):
        index.setdefault(path.name, (path, split))

    frames, names = [], None
    for doc_path in docs:
        with open(doc_path) as f:
            doc = json.load(f)
        if names is None and doc.get("categories"):
            names = [c["name"] for c in sorted(doc["categories"], key=lambda c: c["id"])]

        by_image = defaultdict(list)
        for a in doc.get("annotations", []):
            x, y, w, h = a["bbox"]
            by_image[a["image_id"]].append(
                (a["category_id"] - 1, int(x), int(y), int(x + w), int(y + h), a.get("text")))

        fallback_split = doc_path.stem.replace("annotations_", "") \
            if doc_path.stem != "annotations" else ""
        for img in doc.get("images", []):
            file_name = img["file_name"]
            path, split = index.get(Path(file_name).name,
                                    (root / "images" / file_name, fallback_split))
            frames.append(Frame(path, rel_name(path, root),
                                img.get("variant", variant_of(path)),
                                split or fallback_split, by_image.get(img["id"], [])))
    return frames, names or load_class_names(root)


def load_yolo(root):
    """Same shape, read back from the .txt labels. Needs each image to size the
    normalised numbers, so it reads dimensions as it goes."""
    entries = find_images(root)
    if not entries:
        sys.exit(f"[view] no images found under {root}")

    frames = []
    for path, split in entries:
        label_path = label_for(path)
        boxes = []
        if label_path:
            img = cv2.imread(str(path))
            if img is not None:
                h, w = img.shape[:2]
                for line in label_path.read_text().split("\n"):
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:5])
                    boxes.append((cls,
                                  int((cx - bw / 2) * w), int((cy - bh / 2) * h),
                                  int((cx + bw / 2) * w), int((cy + bh / 2) * h), None))
        frames.append(Frame(path, rel_name(path, root), variant_of(path), split, boxes))
    return frames, load_class_names(root)


def print_stats(frames, names):
    per_class = Counter()
    per_variant = Counter()
    per_split = Counter()
    empty = 0
    for f in frames:
        per_variant[f.variant] += 1
        per_split[f.split or "flat"] += 1
        if not f.boxes:
            empty += 1
        for b in f.boxes:
            per_class[b[0]] += 1

    print(f"images      : {len(frames)}")
    print(f"annotations : {sum(per_class.values())}")
    for i, name in enumerate(names):
        print(f"  {i} {name:<10}: {per_class.get(i, 0)}")
    stray = sorted(c for c in per_class if c >= len(names) or c < 0)
    for c in stray:
        print(f"  {c} <not in classes>: {per_class[c]}")
    print("splits      : " + ", ".join(f"{k}={v}" for k, v in sorted(per_split.items())))
    print("variants    : " + ", ".join(f"{k}={v}" for k, v in sorted(per_variant.items())))
    if empty:
        print(f"images with no boxes: {empty}")
    per_image = [len(f.boxes) for f in frames]
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
            name = names[cls] if 0 <= cls < len(names) else str(cls)
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
    ap.add_argument("--dataset", default="dataset",
                     help="dataset directory, or a data.yaml inside one")
    ap.add_argument("--source", choices=["auto", "coco", "yolo"], default="auto",
                     help="auto: annotations.json if there is one, else the "
                          "per-image labels/*.txt")
    ap.add_argument("--split", default="all",
                     help="all (default), or one of the splits found in the "
                          "folder — typically train or val")
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

    root = resolve_root(args.dataset)
    if not root.is_dir():
        sys.exit(f"[view] {root} is not a directory")

    source = args.source
    if source == "auto":
        source = "coco" if coco_docs(root) else "yolo"
    frames, names = load_coco(root) if source == "coco" else load_yolo(root)

    splits = sorted({f.split for f in frames if f.split})
    if args.split != "all":
        if args.split not in splits:
            sys.exit(f"[view] no split named '{args.split}' in {root} — "
                     f"found {', '.join(splits) if splits else 'none (flat folder)'}")
        frames = [f for f in frames if f.split == args.split]
    if args.only != "all":
        frames = [f for f in frames if f.variant == args.only]
    if not frames:
        sys.exit(f"[view] nothing to show in {root} "
                 f"(--split {args.split} --only {args.only})")

    yml = find_yaml(root)
    print(f"[view] {root.resolve()} — {len(frames)} images, "
          f"{sum(len(f.boxes) for f in frames)} boxes, source={source}"
          + (f", splits: {', '.join(splits)}" if splits else ""))
    print(f"[view] classes: " + ", ".join(f"{i}={n}" for i, n in enumerate(names))
          + (f"  (from {yml.name})" if yml else ""))
    unlabelled = sum(1 for f in frames if not f.boxes)
    if unlabelled:
        print(f"[view] {unlabelled} image(s) with no boxes — missing or empty .txt")

    if args.stats:
        print_stats(frames, names)
        return

    show = {i: True for i in range(max(len(names), 3))}
    caption_mode = args.captions

    if args.export:
        out_dir = Path(args.export)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames, 1):
            image = cv2.imread(str(f.path))
            if image is None:
                print(f"[view] missing {f.name}, skipped")
                continue
            h, w = image.shape[:2]
            scale = min(DISPLAY_MAX_W / w, DISPLAY_MAX_H / h, 1.0)
            view, _ = render(image, f.boxes, names, scale, show, caption_mode)
            # train/frame_0001.jpg and val/frame_0001.jpg flatten to one folder,
            # so the split stays in the filename rather than overwriting.
            cv2.imwrite(str(out_dir / f.name.replace("/", "_")), view)
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
        frame = frames[index]
        image = cv2.imread(str(frame.path))
        if image is None:
            # A missing or unreadable file should not end the session — show a
            # blank plate saying so, and let the keys keep working.
            view = np.zeros((DISPLAY_MAX_H, DISPLAY_MAX_W, 3), np.uint8)
            lines = [f"[{index + 1}/{len(frames)}]  {frame.name}",
                     "MISSING OR UNREADABLE"]
        else:
            h, w = image.shape[:2]
            scale = min(DISPLAY_MAX_W / w, DISPLAY_MAX_H / h, 1.0)
            view, drawn = render(image, frame.boxes, names, scale, show, caption_mode)
            tally = "  ".join(
                f"{'' if show.get(i, True) else '('}{n}: {drawn.get(i, 0)}"
                f"{'' if show.get(i, True) else ' off)'}"
                for i, n in enumerate(names))
            tag = f"{frame.split}, " if frame.split else ""
            lines = [f"[{index + 1}/{len(frames)}]  {frame.name}  ({tag}{frame.variant})",
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
            # Mirrors the split in the name for the same reason --export does.
            preview_dir.mkdir(parents=True, exist_ok=True)
            out = preview_dir / frame.name.replace("/", "_")
            cv2.imwrite(str(out), view)
            print(f"[view] wrote {out}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
