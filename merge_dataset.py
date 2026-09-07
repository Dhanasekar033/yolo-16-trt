#!/usr/bin/env python3
"""
Merge one YOLO dataset folder into another, renaming what would collide.

Two capture runs both numbered from frame_000001, so a straight copy would have
the second run's frame_000001.jpg land on the first run's — and the label file
with it. The overwrite is silent, the count still looks right, and the frames
that vanished are the ones you would never think to check. So nothing is ever
overwritten here: a name already taken in the destination is renamed, and the
image and its .txt are renamed TOGETHER to the same new stem, because a label
that loses its image is not a label any more.

The rename inserts a tag rather than appending one:

    frame_000001.jpg       ->  frame_000001__ds2.jpg
    frame_000001_r180.jpg  ->  frame_000001__ds2_r180.jpg

`_r180` has to stay at the end. make_dataset.py marks the rotated copies with
it and view_dataset.py reads the variant straight off the end of the stem, so a
tag appended after it would turn every rotated frame into an unmarked one.

CLASS NAMES ARE CHECKED FIRST and a mismatch stops the merge. Two datasets can
both be nc=2 and mean entirely different things by index 1; merging those needs
a remap, not a copy, and no error would ever surface it — the model would just
be wrong. Same names in the same order, or this refuses to run.

Files are COPIED by default, so the source is still there if the merge was not
what you wanted. Every rename is written to merge_report.txt in the destination.

    python3 merge_dataset.py --src dataset-2 --dry-run     # what would happen
    python3 merge_dataset.py --src dataset-2               # do it
    python3 merge_dataset.py --src dataset-2 --into train  # all of it into train
    python3 merge_dataset.py --src dataset-2 --move        # don't keep a copy
"""

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
ROT_SUFFIX = "_r180"


def read_yaml(path):
    """Only the names matter here, and only for the compatibility check."""
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        pass
    except Exception:
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


def read_names(root):
    path = root / "classes.txt"
    if path.exists():
        names = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
        if names:
            return names, path.name
    yml = find_yaml(root)
    if yml:
        names = read_yaml(yml).get("names")
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names, key=lambda k: int(k))]
        if isinstance(names, list) and names:
            return [str(n) for n in names], yml.name
    return None, None


def find_pairs(root):
    """[(image, label or None, split), ...] over the layouts these folders take.

    Split is "" for a flat folder, otherwise train/val — which is what lets the
    merge drop each side of the source onto the matching side of the
    destination instead of flattening the split someone already made."""
    pairs, seen = [], set()

    def label_for(image_path):
        parts = list(image_path.parts)
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] == "images":
                swapped = Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")
                return swapped if swapped.exists() else None
        beside = image_path.with_suffix(".txt")
        return beside if beside.exists() else None

    def add(path, split):
        if path.suffix.lower() not in IMG_EXT or not path.is_file():
            return
        key = path.resolve()
        if key in seen:
            return
        seen.add(key)
        pairs.append((path, label_for(path), split))

    images_dir = root / "images"
    if images_dir.is_dir():
        for p in sorted(images_dir.iterdir()):
            add(p, "")
        for sub in sorted(d for d in images_dir.iterdir() if d.is_dir()):
            for p in sorted(sub.rglob("*")):
                add(p, sub.name)
    for sub in sorted(d for d in root.iterdir() if d.is_dir()):
        if sub.name in ("images", "preview", "labels_backup"):
            continue
        if (sub / "images").is_dir():
            for p in sorted((sub / "images").rglob("*")):
                add(p, sub.name)
    for p in sorted(root.iterdir()):
        add(p, "")
    return pairs


def taken_stems(root):
    """Every image stem already in the destination, across every split.

    Deliberately not per-split: frame_000001 in the source's train and in the
    destination's val are still the same name, and letting both exist would
    make the set impossible to talk about — and would collide the moment
    split_dataset.py reshuffles the two sides."""
    return {p.stem for p, _, _ in find_pairs(root)}


def tagged(stem, tag, n=0):
    """frame_7_r180 -> frame_7__ds2_r180, keeping the rotation marker last."""
    base, rot = (stem[:-len(ROT_SUFFIX)], ROT_SUFFIX) \
        if stem.endswith(ROT_SUFFIX) else (stem, "")
    return f"{base}__{tag}{'' if n == 0 else f'_{n}'}{rot}"


def max_class(label_path):
    top = -1
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if parts:
            try:
                top = max(top, int(parts[0]))
            except ValueError:
                pass
    return top


def main():
    ap = argparse.ArgumentParser(
        description="Merge a YOLO dataset folder into another, renaming collisions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--src", required=True, help="folder to merge FROM")
    ap.add_argument("--dst", default="dataset", help="folder to merge INTO")
    ap.add_argument("--tag", default=None,
                    help="what a renamed file is tagged with "
                         "(default: the source folder's name)")
    ap.add_argument("--into", choices=["keep", "train", "val"], default="keep",
                    help="keep: put train in train and val in val; otherwise "
                         "send every source frame to one side")
    ap.add_argument("--move", action="store_true",
                    help="move instead of copy — faster and uses no extra disk, "
                         "but empties the source")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would happen and write nothing")
    ap.add_argument("--force", action="store_true",
                    help="merge even if the class names differ. Only correct if "
                         "the INDICES mean the same thing in both")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    for p in (src, dst):
        if not p.is_dir():
            sys.exit(f"[merge] {p} is not a directory")
    if src.resolve() == dst.resolve():
        sys.exit("[merge] --src and --dst are the same folder")
    tag = args.tag or src.name.replace(" ", "_").strip("-_.") or "src"

    # ---- classes must agree before a single file moves -------------------
    src_names, src_from = read_names(src)
    dst_names, dst_from = read_names(dst)
    print(f"[merge] from {src.resolve()}\n[merge] into {dst.resolve()}")
    print(f"[merge] src classes: "
          f"{', '.join(f'{i}={n}' for i, n in enumerate(src_names or []))} "
          f"({src_from or 'none found'})")
    print(f"[merge] dst classes: "
          f"{', '.join(f'{i}={n}' for i, n in enumerate(dst_names or []))} "
          f"({dst_from or 'none found'})")
    if not src_names or not dst_names:
        sys.exit("[merge] one of the folders has no classes.txt or data.yaml — "
                 "cannot confirm the two use the same class indices")
    if src_names != dst_names and not args.force:
        sys.exit(f"[merge] class names differ:\n"
                 f"          src {src_names}\n          dst {dst_names}\n"
                 f"       Merging these would file one set's boxes under the "
                 f"other set's classes. Remap the source first (see "
                 f"drop_label_class.py --names), or pass --force if the indices "
                 f"really do mean the same thing.")

    pairs = find_pairs(src)
    if not pairs:
        sys.exit(f"[merge] no images found under {src}")

    # ---- plan the whole move before doing any of it ----------------------
    taken = taken_stems(dst)
    plan, renamed, no_label, bad_class = [], [], [], Counter()
    per_split = Counter()
    for image, label, split in pairs:
        if label is None:
            no_label.append(image)
        elif (top := max_class(label)) >= len(dst_names):
            bad_class[top] += 1

        stem = image.stem
        if stem in taken:
            n = 0
            while tagged(stem, tag, n) in taken:
                n += 1
            new_stem = tagged(stem, tag, n)
            renamed.append((stem, new_stem))
        else:
            new_stem = stem
        taken.add(new_stem)

        target = args.into if args.into != "keep" else split
        per_split[target or "flat"] += 1
        dst_image = (dst / "images" / target / (new_stem + image.suffix)
                     if target else dst / "images" / (new_stem + image.suffix))
        dst_label = (dst / "labels" / target / (new_stem + ".txt")
                     if target else dst / "labels" / (new_stem + ".txt"))
        plan.append((image, label, dst_image, dst_label))

    print(f"[merge] {len(plan)} frame(s): " + ", ".join(
        f"{k}={v}" for k, v in sorted(per_split.items())))
    print(f"[merge] {len(renamed)} name collision(s) -> renamed with '__{tag}'")
    for old, new in renamed[:5]:
        print(f"          {old} -> {new}")
    if len(renamed) > 5:
        print(f"          ... and {len(renamed) - 5} more")
    if no_label:
        print(f"[merge] WARNING: {len(no_label)} image(s) have no .txt "
              f"(e.g. {no_label[0].name}) — copied as background frames")
    if bad_class:
        # Indices past the destination's class list would train as a class that
        # does not exist there. Refusing is the only safe answer.
        sys.exit(f"[merge] source labels use class index "
                 f"{sorted(bad_class)} but the destination has only "
                 f"{len(dst_names)} class(es) — fix the source first")

    if args.dry_run:
        print("\n[merge] --dry-run: nothing written")
        return

    move = shutil.move if args.move else shutil.copy2
    done = 0
    for image, label, dst_image, dst_label in plan:
        dst_image.parent.mkdir(parents=True, exist_ok=True)
        move(str(image), str(dst_image))
        if label:
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            move(str(label), str(dst_label))
        done += 1
        if done % 200 == 0:
            print(f"\r[merge] {done}/{len(plan)}", end="", flush=True)
    print(f"\r[merge] {'moved' if args.move else 'copied'} {done} frame(s)")

    report = dst / "merge_report.txt"
    with open(report, "a") as f:
        f.write(f"# {src.resolve()} -> {dst.resolve()}  "
                f"({'move' if args.move else 'copy'}, tag {tag})\n")
        f.write(f"frames: {done}, renamed: {len(renamed)}\n")
        for old, new in renamed:
            f.write(f"  {old} -> {new}\n")
    print(f"[merge] rename log appended to {report}")

    # Read the destination back rather than trusting the plan: a merge that
    # half-worked is worth knowing about before a 12-hour training run.
    after = find_pairs(dst)
    missing = [p.name for p, l, _ in after if l is None]
    counts = Counter(s or "flat" for _, _, s in after)
    print(f"[merge] destination now: {len(after)} frame(s) — " + ", ".join(
        f"{k}={v}" for k, v in sorted(counts.items())))
    if missing:
        print(f"[merge] {len(missing)} frame(s) in the destination have no label "
              f"(e.g. {missing[:3]})")
    print(f"[merge] check it: python3 view_dataset.py --dataset {dst} --stats")
    print(f"[merge] re-split honestly: python3 split_dataset.py --ds {dst}")


if __name__ == "__main__":
    main()
