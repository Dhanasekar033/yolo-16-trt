#!/usr/bin/env python3
"""
Drop one class out of a dataset and close the gap it leaves behind.

Built for the move this dataset is making: it was captured as

    0 label     1 qr_code     2 logo

and the `label` box — the whole card, the thing the other two sit inside — is
no longer a class the model should spend capacity on. Removing it is two edits,
not one, and the second is the one that bites:

    1. every `0 ...` line comes out of every .txt
    2. every REMAINING index shifts down, 1 -> 0 and 2 -> 1

Do only the first and the files still say 1 and 2 while classes.txt says there
are two classes numbered 0 and 1. Ultralytics does not error on that — it reads
index 2 against nc=2 and the run dies deep in the loss, or worse, an index that
still resolves trains qr_code boxes under the name `logo`. Nothing in the
metrics would tell you. So the remap is not optional tidying, it is the point.

The images are never touched. Labels are rewritten IN PLACE after the originals
are copied to <ds>/labels_backup/, and the script refuses to run a second time
over an existing backup — a re-run would otherwise back up the already-stripped
labels on top of the only copy of the originals. classes.txt, data.yaml, and any
COCO annotations*.json are rewritten to match, so the whole folder moves to the
new numbering together or not at all.

    python3 drop_label_class.py --dry-run       # count what would change
    python3 drop_label_class.py                 # do it, on ./dataset
    python3 drop_label_class.py --ds captured
    python3 drop_label_class.py --drop logo     # some other class
    python3 drop_label_class.py --restore       # put the backup back
"""

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_NAMES = ["label", "qr_code", "logo"]


# ---------------------------------------------------------------- reading

def read_yaml(path):
    """The yaml as a dict, PyYAML if installed and a small reader if not.

    Only used to LOOK at names; the rewrite below works on the file's own text
    so that its comments and its `path:` survive untouched."""
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        pass
    except Exception as exc:
        print(f"[drop] {path.name} unreadable ({exc}) — ignoring it")
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
    """(names, where they came from). classes.txt first here, deliberately.

    view_dataset.py prefers data.yaml because that is what the trainer reads,
    but this script rewrites BOTH from one list, and classes.txt is the file
    make_dataset.py wrote the indices against — so it is the one to trust about
    what the numbers in the .txt files currently mean."""
    path = root / "classes.txt"
    if path.exists():
        names = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
        if names:
            return names, path
    yml = find_yaml(root)
    if yml:
        names = read_yaml(yml).get("names")
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names, key=lambda k: int(k))]
        if isinstance(names, list) and names:
            return [str(n) for n in names], yml
    return list(DEFAULT_NAMES), None


def find_label_files(root):
    """Every YOLO .txt in the dataset, whatever shape the folder is in.

    Matches the layouts view_dataset.py reads — flat, split into train/val, the
    train/labels form, and .txt sitting beside the images — because a dataset
    that has been through split_dataset.py is no less in need of this."""
    found, seen = [], set()

    def add(path):
        key = path.resolve()
        if path.is_file() and key not in seen:
            seen.add(key)
            found.append(path)

    labels_dir = root / "labels"
    if labels_dir.is_dir():
        for p in sorted(labels_dir.rglob("*.txt")):
            add(p)
    for sub in sorted(d for d in root.iterdir() if d.is_dir()):
        if sub.name in ("labels", "labels_backup", "preview"):
            continue
        if (sub / "labels").is_dir():
            for p in sorted((sub / "labels").rglob("*.txt")):
                add(p)
    # Loose folders keep the .txt next to the .jpg. classes.txt lives there too
    # and is not a label file, so images are matched first and the txt derived.
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMG_EXT:
            beside = p.with_suffix(".txt")
            if beside.exists():
                add(beside)
    return found


# ---------------------------------------------------------------- rewriting

def remap_of(names, drop_index):
    """old index -> new index, with the dropped one mapped to None."""
    mapping, nxt = {}, 0
    for i in range(len(names)):
        if i == drop_index:
            mapping[i] = None
        else:
            mapping[i] = nxt
            nxt += 1
    return mapping


def rewrite_label_text(text, mapping):
    """(new text, kept, dropped, unknown-indices seen).

    Lines are rebuilt from their parts rather than string-replaced: a naive
    replace of a leading "1" would also hit the coordinates."""
    out, kept, dropped, unknown = [], 0, 0, Counter()
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            cls = int(parts[0])
        except ValueError:
            unknown[parts[0]] += 1
            continue
        new = mapping.get(cls, "missing")
        if new == "missing":
            # An index no class name covers. Dropping it silently would hide a
            # real problem in the labels, so it is counted and reported.
            unknown[str(cls)] += 1
            continue
        if new is None:
            dropped += 1
            continue
        out.append(" ".join([str(new)] + parts[1:]))
        kept += 1
    return ("\n".join(out) + "\n" if out else ""), kept, dropped, unknown


def rewrite_classes_txt(path, names):
    path.write_text("\n".join(names) + "\n")


def rewrite_yaml(path, names, set_path=None):
    """Replace nc: and the names: block, leave every other line alone.

    The `path:` in this file is absolute and points at wherever the dataset was
    when split_dataset.py last ran; regenerating the yaml from scratch here
    would either lose that or invent a new one, and neither is this script's
    business. So the file is edited, not rewritten."""
    lines = path.read_text().splitlines()
    out, in_names, wrote_names, wrote_nc, wrote_path = [], False, False, False, False
    for raw in lines:
        stripped = raw.strip()
        if in_names:
            if raw[:1].isspace() or stripped.startswith("-"):
                continue          # an old name line, replaced below
            in_names = False
        if stripped.split("#", 1)[0].strip().rstrip(":") == "names":
            in_names = True
            out.append("names:")
            out.extend(f"  {i}: {n}" for i, n in enumerate(names))
            wrote_names = True
            continue
        if stripped.startswith("nc:"):
            out.append(f"nc: {len(names)}")
            wrote_nc = True
            continue
        if set_path and stripped.startswith("path:"):
            out.append(f"path: {set_path}")
            wrote_path = True
            continue
        out.append(raw)
    if set_path and not wrote_path:
        # Ahead of train:/val:, which are read relative to it.
        at = next((i for i, l in enumerate(out)
                   if l.strip().startswith(("train:", "val:"))), len(out))
        out.insert(at, f"path: {set_path}")
    if not wrote_nc:
        out.append(f"nc: {len(names)}")
    if not wrote_names:
        out.append("names:")
        out.extend(f"  {i}: {n}" for i, n in enumerate(names))
    path.write_text("\n".join(out) + "\n")


def rewrite_coco(path, drop_index, names):
    """(images, kept boxes, dropped boxes) or None if there is nothing to do.

    COCO category ids are 1-based while the .txt files are 0-based, so the same
    class is `drop_index` in one file and `drop_index + 1` in the other. Getting
    that off by one would delete the wrong class from half the dataset."""
    try:
        with open(path) as f:
            doc = json.load(f)
    except ValueError:
        print(f"[drop] {path.name} is not readable JSON — left alone")
        return None

    cats = sorted(doc.get("categories", []), key=lambda c: c["id"])
    if not cats:
        return None
    old_ids = [c["id"] for c in cats]
    drop_id = old_ids[drop_index] if drop_index < len(old_ids) else None
    remap = {}
    nxt = 1
    for cid in old_ids:
        if cid == drop_id:
            continue
        remap[cid] = nxt
        nxt += 1

    kept_anns, dropped = [], 0
    for a in doc.get("annotations", []):
        new = remap.get(a["category_id"])
        if new is None:
            dropped += 1
            continue
        a["category_id"] = new
        kept_anns.append(a)
    doc["annotations"] = kept_anns
    doc["categories"] = [{"id": i + 1, "name": n} for i, n in enumerate(names)]
    with open(path, "w") as f:
        json.dump(doc, f)
    return len(doc.get("images", [])), len(kept_anns), dropped


# ---------------------------------------------------------------- backup

def backup_targets(root, label_files):
    """The files this script is about to change, so they can be copied first."""
    extra = [p for p in [root / "classes.txt", find_yaml(root)] if p and p.exists()]
    extra += sorted(root.glob("annotations*.json"))
    return label_files + extra


def make_backup(root, backup_dir, files, force):
    if backup_dir.exists():
        if not force:
            sys.exit(
                f"[drop] {backup_dir} already exists — this dataset looks like it "
                f"has been stripped once already.\n"
                f"       Re-running would overwrite the only copy of the original "
                f"labels with the stripped ones.\n"
                f"       Use --restore to put the originals back, or --force with "
                f"a different --backup-dir if you know what you are doing.")
        print(f"[drop] --force: overwriting {backup_dir}")
        shutil.rmtree(backup_dir)
    for src in files:
        dst = backup_dir / src.relative_to(root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return len(files)


def restore(root, backup_dir):
    if not backup_dir.is_dir():
        sys.exit(f"[drop] no backup at {backup_dir}")
    n = 0
    for src in sorted(backup_dir.rglob("*")):
        if src.is_file():
            dst = root / src.relative_to(backup_dir)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            n += 1
    print(f"[drop] restored {n} file(s) from {backup_dir} — the backup is kept; "
          f"delete it yourself once you are happy")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description="Remove one class from a YOLO dataset and reindex the rest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--ds", default="dataset", help="the dataset folder")
    ap.add_argument("--drop", default="label",
                    help="class NAME to remove, as it appears in classes.txt")
    ap.add_argument("--names", default=None,
                    help="comma-separated names for the classes that REMAIN, in "
                         "index order, if they should also be renamed "
                         "(e.g. --names code,artifact). Renaming only rewrites "
                         "classes.txt and data.yaml — the indices in the .txt "
                         "files are decided by --drop, not by this")
    ap.add_argument("--set-path", action="store_true",
                    help="also refresh the ABSOLUTE `path:` in data.yaml to "
                         "where the folder actually is. Ultralytics resolves a "
                         "relative train:/val: against its own datasets dir, "
                         "not against the yaml, so a yaml without this one "
                         "trains on whatever it finds there instead")
    ap.add_argument("--backup-dir", default=None,
                    help="where the originals are copied (default <ds>/labels_backup)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change and write nothing")
    ap.add_argument("--restore", action="store_true",
                    help="copy the backup back over the dataset and exit")
    ap.add_argument("--force", action="store_true",
                    help="allow a second run to replace an existing backup")
    args = ap.parse_args()

    root = Path(args.ds)
    if not root.is_dir():
        sys.exit(f"[drop] {root} is not a directory")
    backup_dir = Path(args.backup_dir) if args.backup_dir else root / "labels_backup"

    if args.restore:
        restore(root, backup_dir)
        return

    names, src = read_names(root)
    print(f"[drop] {root.resolve()}")
    print(f"[drop] classes now: " + ", ".join(f"{i}={n}" for i, n in enumerate(names))
          + (f"  (from {src.name})" if src else "  (assumed — no classes.txt)"))

    if args.drop not in names:
        sys.exit(f"[drop] no class named '{args.drop}' here — nothing to remove. "
                 f"If this dataset has already been stripped, that is the "
                 f"expected answer.")
    if len(names) < 2:
        sys.exit(f"[drop] '{args.drop}' is the only class — removing it would "
                 f"leave a dataset that cannot train")

    drop_index = names.index(args.drop)
    mapping = remap_of(names, drop_index)
    kept_names = [n for i, n in enumerate(names) if i != drop_index]
    new_names = kept_names
    if args.names:
        wanted = [n.strip() for n in args.names.split(",") if n.strip()]
        if len(wanted) != len(kept_names):
            sys.exit(f"[drop] --names gives {len(wanted)} name(s) but "
                     f"{len(kept_names)} class(es) remain after dropping "
                     f"'{args.drop}': {', '.join(kept_names)}")
        # A rename is cosmetic to the .txt files and load-bearing everywhere
        # else: run.py and label-inspector look their classes up BY NAME, so a
        # set renamed here has to be renamed there too or they find nothing.
        print("[drop] rename: " + ", ".join(
            f"{o} -> {w}" for o, w in zip(kept_names, wanted) if o != w) or "none")
        new_names = wanted
    print("[drop] remap: " + ", ".join(
        f"{i} {names[i]} -> " + ("removed" if mapping[i] is None else f"{mapping[i]}")
        for i in range(len(names))))
    print(f"[drop] classes after: " + ", ".join(
        f"{i}={n}" for i, n in enumerate(new_names)))

    label_files = find_label_files(root)
    if not label_files:
        sys.exit(f"[drop] no .txt label files found under {root}")

    # Pass one reads and transforms everything in memory. Nothing is written
    # until every file has parsed, so a bad file cannot leave the dataset half
    # converted — half a dataset on the new numbering and half on the old is the
    # one state that is worse than not starting.
    planned, kept, dropped, unknown = [], 0, 0, Counter()
    emptied = 0
    for path in label_files:
        text, k, d, u = rewrite_label_text(path.read_text(), mapping)
        planned.append((path, text))
        kept += k
        dropped += d
        unknown.update(u)
        if k == 0:
            emptied += 1

    coco_files = sorted(root.glob("annotations*.json"))
    print(f"[drop] {len(label_files)} label file(s): "
          f"{dropped} '{args.drop}' box(es) removed, {kept} kept")
    if emptied:
        print(f"[drop] {emptied} file(s) end up with no boxes at all — YOLO reads "
              f"those as background images, which is fine, but check that is "
              f"what you meant")
    if unknown:
        print(f"[drop] WARNING: {sum(unknown.values())} line(s) carry a class no "
              f"name covers ({', '.join(sorted(unknown))}); they are dropped too")
    if coco_files:
        print(f"[drop] COCO to rewrite: " + ", ".join(p.name for p in coco_files))

    if args.dry_run:
        print("\n[drop] --dry-run: nothing written")
        return

    n = make_backup(root, backup_dir, backup_targets(root, label_files), args.force)
    print(f"[drop] backed up {n} file(s) to {backup_dir}")

    for path, text in planned:
        path.write_text(text)

    rewrite_classes_txt(root / "classes.txt", new_names)
    print(f"[drop] wrote {root / 'classes.txt'}")
    yml = find_yaml(root)
    if yml:
        abs_path = str(root.resolve()) if args.set_path else None
        rewrite_yaml(yml, new_names, abs_path)
        print(f"[drop] wrote {yml}  (nc: {len(new_names)}"
              + (f", path: {abs_path}" if abs_path else "") + ")")
    for path in coco_files:
        res = rewrite_coco(path, drop_index, new_names)
        if res:
            print(f"[drop] wrote {path.name}  "
                  f"({res[0]} images, {res[1]} boxes kept, {res[2]} removed)")

    # Read the labels back and check them against the new class count, because
    # the failure this whole script exists to prevent is a stale index that no
    # trainer will complain about until the model is quietly wrong.
    seen = Counter()
    for path in label_files:
        for line in path.read_text().splitlines():
            parts = line.split()
            if parts:
                seen[int(parts[0])] += 1
    bad = sorted(c for c in seen if c < 0 or c >= len(new_names))
    print("\n[drop] verify: " + ", ".join(
        f"{i}={new_names[i]}: {seen.get(i, 0)}" for i in range(len(new_names))))
    if bad:
        sys.exit(f"[drop] FAILED: index {bad} still present after the remap — "
                 f"restore with --restore and do not train on this")
    print(f"[drop] done. Originals in {backup_dir} (--restore puts them back)")
    print(f"[drop] check it: python3 view_dataset.py --dataset {root} --stats")


if __name__ == "__main__":
    main()
