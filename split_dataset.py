#!/usr/bin/env python3
"""
Flat capture folder -> trainable train/val structure
====================================================
Takes what capture_dataset.py / label_frames.py write —

    <ds>/images/*.jpg   <ds>/labels/*.txt   <ds>/annotations.json

— and turns it into the layout Ultralytics expects:

    <ds>/images/train/*.jpg   <ds>/labels/train/*.txt
    <ds>/images/val/*.jpg     <ds>/labels/val/*.txt
    <ds>/data.yaml            path/train/val rewritten to match
    <ds>/annotations_train.json, annotations_val.json
    <ds>/split_report.txt

    python3 split_dataset.py                    # in place, on ./dataset
    python3 split_dataset.py --ds captured --val-frac 0.2
    python3 split_dataset.py --dry-run          # say what it would do

THE SPLIT IS BY SCENE, NOT BY FRAME, and that is the whole point of this file.

Frames captured by holding S at a machine are not independent samples. The web
sits still between presses, so a run of frames can be the same view photographed
six times over. Split those at random and near-identical frames land on both
sides: the model trains on a picture and is then validated on a copy of it, and
mAP comes back high because the test is rigged. Nothing in the numbers tells you
this happened — that is what makes it worth defending against.

So frames are grouped first. Each is reduced to a small normalised signature, and
any two above --group-threshold cosine similarity are joined into one scene
(transitively, so a slow drift across a run stays one group). Whole GROUPS are
then dealt to train or val, never split. Val ends up near --val-frac but rarely
exactly on it, because a group is indivisible — that is the cost of an honest
split and it is a small one.

The report prints how many distinct scenes were found. Read it: if 500 frames
collapse into 40 scenes, the dataset is 40 samples wearing 500 filenames, and no
split can fix that — only capturing more variety can.

FILES ARE MOVED, not copied, so this does not need a second copy of the images
on disk. It re-runs safely: an already-split folder is gathered back up and
re-split, so changing --val-frac is one command and not a manual cleanup.
"""

import argparse
import glob
import json
import os
import random
import shutil
import sys

import cv2
import numpy as np

CLASSES = ["label", "qr_code", "logo"]
IMG_EXT = (".jpg", ".jpeg", ".png")


def gather(ds):
    """Every image in the dataset, flat or already split, with its label file.

    Looks in images/, images/train/ and images/val/ alike, which is what makes
    the script re-runnable on a folder it has already reorganised.
    """
    pairs = []
    for sub in ("", "train", "val"):
        for p in sorted(glob.glob(os.path.join(ds, "images", sub, "*"))):
            if not p.lower().endswith(IMG_EXT) or os.path.isdir(p):
                continue
            stem = os.path.splitext(os.path.basename(p))[0]
            lab = os.path.join(ds, "labels", sub, stem + ".txt")
            pairs.append((p, lab if os.path.exists(lab) else None, stem))
    return pairs


def signatures(paths, size=(48, 64)):
    """A small contrast-normalised thumbnail per frame, unit length.

    Normalising out mean and standard deviation matters: two captures of the
    same view under a flickering lamp differ in brightness and in nothing else,
    and without this they would score as different scenes and be allowed onto
    opposite sides of the split.
    """
    sigs = []
    for i, p in enumerate(paths):
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            sigs.append(None)
            continue
        s = cv2.resize(im, size, interpolation=cv2.INTER_AREA).astype(np.float32)
        s = (s - s.mean()) / (s.std() + 1e-6)
        sigs.append((s / (np.linalg.norm(s) + 1e-6)).ravel())
        if (i + 1) % 100 == 0:
            print(f"  hashed {i + 1}/{len(paths)}", flush=True)
    return sigs


def scene_groups(sigs, threshold):
    """Union-find over the similarity graph: group id per frame."""
    n = len(sigs)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    ok = [i for i, s in enumerate(sigs) if s is not None]
    if ok:
        M = np.stack([sigs[i] for i in ok])
        C = M @ M.T
        ii, jj = np.where(np.triu(C > threshold, 1))
        for a, b in zip(ii, jj):
            ra, rb = find(ok[a]), find(ok[b])
            if ra != rb:
                parent[ra] = rb
    return [find(i) for i in range(n)]


def deal(groups, val_frac, seed):
    """Assign whole groups to val until it reaches the target share.

    Largest-first would pack val with the few biggest scenes and leave train
    every small one, so groups are shuffled and taken in that order — the split
    is then representative of the range of scenes, not of their sizes.
    """
    members = {}
    for idx, g in enumerate(groups):
        members.setdefault(g, []).append(idx)
    keys = list(members)
    random.Random(seed).shuffle(keys)

    target = int(round(len(groups) * val_frac))
    val, n = set(), 0
    for k in keys:
        if n >= target:
            break
        val.add(k)
        n += len(members[k])
    return ["val" if groups[i] in val else "train" for i in range(len(groups))], members


def split_coco(ds, assign, stems):
    """Split the COCO file alongside the images, if there is one."""
    src = os.path.join(ds, "annotations.json")
    if not os.path.exists(src):
        return None
    try:
        with open(src) as f:
            coco = json.load(f)
    except ValueError:
        print("  WARNING: annotations.json is unreadable — leaving it alone")
        return None

    want = {s: sp for s, sp in zip(stems, assign)}
    by_split = {}
    for sp in ("train", "val"):
        by_split[sp] = {"info": coco.get("info", {}), "licenses": [],
                        "images": [], "annotations": [],
                        "categories": coco.get("categories", [])}
    keep_id = {}
    for im in coco.get("images", []):
        stem = os.path.splitext(im["file_name"])[0]
        sp = want.get(stem)
        if sp is None:
            continue
        keep_id[im["id"]] = sp
        by_split[sp]["images"].append(im)
    for a in coco.get("annotations", []):
        sp = keep_id.get(a["image_id"])
        if sp is not None:
            by_split[sp]["annotations"].append(a)
    for sp in ("train", "val"):
        with open(os.path.join(ds, f"annotations_{sp}.json"), "w") as f:
            json.dump(by_split[sp], f)
    return {sp: (len(by_split[sp]["images"]), len(by_split[sp]["annotations"]))
            for sp in ("train", "val")}


def write_yaml(ds, names):
    """data.yaml with an ABSOLUTE path.

    Ultralytics resolves a relative `path:` against its own datasets directory,
    not against the yaml's own folder, so a relative one silently sends the
    trainer somewhere else entirely. The flip side is that renaming or moving
    this folder invalidates the file — re-run this script and it is rewritten.
    """
    with open(os.path.join(ds, "data.yaml"), "w") as f:
        f.write("# written by split_dataset.py\n")
        f.write(f"path: {os.path.abspath(ds)}\n")
        f.write("train: images/train\nval: images/val\n")
        f.write(f"nc: {len(names)}\nnames:\n")
        for i, n in enumerate(names):
            f.write(f"  {i}: {n}\n")


def main():
    ap = argparse.ArgumentParser(
        description="Reorganise a flat capture folder into train/val for YOLO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--ds", default="dataset", help="the dataset folder")
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="target share of frames in val")
    ap.add_argument("--group-threshold", type=float, default=0.99,
                    help="cosine similarity above which two frames are the same "
                         "scene and may not be split across train and val. "
                         "Lower groups more aggressively")
    ap.add_argument("--no-group", action="store_true",
                    help="plain random split. Only correct if every frame is an "
                         "independent view — check the scene count first")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="report the split without moving anything")
    args = ap.parse_args()

    pairs = gather(args.ds)
    if not pairs:
        print(f"No images under {args.ds}/images/")
        return
    missing = [s for _, l, s in pairs if l is None]
    if missing:
        print(f"WARNING: {len(missing)} image(s) have no label file, e.g. "
              f"{missing[:3]} — they will be moved but cannot train")

    paths = [p for p, _, _ in pairs]
    stems = [s for _, _, s in pairs]
    print(f"{len(pairs)} frame(s) in {args.ds}")

    if args.no_group:
        groups = list(range(len(pairs)))
        print("Grouping disabled — plain random split")
    else:
        print("Hashing frames to find distinct scenes...")
        groups = scene_groups(signatures(paths), args.group_threshold)

    assign, members = deal(groups, args.val_frac, args.seed)
    n_val = sum(1 for a in assign if a == "val")
    n_scenes = len(members)
    sizes = sorted((len(v) for v in members.values()), reverse=True)

    lines = [
        f"frames          : {len(pairs)}",
        f"distinct scenes : {n_scenes}"
        + ("" if args.no_group else f"  (threshold {args.group_threshold})"),
        f"frames/scene    : max {sizes[0]}, median {sizes[len(sizes) // 2]}",
        f"train / val     : {len(pairs) - n_val} / {n_val} frames "
        f"({n_val / len(pairs) * 100:.1f}% val, asked for {args.val_frac * 100:.0f}%)",
    ]
    val_scenes = len({groups[i] for i, a in enumerate(assign) if a == "val"})
    lines.append(f"scenes in val   : {val_scenes} / {n_scenes}")
    print("\n" + "\n".join("  " + l for l in lines))

    if n_scenes < len(pairs) * 0.5:
        print(f"\n  NOTE: {len(pairs)} frames collapse into {n_scenes} scenes. The "
              f"set is {n_scenes} independent sample(s) wearing {len(pairs)} "
              f"filenames.\n        Splitting it honestly cannot add variety — "
              f"capturing with the web moved between presses can.")

    if args.dry_run:
        print("\n--dry-run: nothing moved")
        return

    for sp in ("train", "val"):
        os.makedirs(os.path.join(args.ds, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(args.ds, "labels", sp), exist_ok=True)

    moved = 0
    for (img, lab, stem), sp in zip(pairs, assign):
        dst_i = os.path.join(args.ds, "images", sp, os.path.basename(img))
        if os.path.abspath(img) != os.path.abspath(dst_i):
            shutil.move(img, dst_i)
        if lab:
            dst_l = os.path.join(args.ds, "labels", sp, stem + ".txt")
            if os.path.abspath(lab) != os.path.abspath(dst_l):
                shutil.move(lab, dst_l)
        moved += 1

    # Any now-empty flat folders left behind by a re-run.
    for sub in ("images", "labels"):
        d = os.path.join(args.ds, sub)
        for e in os.listdir(d):
            p = os.path.join(d, e)
            if os.path.isdir(p) and e not in ("train", "val") and not os.listdir(p):
                os.rmdir(p)

    names = CLASSES
    yml = os.path.join(args.ds, "data.yaml")
    if os.path.exists(yml):                    # keep whatever names were in use
        got = []
        for line in open(yml):
            line = line.strip()
            if line[:1].isdigit() and ":" in line:
                got.append(line.split(":", 1)[1].strip())
        if got:
            names = got
    write_yaml(args.ds, names)
    coco = split_coco(args.ds, assign, stems)

    with open(os.path.join(args.ds, "split_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nmoved {moved} frame(s)")
    print(f"  {args.ds}/images/train  {args.ds}/images/val")
    print(f"  {args.ds}/labels/train  {args.ds}/labels/val")
    if coco:
        for sp in ("train", "val"):
            print(f"  {args.ds}/annotations_{sp}.json  "
                  f"({coco[sp][0]} images, {coco[sp][1]} boxes)")
    print(f"  {args.ds}/data.yaml   (path rewritten to {os.path.abspath(args.ds)})")
    print(f"\n  python3 train.py     # data='{args.ds}/data.yaml'")


if __name__ == "__main__":
    main()
