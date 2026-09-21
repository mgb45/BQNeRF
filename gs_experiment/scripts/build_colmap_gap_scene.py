"""Carve an angular coverage gap out of a COLMAP scene, so the epistemic
regime can be tested on real captures under a third party's own pipeline.

Why this is not just "delete some images". 3DGS splits train/test
POSITIONALLY -- `sorted(image_names)[::8]` -- and forces `llffhold = 8`
whenever "360" appears in the source path. Their loader does support an
explicit `sparse/0/test.txt`, but only when `llffhold` is falsy, and that is
not exposed on the command line. Rather than patch their code (which would
weaken the claim that their pipeline is theirs), this script controls the
split the only other way available: by choosing image NAMES, since the split
is a function of sorted order alone.

Two hold-out structures, both producing a source directory their unmodified
`train.py` consumes:

* **trajectory** (default) -- train on a contiguous prefix of the capture
  sequence and evaluate on the unvisited remainder. This is the SLAM case:
  an agent has mapped where it has been and is asked what it can trust about
  where it has not. COLMAP names in these captures follow acquisition order,
  so sorting by name recovers the trajectory.
* **cone** -- withhold an angular wedge around the densest viewing direction.
  A cleaner geometric ablation, but a less plausible one: nothing about a
  real capture removes a symmetric wedge.

Construction:
  * cameras inside a cone of `half_width_deg` around a chosen direction are
    withheld from TRAINING entirely;
  * as many of them as the 1-in-8 split allows become the TEST set, so
    evaluation happens exactly where the training views do not reach;
  * the remaining in-cone cameras are dropped, since leaving them in would
    either leak into training or displace the split;
  * everything is rewritten to a new COLMAP TEXT model with names `g{pos}`
    chosen so that `sorted(names)[::8]` selects precisely the intended test
    set. `points3D.bin` is symlinked, since the point cloud does not depend
    on which images are kept.

The result is a source directory their unmodified `train.py` consumes, where
held-out views sit inside a real coverage hole.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party" / "GS-U"))

import numpy as np

from scene.colmap_loader import qvec2rotmat, read_extrinsics_binary, read_intrinsics_binary


def camera_centres(extrinsics):
    ids = sorted(extrinsics, key=lambda k: extrinsics[k].name)
    centres, names = [], []
    for k in ids:
        im = extrinsics[k]
        R = qvec2rotmat(im.qvec)
        centres.append(-R.T @ im.tvec)
        names.append(im.name)
    return ids, np.array(centres), names


def choose_gap_direction(centres):
    """The densest viewing direction: withholding a cone there removes the
    most coverage, which is the hardest and most honest place to test."""
    c = centres - centres.mean(axis=0)
    d = c / np.linalg.norm(c, axis=1, keepdims=True).clip(1e-12)
    counts = (d @ d.T > np.cos(np.radians(35.0))).sum(axis=1)
    return d[int(np.argmax(counts))]


def plan_trajectory_split(n, train_fraction=0.7):
    """Train on the first `train_fraction` of the capture sequence; evaluate
    on the unvisited tail. Test views are taken from the FAR end of the tail,
    i.e. the part of the trajectory the agent got furthest from ever seeing."""
    n_train = int(round(n * train_fraction))
    train = np.arange(n_train)
    tail = np.arange(n_train, n)
    n_test = min(int(np.ceil(n_train / 7.0)), len(tail))
    test = tail[len(tail) - n_test:]                 # furthest into the unvisited region
    dropped = np.setdiff1d(tail, test)
    return np.sort(test), np.sort(train), np.sort(dropped)


def plan_split(centres, half_width_deg, gap_dir):
    c = centres - centres.mean(axis=0)
    d = c / np.linalg.norm(c, axis=1, keepdims=True).clip(1e-12)
    ang = np.degrees(np.arccos(np.clip(d @ gap_dir, -1, 1)))
    in_cone = np.where(ang <= half_width_deg)[0]
    outside = np.where(ang > half_width_deg)[0]
    # |test| must equal ceil((|test| + |train|) / 8) for the 1-in-8 split to
    # land exactly on the test set: |test| >= |train| / 7.
    n_test = int(np.ceil(len(outside) / 7.0))
    n_test = min(n_test, len(in_cone))
    test = in_cone[np.argsort(ang[in_cone])[:n_test]]          # most central first
    dropped = np.setdiff1d(in_cone, test)
    return np.sort(test), np.sort(outside), np.sort(dropped), ang


def write_model(out_dir, src_dir, image_dirs, extrinsics, intrinsics, ids, names,
                test_idx, train_idx):
    sparse = out_dir / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    order = []                                   # (position, original index)
    t, r = list(test_idx), list(train_idx)
    pos = 0
    while t or r:
        order.append((pos, t.pop(0) if (pos % 8 == 0 and t) else r.pop(0)))
        pos += 1
    assert all(o % 8 != 0 or i in set(test_idx) for o, i in order), "split would not land on test set"

    ext = Path(names[0]).suffix
    rename = {i: f"g{o:05d}{ext}" for o, i in order}
    with open(sparse / "images.txt", "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n#\n")
        for o, i in order:
            im = extrinsics[ids[i]]
            q, tv = im.qvec, im.tvec
            f.write(f"{o + 1} {q[0]} {q[1]} {q[2]} {q[3]} {tv[0]} {tv[1]} {tv[2]} "
                    f"{im.camera_id} {rename[i]}\n")
            f.write(" ".join(f"{x:.2f} {y:.2f} {pid}"
                             for (x, y), pid in zip(im.xys, im.point3D_ids)) + "\n")
    with open(sparse / "cameras.txt", "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n#\n")
        for cid, cam in intrinsics.items():
            f.write(f"{cid} {cam.model} {cam.width} {cam.height} "
                    + " ".join(str(p) for p in cam.params) + "\n")
    src_pts = src_dir / "sparse" / "0" / "points3D.bin"
    dst_pts = sparse / "points3D.bin"
    if not dst_pts.exists():
        os.symlink(os.path.abspath(src_pts), dst_pts)

    for img_dir in image_dirs:
        (out_dir / img_dir).mkdir(parents=True, exist_ok=True)
        for o, i in order:
            link = out_dir / img_dir / rename[i]
            target = src_dir / img_dir / names[i]
            if not target.exists():                       # folders may use .JPG/.jpg
                cands = list((src_dir / img_dir).glob(Path(names[i]).stem + ".*"))
                if not cands:
                    raise FileNotFoundError(target)
                target = cands[0]
            if not link.exists():
                os.symlink(os.path.abspath(target), link)
    return rename, order


def angular_isolation(centres, test_idx, train_idx):
    """How far, in degrees, each held-out view is from the NEAREST training
    view -- the honest measure of whether a hold-out is really a coverage
    hole, whichever way it was constructed."""
    c = centres - centres.mean(axis=0)
    d = c / np.linalg.norm(c, axis=1, keepdims=True).clip(1e-12)
    cos = d[test_idx] @ d[train_idx].T
    return np.degrees(np.arccos(np.clip(cos.max(axis=1), -1, 1)))


def run(scene, half_width_deg=45.0, image_dirs=("images_2", "images_4"), root=None,
        out_root=None, split_mode="trajectory", train_fraction=0.7):
    root = Path(root or "gs_experiment/local_runs/mipnerf360_raw")
    out_root = Path(out_root or "gs_experiment/local_runs/gapscenes")
    src = root / scene
    out = out_root / f"{scene}_{split_mode}"           # no "360" in the path, so llffhold is not forced
    extr = read_extrinsics_binary(src / "sparse" / "0" / "images.bin")
    intr = read_intrinsics_binary(src / "sparse" / "0" / "cameras.bin")
    ids, centres, names = camera_centres(extr)
    if split_mode == "trajectory":
        test_idx, train_idx, dropped = plan_trajectory_split(len(names), train_fraction)
        desc = f"trajectory prefix {train_fraction:.0%}"
    elif split_mode == "cone":
        test_idx, train_idx, dropped, _ = plan_split(
            centres, half_width_deg, choose_gap_direction(centres))
        desc = f"{half_width_deg:.0f} deg cone"
    else:
        raise ValueError(f"unknown split_mode {split_mode!r}")
    iso = angular_isolation(centres, test_idx, train_idx)
    if out.exists():
        shutil.rmtree(out)
    have = [d for d in image_dirs if (src / d).is_dir()]
    rename, order = write_model(out, src, have, extr, intr, ids, names, test_idx, train_idx)
    # Per-test-view isolation, in the order their renderer will emit
    # (00000.npy, 00001.npy, ...) -- i.e. test cameras sorted by their new
    # name, which is positions 0, 8, 16, ... Recording it here avoids having
    # to reconstruct the mapping downstream, where an off-by-one would be
    # invisible.
    pos_of = {i: o for o, i in order}
    test_sorted = sorted(test_idx, key=lambda i: pos_of[i])
    iso_sorted = angular_isolation(centres, np.array(test_sorted), train_idx)
    json.dump({"scene": scene, "split_mode": split_mode, "train_fraction": train_fraction,
               "half_width_deg": half_width_deg,
               "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
               "n_dropped": int(len(dropped)),
               "test_view_order": [rename[i] for i in test_sorted],
               "test_original_names": [names[i] for i in test_sorted],
               "test_isolation_deg": [float(x) for x in iso_sorted]},
              open(out / "gap_manifest.json", "w"), indent=1)
    print(f"{scene}: {len(names)} cams -> {len(train_idx)} train, {len(test_idx)} test, "
          f"{len(dropped)} dropped  [{desc}]")
    print(f"  held-out isolation (deg to NEAREST training view): "
          f"median {np.median(iso):.1f}, min {iso.min():.1f}, max {iso.max():.1f}")
    print(f"  wrote {out}  (image dirs: {', '.join(have)})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--half_width_deg", type=float, default=45.0)
    ap.add_argument("--split_mode", default="trajectory", choices=["trajectory", "cone"])
    ap.add_argument("--train_fraction", type=float, default=0.7)
    # Tanks & Temples and Deep Blending live outside mipnerf360_raw and ship a
    # single full-resolution `images/`, so both defaults have to be reachable
    # from the command line to cover the whole benchmark.
    ap.add_argument("--root", default=None)
    ap.add_argument("--image_dirs", default="images_2,images_4")
    a = ap.parse_args()
    for s in a.scenes:
        run(s, a.half_width_deg, image_dirs=tuple(a.image_dirs.split(",")),
            root=a.root, split_mode=a.split_mode, train_fraction=a.train_fraction)
