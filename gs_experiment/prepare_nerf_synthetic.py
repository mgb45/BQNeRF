"""Prepare a standard NeRF-Synthetic (Blender) scene -- e.g. the classic
"lego" scan -- for this project's pipeline, and build "wide" (full
training-view coverage) and "narrow" (a clustered angular subset of the
same real views) conditions from it.

Why this exists rather than reusing the raw download directly:
- NeRF-Synthetic images are RGBA with a real alpha channel (transparent
  background), not opaque RGB like every scene this pipeline has used so
  far. `train_minimal_gsplat.load_dataset` / `render_reconstruction.
  render_views` both do `Image.open(path).convert("RGB")`, which on an
  RGBA source silently *drops* the alpha channel rather than compositing
  it -- leaving whatever raw RGB was stored under transparent pixels
  (typically black), which would train against ground truth that doesn't
  match what the model actually renders (alpha-composited over
  `background_color`). This script alpha-composites onto a chosen
  background once, up front, so every existing (already-tested) loader
  keeps working unchanged on the output.
- 800x800 source resolution is heavier per-iteration than this minimal,
  unoptimized trainer can reasonably afford at the iteration counts used
  elsewhere in this project; this resizes to a configurable, smaller
  resolution.
- "Wide" is the standard 100-view training split as-is. "Narrow" selects
  a clustered angular subset by real 3D camera-position similarity (dot
  product of normalized positions to a reference view), not by parsing
  an assumed up-axis convention out of the transform matrices -- robust
  to whatever convention a given Blender export used, and analogous to
  camera.py's turntable_arc "narrow cone" pattern, applied to real,
  irregularly-sampled poses instead of a regular ring.

Run: .venv-gsplat/bin/python gs_experiment/prepare_nerf_synthetic.py <raw_scene_dir> <out_dir> --n-narrow 12
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from gs_experiment.nerf_transforms import load_transforms, write_transforms_json


def composite_and_resize(src_path: str, dst_path: str, background_color, resolution: int):
    im = Image.open(src_path)
    if im.mode == "RGBA":
        rgba = np.asarray(im, dtype=np.float32) / 255.0
        rgb, alpha = rgba[..., :3], rgba[..., 3:4]
        bg = np.array(background_color, dtype=np.float32).reshape(1, 1, 3)
        composited = rgb * alpha + bg * (1.0 - alpha)
        im = Image.fromarray((composited * 255.0 + 0.5).astype(np.uint8), mode="RGB")
    else:
        im = im.convert("RGB")
    if resolution != im.size[0]:
        im = im.resize((resolution, resolution), Image.LANCZOS)
    im.save(dst_path)


def prepare_images(raw_scene_dir: str, out_dir: str, split: str, background_color, resolution: int):
    """Some public mirrors of this dataset ship an incomplete image set
    for a split (found the hard way: this run's `test` split lists 200
    frames in transforms_test.json but only ships 36 plain color PNGs,
    mixed in with depth/normal debug renders that aren't ground-truth
    color at all) -- frames missing their color image are skipped rather
    than treated as an error, since a partial held-out set is still a
    valid held-out set for this project's purposes."""
    camera_angle_x, all_frames = load_transforms(os.path.join(raw_scene_dir, f"transforms_{split}.json"))
    out_images_dir = os.path.join(out_dir, split)
    os.makedirs(out_images_dir, exist_ok=True)
    frames = []
    for file_path, c2w in all_frames:
        src = os.path.join(raw_scene_dir, file_path + ".png")
        if not os.path.exists(src):
            continue
        dst = os.path.join(out_dir, file_path + ".png")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        composite_and_resize(src, dst, background_color, resolution)
        frames.append((file_path, c2w))
    if len(frames) < len(all_frames):
        print(f"{split}: {len(frames)}/{len(all_frames)} frames had a color image (rest skipped)")
    return camera_angle_x, frames


def select_narrow_subset(frames, n_narrow: int, reference_idx: int = 0) -> np.ndarray:
    centers = np.array([c2w[:3, 3] for _, c2w in frames])
    dirs = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    ref = dirs[reference_idx]
    similarity = dirs @ ref
    order = np.argsort(-similarity)
    return order[:n_narrow]


def select_gradient_subset(frames, n_per_zone: int, window_fraction: float, reference_idx: int = 0) -> np.ndarray:
    """Generalizes `select_narrow_subset` from one fixed cluster to one
    level of a coverage *gradient*, on a real dataset's fixed, pre-baked
    camera poses (unlike `scene_spec.gradient_scene`, which can place
    cameras anywhere -- a real benchmark's views are what they are).

    Same real-3D-position cosine-similarity ranking as `select_narrow_
    subset`, but rather than always taking the top `n_per_zone` most
    similar views (which conflates "how many views" with "how spread
    out"), this evenly subsamples `n_per_zone` views from a *window* of
    the `window_fraction` most-similar views -- so view *count* is held
    fixed across gradient levels (matching `gradient_scene`'s "hold
    geometry/count equal, vary only angular spread" confound control) and
    only the width of the window views are drawn from actually changes.
    `window_fraction=1.0` draws from the entire pool (widest, most-spread
    condition); a small `window_fraction` draws only from a tight cone
    around the reference view (narrowest condition).
    """
    centers = np.array([c2w[:3, 3] for _, c2w in frames])
    dirs = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    ref = dirs[reference_idx]
    similarity = dirs @ ref
    order = np.argsort(-similarity)

    window_size = max(n_per_zone, int(round(window_fraction * len(frames))))
    window = order[:window_size]
    pick_positions = np.linspace(0, len(window) - 1, n_per_zone).round().astype(int)
    return window[pick_positions]


def write_condition(out_dir: str, camera_angle_x: float, frames, indices, condition_name: str, split_prefix: str):
    scene_dir = os.path.join(out_dir, condition_name)
    os.makedirs(scene_dir, exist_ok=True)
    images_link = os.path.join(scene_dir, split_prefix)
    if not os.path.exists(images_link):
        os.symlink(os.path.abspath(os.path.join(out_dir, split_prefix)), images_link)
    subset = [{"file_path": frames[i][0], "transform_matrix": frames[i][1]} for i in indices]
    write_transforms_json(os.path.join(scene_dir, "transforms.json"), camera_angle_x, subset)
    return scene_dir


def run(raw_scene_dir: str, out_dir: str, n_narrow: int = 12, resolution: int = 400, background_color=(1.0, 1.0, 1.0)):
    os.makedirs(out_dir, exist_ok=True)
    print("compositing + resizing train images...")
    camera_angle_x, train_frames = prepare_images(raw_scene_dir, out_dir, "train", background_color, resolution)
    print("compositing + resizing test images...")
    _, test_frames = prepare_images(raw_scene_dir, out_dir, "test", background_color, resolution)

    narrow_indices = select_narrow_subset(train_frames, n_narrow)
    wide_dir = write_condition(out_dir, camera_angle_x, train_frames, range(len(train_frames)), "wide", "train")
    narrow_dir = write_condition(out_dir, camera_angle_x, train_frames, narrow_indices, "narrow", "train")

    n_eval = min(30, len(test_frames))
    rng = np.random.default_rng(0)
    eval_indices = rng.choice(len(test_frames), size=n_eval, replace=False)
    eval_dir = write_condition(out_dir, camera_angle_x, test_frames, eval_indices, "eval", "test")

    print(f"wide: {len(train_frames)} views -> {wide_dir}")
    print(f"narrow: {n_narrow} views -> {narrow_dir}")
    print(f"eval (held out, from the official test split): {n_eval} views -> {eval_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("raw_scene_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--n-narrow", type=int, default=12)
    parser.add_argument("--resolution", type=int, default=400)
    args = parser.parse_args()
    run(args.raw_scene_dir, args.out_dir, n_narrow=args.n_narrow, resolution=args.resolution)


if __name__ == "__main__":
    main()
