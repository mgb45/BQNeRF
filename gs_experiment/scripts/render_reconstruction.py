"""Shared rendering library: renders ground-truth vs. gsplat reconstruction
for a trained checkpoint (gs_experiment.scripts.train_minimal_gsplat's
output), plus real depth-unprojection (gsplat's own "ED" output) into
world-space query points -- what render_sparse_gp_uncertainty.py builds its
own BQ uncertainty on top of.

Needs torch + gsplat (requirements-gsplat.txt).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from PIL import Image

from gs_experiment.nerf_transforms import fov_x_to_intrinsics, load_transforms, opencv_viewmat_from_c2w
from gs_experiment.ply_io import read_3dgs_ply

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_RUNS = REPO_ROOT / "gs_experiment" / "local_runs"

# The 7 NeRF-Synthetic scenes this project's results use -- `materials`
# deliberately excluded (its best held-out view stays visibly hazy under
# this project's training recipe, unrelated to anything measured here).
SCENES = ["chair", "drums", "ficus", "hotdog", "lego", "mic", "ship"]

# Per-checkpoint local-window radius. NOT shared across checkpoints: `wide`
# (~300k splats) and `budget_500` (500 splats) cover the *same* physical
# scene volume, so a radius chosen for one density gives a wildly different
# number of real neighbors at the other (confirmed directly on lego:
# r=0.08 finds a median of ~1000 neighbors per window in `wide` but a
# median of ~1-2 in `budget_500`, below the >=6 minimum a local GP fit
# needs). r=0.25 was chosen for `budget_500` as the smallest radius giving
# >=6 real neighbors for >90% of candidate window centers. Both radii were
# re-checked on all 7 scenes, not just lego: every scene/checkpoint
# combination gives >=6 real neighbors for at least 86% of candidate
# windows.
WINDOW_RADIUS = {"wide": 0.08, "budget_500": 0.25}

CHECKPOINTS = {
    scene: {
        "wide": {"dir": LOCAL_RUNS / f"{scene}_prepared" / "wide", "window_radius": WINDOW_RADIUS["wide"]},
        "budget_500": {"dir": LOCAL_RUNS / f"{scene}_prepared" / "budget_500", "window_radius": WINDOW_RADIUS["budget_500"]},
    }
    for scene in SCENES
}
EVAL_DIRS = {scene: LOCAL_RUNS / f"{scene}_prepared" / "eval" for scene in SCENES}


def render_views(scene_dir: str, view_indices, checkpoint_dir=None, background_color=(0.05, 0.05, 0.05), device="cuda"):
    """`scene_dir` supplies transforms.json + images (the views to render);
    `checkpoint_dir` (defaults to `scene_dir`) supplies splats.ply. Pass a
    different `checkpoint_dir` to render a checkpoint trained on one split
    (e.g. `wide/`) against another split's views (e.g. a held-out `eval/`),
    for genuine held-out-test-image comparisons rather than reconstruction
    on the views the model was fit to."""
    import gsplat

    checkpoint = read_3dgs_ply(os.path.join(checkpoint_dir or scene_dir, "splats.ply"))
    camera_angle_x, frames = load_transforms(os.path.join(scene_dir, "transforms.json"))

    with Image.open(os.path.join(scene_dir, frames[0][0] + ".png")) as im:
        width, height = im.size
    K = fov_x_to_intrinsics(camera_angle_x, width, height)
    background = torch.tensor(background_color, dtype=torch.float32, device=device)

    positions = torch.tensor(checkpoint["positions"], dtype=torch.float32, device=device)
    scales = torch.tensor(checkpoint["scales"], dtype=torch.float32, device=device)
    rotations = torch.tensor(checkpoint["rotations"], dtype=torch.float32, device=device)
    opacities = torch.tensor(checkpoint["opacities"], dtype=torch.float32, device=device)
    # (N, 3, K) on disk (this project's convention, see ply_io); gsplat wants (N, K, 3)
    sh = torch.tensor(checkpoint["sh_coeffs"], dtype=torch.float32, device=device).transpose(1, 2)
    sh_degree = checkpoint["sh_degree"]

    results = []
    with torch.no_grad():
        for i in view_indices:
            file_path, c2w = frames[i]
            viewmat = torch.tensor(opencv_viewmat_from_c2w(c2w), dtype=torch.float32, device=device)[None]
            Ks = torch.tensor(K, dtype=torch.float32, device=device)[None]

            rendered, _, _ = gsplat.rasterization(
                positions, rotations, scales, opacities, sh, viewmat, Ks,
                width=width, height=height, sh_degree=sh_degree, backgrounds=background,
            )
            recon = rendered[0].clamp(0, 1).cpu().numpy()

            gt_path = os.path.join(scene_dir, file_path + ".png")
            gt = np.asarray(Image.open(gt_path).convert("RGB"), dtype=np.float32) / 255.0

            results.append((i, gt, recon))
    return results, checkpoint


def unproject_depth_grid(depth: np.ndarray, K: np.ndarray, c2w_cv: np.ndarray) -> np.ndarray:
    """Same construction as train_minimal_gsplat.py's function of the same
    name (duplicated rather than imported, so that training doesn't depend on
    this rendering/plotting script -- see train_minimal_gsplat.py's own
    docstring on that function): depth (H, W) in OpenCV camera space ->
    (H, W, 3) world-space points."""
    h, w = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    us, vs = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    x_cam = (us - cx) / fx * depth
    y_cam = (vs - cy) / fy * depth
    cam_points = np.stack([x_cam, y_cam, depth, np.ones_like(depth)], axis=-1)
    world_points = cam_points @ c2w_cv.T
    return world_points[..., :3]



def _render_and_unproject(checkpoint_dir: Path, eval_dir: Path, view_indices, depth_width=112, depth_height=42, device="cuda"):
    """Real GT vs. reconstruction (full res) plus real depth-unprojected
    world points at a lower resolution, for each of `view_indices` in
    `eval_dir` (that scene's own held-out view set) -- what
    render_sparse_gp_uncertainty.py queries its BQ uncertainty terms at.

    Returns a list of dicts: gt (H,W,3), recon (H,W,3), world_points
    (depth_height, depth_width, 3), valid (depth_height, depth_width) bool.
    """
    import gsplat

    results, checkpoint = render_views(str(eval_dir), view_indices, checkpoint_dir=str(checkpoint_dir), device=device)
    camera_angle_x, frames = load_transforms(str(eval_dir / "transforms.json"))

    K_full = fov_x_to_intrinsics(camera_angle_x, results[0][1].shape[1], results[0][1].shape[0])
    K_depth = K_full.copy()
    width, height = results[0][1].shape[1], results[0][1].shape[0]
    K_depth[0, 0] *= depth_width / width
    K_depth[0, 2] *= depth_width / width
    K_depth[1, 1] *= depth_height / height
    K_depth[1, 2] *= depth_height / height
    K_depth_t = torch.tensor(K_depth, dtype=torch.float32, device=device)

    all_positions = torch.tensor(checkpoint["positions"], dtype=torch.float32, device=device)
    all_scales = torch.tensor(checkpoint["scales"], dtype=torch.float32, device=device)
    all_rotations = torch.tensor(checkpoint["rotations"], dtype=torch.float32, device=device)
    all_opacities = torch.tensor(checkpoint["opacities"], dtype=torch.float32, device=device)
    all_sh = torch.tensor(checkpoint["sh_coeffs"], dtype=torch.float32, device=device).transpose(1, 2)
    sh_degree = checkpoint["sh_degree"]

    out = []
    with torch.no_grad():
        for view_i, gt, recon in results:
            _, c2w = frames[view_i]
            viewmat_np = opencv_viewmat_from_c2w(c2w)
            viewmat = torch.tensor(viewmat_np, dtype=torch.float32, device=device)
            c2w_cv = np.linalg.inv(viewmat_np)

            rendered_lo, alpha_lo, _ = gsplat.rasterization(
                all_positions, all_rotations, all_scales, all_opacities, all_sh, viewmat[None], K_depth_t[None],
                width=depth_width, height=depth_height, sh_degree=sh_degree, render_mode="ED",
            )
            depth_map = rendered_lo[0, ..., 0].cpu().numpy()
            alpha_map = alpha_lo[0, ..., 0].cpu().numpy()
            valid = alpha_map > 0.5
            world_points = unproject_depth_grid(depth_map, K_depth, c2w_cv)
            out.append({"gt": gt, "recon": recon, "world_points": world_points, "valid": valid})
    return out
