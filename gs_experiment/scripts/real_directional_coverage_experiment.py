"""Consolidated CLI for the real-geometry directional-uncertainty coverage
experiment that used to be two separate one-off scripts:

  * gap_directional_experiment.py            (--dataset lego)
  * gap_capture_experiment.py                 (--dataset bonsai)

Both train their own set of real gsplat checkpoints and query rendering-
aware directional vs. position-only BQ variance across them: starting
from the full training-view pool, remove a single deliberate angular gap
of increasing half-width around one reference view's direction, leaving
every other view in the pool untouched
(`prepare_nerf_synthetic.select_gap_subset`). Differs only in *dataset*
(synthetic NeRF-Synthetic "lego" vs. a genuinely photographed,
COLMAP-posed Mip-NeRF360 "bonsai" capture).

An earlier "subsample" design (hold view count fixed, vary the angular
width views are drawn from) was tried first and is retired -- it
confounded angular spread with global view thinning (see FINDINGS.md
section 4); the gap design above avoids that and is what every current
result uses. See git history to resurrect the subsample design/code.

Needs torch + gsplat (requirements-gsplat.txt); the bonsai mode also
needs PIL (image resizing) and colmap_loader.

Run examples:
  .venv-gsplat/bin/python gs_experiment/real_directional_coverage_experiment.py \\
      --dataset lego gs_experiment/local_runs/lego_prepared

  .venv-gsplat/bin/python gs_experiment/real_directional_coverage_experiment.py \\
      --dataset bonsai gs_experiment/local_runs/bonsai_prepared
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from gs_experiment.kernels import DirectionalKernel
from gs_experiment.camera import CameraPose
from gs_experiment.nerf_transforms import fov_x_to_intrinsics, load_transforms, opencv_viewmat_from_c2w
from gs_experiment.pixel_uncertainty import LocalUncertaintyEngine, make_default_3d_position_kernel
from gs_experiment.scripts.prepare_nerf_synthetic import select_gap_subset, write_condition
from gs_experiment.scripts.render_reconstruction import render_views
from gs_experiment.splat_scene import load_from_gsplat_checkpoint, splat_observations
from gs_experiment.scripts.train_minimal_gsplat import train, train_with_reference_strategy

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_IDX = 0


# =====================================================================
# --dataset lego  (was gap_directional_experiment.py)
# =====================================================================
LEGO_GAP_HALF_WIDTHS_DEG = [0.0, 15.0, 30.0, 50.0, 75.0]

LEGO_GAP_TRAIN_KWARGS = dict(
    # Matches evaluate_checkpoint.py's TRAIN_KWARGS, the recipe that gives every other
    # NeRF-Synthetic scene in this project real reconstruction quality (ROADMAP.md item
    # 1's quality-bar fix) -- background_color=(1,1,1) matters most: without it this
    # dict silently fell back to train()'s dark (0.05,0.05,0.05) default while
    # prepare_nerf_synthetic.py composites onto white, a real background mismatch that
    # alone produced a degenerate, hazy reconstruction at every gap condition.
    n_splats=5000, bounds=((-2.5, 2.5), (-2.5, 2.5), (-2.5, 2.5)), sh_degree=3, n_iters=30000, seed=0,
    init_scale=0.05, opacity_reg_weight=0.01, densify=True, densify_interval=100, densify_start=500,
    densify_end=15000, min_opacity=0.005, max_splats=300000, log_every=2000, position_lr_final=2e-5,
    background_color=(1.0, 1.0, 1.0),
)
# Marginal-likelihood-fitted (hyperparams.py::fit_kernel_param_pooled_nd), pooled across 9
# real checkpoints spanning both this experiment's lego-gap checkpoints and the cross-scene
# gallery figure's checkpoints -- not hand-picked. This replaces an old sigma=0.9/window_radius=1.6
# pair that turned out to be catastrophically wrong for this data (held-out log marginal
# likelihood -35.8 MILLION vs -8607 at the fitted value -- not merely suboptimal), inherited
# from an older, unreconciled convention and never checked against the data until a cross-figure
# magnitude comparison exposed it (see git history / conversation this was caught in). Matches
# render_scene_gallery.py's SIGMA/WINDOW_RADIUS exactly, so results are comparable across both
# figures, not just internally consistent within each.
LEGO_GAP_SIGMA = 0.0694
LEGO_GAP_WINDOW_RADIUS = 0.08
LEGO_GAP_KAPPA = 4.0
LEGO_GAP_GATE_BACKGROUND_COLOR = (1.0, 1.0, 1.0)  # matches prepare_nerf_synthetic's own default compositing background


def lego_gap_build_conditions(prepared_dir: str, gap_half_widths=LEGO_GAP_HALF_WIDTHS_DEG, condition_prefix: str = "gap"):
    wide_dir = os.path.join(prepared_dir, "wide")
    camera_angle_x, frames = load_transforms(os.path.join(wide_dir, "transforms.json"))

    centers = np.array([c2w[:3, 3] for _, c2w in frames])
    dirs = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    gap_center_direction = dirs[REFERENCE_IDX]

    zone_dirs = []
    n_views = []
    for i, hw in enumerate(gap_half_widths):
        idx = select_gap_subset(frames, gap_half_width_deg=hw, reference_idx=REFERENCE_IDX)
        zone_dir = write_condition(prepared_dir, camera_angle_x, frames, idx, f"{condition_prefix}_{i}", "train")
        zone_dirs.append(zone_dir)
        n_views.append(len(idx))

    return zone_dirs, np.array(gap_half_widths), np.array(n_views), gap_center_direction


def lego_gap_train_zones(zone_dirs, train_kwargs=LEGO_GAP_TRAIN_KWARGS):
    for zone_dir in zone_dirs:
        ply_path = os.path.join(zone_dir, "splats.ply")
        if not os.path.exists(ply_path):
            print(f"training {zone_dir}...")
            train(zone_dir, ply_path, **train_kwargs)
        else:
            print(f"reusing existing checkpoint at {ply_path}")


def lego_gap_check_reconstruction_quality(zone_dirs, eval_dir: str, gap_half_widths, gap_center_direction, reference_idx=REFERENCE_IDX):
    """Overall held-out PSNR *and* PSNR restricted to eval views that fall
    inside each condition's own gap -- a real local quality drop inside
    the gap is expected and isn't a confound; a global drop would be."""
    _, eval_frames = load_transforms(os.path.join(eval_dir, "transforms.json"))
    n_eval = len(eval_frames)

    eval_centers = np.array([c2w[:3, 3] for _, c2w in eval_frames])
    eval_dirs = eval_centers / np.linalg.norm(eval_centers, axis=1, keepdims=True)
    eval_angular_dist_deg = np.degrees(np.arccos(np.clip(eval_dirs @ gap_center_direction, -1.0, 1.0)))

    overall_psnrs, gap_psnrs = [], []
    for zone_dir, hw in zip(zone_dirs, gap_half_widths):
        eval_copy_dir = zone_dir + "_eval"
        os.makedirs(eval_copy_dir, exist_ok=True)
        shutil.copy(os.path.join(eval_dir, "transforms.json"), os.path.join(eval_copy_dir, "transforms.json"))
        images_link = os.path.join(eval_copy_dir, "test")
        if not os.path.exists(images_link):
            os.symlink(os.path.abspath(os.path.join(eval_dir, "test")), images_link)
        shutil.copy(os.path.join(zone_dir, "splats.ply"), os.path.join(eval_copy_dir, "splats.ply"))

        results, _ = render_views(eval_copy_dir, list(range(n_eval)), background_color=LEGO_GAP_GATE_BACKGROUND_COLOR)
        per_view_psnr = np.array(
            [-10.0 * np.log10(max(float(np.mean((gt - recon) ** 2)), 1e-10)) for _, gt, recon in results]
        )
        overall_psnrs.append(float(per_view_psnr.mean()))

        in_gap = eval_angular_dist_deg <= max(hw, 1e-6)
        gap_psnr = float(per_view_psnr[in_gap].mean()) if in_gap.any() else float("nan")
        gap_psnrs.append(gap_psnr)
        print(
            f"{zone_dir}: overall held-out PSNR = {overall_psnrs[-1]:.2f}dB over {n_eval} views; "
            f"in-gap PSNR = {gap_psnr:.2f}dB over {int(in_gap.sum())} views"
        )

    return np.array(overall_psnrs), np.array(gap_psnrs)


def synthetic_camera_for_query(scene, query_point: np.ndarray, query_direction: np.ndarray) -> CameraPose:
    """A stand-in `CameraPose` for `lego_gap_analyze`/`bonsai_gap_analyze`'s
    directional query -- these ask "how uncertain is BQ at this fixed
    world point, looking in this fixed direction," not "at this specific
    training view," so there's no real camera to reuse for the
    rendering-aware directional methods' occlusion/bearing computation
    (they need a `CameraPose` to build a `CameraSplatIndex`/bearing frame
    from). Built here rather than left unset: centered along
    `query_direction` from `query_point`, looking back at it, at a
    distance matching this checkpoint's own real training cameras (the
    median distance of `scene.cameras` from the scene centroid) -- a
    representative real scale already present in the loaded checkpoint,
    not an arbitrary constant. `up` is a fixed world-up vector; only the
    bearing-space geometry (not the image plane orientation) matters for
    occlusion attribution, so any `up` not parallel to `forward` works.
    """
    centers = np.array([c.center for c in scene.cameras])
    centroid = centers.mean(axis=0)
    distance = float(np.median(np.linalg.norm(centers - centroid, axis=1)))
    center = query_point + distance * query_direction
    forward = -query_direction
    up = np.array([0.0, 0.0, 1.0]) if abs(forward[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    return CameraPose(center=center, forward=forward, up=up)


def lego_gap_analyze(zone_dirs, gap_half_widths, n_views, query_direction, overall_psnrs=None, gap_psnrs=None):
    directional_vars, spatial_vars = [], []
    query_point = np.zeros(3)

    for zone_dir in zone_dirs:
        scene = load_from_gsplat_checkpoint(zone_dir, attribution_angular_tol=0.01)
        positions, directions, values, obs_opacities, obs_scales, obs_rotations = splat_observations(
            scene, include_render_attrs=True
        )
        bounds = tuple((positions[:, d].min() - 1.0, positions[:, d].max() + 1.0) for d in range(3))

        pos_kernel = make_default_3d_position_kernel(sigma=LEGO_GAP_SIGMA)
        dir_kernel = DirectionalKernel(kappa=LEGO_GAP_KAPPA)
        engine = LocalUncertaintyEngine(
            positions=positions, values=values, pos_kernel=pos_kernel, scene_bounds=bounds,
            directions=directions, dir_kernel=dir_kernel,
            opacities=obs_opacities, scales=obs_scales, rotations=obs_rotations,
        )

        camera = synthetic_camera_for_query(scene, query_point, query_direction)
        camera_index = engine.build_bearing_index(camera)
        dir_result = engine.rendering_aware_variance_along_ray_directional(
            query_point, query_direction, camera_index, LEGO_GAP_WINDOW_RADIUS
        )
        spatial_result = engine.rendering_aware_variance(query_point, LEGO_GAP_WINDOW_RADIUS)
        directional_vars.append(dir_result.variance)
        spatial_vars.append(spatial_result.variance)

    directional_vars = np.array(directional_vars)
    spatial_vars = np.array(spatial_vars)
    order = np.argsort(gap_half_widths)

    header = f"{'gap (deg)':>10}{'n_views':>9}"
    if overall_psnrs is not None:
        header += f"{'overall PSNR':>14}{'in-gap PSNR':>14}"
    header += f"{'directional var':>18}{'spatial-only var':>18}"
    print(f"\n{header}")
    for i in order:
        row = f"{gap_half_widths[i]:>10.1f}{n_views[i]:>9d}"
        if overall_psnrs is not None:
            row += f"{overall_psnrs[i]:>14.2f}{gap_psnrs[i]:>14.2f}"
        row += f"{directional_vars[i]:>18.5f}{spatial_vars[i]:>18.5f}"
        print(row)

    dir_sorted = directional_vars[order]
    is_monotonic = bool(np.all(np.diff(dir_sorted) >= -1e-12))
    rho = float(np.corrcoef(np.argsort(np.argsort(gap_half_widths)), np.argsort(np.argsort(directional_vars)))[0, 1])
    print(f"\ndirectional variance monotonically increasing with gap width: {is_monotonic}")
    print(f"rank correlation (gap width vs. directional variance): rho={rho:.3f}")
    print(f"directional variance range (widest/narrowest): {dir_sorted[-1] / max(dir_sorted[0], 1e-12):.2f}x")
    print(f"spatial-only variance range (control): {spatial_vars.max() / max(spatial_vars.min(), 1e-12):.2f}x")
    if overall_psnrs is not None:
        print(
            f"overall held-out PSNR range across conditions: {overall_psnrs.min():.2f}-{overall_psnrs.max():.2f}dB "
            "(should stay tight if the gap design avoids the earlier global-thinning confound)"
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(gap_half_widths[order], directional_vars[order], "o-", color="tab:red", label="directional variance")
    ax1.set_xlabel("gap half-width around query direction (deg)")
    ax1.set_ylabel("position+direction BQ variance", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.plot(gap_half_widths[order], spatial_vars[order], "s--", color="tab:blue", label="spatial-only variance (control)")
    ax2.set_ylabel("position-only BQ variance", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    fig.suptitle("Real NeRF-Synthetic (lego): directional BQ variance vs. a deliberate real coverage gap")
    fig.tight_layout()
    out_path = RESULTS_DIR / "gap_directional_gradient.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_path}")


def lego_gap_run(
    prepared_dir: str,
    gap_half_widths=LEGO_GAP_HALF_WIDTHS_DEG,
    condition_prefix: str = "gap",
    train_kwargs=LEGO_GAP_TRAIN_KWARGS,
    check_quality: bool = True,
):
    zone_dirs, gap_half_widths, n_views, query_direction = lego_gap_build_conditions(
        prepared_dir, gap_half_widths=gap_half_widths, condition_prefix=condition_prefix,
    )
    lego_gap_train_zones(zone_dirs, train_kwargs=train_kwargs)
    overall_psnrs = gap_psnrs = None
    if check_quality:
        eval_dir = os.path.join(prepared_dir, "eval")
        if os.path.exists(eval_dir):
            overall_psnrs, gap_psnrs = lego_gap_check_reconstruction_quality(
                zone_dirs, eval_dir, gap_half_widths, query_direction
            )
        else:
            print(f"no eval/ split found at {eval_dir}, skipping reconstruction-quality check")
    lego_gap_analyze(zone_dirs, gap_half_widths, n_views, query_direction, overall_psnrs=overall_psnrs, gap_psnrs=gap_psnrs)


# =====================================================================
# --dataset bonsai  (was gap_capture_experiment.py, preferred as
# the more-correct/more-recent iteration of the gap design)
# =====================================================================
BONSAI_GAP_HALF_WIDTHS_DEG = [0.0, 20.0, 40.0, 65.0, 90.0]
BONSAI_GAP_N_EVAL = 30
BONSAI_GAP_MIN_PSNR = 20.0

BONSAI_GAP_TRAIN_KWARGS = dict(
    n_splats=4000, sh_degree=1, n_iters=12000, seed=0,
    init_scale=0.03, opacity_reg_weight=0.003, log_every=1000,
    means_lr_decay_to=0.05,
)
BONSAI_GAP_SIGMA = 0.05
BONSAI_GAP_WINDOW_RADIUS = 0.8
BONSAI_GAP_KAPPA = 4.0
# Real, data-derived object location for this exact COLMAP reconstruction
# (mean, across 5 earlier checkpoints of this same scene, of each
# checkpoint's own median high-opacity splat position) -- not the world
# origin, which was checked and found ~0.8-0.9 units from the nearest
# real splat.
BONSAI_GAP_QUERY_POINT = np.array([0.612, 1.174, 1.507])


def bonsai_gap_load_pool(out_dir: str):
    all_transforms_path = os.path.join(out_dir, "all_transforms.json")
    camera_angle_x, frames = load_transforms(all_transforms_path)
    return camera_angle_x, frames


def bonsai_gap_build_conditions(
    out_dir: str, camera_angle_x: float, frames, gap_half_widths=BONSAI_GAP_HALF_WIDTHS_DEG,
    condition_prefix: str = "gap", seed: int = 0,
):
    centers = np.array([c2w[:3, 3] for _, c2w in frames])
    dirs = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    gap_center_direction = dirs[REFERENCE_IDX]

    # REFERENCE_IDX must never land in eval -- select_gap_subset below is
    # called with reference_idx=0 against train_pool_frames, which only
    # lines up with frames[REFERENCE_IDX]'s direction because index 0 is
    # excluded from the eval draw (so it's always the smallest surviving
    # global index, and np.setdiff1d's sorted output keeps it at position
    # 0 of train_pool_frames).
    rng = np.random.default_rng(seed)
    assert REFERENCE_IDX == 0, "eval draw below assumes REFERENCE_IDX == 0"
    eval_idx = rng.choice(np.arange(1, len(frames)), size=BONSAI_GAP_N_EVAL, replace=False)
    train_pool_idx = np.setdiff1d(np.arange(len(frames)), eval_idx)
    train_pool_frames = [frames[i] for i in train_pool_idx]
    assert np.array_equal(train_pool_frames[0][1], frames[0][1]), "reference frame must stay at position 0"

    eval_dir = write_condition(out_dir, camera_angle_x, frames, eval_idx, "eval", "all")

    zone_dirs, n_views = [], []
    for i, hw in enumerate(gap_half_widths):
        idx_in_pool = select_gap_subset(train_pool_frames, gap_half_width_deg=hw, reference_idx=0)
        # select_gap_subset's reference_idx=0 means "the first frame in
        # train_pool_frames", not global index 0 -- fine here since
        # train_pool_frames[0]'s direction is what matters, but we still
        # want the *global* gap_center_direction (frames[REFERENCE_IDX])
        # for querying later, computed above before any subsetting.
        zone_dir = write_condition(out_dir, camera_angle_x, train_pool_frames, idx_in_pool, f"{condition_prefix}_{i}", "all")
        zone_dirs.append(zone_dir)
        n_views.append(len(idx_in_pool))

    return zone_dirs, np.array(gap_half_widths), np.array(n_views), gap_center_direction, eval_dir


def bonsai_gap_train_zones(zone_dirs, bounds, train_kwargs=BONSAI_GAP_TRAIN_KWARGS):
    # gsplat's own official reference densification strategy
    # (train_with_reference_strategy), not this project's from-scratch
    # `train` -- found the hard way: `train`'s simple percentile-threshold
    # densification produced wildly oscillating train-view PSNR on this
    # real, noisier scene (15-23dB, never converging), unlike its stable
    # behavior on clean synthetic lego. The official strategy's periodic
    # opacity reset (`reset_every`, part of the standard 3DGS algorithm)
    # is exactly the kind of real-scene stabilization `train` doesn't
    # have, and this project already validated it reproduces this
    # project's own findings closely on lego (ROADMAP.md item 4 history).
    for zone_dir in zone_dirs:
        ply_path = os.path.join(zone_dir, "splats.ply")
        if not os.path.exists(ply_path):
            print(f"training {zone_dir}...")
            train_with_reference_strategy(zone_dir, ply_path, bounds=bounds, **train_kwargs)
        else:
            print(f"reusing existing checkpoint at {ply_path}")


def bonsai_gap_object_region_crop_bounds(c2w, camera_angle_x, width, height, query_point, crop_frac=0.35):
    """Pixel-space crop window centered on `query_point`'s real projection
    into this view -- for a Mip-NeRF360-style unbounded real capture,
    whole-frame PSNR conflates "is the object well reconstructed" with
    "is the far, cluttered background well reconstructed" (a much harder,
    and for this experiment largely irrelevant, problem -- confirmed
    directly by looking at renders where the object was sharp and
    recognizable but the background was a blurry wash, dragging the
    whole-frame number down for reasons unrelated to the actual query
    region). Returns None if the point doesn't project inside the frame."""
    K = fov_x_to_intrinsics(camera_angle_x, width, height)
    viewmat = opencv_viewmat_from_c2w(c2w)
    point_cam = viewmat[:3, :3] @ query_point + viewmat[:3, 3]
    if point_cam[2] <= 0:
        return None
    pixel = K @ point_cam
    px, py = pixel[0] / pixel[2], pixel[1] / pixel[2]
    if not (0 <= px < width and 0 <= py < height):
        return None
    half = crop_frac * min(width, height) / 2.0
    x0, x1 = int(max(0, px - half)), int(min(width, px + half))
    y0, y1 = int(max(0, py - half)), int(min(height, py + half))
    return x0, x1, y0, y1


def bonsai_gap_check_reconstruction_quality(zone_dirs, eval_dir: str, gap_half_widths, gap_center_direction, query_point=None):
    camera_angle_x, eval_frames = load_transforms(os.path.join(eval_dir, "transforms.json"))
    n_eval = len(eval_frames)

    eval_centers = np.array([c2w[:3, 3] for _, c2w in eval_frames])
    eval_dirs = eval_centers / np.linalg.norm(eval_centers, axis=1, keepdims=True)
    eval_angular_dist_deg = np.degrees(np.arccos(np.clip(eval_dirs @ gap_center_direction, -1.0, 1.0)))

    overall_psnrs, gap_psnrs, object_psnrs = [], [], []
    for zone_dir, hw in zip(zone_dirs, gap_half_widths):
        eval_copy_dir = zone_dir + "_eval"
        os.makedirs(eval_copy_dir, exist_ok=True)
        shutil.copy(os.path.join(eval_dir, "transforms.json"), os.path.join(eval_copy_dir, "transforms.json"))
        images_link = os.path.join(eval_copy_dir, "all")
        if not os.path.exists(images_link):
            os.symlink(os.path.abspath(os.path.join(eval_dir, "all")), images_link)
        shutil.copy(os.path.join(zone_dir, "splats.ply"), os.path.join(eval_copy_dir, "splats.ply"))

        results, _ = render_views(eval_copy_dir, list(range(n_eval)))
        per_view_psnr = np.array(
            [-10.0 * np.log10(max(float(np.mean((gt - recon) ** 2)), 1e-10)) for _, gt, recon in results]
        )
        overall_psnrs.append(float(per_view_psnr.mean()))

        in_gap = eval_angular_dist_deg <= max(hw, 1e-6)
        gap_psnr = float(per_view_psnr[in_gap].mean()) if in_gap.any() else float("nan")
        gap_psnrs.append(gap_psnr)

        object_view_psnrs = []
        if query_point is not None:
            for j, (_, gt, recon) in enumerate(results):
                height, width = gt.shape[:2]
                crop = bonsai_gap_object_region_crop_bounds(eval_frames[j][1], camera_angle_x, width, height, query_point)
                if crop is None:
                    continue
                x0, x1, y0, y1 = crop
                mse = float(np.mean((gt[y0:y1, x0:x1] - recon[y0:y1, x0:x1]) ** 2))
                object_view_psnrs.append(-10.0 * np.log10(max(mse, 1e-10)))
        object_psnr = float(np.mean(object_view_psnrs)) if object_view_psnrs else float("nan")
        object_psnrs.append(object_psnr)

        below_bar = (per_view_psnr < BONSAI_GAP_MIN_PSNR).sum()
        print(
            f"{zone_dir}: overall held-out PSNR = {overall_psnrs[-1]:.2f}dB over {n_eval} views "
            f"({below_bar}/{n_eval} individual views below {BONSAI_GAP_MIN_PSNR}dB); "
            f"in-gap PSNR = {gap_psnr:.2f}dB over {int(in_gap.sum())} views; "
            f"object-region PSNR = {object_psnr:.2f}dB over {len(object_view_psnrs)} views"
        )

    return np.array(overall_psnrs), np.array(gap_psnrs), np.array(object_psnrs)


def bonsai_gap_analyze(zone_dirs, gap_half_widths, n_views, query_direction, overall_psnrs=None, gap_psnrs=None, object_psnrs=None):
    directional_vars, spatial_vars = [], []
    query_point = BONSAI_GAP_QUERY_POINT

    for zone_dir in zone_dirs:
        scene = load_from_gsplat_checkpoint(zone_dir, attribution_angular_tol=0.01)
        positions, directions, values, obs_opacities, obs_scales, obs_rotations = splat_observations(
            scene, include_render_attrs=True
        )
        bounds3d = tuple((positions[:, d].min() - 1.0, positions[:, d].max() + 1.0) for d in range(3))

        pos_kernel = make_default_3d_position_kernel(sigma=BONSAI_GAP_SIGMA)
        dir_kernel = DirectionalKernel(kappa=BONSAI_GAP_KAPPA)
        engine = LocalUncertaintyEngine(
            positions=positions, values=values, pos_kernel=pos_kernel, scene_bounds=bounds3d,
            directions=directions, dir_kernel=dir_kernel,
            opacities=obs_opacities, scales=obs_scales, rotations=obs_rotations,
        )

        camera = synthetic_camera_for_query(scene, query_point, query_direction)
        camera_index = engine.build_bearing_index(camera)
        dir_result = engine.rendering_aware_variance_along_ray_directional(
            query_point, query_direction, camera_index, BONSAI_GAP_WINDOW_RADIUS
        )
        spatial_result = engine.rendering_aware_variance(query_point, BONSAI_GAP_WINDOW_RADIUS)
        directional_vars.append(dir_result.variance)
        spatial_vars.append(spatial_result.variance)

    directional_vars = np.array(directional_vars)
    spatial_vars = np.array(spatial_vars)
    order = np.argsort(gap_half_widths)

    header = f"{'gap (deg)':>10}{'n_views':>9}"
    if overall_psnrs is not None:
        header += f"{'overall PSNR':>14}{'in-gap PSNR':>14}{'object PSNR':>14}"
    header += f"{'directional var':>18}{'spatial-only var':>18}"
    print(f"\n{header}")
    for i in order:
        row = f"{gap_half_widths[i]:>10.1f}{n_views[i]:>9d}"
        if overall_psnrs is not None:
            obj = object_psnrs[i] if object_psnrs is not None else float("nan")
            row += f"{overall_psnrs[i]:>14.2f}{gap_psnrs[i]:>14.2f}{obj:>14.2f}"
        row += f"{directional_vars[i]:>18.5f}{spatial_vars[i]:>18.5f}"
        print(row)

    dir_sorted = directional_vars[order]
    is_monotonic = bool(np.all(np.diff(dir_sorted) >= -1e-12))
    rho = float(np.corrcoef(np.argsort(np.argsort(gap_half_widths)), np.argsort(np.argsort(directional_vars)))[0, 1])
    print(f"\ndirectional variance monotonically increasing with gap width: {is_monotonic}")
    print(f"rank correlation (gap width vs. directional variance): rho={rho:.3f}")
    print(f"directional variance range (widest/narrowest): {dir_sorted[-1] / max(dir_sorted[0], 1e-12):.2f}x")
    print(f"spatial-only variance range (control): {spatial_vars.max() / max(spatial_vars.min(), 1e-12):.2f}x")
    if overall_psnrs is not None:
        print(
            f"overall held-out PSNR range across conditions: {overall_psnrs.min():.2f}-{overall_psnrs.max():.2f}dB "
            f"(quality bar: {BONSAI_GAP_MIN_PSNR}dB)"
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(gap_half_widths[order], directional_vars[order], "o-", color="tab:red", label="directional variance")
    ax1.set_xlabel("gap half-width around query direction (deg)")
    ax1.set_ylabel("position+direction BQ variance", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.plot(gap_half_widths[order], spatial_vars[order], "s--", color="tab:blue", label="spatial-only variance (control)")
    ax2.set_ylabel("position-only BQ variance", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    fig.suptitle("Real photographed capture (Mip-NeRF360 bonsai): directional BQ variance\nvs. a deliberate real coverage gap")
    fig.tight_layout()
    out_path = RESULTS_DIR / "gap_capture_directional_gradient.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_path}")


def bonsai_gap_run(
    out_dir: str, gap_half_widths=BONSAI_GAP_HALF_WIDTHS_DEG, condition_prefix: str = "gap",
    train_kwargs=BONSAI_GAP_TRAIN_KWARGS, check_quality: bool = True,
):
    camera_angle_x, frames = bonsai_gap_load_pool(out_dir)
    centers = np.array([c2w[:3, 3] for _, c2w in frames])
    margin = 1.0
    bounds = tuple((float(centers[:, d].min() - margin), float(centers[:, d].max() + margin)) for d in range(3))
    print(f"scene bounds (from camera-center extent + margin): {bounds}")

    zone_dirs, gap_half_widths, n_views, query_direction, eval_dir = bonsai_gap_build_conditions(
        out_dir, camera_angle_x, frames, gap_half_widths=gap_half_widths, condition_prefix=condition_prefix,
    )
    bonsai_gap_train_zones(zone_dirs, bounds, train_kwargs=train_kwargs)
    overall_psnrs = gap_psnrs = object_psnrs = None
    if check_quality:
        overall_psnrs, gap_psnrs, object_psnrs = bonsai_gap_check_reconstruction_quality(
            zone_dirs, eval_dir, gap_half_widths, query_direction, query_point=BONSAI_GAP_QUERY_POINT
        )
    bonsai_gap_analyze(
        zone_dirs, gap_half_widths, n_views, query_direction,
        overall_psnrs=overall_psnrs, gap_psnrs=gap_psnrs, object_psnrs=object_psnrs,
    )


# =====================================================================
# dispatcher
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["lego", "bonsai"], required=True)
    parser.add_argument(
        "dirs", nargs="+",
        help=(
            "positional directory args, meaning depends on --dataset: "
            "lego: <prepared_dir>. "
            "bonsai: <out_dir> (already-prepared, must have all_transforms.json)."
        ),
    )
    parser.add_argument("--condition-prefix", default=None)
    parser.add_argument("--gap-half-widths", type=float, nargs="+", default=None)
    parser.add_argument("--n-iters", type=int, default=None)
    parser.add_argument("--max-splats", type=int, default=None, help="lego only")
    parser.add_argument("--n-splats", type=int, default=None)
    args = parser.parse_args()

    if args.dataset == "lego":
        if len(args.dirs) != 1:
            parser.error("--dataset lego expects exactly 1 positional dir arg: <prepared_dir>")
        train_kwargs = dict(LEGO_GAP_TRAIN_KWARGS)
        if args.n_iters is not None:
            train_kwargs["n_iters"] = args.n_iters
        if args.max_splats is not None:
            train_kwargs["max_splats"] = args.max_splats
        if args.n_splats is not None:
            train_kwargs["n_splats"] = args.n_splats
        lego_gap_run(
            args.dirs[0],
            gap_half_widths=args.gap_half_widths if args.gap_half_widths is not None else LEGO_GAP_HALF_WIDTHS_DEG,
            condition_prefix=args.condition_prefix if args.condition_prefix is not None else "gap",
            train_kwargs=train_kwargs,
        )
    elif args.dataset == "bonsai":
        if len(args.dirs) != 1:
            parser.error("--dataset bonsai expects exactly 1 positional dir arg: <out_dir>")
        train_kwargs = dict(BONSAI_GAP_TRAIN_KWARGS)
        if args.n_iters is not None:
            train_kwargs["n_iters"] = args.n_iters
        if args.n_splats is not None:
            train_kwargs["n_splats"] = args.n_splats
        bonsai_gap_run(
            args.dirs[0],
            gap_half_widths=args.gap_half_widths if args.gap_half_widths is not None else BONSAI_GAP_HALF_WIDTHS_DEG,
            condition_prefix=args.condition_prefix if args.condition_prefix is not None else "gap",
            train_kwargs=train_kwargs,
        )


if __name__ == "__main__":
    main()
