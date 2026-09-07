"""Consolidated CLI for the four real-geometry directional-uncertainty
coverage experiments that used to be separate one-off scripts:

  * real_directional_gradient_experiment.py  (--design subsample --dataset lego)
  * real_capture_gradient_experiment.py      (--design subsample --dataset bonsai)
  * gap_directional_experiment.py            (--design gap       --dataset lego)
  * gap_capture_experiment.py                (--design gap       --dataset bonsai)

Each of the four combinations trains its own set of real gsplat
checkpoints and queries directional vs. position-only BQ variance across
them, differing only in *dataset* (synthetic NeRF-Synthetic "lego" vs. a
genuinely photographed, COLMAP-posed Mip-NeRF360 "bonsai" capture) and
*design*:

  - "subsample": hold total view count fixed and vary the angular width
    of the window views are drawn from around a shared reference view
    (`prepare_nerf_synthetic.select_gradient_subset`).
  - "gap": start from the full view pool and remove a single deliberate
    angular gap of increasing half-width around one reference view's
    direction, leaving every other view in the pool untouched
    (`prepare_nerf_synthetic.select_gap_subset`) -- designed to avoid the
    "spread confounded with global view thinning" issue the subsample
    design ran into (see FINDINGS.md section 4).

gap_directional_experiment.py and gap_capture_experiment.py share the
same gap-design methodology but target different datasets (lego vs.
bonsai) and diverge slightly in maturity: the bonsai gap-design functions
here are ported from gap_capture_experiment.py (the more recent,
more-correct iteration -- gsplat's own reference densification strategy,
its own carved held-out eval split, object-region PSNR), with
gap_directional_experiment.py's own CLI arg name kept where
gap_capture_experiment.py had no equivalent (--condition-prefix).

Needs torch + gsplat (requirements-gsplat.txt); the bonsai modes also
need PIL (image resizing) and colmap_loader.

Run examples:
  .venv-gsplat/bin/python gs_experiment/real_directional_coverage_experiment.py \\
      --design subsample --dataset lego gs_experiment/local_runs/lego_prepared

  .venv-gsplat/bin/python gs_experiment/real_directional_coverage_experiment.py \\
      --design subsample --dataset bonsai \\
      gs_experiment/local_runs/mipnerf360_raw/bonsai gs_experiment/local_runs/bonsai_prepared

  .venv-gsplat/bin/python gs_experiment/real_directional_coverage_experiment.py \\
      --design gap --dataset lego gs_experiment/local_runs/lego_prepared

  .venv-gsplat/bin/python gs_experiment/real_directional_coverage_experiment.py \\
      --design gap --dataset bonsai gs_experiment/local_runs/bonsai_prepared
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

from bq_splat.kernels import DirectionalKernel
from gs_experiment.colmap_loader import load_colmap_scene
from gs_experiment.nerf_transforms import (
    fov_x_to_intrinsics,
    load_transforms,
    opencv_viewmat_from_c2w,
    write_transforms_json,
)
from gs_experiment.pixel_uncertainty import LocalUncertaintyEngine, make_default_3d_position_kernel
from gs_experiment.prepare_nerf_synthetic import select_gap_subset, select_gradient_subset, write_condition
from gs_experiment.render_reconstruction import render_views
from gs_experiment.splat_scene import load_from_gsplat_checkpoint, splat_observations
from gs_experiment.train_minimal_gsplat import train, train_with_reference_strategy

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_IDX = 0


# =====================================================================
# --design subsample --dataset lego  (real_directional_gradient_experiment.py)
# =====================================================================
LEGO_GRADIENT_WINDOW_FRACTIONS = [0.06, 0.15, 0.35, 0.6, 1.0]
LEGO_GRADIENT_N_PER_ZONE = 6

LEGO_GRADIENT_TRAIN_KWARGS = dict(
    n_splats=1500, bounds=((-2.5, 2.5), (-2.5, 2.5), (-2.5, 2.5)), sh_degree=1, n_iters=2500, seed=0,
    init_scale=0.05, opacity_reg_weight=0.01, densify=True, densify_interval=300, densify_start=300,
    min_opacity=0.005, max_splats=8000, log_every=500,
)
LEGO_GRADIENT_SIGMA = 0.05
LEGO_GRADIENT_WINDOW_RADIUS = 0.5
LEGO_GRADIENT_KAPPA = 4.0


def lego_gradient_build_conditions(
    prepared_dir: str,
    n_per_zone: int = LEGO_GRADIENT_N_PER_ZONE,
    window_fractions=LEGO_GRADIENT_WINDOW_FRACTIONS,
    condition_prefix: str = "gradient",
):
    wide_dir = os.path.join(prepared_dir, "wide")
    camera_angle_x, frames = load_transforms(os.path.join(wide_dir, "transforms.json"))

    centers = np.array([c2w[:3, 3] for _, c2w in frames])
    dirs = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    ref_dir = dirs[REFERENCE_IDX]
    query_direction = dirs[np.argmin(dirs @ ref_dir)]  # real, most-dissimilar-to-reference view direction

    zone_dirs = []
    spreads = []
    for i, window_fraction in enumerate(window_fractions):
        idx = select_gradient_subset(frames, n_per_zone=n_per_zone, window_fraction=window_fraction, reference_idx=REFERENCE_IDX)
        zone_dir = write_condition(prepared_dir, camera_angle_x, frames, idx, f"{condition_prefix}_{i}", "train")
        zone_dirs.append(zone_dir)
        # real measured angular spread of this zone's selected views (not
        # the arbitrary window_fraction knob): min cosine similarity to
        # the reference view among the views actually selected.
        min_sim = float((dirs[idx] @ ref_dir).min())
        spreads.append(np.degrees(np.arccos(np.clip(min_sim, -1.0, 1.0))))

    return zone_dirs, np.array(spreads), query_direction


def lego_gradient_train_zones(zone_dirs, train_kwargs=LEGO_GRADIENT_TRAIN_KWARGS):
    ply_paths = []
    for zone_dir in zone_dirs:
        ply_path = os.path.join(zone_dir, "splats.ply")
        if not os.path.exists(ply_path):
            print(f"training {zone_dir}...")
            train(zone_dir, ply_path, **train_kwargs)
        else:
            print(f"reusing existing checkpoint at {ply_path}")
        ply_paths.append(ply_path)
    return ply_paths


def lego_gradient_check_reconstruction_quality(zone_dirs, eval_dir: str):
    """Real held-out PSNR per condition, against the shared official-
    test-split eval set every lego_prepared/bonsai_prepared directory
    already has -- computed and reported *before* trusting any BQ number
    built on top of these checkpoints, closing exactly the gap that made
    the first real-scene attempt's result uninterpretable (FINDINGS.md
    section 37): a BQ variance computed on a checkpoint nobody checked
    could actually reconstruct the scene."""
    _, eval_frames = load_transforms(os.path.join(eval_dir, "transforms.json"))
    n_eval = len(eval_frames)

    psnrs = []
    for zone_dir in zone_dirs:
        eval_copy_dir = zone_dir + "_eval"
        os.makedirs(eval_copy_dir, exist_ok=True)
        shutil.copy(os.path.join(eval_dir, "transforms.json"), os.path.join(eval_copy_dir, "transforms.json"))
        images_link = os.path.join(eval_copy_dir, "test")
        if not os.path.exists(images_link):
            os.symlink(os.path.abspath(os.path.join(eval_dir, "test")), images_link)
        shutil.copy(os.path.join(zone_dir, "splats.ply"), os.path.join(eval_copy_dir, "splats.ply"))

        results, _ = render_views(eval_copy_dir, list(range(n_eval)))
        psnr = float(np.mean([-10.0 * np.log10(max(float(np.mean((gt - recon) ** 2)), 1e-10)) for _, gt, recon in results]))
        psnrs.append(psnr)
        print(f"{zone_dir}: held-out PSNR over {n_eval} eval views = {psnr:.2f}dB")

    return np.array(psnrs)


def lego_gradient_analyze(zone_dirs, spreads_deg, query_direction, psnrs=None):
    directional_vars, spatial_vars = [], []
    query_point = np.zeros(3)

    for zone_dir in zone_dirs:
        scene = load_from_gsplat_checkpoint(zone_dir, attribution_angular_tol=0.01)
        positions, directions, values = splat_observations(scene)
        bounds = tuple((positions[:, d].min() - 1.0, positions[:, d].max() + 1.0) for d in range(3))

        pos_kernel = make_default_3d_position_kernel(sigma=LEGO_GRADIENT_SIGMA)
        dir_kernel = DirectionalKernel(kappa=LEGO_GRADIENT_KAPPA)
        engine = LocalUncertaintyEngine(
            positions=positions, values=values, pos_kernel=pos_kernel, scene_bounds=bounds,
            directions=directions, dir_kernel=dir_kernel,
        )

        dir_result = engine.directional_variance(query_point, query_direction, LEGO_GRADIENT_WINDOW_RADIUS)
        spatial_result = engine.spatial_only_variance(query_point, LEGO_GRADIENT_WINDOW_RADIUS)
        directional_vars.append(dir_result.variance)
        spatial_vars.append(spatial_result.variance)

    directional_vars = np.array(directional_vars)
    spatial_vars = np.array(spatial_vars)

    order = np.argsort(spreads_deg)
    if psnrs is not None:
        print(f"\n{'zone':>4}{'spread (deg)':>14}{'held-out PSNR':>15}{'directional var':>18}{'spatial-only var':>18}")
        for i in order:
            print(f"{i:>4}{spreads_deg[i]:>14.1f}{psnrs[i]:>15.2f}{directional_vars[i]:>18.5f}{spatial_vars[i]:>18.5f}")
    else:
        print(f"\n{'zone':>4}{'spread (deg)':>14}{'directional var':>18}{'spatial-only var':>18}")
        for i in order:
            print(f"{i:>4}{spreads_deg[i]:>14.1f}{directional_vars[i]:>18.5f}{spatial_vars[i]:>18.5f}")

    dir_sorted = directional_vars[order]
    is_monotonic = bool(np.all(np.diff(dir_sorted) <= 1e-12))
    rho = float(np.corrcoef(np.argsort(np.argsort(spreads_deg)), np.argsort(np.argsort(-directional_vars)))[0, 1])
    print(f"\ndirectional variance monotonically decreasing as spread widens: {is_monotonic}")
    print(f"rank correlation (spread vs. directional variance): rho={rho:.3f}")
    print(f"directional variance range (narrowest/widest): {dir_sorted[0] / max(dir_sorted[-1], 1e-12):.2f}x")
    print(f"spatial-only variance range (control): {spatial_vars.max() / max(spatial_vars.min(), 1e-12):.2f}x")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(spreads_deg[order], directional_vars[order], "o-", color="tab:red", label="directional variance")
    ax1.set_xlabel("real angular spread of selected views (deg, min similarity to reference)")
    ax1.set_ylabel("position+direction BQ variance", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.plot(spreads_deg[order], spatial_vars[order], "s--", color="tab:blue", label="spatial-only variance (control)")
    ax2.set_ylabel("position-only BQ variance", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    fig.suptitle("Real NeRF-Synthetic (lego): directional BQ variance vs. real view-coverage spread")
    fig.tight_layout()
    out_path = RESULTS_DIR / "real_directional_gradient.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_path}")


def lego_gradient_run(
    prepared_dir: str,
    n_per_zone: int = LEGO_GRADIENT_N_PER_ZONE,
    window_fractions=LEGO_GRADIENT_WINDOW_FRACTIONS,
    condition_prefix: str = "gradient",
    train_kwargs=LEGO_GRADIENT_TRAIN_KWARGS,
    check_quality: bool = True,
):
    zone_dirs, spreads_deg, query_direction = lego_gradient_build_conditions(
        prepared_dir, n_per_zone=n_per_zone, window_fractions=window_fractions, condition_prefix=condition_prefix,
    )
    lego_gradient_train_zones(zone_dirs, train_kwargs=train_kwargs)
    psnrs = None
    if check_quality:
        eval_dir = os.path.join(prepared_dir, "eval")
        if os.path.exists(eval_dir):
            psnrs = lego_gradient_check_reconstruction_quality(zone_dirs, eval_dir)
        else:
            print(f"no eval/ split found at {eval_dir}, skipping reconstruction-quality check")
    lego_gradient_analyze(zone_dirs, spreads_deg, query_direction, psnrs=psnrs)


# =====================================================================
# --design subsample --dataset bonsai  (real_capture_gradient_experiment.py)
# =====================================================================
BONSAI_GRADIENT_WINDOW_FRACTIONS = [0.06, 0.15, 0.35, 0.6, 1.0]
BONSAI_GRADIENT_N_PER_ZONE = 8
BONSAI_GRADIENT_TARGET_WIDTH = 400

BONSAI_GRADIENT_SIGMA = 0.05
BONSAI_GRADIENT_WINDOW_RADIUS = 0.8
BONSAI_GRADIENT_KAPPA = 4.0
# World origin (COLMAP's own SfM-chosen frame) is NOT reliably where the
# photographed object actually sits -- checked directly, not assumed: the
# nearest real splat to the origin was consistently ~0.8-0.9 units away
# with only a handful of real neighbors within a 0.3 window, in every one
# of the 5 checkpoints. A camera-center centroid near the origin (true
# here) does not imply the object itself is there. BONSAI_GRADIENT_QUERY_POINT
# below is the mean, across all 5 checkpoints, of each checkpoint's own
# median high-opacity splat position -- a real, data-derived "where the
# object actually is" estimate, not assumed from dataset convention the
# way NeRF-Synthetic's object-centered origin could be.
BONSAI_GRADIENT_QUERY_POINT = np.array([0.612, 1.174, 1.507])

BONSAI_GRADIENT_TRAIN_KWARGS = dict(
    n_splats=2000, sh_degree=1, n_iters=3000, seed=0,
    init_scale=0.03, opacity_reg_weight=0.01, densify=True, densify_interval=300, densify_start=300,
    min_opacity=0.005, max_splats=15000, log_every=500,
)


def bonsai_gradient_prepare_images(colmap_scene_dir: str, out_dir: str):
    camera_angle_x, frames = load_colmap_scene(os.path.join(colmap_scene_dir, "sparse", "0"))
    src_images_dir = os.path.join(colmap_scene_dir, "images")

    all_dir = os.path.join(out_dir, "all")
    os.makedirs(all_dir, exist_ok=True)

    kept_frames = []
    for file_stem, c2w in frames:
        src = os.path.join(src_images_dir, file_stem + ".JPG")
        if not os.path.exists(src):
            src = os.path.join(src_images_dir, file_stem + ".jpg")
        if not os.path.exists(src):
            continue
        im = Image.open(src).convert("RGB")
        w, h = im.size
        target_height = round(BONSAI_GRADIENT_TARGET_WIDTH * h / w)
        im = im.resize((BONSAI_GRADIENT_TARGET_WIDTH, target_height), Image.LANCZOS)
        dst = os.path.join(all_dir, file_stem + ".png")
        im.save(dst)
        kept_frames.append((f"all/{file_stem}", c2w))

    if len(kept_frames) < len(frames):
        print(f"{len(kept_frames)}/{len(frames)} frames had a matching image file (rest skipped)")

    write_transforms_json(
        os.path.join(out_dir, "all_transforms.json"), camera_angle_x,
        [{"file_path": fp, "transform_matrix": c2w} for fp, c2w in kept_frames],
    )
    return camera_angle_x, kept_frames


def bonsai_gradient_build_conditions(out_dir: str, camera_angle_x: float, frames):
    centers = np.array([c2w[:3, 3] for _, c2w in frames])
    dirs = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    print(f"camera center centroid (should be near origin): {centers.mean(axis=0)}")

    ref_dir = dirs[REFERENCE_IDX]
    query_direction = dirs[np.argmin(dirs @ ref_dir)]

    zone_dirs, spreads = [], []
    for i, window_fraction in enumerate(BONSAI_GRADIENT_WINDOW_FRACTIONS):
        idx = select_gradient_subset(
            frames, n_per_zone=BONSAI_GRADIENT_N_PER_ZONE, window_fraction=window_fraction, reference_idx=REFERENCE_IDX
        )
        # write_condition expects frame file_paths relative to out_dir and
        # symlinks split_prefix -- our images already live under
        # "all/", referenced via that same relative path in `frames`, so
        # split_prefix="all" points write_condition's symlink at the
        # directory that's already there.
        zone_dir = write_condition(out_dir, camera_angle_x, frames, idx, f"gradient_{i}", "all")
        zone_dirs.append(zone_dir)
        min_sim = float((dirs[idx] @ ref_dir).min())
        spreads.append(np.degrees(np.arccos(np.clip(min_sim, -1.0, 1.0))))

    return zone_dirs, np.array(spreads), query_direction


def bonsai_gradient_train_zones(zone_dirs, bounds):
    ply_paths = []
    for zone_dir in zone_dirs:
        ply_path = os.path.join(zone_dir, "splats.ply")
        if not os.path.exists(ply_path):
            print(f"training {zone_dir}...")
            train(zone_dir, ply_path, bounds=bounds, **BONSAI_GRADIENT_TRAIN_KWARGS)
        else:
            print(f"reusing existing checkpoint at {ply_path}")
        ply_paths.append(ply_path)
    return ply_paths


def bonsai_gradient_analyze(zone_dirs, spreads_deg, query_direction):
    directional_vars, spatial_vars, n_neighbors_list = [], [], []
    query_point = BONSAI_GRADIENT_QUERY_POINT

    for zone_dir in zone_dirs:
        scene = load_from_gsplat_checkpoint(zone_dir, attribution_angular_tol=0.01)
        positions, directions, values = splat_observations(scene)
        bounds3d = tuple((positions[:, d].min() - 1.0, positions[:, d].max() + 1.0) for d in range(3))

        pos_kernel = make_default_3d_position_kernel(sigma=BONSAI_GRADIENT_SIGMA)
        dir_kernel = DirectionalKernel(kappa=BONSAI_GRADIENT_KAPPA)
        engine = LocalUncertaintyEngine(
            positions=positions, values=values, pos_kernel=pos_kernel, scene_bounds=bounds3d,
            directions=directions, dir_kernel=dir_kernel,
        )
        n_neighbors_list.append(len(engine.local_neighbors(query_point, BONSAI_GRADIENT_WINDOW_RADIUS)))
        dir_result = engine.directional_variance(query_point, query_direction, BONSAI_GRADIENT_WINDOW_RADIUS)
        spatial_result = engine.spatial_only_variance(query_point, BONSAI_GRADIENT_WINDOW_RADIUS)
        directional_vars.append(dir_result.variance)
        spatial_vars.append(spatial_result.variance)

    directional_vars = np.array(directional_vars)
    spatial_vars = np.array(spatial_vars)

    order = np.argsort(spreads_deg)
    print(f"\n{'zone':>4}{'spread (deg)':>14}{'n_neighbors':>13}{'directional var':>18}{'spatial-only var':>18}")
    for i in order:
        print(f"{i:>4}{spreads_deg[i]:>14.1f}{n_neighbors_list[i]:>13}{directional_vars[i]:>18.5f}{spatial_vars[i]:>18.5f}")

    dir_sorted = directional_vars[order]
    is_monotonic = bool(np.all(np.diff(dir_sorted) <= 1e-12))
    rho = float(np.corrcoef(np.argsort(np.argsort(spreads_deg)), np.argsort(np.argsort(-directional_vars)))[0, 1])
    print(f"\ndirectional variance monotonically decreasing as spread widens: {is_monotonic}")
    print(f"rank correlation (spread vs. directional variance): rho={rho:.3f}")
    print(f"directional variance range (narrowest/widest): {dir_sorted[0] / max(dir_sorted[-1], 1e-12):.2f}x")
    print(f"spatial-only variance range (control): {spatial_vars.max() / max(spatial_vars.min(), 1e-12):.2f}x")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(spreads_deg[order], directional_vars[order], "o-", color="tab:red", label="directional variance")
    ax1.set_xlabel("real angular spread of selected views (deg, min similarity to reference)")
    ax1.set_ylabel("position+direction BQ variance", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(spreads_deg[order], spatial_vars[order], "s--", color="tab:blue", label="spatial-only variance (control)")
    ax2.set_ylabel("position-only BQ variance", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    fig.suptitle("Real captured scene (Mip-NeRF360 bonsai): directional BQ variance\nvs. real view-coverage spread")
    fig.tight_layout()
    out_path = RESULTS_DIR / "real_capture_directional_gradient.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_path}")


def bonsai_gradient_run(colmap_scene_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    all_transforms_path = os.path.join(out_dir, "all_transforms.json")
    if os.path.exists(all_transforms_path):
        print("reusing already-prepared images")
        camera_angle_x, frames = load_transforms(all_transforms_path)
    else:
        camera_angle_x, frames = bonsai_gradient_prepare_images(colmap_scene_dir, out_dir)

    centers = np.array([c2w[:3, 3] for _, c2w in frames])
    margin = 1.0
    bounds = tuple((float(centers[:, d].min() - margin), float(centers[:, d].max() + margin)) for d in range(3))
    print(f"scene bounds (from camera-center extent + margin): {bounds}")

    zone_dirs, spreads_deg, query_direction = bonsai_gradient_build_conditions(out_dir, camera_angle_x, frames)
    bonsai_gradient_train_zones(zone_dirs, bounds)
    bonsai_gradient_analyze(zone_dirs, spreads_deg, query_direction)


# =====================================================================
# --design gap --dataset lego  (gap_directional_experiment.py)
# =====================================================================
LEGO_GAP_HALF_WIDTHS_DEG = [0.0, 15.0, 30.0, 50.0, 75.0]

LEGO_GAP_TRAIN_KWARGS = dict(
    n_splats=3000, bounds=((-2.5, 2.5), (-2.5, 2.5), (-2.5, 2.5)), sh_degree=1, n_iters=10000, seed=0,
    init_scale=0.05, opacity_reg_weight=0.01, densify=True, densify_interval=300, densify_start=300,
    min_opacity=0.005, max_splats=35000, log_every=1000,
)
LEGO_GAP_SIGMA = 0.9
LEGO_GAP_WINDOW_RADIUS = 1.6
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


def lego_gap_analyze(zone_dirs, gap_half_widths, n_views, query_direction, overall_psnrs=None, gap_psnrs=None):
    directional_vars, spatial_vars = [], []
    query_point = np.zeros(3)

    for zone_dir in zone_dirs:
        scene = load_from_gsplat_checkpoint(zone_dir, attribution_angular_tol=0.01)
        positions, directions, values = splat_observations(scene)
        bounds = tuple((positions[:, d].min() - 1.0, positions[:, d].max() + 1.0) for d in range(3))

        pos_kernel = make_default_3d_position_kernel(sigma=LEGO_GAP_SIGMA)
        dir_kernel = DirectionalKernel(kappa=LEGO_GAP_KAPPA)
        engine = LocalUncertaintyEngine(
            positions=positions, values=values, pos_kernel=pos_kernel, scene_bounds=bounds,
            directions=directions, dir_kernel=dir_kernel,
        )

        dir_result = engine.directional_variance(query_point, query_direction, LEGO_GAP_WINDOW_RADIUS)
        spatial_result = engine.spatial_only_variance(query_point, LEGO_GAP_WINDOW_RADIUS)
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
# --design gap --dataset bonsai  (gap_capture_experiment.py, preferred as
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
# real splat. See bonsai_gradient_build_conditions above for the derivation.
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
        positions, directions, values = splat_observations(scene)
        bounds3d = tuple((positions[:, d].min() - 1.0, positions[:, d].max() + 1.0) for d in range(3))

        pos_kernel = make_default_3d_position_kernel(sigma=BONSAI_GAP_SIGMA)
        dir_kernel = DirectionalKernel(kappa=BONSAI_GAP_KAPPA)
        engine = LocalUncertaintyEngine(
            positions=positions, values=values, pos_kernel=pos_kernel, scene_bounds=bounds3d,
            directions=directions, dir_kernel=dir_kernel,
        )

        dir_result = engine.directional_variance(query_point, query_direction, BONSAI_GAP_WINDOW_RADIUS)
        spatial_result = engine.spatial_only_variance(query_point, BONSAI_GAP_WINDOW_RADIUS)
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
    parser.add_argument("--design", choices=["subsample", "gap"], required=True)
    parser.add_argument("--dataset", choices=["lego", "bonsai"], required=True)
    parser.add_argument(
        "dirs", nargs="+",
        help=(
            "positional directory args, meaning depends on --design/--dataset: "
            "subsample+lego: <prepared_dir>. "
            "subsample+bonsai: <colmap_scene_dir> <out_dir>. "
            "gap+lego: <prepared_dir>. "
            "gap+bonsai: <out_dir> (already-prepared, must have all_transforms.json)."
        ),
    )
    parser.add_argument("--n-per-zone", type=int, default=None, help="subsample+lego only")
    parser.add_argument("--window-fractions", type=float, nargs="+", default=None, help="subsample+lego only")
    parser.add_argument("--condition-prefix", default=None, help="subsample+lego, gap+lego, gap+bonsai")
    parser.add_argument("--gap-half-widths", type=float, nargs="+", default=None, help="gap+lego, gap+bonsai")
    parser.add_argument("--n-iters", type=int, default=None, help="subsample+lego, gap+lego, gap+bonsai")
    parser.add_argument("--max-splats", type=int, default=None, help="subsample+lego, gap+lego only")
    parser.add_argument("--n-splats", type=int, default=None, help="subsample+lego, gap+lego, gap+bonsai")
    args = parser.parse_args()

    if args.design == "subsample" and args.dataset == "lego":
        if len(args.dirs) != 1:
            parser.error("--design subsample --dataset lego expects exactly 1 positional dir arg: <prepared_dir>")
        train_kwargs = dict(LEGO_GRADIENT_TRAIN_KWARGS)
        if args.n_iters is not None:
            train_kwargs["n_iters"] = args.n_iters
        if args.max_splats is not None:
            train_kwargs["max_splats"] = args.max_splats
        if args.n_splats is not None:
            train_kwargs["n_splats"] = args.n_splats
        lego_gradient_run(
            args.dirs[0],
            n_per_zone=args.n_per_zone if args.n_per_zone is not None else LEGO_GRADIENT_N_PER_ZONE,
            window_fractions=args.window_fractions if args.window_fractions is not None else LEGO_GRADIENT_WINDOW_FRACTIONS,
            condition_prefix=args.condition_prefix if args.condition_prefix is not None else "gradient",
            train_kwargs=train_kwargs,
        )
    elif args.design == "subsample" and args.dataset == "bonsai":
        if len(args.dirs) != 2:
            parser.error("--design subsample --dataset bonsai expects exactly 2 positional dir args: <colmap_scene_dir> <out_dir>")
        bonsai_gradient_run(args.dirs[0], args.dirs[1])
    elif args.design == "gap" and args.dataset == "lego":
        if len(args.dirs) != 1:
            parser.error("--design gap --dataset lego expects exactly 1 positional dir arg: <prepared_dir>")
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
    elif args.design == "gap" and args.dataset == "bonsai":
        if len(args.dirs) != 1:
            parser.error("--design gap --dataset bonsai expects exactly 1 positional dir arg: <out_dir>")
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
    else:
        parser.error(f"unsupported --design/--dataset combination: {args.design}/{args.dataset}")


if __name__ == "__main__":
    main()
