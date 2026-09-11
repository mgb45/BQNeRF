"""Builds the real gsplat checkpoints render_coverage_uncertainty_sweep.py renders:
starting from lego's full 100-view training pool, remove a single deliberate
angular gap of increasing half-width around one reference view's direction,
leaving every other view in the pool untouched
(`prepare_nerf_synthetic.select_gap_subset`), and train one checkpoint per gap
condition.

An earlier "subsample" design (hold view count fixed, vary the angular width
views are drawn from) was tried first and is retired -- it confounded angular
spread with global view thinning (see FINDINGS.md section 4); the gap design
here avoids that and is what the coverage-sweep figure uses. See git history
to resurrect the subsample design/code, and for the retired bonsai
(real-COLMAP-capture) variant of this same gap design.

Needs torch + gsplat (requirements-gsplat.txt).

Run: .venv-gsplat/bin/python gs_experiment/scripts/real_directional_coverage_experiment.py \\
    gs_experiment/local_runs/lego_prepared
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from gs_experiment.nerf_transforms import load_transforms
from gs_experiment.scripts.prepare_nerf_synthetic import select_gap_subset, write_condition
from gs_experiment.scripts.render_reconstruction import render_views
from gs_experiment.scripts.train_minimal_gsplat import train

REFERENCE_IDX = 0

LEGO_GAP_HALF_WIDTHS_DEG = [0.0, 15.0, 30.0, 50.0, 75.0]

LEGO_GAP_TRAIN_KWARGS = dict(
    # Matches train_minimal_gsplat.DEFAULT_TRAIN_KWARGS, the recipe that gives every
    # other NeRF-Synthetic scene in this project real reconstruction quality (ROADMAP.md
    # item 1's quality-bar fix) -- background_color=(1,1,1) matters most: without it this
    # dict silently fell back to train()'s dark (0.05,0.05,0.05) default while
    # prepare_nerf_synthetic.py composites onto white, a real background mismatch that
    # alone produced a degenerate, hazy reconstruction at every gap condition.
    n_splats=5000, bounds=((-2.5, 2.5), (-2.5, 2.5), (-2.5, 2.5)), sh_degree=3, n_iters=30000, seed=0,
    init_scale=0.05, opacity_reg_weight=0.01, densify=True, densify_interval=100, densify_start=500,
    densify_end=15000, min_opacity=0.005, max_splats=300000, log_every=2000, position_lr_final=2e-5,
    background_color=(1.0, 1.0, 1.0),
)
# Marginal-likelihood-fitted (hyperparams.py::fit_kernel_param_pooled_nd), pooled across the
# 5 real lego-gap checkpoints this experiment itself produces -- one shared sigma across gap
# conditions is deliberate (a different sigma per condition would confound "does variance grow
# with the gap" with "did the bandwidth happen to fit differently this time"), not an oversight.
# This value (2025-09) is a refit of an earlier pooled fit after a real bug was found and fixed:
# `SplatScene.colors` (splat_scene.py) was the raw SH-DC coefficient, not a real color (3DGS
# stores SH coefficients as offsets from mid-gray, real_color = SH_C0*raw + 0.5) -- see
# gs_experiment/results/FINDINGS.md section 4 for the full story and the fitted-bandwidth deltas
# this produced project-wide (roughly 1.7x-2.0x on every checkpoint spot-checked). The prior
# value here (0.0694, pooled across 9 checkpoints including the gallery figure's, before that
# bug was found) replaced an even older, more badly wrong sigma=0.9/window_radius=1.6 pair
# (held-out log marginal likelihood -35.8 MILLION vs -8607 at the fitted value) -- see git
# history for that story. This refit pools only the 5 gap checkpoints (not also the gallery
# checkpoints, unlike the prior value) since that's a cleaner, directly-reproducible set and
# render_scene_gallery.py now fits its own sigma per-checkpoint at runtime anyway (see that
# script; it no longer shares a hardcoded constant with this one).
LEGO_GAP_SIGMA = 0.13926
LEGO_GAP_WINDOW_RADIUS = 0.08
# Real homoscedastic observation-noise variance, jointly fit alongside a
# bandwidth via hyperparams.fit_kernel_param_and_noise_pooled_nd (see that
# function's docstring, and gs_experiment.quadrature._rendering_aware_moments
# for the full model/motivation -- real splat positions routinely include
# near-duplicate points that force a noiseless fit toward an artificially
# short bandwidth), pooled across real local (position, color) windows
# gathered via kernel_family_ablation.sample_sigma_windows(window_radius=
# LEGO_GAP_WINDOW_RADIUS) from the SAME 5 real lego-gap checkpoints
# LEGO_GAP_SIGMA above is pooled across (125 windows total, 25 per
# checkpoint). This joint fit's own bandwidth (0.6121) differs substantially
# from LEGO_GAP_SIGMA's noiseless one (0.1393) -- not a contradiction, the
# noiseless fit is forced short specifically to keep exactly interpolating
# through near-duplicate points, which this fit no longer needs to do -- and
# is NOT used in place of LEGO_GAP_SIGMA anywhere (every existing caller of
# LEGO_GAP_SIGMA/LEGO_GAP_KAPPA is left exactly as before); this noise
# variance is instead layered ON TOP of the existing noiseless LEGO_GAP_SIGMA
# fit's Gram-matrix diagonal by callers that opt in (see
# render_coverage_uncertainty_sweep.py's `noise_variance` argument and
# likelihood_training_experiment.py's `LEGO_BQ_NOISE_VARIANCE` alias). Real
# marginal-likelihood gain from adding noise: log marginal likelihood
# 5230.79 (noiseless) -> 7704.97 (noise-aware), +2474.17 units, not a close
# call -- see gs_experiment/results/FINDINGS.md's noise-variance section.
LEGO_GAP_NOISE_VARIANCE = 0.005532
# Also marginal-likelihood-fitted (DirectionalKernel via fit_kernel_param_pooled_nd), pooled
# across real per-splat multi-view (direction, color) observations from 7 real checkpoints.
# Held-out log marginal likelihood 1413 at this value vs 202 at the old hardcoded 4.0 -- that
# value was ~5x too concentrated (real per-splat color varies less with viewing angle than it
# implied), which would have made every gap condition look more under-covered than the real
# data supports. Matches render_reconstruction.py's compute_uncertainty_maps default exactly.
LEGO_GAP_KAPPA = 0.745
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


def lego_gap_run(
    prepared_dir: str,
    gap_half_widths=LEGO_GAP_HALF_WIDTHS_DEG,
    condition_prefix: str = "gap",
    train_kwargs=LEGO_GAP_TRAIN_KWARGS,
    check_quality: bool = True,
):
    zone_dirs, gap_half_widths, _, query_direction = lego_gap_build_conditions(
        prepared_dir, gap_half_widths=gap_half_widths, condition_prefix=condition_prefix,
    )
    lego_gap_train_zones(zone_dirs, train_kwargs=train_kwargs)
    if check_quality:
        eval_dir = os.path.join(prepared_dir, "eval")
        if os.path.exists(eval_dir):
            lego_gap_check_reconstruction_quality(zone_dirs, eval_dir, gap_half_widths, query_direction)
        else:
            print(f"no eval/ split found at {eval_dir}, skipping reconstruction-quality check")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prepared_dir")
    parser.add_argument("--condition-prefix", default="gap")
    parser.add_argument("--gap-half-widths", type=float, nargs="+", default=None)
    parser.add_argument("--n-iters", type=int, default=None)
    parser.add_argument("--max-splats", type=int, default=None)
    parser.add_argument("--n-splats", type=int, default=None)
    args = parser.parse_args()

    train_kwargs = dict(LEGO_GAP_TRAIN_KWARGS)
    if args.n_iters is not None:
        train_kwargs["n_iters"] = args.n_iters
    if args.max_splats is not None:
        train_kwargs["max_splats"] = args.max_splats
    if args.n_splats is not None:
        train_kwargs["n_splats"] = args.n_splats
    lego_gap_run(
        args.prepared_dir,
        gap_half_widths=args.gap_half_widths if args.gap_half_widths is not None else LEGO_GAP_HALF_WIDTHS_DEG,
        condition_prefix=args.condition_prefix,
        train_kwargs=train_kwargs,
    )


if __name__ == "__main__":
    main()
