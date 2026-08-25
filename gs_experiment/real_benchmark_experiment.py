"""Real-benchmark version of the differentiation experiment: does
position-only BQ variance flag genuinely thin/fine structure as more
uncertain than thick/simple structure, on a real, standardized benchmark
scene (NeRF-Synthetic's "lego", prepared via prepare_nerf_synthetic.py --
not a hand-built rod cluster like differentiation_experiment.py's scene)?

"Thin" and "thick" regions aren't hand-annotated (no 3D ground-truth
labels exist for this off-the-shelf object) -- they're identified
automatically from a trained checkpoint's own per-splat scale, which is
already the natural GS-native proxy for local feature size (small-scale
splats accumulate on fine detail; large-scale splats accumulate on flat,
simple surfaces -- this is exactly the signal `train_minimal_gsplat`'s
own scale parameter is optimized against). This generalizes to any real
scene, not just this one: no manual annotation step is needed to define
"fine structure" the way the hand-built rod scene could just declare it
by construction.

Two comparisons, both on real reconstructed geometry:
  1. Within the wide (100-view) checkpoint: thin-region vs. thick-region
     BQ position-only variance -- the core milestone-2 claim, now on real
     complex geometry instead of synthetic thin rods.
  2. Cross-checkpoint: wide (100-view) vs. narrow (12-view) BQ variance
     at matching thin-region query points -- the observation-count sanity
     check already established at toy scale, now checked on real data.

Needs torch + gsplat (requirements-gsplat.txt) only insofar as the
checkpoints were already trained by train_minimal_gsplat.py -- this
script itself is pure numpy/scipy (bq_splat, pixel_uncertainty).

Run: .venv-gsplat/bin/python gs_experiment/real_benchmark_experiment.py <lego_prepared_dir>
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from gs_experiment.pixel_uncertainty import LocalUncertaintyEngine, make_default_3d_position_kernel
from gs_experiment.ply_io import read_3dgs_ply


def load_checkpoint_engine(ply_path: str, sigma: float, min_opacity: float = 0.1):
    ck = read_3dgs_ply(ply_path)
    keep = ck["opacities"] > min_opacity
    positions = ck["positions"][keep]
    scales = ck["scales"][keep]
    colors = ck["sh_coeffs"][keep, :, 0].mean(axis=1)  # flat DC-term proxy, values don't affect variance anyway

    bounds = tuple((positions[:, d].min() - 0.3, positions[:, d].max() + 0.3) for d in range(3))
    pos_kernel = make_default_3d_position_kernel(sigma=sigma)
    engine = LocalUncertaintyEngine(positions=positions, values=colors, pos_kernel=pos_kernel, scene_bounds=bounds)

    median_scale = np.median(scales, axis=1)  # one feature-size proxy per splat
    return engine, positions, median_scale


def sample_query_points(positions, median_scale, n_samples, quantile, rng, low=True):
    """Query points drawn from the splats in the bottom (`low=True`) or
    top (`low=False`) `quantile` of their own median scale -- i.e. the
    finest or thickest reconstructed structure, per-splat, not per-pixel
    or per hand-picked region."""
    threshold = np.quantile(median_scale, quantile if low else 1 - quantile)
    mask = median_scale <= threshold if low else median_scale >= threshold
    idx = np.where(mask)[0]
    chosen = rng.choice(idx, size=min(n_samples, len(idx)), replace=False)
    return positions[chosen]


def run(lego_dir: str, n_samples: int = 60, quantile: float = 0.2, sigma: float = 0.15, window_radius: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)

    wide_ply = os.path.join(lego_dir, "wide", "splats.ply")
    narrow_ply = os.path.join(lego_dir, "narrow", "splats.ply")

    wide_engine, wide_pos, wide_scale = load_checkpoint_engine(wide_ply, sigma=sigma)
    narrow_engine, narrow_pos, narrow_scale = load_checkpoint_engine(narrow_ply, sigma=sigma)

    print(f"wide checkpoint: {len(wide_pos)} splats above opacity floor, median scale {np.median(wide_scale):.4f}")
    print(f"narrow checkpoint: {len(narrow_pos)} splats above opacity floor, median scale {np.median(narrow_scale):.4f}")

    thin_points = sample_query_points(wide_pos, wide_scale, n_samples, quantile, rng, low=True)
    thick_points = sample_query_points(wide_pos, wide_scale, n_samples, quantile, rng, low=False)

    def mean_variance(engine, points):
        variances = [engine.spatial_only_variance(p, window_radius).variance for p in points]
        return float(np.mean(variances)), variances

    print("\n=== 1. within the wide checkpoint: thin vs. thick real structure ===")
    thin_wide_mean, _ = mean_variance(wide_engine, thin_points)
    thick_wide_mean, _ = mean_variance(wide_engine, thick_points)
    print(f"thin-region (bottom {quantile:.0%} scale) BQ position-only variance:  {thin_wide_mean:.5f}")
    print(f"thick-region (top {quantile:.0%} scale) BQ position-only variance:    {thick_wide_mean:.5f}")
    print(f"ratio (thin/thick): {thin_wide_mean / thick_wide_mean:.2f}x")

    print("\n=== 2. cross-checkpoint at the same thin-region query points: wide (100-view) vs. narrow (12-view) ===")
    thin_wide_mean2, _ = mean_variance(wide_engine, thin_points)
    thin_narrow_mean, _ = mean_variance(narrow_engine, thin_points)
    print(f"wide checkpoint BQ variance at thin points:   {thin_wide_mean2:.5f}")
    print(f"narrow checkpoint BQ variance at thin points: {thin_narrow_mean:.5f}")
    print(f"ratio (narrow/wide): {thin_narrow_mean / thin_wide_mean2:.2f}x")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lego_dir")
    parser.add_argument("--n-samples", type=int, default=60)
    parser.add_argument("--quantile", type=float, default=0.2)
    parser.add_argument("--sigma", type=float, default=0.15)
    parser.add_argument("--window-radius", type=float, default=0.2)
    args = parser.parse_args()
    run(args.lego_dir, n_samples=args.n_samples, quantile=args.quantile, sigma=args.sigma, window_radius=args.window_radius)


if __name__ == "__main__":
    main()
