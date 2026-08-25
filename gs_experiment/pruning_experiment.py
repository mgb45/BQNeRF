"""ROADMAP.md milestone 3: densification/pruning combination experiment.
"combination not competition": use BQ variance alongside a
opacity/visibility-based pruning criterion, and check whether the
combination reaches better reconstruction quality than the heuristic-only
baseline at the same, reduced splat count.

Post-hoc on an already-trained, already-densified checkpoint (no
retraining): prune down to a target splat count two ways --

  (a) opacity-only (the standard 3DGS heuristic: drop the lowest-opacity
      splats first)
  (b) opacity + BQ position-only variance, combined by rank: a splat with
      moderate-to-high opacity but high local BQ variance (evidence its
      neighborhood is still under-resolved) is protected from being
      pruned as readily as an opacity-matched splat in a low-BQ-variance
      (already well-resolved) region would be. The BQ term only applies
      above `min_opacity_for_bq` (default 0.3, calibrated empirically --
      see FINDINGS.md): BQ variance is *also* high in genuinely empty
      space (little/no local data, correctly but unhelpfully for pruning
      purposes), so applying it unconditionally protects near-zero-
      opacity junk at low keep-counts, which measurably hurt PSNR before
      this floor was added.

then render both pruned checkpoints and compare PSNR against ground
truth, at the *same* splat count -- the direct test of ROADMAP.md's
"reaches equal quality at fewer splats" framing (equivalently: better
quality at equal, reduced, splat count).

BQ variance is read from differentiation_experiment.py's cached 2D grid
(`<checkpoint>_grid_cache.npz`, produced by a differentiation_experiment.py
run against the same checkpoint) via interpolation, rather than
recomputing fresh per-splat BQ solves for up to 15000 splats -- orders of
magnitude cheaper, and precise enough for a splat-count-level comparison.
Run differentiation_experiment.py --checkpoint <scene_dir> first if the
cache doesn't exist yet.

Needs torch + gsplat (requirements-gsplat.txt).

Run: .venv-gsplat/bin/python gs_experiment/pruning_experiment.py <scene_dir> --keep-counts 4000 6000 9000
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from gs_experiment.ply_io import read_3dgs_ply, write_3dgs_ply
from gs_experiment.render_reconstruction import render_views

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_bq_interpolator(cache_path: str):
    data = np.load(cache_path)
    xs, ys, grid = data["xs"], data["ys"], data["spatial_grid"]
    # spatial_grid[j, i] corresponds to (xs[i], ys[j]) -- differentiation_
    # experiment.py's convention (imshow-style, row=y, col=x) -- so the
    # interpolator axes are (ys, xs) in that order to match.
    return RegularGridInterpolator((ys, xs), grid, bounds_error=False, fill_value=None)


def rank_score(values: np.ndarray) -> np.ndarray:
    """Ascending rank, normalized to [0, 1] -- higher value -> higher score."""
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values))
    return ranks / max(len(values) - 1, 1)


def prune_checkpoint(checkpoint: dict, keep_idx: np.ndarray) -> dict:
    return dict(
        positions=checkpoint["positions"][keep_idx],
        scales=checkpoint["scales"][keep_idx],
        rotations=checkpoint["rotations"][keep_idx],
        opacities=checkpoint["opacities"][keep_idx],
        sh_coeffs=checkpoint["sh_coeffs"][keep_idx],
        sh_degree=checkpoint["sh_degree"],
    )


def render_and_score(pruned_checkpoint: dict, scene_dir: str, view_indices, tmp_root: str, label: str):
    scene_copy = os.path.join(tmp_root, label)
    os.makedirs(scene_copy, exist_ok=True)
    write_3dgs_ply(os.path.join(scene_copy, "splats.ply"), **pruned_checkpoint)
    shutil.copy(os.path.join(scene_dir, "transforms.json"), os.path.join(scene_copy, "transforms.json"))
    if not os.path.exists(os.path.join(scene_copy, "images")):
        os.symlink(os.path.abspath(os.path.join(scene_dir, "images")), os.path.join(scene_copy, "images"))

    results, _ = render_views(scene_copy, view_indices)
    psnrs = []
    for i, gt, recon in results:
        mse = float(np.mean((gt - recon) ** 2))
        psnrs.append(-10.0 * np.log10(max(mse, 1e-10)))
    return float(np.mean(psnrs)), psnrs


def run(scene_dir: str, keep_counts, bq_weight: float = 1.0, min_opacity_for_bq: float = 0.0, view_indices=None, seed: int = 0):
    checkpoint = read_3dgs_ply(os.path.join(scene_dir, "splats.ply"))
    n_total = checkpoint["positions"].shape[0]
    cache_path = RESULTS_DIR / "differentiation_experiment_real_grid_cache.npz"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"{cache_path} not found -- run differentiation_experiment.py --checkpoint {scene_dir} first"
        )
    bq_interp = load_bq_interpolator(str(cache_path))

    positions = checkpoint["positions"]
    opacities = checkpoint["opacities"]
    bq_vals = bq_interp(np.stack([positions[:, 1], positions[:, 0]], axis=1))  # (y, x) order
    bq_vals = np.nan_to_num(bq_vals, nan=float(np.nanmedian(bq_vals)))

    opacity_score = rank_score(opacities)
    bq_score = rank_score(bq_vals)
    # BQ variance is also high in genuinely empty space (little/no local
    # data -> high posterior variance, correctly, but not usefully for a
    # pruning decision -- protecting near-zero-opacity splats sitting in
    # empty space wastes keep-budget on splats that barely render at all).
    # Restricting the BQ boost to splats that already clear a minimal
    # opacity floor keeps BQ voting among plausible candidates rather than
    # among obvious junk.
    bq_eligible = opacities > min_opacity_for_bq
    combined_score = opacity_score + bq_weight * bq_score * bq_eligible

    view_indices = view_indices or [0, 8, 16, 24, 32, 41, 45, 48]

    print(f"checkpoint: {n_total} splats, {scene_dir}")
    with tempfile.TemporaryDirectory() as tmp_root:
        for keep_count in keep_counts:
            keep_count = min(keep_count, n_total)

            opacity_only_idx = np.argsort(opacity_score)[-keep_count:]
            combined_idx = np.argsort(combined_score)[-keep_count:]

            saved_by_bq = np.setdiff1d(combined_idx, opacity_only_idx)
            print(
                f"\n--- keep_count={keep_count} ---\n"
                f"{len(saved_by_bq)} splats kept by the BQ-combined criterion that opacity-only would have "
                f"pruned (mean opacity of those: {opacities[saved_by_bq].mean():.3f}, "
                f"mean BQ variance: {bq_vals[saved_by_bq].mean():.3f} vs. scene median {np.median(bq_vals):.3f})"
            )

            pruned_opacity = prune_checkpoint(checkpoint, opacity_only_idx)
            pruned_combined = prune_checkpoint(checkpoint, combined_idx)

            psnr_opacity, _ = render_and_score(pruned_opacity, scene_dir, view_indices, tmp_root, f"opacity_{keep_count}")
            psnr_combined, _ = render_and_score(pruned_combined, scene_dir, view_indices, tmp_root, f"combined_{keep_count}")

            print(f"opacity-only PSNR:   {psnr_opacity:.2f}dB")
            print(f"BQ-combined PSNR:    {psnr_combined:.2f}dB")
            print(f"delta (combined - opacity-only): {psnr_combined - psnr_opacity:+.2f}dB")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene_dir")
    parser.add_argument("--keep-counts", type=int, nargs="+", default=[4000, 6000, 9000])
    parser.add_argument("--bq-weight", type=float, default=1.0)
    parser.add_argument(
        "--min-opacity-for-bq", type=float, default=0.3,
        help="calibrated empirically (see gs_experiment/results/FINDINGS.md): too low and BQ-combined "
        "protects near-zero-opacity splats in empty space, hurting PSNR at loose budgets; 0.3 gave a "
        "clean win at tight budgets and a no-op (never worse) at loose ones",
    )
    args = parser.parse_args()
    run(args.scene_dir, args.keep_counts, bq_weight=args.bq_weight, min_opacity_for_bq=args.min_opacity_for_bq)


if __name__ == "__main__":
    main()
