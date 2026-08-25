"""ROADMAP.md item 2: kernel hyperparameters as fitted, first-class
quantities, at real GS scale -- not hardcoded, as every gs_experiment
script currently does (`sigma=0.05`, picked once, never checked against
the data).

`bq_splat/hyperparams.py` already does marginal-likelihood bandwidth
fitting (bq_splat/results/FINDINGS.md sections 5 and 7), but only ever
against toy 1D scenes. This script does the same thing -- maximize the GP
log marginal likelihood -- against a real trained checkpoint's own local
neighborhoods, using the new `log_marginal_likelihood_nd` /
`fit_kernel_param_pooled_nd` (bq_splat/hyperparams.py), and then checks
whether the fitted bandwidth actually changes anything that matters: does
it shift the BQ-variance-vs-sparsity correlation
(sparsity_correlation_experiment.py) away from what the hardcoded sigma=0.05
already found (r=-0.74, gs_experiment/results/FINDINGS.md section 20)?

Procedure:
  1. Sample query points across the checkpoint, take each one's local
     window (same ball-query convention as LocalUncertaintyEngine) as one
     "dataset" -- (local_positions, local_colors).
  2. Split windows into a fit set and a disjoint held-out set (same
     validate_trainable_kernel_heldout.py spirit as bq_splat/results/
     FINDINGS.md section 7 -- does a bandwidth fit on one part of the
     checkpoint generalize to another part, or is it overfitting to the
     specific windows it was fit on).
  3. Fit a shared bandwidth on the fit set via pooled marginal likelihood,
     for both RBF and Matern-3/2.
  4. Report: fitted value vs. the hardcoded 0.05; held-out pooled log
     marginal likelihood at the fitted value vs. at 0.05 (does fitting
     actually generalize, or just to the fitting windows); and the
     sparsity-correlation Pearson r at both sigma values, to check whether
     the earlier finding is sigma-sensitive.

Needs torch + gsplat only insofar as the checkpoint was already trained;
this script itself is pure numpy/scipy.

Run: .venv-gsplat/bin/python scripts/fit_hyperparameters_real_checkpoint.py <lego_prepared_dir>/wide/splats.ply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.stats import pearsonr

from bq_splat.hyperparams import fit_kernel_param_pooled_nd, pooled_log_marginal_likelihood_nd
from bq_splat.kernels import MaternKernel, ProductKernel, RBFKernel
from gs_experiment.pixel_uncertainty import LocalUncertaintyEngine, make_default_3d_matern_kernel, make_default_3d_position_kernel
from gs_experiment.ply_io import read_3dgs_ply

RESULTS_DIR = Path(__file__).resolve().parents[1] / "gs_experiment" / "results"


def collect_windows(positions, colors, query_points, window_radius, max_window_size, seed):
    """One (local_positions, local_colors) dataset per query point, capped
    at max_window_size (a random subsample, not a truncation -- keeps the
    per-window fit representative rather than spatially biased) so pooled
    marginal-likelihood fitting over many windows stays cheap: each grid
    point in fit_kernel_param_pooled_nd does one Cholesky per window."""
    from scipy.spatial import cKDTree

    tree = cKDTree(positions)
    rng = np.random.default_rng(seed)
    datasets = []
    for p in query_points:
        idx = np.array(tree.query_ball_point(p, window_radius), dtype=int)
        if len(idx) < 6:
            continue
        if len(idx) > max_window_size:
            idx = rng.choice(idx, size=max_window_size, replace=False)
        datasets.append((positions[idx], colors[idx]))
    return datasets


def run(
    ply_path: str,
    n_fit_windows: int = 25,
    n_heldout_windows: int = 25,
    window_radius: float = 0.08,
    max_window_size: int = 60,
    min_opacity: float = 0.1,
    hardcoded_sigma: float = 0.05,
    seed: int = 0,
):
    ck = read_3dgs_ply(ply_path)
    keep = ck["opacities"] > min_opacity
    positions = ck["positions"][keep]
    colors = ck["sh_coeffs"][keep, :, 0].mean(axis=1)
    print(f"{len(positions)} splats above opacity {min_opacity}\n")

    rng = np.random.default_rng(seed)
    n_total = n_fit_windows + n_heldout_windows
    query_idx = rng.choice(len(positions), size=min(n_total, len(positions)), replace=False)
    fit_points = positions[query_idx[:n_fit_windows]]
    heldout_points = positions[query_idx[n_fit_windows:n_total]]

    fit_datasets = collect_windows(positions, colors, fit_points, window_radius, max_window_size, seed)
    heldout_datasets = collect_windows(positions, colors, heldout_points, window_radius, max_window_size, seed + 1)
    print(f"{len(fit_datasets)} fit windows, {len(heldout_datasets)} held-out windows "
          f"(window_radius={window_radius}, capped at {max_window_size} points/window)\n")

    results = {}
    for name, factory, default_kernel in (
        ("rbf", lambda s: ProductKernel([RBFKernel(sigma=s)] * 3), make_default_3d_position_kernel(hardcoded_sigma)),
        ("matern32", lambda r: ProductKernel([MaternKernel(rho=r)] * 3), make_default_3d_matern_kernel(hardcoded_sigma)),
    ):
        fit = fit_kernel_param_pooled_nd(fit_datasets, factory, bounds=(0.005, 1.0), n_grid=25)
        lml_heldout_fitted = pooled_log_marginal_likelihood_nd(heldout_datasets, factory(fit.param))
        lml_heldout_hardcoded = pooled_log_marginal_likelihood_nd(heldout_datasets, default_kernel)

        print(f"[{name}]")
        print(f"  fitted bandwidth: {fit.param:.4f}  (hardcoded value in use: {hardcoded_sigma})")
        print(f"  pooled log marginal likelihood on FIT windows at fitted value: {fit.log_marginal_likelihood:.2f}")
        print(f"  pooled log marginal likelihood on HELD-OUT windows: "
              f"fitted={lml_heldout_fitted:.2f}  hardcoded={lml_heldout_hardcoded:.2f}  "
              f"(higher is better; fitted generalizing means fitted >= hardcoded here)\n")
        results[name] = fit.param

    print("=== does the fitted bandwidth change the sparsity-correlation finding? ===")
    corr_query_idx = rng.choice(len(positions), size=min(150, len(positions)), replace=False)
    corr_points = positions[corr_query_idx]
    for name, sigma_value, label in (
        ("rbf", hardcoded_sigma, "hardcoded sigma=0.05"),
        ("rbf", results["rbf"], "fitted sigma"),
    ):
        kernel = make_default_3d_position_kernel(sigma_value)
        bounds = tuple((positions[:, d].min() - 0.3, positions[:, d].max() + 0.3) for d in range(3))
        engine = LocalUncertaintyEngine(positions=positions, values=colors, pos_kernel=kernel, scene_bounds=bounds)
        local_counts = np.array(
            [engine.tree.query_ball_point(p, window_radius, return_length=True) for p in corr_points]
        )
        bq_variances = np.array([engine.spatial_only_variance(p, window_radius).variance for p in corr_points])
        r, p = pearsonr(np.log1p(local_counts), bq_variances)
        print(f"  {label} ({sigma_value:.4f}): Pearson r={r:.3f}  p={p:.2e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ply_path")
    parser.add_argument("--n-fit-windows", type=int, default=25)
    parser.add_argument("--n-heldout-windows", type=int, default=25)
    parser.add_argument("--window-radius", type=float, default=0.08)
    args = parser.parse_args()
    run(args.ply_path, n_fit_windows=args.n_fit_windows, n_heldout_windows=args.n_heldout_windows, window_radius=args.window_radius)


if __name__ == "__main__":
    main()
