"""ROADMAP.md item 6: does the kernel family (RBF vs. Matern-3/2) matter
for the two headline real-data claims -- sparsity correlation
(sparsity_correlation_experiment.py) and calibration
(calibration_experiment.py) -- when both kernels use a properly *fitted*
bandwidth (item 2's `fit_kernel_param_pooled_nd`), not an arbitrary shared
numeric value? Every real-data RBF-vs-Matern comparison so far
(gs_experiment/results/FINDINGS.md §22) only checked whether the two
kernels agree on spatial *pattern* (they do, correlation 0.98) while
differing ~150x in absolute *scale* -- never whether they differ on the
sparsity-correlation or calibration numbers this project's other claims
are built on, and never with a fitted (not hand-picked) bandwidth for
each.

Reuses the bandwidths fit on all three real checkpoints used throughout
items 4-6 (lego wide, and both thin-rod trainers) -- lego's from
scripts/fit_hyperparameters_real_checkpoint.py's earlier run (item 2,
gs_experiment/results/FINDINGS.md section 26); the two thin-rod
checkpoints' fit fresh in this same session as this ablation's own
prerequisite, closing item 2's open question ("is Matern's much larger
correction on lego a general kernel-family property, or specific to that
scene's geometry" -- answer, from the fresh fits: scene-specific, not
general -- lego's Matern needed the bigger correction, the thin-rod
scenes' RBF did).

Needs torch + gsplat only insofar as the checkpoints were already trained;
this script itself is pure numpy/scipy.

Run: .venv-gsplat/bin/python gs_experiment/kernel_family_ablation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.stats import pearsonr

from gs_experiment.pixel_uncertainty import LocalUncertaintyEngine, make_default_3d_matern_kernel, make_default_3d_position_kernel
from gs_experiment.ply_io import read_3dgs_ply

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# (path, window_radius, {"rbf": fitted_bandwidth, "matern32": fitted_bandwidth})
CHECKPOINTS = [
    dict(
        label="lego_wide", path="gs_experiment/local_runs/lego_prepared/wide/splats.ply", window_radius=0.08,
        bandwidths={"rbf": 0.0624, "matern32": 0.0234},
    ),
    dict(
        label="thinrod_fromscratch", path="gs_experiment/local_runs/nbv_out/nll_experiment/baseline/splats.ply", window_radius=0.15,
        bandwidths={"rbf": 0.1135, "matern32": 0.0597},
    ),
    dict(
        label="thinrod_referencestrategy", path="gs_experiment/local_runs/nbv_out/reference_strategy/splats.ply", window_radius=0.15,
        # matern32's fit here generalized *worse* to held-out windows than
        # the hardcoded 0.05 (fitted=-1032.01 vs hardcoded=-560.36 pooled
        # log marginal likelihood) -- a real overfitting signal, so this
        # checkpoint's matern32 entry uses the hardcoded value instead of
        # trusting an overfit one, noted explicitly rather than silently
        # picking whichever number was on hand.
        bandwidths={"rbf": 0.1226, "matern32": 0.05},
    ),
]


def build_engine(path: str, kernel_family: str, bandwidth: float, min_opacity: float = 0.1):
    ck = read_3dgs_ply(path)
    keep = ck["opacities"] > min_opacity
    positions = ck["positions"][keep]
    colors = ck["sh_coeffs"][keep, :, 0].mean(axis=1)
    bounds = tuple((positions[:, d].min() - 0.3, positions[:, d].max() + 0.3) for d in range(3))
    kernel = make_default_3d_position_kernel(bandwidth) if kernel_family == "rbf" else make_default_3d_matern_kernel(bandwidth)
    return LocalUncertaintyEngine(positions=positions, values=colors, pos_kernel=kernel, scene_bounds=bounds)


def sparsity_correlation(engine, window_radius, n_samples=150, seed=0):
    rng = np.random.default_rng(seed)
    query_idx = rng.choice(len(engine.positions), size=min(n_samples, len(engine.positions)), replace=False)
    query_points = engine.positions[query_idx]
    local_counts = np.array([engine.tree.query_ball_point(p, window_radius, return_length=True) for p in query_points])
    bq_variances = np.array([engine.spatial_only_variance(p, window_radius).variance for p in query_points])
    r, p = pearsonr(np.log1p(local_counts), bq_variances)
    return r, p


def calibration(engine, window_radius, n_samples=300, seed=0):
    rng = np.random.default_rng(seed)
    query_idx = rng.choice(len(engine.positions), size=min(n_samples, len(engine.positions)), replace=False)
    means, variances, actuals = [], [], []
    for i in query_idx:
        result = engine.spatial_only_variance(engine.positions[i], window_radius, exclude_idx=i)
        means.append(result.mean)
        variances.append(max(result.variance, 1e-8))
        actuals.append(engine.values[i])
    means, variances, actuals = np.array(means), np.array(variances), np.array(actuals)
    sq_err = (means - actuals) ** 2
    r, p = pearsonr(variances, sq_err)
    nll_bq = float(np.mean(0.5 * (sq_err / variances + np.log(variances)) + 0.5 * np.log(2 * np.pi)))
    nll_constant = float(np.mean(0.5 * (sq_err / variances.mean() + np.log(variances.mean())) + 0.5 * np.log(2 * np.pi)))
    return r, p, nll_bq, nll_constant


def run():
    print(f"{'checkpoint':<28}{'kernel':<10}{'bandwidth':>10}{'sparsity r':>13}{'calib r':>10}{'NLL(bq)':>12}{'NLL(const)':>12}")
    rows = []
    for ckpt in CHECKPOINTS:
        for kernel_family in ("rbf", "matern32"):
            bandwidth = ckpt["bandwidths"][kernel_family]
            engine = build_engine(ckpt["path"], kernel_family, bandwidth)
            sp_r, sp_p = sparsity_correlation(engine, ckpt["window_radius"])
            cal_r, cal_p, nll_bq, nll_const = calibration(engine, ckpt["window_radius"])
            print(f"{ckpt['label']:<28}{kernel_family:<10}{bandwidth:>10.4f}{sp_r:>13.3f}{cal_r:>10.3f}{nll_bq:>12.3f}{nll_const:>12.3f}")
            rows.append(dict(label=ckpt["label"], kernel=kernel_family, bandwidth=bandwidth, sparsity_r=sp_r, calib_r=cal_r, nll_bq=nll_bq, nll_const=nll_const))

    print("\n=== per-checkpoint RBF vs. Matern deltas ===")
    for ckpt in CHECKPOINTS:
        label = ckpt["label"]
        rbf_row = next(r for r in rows if r["label"] == label and r["kernel"] == "rbf")
        mat_row = next(r for r in rows if r["label"] == label and r["kernel"] == "matern32")
        print(
            f"{label}: sparsity_r rbf={rbf_row['sparsity_r']:.3f} matern={mat_row['sparsity_r']:.3f} "
            f"(delta {mat_row['sparsity_r']-rbf_row['sparsity_r']:+.3f})   "
            f"calib_r rbf={rbf_row['calib_r']:.3f} matern={mat_row['calib_r']:.3f} "
            f"(delta {mat_row['calib_r']-rbf_row['calib_r']:+.3f})"
        )


if __name__ == "__main__":
    run()
