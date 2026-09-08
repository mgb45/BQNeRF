"""Consolidated CLI for evaluating a trained gsplat checkpoint's BQ
position-only variance against the various real-data checks accumulated
over ROADMAP.md items 4-7. Each `--check` mode below is a mechanical
merge of a former one-off experiment script (result already recorded in
gs_experiment/results/FINDINGS.md) -- same math, same defaults, same
report format, just one dispatcher instead of seven files:

  sparsity          local splat density vs. BQ position-only variance,
                     correlation on one real checkpoint
                     (was sparsity_correlation_experiment.py)
  calibration        leave-one-out calibration: does BQ variance track
                     actual squared error (Pearson r, AUSE, Gaussian NLL)
                     (was calibration_experiment.py)
  kernel-ablation     RBF vs. Matern-3/2, both fitted-bandwidth, on the
                     sparsity + calibration checks across three fixed
                     real checkpoints (was kernel_family_ablation.py)
  window-ablation     sensitivity of the sparsity-correlation claim to
                     the window_radius hyperparameter, swept 0.2x-8x
                     (was window_radius_ablation.py)
  visibility-trend    BQ variance vs. training-view coverage, across five
                     lego checkpoints at 100/50/25/12 views
                     (was visibility_trend_experiment.py)
  wide-vs-narrow      thin-vs-thick real structure, and wide-vs-narrow
                     cross-checkpoint, on NeRF-Synthetic lego
                     (was real_benchmark_experiment.py)
  multi-scene         drives download/prepare/train/evaluate (sparsity +
                     calibration together) across named NeRF-Synthetic
                     scenes (was multi_scene_experiment.py)

Needs torch + gsplat only insofar as the checkpoints being evaluated were
already trained (and, for `multi-scene`, to train any not yet on disk);
every other mode is pure numpy/scipy against an already-trained checkpoint.

Run: .venv-gsplat/bin/python gs_experiment/evaluate_checkpoint.py sparsity <ply_path>
     .venv-gsplat/bin/python gs_experiment/evaluate_checkpoint.py calibration <ply_path> --sigma 0.05 --window-radius 0.15
     .venv-gsplat/bin/python gs_experiment/evaluate_checkpoint.py kernel-ablation
     .venv-gsplat/bin/python gs_experiment/evaluate_checkpoint.py window-ablation
     .venv-gsplat/bin/python gs_experiment/evaluate_checkpoint.py visibility-trend <lego_prepared_dir>
     .venv-gsplat/bin/python gs_experiment/evaluate_checkpoint.py wide-vs-narrow <lego_prepared_dir>
     .venv-gsplat/bin/python gs_experiment/evaluate_checkpoint.py multi-scene --scenes chair,drums,ficus
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

from gs_experiment.pixel_uncertainty import (
    LocalUncertaintyEngine,
    make_default_3d_matern_kernel,
    make_default_3d_position_kernel,
)
from gs_experiment.ply_io import read_3dgs_ply
from gs_experiment.prepare_nerf_synthetic import run as prepare_scene
from gs_experiment.train_minimal_gsplat import train

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# --check sparsity   (was sparsity_correlation_experiment.py)
#
# Does BQ position-only variance track local splat sparsity, on real
# reconstructed geometry -- the direct, minimal demonstration that
# rendering-as-quadrature gives a real uncertainty signal essentially for
# free, closed-form, from the same kernel structure already used to
# represent the scene.
# =====================================================================


def run_sparsity_correlation(
    ply_path: str,
    n_samples: int = 150,
    sigma: float = 0.05,
    window_radius: float = 0.08,
    min_opacity: float = 0.1,
    seed: int = 0,
):
    ck = read_3dgs_ply(ply_path)
    keep = ck["opacities"] > min_opacity
    positions = ck["positions"][keep]
    colors = ck["sh_coeffs"][keep, :, 0].mean(axis=1)
    print(f"{len(positions)} splats above opacity {min_opacity}")

    bounds = tuple((positions[:, d].min() - 0.3, positions[:, d].max() + 0.3) for d in range(3))
    pos_kernel = make_default_3d_position_kernel(sigma=sigma)
    engine = LocalUncertaintyEngine(positions=positions, values=colors, pos_kernel=pos_kernel, scene_bounds=bounds)

    rng = np.random.default_rng(seed)
    query_idx = rng.choice(len(positions), size=min(n_samples, len(positions)), replace=False)
    query_points = positions[query_idx]

    # true local density via a direct KD-tree count, not engine.
    # local_neighbors' output: that method applies LocalUncertaintyEngine's
    # max_neighbors cap (subsampling above 400), which would silently
    # flatten the density signal for any query with more than 400 true
    # neighbors -- the BQ solve itself still (correctly) uses the capped
    # set, so this measures true density against the variance BQ actually
    # reports, not a self-referential comparison.
    local_counts = np.array(
        [engine.tree.query_ball_point(p, window_radius, return_length=True) for p in query_points]
    )
    bq_variances = np.array([engine.spatial_only_variance(p, window_radius).variance for p in query_points])

    log_count = np.log1p(local_counts)
    pearson_r, pearson_p = pearsonr(log_count, bq_variances)
    spearman_r, spearman_p = spearmanr(local_counts, bq_variances)

    print(f"local splat count: min={local_counts.min()} median={np.median(local_counts):.0f} max={local_counts.max()}")
    print(f"BQ variance: min={bq_variances.min():.6f} median={np.median(bq_variances):.6f} max={bq_variances.max():.6f}")
    print(f"\nPearson correlation (log(1+count) vs. BQ variance): r={pearson_r:.3f}  p={pearson_p:.2e}")
    print(f"Spearman rank correlation (count vs. BQ variance):    rho={spearman_r:.3f}  p={spearman_p:.2e}")

    sparse_thresh, dense_thresh = np.quantile(local_counts, [0.2, 0.8])
    sparse_var = bq_variances[local_counts <= sparse_thresh]
    dense_var = bq_variances[local_counts >= dense_thresh]
    print(
        f"\nbottom-20% density (<= {sparse_thresh:.0f} neighbors): mean BQ variance = {sparse_var.mean():.6f}\n"
        f"top-20% density (>= {dense_thresh:.0f} neighbors):    mean BQ variance = {dense_var.mean():.6f}\n"
        f"ratio (sparse/dense): {sparse_var.mean() / dense_var.mean():.2f}x"
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(local_counts, bq_variances, s=14, alpha=0.6, edgecolors="none")
    ax.set_xlabel("local splat count within window (sparsity proxy)")
    ax.set_ylabel("BQ position-only variance")
    ax.set_title(f"BQ variance vs. local splat density\n(Spearman rho={spearman_r:.2f}, {ply_path})", fontsize=10)
    fig.tight_layout()
    out = RESULTS_DIR / "sparsity_correlation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out}")


# =====================================================================
# --check calibration   (was calibration_experiment.py)
#
# Is BQ variance calibrated, not just correlated with error? Leave-one-out
# cross-validation on real splat colors: remove a splat from its own local
# BQ neighborhood, predict its color from its (real) neighbors alone, and
# compare the BQ posterior mean/variance against the splat's own real,
# held-out color.
# =====================================================================


def leave_one_out_predictions(engine: LocalUncertaintyEngine, query_idx, window_radius: float):
    means, variances, actuals = [], [], []
    for i in query_idx:
        result = engine.spatial_only_variance(engine.positions[i], window_radius, exclude_idx=i)
        means.append(result.mean)
        variances.append(result.variance)
        actuals.append(engine.values[i])
    return np.array(means), np.array(variances), np.array(actuals)


def sparsification_curve(sq_err: np.ndarray, order_by: np.ndarray, descending: bool = True):
    """Drop points one at a time in the order given by `order_by` (its
    largest values dropped first if descending), return the mean squared
    error of the points *remaining* at each drop fraction -- the standard
    sparsification-curve construction."""
    n = len(sq_err)
    order = np.argsort(-order_by if descending else order_by)
    sorted_err = sq_err[order]
    remaining_mean = np.array([sorted_err[k:].mean() if k < n else 0.0 for k in range(n + 1)])
    fractions = np.arange(n + 1) / n
    return fractions, remaining_mean


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """np.trapz was renamed np.trapezoid in numpy 2.0 (removed entirely in
    some intermediate releases) -- implemented directly rather than
    depending on either name being present in whatever numpy this runs
    under."""
    return float(np.sum((y[1:] + y[:-1]) * np.diff(x) / 2.0))


def ause(fractions: np.ndarray, curve: np.ndarray, oracle_curve: np.ndarray) -> float:
    return _trapezoid(curve - oracle_curve, fractions)


def gaussian_nll(sq_err: np.ndarray, variance: np.ndarray) -> float:
    return float(np.mean(0.5 * (sq_err / variance + np.log(variance)) + 0.5 * np.log(2 * np.pi)))


def run_calibration(
    ply_path: str,
    n_samples: int = 300,
    sigma: float = 0.05,
    window_radius: float = 0.15,
    min_opacity: float = 0.1,
    max_neighbors: int = 150,
    variance_floor: float = 1e-8,
    seed: int = 0,
    label: str = "",
):
    ck = read_3dgs_ply(ply_path)
    keep = ck["opacities"] > min_opacity
    positions = ck["positions"][keep]
    colors = ck["sh_coeffs"][keep, :, 0].mean(axis=1)
    print(f"{label or ply_path}: {len(positions)} splats above opacity {min_opacity}")

    bounds = tuple((positions[:, d].min() - 0.3, positions[:, d].max() + 0.3) for d in range(3))
    pos_kernel = make_default_3d_position_kernel(sigma=sigma)
    engine = LocalUncertaintyEngine(
        positions=positions, values=colors, pos_kernel=pos_kernel, scene_bounds=bounds, max_neighbors=max_neighbors, seed=seed,
    )

    rng = np.random.default_rng(seed)
    query_idx = rng.choice(len(positions), size=min(n_samples, len(positions)), replace=False)

    means, variances, actuals = leave_one_out_predictions(engine, query_idx, window_radius)
    variances = np.clip(variances, variance_floor, None)
    sq_err = (means - actuals) ** 2

    pearson_r, pearson_p = pearsonr(variances, sq_err)
    print(f"  leave-one-out: Pearson r(BQ variance, squared error) = {pearson_r:.3f}  p={pearson_p:.2e}")

    frac_bq, curve_bq = sparsification_curve(sq_err, variances, descending=True)
    frac_oracle, curve_oracle = sparsification_curve(sq_err, sq_err, descending=True)
    frac_random, curve_random = sparsification_curve(sq_err, rng.permutation(len(sq_err)).astype(float), descending=True)

    ause_bq = ause(frac_bq, curve_bq, curve_oracle)
    ause_random = ause(frac_random, curve_random, curve_oracle)
    print(f"  AUSE (BQ ordering vs. oracle): {ause_bq:.6f}   AUSE (random ordering vs. oracle, for scale): {ause_random:.6f}")

    nll_bq = gaussian_nll(sq_err, variances)
    nll_constant = gaussian_nll(sq_err, np.full_like(variances, variances.mean()))
    print(f"  held-out Gaussian NLL: per-point BQ variance = {nll_bq:.4f}   constant (mean) variance = {nll_constant:.4f}"
          f"  (lower is better; per-point beats constant means the *shape* of the variance helps, not just its scale)")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(frac_oracle, curve_oracle, label="oracle (drop by true error)", linewidth=2)
    ax.plot(frac_bq, curve_bq, label="BQ variance ordering", linewidth=2)
    ax.plot(frac_random, curve_random, label="random ordering", linestyle="--", color="gray")
    ax.set_xlabel("fraction of points dropped (highest first)")
    ax.set_ylabel("mean squared error of points remaining")
    ax.set_title(f"Sparsification curve: leave-one-out calibration\n{label or ply_path}\nAUSE={ause_bq:.5f}", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = RESULTS_DIR / f"calibration_sparsification_{label or 'checkpoint'}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")

    return dict(pearson_r=pearson_r, ause_bq=ause_bq, ause_random=ause_random, nll_bq=nll_bq, nll_constant=nll_constant)


# =====================================================================
# --check kernel-ablation   (was kernel_family_ablation.py)
#
# Does the kernel family (RBF vs. Matern-3/2) matter for the sparsity-
# correlation and calibration checks above, when both kernels use a
# properly *fitted* bandwidth (not an arbitrary shared numeric value)?
# Reuses the bandwidths fit on three real checkpoints (lego wide, and both
# thin-rod trainers).
# =====================================================================

# (path, window_radius, {"rbf": fitted_bandwidth, "matern32": fitted_bandwidth})
KERNEL_ABLATION_CHECKPOINTS = [
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


def run_kernel_ablation():
    print(f"{'checkpoint':<28}{'kernel':<10}{'bandwidth':>10}{'sparsity r':>13}{'calib r':>10}{'NLL(bq)':>12}{'NLL(const)':>12}")
    rows = []
    for ckpt in KERNEL_ABLATION_CHECKPOINTS:
        for kernel_family in ("rbf", "matern32"):
            bandwidth = ckpt["bandwidths"][kernel_family]
            engine = build_engine(ckpt["path"], kernel_family, bandwidth)
            sp_r, sp_p = sparsity_correlation(engine, ckpt["window_radius"])
            cal_r, cal_p, nll_bq, nll_const = calibration(engine, ckpt["window_radius"])
            print(f"{ckpt['label']:<28}{kernel_family:<10}{bandwidth:>10.4f}{sp_r:>13.3f}{cal_r:>10.3f}{nll_bq:>12.3f}{nll_const:>12.3f}")
            rows.append(dict(label=ckpt["label"], kernel=kernel_family, bandwidth=bandwidth, sparsity_r=sp_r, calib_r=cal_r, nll_bq=nll_bq, nll_const=nll_const))

    print("\n=== per-checkpoint RBF vs. Matern deltas ===")
    for ckpt in KERNEL_ABLATION_CHECKPOINTS:
        label = ckpt["label"]
        rbf_row = next(r for r in rows if r["label"] == label and r["kernel"] == "rbf")
        mat_row = next(r for r in rows if r["label"] == label and r["kernel"] == "matern32")
        print(
            f"{label}: sparsity_r rbf={rbf_row['sparsity_r']:.3f} matern={mat_row['sparsity_r']:.3f} "
            f"(delta {mat_row['sparsity_r']-rbf_row['sparsity_r']:+.3f})   "
            f"calib_r rbf={rbf_row['calib_r']:.3f} matern={mat_row['calib_r']:.3f} "
            f"(delta {mat_row['calib_r']-rbf_row['calib_r']:+.3f})"
        )


# =====================================================================
# --check window-ablation   (was window_radius_ablation.py)
#
# How sensitive is the sparsity-correlation claim to the window_radius
# hyperparameter? Fixes sigma at each checkpoint's already-established
# value and sweeps window_radius across 0.2x-8x that value, on the same
# three real checkpoints used by kernel-ablation.
# =====================================================================

WINDOW_ABLATION_CHECKPOINTS = [
    dict(label="lego_wide", path="gs_experiment/local_runs/lego_prepared/wide/splats.ply", sigma=0.05, base_window=0.08),
    dict(label="thinrod_fromscratch", path="gs_experiment/local_runs/nbv_out/nll_experiment/baseline/splats.ply", sigma=0.05, base_window=0.15),
    dict(label="thinrod_referencestrategy", path="gs_experiment/local_runs/nbv_out/reference_strategy/splats.ply", sigma=0.05, base_window=0.15),
]

MULTIPLIERS = [0.2, 0.5, 1.0, 2.0, 4.0, 8.0]


def sweep_one_checkpoint(path: str, sigma: float, base_window: float, n_samples: int = 150, min_opacity: float = 0.1, seed: int = 0):
    ck = read_3dgs_ply(path)
    keep = ck["opacities"] > min_opacity
    positions = ck["positions"][keep]
    colors = ck["sh_coeffs"][keep, :, 0].mean(axis=1)

    bounds = tuple((positions[:, d].min() - 0.3, positions[:, d].max() + 0.3) for d in range(3))
    pos_kernel = make_default_3d_position_kernel(sigma=sigma)
    engine = LocalUncertaintyEngine(positions=positions, values=colors, pos_kernel=pos_kernel, scene_bounds=bounds)

    rng = np.random.default_rng(seed)
    query_idx = rng.choice(len(positions), size=min(n_samples, len(positions)), replace=False)
    query_points = positions[query_idx]

    rows = []
    for mult in MULTIPLIERS:
        window_radius = base_window * mult
        local_counts = np.array(
            [engine.tree.query_ball_point(p, window_radius, return_length=True) for p in query_points]
        )
        bq_variances = np.array([engine.spatial_only_variance(p, window_radius).variance for p in query_points])
        r, p = pearsonr(np.log1p(local_counts), bq_variances)
        rows.append(dict(mult=mult, window_radius=window_radius, r=r, p=p, median_count=float(np.median(local_counts))))
    return len(positions), rows


def run_window_ablation():
    fig, ax = plt.subplots(figsize=(7, 5))
    summary_rows = []

    for ckpt in WINDOW_ABLATION_CHECKPOINTS:
        n_splats, rows = sweep_one_checkpoint(ckpt["path"], ckpt["sigma"], ckpt["base_window"])
        print(f"\n=== {ckpt['label']} ({n_splats} splats, sigma={ckpt['sigma']}, base_window={ckpt['base_window']}) ===")
        print(f"{'mult':>6}{'window_radius':>16}{'median_count':>14}{'r':>10}{'p':>12}")
        for row in rows:
            print(f"{row['mult']:>6.1f}{row['window_radius']:>16.3f}{row['median_count']:>14.1f}{row['r']:>10.3f}{row['p']:>12.2e}")
            summary_rows.append(dict(label=ckpt["label"], **row))

        mults = [row["mult"] for row in rows]
        rs = [row["r"] for row in rows]
        ax.plot(mults, rs, marker="o", label=ckpt["label"])

    ax.axhline(0.0, color="gray", linewidth=1, linestyle=":")
    ax.axvline(1.0, color="gray", linewidth=1, linestyle=":", label="each checkpoint's established window_radius")
    ax.set_xscale("log")
    ax.set_xlabel("window_radius / established value (log scale)")
    ax.set_ylabel("Pearson r(log(1+local count), BQ variance)")
    ax.set_title("Sensitivity of the sparsity-correlation claim to window_radius")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = RESULTS_DIR / "window_radius_ablation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out}")

    print("\n=== summary: does the correlation survive at 0.2x-8x the established window_radius? ===")
    for ckpt in WINDOW_ABLATION_CHECKPOINTS:
        label = ckpt["label"]
        label_rows = [r for r in summary_rows if r["label"] == label]
        signs = set(np.sign(r["r"]) for r in label_rows if r["p"] < 0.05)
        strong = [r for r in label_rows if r["p"] < 0.05 and abs(r["r"]) > 0.3]
        print(
            f"{label}: significant-r sign(s) across the sweep = {signs}; "
            f"{len(strong)}/{len(label_rows)} multipliers give |r|>0.3 and p<0.05"
        )


# =====================================================================
# --check visibility-trend   (was visibility_trend_experiment.py)
#
# Does BQ position-only variance grow as training-view coverage shrinks?
# Five independently-trained checkpoints of the same real object
# (wide/rand50/rand25/rand12 at 100/50/25/12 training views, plus the
# original angularly-clustered "narrow" 12-view checkpoint), each
# evaluated at the same fixed set of real-world query points.
# =====================================================================

CONDITIONS = [("wide", 100), ("rand50", 50), ("rand25", 25), ("rand12", 12), ("narrow", 12)]


def load_engine(ply_path: str, sigma: float, min_opacity: float = 0.1):
    ck = read_3dgs_ply(ply_path)
    keep = ck["opacities"] > min_opacity
    positions = ck["positions"][keep]
    colors = ck["sh_coeffs"][keep, :, 0].mean(axis=1)
    bounds = tuple((positions[:, d].min() - 0.3, positions[:, d].max() + 0.3) for d in range(3))
    pos_kernel = make_default_3d_position_kernel(sigma=sigma)
    engine = LocalUncertaintyEngine(positions=positions, values=colors, pos_kernel=pos_kernel, scene_bounds=bounds)
    return engine, positions


def run_visibility_trend(lego_dir: str, n_query_points: int = 150, sigma: float = 0.05, window_radius: float = 0.08, seed: int = 0):
    rng = np.random.default_rng(seed)

    # fixed query points, defined once in world space from the wide
    # (most complete) checkpoint's own splat positions -- every other
    # checkpoint is queried at these exact same xyz locations, since all
    # four share one coordinate system (the same original transforms.json).
    wide_ply = os.path.join(lego_dir, "wide", "splats.ply")
    _, wide_positions = load_engine(wide_ply, sigma)
    query_idx = rng.choice(len(wide_positions), size=n_query_points, replace=False)
    query_points = wide_positions[query_idx]

    means, medians = {}, {}
    for label, n_views in CONDITIONS:
        ply_path = os.path.join(lego_dir, label, "splats.ply")
        engine, _ = load_engine(ply_path, sigma)
        variances = np.array([engine.spatial_only_variance(p, window_radius).variance for p in query_points])
        means[label] = variances.mean()
        medians[label] = np.median(variances)
        print(f"{label} ({n_views} views): mean BQ variance = {variances.mean():.6f}  median = {np.median(variances):.6f}")

    # two separate questions, deliberately not conflated into one trend:
    random_labels = ["wide", "rand50", "rand25", "rand12"]  # count varies, angular spread stays full
    random_means = [means[l] for l in random_labels]
    monotonic = all(random_means[i] <= random_means[i + 1] for i in range(len(random_means) - 1))
    print(f"\n[count, full angular spread held fixed] monotonically non-decreasing as views drop 100->12: {monotonic}")
    print(f"  ratio (rand12/wide): {means['rand12'] / means['wide']:.2f}x")
    print(f"\n[clustering, count held fixed at 12] random-12 vs. angularly-clustered narrow-12:")
    print(f"  ratio (narrow/rand12): {means['narrow'] / means['rand12']:.2f}x")

    fig, ax = plt.subplots(figsize=(7, 5))
    labels_ordered = [c[0] for c in CONDITIONS]
    x = np.arange(len(labels_ordered))
    ax.bar(x, [means[l] for l in labels_ordered], color=["#4c72b0"] * 4 + ["#c44e52"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n({dict(CONDITIONS)[l]} views)" for l in labels_ordered])
    ax.set_ylabel("mean BQ position-only variance\n(same fixed query points, all conditions)")
    ax.set_title(f"BQ variance: view count vs. angular clustering\n({lego_dir})", fontsize=10)
    fig.tight_layout()
    out = RESULTS_DIR / "visibility_trend.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out}")


# =====================================================================
# --check wide-vs-narrow   (was real_benchmark_experiment.py)
#
# Real-benchmark version of the differentiation experiment: does
# position-only BQ variance flag genuinely thin/fine structure as more
# uncertain than thick/simple structure, on NeRF-Synthetic lego? "Thin"
# and "thick" regions are identified from a trained checkpoint's own
# per-splat scale (the natural GS-native proxy for local feature size).
# =====================================================================


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


def run_wide_vs_narrow(lego_dir: str, n_samples: int = 60, quantile: float = 0.2, sigma: float = 0.15, window_radius: float = 0.2, seed: int = 0):
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


# =====================================================================
# --check multi-scene   (was multi_scene_experiment.py)
#
# Extends the sparsity-correlation and calibration checks above across
# the complete standard 8-scene NeRF-Synthetic benchmark, not just lego.
# Trains each new scene at a deliberately lighter budget than lego's
# original 80,000-splat-cap run -- enough for a real, densified checkpoint
# with genuine view-coverage-dependent splat density, not a publication-
# quality reconstruction.
# =====================================================================

RAW_ROOT = "gs_experiment/local_runs/nerf_synthetic_raw"
PREPARED_ROOT = "gs_experiment/local_runs"

TRAIN_KWARGS = dict(
    n_splats=2000, bounds=((-2.5, 2.5), (-2.5, 2.5), (-2.5, 2.5)), sh_degree=1, n_iters=2500, seed=0,
    init_scale=0.05, opacity_reg_weight=0.01, densify=True, densify_interval=300, densify_start=300,
    max_splats=15000, log_every=500,
)
MULTI_SCENE_SIGMA = 0.05
MULTI_SCENE_WINDOW_RADIUS = 0.08


def download_scene(scene: str):
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="pablovela5620/nerf-synthetic-mirror", repo_type="dataset",
        allow_patterns=[f"{scene}/*"], local_dir=RAW_ROOT,
    )


def prepare_and_train(scene: str, skip_download: bool = False, skip_prepare: bool = False, skip_train: bool = False):
    raw_dir = os.path.join(RAW_ROOT, scene)
    prepared_dir = os.path.join(PREPARED_ROOT, f"{scene}_prepared")
    ply_path = os.path.join(prepared_dir, "wide", "splats.ply")

    if not skip_download and not os.path.exists(raw_dir):
        print(f"[{scene}] downloading...")
        download_scene(scene)

    if not skip_prepare and not os.path.exists(os.path.join(prepared_dir, "wide", "transforms.json")):
        print(f"[{scene}] preparing...")
        prepare_scene(raw_dir, prepared_dir)

    if not skip_train and not os.path.exists(ply_path):
        print(f"[{scene}] training (lighter budget than lego's original)...")
        train(os.path.join(prepared_dir, "wide"), ply_path, **TRAIN_KWARGS)

    return ply_path


def evaluate(scene: str, ply_path: str, n_samples: int = 150, seed: int = 0):
    from scipy.spatial import cKDTree

    engine = build_engine(ply_path, "rbf", MULTI_SCENE_SIGMA)
    n_splats = len(engine.positions)

    import numpy as np

    rng = np.random.default_rng(seed)
    query_idx = rng.choice(n_splats, size=min(n_samples, n_splats), replace=False)
    query_points = engine.positions[query_idx]
    local_counts = np.array([engine.tree.query_ball_point(p, MULTI_SCENE_WINDOW_RADIUS, return_length=True) for p in query_points])
    bq_variances = np.array([engine.spatial_only_variance(p, MULTI_SCENE_WINDOW_RADIUS).variance for p in query_points])
    sparsity_r, sparsity_p = pearsonr(np.log1p(local_counts), bq_variances)

    calib_r, calib_p, nll_bq, nll_const = calibration(engine, MULTI_SCENE_WINDOW_RADIUS, n_samples=n_samples, seed=seed)

    return dict(
        scene=scene, n_splats=n_splats, median_local_count=float(np.median(local_counts)),
        sparsity_r=sparsity_r, sparsity_p=sparsity_p, calib_r=calib_r, calib_p=calib_p,
        nll_bq=nll_bq, nll_const=nll_const,
    )


def run_multi_scene(scenes, skip_download=False, skip_prepare=False, skip_train=False):
    results = []
    for scene in scenes:
        ply_path = prepare_and_train(scene, skip_download=skip_download, skip_prepare=skip_prepare, skip_train=skip_train)
        result = evaluate(scene, ply_path)
        results.append(result)
        print(
            f"[{scene}] n_splats={result['n_splats']}  median_local_count={result['median_local_count']:.1f}  "
            f"sparsity_r={result['sparsity_r']:.3f} (p={result['sparsity_p']:.1e})  "
            f"calib_r={result['calib_r']:.3f} (p={result['calib_p']:.1e})  "
            f"NLL(bq)={result['nll_bq']:.2f}  NLL(const)={result['nll_const']:.2f}"
        )

    print(f"\n{'scene':<12}{'n_splats':>10}{'sparsity_r':>13}{'calib_r':>10}{'NLL(bq)':>12}{'NLL(const)':>12}")
    for r in results:
        print(f"{r['scene']:<12}{r['n_splats']:>10}{r['sparsity_r']:>13.3f}{r['calib_r']:>10.3f}{r['nll_bq']:>12.2f}{r['nll_const']:>12.2f}")
    return results


# =====================================================================
# dispatcher
# =====================================================================


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="check", required=True, metavar="check")

    p_sparsity = sub.add_parser("sparsity", help="local splat density vs. BQ position-only variance correlation")
    p_sparsity.add_argument("ply_path")
    p_sparsity.add_argument("--n-samples", type=int, default=150)
    p_sparsity.add_argument("--sigma", type=float, default=0.05)
    p_sparsity.add_argument("--window-radius", type=float, default=0.08)

    p_calibration = sub.add_parser("calibration", help="leave-one-out calibration: BQ variance vs. squared error")
    p_calibration.add_argument("ply_path")
    p_calibration.add_argument("--n-samples", type=int, default=300)
    p_calibration.add_argument("--sigma", type=float, default=0.05)
    p_calibration.add_argument("--window-radius", type=float, default=0.15)
    p_calibration.add_argument("--label", default="")

    sub.add_parser("kernel-ablation", help="RBF vs. Matern-3/2, fitted bandwidths, on three fixed real checkpoints")

    sub.add_parser("window-ablation", help="sweep window_radius 0.2x-8x on three fixed real checkpoints")

    p_visibility = sub.add_parser("visibility-trend", help="BQ variance vs. training-view coverage across five lego checkpoints")
    p_visibility.add_argument("lego_dir")
    p_visibility.add_argument("--n-query-points", type=int, default=150)
    p_visibility.add_argument("--sigma", type=float, default=0.05)
    p_visibility.add_argument("--window-radius", type=float, default=0.08)

    p_wide_narrow = sub.add_parser("wide-vs-narrow", help="thin-vs-thick and wide-vs-narrow real-benchmark checks on lego")
    p_wide_narrow.add_argument("lego_dir")
    p_wide_narrow.add_argument("--n-samples", type=int, default=60)
    p_wide_narrow.add_argument("--quantile", type=float, default=0.2)
    p_wide_narrow.add_argument("--sigma", type=float, default=0.15)
    p_wide_narrow.add_argument("--window-radius", type=float, default=0.2)

    p_multi_scene = sub.add_parser("multi-scene", help="download/prepare/train/evaluate across named NeRF-Synthetic scenes")
    p_multi_scene.add_argument("--scenes", required=True, help="comma-separated scene names, e.g. chair,drums,ficus")
    p_multi_scene.add_argument("--skip-download", action="store_true")
    p_multi_scene.add_argument("--skip-prepare", action="store_true")
    p_multi_scene.add_argument("--skip-train", action="store_true")

    args = parser.parse_args()

    if args.check == "sparsity":
        run_sparsity_correlation(args.ply_path, n_samples=args.n_samples, sigma=args.sigma, window_radius=args.window_radius)
    elif args.check == "calibration":
        run_calibration(args.ply_path, n_samples=args.n_samples, sigma=args.sigma, window_radius=args.window_radius, label=args.label)
    elif args.check == "kernel-ablation":
        run_kernel_ablation()
    elif args.check == "window-ablation":
        run_window_ablation()
    elif args.check == "visibility-trend":
        run_visibility_trend(args.lego_dir, n_query_points=args.n_query_points, sigma=args.sigma, window_radius=args.window_radius)
    elif args.check == "wide-vs-narrow":
        run_wide_vs_narrow(args.lego_dir, n_samples=args.n_samples, quantile=args.quantile, sigma=args.sigma, window_radius=args.window_radius)
    elif args.check == "multi-scene":
        scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
        run_multi_scene(scenes, skip_download=args.skip_download, skip_prepare=args.skip_prepare, skip_train=args.skip_train)


if __name__ == "__main__":
    main()
