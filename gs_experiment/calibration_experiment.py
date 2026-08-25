"""ROADMAP.md item 5: is BQ variance calibrated, not just correlated with
error? Every real-data result so far (sparsity correlation, differentiation
ratios) is a correlation or ratio -- none of them checks whether a claimed
"2x higher variance" region actually has ~2x the squared error, which is
what "calibrated" means and what any downstream use of the variance
(sparsification, active-view budgeting, a paper claim) actually needs.

Protocol: leave-one-out cross-validation on real splat colors. For many
real splats in a real trained checkpoint, remove that splat from its own
local BQ neighborhood (`LocalUncertaintyEngine`'s new `exclude_idx`),
predict its color from its (real) neighbors alone, and compare the BQ
posterior mean/variance against the splat's own real, held-out color. This
gives real (predicted_mean, predicted_variance, actual_value) triples --
not a synthetic calibration check, an honest one, on a real checkpoint,
with a real "ground truth" (the splat's own value) never seen by the
prediction.

Reports:
  - Pearson r between BQ variance and squared error (a direct, simpler
    calibration signal, complementing the sparsity-vs-variance correlation
    already checked elsewhere).
  - A sparsification curve: order query points by *predicted* variance,
    descending, progressively drop the most uncertain and track mean
    squared error of what's left, against the oracle (drop by *actual*
    error) and random-order baselines.
  - AUSE (Area Under the Sparsification Error curve): the standard metric,
    area between the BQ-ordering curve and the oracle curve -- 0 is
    perfect, closer to the random-order curve's area is worse.
  - Held-out Gaussian NLL under the real per-point BQ variance vs. under a
    constant-variance baseline (the population mean variance for every
    point) -- does the *shape* of the per-point variance actually help,
    not just its overall scale.

Needs torch + gsplat only insofar as the checkpoint was already trained;
this script itself is pure numpy/scipy.

Run: .venv-gsplat/bin/python gs_experiment/calibration_experiment.py <ply_path> --sigma 0.05 --window-radius 0.15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from gs_experiment.pixel_uncertainty import LocalUncertaintyEngine, make_default_3d_position_kernel
from gs_experiment.ply_io import read_3dgs_ply

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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


def run(
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ply_path")
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--window-radius", type=float, default=0.15)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    run(args.ply_path, n_samples=args.n_samples, sigma=args.sigma, window_radius=args.window_radius, label=args.label)


if __name__ == "__main__":
    main()
