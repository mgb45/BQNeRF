"""Does BQ position-only variance track local splat sparsity, on real
reconstructed geometry? This is the more fundamental claim behind
"uncertainty nearly for free from recognizing rendering as Bayesian
quadrature": posterior variance in a GP-quadrature formulation is, by
construction, a function of how many/how well-spread the local quadrature
nodes (splats) are -- not a claim that needs thin-vs-thick geometric
classification to test, just a direct check that local node density and
BQ variance actually correlate the way the closed-form math says they
should, on a real checkpoint rather than only in the abstract.

Samples many query points across the real, trained checkpoint (not
filtered by splat scale, unlike real_benchmark_experiment.py's thin/thick
split), and for each measures two independent quantities:
  - local splat count within a fixed window (a trivial, non-BQ
    statistic -- just a KD-tree ball query)
  - BQ position-only variance at that same point, same window

then reports the correlation between them. A strong negative correlation
(sparse regions -> high variance) is the direct, minimal demonstration
that this project's central claim -- rendering-as-quadrature gives you a
real uncertainty signal essentially for free, closed-form, from the same
kernel structure already used to represent the scene -- actually holds on
real data, without needing any harder claim about *why* a region is
sparse (few views, thin geometry, or anything else).

Needs torch + gsplat only insofar as the checkpoint was already trained;
this script itself is pure numpy/scipy.

Run: .venv-gsplat/bin/python gs_experiment/sparsity_correlation_experiment.py <lego_prepared_dir>/wide/splats.ply
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
from scipy.stats import pearsonr, spearmanr

from gs_experiment.pixel_uncertainty import LocalUncertaintyEngine, make_default_3d_position_kernel
from gs_experiment.ply_io import read_3dgs_ply

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run(
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ply_path")
    parser.add_argument("--n-samples", type=int, default=150)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--window-radius", type=float, default=0.08)
    args = parser.parse_args()
    run(args.ply_path, n_samples=args.n_samples, sigma=args.sigma, window_radius=args.window_radius)


if __name__ == "__main__":
    main()
