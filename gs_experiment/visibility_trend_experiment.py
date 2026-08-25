"""Does BQ position-only variance grow as training-view coverage shrinks, on
real reconstructed geometry? A graded version of
real_benchmark_experiment.py's two-point (wide/narrow) cross-checkpoint
comparison: five independently-trained checkpoints of the same real
object (NeRF-Synthetic lego) -- wide/rand50/rand25/rand12 at 100/50/25/12
training views, each a random subset of the same 100-view split (full
angular spread, just sparser), plus the original "narrow" checkpoint (12
views, angularly clustered) -- each evaluated at the *same* fixed set of
real-world query points.

First run with only wide/rand50/rand25/narrow found something more
specific than a smooth decay: BQ variance was nearly flat across
100->50->25 (random subsampling that keeps full angular spread barely
moves it) and only jumped sharply at narrow's 12 clustered views. rand12
was added specifically to isolate the two candidate explanations --
dropping to few views vs. dropping to few *clustered* views -- by holding
count fixed at 12 while varying only whether those 12 are random or
clustered.

The point isn't a claim about *why* a region is uncertain (that was
real_benchmark_experiment.py's harder, more specific thin/thick
question) -- it's the more fundamental, "nearly free" claim: the same
closed-form BQ variance, computed from nothing but each checkpoint's own
splat positions, tracks visibility coverage as a smooth trend, not just
a two-point difference.

Needs torch + gsplat only insofar as the checkpoints were already
trained; this script itself is pure numpy/scipy.

Run: .venv-gsplat/bin/python gs_experiment/visibility_trend_experiment.py <lego_prepared_dir>
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

from gs_experiment.pixel_uncertainty import LocalUncertaintyEngine, make_default_3d_position_kernel
from gs_experiment.ply_io import read_3dgs_ply

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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


def run(lego_dir: str, n_query_points: int = 150, sigma: float = 0.05, window_radius: float = 0.08, seed: int = 0):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lego_dir")
    parser.add_argument("--n-query-points", type=int, default=150)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--window-radius", type=float, default=0.08)
    args = parser.parse_args()
    run(args.lego_dir, n_query_points=args.n_query_points, sigma=args.sigma, window_radius=args.window_radius)


if __name__ == "__main__":
    main()
