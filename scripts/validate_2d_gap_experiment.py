"""2D bridge experiment (see conversation / ROADMAP.md): rerun the toy
differentiation experiment from validate_milestone1.py's gap_experiment()
over an image-plane domain instead of a 1D ray, using splat centers with
GS-realistic 2D scatter instead of samples along a depth axis. Checks
whether the same "high variance in a well-observed-but-under-resolved
region" signal survives the move to 2D before ever touching gsplat.

Run: .venv/bin/python scripts/validate_2d_gap_experiment.py
Output: bq_splat/results/gap_experiment_2d.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bq_splat.kernels import ProductKernel, RBFKernel
from bq_splat.quadrature import bayesian_quadrature_nd
from bq_splat.toy_scene_2d import gap_nodes_2d, make_mixture_scene_2d

RESULTS_DIR = Path(__file__).resolve().parents[1] / "bq_splat" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run(seed=1, n_nodes=250, grid_res=45, window_radius=1.4):
    rng = np.random.default_rng(seed)
    domain = ((0.0, 10.0), (0.0, 10.0))
    (x0, x1), (y0, y1) = domain
    kernel = ProductKernel([RBFKernel(sigma=0.5), RBFKernel(sigma=0.5)])

    scene = make_mixture_scene_2d(rng, domain=domain, n_bumps=10, min_width=0.2, max_width=0.5)
    nodes, (gap_center, gap_radius) = gap_nodes_2d(
        rng, domain, n=n_nodes, gap_center_frac=(0.5, 0.5), gap_radius_frac=0.18, thin_prob=0.9
    )
    values = scene.g_true(nodes)

    margin = 0.3
    xs = np.linspace(x0 + margin, x1 - margin, grid_res)
    ys = np.linspace(y0 + margin, y1 - margin, grid_res)
    variance_grid = np.full((grid_res, grid_res), np.nan)

    for i, qx in enumerate(xs):
        for j, qy in enumerate(ys):
            lo_x, hi_x = max(x0, qx - window_radius), min(x1, qx + window_radius)
            lo_y, hi_y = max(y0, qy - window_radius), min(y1, qy + window_radius)
            mask = (nodes[:, 0] >= lo_x) & (nodes[:, 0] <= hi_x) & (nodes[:, 1] >= lo_y) & (nodes[:, 1] <= hi_y)
            local_nodes = nodes[mask]
            local_values = values[mask]
            result = bayesian_quadrature_nd(local_nodes, local_values, kernel, [(lo_x, hi_x), (lo_y, hi_y)])
            variance_grid[j, i] = result.variance  # row=y, col=x for imshow

    # true image, for reference
    img_res = 120
    img_xs = np.linspace(x0, x1, img_res)
    img_ys = np.linspace(y0, y1, img_res)
    grid_x, grid_y = np.meshgrid(img_xs, img_ys)
    true_img = scene.g_true(np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)).reshape(img_res, img_res)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].imshow(true_img, extent=[x0, x1, y0, y1], origin="lower", cmap="viridis")
    axes[0].scatter(nodes[:, 0], nodes[:, 1], s=6, color="white", edgecolor="black", linewidth=0.3, label="splat centers")
    gap_circle = plt.Circle(gap_center, gap_radius, fill=False, color="orange", linewidth=2, label="sparse-coverage gap")
    axes[0].add_patch(gap_circle)
    axes[0].set_title("true signal g(x,y) + splat centers")
    axes[0].legend(loc="upper right", fontsize=7)
    axes[0].set_xlim(x0, x1)
    axes[0].set_ylim(y0, y1)

    im = axes[1].imshow(variance_grid, extent=[x0, x1, y0, y1], origin="lower", cmap="inferno")
    gap_circle2 = plt.Circle(gap_center, gap_radius, fill=False, color="cyan", linewidth=2)
    axes[1].add_patch(gap_circle2)
    axes[1].set_title("local BQ posterior variance")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle("2D bridge experiment: sparse-but-visible region, image-plane domain")
    fig.tight_layout()
    out = RESULTS_DIR / "gap_experiment_2d.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")

    xx, yy = np.meshgrid(xs, ys)
    in_gap = np.sqrt((xx - gap_center[0]) ** 2 + (yy - gap_center[1]) ** 2) < gap_radius
    mean_in = np.nanmean(variance_grid[in_gap])
    mean_out = np.nanmean(variance_grid[~in_gap])
    print(f"Mean local BQ variance inside gap:  {mean_in:.5f}")
    print(f"Mean local BQ variance outside gap: {mean_out:.5f}")
    print(f"Ratio (inside / outside): {mean_in / mean_out:.2f}x")
    print(f"NaN fraction in variance grid: {np.mean(np.isnan(variance_grid)):.3f}")


if __name__ == "__main__":
    run()
