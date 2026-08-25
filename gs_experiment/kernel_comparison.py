"""RBF vs. Matern-3/2 kernel comparison on a real trained gsplat
checkpoint -- ROADMAP.md flags this as unresolved: the kernel-choice
question bq_splat/results/FINDINGS.md sections 5-7 validated at toy scale
("fitted Matern beats Riemann outright at n=20/40 nodes... RBF's earlier
per-scene gains were mostly overfitting") was never run against a real
trained GS checkpoint. This reuses gs_experiment.differentiation_
experiment's real-scene machinery unmodified -- only the position kernel
changes between runs, everything else (scene, cameras, zones, query
direction, neighbor cap) is held fixed, so any difference in the reported
numbers is attributable to the kernel choice itself.

Run: .venv-gsplat/bin/python gs_experiment/kernel_comparison.py <scene_dir> [--separation 18.0] [--angular-tol 0.01]
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

from gs_experiment.differentiation_experiment import _build_real_scene
from gs_experiment.pixel_uncertainty import LocalUncertaintyEngine, make_default_3d_matern_kernel, make_default_3d_position_kernel
from gs_experiment.splat_scene import splat_observations

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def compare(checkpoint_dir, separation=18.0, attribution_angular_tol=0.01, grid_res=35, window_radius=1.6, bandwidth=0.9):
    scene, geom = _build_real_scene(checkpoint_dir, separation=separation, attribution_angular_tol=attribution_angular_tol)
    positions, directions, values = splat_observations(scene)

    bounds = geom["bounds"]
    wide_zone_center = geom["wide_zone_center"]
    narrow_zone_center = geom["narrow_zone_center"]
    zone_radius = geom["zone_radius"]
    slice_z = geom["slice_z"]

    (x0, x1), (y0, y1), _ = bounds
    margin = 0.5
    xs = np.linspace(x0 + margin, x1 - margin, grid_res)
    ys = np.linspace(y0 + margin, y1 - margin, grid_res)
    xx, yy = np.meshgrid(xs, ys)
    in_wide = np.linalg.norm(np.stack([xx, yy], axis=-1) - wide_zone_center[:2], axis=-1) < zone_radius
    in_narrow = np.linalg.norm(np.stack([xx, yy], axis=-1) - narrow_zone_center[:2], axis=-1) < zone_radius

    kernels = {
        "RBF": make_default_3d_position_kernel(sigma=bandwidth),
        "Matern-3/2": make_default_3d_matern_kernel(rho=bandwidth),
    }

    grids = {}
    summary = {}
    for name, pos_kernel in kernels.items():
        engine = LocalUncertaintyEngine(positions=positions, values=values, pos_kernel=pos_kernel, scene_bounds=bounds)
        grid = np.full((grid_res, grid_res), np.nan)
        for i, qx in enumerate(xs):
            for j, qy in enumerate(ys):
                q = np.array([qx, qy, slice_z])
                grid[j, i] = engine.spatial_only_variance(q, window_radius).variance
        grids[name] = grid
        wide_mean = float(np.nanmean(grid[in_wide]))
        narrow_mean = float(np.nanmean(grid[in_narrow]))
        summary[name] = dict(wide=wide_mean, narrow=narrow_mean, ratio=narrow_mean / wide_mean)
        print(f"{name:>12}: wide={wide_mean:.4f}  narrow={narrow_mean:.4f}  ratio(narrow/wide)={narrow_mean / wide_mean:.2f}x")

    rbf_grid, matern_grid = grids["RBF"], grids["Matern-3/2"]
    finite = np.isfinite(rbf_grid) & np.isfinite(matern_grid)
    corr = float(np.corrcoef(rbf_grid[finite], matern_grid[finite])[0, 1])
    print(f"RBF vs Matern-3/2 position-only variance, Pearson correlation across the grid: {corr:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    # independent per-panel scaling, deliberately: RBF and Matern-3/2
    # posterior variance live on very different absolute scales for the
    # same nominal bandwidth (less-smooth Matern shrinks less per nearby
    # observation), so a shared vmax makes RBF's own within-panel
    # structure vanish -- the printed wide/narrow/ratio numbers and the
    # cross-kernel correlation are the actual quantitative comparison,
    # not a visual scale match.
    for ax, name in zip(axes, kernels):
        im = ax.imshow(grids[name], extent=[x0, x1, y0, y1], origin="lower", cmap="inferno", aspect="auto", vmin=0)
        for center, color in [(wide_zone_center, "lime"), (narrow_zone_center, "cyan")]:
            ax.add_patch(plt.Circle(center[:2], zone_radius, fill=False, color=color, linewidth=2))
        r = summary[name]["ratio"]
        ax.set_title(f"{name} position-only variance\nratio(narrow/wide)={r:.2f}x", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"RBF vs Matern-3/2, same real checkpoint ({checkpoint_dir}), correlation={corr:.3f}")
    fig.tight_layout()
    out = RESULTS_DIR / "kernel_comparison_real.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")

    return summary, corr


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene_dir")
    parser.add_argument("--separation", type=float, default=18.0)
    parser.add_argument("--angular-tol", type=float, default=0.01)
    parser.add_argument("--bandwidth", type=float, default=0.9)
    args = parser.parse_args()
    compare(args.scene_dir, separation=args.separation, attribution_angular_tol=args.angular_tol, bandwidth=args.bandwidth)


if __name__ == "__main__":
    main()
