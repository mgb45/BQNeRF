"""ROADMAP.md milestone 2 scaffold: the real differentiation experiment,
wired end-to-end and runnable now against a mock scene (no GPU, no
gsplat/torch needed), ready to point at `load_from_gsplat_checkpoint` once
a real trained scene and GPU access are available.

Reuses, without modification: bq_splat's kernels/quadrature (validated in
sections 1-9 of bq_splat/results/FINDINGS.md), the KD-tree + vv-caching
optimizations from benchmark_local_bq_scaling.py (now via
pixel_uncertainty.LocalUncertaintyEngine), and the same "hold spatial
density exactly equal between zones by construction" methodology
validated in scripts/validate_directional_combined.py -- extended here to
real 3D camera poses (camera.py's turntable helpers) instead of the toy's
2D angle parameterization.

Compares three signals over a 2D slice of the 3D scene:
  (a) position-only BQ variance -- blind to viewing direction
  (b) position+direction BQ variance -- queried from a direction chosen to
      lie outside the narrow zone's camera arc
  (c) a non-BQ visibility proxy (visibility_baseline.py) -- a genuinely
      different mechanism, for the "combination not competition" comparison
      ROADMAP.md calls for once this points at a real scene and a real
      alternative like GAVIS or PUP.

Run: .venv/bin/python gs_experiment/differentiation_experiment.py
Output: gs_experiment/results/differentiation_experiment_mock.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bq_splat.kernels import DirectionalKernel
from gs_experiment.camera import directions_from_positions_to_camera, turntable_arc, turntable_ring
from gs_experiment.pixel_uncertainty import LocalUncertaintyEngine, make_default_3d_position_kernel
from gs_experiment.splat_scene import make_mock_scene, splat_observations
from gs_experiment.visibility_baseline import visibility_uncertainty_proxy

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run(seed=0, grid_res=35, window_radius=1.6, sigma=0.9, kappa=4.0):
    rng = np.random.default_rng(seed)
    bounds = ((-5.0, 5.0), (-5.0, 5.0), (-1.0, 1.0))

    wide_cameras = turntable_ring(radius=8.0, n_views=14)
    narrow_cameras = turntable_arc(radius=8.0, n_views=14, theta_center_deg=0.0, half_width_deg=15.0)

    zone_radius = 1.2
    narrow_zone_center = np.array([2.5, 2.5, 0.0])
    wide_zone_center = np.array([-2.5, -2.5, 0.0])

    scene = make_mock_scene(
        rng,
        n_splats=350,
        bounds=bounds,
        wide_cameras=wide_cameras,
        narrow_cameras=narrow_cameras,
        narrow_zone_center=narrow_zone_center,
        narrow_zone_radius=zone_radius,
    )
    positions, directions, values = splat_observations(scene)

    pos_kernel = make_default_3d_position_kernel(sigma=sigma)
    dir_kernel = DirectionalKernel(kappa=kappa)
    engine = LocalUncertaintyEngine(
        positions=positions, values=values, pos_kernel=pos_kernel, scene_bounds=bounds,
        directions=directions, dir_kernel=dir_kernel,
    )

    # query direction chosen to lie outside the narrow zone's camera arc,
    # same pattern as validate_directional_combined.py: negate a direction
    # actually inside the observed cone.
    narrow_typical_dir = directions_from_positions_to_camera(
        narrow_zone_center.reshape(1, -1), narrow_cameras[len(narrow_cameras) // 2]
    )[0]
    query_direction = -narrow_typical_dir

    (x0, x1), (y0, y1), _ = bounds
    margin = 0.5
    xs = np.linspace(x0 + margin, x1 - margin, grid_res)
    ys = np.linspace(y0 + margin, y1 - margin, grid_res)
    spatial_grid = np.full((grid_res, grid_res), np.nan)
    directional_grid = np.full((grid_res, grid_res), np.nan)
    visibility_grid = np.full((grid_res, grid_res), np.nan)

    for i, qx in enumerate(xs):
        for j, qy in enumerate(ys):
            q = np.array([qx, qy, 0.0])
            spatial_grid[j, i] = engine.spatial_only_variance(q, window_radius).variance
            directional_grid[j, i] = engine.directional_variance(q, query_direction, window_radius).variance

            idx = engine.local_neighbors(q, window_radius)
            visibility_grid[j, i] = visibility_uncertainty_proxy(directions[idx])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    titles = ["(a) position-only BQ variance\n(blind to direction)", "(b) position+direction BQ variance", "(c) visibility proxy (non-BQ)"]
    grids = [spatial_grid, directional_grid, visibility_grid]
    for ax, grid, title in zip(axes, grids, titles):
        im = ax.imshow(grid, extent=[x0, x1, y0, y1], origin="lower", cmap="inferno")
        for center, color in [(wide_zone_center, "lime"), (narrow_zone_center, "cyan")]:
            ax.add_patch(plt.Circle(center[:2], zone_radius, fill=False, color=color, linewidth=2))
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("gs_experiment scaffold, mock scene: 3D positions + real camera poses (no GPU/gsplat needed)")
    fig.tight_layout()
    out = RESULTS_DIR / "differentiation_experiment_mock.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")

    xx, yy = np.meshgrid(xs, ys)
    in_wide = np.linalg.norm(np.stack([xx, yy], axis=-1) - wide_zone_center[:2], axis=-1) < zone_radius
    in_narrow = np.linalg.norm(np.stack([xx, yy], axis=-1) - narrow_zone_center[:2], axis=-1) < zone_radius

    for name, grid in [("position-only", spatial_grid), ("position+direction", directional_grid), ("visibility proxy", visibility_grid)]:
        wide_mean = np.nanmean(grid[in_wide])
        narrow_mean = np.nanmean(grid[in_narrow])
        print(f"{name:>20}: wide={wide_mean:.4f}  narrow={narrow_mean:.4f}  ratio(narrow/wide)={narrow_mean/wide_mean:.2f}x")


if __name__ == "__main__":
    print("Running on a MOCK scene -- see splat_scene.load_from_gsplat_checkpoint for the real-data path.")
    run()
