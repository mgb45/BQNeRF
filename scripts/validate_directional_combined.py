"""The actual claim from the conversation, tested directly: can two regions
with IDENTICAL spatial splat density be told apart by whether they were
observed from a wide spread of directions or a narrow cone -- something a
position-only kernel is blind to by construction, but a position+direction
kernel should catch.

Setup: splat positions scattered at roughly uniform density over the whole
2D domain (no spatial "gap" this time -- spatial coverage is deliberately
matched everywhere, unlike validate_2d_gap_experiment.py). Two equal-size
circular zones get different angular treatment: "wide" zone splats are each
observed from directions spread across most of the circle; "narrow" zone
splats are each observed from a tight cone. Every other splat gets wide
coverage too. Then:

  (a) pure spatial-only BQ variance (bayesian_quadrature_nd, direction
      ignored) over the domain -- should look similar in both zones, since
      spatial density is matched.
  (b) position+direction BQ variance (bayesian_quadrature_directional),
      queried at one fixed "novel" direction chosen to lie outside the
      narrow zone's cone -- should spike specifically over the narrow zone.

Run: .venv/bin/python scripts/validate_directional_combined.py
Output: bq_splat/results/directional_combined.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bq_splat.kernels import DirectionalKernel, ProductKernel, RBFKernel
from bq_splat.quadrature import bayesian_quadrature_directional, bayesian_quadrature_nd
from bq_splat.toy_scene_2d import make_mixture_scene_2d

RESULTS_DIR = Path(__file__).resolve().parents[1] / "bq_splat" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def angle_to_unit_vector(theta):
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    return np.stack([np.cos(theta), np.sin(theta)], axis=1)


def build_scene(rng, domain, n_background=150, n_per_zone=18, n_dirs_per_splat=6, narrow_half_width=0.3, zone_radius=1.4, window_radius=1.6):
    """Two attempts at "spatial density matched" zones failed before this
    one, and it's worth recording why: independently random placement
    (even with equal *counts* per zone) can still differ in how *spread
    out* those points are within the disk -- 18 points is small enough for
    that clumping-by-chance to matter, and it showed up as a 0.32x
    position-only-variance ratio between zones meant to be identical.

    The fix used here is exact rather than statistical: both zones get the
    IDENTICAL set of relative offsets from their own center (one offset
    pattern, generated once, translated twice). Since a stationary kernel's
    behavior only depends on relative positions (the same fact that made
    caching `vv` by window size exact in benchmark_local_bq_scaling.py),
    this guarantees position-only BQ variance is equal between zones up to
    floating point, not just on average. Background splats are kept
    outside both zones' (zone_radius + window_radius) so they can't leak
    into either zone's local windows and reintroduce an asymmetry.
    """
    (x0, x1), (y0, y1) = domain
    wide_center = np.array([3.0, 3.0])
    narrow_center = np.array([7.0, 7.0])
    narrow_cone_center_theta = 0.0  # narrow zone's splats are all seen from near theta=0
    exclusion_radius = zone_radius + window_radius

    raw_background = np.stack([rng.uniform(x0, x1, size=n_background * 3), rng.uniform(y0, y1, size=n_background * 3)], axis=1)
    far_enough = (
        (np.linalg.norm(raw_background - wide_center, axis=1) > exclusion_radius)
        & (np.linalg.norm(raw_background - narrow_center, axis=1) > exclusion_radius)
    )
    background_positions = raw_background[far_enough][:n_background]

    offsets = []
    while len(offsets) < n_per_zone:
        o = rng.uniform(-zone_radius, zone_radius, size=2)
        if np.linalg.norm(o) < zone_radius:
            offsets.append(o)
    offsets = np.array(offsets)

    all_positions, all_directions, all_values = [], [], []
    scene = make_mixture_scene_2d(rng, domain=domain, n_bumps=8, min_width=0.4, max_width=0.9)

    def add_splat(p, thetas):
        value = float(scene.g_true(p.reshape(1, -1))[0])
        for theta in thetas:
            all_positions.append(p)
            all_directions.append(theta)
            all_values.append(value)

    for p in background_positions:
        add_splat(p, rng.uniform(-np.pi, np.pi, size=n_dirs_per_splat))
    for offset in offsets:
        add_splat(wide_center + offset, rng.uniform(-np.pi, np.pi, size=n_dirs_per_splat))
    for offset in offsets:
        thetas = rng.uniform(
            narrow_cone_center_theta - narrow_half_width, narrow_cone_center_theta + narrow_half_width,
            size=n_dirs_per_splat,
        )
        add_splat(narrow_center + offset, thetas)

    return (
        np.array(all_positions),
        angle_to_unit_vector(np.array(all_directions)),
        np.array(all_values),
        scene,
        dict(wide_center=wide_center, wide_radius=zone_radius, narrow_center=narrow_center, narrow_radius=zone_radius),
    )


def run(seed=0, grid_res=40, window_radius=1.6, kappa=4.0):
    rng = np.random.default_rng(seed)
    domain = ((0.0, 10.0), (0.0, 10.0))
    (x0, x1), (y0, y1) = domain

    positions, directions, values, scene, zones = build_scene(rng, domain)

    pos_kernel = ProductKernel([RBFKernel(sigma=0.6), RBFKernel(sigma=0.6)])
    dir_kernel = DirectionalKernel(kappa=kappa)
    query_direction = angle_to_unit_vector(np.pi)  # deliberately outside the narrow zone's cone (centered at 0)

    margin = 0.3
    xs = np.linspace(x0 + margin, x1 - margin, grid_res)
    ys = np.linspace(y0 + margin, y1 - margin, grid_res)
    spatial_only_grid = np.full((grid_res, grid_res), np.nan)
    directional_grid = np.full((grid_res, grid_res), np.nan)

    for i, qx in enumerate(xs):
        for j, qy in enumerate(ys):
            lo_x, hi_x = max(x0, qx - window_radius), min(x1, qx + window_radius)
            lo_y, hi_y = max(y0, qy - window_radius), min(y1, qy + window_radius)
            mask = (
                (positions[:, 0] >= lo_x) & (positions[:, 0] <= hi_x)
                & (positions[:, 1] >= lo_y) & (positions[:, 1] <= hi_y)
            )
            local_positions = positions[mask]
            local_directions = directions[mask]
            local_values = values[mask]
            bounds = [(lo_x, hi_x), (lo_y, hi_y)]

            spatial_result = bayesian_quadrature_nd(local_positions, local_values, pos_kernel, bounds)
            spatial_only_grid[j, i] = spatial_result.variance

            dir_result = bayesian_quadrature_directional(
                local_positions, local_directions, local_values, pos_kernel, dir_kernel, bounds, query_direction
            )
            directional_grid[j, i] = dir_result.variance

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    img_res = 100
    img_x, img_y = np.meshgrid(np.linspace(x0, x1, img_res), np.linspace(y0, y1, img_res))
    true_img = scene.g_true(np.stack([img_x.ravel(), img_y.ravel()], axis=1)).reshape(img_res, img_res)
    axes[0].imshow(true_img, extent=[x0, x1, y0, y1], origin="lower", cmap="viridis")
    unique_positions = np.unique(positions, axis=0)
    axes[0].scatter(unique_positions[:, 0], unique_positions[:, 1], s=4, color="white", edgecolor="black", linewidth=0.2)
    for center, radius, color, label in [
        (zones["wide_center"], zones["wide_radius"], "lime", "wide-angle zone"),
        (zones["narrow_center"], zones["narrow_radius"], "orange", "narrow-cone zone"),
    ]:
        axes[0].add_patch(plt.Circle(center, radius, fill=False, color=color, linewidth=2, label=label))
    axes[0].set_title("true signal + splat positions\n(spatial density matched everywhere)")
    axes[0].legend(loc="upper left", fontsize=7)
    axes[0].set_xlim(x0, x1)
    axes[0].set_ylim(y0, y1)

    im1 = axes[1].imshow(spatial_only_grid, extent=[x0, x1, y0, y1], origin="lower", cmap="inferno")
    for center, radius, color in [(zones["wide_center"], zones["wide_radius"], "lime"), (zones["narrow_center"], zones["narrow_radius"], "cyan")]:
        axes[1].add_patch(plt.Circle(center, radius, fill=False, color=color, linewidth=2))
    axes[1].set_title("(a) position-only BQ variance\n(blind to direction)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(directional_grid, extent=[x0, x1, y0, y1], origin="lower", cmap="inferno")
    for center, radius, color in [(zones["wide_center"], zones["wide_radius"], "lime"), (zones["narrow_center"], zones["narrow_radius"], "cyan")]:
        axes[2].add_patch(plt.Circle(center, radius, fill=False, color=color, linewidth=2))
    axes[2].set_title(f"(b) position+direction BQ variance\n(queried from theta=pi, outside narrow cone)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle("Combined experiment: does directionality catch what position-only BQ misses?")
    fig.tight_layout()
    out = RESULTS_DIR / "directional_combined.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")

    xx, yy = np.meshgrid(xs, ys)
    in_wide = np.linalg.norm(np.stack([xx, yy], axis=-1) - zones["wide_center"], axis=-1) < zones["wide_radius"]
    in_narrow = np.linalg.norm(np.stack([xx, yy], axis=-1) - zones["narrow_center"], axis=-1) < zones["narrow_radius"]

    print("\n--- position-only (spatial) BQ variance ---")
    print(f"wide zone:   mean={np.nanmean(spatial_only_grid[in_wide]):.5f}")
    print(f"narrow zone: mean={np.nanmean(spatial_only_grid[in_narrow]):.5f}")
    print(f"ratio (narrow/wide): {np.nanmean(spatial_only_grid[in_narrow]) / np.nanmean(spatial_only_grid[in_wide]):.2f}x")

    print("\n--- position+direction BQ variance (queried at theta=pi) ---")
    print(f"wide zone:   mean={np.nanmean(directional_grid[in_wide]):.5f}")
    print(f"narrow zone: mean={np.nanmean(directional_grid[in_narrow]):.5f}")
    print(f"ratio (narrow/wide): {np.nanmean(directional_grid[in_narrow]) / np.nanmean(directional_grid[in_wide]):.2f}x")


if __name__ == "__main__":
    run()
