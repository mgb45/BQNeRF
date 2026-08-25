"""ROADMAP.md milestone 2: the real differentiation experiment.

Runnable in two modes:
  - mock (default, no GPU/gsplat/torch needed): a synthetic scene with
    camera-index assignment by fiat, for pipeline development/regression
    testing.
  - real (--checkpoint <scene_dir>): a real trained gsplat checkpoint,
    loaded via gs_experiment.splat_scene.load_from_gsplat_checkpoint --
    e.g. one produced by gs_experiment.scene_spec.differentiation_scene +
    gs_experiment.blender_render + gs_experiment.train_minimal_gsplat.
    Per gs_experiment/README.md's "once GPU access is available" plan,
    nothing else in the pipeline changes: `scene` and its zone geometry
    are the only things that differ between modes, since SplatScene/
    splat_observations are the interface boundary.

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

Run (mock): .venv/bin/python gs_experiment/differentiation_experiment.py
Run (real): .venv-gsplat/bin/python gs_experiment/differentiation_experiment.py --checkpoint <scene_dir>
Output: gs_experiment/results/differentiation_experiment_{mock,real}.png
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

from bq_splat.kernels import DirectionalKernel
from gs_experiment.camera import directions_from_positions_to_camera, turntable_arc, turntable_ring
from gs_experiment.pixel_uncertainty import LocalUncertaintyEngine, make_default_3d_position_kernel
from gs_experiment.splat_scene import make_mock_scene, splat_observations
from gs_experiment.visibility_baseline import visibility_uncertainty_proxy

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _build_mock_scene(seed):
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

    narrow_typical_dir = directions_from_positions_to_camera(
        narrow_zone_center.reshape(1, -1), narrow_cameras[len(narrow_cameras) // 2]
    )[0]
    query_direction = -narrow_typical_dir

    return scene, dict(
        bounds=bounds,
        wide_zone_center=wide_zone_center,
        narrow_zone_center=narrow_zone_center,
        zone_radius=zone_radius,
        query_direction=query_direction,
        slice_z=0.0,
    )


def _build_real_scene(checkpoint_dir, separation=18.0, attribution_angular_tol=0.01):
    from gs_experiment.splat_scene import load_from_gsplat_checkpoint

    # visibility_attribution's default angular_tol (0.05) was validated
    # against make_occluder_scene's single isolated occluder/target pair;
    # empirically, against a cluster of ~14 closely-packed thin rods it
    # over-triggers the soft-z-buffer occlusion test massively (measured:
    # >99.9% of otherwise-valid (splat, camera) observations flagged
    # occluded at 0.05, vs. a few thousand real observations surviving at
    # 0.01) -- tight object packing, not a bug in the test itself, but a
    # scene-density sensitivity worth a smaller default here.
    scene = load_from_gsplat_checkpoint(checkpoint_dir, attribution_angular_tol=attribution_angular_tol)

    wide_zone_center = np.array([0.0, 0.0, 0.0])
    narrow_zone_center = np.array([separation, 0.0, 0.0])
    zone_radius = 1.5
    # bounds span both rod clusters (see scene_spec.differentiation_scene's
    # thin_rod_cluster spread=0.8, rod half-length ~0.25) plus margin
    bounds = ((-2.0, separation + 2.0), (-2.0, 2.0), (-1.5, 1.5))

    # _build_mock_scene picks a discriminating query direction by negating
    # a direction actually inside the narrow zone's observed cone -- that
    # only works when "negate" is equivalent to "azimuth the ring also
    # covers," which held for the mock scene's geometry but does NOT hold
    # here: both camera rigs share the same phi_deg elevation
    # (scene_spec.differentiation_scene), so negating a narrow-zone
    # direction flips elevation as well as azimuth and lands on a band
    # neither rig observes, showing no differentiation for either zone
    # (an empirically-found bug, not a hypothetical one -- both zones came
    # back with near-identical directional variance until this was
    # fixed). Instead, search every camera's direction-as-seen-from-the-
    # wide-cluster for the one least similar (by dot product) to anything
    # any camera's direction-as-seen-from-the-narrow-cluster looks like --
    # this is robust to whatever elevation/azimuth convention the rigs
    # actually use, and by construction favors a direction the wide ring's
    # full 360-degree sweep covers but the narrow arc's ~24-degree sweep
    # does not.
    wide_dirs = np.array(
        [directions_from_positions_to_camera(wide_zone_center.reshape(1, -1), cam)[0] for cam in scene.cameras]
    )
    narrow_dirs = np.array(
        [directions_from_positions_to_camera(narrow_zone_center.reshape(1, -1), cam)[0] for cam in scene.cameras]
    )
    max_similarity_to_narrow = (wide_dirs @ narrow_dirs.T).max(axis=1)
    query_direction = wide_dirs[np.argmin(max_similarity_to_narrow)]

    return scene, dict(
        bounds=bounds,
        wide_zone_center=wide_zone_center,
        narrow_zone_center=narrow_zone_center,
        zone_radius=zone_radius,
        query_direction=query_direction,
        slice_z=0.0,
    )


def run(
    scene=None,
    bounds=None,
    wide_zone_center=None,
    narrow_zone_center=None,
    zone_radius=1.2,
    query_direction=None,
    slice_z=0.0,
    seed=0,
    grid_res=35,
    window_radius=1.6,
    sigma=0.9,
    kappa=4.0,
    out_name="differentiation_experiment_mock.png",
    title="gs_experiment, mock scene: 3D positions + real camera poses (no GPU/gsplat needed)",
):
    if scene is None:
        scene, geom = _build_mock_scene(seed)
        bounds = geom["bounds"]
        wide_zone_center = geom["wide_zone_center"]
        narrow_zone_center = geom["narrow_zone_center"]
        zone_radius = geom["zone_radius"]
        query_direction = geom["query_direction"]
        slice_z = geom["slice_z"]
    elif any(v is None for v in (bounds, wide_zone_center, narrow_zone_center, query_direction)):
        raise ValueError("bounds/wide_zone_center/narrow_zone_center/query_direction are required when scene is given")

    positions, directions, values = splat_observations(scene)

    pos_kernel = make_default_3d_position_kernel(sigma=sigma)
    dir_kernel = DirectionalKernel(kappa=kappa)
    # Two engines, deliberately: splat_observations expands one row per
    # (splat, camera) pair, which is correct input for the directional
    # kernel (each camera really is a distinct direction observation) but
    # wrong for position-only variance -- "position-only, blind to
    # direction" should mean variance doesn't depend on how many cameras
    # saw a splat, and feeding camera-duplicated rows into spatial_only_
    # variance quietly breaks that (a splat seen by 30 cameras contributes
    # 30x the row-weight of one seen by 1, camera-count information
    # leaking into a signal that's supposed to be blind to it). Checked
    # empirically (gs_experiment/results/FINDINGS.md) whether this
    # actually changed the differentiation result -- it didn't
    # (deduplicated and duplicated versions agree closely) -- but the
    # deduplicated version is the conceptually correct one regardless, so
    # it's what position-only queries use here.
    spatial_engine = LocalUncertaintyEngine(
        positions=scene.positions, values=scene.colors, pos_kernel=pos_kernel, scene_bounds=bounds,
    )
    directional_engine = LocalUncertaintyEngine(
        positions=positions, values=values, pos_kernel=pos_kernel, scene_bounds=bounds,
        directions=directions, dir_kernel=dir_kernel,
    )

    (x0, x1), (y0, y1), _ = bounds
    margin = 0.5
    xs = np.linspace(x0 + margin, x1 - margin, grid_res)
    ys = np.linspace(y0 + margin, y1 - margin, grid_res)
    spatial_grid = np.full((grid_res, grid_res), np.nan)
    directional_grid = np.full((grid_res, grid_res), np.nan)
    visibility_grid = np.full((grid_res, grid_res), np.nan)

    for i, qx in enumerate(xs):
        for j, qy in enumerate(ys):
            q = np.array([qx, qy, slice_z])
            spatial_grid[j, i] = spatial_engine.spatial_only_variance(q, window_radius).variance
            directional_grid[j, i] = directional_engine.directional_variance(q, query_direction, window_radius).variance

            idx = directional_engine.local_neighbors(q, window_radius)
            visibility_grid[j, i] = visibility_uncertainty_proxy(directions[idx])

    # cached for reuse by other scripts (e.g. the milestone-3 pruning
    # experiment) that want per-splat BQ variance without paying for a
    # fresh set of BQ solves -- interpolating this already-computed grid
    # is orders of magnitude cheaper than re-querying LocalUncertaintyEngine
    # per splat, and precise enough for a splat-count-level comparison
    # rather than a per-pixel one.
    cache_path = RESULTS_DIR / (Path(out_name).stem + "_grid_cache.npz")
    np.savez(
        cache_path, xs=xs, ys=ys, slice_z=slice_z, spatial_grid=spatial_grid,
        directional_grid=directional_grid, visibility_grid=visibility_grid,
    )
    print(f"Saved grid cache {cache_path}")

    xx, yy = np.meshgrid(xs, ys)
    in_wide = np.linalg.norm(np.stack([xx, yy], axis=-1) - wide_zone_center[:2], axis=-1) < zone_radius
    in_narrow = np.linalg.norm(np.stack([xx, yy], axis=-1) - narrow_zone_center[:2], axis=-1) < zone_radius
    in_either_zone = in_wide | in_narrow

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    panel_titles = ["(a) position-only BQ variance\n(blind to direction)", "(b) position+direction BQ variance", "(c) visibility proxy (non-BQ)"]
    grids = [spatial_grid, directional_grid, visibility_grid]
    for ax, grid, panel_title in zip(axes, grids, panel_titles):
        # vmax capped to what's actually inside the two zones being
        # compared (with headroom), not the grid's global max: a region
        # far from either zone with too little local data to constrain
        # anything reads as very high variance (correctly -- that's not a
        # bug), but on a shared color scale it's a much larger swing than
        # the wide-vs-narrow comparison this plot exists to show, and
        # silently crushes both zones to the same-looking dark color.
        # Cells outside this range still render (clipped, not hidden),
        # just at the same saturated color -- the printed wide/narrow/
        # ratio numbers below are the actual quantitative comparison in
        # every case, this is a visibility fix for the figure only.
        zone_vmax = float(np.nanmax(grid[in_either_zone])) * 1.3 if in_either_zone.any() else None
        im = ax.imshow(grid, extent=[x0, x1, y0, y1], origin="lower", cmap="inferno", aspect="auto", vmin=0, vmax=zone_vmax)
        for center, color in [(wide_zone_center, "lime"), (narrow_zone_center, "cyan")]:
            ax.add_patch(plt.Circle(center[:2], zone_radius, fill=False, color=color, linewidth=2))
        ax.set_title(panel_title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    out = RESULTS_DIR / out_name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")

    for name, grid in [("position-only", spatial_grid), ("position+direction", directional_grid), ("visibility proxy", visibility_grid)]:
        wide_mean = np.nanmean(grid[in_wide])
        narrow_mean = np.nanmean(grid[in_narrow])
        print(f"{name:>20}: wide={wide_mean:.4f}  narrow={narrow_mean:.4f}  ratio(narrow/wide)={narrow_mean/wide_mean:.2f}x")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint", default=None, help="scene_dir for load_from_gsplat_checkpoint; omit to run the mock scene"
    )
    parser.add_argument("--separation", type=float, default=18.0, help="must match scene_spec.differentiation_scene's separation")
    parser.add_argument(
        "--angular-tol", type=float, default=0.01, help="visibility_attribution occlusion angular_tol (see _build_real_scene)"
    )
    args = parser.parse_args()

    if args.checkpoint is None:
        print("Running on a MOCK scene -- pass --checkpoint <scene_dir> for the real-data path.")
        run()
    else:
        print(f"Running on a REAL trained checkpoint: {args.checkpoint}")
        scene, geom = _build_real_scene(args.checkpoint, separation=args.separation, attribution_angular_tol=args.angular_tol)
        run(
            scene=scene,
            out_name="differentiation_experiment_real.png",
            title=f"gs_experiment, real trained checkpoint ({args.checkpoint})",
            **geom,
        )


if __name__ == "__main__":
    main()
