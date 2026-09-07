"""Consolidated CLI for the hand-built-scene experiments -- everything
that operates on gs_experiment/scene_spec.py's hand-built scenes
(differentiation_scene, nbv_test_scene, gradient_scene) rather than a real
trained checkpoint alone, unlike most of the other experiment scripts.
Grouped here because they share that "designed scene" property but not
much else (each has its own CLI args, some need torch/gsplat and some
don't), so dispatch is by experiment name via subparsers rather than by
shared parameters.

Replaces (each experiment keeps its source script's own remaining CLI
args, verbatim math/plotting/report format -- this is a mechanical merge,
not a re-derivation):

  differentiation         <- differentiation_experiment.py
      ROADMAP.md milestone 2's go/no-go test: does position-only BQ
      variance flag a well-observed-but-poorly-resolved region? Mock mode
      (default, no GPU/gsplat/torch needed) or --checkpoint <scene_dir>
      for a real trained gsplat checkpoint.

  declustering-isolation  <- validate_declustering_isolation.py
      Controlled follow-up to `differentiation`: is the go/no-go result
      driven by view-count itself, or by the splat-clustering/redundancy
      confound? Post-hoc on an already-trained checkpoint.

  pruning                 <- pruning_experiment.py
      ROADMAP.md milestone 3: BQ variance + opacity-based pruning,
      compared against an opacity-only baseline at equal splat count.
      Needs a `differentiation --checkpoint <scene_dir>` grid cache first.

  nbv                     <- nbv_experiment.py
      ROADMAP.md milestone 4: BQ variance + a visibility proxy for
      next-best-view candidate scoring, on scene_spec.nbv_test_scene.

  directional-gradient     <- directional_gradient_experiment.py
      The designed-scene (scene_spec.gradient_scene) version of the
      coverage-gradient test: does directional BQ variance rise
      monotonically with a designed, real-camera view-coverage gradient?
      Two subcommands (`prepare`, `train-and-analyze`) since Blender's
      `bpy` can only run inside a separate `blender --background` process
      between them.

Not here: the real-geometry directional-coverage experiments (those read
real checkpoints only, no hand-built scene_spec scene) -- consolidated
separately into real_directional_coverage_experiment.py.

Run examples:
  .venv/bin/python -m gs_experiment.designed_scene_experiments differentiation
  .venv-gsplat/bin/python -m gs_experiment.designed_scene_experiments differentiation --checkpoint <scene_dir>
  .venv-gsplat/bin/python -m gs_experiment.designed_scene_experiments declustering-isolation <scene_dir>
  .venv-gsplat/bin/python -m gs_experiment.designed_scene_experiments pruning <scene_dir> --keep-counts 4000 6000 9000
  .venv-gsplat/bin/python -m gs_experiment.designed_scene_experiments nbv <nbv_dir> <info_npz>
  .venv-gsplat/bin/python -m gs_experiment.designed_scene_experiments directional-gradient prepare <out_dir>
  blender --background --python gs_experiment/blender_render.py -- <out_dir>/scene_spec.json <out_dir>
  .venv-gsplat/bin/python -m gs_experiment.designed_scene_experiments directional-gradient train-and-analyze <out_dir>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from bq_splat.kernels import DirectionalKernel
from gs_experiment.camera import (
    directions_from_positions_to_camera,
    translate_camera,
    turntable_arc,
    turntable_camera,
    turntable_ring,
)
from gs_experiment.nerf_transforms import load_transforms, write_transforms_json
from gs_experiment.pixel_uncertainty import LocalUncertaintyEngine, make_default_3d_position_kernel
from gs_experiment.scene_spec import gradient_scene
from gs_experiment.splat_scene import load_from_gsplat_checkpoint, make_mock_scene, splat_observations
from gs_experiment.visibility_baseline import resultant_length, visibility_uncertainty_proxy

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# differentiation  (differentiation_experiment.py)
# ---------------------------------------------------------------------------
# ROADMAP.md milestone 2: the real differentiation experiment.
#
# Runnable in two modes:
#   - mock (default, no GPU/gsplat/torch needed): a synthetic scene with
#     camera-index assignment by fiat, for pipeline development/regression
#     testing.
#   - real (--checkpoint <scene_dir>): a real trained gsplat checkpoint,
#     loaded via gs_experiment.splat_scene.load_from_gsplat_checkpoint --
#     e.g. one produced by gs_experiment.scene_spec.differentiation_scene +
#     gs_experiment.blender_render + gs_experiment.train_minimal_gsplat.
#     Per gs_experiment/README.md's "once GPU access is available" plan,
#     nothing else in the pipeline changes: `scene` and its zone geometry
#     are the only things that differ between modes, since SplatScene/
#     splat_observations are the interface boundary.
#
# Reuses, without modification: bq_splat's kernels/quadrature (validated in
# sections 1-9 of bq_splat/results/FINDINGS.md), the KD-tree + vv-caching
# optimizations from bq_splat/validate.py --check scaling (now via
# pixel_uncertainty.LocalUncertaintyEngine), and the same "hold spatial
# density exactly equal between zones by construction" methodology
# validated in bq_splat/validate.py --check directional-combined -- extended here to
# real 3D camera poses (camera.py's turntable helpers) instead of the toy's
# 2D angle parameterization.
#
# Compares three signals over a 2D slice of the 3D scene:
#   (a) position-only BQ variance -- blind to viewing direction
#   (b) position+direction BQ variance -- queried from a direction chosen to
#       lie outside the narrow zone's camera arc
#   (c) a non-BQ visibility proxy (visibility_baseline.py) -- a genuinely
#       different mechanism, for the "combination not competition" comparison
#       ROADMAP.md calls for once this points at a real scene and a real
#       alternative like GAVIS or PUP.
#
# Output: gs_experiment/results/differentiation_experiment_{mock,real}.png


def _differentiation_build_mock_scene(seed):
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


def _differentiation_build_real_scene(checkpoint_dir, separation=18.0, attribution_angular_tol=0.01):
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

    # _differentiation_build_mock_scene picks a discriminating query
    # direction by negating a direction actually inside the narrow zone's
    # observed cone -- that only works when "negate" is equivalent to
    # "azimuth the ring also covers," which held for the mock scene's
    # geometry but does NOT hold here: both camera rigs share the same
    # phi_deg elevation (scene_spec.differentiation_scene), so negating a
    # narrow-zone direction flips elevation as well as azimuth and lands on
    # a band neither rig observes, showing no differentiation for either
    # zone (an empirically-found bug, not a hypothetical one -- both zones
    # came back with near-identical directional variance until this was
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


def run_differentiation(
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
        scene, geom = _differentiation_build_mock_scene(seed)
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


# ---------------------------------------------------------------------------
# declustering-isolation  (validate_declustering_isolation.py)
# ---------------------------------------------------------------------------
# Controlled test for gs_experiment/results/FINDINGS.md section 12's open
# question: is the demonstrated go/no-go result (position-only BQ variance
# ranking the wide zone as more uncertain than the narrow zone) driven by
# view-count itself, or by the splat-clustering/redundancy confound flagged
# there as not yet isolated (more views -> more consistent gradient -> more
# densification cycles -> more spatially redundant splats, since
# clone/split placement puts children near their parent)?
#
# Post-hoc on the already-trained checkpoint, no retraining needed. Three
# conditions, same query points/kernel/window throughout, only the wide
# zone's splat population changes:
#
#   1. original wide zone (baseline, reproduces FINDINGS.md section 11)
#   2. wide zone randomly subsampled to the narrow zone's splat COUNT
#      (holds count fixed, doesn't touch spacing/redundancy pattern)
#   3. wide zone greedily declustered (Poisson-disk-style minimum-distance
#      rejection) to match the narrow zone's median nearest-neighbor
#      SPACING (changes both count and redundancy pattern together, since
#      they're coupled by construction here)
#
# If (2) still shows the wide-higher-variance effect but (3) removes or
# reverses it, that's evidence the effect is about spatial redundancy
# specifically, not raw count -- supporting FINDINGS.md section 12's leading
# hypothesis. If both (2) and (3) still show the effect, count/redundancy
# isn't the driver and something else (e.g. genuine coverage-extent
# differences) needs to be considered instead.


def greedy_decluster(positions: np.ndarray, min_dist: float, order: np.ndarray) -> np.ndarray:
    """Greedily keep points at least `min_dist` apart, processing in
    `order`. Poisson-disk-style rejection subsampling: not the only way
    to decluster, but simple and doesn't need a target count picked in
    advance -- `min_dist` directly controls the resulting spacing, which
    is the statistic being matched here. O(n * n_kept), fine at this
    scale (thousands of candidates, expected n_kept in the low
    thousands)."""
    n = positions.shape[0]
    keep = np.zeros(n, dtype=bool)
    kept = np.empty((n, 3))
    n_kept = 0
    for i in order:
        p = positions[i]
        if n_kept > 0:
            d = np.linalg.norm(kept[:n_kept] - p, axis=1).min()
            if d < min_dist:
                continue
        keep[i] = True
        kept[n_kept] = p
        n_kept += 1
    return keep


def median_nn_distance(positions: np.ndarray) -> float:
    from scipy.spatial import cKDTree

    tree = cKDTree(positions)
    dists, _ = tree.query(positions, k=2)
    return float(np.median(dists[:, 1]))


def run_declustering_isolation(scene_dir: str, zone_radius: float = 1.6, sigma: float = 0.9, window_radius: float = 1.6, seed: int = 0):
    scene = load_from_gsplat_checkpoint(scene_dir, attribution_angular_tol=0.01)
    positions, colors = scene.positions, scene.colors

    wide_center = np.array([0.0, 0.0, 0.0])
    narrow_center = np.array([18.0, 0.0, 0.0])
    d_wide = np.linalg.norm(positions - wide_center, axis=1)
    d_narrow = np.linalg.norm(positions - narrow_center, axis=1)
    wide_mask = d_wide < zone_radius
    narrow_mask = d_narrow < zone_radius
    other_mask = ~wide_mask & ~narrow_mask

    wide_pos, wide_col = positions[wide_mask], colors[wide_mask]
    narrow_pos, narrow_col = positions[narrow_mask], colors[narrow_mask]
    other_pos, other_col = positions[other_mask], colors[other_mask]

    narrow_spacing = median_nn_distance(narrow_pos)
    wide_spacing = median_nn_distance(wide_pos)
    print(f"baseline: wide n={len(wide_pos)} median_NN={wide_spacing:.5f}  narrow n={len(narrow_pos)} median_NN={narrow_spacing:.5f}")

    rng = np.random.default_rng(seed)
    bounds = tuple((positions[:, d].min() - 1.0, positions[:, d].max() + 1.0) for d in range(3))
    pos_kernel = make_default_3d_position_kernel(sigma=sigma)

    def zone_ratio(wide_pos_variant, wide_col_variant, label):
        all_pos = np.concatenate([wide_pos_variant, narrow_pos, other_pos], axis=0)
        all_col = np.concatenate([wide_col_variant, narrow_col, other_col], axis=0)
        engine = LocalUncertaintyEngine(positions=all_pos, values=all_col, pos_kernel=pos_kernel, scene_bounds=bounds, seed=seed)
        wide_var = engine.spatial_only_variance(wide_center, window_radius).variance
        narrow_var = engine.spatial_only_variance(narrow_center, window_radius).variance
        ratio = narrow_var / wide_var
        print(f"{label}: n_wide={len(wide_pos_variant)}  wide_var={wide_var:.4f}  narrow_var={narrow_var:.4f}  ratio={ratio:.3f}x")
        return ratio

    print()
    r1 = zone_ratio(wide_pos, wide_col, "(1) original wide")

    # (2) random subsample of wide to match narrow's count -- holds count
    # fixed, leaves spacing/redundancy pattern (relative to the original
    # population's own structure) otherwise unperturbed
    n_target = len(narrow_pos)
    sub_idx = rng.choice(len(wide_pos), size=min(n_target, len(wide_pos)), replace=False)
    r2 = zone_ratio(wide_pos[sub_idx], wide_col[sub_idx], "(2) wide random-subsampled to narrow's count")

    # (3) greedy decluster wide to match narrow's median spacing
    order = rng.permutation(len(wide_pos))
    keep = greedy_decluster(wide_pos, min_dist=narrow_spacing, order=order)
    declustered_spacing = median_nn_distance(wide_pos[keep]) if keep.sum() > 1 else float("nan")
    r3 = zone_ratio(
        wide_pos[keep], wide_col[keep],
        f"(3) wide declustered to spacing~{declustered_spacing:.5f} (target {narrow_spacing:.5f})",
    )

    print()
    print(f"Summary: ratio(narrow/wide) original={r1:.3f}x  count-matched={r2:.3f}x  spacing-matched={r3:.3f}x")
    print("If (3) moves toward 1x while (2) doesn't: redundancy/spacing is the driver, not raw count.")
    print("If both (2) and (3) stay well below 1x: count/redundancy isn't the (whole) story.")


# ---------------------------------------------------------------------------
# pruning  (pruning_experiment.py)
# ---------------------------------------------------------------------------
# ROADMAP.md milestone 3: densification/pruning combination experiment.
# "combination not competition": use BQ variance alongside a
# opacity/visibility-based pruning criterion, and check whether the
# combination reaches better reconstruction quality than the heuristic-only
# baseline at the same, reduced splat count.
#
# Post-hoc on an already-trained, already-densified checkpoint (no
# retraining): prune down to a target splat count two ways --
#
#   (a) opacity-only (the standard 3DGS heuristic: drop the lowest-opacity
#       splats first)
#   (b) opacity + BQ position-only variance, combined by rank: a splat with
#       moderate-to-high opacity but high local BQ variance (evidence its
#       neighborhood is still under-resolved) is protected from being
#       pruned as readily as an opacity-matched splat in a low-BQ-variance
#       (already well-resolved) region would be. The BQ term only applies
#       above `min_opacity_for_bq` (default 0.3, calibrated empirically --
#       see FINDINGS.md): BQ variance is *also* high in genuinely empty
#       space (little/no local data, correctly but unhelpfully for pruning
#       purposes), so applying it unconditionally protects near-zero-
#       opacity junk at low keep-counts, which measurably hurt PSNR before
#       this floor was added.
#
# then render both pruned checkpoints and compare PSNR against ground
# truth, at the *same* splat count -- the direct test of ROADMAP.md's
# "reaches equal quality at fewer splats" framing (equivalently: better
# quality at equal, reduced, splat count).
#
# BQ variance is read from the `differentiation` experiment's cached 2D
# grid (`<checkpoint>_grid_cache.npz`, produced by a
# `differentiation --checkpoint <scene_dir>` run against the same
# checkpoint) via interpolation, rather than recomputing fresh per-splat BQ
# solves for up to 15000 splats -- orders of magnitude cheaper, and precise
# enough for a splat-count-level comparison. Run
# `differentiation --checkpoint <scene_dir>` first if the cache doesn't
# exist yet.
#
# Needs torch + gsplat (requirements-gsplat.txt).


def load_bq_interpolator(cache_path: str):
    data = np.load(cache_path)
    xs, ys, grid = data["xs"], data["ys"], data["spatial_grid"]
    # spatial_grid[j, i] corresponds to (xs[i], ys[j]) -- run_differentiation's
    # convention (imshow-style, row=y, col=x) -- so the interpolator axes
    # are (ys, xs) in that order to match.
    return RegularGridInterpolator((ys, xs), grid, bounds_error=False, fill_value=None)


def rank_score(values: np.ndarray) -> np.ndarray:
    """Ascending rank, normalized to [0, 1] -- higher value -> higher score."""
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values))
    return ranks / max(len(values) - 1, 1)


def prune_checkpoint(checkpoint: dict, keep_idx: np.ndarray) -> dict:
    return dict(
        positions=checkpoint["positions"][keep_idx],
        scales=checkpoint["scales"][keep_idx],
        rotations=checkpoint["rotations"][keep_idx],
        opacities=checkpoint["opacities"][keep_idx],
        sh_coeffs=checkpoint["sh_coeffs"][keep_idx],
        sh_degree=checkpoint["sh_degree"],
    )


def render_and_score(pruned_checkpoint: dict, scene_dir: str, view_indices, tmp_root: str, label: str):
    from gs_experiment.ply_io import write_3dgs_ply
    from gs_experiment.render_reconstruction import render_views

    scene_copy = os.path.join(tmp_root, label)
    os.makedirs(scene_copy, exist_ok=True)
    write_3dgs_ply(os.path.join(scene_copy, "splats.ply"), **pruned_checkpoint)
    shutil.copy(os.path.join(scene_dir, "transforms.json"), os.path.join(scene_copy, "transforms.json"))
    if not os.path.exists(os.path.join(scene_copy, "images")):
        os.symlink(os.path.abspath(os.path.join(scene_dir, "images")), os.path.join(scene_copy, "images"))

    results, _ = render_views(scene_copy, view_indices)
    psnrs = []
    for i, gt, recon in results:
        mse = float(np.mean((gt - recon) ** 2))
        psnrs.append(-10.0 * np.log10(max(mse, 1e-10)))
    return float(np.mean(psnrs)), psnrs


def run_pruning(scene_dir: str, keep_counts, bq_weight: float = 1.0, min_opacity_for_bq: float = 0.0, view_indices=None, seed: int = 0):
    from gs_experiment.ply_io import read_3dgs_ply

    checkpoint = read_3dgs_ply(os.path.join(scene_dir, "splats.ply"))
    n_total = checkpoint["positions"].shape[0]
    cache_path = RESULTS_DIR / "differentiation_experiment_real_grid_cache.npz"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"{cache_path} not found -- run `differentiation --checkpoint {scene_dir}` first"
        )
    bq_interp = load_bq_interpolator(str(cache_path))

    positions = checkpoint["positions"]
    opacities = checkpoint["opacities"]
    bq_vals = bq_interp(np.stack([positions[:, 1], positions[:, 0]], axis=1))  # (y, x) order
    bq_vals = np.nan_to_num(bq_vals, nan=float(np.nanmedian(bq_vals)))

    opacity_score = rank_score(opacities)
    bq_score = rank_score(bq_vals)
    # BQ variance is also high in genuinely empty space (little/no local
    # data -> high posterior variance, correctly, but not usefully for a
    # pruning decision -- protecting near-zero-opacity splats sitting in
    # empty space wastes keep-budget on splats that barely render at all).
    # Restricting the BQ boost to splats that already clear a minimal
    # opacity floor keeps BQ voting among plausible candidates rather than
    # among obvious junk.
    bq_eligible = opacities > min_opacity_for_bq
    combined_score = opacity_score + bq_weight * bq_score * bq_eligible

    view_indices = view_indices or [0, 8, 16, 24, 32, 41, 45, 48]

    print(f"checkpoint: {n_total} splats, {scene_dir}")
    with tempfile.TemporaryDirectory() as tmp_root:
        for keep_count in keep_counts:
            keep_count = min(keep_count, n_total)

            opacity_only_idx = np.argsort(opacity_score)[-keep_count:]
            combined_idx = np.argsort(combined_score)[-keep_count:]

            saved_by_bq = np.setdiff1d(combined_idx, opacity_only_idx)
            print(
                f"\n--- keep_count={keep_count} ---\n"
                f"{len(saved_by_bq)} splats kept by the BQ-combined criterion that opacity-only would have "
                f"pruned (mean opacity of those: {opacities[saved_by_bq].mean():.3f}, "
                f"mean BQ variance: {bq_vals[saved_by_bq].mean():.3f} vs. scene median {np.median(bq_vals):.3f})"
            )

            pruned_opacity = prune_checkpoint(checkpoint, opacity_only_idx)
            pruned_combined = prune_checkpoint(checkpoint, combined_idx)

            psnr_opacity, _ = render_and_score(pruned_opacity, scene_dir, view_indices, tmp_root, f"opacity_{keep_count}")
            psnr_combined, _ = render_and_score(pruned_combined, scene_dir, view_indices, tmp_root, f"combined_{keep_count}")

            print(f"opacity-only PSNR:   {psnr_opacity:.2f}dB")
            print(f"BQ-combined PSNR:    {psnr_combined:.2f}dB")
            print(f"delta (combined - opacity-only): {psnr_combined - psnr_opacity:+.2f}dB")


# ---------------------------------------------------------------------------
# nbv  (nbv_experiment.py)
# ---------------------------------------------------------------------------
# ROADMAP.md milestone 4: active-view / next-best-view (NBV) combination
# experiment. "Use BQ variance alongside a visibility proxy for
# candidate-view scoring; check whether the combined signal selects views
# that improve reconstruction in under-resolved regions faster than either
# signal alone."
#
# Uses scene_spec.nbv_test_scene: a single thin-rod cluster observed from a
# narrow training arc, a discrete pool of candidate next-view poses, and a
# disjoint held-out evaluation ring (never a candidate, never trained on
# until it's used for evaluation).
#
# Pipeline:
#   1. Train a baseline checkpoint on the training arc alone.
#   2. Score every candidate view two ways, using the baseline checkpoint's
#      real splat positions/observed directions (no retraining needed for
#      scoring itself -- this is BQ's actual practical advantage, "closed-
#      form, essentially free to compute", exercised for real here):
#        (a) BQ: position+direction variance at the cluster center, queried
#            at the candidate's viewing direction -- high variance means
#            that direction is under-covered by the training arc.
#        (b) visibility: how much adding the candidate's direction would
#            reduce the mean resultant length of the already-observed
#            direction set (bigger reduction = more angular-diversity
#            gain) -- a genuinely different, non-BQ mechanism, computed
#            with visibility_baseline.resultant_length.
#        (c) combined: normalized sum of (a) and (b).
#   3. Retrain two more checkpoints -- training arc + the top-combined
#      candidate, training arc + the worst-combined (most redundant)
#      candidate -- and evaluate all three (baseline, +best, +worst) on the
#      held-out eval ring via PSNR, to check whether the BQ+visibility
#      combination actually picks a view that helps more than a poor one.
#
# Needs torch + gsplat (requirements-gsplat.txt).


def make_subset_scene_dir(source_dir: str, frame_indices, out_dir: str):
    camera_angle_x, frames = load_transforms(os.path.join(source_dir, "transforms.json"))
    subset = [{"file_path": frames[i][0], "transform_matrix": frames[i][1]} for i in frame_indices]
    os.makedirs(out_dir, exist_ok=True)
    write_transforms_json(os.path.join(out_dir, "transforms.json"), camera_angle_x, subset)
    images_link = os.path.join(out_dir, "images")
    if not os.path.exists(images_link):
        os.symlink(os.path.abspath(os.path.join(source_dir, "images")), images_link)


def score_candidates(baseline_dir: str, radius: float, window_radius: float = 1.6, angular_tol: float = 0.01):
    scene = load_from_gsplat_checkpoint(baseline_dir, attribution_angular_tol=angular_tol)
    positions, directions, values = splat_observations(scene)

    pos_margin = 1.0
    bounds = tuple((positions[:, d].min() - pos_margin, positions[:, d].max() + pos_margin) for d in range(3))
    pos_kernel = make_default_3d_position_kernel(sigma=0.9)
    dir_kernel = DirectionalKernel(kappa=4.0)
    engine = LocalUncertaintyEngine(
        positions=positions, values=values, pos_kernel=pos_kernel, scene_bounds=bounds,
        directions=directions, dir_kernel=dir_kernel,
    )

    center = np.zeros(3)
    existing_idx = engine.local_neighbors(center, window_radius)
    existing_dirs = directions[existing_idx]
    base_resultant = resultant_length(existing_dirs)
    print(f"baseline: {len(existing_idx)} local (splat,camera) observations, resultant_length={base_resultant:.4f}")

    return engine, center, existing_dirs, base_resultant


def run_nbv(nbv_dir: str, info_path: str, radius: float = 6.5, n_iters: int = 3000, seed: int = 0):
    from gs_experiment.render_reconstruction import render_views
    from gs_experiment.train_minimal_gsplat import train

    info = dict(np.load(info_path))
    train_idx = info["train_idx"].tolist()
    candidate_idx = info["candidate_idx"].tolist()
    eval_idx = info["eval_idx"].tolist()
    candidate_thetas = info["candidate_thetas"]

    baseline_dir = os.path.join(nbv_dir, "baseline")
    make_subset_scene_dir(nbv_dir, train_idx, baseline_dir)

    print("=== training baseline (train arc alone) ===")
    train(
        baseline_dir, os.path.join(baseline_dir, "splats.ply"),
        n_splats=1500, bounds=((-2.5, 2.5), (-2.5, 2.5), (-2.5, 2.5)), sh_degree=1, n_iters=n_iters, seed=seed,
        init_scale=0.1, opacity_reg_weight=0.003, densify=True, densify_interval=300, densify_start=300,
        densify_grad_percentile=80.0, min_opacity=0.005, max_splats=6000, log_every=1000,
    )

    engine, center, existing_dirs, base_resultant = score_candidates(baseline_dir, radius=radius)

    bq_scores, vis_scores = [], []
    for theta in candidate_thetas:
        cam = turntable_camera(radius, 35.0, float(theta))
        cand_dir = directions_from_positions_to_camera(center.reshape(1, -1), cam)[0]
        bq_scores.append(engine.directional_variance(center, cand_dir, 1.6).variance)
        new_resultant = resultant_length(np.vstack([existing_dirs, cand_dir]))
        vis_scores.append(base_resultant - new_resultant)  # positive = diversity gain

    bq_scores = np.array(bq_scores)
    vis_scores = np.array(vis_scores)
    bq_norm = (bq_scores - bq_scores.min()) / (np.ptp(bq_scores) + 1e-12)
    vis_norm = (vis_scores - vis_scores.min()) / (np.ptp(vis_scores) + 1e-12)
    combined = bq_norm + vis_norm

    print("\ntheta   bq_var   vis_gain   combined")
    for t, b, v, c in sorted(zip(candidate_thetas, bq_scores, vis_scores, combined), key=lambda r: -r[3]):
        print(f"{t:6.1f}  {b:7.3f}  {v:8.4f}  {c:7.3f}")

    rank_corr = np.corrcoef(np.argsort(np.argsort(bq_scores)), np.argsort(np.argsort(vis_scores)))[0, 1]
    print(f"\nBQ vs visibility candidate-ranking correlation: {rank_corr:.3f}")

    best_local = int(np.argmax(combined))
    worst_local = int(np.argmin(combined))
    best_global = candidate_idx[best_local]
    worst_global = candidate_idx[worst_local]
    print(
        f"best candidate: theta={candidate_thetas[best_local]:.1f} (combined={combined[best_local]:.3f})  "
        f"worst candidate: theta={candidate_thetas[worst_local]:.1f} (combined={combined[worst_local]:.3f})"
    )

    results = {}
    for label, extra_idx in [("baseline", []), ("plus_best", [best_global]), ("plus_worst", [worst_global])]:
        scene_dir = os.path.join(nbv_dir, label)
        if extra_idx:
            make_subset_scene_dir(nbv_dir, train_idx + extra_idx, scene_dir)
            print(f"\n=== training {label} ===")
            train(
                scene_dir, os.path.join(scene_dir, "splats.ply"),
                n_splats=1500, bounds=((-2.5, 2.5), (-2.5, 2.5), (-2.5, 2.5)), sh_degree=1, n_iters=n_iters, seed=seed,
                init_scale=0.1, opacity_reg_weight=0.003, densify=True, densify_interval=300, densify_start=300,
                densify_grad_percentile=80.0, min_opacity=0.005, max_splats=6000, log_every=1000,
            )
        else:
            scene_dir = baseline_dir  # already trained above

        eval_scene_dir = os.path.join(nbv_dir, f"{label}_eval")
        make_subset_scene_dir(nbv_dir, eval_idx, eval_scene_dir)
        shutil.copy(os.path.join(scene_dir, "splats.ply"), os.path.join(eval_scene_dir, "splats.ply"))

        eval_results, _ = render_views(eval_scene_dir, list(range(len(eval_idx))))
        psnrs = [-10.0 * np.log10(max(float(np.mean((gt - recon) ** 2)), 1e-10)) for _, gt, recon in eval_results]
        results[label] = float(np.mean(psnrs))
        print(f"{label}: mean held-out PSNR over {len(eval_idx)} eval views = {results[label]:.2f}dB")

    print("\n=== summary ===")
    print(f"baseline (train arc alone):        {results['baseline']:.2f}dB")
    print(f"+ best (BQ+visibility combined):   {results['plus_best']:.2f}dB  (delta {results['plus_best']-results['baseline']:+.2f}dB)")
    print(f"+ worst (most redundant candidate): {results['plus_worst']:.2f}dB  (delta {results['plus_worst']-results['baseline']:+.2f}dB)")


# ---------------------------------------------------------------------------
# directional-gradient  (directional_gradient_experiment.py)
# ---------------------------------------------------------------------------
# A realistic experiment with views deliberately chosen to produce a
# view-direction *uncertainty gradient*, not the binary wide-vs-narrow split
# every prior directional result in this project used (toy scale:
# bq_splat/results/FINDINGS.md section 9; real scale: gs_experiment/
# results/FINDINGS.md sections 17-19, 22). `scene_spec.gradient_scene`
# builds 5 identical thin-rod clusters (spatial density held equal across
# zones, isolating the directional effect the same way `differentiation_scene`
# and `validate_directional_combined.py` already do) along a line, each
# observed by its own turntable-arc camera rig, all centered on the *same*
# azimuth but with angular half-width increasing linearly from zone 0
# (narrowest, most under-covered) to the last zone (widest, effectively a
# full ring) -- a real, monotonic angular-coverage gradient, real Blender
# rendering, real gsplat training, not a synthetic/toy signal.
#
# A single fixed query direction -- the azimuth diametrically opposite the
# shared arc center -- is genuinely consistent across every zone (since
# `theta_center_deg` doesn't vary, unlike each zone's own local convention),
# computed the same robust way `differentiation`'s real-scene builder does
# (a real camera pose's direction-to-a-point, not hand-derived spherical
# trigonometry -- see _differentiation_build_real_scene's comment about a
# real elevation bug from doing it the naive way).
#
# Pipeline (three separate steps, since Blender's `bpy` can only run inside
# a Blender process):
#   1. `directional-gradient prepare <out_dir>`
#      -- builds the scene spec + zone metadata, writes JSON.
#   2. `blender --background --python gs_experiment/blender_render.py -- <out_dir>/scene_spec.json <out_dir>`
#      -- real rendering (see that module's docstring).
#   3. `directional-gradient train-and-analyze <out_dir>`
#      -- real gsplat training with densification, then queries directional
#      and position-only BQ variance at each zone's center, reporting
#      whether variance actually rises monotonically with the designed
#      coverage gradient.


def directional_gradient_prepare(out_dir: str, n_zones: int = 5, n_views_per_zone: int = 10, radius: float = 6.5):
    os.makedirs(out_dir, exist_ok=True)
    spec, info = gradient_scene(n_zones=n_zones, n_views_per_zone=n_views_per_zone, radius=radius)

    with open(os.path.join(out_dir, "scene_spec.json"), "w") as fh:
        json.dump(spec.to_json_dict(), fh, indent=2)

    np.savez(
        os.path.join(out_dir, "gradient_info.npz"),
        zone_centers=info["zone_centers"], half_widths_deg=info["half_widths_deg"],
        zone_camera_ranges=np.array(info["zone_camera_ranges"]),
        theta_center_deg=info["theta_center_deg"], query_theta_deg=info["query_theta_deg"], radius=radius,
    )
    print(f"wrote {len(spec.objects)} objects, {len(spec.cameras)} cameras across {n_zones} zones to {out_dir}")
    print(f"half-widths (deg): {info['half_widths_deg']}")
    print(f"shared query azimuth: {info['query_theta_deg']:.1f} deg")
    print(f"\nNext: blender --background --python gs_experiment/blender_render.py -- "
          f"{out_dir}/scene_spec.json {out_dir}")


def directional_gradient_train_and_analyze(
    out_dir: str, n_splats: int = 1500, n_iters: int = 3000, seed: int = 0,
    window_radius: float = 1.6, sigma: float = 0.9, kappa: float = 4.0, min_opacity: float = 0.1,
):
    from gs_experiment.train_minimal_gsplat import train

    info = np.load(os.path.join(out_dir, "gradient_info.npz"))
    zone_centers = info["zone_centers"]
    half_widths_deg = info["half_widths_deg"]
    query_theta_deg = float(info["query_theta_deg"])
    radius = float(info["radius"])
    n_zones = len(zone_centers)

    ply_path = os.path.join(out_dir, "splats.ply")
    if not os.path.exists(ply_path):
        print("training...")
        span = float(zone_centers[:, 0].max() - zone_centers[:, 0].min())
        bounds = ((-2.0, span + 2.0), (-2.5, 2.5), (-2.5, 2.5))
        train(
            out_dir, ply_path, n_splats=n_splats, bounds=bounds, sh_degree=1, n_iters=n_iters, seed=seed,
            init_scale=0.1, opacity_reg_weight=0.003, densify=True, densify_interval=300, densify_start=300,
            densify_grad_percentile=80.0, min_opacity=0.005, max_splats=8000, log_every=500,
        )
    else:
        print(f"reusing existing checkpoint at {ply_path}")

    scene = load_from_gsplat_checkpoint(out_dir, attribution_angular_tol=0.01)
    positions, directions, values = splat_observations(scene)
    bounds3d = tuple((positions[:, d].min() - 1.0, positions[:, d].max() + 1.0) for d in range(3))

    pos_kernel = make_default_3d_position_kernel(sigma=sigma)
    dir_kernel = DirectionalKernel(kappa=kappa)
    engine = LocalUncertaintyEngine(
        positions=positions, values=values, pos_kernel=pos_kernel, scene_bounds=bounds3d,
        directions=directions, dir_kernel=dir_kernel,
    )

    directional_vars, spatial_vars = [], []
    for center in zone_centers:
        query_cam = translate_camera(turntable_camera(radius, 35.0, query_theta_deg), center)
        query_direction = directions_from_positions_to_camera(center.reshape(1, -1), query_cam)[0]

        dir_result = engine.directional_variance(center, query_direction, window_radius)
        spatial_result = engine.spatial_only_variance(center, window_radius)
        directional_vars.append(dir_result.variance)
        spatial_vars.append(spatial_result.variance)

    directional_vars = np.array(directional_vars)
    spatial_vars = np.array(spatial_vars)

    print(f"\n{'zone':>4}{'half-width (deg)':>18}{'directional var':>18}{'spatial-only var':>18}")
    for i in range(n_zones):
        print(f"{i:>4}{half_widths_deg[i]:>18.1f}{directional_vars[i]:>18.5f}{spatial_vars[i]:>18.5f}")

    is_monotonic = bool(np.all(np.diff(directional_vars) <= 1e-12))
    spearman_rho = float(np.corrcoef(np.argsort(np.argsort(-half_widths_deg)), np.argsort(np.argsort(directional_vars)))[0, 1])
    spatial_range_ratio = float(spatial_vars.max() / max(spatial_vars.min(), 1e-12))

    print(f"\ndirectional variance strictly monotonically decreasing with half-width: {is_monotonic}")
    print(f"rank correlation (narrowing half-width vs. rising directional variance): rho={spearman_rho:.3f}")
    print(f"directional variance range (narrowest/widest zone): {directional_vars[0] / max(directional_vars[-1], 1e-12):.2f}x")
    print(f"spatial-only variance max/min across zones (should be small -- geometry is matched): {spatial_range_ratio:.2f}x")

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(half_widths_deg, directional_vars, "o-", color="tab:red", label="directional variance")
    ax1.set_xlabel("zone's camera-arc half-width (deg) -- designed coverage gradient")
    ax1.set_ylabel("position+direction BQ variance", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.plot(half_widths_deg, spatial_vars, "s--", color="tab:blue", label="spatial-only variance (control)")
    ax2.set_ylabel("position-only BQ variance", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    fig.suptitle("Directional BQ variance vs. a designed view-coverage gradient\n(5 zones, identical geometry, shared query direction)")
    fig.tight_layout()
    out_path = RESULTS_DIR / "directional_gradient.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_path}")


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="experiment", required=True, metavar="experiment")

    p_diff = sub.add_parser("differentiation", help="go/no-go: position-only BQ variance vs. a poorly-resolved region")
    p_diff.add_argument(
        "--checkpoint", default=None, help="scene_dir for load_from_gsplat_checkpoint; omit to run the mock scene"
    )
    p_diff.add_argument("--separation", type=float, default=18.0, help="must match scene_spec.differentiation_scene's separation")
    p_diff.add_argument(
        "--angular-tol", type=float, default=0.01, help="visibility_attribution occlusion angular_tol (see _differentiation_build_real_scene)"
    )

    p_decl = sub.add_parser("declustering-isolation", help="isolate the redundancy confound in the differentiation result")
    p_decl.add_argument("scene_dir")
    p_decl.add_argument("--seed", type=int, default=0)

    p_prune = sub.add_parser("pruning", help="BQ variance + opacity-based pruning vs. opacity-only")
    p_prune.add_argument("scene_dir")
    p_prune.add_argument("--keep-counts", type=int, nargs="+", default=[4000, 6000, 9000])
    p_prune.add_argument("--bq-weight", type=float, default=1.0)
    p_prune.add_argument(
        "--min-opacity-for-bq", type=float, default=0.3,
        help="calibrated empirically (see gs_experiment/results/FINDINGS.md): too low and BQ-combined "
        "protects near-zero-opacity splats in empty space, hurting PSNR at loose budgets; 0.3 gave a "
        "clean win at tight budgets and a no-op (never worse) at loose ones",
    )

    p_nbv = sub.add_parser("nbv", help="BQ variance + visibility proxy for next-best-view candidate scoring")
    p_nbv.add_argument("nbv_dir")
    p_nbv.add_argument("info_path")
    p_nbv.add_argument("--radius", type=float, default=6.5)
    p_nbv.add_argument("--n-iters", type=int, default=3000)
    p_nbv.add_argument("--seed", type=int, default=0)

    p_dg = sub.add_parser("directional-gradient", help="designed-scene coverage-gradient test (scene_spec.gradient_scene)")
    dg_sub = p_dg.add_subparsers(dest="cmd", required=True)
    p_dg_prepare = dg_sub.add_parser("prepare")
    p_dg_prepare.add_argument("out_dir")
    p_dg_prepare.add_argument("--n-zones", type=int, default=5)
    p_dg_prepare.add_argument("--n-views-per-zone", type=int, default=10)
    p_dg_train = dg_sub.add_parser("train-and-analyze")
    p_dg_train.add_argument("out_dir")
    p_dg_train.add_argument("--n-iters", type=int, default=3000)
    p_dg_train.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if args.experiment == "differentiation":
        if args.checkpoint is None:
            print("Running on a MOCK scene -- pass --checkpoint <scene_dir> for the real-data path.")
            run_differentiation()
        else:
            print(f"Running on a REAL trained checkpoint: {args.checkpoint}")
            scene, geom = _differentiation_build_real_scene(
                args.checkpoint, separation=args.separation, attribution_angular_tol=args.angular_tol
            )
            run_differentiation(
                scene=scene,
                out_name="differentiation_experiment_real.png",
                title=f"gs_experiment, real trained checkpoint ({args.checkpoint})",
                **geom,
            )
    elif args.experiment == "declustering-isolation":
        run_declustering_isolation(args.scene_dir, seed=args.seed)
    elif args.experiment == "pruning":
        run_pruning(args.scene_dir, args.keep_counts, bq_weight=args.bq_weight, min_opacity_for_bq=args.min_opacity_for_bq)
    elif args.experiment == "nbv":
        run_nbv(args.nbv_dir, args.info_path, radius=args.radius, n_iters=args.n_iters, seed=args.seed)
    elif args.experiment == "directional-gradient":
        if args.cmd == "prepare":
            directional_gradient_prepare(args.out_dir, n_zones=args.n_zones, n_views_per_zone=args.n_views_per_zone)
        elif args.cmd == "train-and-analyze":
            directional_gradient_train_and_analyze(args.out_dir, n_iters=args.n_iters, seed=args.seed)


if __name__ == "__main__":
    main()
