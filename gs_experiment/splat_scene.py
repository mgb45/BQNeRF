"""A loaded (or mocked) 3D Gaussian Splatting scene.

Scope note, worth being explicit about: this operates on 3D world-space
splat positions and 3D world-space query points, not 2D image-plane
pixels. Mapping a world-space uncertainty value back to a specific
camera's per-pixel image is a reprojection step (each pixel's uncertainty
would come from the world-space point(s) its ray intersects, weighted by
the same alpha-compositing weights gsplat already computes during
rendering) -- deferred until this is wired to a live gsplat renderer,
since that projection/ray-intersection logic is exactly what gsplat's own
rasterizer already provides and shouldn't be reimplemented here.

Also worth restating: `scales`/`rotations` are kept as metadata for
standard rendering, but are NOT fed into the BQ kernel's bandwidth. The
validated BQ machinery uses one shared or pooled-fit bandwidth
(see gs_experiment/results/FINDINGS.md), not per-splat
heterogeneous covariances -- using each splat's own learned covariance as
its own kernel bandwidth is a real, mathematically plausible extension
(closer to the original derivation's "splats as weighted kernel nodes"
framing) but it is a second, unvalidated change; stacking it on top of the
GPU/gsplat integration at the same time would make it hard to tell which
change caused which result. Left as documented future work.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from gs_experiment.camera import CameraPose, directions_from_positions_to_camera
from gs_experiment.spherical_harmonics import eval_sh
from gs_experiment.visibility_attribution import (
    attribute_observations,
    invert_to_observed_camera_idx,
    subsample_observed_camera_idx,
)


@dataclass
class SplatScene:
    positions: np.ndarray  # (N, 3)
    colors: np.ndarray  # (N,) scalar -- ignored if sh_coeffs is set (see splat_observations)
    opacities: np.ndarray  # (N,)
    scales: np.ndarray  # (N, 3) -- metadata only, see module docstring
    rotations: np.ndarray  # (N, 4) quaternions -- metadata only, see module docstring

    # Observations: which cameras plausibly saw each splat, needed for the
    # directional kernel. observed_camera_idx[i] is a list of indices into
    # `cameras` for splat i. Real data would derive this from each
    # training view's actual contribution (e.g. non-negligible rendering
    # weight) rather than "every camera sees every splat" -- the mock
    # scene below approximates it by simple visibility (in front of the
    # camera, not distance-gated) since it isn't rendering anything for real.
    observed_camera_idx: List[np.ndarray]
    cameras: List[CameraPose]

    # Optional: real spherical-harmonic color, (N, n_channels, n_coeffs).
    # When set, splat_observations evaluates genuinely view-dependent color
    # per observation instead of falling back to the flat `colors` field.
    sh_coeffs: Optional[np.ndarray] = None
    sh_degree: int = 0


def make_mock_scene(
    rng: np.random.Generator,
    n_splats: int = 200,
    bounds=((-5.0, 5.0), (-5.0, 5.0), (-1.0, 1.0)),
    wide_cameras: Optional[List[CameraPose]] = None,
    narrow_cameras: Optional[List[CameraPose]] = None,
    narrow_zone_center: Optional[np.ndarray] = None,
    narrow_zone_radius: float = 1.5,
) -> SplatScene:
    """Synthetic scene for testing the gs_experiment pipeline without any
    real gsplat checkpoint or GPU. Splats scatter uniformly in `bounds`;
    splats within `narrow_zone_radius` of `narrow_zone_center` are marked
    as observed only by `narrow_cameras`, everything else by
    `wide_cameras` -- two zones with identical spatial splat density but
    different angular coverage, isolating the directional signal from the
    spatial one.
    """
    (x0, x1), (y0, y1), (z0, z1) = bounds
    positions = np.stack(
        [rng.uniform(x0, x1, n_splats), rng.uniform(y0, y1, n_splats), rng.uniform(z0, z1, n_splats)], axis=1
    )
    colors = rng.uniform(0.2, 1.0, n_splats)
    opacities = rng.uniform(0.5, 1.0, n_splats)
    scales = rng.uniform(0.02, 0.08, size=(n_splats, 3))
    rotations = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_splats, 1))  # identity quaternions

    wide_cameras = wide_cameras or []
    narrow_cameras = narrow_cameras or []
    all_cameras = wide_cameras + narrow_cameras
    wide_idx = np.arange(len(wide_cameras))
    narrow_idx = np.arange(len(wide_cameras), len(all_cameras))

    observed_camera_idx = []
    for p in positions:
        if narrow_zone_center is not None and np.linalg.norm(p - narrow_zone_center) < narrow_zone_radius:
            observed_camera_idx.append(narrow_idx if len(narrow_idx) > 0 else wide_idx)
        else:
            observed_camera_idx.append(wide_idx if len(wide_idx) > 0 else narrow_idx)

    return SplatScene(
        positions=positions,
        colors=colors,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        observed_camera_idx=observed_camera_idx,
        cameras=all_cameras,
    )


def splat_observations(scene: SplatScene, include_render_attrs: bool = False):
    """Expand a SplatScene into parallel (position, direction, value)
    arrays -- one row per (splat, observing-camera) pair -- the input
    format the directional kernel (DirectionalKernel, combined with a
    position kernel in gs_experiment.quadrature) expects.

    `value` is genuinely view-dependent (`eval_sh(scene.sh_coeffs[i],
    direction, scene.sh_degree)`) when `scene.sh_coeffs` is set; otherwise
    it falls back to the flat `scene.colors[i]` for every observation of a
    splat, same as before SH support existed. The flat-color path is a
    known simplification (the same value regardless of viewing direction),
    kept only for scenes that don't set sh_coeffs.

    `include_render_attrs=True` additionally returns `opacities`/`scales`/
    `rotations` arrays, each splat's own value repeated once per observing
    camera (same indexing as `positions`/`directions`/`values` above) --
    for callers that need to construct a `LocalUncertaintyEngine` with
    `opacities=`/`scales=`/`rotations=` set from this expanded, directional
    form (rendering_aware_variance_along_ray_directional and friends need
    real per-candidate opacity/covariance, not just position/direction/
    value). Default stays `False` so every existing 3-tuple-unpacking
    caller is unaffected.

    Vectorized per camera, not per (splat, camera) pair: an earlier version
    called `directions_from_positions_to_camera`/`eval_sh` once per row in a
    nested Python loop (both already accept batched array input, so this
    bought nothing) -- profiled at >120s of a 277s real-checkpoint uncertainty-
    map call, 3.4M single-row calls, before this fix. Grouping rows by camera
    (typically ~100 groups, not ~3.4M rows) and writing into a pre-sized
    output array via boolean-mask assignment preserves the exact original
    splat-major row order (mask assignment lands each camera's rows back at
    their original flat positions, regardless of which order the groups are
    processed in) -- same output, not an approximation.
    """
    n_per_splat = np.array([len(idx) for idx in scene.observed_camera_idx], dtype=np.int64)
    n_splats = len(scene.observed_camera_idx)
    splat_idx_flat = np.repeat(np.arange(n_splats), n_per_splat)
    cam_idx_flat = np.concatenate(scene.observed_camera_idx) if n_splats > 0 else np.array([], dtype=np.int64)

    positions_flat = scene.positions[splat_idx_flat]
    directions_flat = np.empty((splat_idx_flat.shape[0], 3), dtype=float)
    values_flat = np.empty(splat_idx_flat.shape[0], dtype=float) if scene.sh_coeffs is not None else None
    for c in np.unique(cam_idx_flat):
        rows = cam_idx_flat == c
        directions_flat[rows] = directions_from_positions_to_camera(positions_flat[rows], scene.cameras[c])
        if scene.sh_coeffs is not None:
            # Evaluated per camera group, same as directions above: eval_sh on
            # the full flat array would gather scene.sh_coeffs[splat_idx_flat],
            # duplicating every splat's (n_channels, n_coeffs) coefficients once
            # per observing camera -- for a dense checkpoint that's a
            # (n_splat_camera_pairs, 3, 16) float64 array, tens of GB at
            # millions of splats. Per camera, splat_idx_flat[rows] has no
            # duplicates (each splat appears once per camera it's observed by),
            # so this bounds peak memory by the camera with the most observed
            # splats instead of the sum over all cameras.
            colors = eval_sh(scene.sh_coeffs[splat_idx_flat[rows]], directions_flat[rows], scene.sh_degree)
            values_flat[rows] = colors.mean(axis=-1)  # collapse channels to one scalar, matching the flat-color path

    if scene.sh_coeffs is None:
        values_flat = scene.colors[splat_idx_flat]

    if include_render_attrs:
        return (
            positions_flat, directions_flat, values_flat,
            scene.opacities[splat_idx_flat], scene.scales[splat_idx_flat], scene.rotations[splat_idx_flat],
        )
    return positions_flat, directions_flat, values_flat


# Two independent, additive costs make up load_from_gsplat_checkpoint's/
# splat_observations' host RSS, both measured directly (not estimated) against
# real checkpoints from this project's own local_runs/lego_prepared/ -- one at
# 1,000,000 splats, one at 3,000,000 (the exact checkpoint a --budgets 3000000
# run of splat_budget_uncertainty_sweep.py produced right before OOM-killing the
# host, repeatedly, on a 30GB machine):
#
#   n_splats=1,000,000, cap=20  -> 19,683,660 rows: attribution baseline
#       2757 MiB, +2467 MiB after row-expansion (125.4 bytes/row marginal).
#   n_splats=3,000,000, cap=20  -> 59,168,609 rows: attribution baseline
#       7007 MiB, +6948 MiB after row-expansion (123.1 bytes/row marginal).
#
# The row-expansion cost is what `max_observations_per_splat` (above, in
# load_from_gsplat_checkpoint) caps; the attribution-baseline cost (reading the
# checkpoint + gpu_visibility_attribution's per-camera index lists -- scales
# with n_splats, not rows) is NOT bounded by that cap, and at 3M splats alone
# it's already ~7GB -- larger than the whole memory budget below. That's the
# actual failure mode a row-only guard misses: a budget whose *row* count fits
# under a cap can still have an unrelated, unbounded *baseline* cost that
# doesn't. Fit as a line through the two measured points above (both terms, in
# bytes):
_ATTRIBUTION_FIXED_OVERHEAD_BYTES = 700 * 1024 * 1024  # process/import/checkpoint-read floor
_ATTRIBUTION_BYTES_PER_SPLAT = 2200  # measured ~2125 B/splat marginal; rounded up for margin
_BYTES_PER_OBSERVATION_ROW = 140  # measured ~123-125 B/row; rounded up for margin

# Target ceiling for ONE load_from_gsplat_checkpoint + splat_observations call.
# compute_uncertainty_maps calls this path once per (checkpoint, sigma) pair, so
# a caller scoring the same checkpoint at both a fixed and a refit sigma pays
# this twice in one process -- keep it well under a small machine's free RAM
# (not just under it) to leave room for both calls plus normal desktop load and
# matplotlib/render buffers. Lower this further on a smaller machine; raise it
# only after remeasuring against a real checkpoint at the new budget, not by
# just guessing a bigger number.
PER_CALL_MEMORY_BUDGET_BYTES = 5 * 1024**3
MIN_OBSERVATIONS_PER_SPLAT = 8  # floor -- below this the directional kernel sees too few real
# observations per splat for its per-camera fit to mean much, regardless of memory pressure.


def _attribution_baseline_bytes(n_splats: int) -> int:
    return _ATTRIBUTION_FIXED_OVERHEAD_BYTES + _ATTRIBUTION_BYTES_PER_SPLAT * n_splats


def max_observations_per_splat_for_budget(budget: int, n_training_views: int = 100) -> Optional[int]:
    """None (no cap) whenever `n_training_views` itself already bounds
    per-splat observation count below what PER_CALL_MEMORY_BUDGET_BYTES
    allows -- keeps small budgets byte-for-byte reproducible rather than
    introducing subsampling noise where it isn't needed.

    Raises ValueError if `budget`'s attribution baseline alone (before a
    single observation row is added) already exceeds the memory budget --
    that budget can't be made safe by capping rows at all, regardless of
    how low; refuse rather than silently under-cap it (see the module-level
    comment above for why a row-only guard misses exactly this case).
    """
    baseline = _attribution_baseline_bytes(budget)
    if baseline >= PER_CALL_MEMORY_BUDGET_BYTES:
        raise ValueError(
            f"budget={budget:,}: attribution alone costs an estimated {baseline / 1024**3:.1f}GB, "
            f"already at or over PER_CALL_MEMORY_BUDGET_BYTES={PER_CALL_MEMORY_BUDGET_BYTES / 1024**3:.1f}GB "
            f"-- no observation cap can make this budget safe. This is the exact shape of budget that "
            f"OOM-killed the host repeatedly before this guard existed; lower the budget rather than bypass this."
        )
    remaining = PER_CALL_MEMORY_BUDGET_BYTES - baseline
    max_rows = remaining // _BYTES_PER_OBSERVATION_ROW
    cap = max(MIN_OBSERVATIONS_PER_SPLAT, max_rows // budget)
    if cap * budget * _BYTES_PER_OBSERVATION_ROW > remaining and cap == MIN_OBSERVATIONS_PER_SPLAT:
        raise ValueError(
            f"budget={budget:,}: even the MIN_OBSERVATIONS_PER_SPLAT={MIN_OBSERVATIONS_PER_SPLAT} floor "
            f"would exceed the remaining {remaining / 1024**3:.1f}GB after attribution's "
            f"{baseline / 1024**3:.1f}GB baseline -- refuse rather than exceed PER_CALL_MEMORY_BUDGET_BYTES."
        )
    return None if cap >= n_training_views else cap


def fit_kernel_hyperparams(
    scene: SplatScene,
    sigma_bounds=(0.005, 1.0),
    kappa_bounds=(0.05, 20.0),
    n_windows: int = 25,
    max_window_size: int = 60,
    window_radius: float = 0.08,
    min_opacity: float = 0.1,
    min_observations_for_kappa: int = 3,
    seed: int = 0,
):
    """Marginal-likelihood-fit both the position kernel's bandwidth (sigma) and
    the directional kernel's concentration (kappa) against THIS scene's own real
    data, instead of reusing a value pooled once across a different calibration
    set (gs_experiment.hyperparams.fit_kernel_param_pooled_nd -- the same
    procedure a held-out check once validated for sigma against a real
    checkpoint's own local windows, and the same procedure the project's
    original pooled kappa fit used, applied per-scene here instead of pooled
    across a fixed set of checkpoints; see git history for both). Exists
    because a bandwidth tuned at one checkpoint's splat density/coverage has no
    reason to be right for a checkpoint at a very different density (confirmed
    directly: a 300k-splat-tuned sigma is a real bandwidth mismatch at 500
    splats -- an earlier, retired sweep script first demonstrated this with an
    explicit fixed-vs-refit-sigma comparison for sigma alone; this generalizes
    that to kappa and to every caller of compute_uncertainty_maps, not just
    one sweep).

    Sigma comes from local (position, color) windows: `n_windows` splats above
    `min_opacity` are sampled as window centers, each paired with its neighbors
    within `window_radius` (capped at `max_window_size`, a random subsample,
    not a truncation, so the fit isn't spatially biased toward whichever
    neighbors happen to sort first).

    Kappa comes from per-splat multi-view (direction, color) groups: splats
    above `min_opacity` with at least `min_observations_for_kappa` observing
    cameras are sampled (up to `n_windows` of them), each contributing one
    window of (viewing direction, observed color) pairs across its own real
    observing cameras -- whether color varies with viewing direction is a
    per-splat question, unlike sigma's spatial-neighborhood one, so kappa's
    windows are per-splat groups, not spatial neighborhoods.

    Returns `(sigma, kappa)`, either `None` if there wasn't enough real data to
    fit it (too few above-threshold splats for sigma; too few multi-view
    splats for kappa) -- callers should fall back to a documented default in
    that case, not silently use an ill-fit value.
    """
    from scipy.spatial import cKDTree

    from gs_experiment.hyperparams import fit_kernel_param_pooled_nd
    from gs_experiment.kernels import DirectionalKernel, ProductKernel, RBFKernel

    rng = np.random.default_rng(seed)
    keep = scene.opacities > min_opacity
    positions = scene.positions[keep]
    colors = scene.colors[keep]
    opac_idx = np.nonzero(keep)[0]

    sigma = None
    if len(positions) >= 6:
        tree = cKDTree(positions)
        query_idx = rng.choice(len(positions), size=min(n_windows, len(positions)), replace=False)
        sigma_datasets = []
        for p in positions[query_idx]:
            idx = np.array(tree.query_ball_point(p, window_radius), dtype=int)
            if len(idx) < 6:
                continue
            if len(idx) > max_window_size:
                idx = rng.choice(idx, size=max_window_size, replace=False)
            sigma_datasets.append((positions[idx], colors[idx]))
        if sigma_datasets:
            fit = fit_kernel_param_pooled_nd(
                sigma_datasets, lambda s: ProductKernel([RBFKernel(sigma=s)] * 3), bounds=sigma_bounds, n_grid=25,
            )
            sigma = float(fit.param)

    kappa = None
    eligible = [i for i in opac_idx if len(scene.observed_camera_idx[i]) >= min_observations_for_kappa]
    if eligible:
        chosen = rng.choice(eligible, size=min(n_windows, len(eligible)), replace=False)
        kappa_datasets = []
        for i in chosen:
            cams = scene.observed_camera_idx[i]
            if len(cams) > max_window_size:
                cams = rng.choice(cams, size=max_window_size, replace=False)
            directions = np.stack(
                [directions_from_positions_to_camera(scene.positions[i][None, :], scene.cameras[c])[0] for c in cams]
            )
            if scene.sh_coeffs is not None:
                colors_i = eval_sh(scene.sh_coeffs[i][None, :, :], directions, scene.sh_degree).mean(axis=-1)
            else:
                colors_i = np.full(len(cams), scene.colors[i])
            kappa_datasets.append((directions, colors_i))
        fit = fit_kernel_param_pooled_nd(
            kappa_datasets, lambda k: DirectionalKernel(kappa=k), bounds=kappa_bounds, n_grid=25,
        )
        kappa = float(fit.param)

    return sigma, kappa


def make_occluder_scene(rng: np.random.Generator, n_wall_splats: int = 60, n_target_splats: int = 40, n_cameras_per_side: int = 6):
    """A more realistic scene than make_mock_scene's zone-based fiat
    assignment: a "wall" of splats at x=wall_x, a cluster of "target"
    splats behind it, front cameras (which the wall should occlude the
    targets from) and back cameras (which should see the targets directly,
    nothing in the way). observed_camera_idx comes from real frustum +
    occlusion attribution (visibility_attribution.py), not an assignment
    rule -- this is the integration test that SH color and real visibility
    attribution actually compose with the rest of the pipeline, not just
    that each works in isolation.
    """
    from gs_experiment.camera import CameraPose
    from gs_experiment.spherical_harmonics import random_sh_coeffs

    wall_x = 3.0
    wall_y = rng.uniform(-2.0, 2.0, n_wall_splats)
    wall_z = rng.uniform(-2.0, 2.0, n_wall_splats)
    wall_positions = np.stack([np.full(n_wall_splats, wall_x), wall_y, wall_z], axis=1)

    target_x = rng.uniform(wall_x + 1.0, wall_x + 2.5, n_target_splats)
    target_y = rng.uniform(-1.0, 1.0, n_target_splats)
    target_z = rng.uniform(-1.0, 1.0, n_target_splats)
    target_positions = np.stack([target_x, target_y, target_z], axis=1)

    positions = np.concatenate([wall_positions, target_positions], axis=0)
    n_splats = positions.shape[0]

    front_cameras = [
        CameraPose(center=np.array([-5.0, y, 0.0]), forward=np.array([1.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))
        for y in np.linspace(-1.5, 1.5, n_cameras_per_side)
    ]
    back_cameras = [
        CameraPose(center=np.array([wall_x + 6.0, y, 0.0]), forward=np.array([-1.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))
        for y in np.linspace(-1.5, 1.5, n_cameras_per_side)
    ]
    all_cameras = front_cameras + back_cameras

    per_camera = attribute_observations(positions, all_cameras, fov_deg=70.0, angular_tol=0.08, depth_margin=0.05)
    observed_camera_idx = invert_to_observed_camera_idx(per_camera, n_splats)

    sh_coeffs = random_sh_coeffs(rng, n_splats, degree=2)
    colors = sh_coeffs[:, :, 0].mean(axis=1)  # unused fallback value, sh_coeffs takes priority

    opacities = rng.uniform(0.5, 1.0, n_splats)
    scales = rng.uniform(0.02, 0.08, size=(n_splats, 3))
    rotations = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_splats, 1))

    scene = SplatScene(
        positions=positions,
        colors=colors,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        observed_camera_idx=observed_camera_idx,
        cameras=all_cameras,
        sh_coeffs=sh_coeffs,
        sh_degree=2,
    )
    return scene, dict(
        wall_x=wall_x,
        n_wall_splats=n_wall_splats,
        front_camera_idx=np.arange(len(front_cameras)),
        back_camera_idx=np.arange(len(front_cameras), len(all_cameras)),
    )


def load_from_gsplat_checkpoint(
    scene_dir: str,
    ply_filename: str = "splats.ply",
    fov_deg: Optional[float] = None,
    attribution_angular_tol: float = 0.05,
    attribution_depth_margin: float = 0.05,
    use_gpu_attribution: bool = False,
    attribution_min_opacity: float = 0.0,
    max_observations_per_splat: Optional[int] = None,
    attribution_seed: int = 0,
) -> SplatScene:
    """Real loader: reads `<scene_dir>/<ply_filename>` (a standard 3DGS
    .ply checkpoint, see gs_experiment.ply_io) for
    positions/scales/rotations/opacities/SH colors, and
    `<scene_dir>/transforms.json` (see gs_experiment.nerf_transforms, the
    same NeRF-synthetic-style schema this project's data always uses)
    for training camera poses -- the two files
    gs_experiment.scripts.train_minimal_gsplat's trainer produces together.

    `fov_deg` defaults to the shared `camera_angle_x` recorded in
    transforms.json (converted to degrees); pass it explicitly only if a
    different value is wanted for the visibility-attribution frustum test
    itself (e.g. a deliberately looser/tighter cone than what the cameras
    were actually rendered with).

    observed_camera_idx is not stored anywhere in a real checkpoint (the
    training pipeline doesn't record which views constrained which
    splat), so it's approximated the same way gs_experiment.splat_scene.
    make_occluder_scene demonstrates real geometric attribution works:
    frustum + soft-z-buffer occlusion
    (gs_experiment.visibility_attribution.attribute_observations), not an
    assignment rule.

    `use_gpu_attribution=True` uses `gpu_visibility_attribution.
    batched_attribute_observations` instead -- the same attribution,
    verified to match exactly (tests/gs_experiment/test_gpu_visibility_attribution.py;
    also checked directly against attribute_observations on a real 80k-splat/
    100-camera checkpoint: identical index sets on every camera), just
    ~100x faster on a real checkpoint by batching every camera's occlusion
    z-buffer into one GPU pass instead of a 100-iteration Python loop. Needs
    torch (lazily imported here, not at module level, so this module and the
    default `pytest tests/` suite stay importable without it); default stays
    `False` so this function's behavior is unchanged for every existing caller.

    `attribution_min_opacity`: splats below this opacity can't hard-occlude
    others during attribution (see `occlusion_mask`'s docstring) -- default
    0.0 keeps the old, opacity-blind behavior. Real checkpoints reliably
    have a few percent of splats that are GS-training floaters (drifted
    outside the intended training volume, near-zero opacity, a normal
    optimization artifact, not a bug in training itself); confirmed
    directly that these alone caused an 8x collapse in real per-splat
    camera attribution on an otherwise-identical, floater-free checkpoint
    of the same scene. Pass e.g. 0.1 (already this project's convention
    elsewhere, see fit_kernel_hyperparams' own min_opacity default) for
    real-checkpoint use.

    `max_observations_per_splat`: caps each splat's `observed_camera_idx`
    at this many cameras (uniform random subsample without replacement,
    seeded by `attribution_seed`) -- see
    `visibility_attribution.subsample_observed_camera_idx`'s docstring for
    why this exists: `splat_scene.splat_observations` expands
    `observed_camera_idx` into one row per (splat, observing-camera) pair,
    and at high splat counts with dense multi-view coverage that row count
    (not just splat count) is what determines whether the directional
    `LocalUncertaintyEngine` fits in host memory. `None` (the default)
    keeps every observation, i.e. unchanged from before this parameter
    existed.
    """
    from gs_experiment.nerf_transforms import camera_pose_from_c2w, load_transforms
    from gs_experiment.ply_io import read_3dgs_ply

    ply_path = os.path.join(scene_dir, ply_filename)
    transforms_path = os.path.join(scene_dir, "transforms.json")

    checkpoint = read_3dgs_ply(ply_path)
    camera_angle_x, frames = load_transforms(transforms_path)
    cameras = [camera_pose_from_c2w(c2w) for _, c2w in frames]

    if fov_deg is None:
        fov_deg = np.degrees(camera_angle_x)

    positions = checkpoint["positions"]
    opacities = checkpoint["opacities"]
    if use_gpu_attribution:
        from gs_experiment.gpu_visibility_attribution import batched_attribute_observations

        per_camera = batched_attribute_observations(
            positions, cameras, fov_deg=fov_deg, angular_tol=attribution_angular_tol, depth_margin=attribution_depth_margin,
            opacities=opacities, min_opacity=attribution_min_opacity,
        )
    else:
        per_camera = attribute_observations(
            positions, cameras, fov_deg=fov_deg, angular_tol=attribution_angular_tol, depth_margin=attribution_depth_margin,
            opacities=opacities, min_opacity=attribution_min_opacity,
        )
    observed_camera_idx = invert_to_observed_camera_idx(per_camera, positions.shape[0])
    if max_observations_per_splat is not None:
        observed_camera_idx = subsample_observed_camera_idx(
            observed_camera_idx, max_observations_per_splat, seed=attribution_seed
        )

    sh_coeffs = checkpoint["sh_coeffs"]
    colors = sh_coeffs[:, :, 0].mean(axis=1)  # unused fallback, sh_coeffs takes priority (see splat_observations)

    return SplatScene(
        positions=positions,
        colors=colors,
        opacities=checkpoint["opacities"],
        scales=checkpoint["scales"],
        rotations=checkpoint["rotations"],
        observed_camera_idx=observed_camera_idx,
        cameras=cameras,
        sh_coeffs=sh_coeffs,
        sh_degree=checkpoint["sh_degree"],
    )
