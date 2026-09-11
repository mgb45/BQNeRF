"""A loaded (or mocked) 3D Gaussian Splatting scene.

Scope note, worth being explicit about: this operates on 3D world-space
splat positions and 3D world-space query points, not 2D image-plane
pixels. Mapping a world-space uncertainty value back to a specific
camera's per-pixel image is a reprojection step (each pixel's uncertainty
would come from the world-space point(s) its ray intersects, weighted by
the same alpha-compositing weights gsplat already computes during
rendering) -- gpu_uncertainty.py/gpu_sh_directional_uncertainty.py do
exactly this via real bearing-space candidate search, not a reimplemented
rasterizer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from gs_experiment.camera import CameraPose
from gs_experiment.spherical_harmonics import SH_C0
from gs_experiment.visibility_attribution import (
    attribute_observations,
    invert_to_observed_camera_idx,
    subsample_observed_camera_idx,
)


@dataclass
class SplatScene:
    positions: np.ndarray  # (N, 3)
    colors: np.ndarray  # (N,) scalar, DC-only fallback -- see load_from_gsplat_checkpoint
    opacities: np.ndarray  # (N,)
    scales: np.ndarray  # (N, 3)
    rotations: np.ndarray  # (N, 4) quaternions

    # Observations: which real training cameras plausibly saw each splat --
    # observed_camera_idx[i] is a list of indices into `cameras` for splat i.
    # Real data derives this from geometric visibility attribution (frustum +
    # soft occlusion, visibility_attribution.attribute_observations), not an
    # assignment rule. Feeds gpu_sh_directional_uncertainty.
    # accumulate_sh_precision's real per-(splat, training-camera)
    # alpha-compositing weight (rerendering each training camera).
    observed_camera_idx: List[np.ndarray]
    cameras: List[CameraPose]

    # Optional: real spherical-harmonic color, (N, n_channels, n_coeffs).
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
    `wide_cameras`.
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


# Two independent, additive costs make up load_from_gsplat_checkpoint's
# host RSS, both measured directly (not estimated) against real checkpoints:
#
#   n_splats=1,000,000  -> attribution baseline 2757 MiB
#   n_splats=3,000,000  -> attribution baseline 7007 MiB
#
# Fit as a line through the two measured points above (bytes):
_ATTRIBUTION_FIXED_OVERHEAD_BYTES = 700 * 1024 * 1024  # process/import/checkpoint-read floor
_ATTRIBUTION_BYTES_PER_SPLAT = 2200  # measured ~2125 B/splat marginal; rounded up for margin


def fit_kernel_hyperparams(
    scene: SplatScene,
    sigma_bounds=(0.005, 1.0),
    n_windows: int = 25,
    max_window_size: int = 60,
    window_radius: float = 0.08,
    min_opacity: float = 0.1,
    seed: int = 0,
) -> Optional[float]:
    """Marginal-likelihood-fit the position kernel's bandwidth (sigma)
    against THIS scene's own real data, instead of reusing a value pooled
    once across a different calibration set
    (gs_experiment.hyperparams.fit_kernel_param_pooled_nd) -- a bandwidth
    tuned at one checkpoint's splat density/coverage has no reason to be
    right for a checkpoint at a very different density (confirmed
    directly: a 300k-splat-tuned sigma is a real bandwidth mismatch at 500
    splats).

    Sigma comes from local (position, color) windows: `n_windows` splats
    above `min_opacity` are sampled as window centers, each paired with its
    neighbors within `window_radius` (capped at `max_window_size`, a
    random subsample, not a truncation, so the fit isn't spatially biased
    toward whichever neighbors happen to sort first).

    Returns `sigma`, or `None` if there wasn't enough real data to fit it
    (too few above-threshold splats) -- callers should fall back to a
    documented default in that case, not silently use an ill-fit value.
    """
    from scipy.spatial import cKDTree

    from gs_experiment.hyperparams import fit_kernel_param_pooled_nd
    from gs_experiment.kernels import ProductKernel, RBFKernel

    rng = np.random.default_rng(seed)
    keep = scene.opacities > min_opacity
    positions = scene.positions[keep]
    colors = scene.colors[keep]

    if len(positions) < 6:
        return None

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
    if not sigma_datasets:
        return None

    fit = fit_kernel_param_pooled_nd(
        sigma_datasets, lambda s: ProductKernel([RBFKernel(sigma=s)] * 3), bounds=sigma_bounds, n_grid=25,
    )
    return float(fit.param)


def fit_kernel_hyperparams_with_noise(
    scene: SplatScene,
    sigma_bounds=(0.005, 1.0),
    noise_bounds=(1e-5, 0.5),
    n_windows: int = 25,
    max_window_size: int = 60,
    window_radius: float = 0.08,
    min_opacity: float = 0.1,
    seed: int = 0,
) -> tuple:
    """`fit_kernel_hyperparams`'s noise-aware sibling: fits the position
    kernel's bandwidth (sigma) *jointly* with a real homoscedastic
    observation-noise variance (`gs_experiment.hyperparams.
    fit_kernel_param_and_noise_pooled_nd`), instead of the noiseless-
    interpolation assumption `fit_kernel_hyperparams` makes. Motivation:
    real splat positions routinely include near-duplicate points (observed
    directly -- pairwise distances as small as 1e-4 units under a
    ~0.1-unit noiseless-fit bandwidth), which forces the noiseless fit
    toward an artificially short bandwidth just to keep exactly
    interpolating through them, at a real, large marginal-likelihood cost
    (confirmed directly: +180 to +780 log-likelihood units from adding
    noise, across 8 real scene/checkpoint combinations checked) -- and a
    visibly worse-conditioned, speckle-prone posterior downstream. See
    `gs_experiment.quadrature._rendering_aware_moments`'s docstring for the
    full noise model and motivation.

    Returns `(sigma, noise_variance)`, `None`/`None` if there wasn't enough
    real data to fit them (same condition `fit_kernel_hyperparams` uses).
    """
    from scipy.spatial import cKDTree

    from gs_experiment.hyperparams import fit_kernel_param_and_noise_pooled_nd
    from gs_experiment.kernels import ProductKernel, RBFKernel

    rng = np.random.default_rng(seed)
    keep = scene.opacities > min_opacity
    positions = scene.positions[keep]
    colors = scene.colors[keep]

    if len(positions) < 6:
        return None, None

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
    if not sigma_datasets:
        return None, None

    fit = fit_kernel_param_and_noise_pooled_nd(
        sigma_datasets, lambda s: ProductKernel([RBFKernel(sigma=s)] * 3),
        bounds=sigma_bounds, noise_bounds=noise_bounds, n_grid=15,
    )
    return float(fit.param), float(fit.noise_variance)


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
    # Real (DC-only, view-independent) color, not the raw SH coefficient --
    # 3DGS stores SH coefficients as offsets from a mid-gray baseline
    # (eval_sh's own "+ 0.5" convention; see that function's docstring), so
    # a raw sh_coeffs[:,:,0] value is not itself a color.
    colors = SH_C0 * sh_coeffs[:, :, 0].mean(axis=1) + 0.5

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
    verified to match exactly, just ~100x faster on a real checkpoint by
    batching every camera's occlusion z-buffer into one GPU pass instead of
    a 100-iteration Python loop. Needs torch (lazily imported here, not at
    module level, so this module and the default `pytest tests/` suite
    stay importable without it); default stays `False` so this function's
    behavior is unchanged for every existing caller.

    `attribution_min_opacity`: splats below this opacity can't hard-occlude
    others during attribution (see `occlusion_mask`'s docstring) -- default
    0.0 keeps the old, opacity-blind behavior. Real checkpoints reliably
    have a few percent of splats that are GS-training floaters (drifted
    outside the intended training volume, near-zero opacity, a normal
    optimization artifact, not a bug in training itself); confirmed
    directly that these alone caused an 8x collapse in real per-splat
    camera attribution on an otherwise-identical, floater-free checkpoint
    of the same scene. Pass e.g. 0.1 for real-checkpoint use.

    `max_observations_per_splat`: caps each splat's `observed_camera_idx`
    at this many cameras (uniform random subsample without replacement,
    seeded by `attribution_seed`) -- see
    `visibility_attribution.subsample_observed_camera_idx`'s docstring for
    why this exists: at high splat counts with dense multi-view coverage,
    `gpu_sh_directional_uncertainty.accumulate_sh_precision`'s per-camera
    rerendering cost scales with total observation-row count, not just
    splat count. `None` (the default) keeps every observation.
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
    # Real (DC-only, view-independent) color -- SH_C0 * raw + 0.5, matching
    # eval_sh's own degree-0 formula exactly (3DGS stores SH coefficients as
    # offsets from a mid-gray baseline, not a color on their own).
    colors = SH_C0 * sh_coeffs[:, :, 0].mean(axis=1) + 0.5

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
