"""Frustum + occlusion visibility attribution: which cameras plausibly
observed each splat, for real data where (unlike the mock scene, which
assigns this by fiat for controlled experiments) there's no ground-truth
record of which training views constrained which splat.

Originally a cheap proxy for the real work this docstring used to say was
"deferred until there's an actual renderer to hook into": rendering every
training view and recording each splat's alpha-weighted contribution.
That renderer hook now exists (gs_experiment/gsplat_rendering_weights.py,
via gsplat's own differentiable projection) for a single query ray/pixel;
`attribute_observations` below still uses the cheaper geometric filters,
since attributing *every* splat to *every* camera via the real renderer
for a whole scene is a much larger cost than this module's original
per-query use case needs:

  1. Frustum test: is the splat within the camera's field of view and in
     front of it.
  2. Occlusion test: a soft z-buffer -- project all splats into each
     camera's local angular coordinates (bearing, not full pixel
     projection) and flag a splat as occluded if another splat sits at
     a similar bearing but meaningfully closer to the camera.
  3. ray_transmittance_weights: a continuous, depth-ordered analogue of
     (2) for one specific ray -- real per-splat opacity turned into a
     genuine alpha-compositing transmittance weight, still via bearing
     proximity rather than the real anisotropic 2D footprint (contrast
     with gsplat_rendering_weights.gsplat_alpha_compositing_weights).

All of the above are pure numpy/scipy, no torch/gsplat dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from gs_experiment.camera import CameraPose, camera_local_frame


def project_to_camera_local(positions: np.ndarray, camera: CameraPose):
    """Project `positions` (N, 3) into camera-local angular bearing
    (bearing_x, bearing_y) and depth along the camera's forward axis.
    Bearing here is the tangent-plane projection (position component along
    right/up, divided by forward-depth) -- a pinhole-like angular
    coordinate, not a full intrinsics-based pixel projection (no focal
    length/principal point needed for a bearing-based occlusion test).
    Returns (bearing_x, bearing_y, depth), each shape (N,).
    """
    positions = np.asarray(positions, dtype=float)
    right, up, forward = camera_local_frame(camera)
    rel = positions - camera.center[None, :]
    depth = rel @ forward
    safe_depth = np.where(np.abs(depth) < 1e-9, np.nan, depth)
    bearing_x = (rel @ right) / safe_depth
    bearing_y = (rel @ up) / safe_depth
    return bearing_x, bearing_y, depth


def in_frustum(positions: np.ndarray, camera: CameraPose, fov_deg: float = 60.0, near: float = 1e-3, far: float = np.inf) -> np.ndarray:
    """Boolean mask: is each position in front of `camera`, within `near`/
    `far`, and within a square field of view of half-angle `fov_deg`/2 in
    both bearing axes."""
    bearing_x, bearing_y, depth = project_to_camera_local(positions, camera)
    half_tan = np.tan(np.deg2rad(fov_deg) / 2.0)
    in_front = (depth > near) & (depth < far)
    within_fov = (np.abs(bearing_x) < half_tan) & (np.abs(bearing_y) < half_tan)
    return in_front & within_fov & ~np.isnan(bearing_x)


def occlusion_mask(positions: np.ndarray, camera: CameraPose, angular_tol: float, depth_margin: float = 0.05) -> np.ndarray:
    """Boolean mask: is each position occluded by another position that
    projects to a similar bearing (within `angular_tol`, in the tangent-
    plane bearing units from project_to_camera_local) but is closer to the
    camera by more than `depth_margin` * that closer point's own depth
    (a relative, scale-aware margin rather than an absolute one, since
    "close" means different absolute distances near vs. far from the
    camera).
    """
    bearing_x, bearing_y, depth = project_to_camera_local(positions, camera)
    n = positions.shape[0]
    occluded = np.zeros(n, dtype=bool)

    valid = ~np.isnan(bearing_x)
    if valid.sum() < 2:
        return occluded

    valid_idx = np.where(valid)[0]
    bearings = np.stack([bearing_x[valid_idx], bearing_y[valid_idx]], axis=1)
    tree = cKDTree(bearings)
    neighbor_lists = tree.query_ball_point(bearings, angular_tol)

    for local_i, neighbors in enumerate(neighbor_lists):
        i = valid_idx[local_i]
        my_depth = depth[i]
        for local_j in neighbors:
            j = valid_idx[local_j]
            if j == i:
                continue
            if depth[j] < my_depth - depth_margin * abs(my_depth):
                occluded[i] = True
                break

    return occluded


def ray_transmittance_weights(
    positions: np.ndarray, opacities: np.ndarray, camera: CameraPose, reference_bearing, angular_tol: float = 0.05
) -> np.ndarray:
    """Real, depth-ordered alpha-compositing transmittance weight
    `w_i = T_i * alpha_i` for each of `positions`, along the single ray at
    `reference_bearing` (a (bearing_x, bearing_y) pair, in `camera`'s local
    angular coordinates -- see project_to_camera_local) -- the continuous,
    occlusion-aware analogue of `occlusion_mask`'s binary yes/no flag.
    `alpha_i` is real per-splat opacity used directly as the discrete alpha
    in the standard alpha-compositing formula
    (`PROOF_alpha_compositing_equivalence.md` Theorem A: `T_i = prod_{j
    closer, same ray} (1 - alpha_j)`), not a synthetic Gaussian bump --
    this is what `gs_experiment/pixel_uncertainty.py`'s
    `rendering_aware_variance` (opacity-averaged, occlusion-blind) is
    missing, and what its own docstring names as the next step.

    Splats more than `angular_tol` from `reference_bearing`, behind the
    camera, or with undefined bearing (see project_to_camera_local) get
    weight 0 regardless of depth or opacity -- they aren't plausibly on
    this ray at all. Splats behind a closer, sufficiently opaque splat on
    the same ray get a small weight via the accumulated transmittance
    product, exactly as real alpha compositing would down-weight them --
    not a hard cutoff the way occlusion_mask's boolean flag is.
    """
    positions = np.asarray(positions, dtype=float)
    opacities = np.asarray(opacities, dtype=float)
    n = positions.shape[0]
    weights = np.zeros(n)
    if n == 0:
        return weights

    bearing_x, bearing_y, depth = project_to_camera_local(positions, camera)
    ref_x, ref_y = reference_bearing
    angular_dist = np.sqrt((bearing_x - ref_x) ** 2 + (bearing_y - ref_y) ** 2)
    on_ray = (angular_dist < angular_tol) & ~np.isnan(bearing_x) & (depth > 0)
    on_ray_idx = np.where(on_ray)[0]
    if on_ray_idx.size == 0:
        return weights

    depth_order = on_ray_idx[np.argsort(depth[on_ray_idx])]
    transmittance = 1.0
    for i in depth_order:
        alpha = float(np.clip(opacities[i], 0.0, 1.0))
        weights[i] = transmittance * alpha
        transmittance *= 1.0 - alpha
    return weights


@dataclass
class CameraSplatIndex:
    """A real per-camera bearing/depth index over an entire scene's
    splats, built once and reused for many per-pixel
    `LocalUncertaintyEngine.rendering_aware_variance_along_ray` queries
    against the same camera.

    Replaces gathering candidates via a 3D-world-space ball query
    (`LocalUncertaintyEngine.local_neighbors`) before handing them to
    `ray_transmittance_weights`: on a real, densely-packed checkpoint, a
    generously-sized 3D radius can return a large fraction of the *entire*
    scene (confirmed directly: ~half of 300k splats within a 1.6-unit
    radius on a real NeRF-Synthetic checkpoint), and `local_neighbors`
    then keeps only `max_neighbors` of those *uniformly at random* --
    almost never the splat that's actually relevant to a given pixel,
    since relevance to one ray is about angular/depth alignment, not raw
    3D Euclidean distance. Indexing by real bearing instead (this class)
    and ranking any overflow by bearing distance (not randomly) fixes
    both the correctness gap and the near-total candidate waste.
    """

    indices: np.ndarray  # into the original positions/opacities arrays -- in front of the camera only
    bearings: np.ndarray  # (M, 2)
    depths: np.ndarray  # (M,)
    camera: CameraPose
    _tree: Optional[cKDTree] = field(default=None, repr=False)

    def __post_init__(self):
        self._tree = cKDTree(self.bearings) if len(self.indices) > 0 else None

    @classmethod
    def build(cls, positions: np.ndarray, camera: CameraPose) -> "CameraSplatIndex":
        bearing_x, bearing_y, depth = project_to_camera_local(positions, camera)
        valid = (depth > 0) & ~np.isnan(bearing_x)
        idx = np.where(valid)[0]
        return cls(indices=idx, bearings=np.stack([bearing_x[idx], bearing_y[idx]], axis=1), depths=depth[idx], camera=camera)

    def query(self, reference_bearing, angular_tol: float, max_candidates: Optional[int] = None) -> np.ndarray:
        """Indices (into the original positions/opacities arrays) of
        splats within `angular_tol` of `reference_bearing`. If more than
        `max_candidates` qualify, keeps the ones *nearest in bearing* --
        a real relevance ranking, not a random, potentially-irrelevant
        subsample."""
        if self._tree is None:
            return np.empty(0, dtype=int)
        local = np.array(self._tree.query_ball_point(reference_bearing, angular_tol), dtype=int)
        if local.size == 0:
            return np.empty(0, dtype=int)
        if max_candidates is not None and local.size > max_candidates:
            dists = np.linalg.norm(self.bearings[local] - np.asarray(reference_bearing), axis=1)
            local = local[np.argsort(dists)[:max_candidates]]
        return self.indices[local]


def attribute_observations(positions: np.ndarray, cameras: list, fov_deg: float = 60.0, angular_tol: float = 0.05, depth_margin: float = 0.05):
    """For each camera, which splat indices does it plausibly observe
    (in frustum and not occluded). Returns a list of length len(cameras),
    each entry an array of splat indices -- the camera-indexed view of the
    same information SplatScene.observed_camera_idx stores splat-indexed;
    invert this (see gs_experiment.splat_scene) to populate that field for
    real data.
    """
    per_camera = []
    for camera in cameras:
        visible = in_frustum(positions, camera, fov_deg=fov_deg)
        if visible.any():
            occluded = np.zeros(positions.shape[0], dtype=bool)
            occluded[visible] = occlusion_mask(positions[visible], camera, angular_tol, depth_margin)
            visible_idx = np.where(visible)[0]
            per_camera.append(visible_idx[~occluded[visible_idx]])
        else:
            per_camera.append(np.array([], dtype=int))
    return per_camera


def invert_to_observed_camera_idx(per_camera_visible: list, n_splats: int) -> list:
    """Invert attribute_observations's camera-indexed output into the
    splat-indexed observed_camera_idx list SplatScene expects."""
    observed = [[] for _ in range(n_splats)]
    for cam_idx, splat_indices in enumerate(per_camera_visible):
        for s in splat_indices:
            observed[s].append(cam_idx)
    return [np.array(cams, dtype=int) for cams in observed]
