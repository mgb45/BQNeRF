"""Frustum + occlusion visibility attribution: which cameras plausibly
observed each splat, for real data where (unlike the mock scene, which
assigns this by fiat for controlled experiments) there's no ground-truth
record of which training views constrained which splat.

Deliberately a cheap proxy, not a faithful reproduction of what the
splatting/training pipeline actually did (that would mean rendering every
training view and recording each splat's alpha-weighted contribution --
real work, deferred until there's an actual renderer to hook into). This
gives two purely-geometric filters instead:

  1. Frustum test: is the splat within the camera's field of view and in
     front of it.
  2. Occlusion test: a soft z-buffer -- project all splats into each
     camera's local angular coordinates (bearing, not full pixel
     projection) and flag a splat as occluded if another splat sits at
     a similar bearing but meaningfully closer to the camera.

Both are pure numpy/scipy, no torch/gsplat dependency.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from gs_experiment.camera import CameraPose


def camera_local_frame(camera: CameraPose):
    """Right-handed (right, up, forward) basis for `camera`."""
    forward = camera.forward / np.linalg.norm(camera.forward)
    up = camera.up / np.linalg.norm(camera.up)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)  # re-orthogonalize
    return right, up, forward


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
