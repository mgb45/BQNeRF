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

Also worth restating (see bq_splat/README.md and the design discussion this
package follows from): `scales`/`rotations` are kept as metadata for
standard rendering, but are NOT fed into the BQ kernel's bandwidth. The
validated BQ machinery (bq_splat) uses one shared or pooled-fit bandwidth
(see bq_splat/results/FINDINGS.md sections 5, 7), not per-splat
heterogeneous covariances -- using each splat's own learned covariance as
its own kernel bandwidth is a real, mathematically plausible extension
(closer to the original derivation's "splats as weighted kernel nodes"
framing) but it is a second, unvalidated change; stacking it on top of the
GPU/gsplat integration at the same time would make it hard to tell which
change caused which result. Left as documented future work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from gs_experiment.camera import CameraPose, directions_from_positions_to_camera


@dataclass
class SplatScene:
    positions: np.ndarray  # (N, 3)
    colors: np.ndarray  # (N,) scalar for now -- real gsplat is (N, 3) or spherical-harmonic coefficients
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
    `wide_cameras` -- the 3D, real-camera-pose analogue of
    scripts/validate_directional_combined.py's controlled zones.
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


def splat_observations(scene: SplatScene):
    """Expand a SplatScene into parallel (position, direction, value)
    arrays -- one row per (splat, observing-camera) pair -- the input
    format bayesian_quadrature_directional expects. `value` here is just
    each splat's color; a real pipeline would use each training view's
    actual rendered contribution instead."""
    positions, directions, values = [], [], []
    for i, cam_idx in enumerate(scene.observed_camera_idx):
        for c in cam_idx:
            camera = scene.cameras[c]
            direction = directions_from_positions_to_camera(scene.positions[i : i + 1], camera)[0]
            positions.append(scene.positions[i])
            directions.append(direction)
            values.append(scene.colors[i])
    return np.array(positions), np.array(directions), np.array(values)


def load_from_gsplat_checkpoint(path: str) -> SplatScene:
    """Real loader, not yet callable without the optional gsplat/plyfile
    dependencies (see requirements-gsplat.txt) and, more fundamentally,
    without a trained checkpoint to point it at. Documented here so the
    interface exists and the rest of this package can be written and
    tested against it now, via make_mock_scene, without waiting for GPU
    access.

    Expected real implementation: read a 3DGS .ply checkpoint (plyfile),
    extract positions/scales/rotations/opacities/spherical-harmonic colors
    per the standard 3DGS point-cloud schema, and load the corresponding
    training cameras from the scene's transforms.json (or COLMAP
    sparse-reconstruction output). observed_camera_idx would come from
    which training views the reconstruction pipeline actually used for
    each splat, or be approximated by frustum + occlusion checks.
    """
    raise NotImplementedError(
        "Real gsplat checkpoint loading needs the optional torch/gsplat/plyfile "
        "dependencies (requirements-gsplat.txt) and a trained scene -- not available "
        "in this environment. Use make_mock_scene() for now; this function's docstring "
        "records the intended real implementation."
    )
