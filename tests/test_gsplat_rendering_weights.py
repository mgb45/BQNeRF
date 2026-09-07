"""GPU-only tests for gs_experiment.gsplat_rendering_weights: real, gsplat-
computed alpha-compositing weights. Skipped automatically (not failed) when
torch/gsplat or a CUDA device aren't available -- see
../requirements-gsplat.txt for setup -- so `pytest tests/` stays green in a
GPU-free environment; run explicitly with a GPU + gsplat env to exercise
these for real.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gsplat")

if not torch.cuda.is_available():
    pytest.skip("gsplat's CUDA kernels need a real GPU", allow_module_level=True)

from gs_experiment.camera import CameraPose, project_point_to_pixel, viewmat_from_camera_pose  # noqa: E402
from gs_experiment.gsplat_rendering_weights import gsplat_alpha_compositing_weights  # noqa: E402


def make_camera():
    return CameraPose(center=np.array([0.0, 0.0, 0.0]), forward=np.array([1.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))


def default_intrinsics():
    return np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])


def test_gsplat_weight_at_a_splats_own_projected_center_equals_its_opacity():
    """At the exact pixel a splat projects to, sigma=0 so alpha=opacity
    exactly (see this module's own docstring derivation) -- a single,
    unoccluded splat's weight should equal its opacity to high precision."""
    camera = make_camera()
    K = default_intrinsics()
    positions = np.array([[5.0, 0.0, 0.0]])
    opacities = np.array([0.8])
    scales = np.array([[0.1, 0.1, 0.1]])
    rotations = np.array([[1.0, 0.0, 0.0, 0.0]])
    viewmat = viewmat_from_camera_pose(camera)
    pixel = project_point_to_pixel(positions[0], viewmat, K)

    weights = gsplat_alpha_compositing_weights(positions, opacities, scales, rotations, camera, K, 100, 100, pixel)
    assert abs(weights[0] - 0.8) < 1e-4


def test_gsplat_weight_decays_away_from_the_splats_projected_center():
    camera = make_camera()
    K = default_intrinsics()
    positions = np.array([[5.0, 0.0, 0.0]])
    opacities = np.array([0.9])
    scales = np.array([[0.1, 0.1, 0.1]])
    rotations = np.array([[1.0, 0.0, 0.0, 0.0]])
    viewmat = viewmat_from_camera_pose(camera)
    pixel = project_point_to_pixel(positions[0], viewmat, K)
    far_pixel = pixel + np.array([20.0, 0.0])

    at_center = gsplat_alpha_compositing_weights(positions, opacities, scales, rotations, camera, K, 100, 100, pixel)
    far_away = gsplat_alpha_compositing_weights(positions, opacities, scales, rotations, camera, K, 100, 100, far_pixel)
    assert far_away[0] < at_center[0]


def test_gsplat_weight_fully_opaque_occluder_zeroes_out_whatever_is_behind_it():
    """The real-footprint analogue of
    test_gs_visibility_attribution.py::test_ray_transmittance_weights_fully_opaque_occluder_zeroes_out_whatever_is_behind_it,
    now using gsplat's own projected footprint/depth ordering."""
    camera = make_camera()
    K = default_intrinsics()
    positions = np.array([[2.0, 0.0, 0.0], [5.0, 0.0, 0.0]])  # occluder, then target -- same bearing
    opacities = np.array([1.0, 0.9])
    scales = np.array([[0.5, 0.5, 0.5], [0.1, 0.1, 0.1]])  # occluder large enough to cover the target's footprint
    rotations = np.tile([1.0, 0.0, 0.0, 0.0], (2, 1))
    viewmat = viewmat_from_camera_pose(camera)
    pixel = project_point_to_pixel(np.array([3.5, 0.0, 0.0]), viewmat, K)  # between the two, same bearing

    weights = gsplat_alpha_compositing_weights(positions, opacities, scales, rotations, camera, K, 100, 100, pixel)
    assert abs(weights[0] - 1.0) < 1e-3
    # opacity 1.0 is clamped to alpha<=0.999 (gsplat's own numerical-stability
    # convention, mirrored here), so a residual ~0.001 transmittance -- not
    # literally 0 -- is the physically-correct answer, and still negligible.
    assert weights[1] < 1e-3


def test_gsplat_weight_is_zero_for_a_splat_behind_the_camera():
    camera = make_camera()
    K = default_intrinsics()
    positions = np.array([[-5.0, 0.0, 0.0]])
    opacities = np.array([1.0])
    scales = np.array([[0.1, 0.1, 0.1]])
    rotations = np.array([[1.0, 0.0, 0.0, 0.0]])

    weights = gsplat_alpha_compositing_weights(
        positions, opacities, scales, rotations, camera, K, 100, 100, np.array([50.0, 50.0])
    )
    assert weights[0] == 0.0


def test_gsplat_weight_empty_positions_returns_empty_array():
    camera = make_camera()
    K = default_intrinsics()
    weights = gsplat_alpha_compositing_weights(
        np.empty((0, 3)), np.empty(0), np.empty((0, 3)), np.empty((0, 4)), camera, K, 100, 100, np.array([50.0, 50.0])
    )
    assert weights.shape == (0,)
