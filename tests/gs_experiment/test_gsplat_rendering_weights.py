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
from gs_experiment.gsplat_rendering_weights import (  # noqa: E402
    GsplatCameraProjection,
    gsplat_alpha_compositing_weights,
    gsplat_covariances,
)
from gs_experiment.pixel_uncertainty import quat_scale_to_covariance  # noqa: E402


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


def test_gsplat_camera_projection_query_pixel_matches_the_per_call_function():
    """GsplatCameraProjection.query_pixel (cached projection, plain-numpy
    compositing) must agree with gsplat_alpha_compositing_weights (fresh
    per-call CUDA projection) on the same scene/pixel -- same underlying
    math, two entry points."""
    camera = make_camera()
    K = default_intrinsics()
    positions = np.array([[2.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    opacities = np.array([0.6, 0.9])
    scales = np.array([[0.3, 0.3, 0.3], [0.1, 0.1, 0.1]])
    rotations = np.tile([1.0, 0.0, 0.0, 0.0], (2, 1))
    pixel = project_point_to_pixel(np.array([3.5, 0.0, 0.0]), viewmat_from_camera_pose(camera), K)

    direct = gsplat_alpha_compositing_weights(positions, opacities, scales, rotations, camera, K, 100, 100, pixel)

    projection = GsplatCameraProjection.build(positions, opacities, scales, rotations, camera, K, 100, 100)
    idx, cached = projection.query_pixel(pixel, pixel_radius=1000.0)
    cached_full = np.zeros(2)
    cached_full[idx] = cached

    np.testing.assert_allclose(direct, cached_full, atol=1e-4)


def test_gsplat_camera_projection_finds_the_relevant_splat_among_many_distractors():
    """The real bug this class fixes, reproduced at small scale: a cluster
    of distractor splats that a 3D-world-space ball query around the
    query point's own 3D location would treat as equally "nearby" (or
    even nearer) as the one splat actually rendering this pixel, but
    which project to wildly different screen locations. Real per-pixel
    relevance is about *pixel-space* proximity, which this class selects
    on directly -- confirmed here by finding the truly relevant splat
    regardless of how many pixel-irrelevant distractors are also close in
    3D world space."""
    camera = make_camera()
    K = default_intrinsics()
    rng = np.random.default_rng(0)

    relevant_position = np.array([5.0, 0.0, 0.0])
    relevant_opacity = 0.95
    n_distractors = 500
    # distractors: within 3D distance ~0.3-1.5 of the relevant splat (well
    # inside what a generous 3D ball-query radius would treat as "nearby"),
    # but scattered widely in world-space y/z -- very different bearings,
    # hence very different projected pixel locations.
    distractor_offsets = np.stack(
        [np.zeros(n_distractors), rng.uniform(-3.0, 3.0, n_distractors), rng.uniform(-3.0, 3.0, n_distractors)], axis=1
    )
    positions = np.concatenate([relevant_position[None, :], relevant_position[None, :] + distractor_offsets], axis=0)
    opacities = np.concatenate([[relevant_opacity], np.full(n_distractors, 0.9)])
    scales = np.full((n_distractors + 1, 3), 0.1)
    rotations = np.tile([1.0, 0.0, 0.0, 0.0], (n_distractors + 1, 1))

    pixel = project_point_to_pixel(relevant_position, viewmat_from_camera_pose(camera), K)
    projection = GsplatCameraProjection.build(positions, opacities, scales, rotations, camera, K, 100, 100)
    idx, weights = projection.query_pixel(pixel, pixel_radius=64.0, max_candidates=2000)

    assert 0 in idx  # the actually-relevant splat must be found
    relevant_weight = weights[list(idx).index(0)]
    assert abs(relevant_weight - relevant_opacity) < 1e-2  # and correctly dominate at its own pixel
    assert len(idx) < n_distractors  # most 3D-nearby-but-pixel-irrelevant distractors were correctly excluded


def test_gsplat_covariances_matches_the_plain_numpy_formula():
    """gsplat_covariances (real GPU gsplat.quat_scale_to_covar_preci) and
    quat_scale_to_covariance (plain numpy, used when no GPU/gsplat is
    available, e.g. rendering_aware_variance_along_ray) must agree --
    same formula, two implementations."""
    rng = np.random.default_rng(0)
    n = 8
    scales = rng.uniform(0.05, 0.5, size=(n, 3))
    rotations = rng.normal(size=(n, 4))  # quat_scale_to_covariance normalizes internally

    numpy_covars = quat_scale_to_covariance(rotations, scales)
    gpu_covars = gsplat_covariances(scales, rotations)

    np.testing.assert_allclose(numpy_covars, gpu_covars, atol=1e-5, rtol=1e-4)
