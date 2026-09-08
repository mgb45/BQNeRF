import numpy as np

from gs_experiment.camera import CameraPose
from gs_experiment.visibility_attribution import (
    CameraSplatIndex,
    attribute_observations,
    in_frustum,
    invert_to_observed_camera_idx,
    occlusion_mask,
    project_to_camera_local,
    ray_transmittance_weights,
)


def make_camera(center=(0.0, 0.0, 0.0), forward=(1.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)):
    return CameraPose(center=np.array(center, dtype=float), forward=np.array(forward, dtype=float), up=np.array(up, dtype=float))


def test_project_to_camera_local_depth_matches_forward_distance():
    camera = make_camera()
    positions = np.array([[3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
    _, _, depth = project_to_camera_local(positions, camera)
    np.testing.assert_allclose(depth, [3.0, 0.0, -2.0], atol=1e-9)


def test_project_to_camera_local_bearing_zero_on_axis():
    camera = make_camera()
    positions = np.array([[5.0, 0.0, 0.0]])
    bx, by, _ = project_to_camera_local(positions, camera)
    assert abs(bx[0]) < 1e-9
    assert abs(by[0]) < 1e-9


def test_in_frustum_accepts_forward_rejects_behind_and_off_axis():
    camera = make_camera()
    positions = np.array(
        [
            [5.0, 0.0, 0.0],  # dead ahead -- in frustum
            [-5.0, 0.0, 0.0],  # behind the camera -- not in frustum
            [5.0, 20.0, 0.0],  # far off to the side -- outside a 60deg fov
        ]
    )
    mask = in_frustum(positions, camera, fov_deg=60.0)
    assert mask[0] == True
    assert mask[1] == False
    assert mask[2] == False


def test_occlusion_mask_flags_splat_behind_a_closer_occluder():
    camera = make_camera()
    positions = np.array(
        [
            [2.0, 0.0, 0.0],  # occluder, close to camera
            [5.0, 0.0, 0.0],  # target, same bearing, further away -- should be occluded
            [5.0, 3.0, 0.0],  # same depth as target, different bearing -- should NOT be occluded
        ]
    )
    occluded = occlusion_mask(positions, camera, angular_tol=0.1, depth_margin=0.05)
    assert occluded[1] == True
    assert occluded[2] == False
    assert occluded[0] == False  # the occluder itself isn't behind anything


def _occlusion_mask_reference(positions, camera, angular_tol, depth_margin=0.05):
    """Brute-force O(n^2) reference for occlusion_mask, kept only in this
    test as a cross-check for the vectorized query_pairs implementation --
    a direct, unoptimized transcription of "is any other point within
    angular_tol in bearing and meaningfully closer," with no scipy spatial
    indexing at all, so it can't share a bug with the real implementation."""
    bearing_x, bearing_y, depth = project_to_camera_local(positions, camera)
    n = positions.shape[0]
    occluded = np.zeros(n, dtype=bool)
    for i in range(n):
        if np.isnan(bearing_x[i]):
            continue
        for j in range(n):
            if i == j or np.isnan(bearing_x[j]):
                continue
            dist = np.hypot(bearing_x[i] - bearing_x[j], bearing_y[i] - bearing_y[j])
            if dist < angular_tol and depth[j] < depth[i] - depth_margin * abs(depth[i]):
                occluded[i] = True
                break
    return occluded


def test_occlusion_mask_matches_a_brute_force_reference_on_random_scenes():
    """Regression check for the grid/z-buffer rewrite against an
    independent, deliberately naive O(n^2) reference (exact circular
    angular_tol radius) across several random scenes and angular
    tolerances, not just the two hand-placed cases above.

    Not required to match exactly: the grid version's 3x3-cell-block
    neighbor search is a provable *superset* of the true circular-radius
    neighbor set (see occlusion_mask's docstring for the proof sketch),
    so it can flag a point as occluded by a neighbor just past the true
    circular radius, near a cell corner -- a false positive relative to
    the exact reference, never a false negative. The false-positive rate
    is real and not tiny (the 3x3-square-vs-circle area ratio is ~2.86x,
    9/pi) -- measured up to ~0.21 at angular_tol=0.1 on this exact scene,
    so the threshold below is set with real headroom above that, not
    tightened to look reassuring. Checked directly: every
    reference-occluded point must still be occluded here (the property
    that actually matters -- occlusion_mask is only ever used to decide
    whether to *exclude* a splat from a camera's attribution, so a missed
    occlusion would silently corrupt attribution, while an extra one just
    makes the proxy a bit more conservative)."""
    camera = make_camera()
    rng = np.random.default_rng(0)
    for angular_tol in (0.02, 0.1, 0.5):
        positions = np.stack(
            [rng.uniform(3.0, 8.0, 80), rng.uniform(-3.0, 3.0, 80), rng.uniform(-3.0, 3.0, 80)], axis=1
        )
        expected = _occlusion_mask_reference(positions, camera, angular_tol=angular_tol)
        got = occlusion_mask(positions, camera, angular_tol=angular_tol)

        assert np.all(got[expected])  # no false negatives: every true occlusion is still flagged
        false_positive_rate = (got & ~expected).sum() / len(positions)
        assert false_positive_rate < 0.35


def test_attribute_observations_and_invert_round_trip():
    camera_front = make_camera(center=(0.0, 0.0, 0.0), forward=(1.0, 0.0, 0.0))
    camera_back = make_camera(center=(0.0, 0.0, 0.0), forward=(-1.0, 0.0, 0.0))
    positions = np.array([[5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]])  # splat 0 in front of camera_front; splat 1 in front of camera_back

    per_camera = attribute_observations(positions, [camera_front, camera_back], fov_deg=60.0)
    assert 0 in per_camera[0] and 1 not in per_camera[0]
    assert 1 in per_camera[1] and 0 not in per_camera[1]

    observed = invert_to_observed_camera_idx(per_camera, n_splats=2)
    assert list(observed[0]) == [0]
    assert list(observed[1]) == [1]


def test_attribute_observations_respects_occlusion_end_to_end():
    camera = make_camera()
    positions = np.array([[2.0, 0.0, 0.0], [5.0, 0.0, 0.0]])  # occluder then target, same bearing
    per_camera = attribute_observations(positions, [camera], fov_deg=60.0, angular_tol=0.1)
    assert 0 in per_camera[0]
    assert 1 not in per_camera[0]


def test_ray_transmittance_weights_zeroes_splats_off_the_reference_bearing():
    camera = make_camera()
    positions = np.array([[5.0, 0.0, 0.0], [5.0, 3.0, 0.0]])  # on-ray, off-ray
    opacities = np.array([0.8, 0.9])
    weights = ray_transmittance_weights(positions, opacities, camera, reference_bearing=(0.0, 0.0), angular_tol=0.1)
    assert weights[0] > 0.0
    assert weights[1] == 0.0


def test_ray_transmittance_weights_zeroes_splats_behind_the_camera():
    camera = make_camera()
    positions = np.array([[-5.0, 0.0, 0.0]])  # behind the camera, same nominal bearing
    opacities = np.array([1.0])
    weights = ray_transmittance_weights(positions, opacities, camera, reference_bearing=(0.0, 0.0), angular_tol=0.1)
    assert weights[0] == 0.0


def test_ray_transmittance_weights_recovers_standard_alpha_compositing_along_one_ray():
    """Two on-ray splats, near then far: matches
    PROOF_alpha_compositing_equivalence.md Theorem A's discrete formula
    directly -- w_0 = alpha_0, w_1 = (1 - alpha_0) * alpha_1."""
    camera = make_camera()
    positions = np.array([[2.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    alpha0, alpha1 = 0.6, 0.7
    opacities = np.array([alpha0, alpha1])
    weights = ray_transmittance_weights(positions, opacities, camera, reference_bearing=(0.0, 0.0), angular_tol=0.1)
    assert abs(weights[0] - alpha0) < 1e-9
    assert abs(weights[1] - (1 - alpha0) * alpha1) < 1e-9


def test_ray_transmittance_weights_fully_opaque_occluder_zeroes_out_whatever_is_behind_it():
    camera = make_camera()
    positions = np.array([[2.0, 0.0, 0.0], [5.0, 0.0, 0.0]])  # fully opaque occluder, then a target
    opacities = np.array([1.0, 0.9])
    weights = ray_transmittance_weights(positions, opacities, camera, reference_bearing=(0.0, 0.0), angular_tol=0.1)
    assert abs(weights[0] - 1.0) < 1e-9
    assert weights[1] == 0.0


def test_ray_transmittance_weights_all_zero_returns_all_zero_not_an_error():
    camera = make_camera()
    positions = np.array([[5.0, 10.0, 0.0]])  # off-ray
    opacities = np.array([0.9])
    weights = ray_transmittance_weights(positions, opacities, camera, reference_bearing=(0.0, 0.0), angular_tol=0.1)
    assert np.all(weights == 0.0)


def test_camera_splat_index_finds_the_relevant_splat_among_many_distractors_within_3d_ball_query_range():
    """The real bug this class fixes, at small scale: a dense cluster of
    "distractor" splats sitting within what would be a generous 3D ball-
    query radius of a query point, but at wildly different bearings (as
    if they were on a totally different part of the object) -- a 3D
    Euclidean neighbor search can't tell them apart from the one splat
    that's actually on this ray, but bearing-space indexing does, by
    construction."""
    camera = make_camera()
    rng = np.random.default_rng(0)
    relevant = np.array([[5.0, 0.0, 0.0]])
    # distractors: within 3D distance ~0.3-1.5 of the relevant splat, but
    # scattered across a wide range of world-space y/z (very different bearings)
    distractors = relevant + np.stack(
        [np.zeros(500), rng.uniform(-3.0, 3.0, 500), rng.uniform(-3.0, 3.0, 500)], axis=1
    )
    positions = np.concatenate([relevant, distractors], axis=0)

    index = CameraSplatIndex.build(positions, camera)
    found = index.query(reference_bearing=(0.0, 0.0), angular_tol=0.05, max_candidates=None)

    assert 0 in found  # the actually-relevant splat (index 0) must be found
    assert len(found) < len(positions) / 2  # and the search correctly excluded most distractors


def test_camera_splat_index_max_candidates_keeps_nearest_in_bearing_not_random():
    camera = make_camera()
    # three on-ray-ish splats at increasing bearing distance from the reference
    positions = np.array([[5.0, 0.0, 0.0], [5.0, 0.3, 0.0], [5.0, 0.6, 0.0]])
    index = CameraSplatIndex.build(positions, camera)
    found = index.query(reference_bearing=(0.0, 0.0), angular_tol=1.0, max_candidates=1)
    assert list(found) == [0]  # the nearest-in-bearing one, deterministically -- not a random pick among the 3


def test_camera_splat_index_query_returns_empty_for_no_valid_splats():
    camera = make_camera()
    positions = np.array([[-5.0, 0.0, 0.0]])  # behind the camera
    index = CameraSplatIndex.build(positions, camera)
    found = index.query(reference_bearing=(0.0, 0.0), angular_tol=0.5)
    assert found.shape == (0,)
