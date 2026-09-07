import numpy as np

from gs_experiment.camera import CameraPose
from gs_experiment.visibility_attribution import (
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
