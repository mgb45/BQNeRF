import numpy as np
import pytest

from gs_experiment.camera import (
    CameraPose,
    camera_local_frame,
    directions_from_positions_to_camera,
    project_point_to_pixel,
    turntable_arc,
    turntable_camera,
    turntable_ring,
    viewmat_from_camera_pose,
)


def test_turntable_camera_center_at_correct_radius():
    cam = turntable_camera(t=5.0, phi_deg=0.0, theta_deg=30.0)
    assert abs(np.linalg.norm(cam.center) - 5.0) < 1e-9


def test_turntable_camera_forward_and_up_are_unit_and_orthogonal():
    cam = turntable_camera(t=5.0, phi_deg=20.0, theta_deg=70.0)
    assert abs(np.linalg.norm(cam.forward) - 1.0) < 1e-9
    assert abs(np.linalg.norm(cam.up) - 1.0) < 1e-9
    assert abs(np.dot(cam.forward, cam.up)) < 1e-9


def test_turntable_ring_covers_full_circle_of_radii():
    ring = turntable_ring(radius=4.0, n_views=8)
    assert len(ring) == 8
    for cam in ring:
        assert abs(np.linalg.norm(cam.center) - 4.0) < 1e-9


def test_turntable_arc_stays_within_half_width():
    arc = turntable_arc(radius=4.0, n_views=10, theta_center_deg=0.0, half_width_deg=10.0)
    ring_reference = turntable_camera(4.0, 0.0, 0.0)
    for cam in arc:
        cos_sep = np.dot(cam.center, ring_reference.center) / (
            np.linalg.norm(cam.center) * np.linalg.norm(ring_reference.center)
        )
        angle_deg = np.degrees(np.arccos(np.clip(cos_sep, -1.0, 1.0)))
        assert angle_deg < 15.0  # a bit more than half_width to allow for phi/projection slack


def test_directions_from_positions_to_camera_are_unit_vectors_pointing_at_camera():
    cam = turntable_camera(5.0, 0.0, 0.0)
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    dirs = directions_from_positions_to_camera(positions, cam)
    norms = np.linalg.norm(dirs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-9)

    expected0 = cam.center / np.linalg.norm(cam.center)
    np.testing.assert_allclose(dirs[0], expected0, atol=1e-9)


def test_camera_local_frame_is_orthonormal():
    cam = turntable_camera(5.0, 20.0, 70.0)
    right, up, forward = camera_local_frame(cam)
    for v in (right, up, forward):
        assert abs(np.linalg.norm(v) - 1.0) < 1e-9
    assert abs(np.dot(right, up)) < 1e-9
    assert abs(np.dot(right, forward)) < 1e-9
    assert abs(np.dot(up, forward)) < 1e-9


def test_viewmat_from_camera_pose_is_a_proper_rotation_plus_translation():
    cam = turntable_camera(5.0, 20.0, 70.0)
    viewmat = viewmat_from_camera_pose(cam)
    rotation = viewmat[:3, :3]
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
    assert abs(np.linalg.det(rotation) - 1.0) < 1e-9  # a proper rotation, not a reflection


def test_viewmat_from_camera_pose_maps_camera_center_to_the_origin():
    cam = turntable_camera(5.0, 20.0, 70.0)
    viewmat = viewmat_from_camera_pose(cam)
    center_in_camera_space = viewmat[:3, :3] @ cam.center + viewmat[:3, 3]
    np.testing.assert_allclose(center_in_camera_space, np.zeros(3), atol=1e-9)


def test_project_point_to_pixel_dead_ahead_point_lands_at_the_principal_point():
    cam = CameraPose(center=np.array([0.0, 0.0, 0.0]), forward=np.array([1.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))
    viewmat = viewmat_from_camera_pose(cam)
    K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    pixel = project_point_to_pixel(np.array([5.0, 0.0, 0.0]), viewmat, K)
    np.testing.assert_allclose(pixel, [50.0, 50.0], atol=1e-9)


def test_project_point_to_pixel_raises_for_a_point_behind_the_camera():
    cam = CameraPose(center=np.array([0.0, 0.0, 0.0]), forward=np.array([1.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))
    viewmat = viewmat_from_camera_pose(cam)
    K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError):
        project_point_to_pixel(np.array([-5.0, 0.0, 0.0]), viewmat, K)
