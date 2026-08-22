import numpy as np

from gs_experiment.camera import (
    directions_from_positions_to_camera,
    turntable_arc,
    turntable_camera,
    turntable_ring,
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
