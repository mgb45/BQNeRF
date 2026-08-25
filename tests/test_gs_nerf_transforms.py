import numpy as np

from gs_experiment.camera import turntable_camera
from gs_experiment.nerf_transforms import (
    camera_pose_from_c2w,
    fov_x_to_intrinsics,
    load_transforms,
    opencv_viewmat_from_c2w,
    write_transforms_json,
)


def _c2w_from_camera_pose(camera):
    right = np.cross(camera.forward, camera.up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, camera.forward)
    back = -camera.forward
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = back
    c2w[:3, 3] = camera.center
    return c2w


def test_camera_pose_from_c2w_round_trips_turntable_camera():
    cam = turntable_camera(t=5.0, phi_deg=20.0, theta_deg=70.0)
    c2w = _c2w_from_camera_pose(cam)

    recovered = camera_pose_from_c2w(c2w)

    np.testing.assert_allclose(recovered.center, cam.center, atol=1e-10)
    np.testing.assert_allclose(recovered.forward, cam.forward, atol=1e-10)
    np.testing.assert_allclose(recovered.up, cam.up, atol=1e-10)


def test_opencv_viewmat_places_camera_at_origin_looking_down_positive_z():
    cam = turntable_camera(t=5.0, phi_deg=0.0, theta_deg=0.0)
    c2w = _c2w_from_camera_pose(cam)

    world2cam = opencv_viewmat_from_c2w(c2w)

    # the camera's own center, in its own (OpenCV) camera frame, is the origin
    cam_center_in_cam_frame = world2cam @ np.array([*cam.center, 1.0])
    np.testing.assert_allclose(cam_center_in_cam_frame[:3], 0.0, atol=1e-10)

    # a point straight ahead of the camera (along `forward`) must land on
    # +z in OpenCV's camera-space convention (x right, y down, z forward)
    point_ahead = cam.center + cam.forward * 2.0
    point_in_cam_frame = world2cam @ np.array([*point_ahead, 1.0])
    assert point_in_cam_frame[2] > 0
    np.testing.assert_allclose(point_in_cam_frame[:2], 0.0, atol=1e-10)


def test_write_then_load_transforms_json_round_trips(tmp_path):
    frames = [
        {"file_path": "images/r_000", "transform_matrix": np.eye(4)},
        {"file_path": "images/r_001", "transform_matrix": np.eye(4) * 2},
    ]
    path = tmp_path / "transforms.json"
    write_transforms_json(str(path), camera_angle_x=0.7, frames=frames)

    camera_angle_x, loaded_frames = load_transforms(str(path))

    assert camera_angle_x == 0.7
    assert [f[0] for f in loaded_frames] == ["images/r_000", "images/r_001"]
    np.testing.assert_allclose(loaded_frames[0][1], np.eye(4))


def test_fov_x_to_intrinsics_centers_principal_point():
    K = fov_x_to_intrinsics(camera_angle_x=2 * np.arctan(1.0), width=200, height=200)
    assert K[0, 2] == 100.0
    assert K[1, 2] == 100.0
    np.testing.assert_allclose(K[0, 0], 100.0, atol=1e-9)
