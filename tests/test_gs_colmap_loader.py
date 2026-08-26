import numpy as np

from gs_experiment.colmap_loader import ColmapCamera, ColmapImage, camera_angle_x_from_camera, colmap_image_to_c2w_opengl, qvec_to_rotmat
from gs_experiment.nerf_transforms import opencv_viewmat_from_c2w


def test_qvec_to_rotmat_identity_quaternion_gives_identity():
    R = qvec_to_rotmat(np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(R, np.eye(3), atol=1e-12)


def test_qvec_to_rotmat_produces_a_valid_rotation_matrix():
    # an arbitrary, non-trivial unit quaternion
    q = np.array([0.82376014, -0.22333718, -0.44771114, -0.26663545])
    q = q / np.linalg.norm(q)
    R = qvec_to_rotmat(q)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)


def test_colmap_image_to_c2w_opengl_round_trips_through_opencv_viewmat_from_c2w():
    """colmap_image_to_c2w_opengl is documented as the exact inverse of
    nerf_transforms.opencv_viewmat_from_c2w's operation -- checked
    directly: build a w2c in COLMAP's convention from a known camera
    pose, recover c2w via colmap_image_to_c2w_opengl, then re-derive the
    w2c via this project's own opencv_viewmat_from_c2w and confirm it
    matches what we started with."""
    rng = np.random.default_rng(0)
    q = rng.normal(size=4)
    q = q / np.linalg.norm(q)
    t = rng.normal(size=3) * 2.0

    image = ColmapImage(image_id=1, qvec=q, tvec=t, camera_id=1, name="foo.jpg")
    c2w_opengl = colmap_image_to_c2w_opengl(image)

    w2c_cv_recovered = opencv_viewmat_from_c2w(c2w_opengl)

    R = qvec_to_rotmat(q)
    w2c_cv_expected = np.eye(4)
    w2c_cv_expected[:3, :3] = R
    w2c_cv_expected[:3, 3] = t

    np.testing.assert_allclose(w2c_cv_recovered, w2c_cv_expected, atol=1e-8)


def test_camera_angle_x_from_camera_matches_known_focal_length():
    # fx such that horizontal FOV is exactly 90 degrees for a 100px-wide image:
    # fx = (width/2) / tan(45deg) = 50
    camera = ColmapCamera(camera_id=1, model="PINHOLE", width=100, height=100, params=np.array([50.0, 50.0, 50.0, 50.0]))
    angle = camera_angle_x_from_camera(camera)
    assert np.isclose(np.degrees(angle), 90.0, atol=1e-6)


def test_camera_angle_x_from_camera_is_resolution_independent():
    """Scaling width and fx by the same factor (e.g. a downsampled image
    from the same physical camera) should not change the derived FOV --
    this is what lets real_capture_gradient_experiment.py use a
    resolution the COLMAP calibration wasn't computed at."""
    cam_full = ColmapCamera(camera_id=1, model="PINHOLE", width=3118, height=2078, params=np.array([3222.7, 3222.7, 1559.0, 1039.0]))
    cam_half = ColmapCamera(camera_id=1, model="PINHOLE", width=1559, height=1039, params=np.array([1611.35, 1611.35, 779.5, 519.5]))
    assert np.isclose(camera_angle_x_from_camera(cam_full), camera_angle_x_from_camera(cam_half), atol=1e-6)
