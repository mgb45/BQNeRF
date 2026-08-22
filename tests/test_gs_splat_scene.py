import numpy as np
import pytest

from gs_experiment.camera import turntable_arc, turntable_ring
from gs_experiment.splat_scene import load_from_gsplat_checkpoint, make_mock_scene, splat_observations


def test_make_mock_scene_assigns_narrow_zone_splats_only_narrow_cameras():
    rng = np.random.default_rng(0)
    bounds = ((-5.0, 5.0), (-5.0, 5.0), (-1.0, 1.0))
    wide_cameras = turntable_ring(radius=8.0, n_views=6)
    narrow_cameras = turntable_arc(radius=8.0, n_views=6, theta_center_deg=0.0, half_width_deg=10.0)
    narrow_center = np.array([2.0, 2.0, 0.0])

    scene = make_mock_scene(
        rng, n_splats=100, bounds=bounds, wide_cameras=wide_cameras, narrow_cameras=narrow_cameras,
        narrow_zone_center=narrow_center, narrow_zone_radius=1.2,
    )

    n_wide_cams = len(wide_cameras)
    for i, p in enumerate(scene.positions):
        cam_idx = scene.observed_camera_idx[i]
        if np.linalg.norm(p - narrow_center) < 1.2:
            assert np.all(cam_idx >= n_wide_cams)
        else:
            assert np.all(cam_idx < n_wide_cams)


def test_splat_observations_produces_one_row_per_splat_camera_pair():
    rng = np.random.default_rng(1)
    bounds = ((-5.0, 5.0), (-5.0, 5.0), (-1.0, 1.0))
    wide_cameras = turntable_ring(radius=8.0, n_views=5)
    scene = make_mock_scene(rng, n_splats=20, bounds=bounds, wide_cameras=wide_cameras)

    positions, directions, values = splat_observations(scene)

    expected_rows = sum(len(idx) for idx in scene.observed_camera_idx)
    assert positions.shape == (expected_rows, 3)
    assert directions.shape == (expected_rows, 3)
    assert values.shape == (expected_rows,)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-9)


def test_load_from_gsplat_checkpoint_is_a_documented_stub():
    with pytest.raises(NotImplementedError):
        load_from_gsplat_checkpoint("/nonexistent/checkpoint.ply")
