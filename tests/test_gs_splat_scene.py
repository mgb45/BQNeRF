import numpy as np
import pytest

from gs_experiment.camera import turntable_arc, turntable_ring
from gs_experiment.splat_scene import (
    load_from_gsplat_checkpoint,
    make_mock_scene,
    make_occluder_scene,
    splat_observations,
)
from gs_experiment.spherical_harmonics import eval_sh


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


def test_splat_observations_uses_sh_when_present_and_is_view_dependent():
    rng = np.random.default_rng(0)
    bounds = ((-5.0, 5.0), (-5.0, 5.0), (-1.0, 1.0))
    wide_cameras = turntable_ring(radius=8.0, n_views=6)
    scene = make_mock_scene(rng, n_splats=10, bounds=bounds, wide_cameras=wide_cameras)

    from gs_experiment.spherical_harmonics import random_sh_coeffs

    scene.sh_coeffs = random_sh_coeffs(rng, n_splats=10, degree=2, scale=1.0)
    scene.sh_degree = 2

    _, directions, values = splat_observations(scene)

    # recompute expected values in the same (splat, camera) iteration order
    # splat_observations uses, directly from eval_sh, to check the wiring
    # rather than just that *some* values came out.
    expected = []
    row = 0
    for i, cam_idx in enumerate(scene.observed_camera_idx):
        for _ in cam_idx:
            expected.append(float(np.mean(eval_sh(scene.sh_coeffs[i], directions[row], scene.sh_degree))))
            row += 1
    np.testing.assert_allclose(values, np.array(expected), atol=1e-12)

    # and it should actually be direction-dependent -- not silently
    # collapsing back to the flat-color path
    assert len(set(np.round(values, 6))) > 1


def test_make_occluder_scene_front_cameras_do_not_see_targets_back_cameras_do():
    rng = np.random.default_rng(1)
    scene, info = make_occluder_scene(rng, n_wall_splats=60, n_target_splats=40, n_cameras_per_side=6)

    n_wall = info["n_wall_splats"]
    target_indices = np.arange(n_wall, scene.positions.shape[0])
    front_cams = set(info["front_camera_idx"].tolist())
    back_cams = set(info["back_camera_idx"].tolist())

    front_sees_any_target = any(
        set(scene.observed_camera_idx[i].tolist()) & front_cams for i in target_indices
    )
    back_sees_most_targets = sum(
        bool(set(scene.observed_camera_idx[i].tolist()) & back_cams) for i in target_indices
    )

    assert not front_sees_any_target, "wall should occlude every target splat from every front camera"
    assert back_sees_most_targets > 0.5 * len(target_indices), "back cameras should see most targets directly"


def test_make_occluder_scene_pipeline_runs_through_splat_observations():
    rng = np.random.default_rng(2)
    scene, _ = make_occluder_scene(rng, n_wall_splats=30, n_target_splats=20, n_cameras_per_side=4)
    positions, directions, values = splat_observations(scene)
    assert positions.shape[0] > 0
    assert np.all(np.isfinite(values))
