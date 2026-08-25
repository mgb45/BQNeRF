import numpy as np

from bq_splat.kernels import DirectionalKernel
from gs_experiment.camera import turntable_arc, turntable_ring
from gs_experiment.pixel_uncertainty import LocalUncertaintyEngine, make_default_3d_position_kernel
from gs_experiment.splat_scene import make_mock_scene, splat_observations


def build_engine(seed=0):
    rng = np.random.default_rng(seed)
    bounds = ((-5.0, 5.0), (-5.0, 5.0), (-1.0, 1.0))
    wide_cameras = turntable_ring(radius=8.0, n_views=10)
    narrow_cameras = turntable_arc(radius=8.0, n_views=10, theta_center_deg=0.0, half_width_deg=15.0)
    scene = make_mock_scene(
        rng, n_splats=150, bounds=bounds, wide_cameras=wide_cameras, narrow_cameras=narrow_cameras,
        narrow_zone_center=np.array([2.0, 2.0, 0.0]), narrow_zone_radius=1.2,
    )
    positions, directions, values = splat_observations(scene)
    pos_kernel = make_default_3d_position_kernel(sigma=0.9)
    dir_kernel = DirectionalKernel(kappa=4.0)
    engine = LocalUncertaintyEngine(
        positions=positions, values=values, pos_kernel=pos_kernel, scene_bounds=bounds,
        directions=directions, dir_kernel=dir_kernel,
    )
    return engine, scene


def test_engine_builds_and_local_neighbors_returns_indices_within_radius():
    engine, _ = build_engine()
    q = np.array([0.0, 0.0, 0.0])
    idx = engine.local_neighbors(q, radius=1.5)
    if idx.shape[0] > 0:
        dists = np.linalg.norm(engine.positions[idx] - q, axis=1)
        assert np.all(dists <= 1.5 + 1e-9)


def test_spatial_only_variance_is_finite_and_nonnegative():
    engine, _ = build_engine()
    for q in [np.array([0.0, 0.0, 0.0]), np.array([4.5, 4.5, 0.0]), np.array([-4.5, -4.5, 0.0])]:
        result = engine.spatial_only_variance(q, radius=1.5)
        assert np.isfinite(result.mean)
        assert result.variance >= 0.0


def test_directional_variance_is_finite_and_nonnegative():
    engine, _ = build_engine()
    query_direction = np.array([0.0, 0.0, 1.0])
    for q in [np.array([0.0, 0.0, 0.0]), np.array([2.0, 2.0, 0.0])]:
        result = engine.directional_variance(q, query_direction, radius=1.5)
        assert np.isfinite(result.mean)
        assert result.variance >= 0.0


def test_vv_cache_reused_for_repeated_window_size():
    engine, _ = build_engine()
    engine.spatial_only_variance(np.array([0.0, 0.0, 0.0]), radius=1.5)
    cache_size_after_one = len(engine._vv_cache)
    engine.spatial_only_variance(np.array([1.0, -1.0, 0.0]), radius=1.5)  # same window size, different center
    assert len(engine._vv_cache) == cache_size_after_one  # no new cache entry for an interior, same-size window


def test_precomputed_vv_matches_recomputed_vv():
    """The engine's cached vv should agree with directly calling
    pos_kernel.vv on the same bounds -- catches any mismatch between the
    cache key and what bayesian_quadrature_nd would compute on its own."""
    engine, _ = build_engine()
    from gs_experiment.pixel_uncertainty import box_bounds

    q = np.array([0.0, 0.0, 0.0])
    bounds = box_bounds(q, 1.5, engine.scene_bounds)
    cached = engine._cached_vv(bounds)
    direct = float(engine.pos_kernel.vv(bounds))
    assert abs(cached - direct) < 1e-9


def test_exclude_idx_removes_self_from_a_ball_query_centered_on_it():
    """Querying at a real splat's own position always finds that splat at
    distance 0 -- exclude_idx must filter it out, the basis for a
    leave-one-out calibration check (gs_experiment/calibration_experiment.py)
    not trivially seeing its own held-out answer."""
    engine, _ = build_engine()
    self_idx = 7
    q = engine.positions[self_idx]

    idx_with_self = engine.local_neighbors(q, radius=1.5)
    idx_without_self = engine.local_neighbors(q, radius=1.5, exclude_idx=self_idx)

    assert self_idx in idx_with_self
    assert self_idx not in idx_without_self
    assert len(idx_without_self) == len(idx_with_self) - 1


def test_spatial_only_variance_exclude_idx_changes_the_result():
    """A real behavioral check, not just an index-membership check: leaving
    a point's own (position, value) out of the local BQ solve should
    generally change both the posterior mean and variance relative to
    including it -- if it didn't, exclude_idx wouldn't be doing anything."""
    engine, _ = build_engine()
    self_idx = 7
    q = engine.positions[self_idx]

    with_self = engine.spatial_only_variance(q, radius=1.5)
    without_self = engine.spatial_only_variance(q, radius=1.5, exclude_idx=self_idx)

    assert with_self.variance != without_self.variance


def test_directional_variance_higher_for_query_outside_narrow_zone_cone():
    """The gs_experiment-level analogue of
    scripts/validate_directional_combined.py's core claim, using real 3D
    camera poses instead of a 2D angle parameterization."""
    engine, scene = build_engine()
    from gs_experiment.camera import directions_from_positions_to_camera

    narrow_center = np.array([2.0, 2.0, 0.0])
    narrow_camera = scene.cameras[-1]  # one of the narrow-arc cameras
    typical_dir = directions_from_positions_to_camera(narrow_center.reshape(1, -1), narrow_camera)[0]
    outside_dir = -typical_dir

    narrow_result = engine.directional_variance(narrow_center, outside_dir, radius=1.5)
    narrow_result_inside = engine.directional_variance(narrow_center, typical_dir, radius=1.5)
    assert narrow_result.variance > narrow_result_inside.variance
