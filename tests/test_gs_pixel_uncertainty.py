import numpy as np
import pytest

from bq_splat.kernels import DirectionalKernel, MaternKernel, ProductKernel
from gs_experiment.camera import CameraPose, turntable_arc, turntable_ring
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


def opacities_per_observation(scene):
    """Per-(splat, observing-camera) opacity, aligned row-for-row with
    splat_observations's output (same iteration order over
    scene.observed_camera_idx) -- the opacities.py module doesn't do this
    expansion itself since only rendering_aware_variance's tests need it."""
    opacities = []
    for i, cam_idx in enumerate(scene.observed_camera_idx):
        for _ in cam_idx:
            opacities.append(scene.opacities[i])
    return np.array(opacities)


def build_engine_with_opacities(seed=0, opacity_scale=1.0):
    rng = np.random.default_rng(seed)
    bounds = ((-5.0, 5.0), (-5.0, 5.0), (-1.0, 1.0))
    wide_cameras = turntable_ring(radius=8.0, n_views=10)
    scene = make_mock_scene(rng, n_splats=150, bounds=bounds, wide_cameras=wide_cameras)
    positions, directions, values = splat_observations(scene)
    opacities = opacity_scale * opacities_per_observation(scene)
    pos_kernel = make_default_3d_position_kernel(sigma=0.9)
    engine = LocalUncertaintyEngine(
        positions=positions, values=values, pos_kernel=pos_kernel, scene_bounds=bounds, opacities=opacities,
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
    leave-one-out calibration check (gs_experiment/evaluate_checkpoint.py calibration)
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
    bq_splat/validate.py --check directional-combined's core claim, using real 3D
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


def test_rendering_aware_variance_is_finite_and_nonnegative():
    engine, _ = build_engine_with_opacities()
    for q in [np.array([0.0, 0.0, 0.0]), np.array([4.5, 4.5, 0.0]), np.array([-4.5, -4.5, 0.0])]:
        result = engine.rendering_aware_variance(q, radius=1.5)
        assert np.isfinite(result.mean)
        assert result.variance >= 0.0


def test_rendering_aware_variance_requires_opacities():
    engine, _ = build_engine()  # no opacities passed
    with pytest.raises(ValueError):
        engine.rendering_aware_variance(np.array([0.0, 0.0, 0.0]), radius=1.5)


def test_rendering_aware_variance_requires_rbf_pos_kernel():
    rng = np.random.default_rng(0)
    bounds = ((-5.0, 5.0), (-5.0, 5.0), (-1.0, 1.0))
    scene = make_mock_scene(rng, n_splats=50, bounds=bounds, wide_cameras=turntable_ring(radius=8.0, n_views=5))
    positions, _, values = splat_observations(scene)
    opacities = opacities_per_observation(scene)
    matern_kernel = ProductKernel([MaternKernel(rho=0.5), MaternKernel(rho=0.5), MaternKernel(rho=0.5)])
    engine = LocalUncertaintyEngine(
        positions=positions, values=values, pos_kernel=matern_kernel, scene_bounds=bounds, opacities=opacities,
    )
    with pytest.raises(ValueError):
        engine.rendering_aware_variance(np.array([0.0, 0.0, 0.0]), radius=1.5)


def test_rendering_aware_variance_is_lower_for_lower_opacity_neighborhood():
    """The a_q closed form makes both the reported mean and variance scale
    with (opacity amplitude)^2-ish quantities (see
    bq_splat/render_weight.py / rendering_aware_prior_variance): a splat
    neighborhood the renderer barely uses (low opacity, e.g. mostly
    transparent) should report a smaller magnitude than an otherwise
    identical high-opacity neighborhood -- consistent with real alpha
    compositing, where a near-transparent splat contributes almost nothing
    to either the rendered color or its uncertainty."""
    low_engine, _ = build_engine_with_opacities(opacity_scale=0.02)
    high_engine, _ = build_engine_with_opacities(opacity_scale=1.0)

    q = np.array([0.0, 0.0, 0.0])
    low_result = low_engine.rendering_aware_variance(q, radius=1.5)
    high_result = high_engine.rendering_aware_variance(q, radius=1.5)

    assert low_result.variance < high_result.variance


def test_rendering_aware_variance_exclude_idx_changes_the_result():
    engine, _ = build_engine_with_opacities()
    self_idx = 7
    q = engine.positions[self_idx]

    with_self = engine.rendering_aware_variance(q, radius=1.5)
    without_self = engine.rendering_aware_variance(q, radius=1.5, exclude_idx=self_idx)

    assert with_self.variance != without_self.variance


def build_occluder_engine(occluder_opacity=1.0):
    """A minimal, exact scene for rendering_aware_variance_along_ray: a
    camera looking down +x, a fully (or near-fully) opaque occluder at
    x=2, and a target at x=5, same bearing -- the same geometry
    tests/test_gs_visibility_attribution.py's alpha-compositing tests use,
    now exercised through the engine end to end."""
    bounds = ((-1.0, 10.0), (-5.0, 5.0), (-5.0, 5.0))
    positions = np.array([[2.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    occluder_color, target_color = 0.0, 10.0
    values = np.array([occluder_color, target_color])
    opacities = np.array([occluder_opacity, 0.9])
    camera = CameraPose(center=np.array([-10.0, 0.0, 0.0]), forward=np.array([1.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))
    pos_kernel = make_default_3d_position_kernel(sigma=1.0)
    engine = LocalUncertaintyEngine(
        positions=positions, values=values, pos_kernel=pos_kernel, scene_bounds=bounds, opacities=opacities,
    )
    return engine, camera, occluder_color, target_color


def test_rendering_aware_variance_along_ray_is_finite_and_nonnegative():
    engine, camera, _, _ = build_occluder_engine()
    result = engine.rendering_aware_variance_along_ray(np.array([3.5, 0.0, 0.0]), camera, radius=3.0)
    assert np.isfinite(result.mean)
    assert result.variance >= 0.0


def test_rendering_aware_variance_along_ray_weights_toward_the_occluder_not_the_occluded_target():
    """The key behavioral fix over rendering_aware_variance: with a fully
    opaque occluder in front of a target on the same ray, the along-ray
    mean should land right on the occluder's color, since the occluder's
    real alpha-compositing transmittance weight completely dominates
    (w_occluder=1.0, w_target=(1-1.0)*0.9=0). The occlusion-blind
    rendering_aware_variance has no such preference built in (query point
    sits equidistant between the two splats) and lands far from it --
    a GP posterior mean is a linear predictor, not a convex combination of
    observed colors, so it isn't just "the average" of 0 and 10, but
    whatever it is, it is not the occluder-dominated answer."""
    engine, camera, occluder_color, target_color = build_occluder_engine(occluder_opacity=1.0)
    query_point = np.array([3.5, 0.0, 0.0])  # equidistant from both splats

    along_ray = engine.rendering_aware_variance_along_ray(query_point, camera, radius=3.0)
    occlusion_blind = engine.rendering_aware_variance(query_point, radius=3.0)

    assert abs(along_ray.mean - occluder_color) < 1e-6
    assert abs(occlusion_blind.mean - occluder_color) > 1.0


def test_rendering_aware_variance_along_ray_partial_occluder_lets_some_target_weight_through():
    """A partially-transparent occluder (opacity 0.3, not 1.0) should still
    let some of the target's transmittance weight through
    ((1-0.3)*0.9 = 0.63, vs. the occluder's own 0.3) -- so the along-ray
    mean should land closer to the target than the fully-opaque case does."""
    fully_opaque_engine, camera, occluder_color, target_color = build_occluder_engine(occluder_opacity=1.0)
    partial_engine, _, _, _ = build_occluder_engine(occluder_opacity=0.3)
    query_point = np.array([3.5, 0.0, 0.0])

    fully_opaque_result = fully_opaque_engine.rendering_aware_variance_along_ray(query_point, camera, radius=3.0)
    partial_result = partial_engine.rendering_aware_variance_along_ray(query_point, camera, radius=3.0)

    assert partial_result.mean > fully_opaque_result.mean


def test_rendering_aware_variance_along_ray_falls_back_gracefully_when_nothing_is_on_ray():
    """A splat that's nearby in plain 3D distance but far off to the side
    of this specific ray (a_q is identically 0 for it) should make the
    posterior collapse toward ~0 mean, ~0 variance, not raise or blow up."""
    engine, camera, _, _ = build_occluder_engine()
    off_ray_query = np.array([3.5, 4.0, 0.0])  # near the target in 3D, but off this camera's forward axis
    result = engine.rendering_aware_variance_along_ray(off_ray_query, camera, radius=1.0, angular_tol=0.01)
    assert np.isfinite(result.mean)
    assert abs(result.mean) < 1e-3
    assert result.variance < 1e-3


def test_rendering_aware_variance_along_ray_requires_opacities():
    rng = np.random.default_rng(0)
    bounds = ((-5.0, 5.0), (-5.0, 5.0), (-1.0, 1.0))
    scene = make_mock_scene(rng, n_splats=20, bounds=bounds, wide_cameras=turntable_ring(radius=8.0, n_views=5))
    positions, _, values = splat_observations(scene)
    pos_kernel = make_default_3d_position_kernel(sigma=0.9)
    engine = LocalUncertaintyEngine(positions=positions, values=values, pos_kernel=pos_kernel, scene_bounds=bounds)
    camera = CameraPose(center=np.array([-10.0, 0.0, 0.0]), forward=np.array([1.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))
    with pytest.raises(ValueError):
        engine.rendering_aware_variance_along_ray(np.array([0.0, 0.0, 0.0]), camera, radius=1.5)
