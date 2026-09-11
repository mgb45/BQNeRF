import numpy as np
import pytest

from gs_experiment.camera import turntable_arc, turntable_ring
from gs_experiment.splat_scene import (
    fit_kernel_hyperparams,
    fit_kernel_hyperparams_with_noise,
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


def test_load_from_gsplat_checkpoint_raises_on_missing_scene_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_from_gsplat_checkpoint(str(tmp_path / "nonexistent"))


def test_load_from_gsplat_checkpoint_round_trips_a_synthetic_checkpoint(tmp_path):
    from gs_experiment.nerf_transforms import write_transforms_json
    from gs_experiment.ply_io import write_3dgs_ply
    from gs_experiment.spherical_harmonics import random_sh_coeffs

    rng = np.random.default_rng(3)
    n = 25
    positions = rng.uniform(-0.5, 0.5, size=(n, 3))
    scales = rng.uniform(0.01, 0.05, size=(n, 3))
    rotations = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    opacities = rng.uniform(0.3, 0.9, size=n)
    sh_coeffs = random_sh_coeffs(rng, n_splats=n, degree=1, scale=0.2)

    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()
    write_3dgs_ply(str(scene_dir / "splats.ply"), positions, scales, rotations, opacities, sh_coeffs, sh_degree=1)

    cameras = turntable_ring(radius=6.0, n_views=8, phi_deg=30.0)

    def _c2w(camera):
        right = np.cross(camera.forward, camera.up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, camera.forward)
        c2w = np.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -camera.forward
        c2w[:3, 3] = camera.center
        return c2w

    frames = [{"file_path": f"images/r_{i:03d}", "transform_matrix": _c2w(c)} for i, c in enumerate(cameras)]
    write_transforms_json(str(scene_dir / "transforms.json"), camera_angle_x=np.deg2rad(50.0), frames=frames)

    scene = load_from_gsplat_checkpoint(str(scene_dir))

    np.testing.assert_allclose(scene.positions, positions, atol=1e-5)
    np.testing.assert_allclose(scene.opacities, opacities, atol=1e-4)
    np.testing.assert_allclose(scene.scales, scales, atol=1e-4)
    np.testing.assert_allclose(scene.sh_coeffs, sh_coeffs, atol=1e-4)
    assert scene.sh_degree == 1
    assert len(scene.cameras) == 8
    assert len(scene.observed_camera_idx) == n
    # every splat sits within a couple of units of the origin and every
    # camera orbits it at radius 6 with a generous FOV, so real frustum
    # attribution should find most splats observed by at least one camera
    # (not necessarily all: with 25 splats packed into a 1-unit cube and
    # only 8 viewpoints, some mutual occlusion is expected, not a bug)
    n_observed = sum(len(idx) > 0 for idx in scene.observed_camera_idx)
    assert n_observed >= n * 0.5


def test_load_from_gsplat_checkpoint_fields_satisfy_their_documented_invariants(tmp_path):
    """A real checkpoint's fields each cross a convention boundary on the
    way in (ply_io's pre-sigmoid logit / log-scale / not-necessarily-unit
    quaternion storage -> SplatScene's real-probability / real-scale /
    unit-quaternion fields), and the colors bug (FINDINGS.md section 4)
    passed 159 tests specifically because nothing checked a value like
    this was semantically sane, only that the pipeline ran. Uses
    deliberately non-unit-norm quaternions and opacities/scales spanning
    several orders of magnitude, unlike the identity-quaternion,
    middle-of-range round-trip test above, so a skipped or doubled
    conversion anywhere in load_from_gsplat_checkpoint's path would show
    up as an out-of-range value here, not just a wrong-but-plausible one."""
    from gs_experiment.nerf_transforms import write_transforms_json
    from gs_experiment.ply_io import write_3dgs_ply
    from gs_experiment.spherical_harmonics import random_sh_coeffs

    rng = np.random.default_rng(11)
    n = 20
    positions = rng.uniform(-2.0, 2.0, size=(n, 3))
    scales = rng.uniform(1e-3, 2.0, size=(n, 3))  # spans orders of magnitude
    rotations = rng.normal(size=(n, 4)) + np.array([1.0, 0.0, 0.0, 0.0])  # deliberately non-unit-norm
    opacities = rng.uniform(0.001, 0.999, size=n)  # spans close to both (0, 1) boundaries
    sh_coeffs = random_sh_coeffs(rng, n_splats=n, degree=2, scale=0.4)

    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()
    write_3dgs_ply(str(scene_dir / "splats.ply"), positions, scales, rotations, opacities, sh_coeffs, sh_degree=2)
    cameras = turntable_ring(radius=6.0, n_views=4, phi_deg=30.0)

    def _c2w(camera):
        right = np.cross(camera.forward, camera.up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, camera.forward)
        c2w = np.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -camera.forward
        c2w[:3, 3] = camera.center
        return c2w

    frames = [{"file_path": f"images/r_{i:03d}", "transform_matrix": _c2w(c)} for i, c in enumerate(cameras)]
    write_transforms_json(str(scene_dir / "transforms.json"), camera_angle_x=np.deg2rad(50.0), frames=frames)

    scene = load_from_gsplat_checkpoint(str(scene_dir))

    assert np.all(np.isfinite(scene.positions))
    assert np.all(scene.opacities > 0.0) and np.all(scene.opacities < 1.0)
    assert np.all(scene.scales > 0.0)
    np.testing.assert_allclose(np.linalg.norm(scene.rotations, axis=1), 1.0, atol=1e-5)
    # colors is the DC-only fallback (SH_C0 * raw + 0.5, see splat_scene.py) --
    # sane for these non-adversarial SH coefficients, not the raw
    # coefficient's much wider, sign-unrestricted range (this exact check,
    # against a real checkpoint, is what caught the colors bug originally).
    assert np.all(scene.colors > -0.5) and np.all(scene.colors < 1.5)


def test_load_from_gsplat_checkpoint_colors_matches_eval_sh_degree_zero(tmp_path):
    """Regression test for a real bug: `colors` was once the raw SH
    coefficient (`sh_coeffs[:,:,0].mean(axis=1)`), not a real color -- 3DGS
    stores SH coefficients as offsets from mid-gray (eval_sh's own "+ 0.5"
    convention), so a real checkpoint's raw `colors` spanned [-2.39, 2.37]
    and the marginal-likelihood-fitted RBF sigma changed by ~2x once
    corrected (gs_experiment/results/FINDINGS.md's calibration-methodology
    section). `colors` must always equal `eval_sh(sh_coeffs, *, degree=0)`
    exactly (direction-independent at degree 0), not merely "look colorful"."""
    from gs_experiment.nerf_transforms import write_transforms_json
    from gs_experiment.ply_io import write_3dgs_ply
    from gs_experiment.spherical_harmonics import random_sh_coeffs

    rng = np.random.default_rng(7)
    n = 12
    positions = rng.uniform(-0.5, 0.5, size=(n, 3))
    scales = rng.uniform(0.01, 0.05, size=(n, 3))
    rotations = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    opacities = rng.uniform(0.3, 0.9, size=n)
    sh_coeffs = random_sh_coeffs(rng, n_splats=n, degree=1, scale=0.2)

    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()
    write_3dgs_ply(str(scene_dir / "splats.ply"), positions, scales, rotations, opacities, sh_coeffs, sh_degree=1)
    cameras = turntable_ring(radius=6.0, n_views=4, phi_deg=30.0)

    def _c2w(camera):
        right = np.cross(camera.forward, camera.up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, camera.forward)
        c2w = np.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -camera.forward
        c2w[:3, 3] = camera.center
        return c2w

    frames = [{"file_path": f"images/r_{i:03d}", "transform_matrix": _c2w(c)} for i, c in enumerate(cameras)]
    write_transforms_json(str(scene_dir / "transforms.json"), camera_angle_x=np.deg2rad(50.0), frames=frames)

    scene = load_from_gsplat_checkpoint(str(scene_dir))

    dummy_directions = np.zeros((n, 3))  # degree 0 ignores direction entirely
    expected = eval_sh(sh_coeffs, dummy_directions, degree=0).mean(axis=-1)
    np.testing.assert_allclose(scene.colors, expected, atol=1e-8)
    # A real color's DC-only approximation should land close to [0, 1] for
    # typical (not adversarially large) SH coefficients -- not the raw
    # coefficient's much wider, sign-unrestricted range.
    assert scene.colors.min() > -0.5 and scene.colors.max() < 1.5


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


def test_splat_observations_return_rgb_matches_scalar_mean_and_uncollapsed_eval_sh():
    """`return_rgb=True`'s `values_rgb` must (a) leave the default 3-tuple
    return unaffected and (b) satisfy `values == values_rgb.mean(axis=-1)`
    exactly (the scalar path is a strict collapse of the RGB path, not an
    independent computation that happens to agree) and (c) actually carry
    3 distinct channels, not silently a broadcast scalar."""
    rng = np.random.default_rng(1)
    bounds = ((-5.0, 5.0), (-5.0, 5.0), (-1.0, 1.0))
    wide_cameras = turntable_ring(radius=8.0, n_views=6)
    scene = make_mock_scene(rng, n_splats=10, bounds=bounds, wide_cameras=wide_cameras)

    from gs_experiment.spherical_harmonics import random_sh_coeffs

    scene.sh_coeffs = random_sh_coeffs(rng, n_splats=10, degree=2, scale=1.0)
    scene.sh_degree = 2

    default_result = splat_observations(scene)
    assert len(default_result) == 3  # return_rgb defaults to False: unchanged 3-tuple

    positions, directions, values, values_rgb = splat_observations(scene, return_rgb=True)
    assert values_rgb.shape == (values.shape[0], 3)
    np.testing.assert_allclose(values, values_rgb.mean(axis=-1), atol=1e-12)
    # a real per-channel signal, not R==G==B broadcast from the scalar
    assert not np.allclose(values_rgb[:, 0], values_rgb[:, 1])


def test_splat_observations_return_rgb_without_sh_broadcasts_flat_color():
    """No sh_coeffs -> no real per-channel signal exists either way; the
    RGB path must still return a valid (n_rows, 3) array (broadcast from
    the flat scalar), not crash or return None."""
    rng = np.random.default_rng(2)
    bounds = ((-5.0, 5.0), (-5.0, 5.0), (-1.0, 1.0))
    wide_cameras = turntable_ring(radius=8.0, n_views=6)
    scene = make_mock_scene(rng, n_splats=6, bounds=bounds, wide_cameras=wide_cameras)
    assert scene.sh_coeffs is None

    _, _, values, values_rgb = splat_observations(scene, return_rgb=True)
    assert values_rgb.shape == (values.shape[0], 3)
    np.testing.assert_allclose(values_rgb[:, 0], values)
    np.testing.assert_allclose(values_rgb[:, 1], values)
    np.testing.assert_allclose(values_rgb[:, 2], values)


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


def test_fit_kernel_hyperparams_with_noise_returns_sane_values_and_matches_kappa():
    """Smoke test against a real (synthetic-but-real-shaped) scene with
    enough splats/observations for both the sigma+noise and kappa fits to
    actually run, not hit the "too little data" None branch. Checks: sigma
    and noise_variance are both real, positive, in-bounds floats; kappa
    matches `fit_kernel_hyperparams`'s own kappa exactly (same fitting
    code, untouched -- the two functions must not silently diverge on the
    one hyperparameter they share)."""
    rng = np.random.default_rng(4)
    bounds = ((-2.0, 2.0), (-2.0, 2.0), (-1.0, 1.0))
    wide_cameras = turntable_ring(radius=8.0, n_views=10)
    scene = make_mock_scene(rng, n_splats=150, bounds=bounds, wide_cameras=wide_cameras)

    sigma, noise_variance, kappa = fit_kernel_hyperparams_with_noise(
        scene, sigma_bounds=(0.01, 2.0), noise_bounds=(1e-4, 1.0), window_radius=0.5, seed=0,
    )
    assert sigma is not None and noise_variance is not None
    assert 0.01 <= sigma <= 2.0
    assert 1e-4 <= noise_variance <= 1.0

    _, kappa_baseline = fit_kernel_hyperparams(scene, window_radius=0.5, seed=0)
    assert kappa == kappa_baseline


def test_fit_kernel_hyperparams_with_noise_returns_none_on_too_little_data():
    """Mirrors fit_kernel_hyperparams's own too-little-data contract: with
    fewer than 6 real (opacity-eligible) splats, sigma/noise_variance must
    come back None together, not a spuriously "confident" fit."""
    rng = np.random.default_rng(5)
    bounds = ((-2.0, 2.0), (-2.0, 2.0), (-1.0, 1.0))
    scene = make_mock_scene(rng, n_splats=3, bounds=bounds, wide_cameras=turntable_ring(radius=8.0, n_views=4))

    sigma, noise_variance, kappa = fit_kernel_hyperparams_with_noise(scene, seed=0)
    assert sigma is None and noise_variance is None
