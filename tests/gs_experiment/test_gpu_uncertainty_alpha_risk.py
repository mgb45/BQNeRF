"""GPU-only tests for gpu_uncertainty.compute_alpha_risk_batched -- the
batched-GPU equivalent of `LocalUncertaintyEngine.
rendering_aware_alpha_risk_along_ray`'s `.alpha_mean`/`.alpha_risk`. This is
u_spatial_BQ(q), the position-only (direction-blind) "finite spatial
representation" term of the renderer-consistent sparse-GP decomposition
(mu_q = C_alpha(q), u_q = u_spatial_BQ(q) + b_q^T Sigma_theta b_q -- see
gs_experiment/sh_directional_uncertainty.py's module docstring).

Skipped automatically (not failed) when torch/CUDA aren't available,
matching every other GPU test file's own convention. Core claim tested:
numerical agreement with the already-validated scalar per-pixel path.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("gpu_uncertainty needs a real GPU", allow_module_level=True)

from gs_experiment.camera import CameraPose  # noqa: E402
from gs_experiment.gpu_uncertainty import compute_alpha_risk_batched  # noqa: E402
from gs_experiment.pixel_uncertainty import LocalUncertaintyEngine, make_default_3d_position_kernel  # noqa: E402

SIGMA = 1.0
RADIUS = 3.0
ANGULAR_TOL = 0.2


def build_position_only_engine(n_splats: int = 24, seed: int = 0, with_covariances: bool = True):
    """A position-only engine (no directions/dir_kernel at all) -- this
    term is deliberately direction-blind."""
    rng = np.random.default_rng(seed)
    positions = np.stack(
        [rng.uniform(4.0, 6.0, n_splats), rng.uniform(-1.5, 1.5, n_splats), rng.uniform(-1.5, 1.5, n_splats)], axis=1
    )
    values = rng.uniform(0.0, 10.0, n_splats)
    opacities = rng.uniform(0.1, 0.9, n_splats)
    camera = CameraPose(center=np.array([-10.0, 0.0, 0.0]), forward=np.array([1.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))
    pos_kernel = make_default_3d_position_kernel(sigma=SIGMA)
    scales = rng.uniform(0.02, 0.08, size=(n_splats, 3)) if with_covariances else None
    rotations = np.tile([1.0, 0.0, 0.0, 0.0], (n_splats, 1)) if with_covariances else None
    engine = LocalUncertaintyEngine(
        positions=positions, values=values, pos_kernel=pos_kernel,
        scene_bounds=((-1.0, 10.0), (-5.0, 5.0), (-5.0, 5.0)),
        opacities=opacities, scales=scales, rotations=rotations,
    )
    return engine, camera


def scalar_alpha_risk(engine, camera_index, points, max_candidates=500):
    means, risks = [], []
    for p in points:
        result = engine.rendering_aware_alpha_risk_along_ray(
            p, camera_index, RADIUS, angular_tol=ANGULAR_TOL, max_candidates=max_candidates,
        )
        means.append(result.alpha_mean)
        risks.append(result.alpha_risk)
    return np.array(means), np.array(risks)


def test_batched_matches_scalar_with_real_covariances():
    engine, camera = build_position_only_engine()
    camera_index = engine.build_bearing_index(camera)

    rng = np.random.default_rng(1)
    n_queries = 30
    points = np.stack(
        [rng.uniform(4.5, 5.5, n_queries), rng.uniform(-1.0, 1.0, n_queries), rng.uniform(-1.0, 1.0, n_queries)], axis=1
    )

    scalar_mean, scalar_risk = scalar_alpha_risk(engine, camera_index, points)
    batched_mean, batched_risk = compute_alpha_risk_batched(
        engine, camera_index, points, angular_tol=ANGULAR_TOL, sigma_rbf=SIGMA, max_candidates=500, device="cuda",
    )

    assert np.allclose(batched_mean, scalar_mean, rtol=1e-6, atol=1e-8)
    assert np.allclose(batched_risk, scalar_risk, rtol=1e-6, atol=1e-8)


def test_batched_matches_scalar_without_covariances():
    engine, camera = build_position_only_engine(seed=11, with_covariances=False)
    camera_index = engine.build_bearing_index(camera)

    rng = np.random.default_rng(12)
    n_queries = 20
    points = np.stack(
        [rng.uniform(4.5, 5.5, n_queries), rng.uniform(-1.0, 1.0, n_queries), rng.uniform(-1.0, 1.0, n_queries)], axis=1
    )

    scalar_mean, scalar_risk = scalar_alpha_risk(engine, camera_index, points)
    batched_mean, batched_risk = compute_alpha_risk_batched(
        engine, camera_index, points, angular_tol=ANGULAR_TOL, sigma_rbf=SIGMA, max_candidates=500, device="cuda",
    )

    assert np.allclose(batched_mean, scalar_mean, rtol=1e-6, atol=1e-8)
    assert np.allclose(batched_risk, scalar_risk, rtol=1e-6, atol=1e-8)


def test_batched_matches_scalar_with_candidate_cap_forcing_ranking():
    """max_candidates smaller than the real candidate count forces the
    bearing-distance overflow ranking -- must still match the scalar path
    (CameraSplatIndex.query's own query_direction=None ranking)."""
    engine, camera = build_position_only_engine(n_splats=40, seed=20)
    camera_index = engine.build_bearing_index(camera)
    point = np.array([[5.0, 0.0, 0.0]])

    scalar_mean, scalar_risk = scalar_alpha_risk(engine, camera_index, point, max_candidates=5)
    batched_mean, batched_risk = compute_alpha_risk_batched(
        engine, camera_index, point, angular_tol=ANGULAR_TOL, sigma_rbf=SIGMA, max_candidates=5, device="cuda",
    )

    assert np.allclose(batched_mean, scalar_mean, rtol=1e-6, atol=1e-8)
    assert np.allclose(batched_risk, scalar_risk, rtol=1e-6, atol=1e-8)


def test_batched_handles_empty_camera_index():
    positions = np.array([[5.0, 0.0, 0.0]])
    values = np.array([1.0])
    opacities = np.array([0.9])
    pos_kernel = make_default_3d_position_kernel(sigma=SIGMA)
    engine = LocalUncertaintyEngine(
        positions=positions, values=values, pos_kernel=pos_kernel, scene_bounds=((-1.0, 10.0), (-5.0, 5.0), (-5.0, 5.0)),
        opacities=opacities,
    )
    camera = CameraPose(center=np.array([50.0, 50.0, 50.0]), forward=np.array([0.0, 0.0, 1.0]), up=np.array([0.0, 1.0, 0.0]))
    camera_index = engine.build_bearing_index(camera)
    assert camera_index.indices.shape[0] == 0

    mean, risk = compute_alpha_risk_batched(
        engine, camera_index, np.array([[5.0, 0.0, 0.0]]), angular_tol=ANGULAR_TOL, sigma_rbf=SIGMA,
        max_candidates=100, device="cuda",
    )
    assert mean[0] == 0.0
    assert np.isfinite(risk[0]) and risk[0] >= 0.0


def test_batched_zero_weight_query_falls_back_like_scalar():
    """Every candidate has weight 0 (query point far off every splat's own
    ray, so none are `on_ray` in ray_transmittance_weights) but candidates
    still exist within the bearing ball -- a distinct corner case from
    truly zero candidates, checked separately."""
    engine, camera = build_position_only_engine(seed=30)
    camera_index = engine.build_bearing_index(camera)
    # A point whose own bearing gathers no real candidates within angular_tol
    # of any splat's transmittance ray (large enough offset, small angular_tol).
    point = np.array([[5.0, 3.0, 3.0]])

    scalar_mean, scalar_risk = scalar_alpha_risk(engine, camera_index, point)
    batched_mean, batched_risk = compute_alpha_risk_batched(
        engine, camera_index, point, angular_tol=ANGULAR_TOL, sigma_rbf=SIGMA, max_candidates=500, device="cuda",
    )
    assert np.allclose(batched_mean, scalar_mean, rtol=1e-6, atol=1e-8)
    assert np.allclose(batched_risk, scalar_risk, rtol=1e-6, atol=1e-8)
