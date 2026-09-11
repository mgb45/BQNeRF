"""GPU-only tests for gpu_uncertainty.compute_directional_variance_batched_mixture
-- the mixture-a_q alternative to compute_directional_variance_batched, being
made the production path (see gs_experiment/results/FINDINGS.md's kernel-
consistency-with-alpha-compositing section). Skipped automatically (not
failed) when torch/CUDA aren't available, matching test_gpu_uncertainty.py's
own convention.

As with test_gpu_uncertainty.py, the core claim tested is numerical
agreement with an independent, already-audited scalar reference
(quadrature.bayesian_quadrature_rendering_aware_mixture_directional), not
just "runs and returns something plausible" -- plus one theory-level test
(test_zero_covariance_recovers_alpha_compositing_weights_exactly) that
directly checks this module's own motivating mathematical claim: with
point splats (zero footprint covariance), the BQ weight vector w* must
equal the real alpha-compositing weights w_i = T_i*alpha_i exactly, for
any sigma_rbf.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("gpu_uncertainty needs a real GPU", allow_module_level=True)

from gs_experiment.camera import CameraPose  # noqa: E402
from gs_experiment.gpu_uncertainty import (  # noqa: E402
    compute_directional_alpha_risk_batched_mixture,
    compute_directional_variance_batched_mixture,
)
from gs_experiment.kernels import DirectionalKernel  # noqa: E402
from gs_experiment.pixel_uncertainty import LocalUncertaintyEngine, make_default_3d_position_kernel  # noqa: E402
from gs_experiment.quadrature import (  # noqa: E402
    bayesian_quadrature_rendering_aware_mixture_directional,
    rendering_aware_alternative_weight_risk_mixture_directional,
)

SIGMA = 1.0
KAPPA = 20.0
RADIUS = 3.0
ANGULAR_TOL = 0.2


def build_mixture_engine(n_splats: int = 12, seed: int = 0, with_covariances: bool = True):
    """Small (scalar reference is an explicit O(n^2) double loop, kept fast
    on purpose -- see its own docstring) real-ish scene with real
    scales/rotations, so the mixture path's per-candidate covariance term
    is actually exercised, not silently zero throughout."""
    rng = np.random.default_rng(seed)
    positions = np.stack(
        [rng.uniform(4.0, 6.0, n_splats), rng.uniform(-1.5, 1.5, n_splats), rng.uniform(-1.5, 1.5, n_splats)], axis=1
    )
    values = rng.uniform(0.0, 10.0, n_splats)
    opacities = rng.uniform(0.1, 0.9, n_splats)
    directions = rng.normal(size=(n_splats, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    dir_kernel = DirectionalKernel(kappa=KAPPA)
    camera = CameraPose(center=np.array([-10.0, 0.0, 0.0]), forward=np.array([1.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))
    pos_kernel = make_default_3d_position_kernel(sigma=SIGMA)
    scales = rng.uniform(0.02, 0.08, size=(n_splats, 3)) if with_covariances else None
    rotations = np.tile([1.0, 0.0, 0.0, 0.0], (n_splats, 1)) if with_covariances else None
    engine = LocalUncertaintyEngine(
        positions=positions, values=values, pos_kernel=pos_kernel,
        scene_bounds=((-1.0, 10.0), (-5.0, 5.0), (-5.0, 5.0)),
        opacities=opacities, directions=directions, dir_kernel=dir_kernel,
        scales=scales, rotations=rotations,
    )
    return engine, camera


def scalar_mixture_result(engine, camera_index, point, direction, noise_variance=0.0, values_override=None):
    """Scalar reference for one query point: gathers the exact same real
    candidates/weights the batched path would (engine._along_ray_local_data,
    the same shared machinery rendering_aware_variance_along_ray_directional
    uses), then solves via the new mixture formula directly."""
    idx, local_positions, local_values, _render_weight, weights = engine._along_ray_local_data(
        point, camera_index, RADIUS, None, ANGULAR_TOL, 500, query_direction=direction
    )
    local_directions = engine.directions[idx]
    local_covariances = engine.covariances()[idx] if engine.scales is not None and engine.rotations is not None else None
    values = local_values if values_override is None else np.asarray(values_override)[idx]
    return bayesian_quadrature_rendering_aware_mixture_directional(
        local_positions, local_directions, values, weights, engine.dir_kernel, direction, SIGMA,
        local_covariances=local_covariances, noise_variance=noise_variance,
    ), weights, idx


def test_batched_matches_scalar_with_real_covariances():
    engine, camera = build_mixture_engine()
    camera_index = engine.build_bearing_index(camera)

    rng = np.random.default_rng(1)
    n_queries = 10
    points = np.stack(
        [rng.uniform(4.5, 5.5, n_queries), rng.uniform(-1.0, 1.0, n_queries), rng.uniform(-1.0, 1.0, n_queries)], axis=1
    )
    directions = rng.normal(size=(n_queries, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    batched_var, batched_prior, batched_mean = compute_directional_variance_batched_mixture(
        engine, camera_index, points, directions, angular_tol=ANGULAR_TOL, sigma_rbf=SIGMA, kappa=KAPPA,
        max_candidates=100, device="cuda", return_mean=True,
    )

    for i in range(n_queries):
        scalar_result, _, _ = scalar_mixture_result(engine, camera_index, points[i], directions[i])
        assert batched_var[i] == pytest.approx(scalar_result.variance, rel=1e-6, abs=1e-8)
        assert batched_mean[i] == pytest.approx(scalar_result.mean, rel=1e-6, abs=1e-8)
        assert batched_var[i] <= batched_prior[i] + 1e-8


def test_batched_matches_scalar_with_values_rgb():
    engine, camera = build_mixture_engine(seed=2)
    camera_index = engine.build_bearing_index(camera)
    n_splats = engine.positions.shape[0]
    rng = np.random.default_rng(3)
    values_rgb = rng.uniform(0.0, 1.0, size=(n_splats, 3))

    n_queries = 8
    points = np.stack(
        [rng.uniform(4.5, 5.5, n_queries), rng.uniform(-1.0, 1.0, n_queries), rng.uniform(-1.0, 1.0, n_queries)], axis=1
    )
    directions = rng.normal(size=(n_queries, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    _, _, batched_mean_rgb = compute_directional_variance_batched_mixture(
        engine, camera_index, points, directions, angular_tol=ANGULAR_TOL, sigma_rbf=SIGMA, kappa=KAPPA,
        max_candidates=100, device="cuda", values_rgb=values_rgb,
    )

    for i in range(n_queries):
        for c in range(3):
            scalar_result, _, _ = scalar_mixture_result(
                engine, camera_index, points[i], directions[i], values_override=values_rgb[:, c]
            )
            assert batched_mean_rgb[i, c] == pytest.approx(scalar_result.mean, rel=1e-6, abs=1e-8)


def test_batched_return_mean_and_rgb_default_unaffected():
    engine, camera = build_mixture_engine(seed=4)
    camera_index = engine.build_bearing_index(camera)
    points = np.array([[5.0, 0.0, 0.0]])
    directions = np.array([[1.0, 0.0, 0.0]])

    default_result = compute_directional_variance_batched_mixture(
        engine, camera_index, points, directions, angular_tol=ANGULAR_TOL, sigma_rbf=SIGMA, kappa=KAPPA,
        max_candidates=100, device="cuda",
    )
    assert len(default_result) == 2


def test_batched_noise_variance_changes_result_and_zero_matches_default():
    """Checks monotonic increase across a wide noise_variance range rather
    than a fixed small delta: the directional mixture construction's
    reduction-from-prior term can be genuinely tiny for a given toy
    point/kappa (posterior close to prior even at noise_variance=0 -- not a
    bug, just weak information content at that specific query), so a
    fixed small noise_variance can move the result by less than a fixed
    atol even though the wiring is correct and monotonic."""
    engine, camera = build_mixture_engine(seed=5)
    camera_index = engine.build_bearing_index(camera)
    points = np.array([[5.0, 0.2, -0.1]])
    directions = np.array([[1.0, 0.0, 0.0]])

    default_var, default_prior = compute_directional_variance_batched_mixture(
        engine, camera_index, points, directions, angular_tol=ANGULAR_TOL, sigma_rbf=SIGMA, kappa=KAPPA,
        max_candidates=100, device="cuda",
    )
    explicit_zero_var, _ = compute_directional_variance_batched_mixture(
        engine, camera_index, points, directions, angular_tol=ANGULAR_TOL, sigma_rbf=SIGMA, kappa=KAPPA,
        max_candidates=100, device="cuda", noise_variance=0.0,
    )
    assert default_var == pytest.approx(explicit_zero_var, rel=1e-10)

    variances_by_noise = []
    for nv in (0.0, 1e-3, 1.0, 1e3):
        var, _ = compute_directional_variance_batched_mixture(
            engine, camera_index, points, directions, angular_tol=ANGULAR_TOL, sigma_rbf=SIGMA, kappa=KAPPA,
            max_candidates=100, device="cuda", noise_variance=nv,
        )
        variances_by_noise.append(float(var[0]))
    # monotonically non-decreasing as noise_variance grows (more noise -> posterior
    # relies less on the data -> variance moves toward the prior, never below it)
    assert all(b >= a - 1e-12 for a, b in zip(variances_by_noise, variances_by_noise[1:]))
    # a large enough noise_variance must produce a real, unambiguous change from noiseless
    assert variances_by_noise[-1] > variances_by_noise[0] + 1e-9


def _build_engine_with_kappa(kappa, seed=6, with_covariances=False):
    rng = np.random.default_rng(seed)
    n_splats = 12
    positions = np.stack(
        [rng.uniform(4.0, 6.0, n_splats), rng.uniform(-1.5, 1.5, n_splats), rng.uniform(-1.5, 1.5, n_splats)], axis=1
    )
    values = rng.uniform(0.0, 10.0, n_splats)
    opacities = rng.uniform(0.1, 0.9, n_splats)
    directions = rng.normal(size=(n_splats, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    dir_kernel = DirectionalKernel(kappa=kappa)
    camera = CameraPose(center=np.array([-10.0, 0.0, 0.0]), forward=np.array([1.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))
    pos_kernel = make_default_3d_position_kernel(sigma=SIGMA)
    scales = rng.uniform(0.02, 0.08, size=(n_splats, 3)) if with_covariances else None
    rotations = np.tile([1.0, 0.0, 0.0, 0.0], (n_splats, 1)) if with_covariances else None
    engine = LocalUncertaintyEngine(
        positions=positions, values=values, pos_kernel=pos_kernel,
        scene_bounds=((-1.0, 10.0), (-5.0, 5.0), (-5.0, 5.0)),
        opacities=opacities, directions=directions, dir_kernel=dir_kernel,
        scales=scales, rotations=rotations,
    )
    return engine, camera


def test_zero_covariance_recovers_alpha_compositing_weights_in_position_only_limit():
    """The exact-recovery claim (mixture a_q -> w*=w_alpha for point splats,
    ANY sigma_rbf) holds only for the POSITION part of the kernel -- it does
    NOT survive the directional factor at realistic kappa, and this is a
    real, structural fact, not a numerical-precision detail. Reason: z_i's
    directional term k_dir(d_i, query_direction) evaluates each candidate
    against ONE FIXED external query direction, while Kxx_ij's directional
    term k_dir(d_i,d_j) is a NODE-vs-NODE alignment -- these don't cancel
    the way the position term does (where both z_i and Kxx_ij use the same
    k_pos(x_i,x_j)/k_pos(x_i,query) structure -- position is genuinely
    "integrated over" per this project's own README framing, direction is
    "evaluated at one query," and those play structurally different roles).
    Recovery holds in the kappa->0 limit (direction term degenerates to a
    constant, cancelling out of both z and Kxx identically), confirmed
    directly below; test_directional_kernel_breaks_exact_recovery documents
    the corresponding negative result at realistic kappa, so this
    scope-limitation is a checked fact, not an assumption."""
    engine, camera = _build_engine_with_kappa(kappa=1e-6)
    camera_index = engine.build_bearing_index(camera)

    point = np.array([5.0, 0.1, -0.2])
    direction = np.array([1.0, 0.0, 0.0])
    idx, _, _, _render_weight, weights = engine._along_ray_local_data(
        point, camera_index, RADIUS, None, ANGULAR_TOL, 500, query_direction=direction
    )
    assert len(idx) >= 3  # a real, non-trivial candidate set, not a degenerate 0/1-candidate case

    rng = np.random.default_rng(7)
    for _ in range(5):
        random_values = rng.uniform(-5.0, 5.0, size=len(engine.positions))
        result, w_alpha, idx2 = scalar_mixture_result(engine, camera_index, point, direction, values_override=random_values)
        expected_mean = float(w_alpha @ random_values[idx2])
        # not bit-exact: a tiny residual from rel_jitter's own (deliberate, separate)
        # diagonal regularization of Kxx, which perturbs Kxx slightly away from the
        # exact K_pos used in z's own derivation -- rel_jitter=1e-4 (this function's
        # own default), so a ~1e-3-scale residual, not 1e-9, is the honest tolerance.
        assert result.mean == pytest.approx(expected_mean, rel=2e-3, abs=2e-3)


def test_directional_kernel_breaks_exact_recovery_at_realistic_kappa():
    """Honest negative result, not swept under the rug: at a realistic
    kappa (this project's own KAPPA=20 default elsewhere in this test
    file), the directional along-ray mixture construction does NOT recover
    w_alpha -- the mean can be off by 100x, not a small residual. See this
    module's own docstring and
    test_zero_covariance_recovers_alpha_compositing_weights_in_position_only_limit's
    docstring for the mechanism (z's query-direction-vs-node alignment does
    not match Kxx's node-vs-node alignment once kappa makes that factor
    actually vary). Consequence: making alpha-compositing-consistency hold
    for the real along-ray directional uncertainty construction needs more
    than swapping a_q to a mixture -- that fix is complete only for the
    position-only quadrature."""
    engine, camera = _build_engine_with_kappa(kappa=KAPPA)
    camera_index = engine.build_bearing_index(camera)

    point = np.array([5.0, 0.1, -0.2])
    direction = np.array([1.0, 0.0, 0.0])
    rng = np.random.default_rng(7)
    random_values = rng.uniform(-5.0, 5.0, size=len(engine.positions))
    result, w_alpha, idx2 = scalar_mixture_result(engine, camera_index, point, direction, values_override=random_values)
    expected_mean = float(w_alpha @ random_values[idx2])
    # deliberately a LOOSE, one-directional check: confirms the mismatch is real and
    # large, not that it equals any specific value (that would overfit this seed).
    assert abs(result.mean - expected_mean) > 0.05 * max(abs(expected_mean), 1.0)


def test_alpha_risk_batched_matches_scalar_reference():
    """Cross-validates compute_directional_alpha_risk_batched_mixture (the
    production, real-checkpoint-scale path) against
    quadrature.rendering_aware_alternative_weight_risk_mixture_directional
    (the O(K^2) scalar reference) -- same standard this file already
    applies to the BQ-optimal (solve-based) mixture path."""
    engine, camera = build_mixture_engine(seed=8)
    camera_index = engine.build_bearing_index(camera)

    rng = np.random.default_rng(9)
    n_queries = 10
    points = np.stack(
        [rng.uniform(4.5, 5.5, n_queries), rng.uniform(-1.0, 1.0, n_queries), rng.uniform(-1.0, 1.0, n_queries)], axis=1
    )
    directions = rng.normal(size=(n_queries, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    batched_mean, batched_risk = compute_directional_alpha_risk_batched_mixture(
        engine, camera_index, points, directions, angular_tol=ANGULAR_TOL, sigma_rbf=SIGMA, kappa=KAPPA,
        max_candidates=100, device="cuda",
    )

    for i in range(n_queries):
        idx, local_positions, local_values, _render_weight, weights = engine._along_ray_local_data(
            points[i], camera_index, RADIUS, None, ANGULAR_TOL, 500, query_direction=directions[i]
        )
        local_directions = engine.directions[idx]
        local_covariances = engine.covariances()[idx]
        scalar_mean, scalar_risk = rendering_aware_alternative_weight_risk_mixture_directional(
            local_positions, local_directions, local_values, weights, engine.dir_kernel, directions[i], SIGMA,
            local_covariances=local_covariances,
        )
        assert batched_mean[i] == pytest.approx(scalar_mean, rel=1e-6, abs=1e-8)
        assert batched_risk[i] == pytest.approx(scalar_risk, rel=1e-6, abs=1e-8)


def test_alpha_risk_batched_noise_variance_wiring():
    engine, camera = build_mixture_engine(seed=10)
    camera_index = engine.build_bearing_index(camera)
    points = np.array([[5.0, 0.1, 0.1]])
    directions = np.array([[1.0, 0.0, 0.0]])

    _, risk_noiseless = compute_directional_alpha_risk_batched_mixture(
        engine, camera_index, points, directions, angular_tol=ANGULAR_TOL, sigma_rbf=SIGMA, kappa=KAPPA,
        max_candidates=100, device="cuda", noise_variance=0.0,
    )
    _, risk_noisy = compute_directional_alpha_risk_batched_mixture(
        engine, camera_index, points, directions, angular_tol=ANGULAR_TOL, sigma_rbf=SIGMA, kappa=KAPPA,
        max_candidates=100, device="cuda", noise_variance=10.0,
    )
    assert risk_noisy[0] > risk_noiseless[0]


def test_alpha_risk_batched_handles_empty_camera_index():
    positions = np.array([[5.0, 0.0, 0.0]])
    values = np.array([1.0])
    opacities = np.array([0.9])
    directions = np.array([[1.0, 0.0, 0.0]])
    dir_kernel = DirectionalKernel(kappa=KAPPA)
    pos_kernel = make_default_3d_position_kernel(sigma=SIGMA)
    engine = LocalUncertaintyEngine(
        positions=positions, values=values, pos_kernel=pos_kernel, scene_bounds=((-1.0, 10.0), (-5.0, 5.0), (-5.0, 5.0)),
        opacities=opacities, directions=directions, dir_kernel=dir_kernel,
    )
    camera = CameraPose(center=np.array([50.0, 50.0, 50.0]), forward=np.array([0.0, 0.0, 1.0]), up=np.array([0.0, 1.0, 0.0]))
    camera_index = engine.build_bearing_index(camera)
    assert camera_index.indices.shape[0] == 0

    mean, risk = compute_directional_alpha_risk_batched_mixture(
        engine, camera_index, np.array([[5.0, 0.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]),
        angular_tol=ANGULAR_TOL, sigma_rbf=SIGMA, kappa=KAPPA, max_candidates=100, device="cuda",
    )
    assert mean[0] == 0.0
    assert np.isfinite(risk[0]) and risk[0] >= 0.0


def test_batched_handles_empty_camera_index():
    """Mirrors test_gpu_uncertainty.py's own empty-camera-index test: zero
    real candidates falls back to compute_directional_variance_batched's
    shared _zero_candidate_variance convention (a near-degenerate small-
    amplitude self-variance), the same fallback the single-Gaussian
    production path uses -- the mixture formulation has no fallback
    concept of its own for "nothing nearby," so it reuses that one."""
    positions = np.array([[5.0, 0.0, 0.0]])
    values = np.array([1.0])
    opacities = np.array([0.9])
    directions = np.array([[1.0, 0.0, 0.0]])
    dir_kernel = DirectionalKernel(kappa=KAPPA)
    pos_kernel = make_default_3d_position_kernel(sigma=SIGMA)
    engine = LocalUncertaintyEngine(
        positions=positions, values=values, pos_kernel=pos_kernel, scene_bounds=((-1.0, 10.0), (-5.0, 5.0), (-5.0, 5.0)),
        opacities=opacities, directions=directions, dir_kernel=dir_kernel,
    )
    camera = CameraPose(center=np.array([50.0, 50.0, 50.0]), forward=np.array([0.0, 0.0, 1.0]), up=np.array([0.0, 1.0, 0.0]))
    camera_index = engine.build_bearing_index(camera)
    assert camera_index.indices.shape[0] == 0

    result, prior = compute_directional_variance_batched_mixture(
        engine, camera_index, np.array([[5.0, 0.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]),
        angular_tol=ANGULAR_TOL, sigma_rbf=SIGMA, kappa=KAPPA, max_candidates=100, device="cuda",
    )
    assert np.all(np.isfinite(result)) and np.all(result >= 0.0)
    assert np.allclose(result, prior, rtol=1e-6, atol=1e-8)
