"""CPU-only tests for quadrature.rendering_aware_alternative_weight_risk_mixture_directional
-- the mixture-a_q, joint-position+direction-kernel analogue of
rendering_aware_alternative_weight_risk (see that function's own tests in
test_render_weight.py for the established pattern this mirrors).
"""

import numpy as np

import pytest

from gs_experiment.kernels import DirectionalKernel
from gs_experiment.quadrature import (
    _mixture_directional_moments,
    bayesian_quadrature_rendering_aware_mixture,
    bayesian_quadrature_rendering_aware_mixture_directional,
    rendering_aware_alternative_weight_risk_mixture,
    rendering_aware_alternative_weight_risk_mixture_directional,
)


def _toy_mixture_scene(seed=0, n=6, with_covariances=True):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-1.0, 1.0, size=(n, 3))
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    values = rng.uniform(0.0, 1.0, size=n)
    weights = rng.uniform(0.05, 0.5, size=n)  # stand-in for real w_i=T_i*alpha_i, any positive weights
    covariances = rng.uniform(0.001, 0.01, size=(n, 3, 3))
    covariances = np.einsum("nij,nkj->nik", covariances, covariances)  # make PSD
    covariances += 1e-4 * np.eye(3)
    if not with_covariances:
        covariances = np.zeros((n, 3, 3))
    query_direction = rng.normal(size=3)
    query_direction /= np.linalg.norm(query_direction)
    return positions, directions, values, weights, covariances, query_direction


def test_risk_reduces_to_bq_variance_and_mean_at_bq_optimal_weights():
    """The general RKHS worst-case-error quadratic form, evaluated at the
    BQ-optimal w* = Kxx^-1 z, must reduce exactly to
    bayesian_quadrature_rendering_aware_mixture_directional's own reported
    mean/variance -- the same correctness check
    test_render_weight.py::test_alternative_weight_risk_reduces_to_bq_variance_at_bq_weights
    already established for the single-Gaussian, position-only case,
    applied here to the mixture+directional construction."""
    positions, directions, values, weights, covariances, query_direction = _toy_mixture_scene(seed=1)
    dir_kernel = DirectionalKernel(kappa=5.0)
    sigma_rbf = 0.3

    bq_result = bayesian_quadrature_rendering_aware_mixture_directional(
        positions, directions, values, weights, dir_kernel, query_direction, sigma_rbf, local_covariances=covariances,
    )

    z0, z, kxx = _mixture_directional_moments(
        positions, directions, weights, dir_kernel, query_direction, sigma_rbf, covariances,
    )
    w_bq = np.linalg.solve(kxx, z)

    mean, risk = rendering_aware_alternative_weight_risk_mixture_directional(
        positions, directions, values, weights, dir_kernel, query_direction, sigma_rbf, w=w_bq,
        local_covariances=covariances,
    )
    assert abs(mean - bq_result.mean) < 1e-9
    assert abs(risk - bq_result.variance) < 1e-9


def test_risk_never_smaller_than_bq_variance_at_other_weights():
    """w* uniquely minimizes e(w)^2 (classical Bayes-Hermite optimality) --
    any other real weight vector, in particular the real alpha-compositing
    stand-in `weights` used directly (not solved for), must give a risk
    that's >= the BQ-optimal variance, never smaller."""
    positions, directions, values, weights, covariances, query_direction = _toy_mixture_scene(seed=2)
    dir_kernel = DirectionalKernel(kappa=5.0)
    sigma_rbf = 0.3

    bq_result = bayesian_quadrature_rendering_aware_mixture_directional(
        positions, directions, values, weights, dir_kernel, query_direction, sigma_rbf, local_covariances=covariances,
    )
    _mean, risk_at_weights = rendering_aware_alternative_weight_risk_mixture_directional(
        positions, directions, values, weights, dir_kernel, query_direction, sigma_rbf, local_covariances=covariances,
    )
    assert risk_at_weights >= bq_result.variance - 1e-9


def test_mean_is_plain_dot_product_regardless_of_kernel_geometry():
    """mean = weights @ values always -- it's the real estimator's own
    predicted value, computed directly, and must not depend on sigma_rbf,
    kappa, or local_covariances at all (only `risk` does)."""
    positions, directions, values, weights, covariances, query_direction = _toy_mixture_scene(seed=3)
    expected_mean = float(weights @ values)

    for kappa in (0.1, 5.0, 50.0):
        for sigma_rbf in (0.1, 1.0):
            dir_kernel = DirectionalKernel(kappa=kappa)
            mean, risk = rendering_aware_alternative_weight_risk_mixture_directional(
                positions, directions, values, weights, dir_kernel, query_direction, sigma_rbf,
                local_covariances=covariances,
            )
            assert mean == expected_mean
            assert risk >= 0.0


def test_w_defaults_to_weights():
    """The common, actual production use case: `w` omitted means score the
    same weights a_q was built from (real alpha compositing's own weights)
    -- must be bit-identical to passing w=weights explicitly."""
    positions, directions, values, weights, covariances, query_direction = _toy_mixture_scene(seed=4)
    dir_kernel = DirectionalKernel(kappa=5.0)
    sigma_rbf = 0.3

    default_mean, default_risk = rendering_aware_alternative_weight_risk_mixture_directional(
        positions, directions, values, weights, dir_kernel, query_direction, sigma_rbf, local_covariances=covariances,
    )
    explicit_mean, explicit_risk = rendering_aware_alternative_weight_risk_mixture_directional(
        positions, directions, values, weights, dir_kernel, query_direction, sigma_rbf, w=weights,
        local_covariances=covariances,
    )
    assert default_mean == explicit_mean
    assert default_risk == explicit_risk


def test_zero_candidates_returns_zero_mean_and_zero_risk():
    dir_kernel = DirectionalKernel(kappa=5.0)
    mean, risk = rendering_aware_alternative_weight_risk_mixture_directional(
        np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros(0), dir_kernel, np.array([1.0, 0.0, 0.0]), 0.3,
    )
    assert mean == 0.0
    assert risk == 0.0


def test_position_only_recovers_alpha_compositing_weights_exactly():
    """The clean, position-only special case: without any directional
    kernel at all AND with point splats (zero footprint covariance),
    w*=weights exactly (up to rel_jitter's own tiny regularization) --
    risk(w=weights) reduces to the BQ-optimal variance directly, no
    bias-correction term needed. (Real, non-zero covariances break EXACT
    equality even in the position-only case -- see the companion test
    right below for how close it gets with real footprints -- this test
    isolates the clean, exact special case.) Contrast with
    test_gpu_uncertainty_mixture.py's
    test_directional_kernel_breaks_exact_recovery_at_realistic_kappa,
    which fails badly (not just approximately) once direction is added,
    even at zero covariance."""
    positions, _directions, values, weights, _covariances, _query_direction = _toy_mixture_scene(
        seed=5, with_covariances=False
    )
    sigma_rbf = 0.3

    bq_result = bayesian_quadrature_rendering_aware_mixture(positions, values, weights, sigma_rbf)
    mean, risk = rendering_aware_alternative_weight_risk_mixture(positions, values, weights, sigma_rbf)
    # tolerance matches rel_jitter's own scale (1e-4 default), not machine precision --
    # the jitter term is a real, separate regularization of Kxx that this construction
    # doesn't (and shouldn't) cancel out; see this test's own docstring.
    assert mean == pytest.approx(bq_result.mean, rel=2e-3)
    assert risk == pytest.approx(bq_result.variance, rel=2e-3, abs=1e-6)


def test_position_only_with_real_covariances_is_close_but_not_exact():
    """With real (non-degenerate) footprint covariances, position-only
    recovery is a controlled approximation, not exact -- confirms this
    isn't silently degrading to the directional case's ~100x-off failure:
    the mismatch here should be small (same order as the mixture-
    directional kappa->0 limit already measured, not the large kappa>0
    one)."""
    positions, _directions, values, weights, covariances, _query_direction = _toy_mixture_scene(seed=5)
    sigma_rbf = 0.3

    bq_result = bayesian_quadrature_rendering_aware_mixture(
        positions, values, weights, sigma_rbf, local_covariances=covariances,
    )
    mean, risk = rendering_aware_alternative_weight_risk_mixture(
        positions, values, weights, sigma_rbf, local_covariances=covariances,
    )
    assert mean == pytest.approx(bq_result.mean, rel=0.1)
    assert risk >= bq_result.variance - 1e-9  # risk(weights) is never below the BQ-optimal variance


def test_position_only_risk_reduces_to_bq_variance_at_bq_optimal_weights():
    positions, _directions, values, weights, covariances, _query_direction = _toy_mixture_scene(seed=6)
    sigma_rbf = 0.3

    bq_result = bayesian_quadrature_rendering_aware_mixture(
        positions, values, weights, sigma_rbf, local_covariances=covariances,
    )
    from gs_experiment.quadrature import _mixture_moments

    z0, z, kxx = _mixture_moments(positions, weights, sigma_rbf, covariances)
    w_bq = np.linalg.solve(kxx, z)
    mean, risk = rendering_aware_alternative_weight_risk_mixture(
        positions, values, weights, sigma_rbf, w=w_bq, local_covariances=covariances,
    )
    assert abs(mean - bq_result.mean) < 1e-9
    assert abs(risk - bq_result.variance) < 1e-9
