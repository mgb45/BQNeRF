"""CPU-only tests for sh_directional_uncertainty.py -- the per-splat SH-
coefficient Bayesian linear regression that gives the directional term of
the renderer-consistent sparse-GP decomposition (see that module's own
docstring)."""

import numpy as np
import pytest

from gs_experiment.sh_directional_uncertainty import (
    directional_variance,
    invert_precision,
    per_splat_coefficient_precision,
    propagate_pixel_directional_uncertainty,
    sh_basis,
)
from gs_experiment.spherical_harmonics import N_COEFFS_FOR_DEGREE, eval_sh


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_sh_basis_matches_eval_sh_exactly(degree):
    rng = np.random.default_rng(degree)
    n, n_channels = 15, 3
    n_coeffs = N_COEFFS_FOR_DEGREE[degree]
    coeffs = rng.normal(size=(n, n_channels, n_coeffs))
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    expected = eval_sh(coeffs, directions, degree)  # (n, n_channels)
    phi = sh_basis(directions, degree)  # (n, n_coeffs)
    actual = np.einsum("nc,ndc->nd", phi, coeffs) + 0.5

    assert np.allclose(actual, expected, atol=1e-12)


def test_sh_basis_is_unit_norm_direction_independent_at_degree_zero():
    directions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    phi = sh_basis(directions, degree=0)
    assert phi.shape == (3, 1)
    assert np.allclose(phi[:, 0], phi[0, 0])


def test_precision_matches_naive_bayesian_linear_regression():
    """Direct check against np.linalg.inv(lam*I + Phi^T diag(beta^2) Phi)
    for a toy case with abstract feature vectors (isolating the pure
    linear-algebra identity from the SH-specific plumbing)."""
    rng = np.random.default_rng(0)
    n_coeffs = 4
    phi = rng.normal(size=(6, n_coeffs))
    beta = rng.uniform(0.1, 1.0, size=6)
    lam = 2.0

    precision = per_splat_coefficient_precision([phi], [beta], lam=lam, n_coeffs=n_coeffs)
    expected = lam * np.eye(n_coeffs) + phi.T @ np.diag(beta**2) @ phi
    assert np.allclose(precision[0], expected, atol=1e-10)


def test_zero_observations_falls_back_to_prior_precision():
    n_coeffs = 4
    lam = 3.0
    precision = per_splat_coefficient_precision([np.zeros((0, n_coeffs))], [np.zeros(0)], lam=lam, n_coeffs=n_coeffs)
    assert np.allclose(precision[0], lam * np.eye(n_coeffs))
    sigma = invert_precision(precision)
    assert np.allclose(sigma[0], np.eye(n_coeffs) / lam)


def test_single_observation_reduces_variance_only_along_its_own_feature_direction():
    """Sherman-Morrison sanity check: with exactly one observation of
    weight 1 at feature vector e0 = [1, 0], variance along e0 itself must
    strictly decrease from the prior, while variance along the ORTHOGONAL
    feature direction e1 = [0, 1] must be untouched -- this observation
    carries zero information about that orthogonal mode."""
    lam = 1.0
    phi = np.array([[1.0, 0.0]])
    beta = np.array([1.0])
    precision = per_splat_coefficient_precision([phi], [beta], lam=lam, n_coeffs=2)
    sigma = invert_precision(precision)[0]

    e0, e1 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    prior_var_e0 = (e0 @ (np.eye(2) / lam) @ e0)
    prior_var_e1 = (e1 @ (np.eye(2) / lam) @ e1)
    posterior_var_e0 = e0 @ sigma @ e0
    posterior_var_e1 = e1 @ sigma @ e1

    assert posterior_var_e0 < prior_var_e0
    assert posterior_var_e1 == pytest.approx(prior_var_e1, abs=1e-12)


def test_directional_variance_matches_manual_quadratic_form():
    rng = np.random.default_rng(1)
    degree = 2
    n_coeffs = N_COEFFS_FOR_DEGREE[degree]
    sigma_theta = rng.normal(size=(n_coeffs, n_coeffs))
    sigma_theta = sigma_theta @ sigma_theta.T  # PSD
    direction = np.array([0.3, -0.5, 0.8])
    direction /= np.linalg.norm(direction)

    phi = sh_basis(direction, degree)
    expected = float(phi @ sigma_theta @ phi)
    actual = directional_variance(sigma_theta, direction, degree)
    assert actual == pytest.approx(expected, rel=1e-10)


def test_directional_variance_batches_over_many_splats_and_directions():
    rng = np.random.default_rng(2)
    degree = 1
    n_coeffs = N_COEFFS_FOR_DEGREE[degree]
    n = 5
    sigma_theta = rng.normal(size=(n, n_coeffs, n_coeffs))
    sigma_theta = np.einsum("nij,nkj->nik", sigma_theta, sigma_theta)
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    s2 = directional_variance(sigma_theta, directions, degree)
    assert s2.shape == (n,)
    for i in range(n):
        expected = directional_variance(sigma_theta[i], directions[i], degree)
        assert s2[i] == pytest.approx(expected, rel=1e-10)


def test_propagate_pixel_directional_uncertainty_is_weighted_sum_of_squares():
    alpha_weights = np.array([0.5, 0.2, 0.1])
    s2 = np.array([1.0, 4.0, 9.0])
    expected = 0.25 * 1.0 + 0.04 * 4.0 + 0.01 * 9.0
    assert propagate_pixel_directional_uncertainty(alpha_weights, s2) == pytest.approx(expected)
