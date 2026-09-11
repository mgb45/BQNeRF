"""Focused tests for kernel_family_ablation.py's own noise_variance
extension (see gs_experiment/results/FINDINGS.md's noise-variance section):
that module builds its own Gram matrix from scratch
(`_generalized_rendering_aware_moments`, to support kernel families beyond
RBF), so it needed its own copy of the same additive-diagonal
`noise_variance` extension already validated elsewhere in this project
(gs_experiment/quadrature.py, gs_experiment/hyperparams.py). Matches the
pattern in tests/gs_experiment/test_render_weight.py's own noise_variance
tests: confirm noise_variance=0 is unchanged, and confirm a real
noise_variance measurably changes the result.
"""

from __future__ import annotations

import numpy as np

from gs_experiment.kernel_family_ablation import (
    FAMILIES,
    _ause,
    _generalized_rendering_aware_moments,
    fit_all_families,
    generalized_rendering_aware_variance,
    make_kernels_per_axis,
)
from gs_experiment.kernels import ProductKernel, RBFKernel
from gs_experiment.pixel_uncertainty import LocalUncertaintyEngine


def _toy_engine(seed=0):
    rng = np.random.default_rng(seed)
    n = 20
    positions = rng.uniform(-1.0, 1.0, size=(n, 3))
    # two near-duplicate points with sharply conflicting values -- the
    # classic case the noise extension exists to relax (see
    # gs_experiment/quadrature.py's _rendering_aware_moments docstring).
    positions[0] = [0.0, 0.0, 0.0]
    positions[1] = [1e-4, 0.0, 0.0]
    values = rng.uniform(0.0, 1.0, size=n)
    values[0], values[1] = 1.0, -1.0
    opacities = rng.uniform(0.2, 1.0, size=n)
    return LocalUncertaintyEngine(
        positions=positions,
        values=values,
        pos_kernel=ProductKernel([RBFKernel(sigma=0.3)] * 3),
        scene_bounds=tuple((positions[:, d].min(), positions[:, d].max()) for d in range(3)),
        opacities=opacities,
        max_neighbors=60,
        seed=seed,
    )


def test_noise_variance_zero_matches_default_exactly():
    engine = _toy_engine()
    kernels_per_axis = make_kernels_per_axis("rbf", 0.3)
    query_point = np.array([0.0, 0.0, 0.0])
    radius = 0.5

    default = generalized_rendering_aware_variance(engine, kernels_per_axis, query_point, radius)
    explicit_zero = generalized_rendering_aware_variance(
        engine, kernels_per_axis, query_point, radius, noise_variance=0.0
    )
    assert default.mean == explicit_zero.mean
    assert default.variance == explicit_zero.variance


def test_noise_variance_measurably_changes_result():
    engine = _toy_engine()
    kernels_per_axis = make_kernels_per_axis("rbf", 0.3)
    query_point = np.array([0.0, 0.0, 0.0])
    radius = 0.5

    noiseless = generalized_rendering_aware_variance(engine, kernels_per_axis, query_point, radius, noise_variance=0.0)
    noisy = generalized_rendering_aware_variance(engine, kernels_per_axis, query_point, radius, noise_variance=0.05)

    assert noisy.mean != noiseless.mean
    assert noisy.variance != noiseless.variance
    # a real observation-noise variance relaxes the near-noiseless-forced
    # extreme fit -- posterior variance should increase (less confident),
    # not decrease, matching the ridge-regression-style behavior tested in
    # tests/gs_experiment/test_render_weight.py's own noise_variance tests.
    assert noisy.variance > noiseless.variance


def test_noise_variance_present_in_generalized_moments():
    engine = _toy_engine()
    kernels_per_axis = make_kernels_per_axis("rbf", 0.3)
    query_point = np.array([0.2, 0.1, -0.1])
    radius = 0.5

    mean0, var0, z0_0 = _generalized_rendering_aware_moments(engine, kernels_per_axis, query_point, radius)
    mean1, var1, z0_1 = _generalized_rendering_aware_moments(
        engine, kernels_per_axis, query_point, radius, noise_variance=0.1
    )
    # z0 (prior variance) does not depend on the observation noise model --
    # only the posterior conditioned on kxx changes.
    assert z0_0 == z0_1
    assert var1 != var0


def test_rbf_noise_family_present_and_fits_jointly():
    """FAMILIES gained a 4th 'rbf_noise' entry alongside the existing
    rbf/matern32/rational_quadratic three -- confirms fit_all_families
    returns a real fitted noise_variance for it (and exactly 0.0, the
    unfitted default, for the three noiseless families)."""
    assert "rbf_noise" in FAMILIES
    assert set(FAMILIES) == {"rbf", "matern32", "rational_quadratic", "rbf_noise"}

    rng = np.random.default_rng(1)
    n = 200
    positions = rng.uniform(-1.0, 1.0, size=(n, 3))
    colors = rng.uniform(0.0, 1.0, size=n)
    opacities = np.full(n, 0.5)

    class _FakeScene:
        pass

    scene = _FakeScene()
    scene.positions = positions
    scene.colors = colors
    scene.opacities = opacities

    fitted = fit_all_families(scene, window_radius=0.3, n_windows=10, seed=1)
    assert set(fitted) == {"rbf", "matern32", "rational_quadratic", "rbf_noise"}
    for name in ("rbf", "matern32", "rational_quadratic"):
        assert fitted[name]["noise_variance"] == 0.0
    assert fitted["rbf_noise"]["noise_variance"] >= 0.0
    assert fitted["rbf_noise"]["param"] > 0.0


def test_ause_perfect_predictor_is_near_zero():
    """When `uncertainty` ranks points in exactly the same order as the true
    squared error, the predicted sparsification curve equals the oracle
    curve everywhere, so AUSE should be ~0 (its best-possible value)."""
    rng = np.random.default_rng(2)
    se = rng.uniform(0.01, 1.0, size=100)
    assert abs(_ause(se, se)) < 1e-9


def test_ause_worst_case_predictor_is_positive():
    """Ranking points by the *reverse* of true error (least-confident-first
    predictions on the most-confident-actual points) should score
    measurably worse than 0 -- confirms the metric actually penalizes bad
    ranking, not just returning ~0 regardless of input."""
    rng = np.random.default_rng(3)
    se = rng.uniform(0.01, 1.0, size=100)
    reversed_uncertainty = -se
    assert _ause(reversed_uncertainty, se) > 0.05


def test_ause_nan_below_two_points():
    assert np.isnan(_ause(np.array([1.0]), np.array([1.0])))
    assert np.isnan(_ause(np.array([]), np.array([])))
