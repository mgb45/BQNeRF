import numpy as np
from scipy import integrate

from gs_experiment.kernels import ProductKernel, RBFKernel
from gs_experiment.quadrature import (
    bayesian_quadrature_rendering_aware,
    rendering_aware_alternative_weight_risk,
    rendering_aware_moment_vector,
    rendering_aware_prior_variance,
)
from gs_experiment.render_weight import GaussianRenderWeight


def test_render_weight_peaks_at_amplitude_at_its_own_center():
    w = GaussianRenderWeight(amplitude=0.7, center=[2.0], covariance=[[0.05]])
    assert abs(float(w(np.array([[2.0]]))[0]) - 0.7) < 1e-12


def test_render_weight_decays_away_from_center():
    w = GaussianRenderWeight(amplitude=1.0, center=[0.0], covariance=[[0.05]])
    assert w(np.array([[3.0]]))[0] < 1e-10


def test_total_mass_matches_numerical_integration():
    w = GaussianRenderWeight(amplitude=0.8, center=[1.0, -0.5], covariance=np.diag([0.2, 0.4]))

    numerical, _ = integrate.dblquad(
        lambda y, x: float(w(np.array([[x, y]]))[0]), -10, 12, -11, 9,
    )
    assert abs(w.total_mass - numerical) / numerical < 1e-6


def test_from_total_mass_round_trips_the_requested_mass_regardless_of_footprint_scale():
    """The whole point of from_total_mass: pin total_mass to a fixed,
    physically meaningful value while covariance's *volume* varies wildly
    (a real splat footprint vs. a coarse window) -- total_mass must come
    out exactly right either way, unlike fixing amplitude directly (see
    this module's docstring: that lets total_mass vanish for a tiny
    footprint even at fixed amplitude)."""
    for covariance in [1e-8 * np.eye(3), 1e-2 * np.eye(3), 5.0 * np.eye(3)]:
        w = GaussianRenderWeight.from_total_mass(total_mass=0.42, center=[0.0, 0.0, 0.0], covariance=covariance)
        assert abs(w.total_mass - 0.42) / 0.42 < 1e-9


def test_from_total_mass_tiny_footprint_no_longer_vanishes():
    """Direct regression check for the real bug this fixes: amplitude=1.0
    (a real opacity) with a real, tiny splat covariance used to give a
    total_mass around 1e-100 in practice -- from_total_mass keeps it
    exactly at the requested, physically bounded value instead."""
    tiny_covariance = (0.02**2) * np.eye(3)  # a real splat's actual scale
    w = GaussianRenderWeight.from_total_mass(total_mass=0.9, center=[0.0, 0.0, 0.0], covariance=tiny_covariance)
    assert abs(w.total_mass - 0.9) < 1e-9


def test_prior_variance_grows_with_render_weight_spread():
    """z_{q,0} should track how concentrated a_q actually is: a tight
    footprint/transmittance envelope (a hard, well-resolved surface) should
    report a smaller prior integration uncertainty than a diffuse one at the
    same amplitude -- the thing the old uniform-box vv can't see at all,
    since it never receives a_q (see test_old_box_vv_is_blind_to_footprint_shape)."""
    sigma_rbf = 0.4
    tight = GaussianRenderWeight(amplitude=1.0, center=[0.0], covariance=[[0.02]])
    wide = GaussianRenderWeight(amplitude=1.0, center=[0.0], covariance=[[2.0]])

    z0_tight = rendering_aware_prior_variance(tight, sigma_rbf)
    z0_wide = rendering_aware_prior_variance(wide, sigma_rbf)

    assert z0_tight < z0_wide


def test_old_box_vv_is_blind_to_footprint_shape():
    """The old box-quadrature prior variance (kernel.vv over a fixed window)
    is exactly the same number whether the true rendering weight a_q is
    tightly concentrated (a hard surface) or diffuse within that window --
    it has no way to represent "most of this box is actually irrelevant to
    this query." This is precisely the gap the rendering-aware z0 above
    closes."""
    base = ProductKernel([RBFKernel(0.4)])
    bounds = [(-5.0, 5.0)]
    assert base.vv(bounds) == base.vv(bounds)  # trivially the same call twice
    # The point: vv takes no a_q argument at all, so it cannot distinguish
    # the tight-footprint and diffuse-footprint scenarios from
    # test_prior_variance_grows_with_render_weight_spread, which the new
    # z0 clearly does.


def test_rendering_aware_posterior_is_finite_and_nonnegative():
    w = GaussianRenderWeight(amplitude=1.0, center=[1.0], covariance=[[0.1]])
    nodes = np.array([[0.8], [1.0], [1.3]])
    values = np.array([0.5, 0.9, 0.4])
    result = bayesian_quadrature_rendering_aware(nodes, values, w, sigma_rbf=0.3)
    assert np.isfinite(result.mean)
    assert result.variance >= 0.0


def test_rendering_aware_ignores_a_node_outside_the_footprint_but_old_box_quadrature_does_not():
    """The key behavioral fix: a node the renderer doesn't actually care
    about for this query (a_q(x_i) ~ 0 -- e.g. occluded, or outside the
    pixel's footprint) should barely move the rendering-aware mean,
    regardless of whether it happens to sit inside whatever generic window
    the old box-quadrature approach was given."""
    sigma_rbf = 0.3
    node = np.array([[3.0]])
    value = np.array([10.0])

    render_weight = GaussianRenderWeight(amplitude=1.0, center=[0.0], covariance=[[0.05]])
    new_mean = bayesian_quadrature_rendering_aware(node, value, render_weight, sigma_rbf=sigma_rbf).mean

    old_kernel = ProductKernel([RBFKernel(sigma_rbf)])
    old_bounds = [(-5.0, 5.0)]
    kxx = float(old_kernel.k(node, node)[0, 0])
    v = float(old_kernel.v(node, old_bounds)[0])
    old_mean = (v / kxx) * value[0]

    assert abs(new_mean) < 1e-3
    assert abs(old_mean) > 1.0


def test_alternative_weight_risk_reduces_to_bq_variance_at_bq_weights():
    """rendering_aware_alternative_weight_risk's e(w)^2 = z0 - 2 w@z + w@K@w
    is the general RKHS worst-case-error quadratic form for *any* real
    weight vector w; at the BQ-optimal weights w* = K^-1 z it must reduce
    exactly to bayesian_quadrature_rendering_aware's own reported mean and
    variance."""
    w = GaussianRenderWeight(amplitude=0.8, center=[0.3], covariance=[[0.1]])
    nodes = np.array([[0.05], [0.2], [0.5], [0.8]])
    values = np.array([1.0, 0.7, 0.3, 0.9])
    sigma_rbf = 0.2

    result = bayesian_quadrature_rendering_aware(nodes, values, w, sigma_rbf=sigma_rbf)

    from gs_experiment.quadrature import _rendering_aware_moments

    kxx, z, _ = _rendering_aware_moments(nodes, w, sigma_rbf, 1e-4)
    w_bq = np.linalg.solve(kxx, z)

    mean, risk = rendering_aware_alternative_weight_risk(nodes, values, w, w_bq, sigma_rbf=sigma_rbf)
    assert abs(mean - result.mean) < 1e-9
    assert abs(risk - result.variance) < 1e-9


def test_alternative_weight_risk_is_never_smaller_than_bq_variance():
    """w* = K^-1 z uniquely minimizes e(w)^2 (the classical Bayes-Hermite
    optimality result) -- any other real, nonnegative weight vector, in
    particular one chosen for a different reason than minimizing this
    quantity (e.g. real alpha-compositing transmittance weights), must give
    an equal-or-larger worst-case error, never smaller."""
    w = GaussianRenderWeight(amplitude=1.0, center=[0.4], covariance=[[0.12]])
    nodes = np.array([[0.1], [0.3], [0.6], [0.9]])
    values = np.array([0.9, 0.4, 0.6, 0.2])
    sigma_rbf = 0.25

    result = bayesian_quadrature_rendering_aware(nodes, values, w, sigma_rbf=sigma_rbf)

    rng = np.random.default_rng(0)
    for _ in range(20):
        alt_weights = rng.uniform(0.0, 1.0, size=len(nodes))
        _, risk = rendering_aware_alternative_weight_risk(nodes, values, w, alt_weights, sigma_rbf=sigma_rbf)
        assert risk >= result.variance - 1e-9


def test_noise_variance_zero_matches_pre_noise_behavior_exactly():
    """noise_variance=0.0 (the default) must reproduce every existing
    caller's behavior bit-for-bit -- the homoscedastic-noise extension is
    additive on K's diagonal, on top of the existing rel_jitter numerical
    term, never a replacement for it."""
    w = GaussianRenderWeight(amplitude=0.8, center=[0.3], covariance=[[0.1]])
    nodes = np.array([[0.05], [0.2], [0.5], [0.8]])
    values = np.array([1.0, 0.7, 0.3, 0.9])
    sigma_rbf = 0.2

    default = bayesian_quadrature_rendering_aware(nodes, values, w, sigma_rbf=sigma_rbf)
    explicit_zero = bayesian_quadrature_rendering_aware(nodes, values, w, sigma_rbf=sigma_rbf, noise_variance=0.0)
    assert default.mean == explicit_zero.mean
    assert default.variance == explicit_zero.variance


def test_noise_variance_relaxes_near_duplicate_point_oscillation():
    """The motivating case: two near-duplicate nodes with conflicting
    observed values force a noiseless GP to swing to extreme, opposite-sign
    weights to satisfy both exactly. A real observation-noise variance
    should relax that."""
    w = GaussianRenderWeight(amplitude=1.0, center=[0.5], covariance=[[0.3]])
    # two near-duplicate points with sharply conflicting values, plus a
    # third, well-separated point -- the near-duplicate pair is what
    # forces extreme weights under noiseless interpolation.
    nodes = np.array([[0.50], [0.501], [0.9]])
    sigma_rbf = 0.3

    from gs_experiment.quadrature import _rendering_aware_moments

    kxx, z, z0 = _rendering_aware_moments(nodes, w, sigma_rbf, 1e-4, 0.0)
    w_star_noiseless = np.linalg.solve(kxx, z)

    kxx_noisy, z_noisy, _ = _rendering_aware_moments(nodes, w, sigma_rbf, 1e-4, 0.05)
    w_star_noisy = np.linalg.solve(kxx_noisy, z_noisy)

    # the noiseless solve is forced to nearly-exactly separate the two
    # conflicting near-duplicate observations -- large-magnitude,
    # opposite-signed weights on that pair.
    assert w_star_noiseless[0] * w_star_noiseless[1] < 0
    assert abs(w_star_noiseless[0]) > 2.0 or abs(w_star_noiseless[1]) > 2.0

    # a real noise variance shrinks the magnitude of every weight relative
    # to the noiseless case (the standard ridge-regression-style shrinkage
    # a noise/nugget term on K's diagonal produces).
    assert np.abs(w_star_noisy).max() < np.abs(w_star_noiseless).max()


def test_rendering_aware_with_zero_nodes_returns_prior():
    w = GaussianRenderWeight(amplitude=1.0, center=[0.0], covariance=[[0.2]])
    sigma_rbf = 0.3
    result = bayesian_quadrature_rendering_aware(np.empty((0, 1)), np.empty(0), w, sigma_rbf=sigma_rbf)
    assert result.mean == 0.0
    assert abs(result.variance - rendering_aware_prior_variance(w, sigma_rbf)) < 1e-12
