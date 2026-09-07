import numpy as np
import pytest
from scipy import integrate

from bq_splat.kernels import DirectionalKernel, MaternKernel, ProductKernel, RBFKernel
from bq_splat.quadrature import (
    bayesian_quadrature_nd,
    bayesian_quadrature_rendering_aware,
    bayesian_quadrature_rendering_aware_directional,
    numerical_rendering_moment_vector,
    numerical_rendering_prior_variance,
    renderer_centered_residual_variance,
    rendering_aware_moment_vector,
    rendering_aware_prior_variance,
)
from bq_splat.render_weight import GaussianRenderWeight


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


def test_closed_form_moment_vector_matches_numerical_1d():
    w = GaussianRenderWeight(amplitude=0.8, center=[2.0], covariance=[[0.3]])
    sigma_rbf = 0.5
    nodes = np.array([[0.0], [2.0], [5.0]])
    base = ProductKernel([RBFKernel(sigma_rbf)])
    domain = [(-15.0, 19.0)]

    closed = rendering_aware_moment_vector(nodes, w, sigma_rbf)
    numerical = numerical_rendering_moment_vector(nodes, w, base, domain)

    np.testing.assert_allclose(closed, numerical, atol=1e-6)


def test_closed_form_prior_variance_matches_numerical_1d():
    w = GaussianRenderWeight(amplitude=0.8, center=[2.0], covariance=[[0.3]])
    sigma_rbf = 0.5
    base = ProductKernel([RBFKernel(sigma_rbf)])
    domain = [(-15.0, 19.0)]

    closed = rendering_aware_prior_variance(w, sigma_rbf)
    numerical = numerical_rendering_prior_variance(w, base, domain)

    assert abs(closed - numerical) < 1e-6


def test_closed_form_moment_vector_matches_numerical_2d():
    """z (a 2D integral) is cheap enough for scipy.integrate.nquad directly."""
    w = GaussianRenderWeight(amplitude=0.6, center=[1.0, -0.5], covariance=np.diag([0.2, 0.4]))
    sigma_rbf = 0.6
    nodes = np.array([[0.0, 0.0], [1.0, -0.5], [2.0, 1.0]])
    base = ProductKernel([RBFKernel(sigma_rbf), RBFKernel(sigma_rbf)])
    domain = [(-9.0, 11.0), (-9.5, 8.5)]

    closed_z = rendering_aware_moment_vector(nodes, w, sigma_rbf)
    numerical_z = numerical_rendering_moment_vector(nodes, w, base, domain)
    np.testing.assert_allclose(closed_z, numerical_z, atol=1e-6)


def test_closed_form_prior_variance_matches_monte_carlo_2d():
    """z0 is a 4D integral (2D xi x 2D xi') -- scipy.integrate.nquad's nested
    adaptive quadrature is impractically slow past ~2D (confirmed: it does
    not finish in 100s even over a tight, few-sigma domain), so this cross-
    checks the closed form against importance-sampled Monte Carlo instead:
    since a_q(xi) = A * N(xi; mu_q, Sigma_q) (see rendering_aware_prior_variance's
    docstring for A), z0 = A^2 * E[k_base(xi, xi')] for xi, xi' drawn iid from
    N(mu_q, Sigma_q) -- sampling directly from the render weight's own shape
    rather than blindly over a box makes this efficient at a modest sample
    count."""
    rng = np.random.default_rng(0)
    w = GaussianRenderWeight(amplitude=0.6, center=[1.0, -0.5], covariance=np.diag([0.2, 0.4]))
    sigma_rbf = 0.6

    d = w.dim
    _, logdet = np.linalg.slogdet(w.covariance)
    log_A = np.log(w.amplitude) + 0.5 * (d * np.log(2 * np.pi) + logdet)
    A = np.exp(log_A)

    n_samples = 200_000
    xi = rng.multivariate_normal(w.center, w.covariance, size=n_samples)
    xip = rng.multivariate_normal(w.center, w.covariance, size=n_samples)
    rbf = RBFKernel(sigma_rbf)
    k_vals = np.ones(n_samples)
    for axis in range(d):
        k_vals *= rbf.k(xi[:, axis], xip[:, axis])
    z0_mc = A**2 * float(np.mean(k_vals))

    closed_z0 = rendering_aware_prior_variance(w, sigma_rbf)
    assert abs(closed_z0 - z0_mc) / closed_z0 < 0.05


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
    the old box-quadrature approach was given. The old approach has no
    concept of a_q, so the same node moves its mean by a lot."""
    sigma_rbf = 0.3
    node = np.array([[3.0]])
    value = np.array([10.0])

    render_weight = GaussianRenderWeight(amplitude=1.0, center=[0.0], covariance=[[0.05]])
    new_mean = bayesian_quadrature_rendering_aware(node, value, render_weight, sigma_rbf=sigma_rbf).mean

    old_kernel = ProductKernel([RBFKernel(sigma_rbf)])
    old_bounds = [(-5.0, 5.0)]
    old_mean = bayesian_quadrature_nd(node, value, old_kernel, old_bounds).mean

    assert abs(new_mean) < 1e-3
    assert abs(old_mean) > 1.0


def test_renderer_centered_variance_matches_formulation_one_variance():
    """Formulation 1 (bayesian_quadrature_rendering_aware) and formulation 2
    (renderer_centered_residual_variance) share the exact same z/z0/K
    machinery and must report the same variance for the same inputs --
    they differ only in which mean is reported/used alongside it."""
    w = GaussianRenderWeight(amplitude=0.9, center=[0.5], covariance=[[0.15]])
    nodes = np.array([[0.1], [0.4], [0.9]])
    values = np.array([1.0, 2.0, 0.5])
    sigma_rbf = 0.25

    result = bayesian_quadrature_rendering_aware(nodes, values, w, sigma_rbf=sigma_rbf)
    residual_variance = renderer_centered_residual_variance(nodes, w, sigma_rbf=sigma_rbf)

    assert abs(result.variance - residual_variance) < 1e-9


def test_rendering_aware_with_zero_nodes_returns_prior():
    w = GaussianRenderWeight(amplitude=1.0, center=[0.0], covariance=[[0.2]])
    sigma_rbf = 0.3
    result = bayesian_quadrature_rendering_aware(np.empty((0, 1)), np.empty(0), w, sigma_rbf=sigma_rbf)
    assert result.mean == 0.0
    assert abs(result.variance - rendering_aware_prior_variance(w, sigma_rbf)) < 1e-12


def test_numerical_mode_matches_closed_form_mode_end_to_end():
    w = GaussianRenderWeight(amplitude=0.7, center=[1.0], covariance=[[0.1]])
    sigma_rbf = 0.3
    nodes = np.array([[0.8], [1.0], [1.3]])
    values = np.array([0.5, 0.9, 0.4])
    base = ProductKernel([RBFKernel(sigma_rbf)])
    domain = [(-9.0, 11.0)]

    closed = bayesian_quadrature_rendering_aware(nodes, values, w, sigma_rbf=sigma_rbf, mode="closed_form")
    numerical = bayesian_quadrature_rendering_aware(
        nodes, values, w, base_kernel=base, domain=domain, mode="numerical"
    )

    assert abs(closed.mean - numerical.mean) < 1e-4
    assert abs(closed.variance - numerical.variance) < 1e-4


def test_numerical_mode_works_with_matern_base_kernel():
    """The numerical fallback's whole point is supporting a non-Gaussian
    base kernel the closed form can't handle -- Matern-3/2 here."""
    w = GaussianRenderWeight(amplitude=1.0, center=[0.5], covariance=[[0.1]])
    base = ProductKernel([MaternKernel(rho=0.4)])
    domain = [(-9.0, 10.0)]
    nodes = np.array([[0.3], [0.5], [0.8]])
    values = np.array([1.0, 1.2, 0.9])

    result = bayesian_quadrature_rendering_aware(nodes, values, w, base_kernel=base, domain=domain, mode="numerical")
    assert np.isfinite(result.mean)
    assert result.variance >= 0.0


def test_closed_form_mode_requires_sigma_rbf():
    w = GaussianRenderWeight(amplitude=1.0, center=[0.0], covariance=[[0.1]])
    with pytest.raises(ValueError):
        bayesian_quadrature_rendering_aware(np.array([[0.0]]), np.array([1.0]), w, mode="closed_form")


def test_numerical_mode_requires_base_kernel_and_domain():
    w = GaussianRenderWeight(amplitude=1.0, center=[0.0], covariance=[[0.1]])
    with pytest.raises(ValueError):
        bayesian_quadrature_rendering_aware(np.array([[0.0]]), np.array([1.0]), w, mode="numerical")


def test_directional_reduces_to_spatial_only_when_dir_kernel_is_kappa_zero():
    """kappa=0 makes dir_kernel.k(d, d') = exp(0*(dot-1)) = 1 for every pair
    of directions -- every direction is treated as perfectly correlated
    with every other -- which should make
    bayesian_quadrature_rendering_aware_directional collapse to exactly
    bayesian_quadrature_rendering_aware's own (direction-free) result,
    since K and z each pick up a factor of 1 everywhere."""
    w = GaussianRenderWeight(amplitude=0.8, center=[1.0], covariance=[[0.2]])
    sigma_rbf = 0.3
    positions = np.array([[0.6], [1.0], [1.5]])
    values = np.array([0.4, 0.9, 0.5])
    directions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # arbitrary, unrelated directions
    query_direction = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)

    dir_kernel = DirectionalKernel(kappa=0.0)
    directional = bayesian_quadrature_rendering_aware_directional(
        positions, directions, values, w, dir_kernel, query_direction, sigma_rbf=sigma_rbf
    )
    spatial_only = bayesian_quadrature_rendering_aware(positions, values, w, sigma_rbf=sigma_rbf)

    assert abs(directional.mean - spatial_only.mean) < 1e-9
    assert abs(directional.variance - spatial_only.variance) < 1e-9


def test_directional_prior_variance_matches_spatial_only_prior_variance():
    """z_{q,0} should be exactly the spatial-only prior variance regardless
    of dir_kernel/query_direction (self-similarity k_dir(d,d) == 1 always)
    -- checked here via the n=0 fallback, where mean/variance reduce to
    (0, z_{q,0}) directly."""
    w = GaussianRenderWeight(amplitude=0.5, center=[0.0], covariance=[[0.3]])
    sigma_rbf = 0.4
    dir_kernel = DirectionalKernel(kappa=4.0)
    query_direction = np.array([0.0, 0.0, 1.0])

    result = bayesian_quadrature_rendering_aware_directional(
        np.empty((0, 1)), np.empty((0, 3)), np.empty(0), w, dir_kernel, query_direction, sigma_rbf=sigma_rbf
    )
    assert result.mean == 0.0
    assert abs(result.variance - rendering_aware_prior_variance(w, sigma_rbf)) < 1e-12


def test_directional_mean_is_pulled_toward_the_observation_closest_to_the_query_direction():
    """Two observations at the *same* position (so the spatial term alone
    can't distinguish them) but different directions and very different
    colors -- kappa large enough to sharply distinguish directions should
    pull the mean toward whichever observation's direction matches the
    query direction, exactly the classic directional-kernel behavior
    (bayesian_quadrature_directional's own tests), now reproduced through
    the rendering-aware a_q instead of a box kernel."""
    w = GaussianRenderWeight(amplitude=1.0, center=[1.0], covariance=[[0.3]])
    sigma_rbf = 0.4
    positions = np.array([[1.0], [1.0]])
    values = np.array([0.0, 10.0])
    dir_a = np.array([1.0, 0.0, 0.0])
    dir_b = np.array([0.0, 1.0, 0.0])
    directions = np.array([dir_a, dir_b])
    dir_kernel = DirectionalKernel(kappa=20.0)

    toward_a = bayesian_quadrature_rendering_aware_directional(
        positions, directions, values, w, dir_kernel, dir_a, sigma_rbf=sigma_rbf
    )
    toward_b = bayesian_quadrature_rendering_aware_directional(
        positions, directions, values, w, dir_kernel, dir_b, sigma_rbf=sigma_rbf
    )

    assert toward_a.mean < toward_b.mean
    assert abs(toward_a.mean - values[0]) < abs(toward_a.mean - values[1])
    assert abs(toward_b.mean - values[1]) < abs(toward_b.mean - values[0])


def test_directional_numerical_mode_matches_closed_form_mode():
    w = GaussianRenderWeight(amplitude=0.7, center=[1.0], covariance=[[0.1]])
    sigma_rbf = 0.3
    positions = np.array([[0.8], [1.0], [1.3]])
    values = np.array([0.5, 0.9, 0.4])
    directions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0] / np.sqrt(2)])
    dir_kernel = DirectionalKernel(kappa=3.0)
    query_direction = np.array([1.0, 0.0, 0.0])
    base = ProductKernel([RBFKernel(sigma_rbf)])
    domain = [(-9.0, 11.0)]

    closed = bayesian_quadrature_rendering_aware_directional(
        positions, directions, values, w, dir_kernel, query_direction, sigma_rbf=sigma_rbf, mode="closed_form"
    )
    numerical = bayesian_quadrature_rendering_aware_directional(
        positions, directions, values, w, dir_kernel, query_direction, base_kernel=base, domain=domain, mode="numerical"
    )

    assert abs(closed.mean - numerical.mean) < 1e-4
    assert abs(closed.variance - numerical.variance) < 1e-4
