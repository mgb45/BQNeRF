import numpy as np
from scipy import integrate

from bq_splat.kernels import MaternKernel, ProductKernel, RBFKernel
from bq_splat.quadrature import bayesian_quadrature_nd


def test_product_rbf_v_matches_numerical_2d_integration():
    kernel = ProductKernel([RBFKernel(sigma=0.4), RBFKernel(sigma=0.3)])
    bounds = [(0.0, 5.0), (0.0, 4.0)]
    x = np.array([1.2, 2.3])

    expected, _ = integrate.dblquad(
        lambda t2, t1: kernel.k(x, np.array([t1, t2])).item(),
        bounds[0][0], bounds[0][1],
        bounds[1][0], bounds[1][1],
    )
    got = float(kernel.v(x, bounds)[0])
    assert abs(got - expected) < 1e-6


def test_product_rbf_vv_matches_numerical_4d_integration_via_nested_v():
    kernel = ProductKernel([RBFKernel(sigma=0.5), RBFKernel(sigma=0.5)])
    bounds = [(0.0, 3.0), (0.0, 3.0)]

    # vv = integral over the domain of v(x) dx -- check via a separate route
    # (numerically integrating our own v) rather than re-deriving vv by hand.
    def v_scalar(x1, x2):
        return float(kernel.v(np.array([x1, x2]), bounds)[0])

    expected, _ = integrate.dblquad(v_scalar, bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1])
    got = kernel.vv(bounds)
    assert abs(got - expected) < 1e-4


def test_product_rbf_is_exact_isotropic_gaussian_in_2d():
    """ProductKernel([RBF(sigma)]*2] should equal the true isotropic 2D
    Gaussian kernel exp(-||x-y||^2 / (2 sigma^2)) / (2 pi sigma^2) exactly,
    since squared Euclidean distance separates into a sum over axes."""
    sigma = 0.6
    kernel = ProductKernel([RBFKernel(sigma=sigma), RBFKernel(sigma=sigma)])
    x = np.array([[1.0, 2.0]])
    y = np.array([[1.7, 1.1]])

    got = kernel.k(x, y)[0, 0]
    r2 = np.sum((x - y) ** 2)
    expected = np.exp(-r2 / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    assert abs(got - expected) < 1e-12


def test_product_kernel_gram_and_v_factorize_as_expected():
    """Direct check of the separability property ProductKernel relies on:
    K_2D(i,j) = K_axis1(i,j) * K_axis2(i,j), and v_2D(x) = v_axis1(x1) *
    v_axis2(x2). (An earlier version of this test tried to check this
    indirectly through bayesian_quadrature_nd's posterior mean and got the
    algebra wrong -- a constant per-axis kernel value scales the Gram matrix
    and its inverse in a way that doesn't cancel the way a first pass at the
    derivation assumed. Checking the factorization directly is both simpler
    and actually correct.)"""
    rng = np.random.default_rng(0)
    kernel_x = RBFKernel(sigma=0.4)
    kernel_y = RBFKernel(sigma=0.7)
    kernel_2d = ProductKernel([kernel_x, kernel_y])

    pts = rng.uniform(0, 5, size=(6, 2))
    K_2d = kernel_2d.k(pts, pts)
    K_x = kernel_x.k(pts[:, 0].reshape(-1, 1), pts[:, 0].reshape(1, -1))
    K_y = kernel_y.k(pts[:, 1].reshape(-1, 1), pts[:, 1].reshape(1, -1))
    np.testing.assert_allclose(K_2d, K_x * K_y, atol=1e-12)

    bounds = [(0.0, 5.0), (0.0, 5.0)]
    v_2d = kernel_2d.v(pts, bounds)
    v_x = kernel_x.v(pts[:, 0], *bounds[0])
    v_y = kernel_y.v(pts[:, 1], *bounds[1])
    np.testing.assert_allclose(v_2d, v_x * v_y, atol=1e-12)

    assert abs(kernel_2d.vv(bounds) - kernel_x.vv(*bounds[0]) * kernel_y.vv(*bounds[1])) < 1e-12


def test_bq_nd_runs_and_gives_finite_sane_result():
    rng = np.random.default_rng(0)
    a, b = 0.0, 10.0
    kernel_2d = ProductKernel([RBFKernel(sigma=0.4), RBFKernel(sigma=0.4)])
    bounds = [(a, b), (a, b)]
    nodes = rng.uniform(a, b, size=(20, 2))
    values = np.sin(nodes[:, 0]) + np.cos(nodes[:, 1])

    result = bayesian_quadrature_nd(nodes, values, kernel_2d, bounds)
    assert np.isfinite(result.mean)
    assert result.variance >= 0.0


def test_matern_product_kernel_gram_is_positive_semidefinite():
    rng = np.random.default_rng(1)
    kernel = ProductKernel([MaternKernel(rho=0.4), MaternKernel(rho=0.6)])
    x = rng.uniform(0, 5, size=(12, 2))
    K = kernel.k(x, x)
    eigvals = np.linalg.eigvalsh(K)
    assert eigvals.min() > -1e-8
