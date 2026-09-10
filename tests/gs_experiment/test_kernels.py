import math

import numpy as np
import pytest
from scipy import integrate

from gs_experiment.kernels import MaternKernel, RationalQuadraticKernel, RBFKernel


def test_rbf_matches_nerf_py_closed_form():
    """Cross-check against the exact erf-based formula in models/nerf.py's
    `rbf_vf`, to make sure the port (via scipy.stats.norm.cdf) is faithful."""
    sig = 0.25
    near, far = 0.0, 1.0
    xs = np.array([0.1, 0.3, 0.5, 0.7, 0.95])

    kernel = RBFKernel(sigma=sig)
    ours = kernel.v(xs, near, far)

    # models/nerf.py `rbf_vf`: 0.5*erf((far-x)/(sig*sqrt2)) - 0.5*erf((near-x)/(sig*sqrt2))
    expected_nerf_py = np.array(
        [
            0.5 * math.erf((far - x) / (sig * math.sqrt(2))) - 0.5 * math.erf((near - x) / (sig * math.sqrt(2)))
            for x in xs
        ]
    )
    np.testing.assert_allclose(ours, expected_nerf_py, atol=1e-10)


def test_rbf_v_integrates_to_kernel_numerically():
    kernel = RBFKernel(sigma=0.3)
    a, b = -1.0, 4.0
    for x in [-0.5, 0.0, 1.7, 3.9, 5.0]:
        expected, _ = integrate.quad(lambda t: float(kernel.k(x, t)), a, b)
        got = float(kernel.v(x, a, b))
        assert abs(got - expected) < 1e-8


def test_rbf_vv_matches_nested_numerical_integration():
    kernel = RBFKernel(sigma=0.4)
    a, b = 0.0, 5.0
    expected, _ = integrate.dblquad(lambda y, x: float(kernel.k(x, y)), a, b, a, b)
    got = kernel.vv(a, b)
    assert abs(got - expected) < 1e-4


def test_matern_v_integrates_to_kernel_numerically():
    kernel = MaternKernel(rho=0.5)
    a, b = -1.0, 3.0
    for x in [-0.5, 0.4, 2.9]:
        expected, _ = integrate.quad(lambda t: float(kernel.k(x, t)), a, b, points=[x] if a < x < b else None)
        got = float(kernel.v(x, a, b))
        assert abs(got - expected) < 1e-6


def test_matern_vv_matches_nested_numerical_integration():
    kernel = MaternKernel(rho=0.6)
    a, b = 0.0, 4.0
    expected, _ = integrate.dblquad(lambda y, x: float(kernel.k(x, y)), a, b, a, b)
    got = kernel.vv(a, b)
    assert abs(got - expected) < 1e-3


def test_rational_quadratic_v_integrates_to_kernel_numerically():
    kernel = RationalQuadraticKernel(l=0.5)
    a, b = -1.0, 3.0
    for x in [-0.5, 0.4, 2.9]:
        expected, _ = integrate.quad(lambda t: float(kernel.k(x, t)), a, b, points=[x] if a < x < b else None)
        got = float(kernel.v(x, a, b))
        assert abs(got - expected) < 1e-6


def test_rational_quadratic_vv_matches_nested_numerical_integration():
    kernel = RationalQuadraticKernel(l=0.6)
    a, b = 0.0, 4.0
    expected, _ = integrate.dblquad(lambda y, x: float(kernel.k(x, y)), a, b, a, b)
    got = kernel.vv(a, b)
    assert abs(got - expected) < 1e-3


def test_rational_quadratic_self_similarity_is_one():
    """k(0) = (1 + 0)^(-alpha) = 1 always, the same normalization
    convention MaternKernel uses (unlike RBFKernel, which is a normalized
    density rather than an amplitude-1 covariance)."""
    kernel = RationalQuadraticKernel(l=0.37, alpha=2.3)
    assert abs(float(kernel.k(1.23, 1.23)) - 1.0) < 1e-12


def test_rational_quadratic_approaches_rbf_as_alpha_grows():
    """As alpha -> infinity, RQ's scale mixture concentrates on a single
    lengthscale and the kernel converges to the amplitude-1 RBF/exponentiated-
    quadratic form exp(-r^2 / (2 l^2)) (GPML sec. 4.2.1) -- checked at a
    large but finite alpha, not RBFKernel itself (which uses a different,
    normalized-density convention -- see RBFKernel's docstring)."""
    l = 0.4
    r = 0.55
    rq = RationalQuadraticKernel(l=l, alpha=1e5)
    expected = np.exp(-(r**2) / (2 * l**2))
    assert abs(float(rq.k(0.0, r)) - expected) < 1e-4


@pytest.mark.parametrize(
    "KernelCls,kwarg",
    [(RBFKernel, dict(sigma=0.3)), (MaternKernel, dict(rho=0.4)), (RationalQuadraticKernel, dict(l=0.4))],
)
def test_kernel_gram_matrix_is_positive_semidefinite(KernelCls, kwarg):
    rng = np.random.default_rng(0)
    kernel = KernelCls(**kwarg)
    x = np.sort(rng.uniform(0, 5, size=12))
    K = kernel.k(x.reshape(-1, 1), x.reshape(1, -1))
    eigvals = np.linalg.eigvalsh(K)
    assert eigvals.min() > -1e-8
