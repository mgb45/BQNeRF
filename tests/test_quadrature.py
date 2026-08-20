import numpy as np

from bq_splat.kernels import RBFKernel
from bq_splat.quadrature import bayesian_quadrature
from bq_splat.reference import true_integral


def test_bq_mean_recovers_true_integral_for_dense_smooth_signal():
    """With many nodes relative to the kernel lengthscale and signal
    smoothness, BQ posterior mean should closely match the true integral."""
    a, b = 0.0, 10.0
    rng = np.random.default_rng(1)
    centers = rng.uniform(a, b, size=4)

    def g_true(t):
        t = np.asarray(t, dtype=float)
        return sum(np.exp(-0.5 * ((t - c) / 0.5) ** 2) for c in centers)

    nodes = np.linspace(a, b, 200)
    values = g_true(nodes)
    kernel = RBFKernel(sigma=0.3)

    result = bayesian_quadrature(nodes, values, kernel, a, b)
    truth = true_integral(g_true, a, b)

    assert abs(result.mean - truth) / abs(truth) < 0.02


def test_bq_variance_shrinks_as_nodes_are_added():
    a, b = 0.0, 10.0
    rng = np.random.default_rng(2)
    kernel = RBFKernel(sigma=0.4)

    def g_true(t):
        return np.sin(np.asarray(t, dtype=float))

    variances = []
    for n in [2, 5, 10, 30]:
        nodes = np.sort(rng.uniform(a, b, size=n))
        values = g_true(nodes)
        result = bayesian_quadrature(nodes, values, kernel, a, b)
        variances.append(result.variance)

    assert all(v1 >= v2 - 1e-9 for v1, v2 in zip(variances, variances[1:]))


def test_bq_variance_is_nonnegative():
    a, b = 0.0, 5.0
    rng = np.random.default_rng(3)
    kernel = RBFKernel(sigma=0.2)
    nodes = np.sort(rng.uniform(a, b, size=8))
    values = rng.normal(size=8)
    result = bayesian_quadrature(nodes, values, kernel, a, b)
    assert result.variance >= 0.0


def test_bq_variance_higher_in_undersampled_gap_than_dense_region():
    """The toy analogue of the paper's central claim: given equal-size
    regions, the one with sparser node coverage should carry higher BQ
    posterior variance, independent of the observed values there."""
    a, b = 0.0, 10.0
    kernel = RBFKernel(sigma=0.3)

    dense_nodes = np.linspace(1.0, 3.0, 20)
    sparse_nodes = np.linspace(1.0, 3.0, 3)

    values_dense = np.ones_like(dense_nodes)
    values_sparse = np.ones_like(sparse_nodes)

    dense_result = bayesian_quadrature(dense_nodes, values_dense, kernel, 1.0, 3.0)
    sparse_result = bayesian_quadrature(sparse_nodes, values_sparse, kernel, 1.0, 3.0)

    assert sparse_result.variance > dense_result.variance
