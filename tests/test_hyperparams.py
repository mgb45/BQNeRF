import numpy as np

from bq_splat.hyperparams import (
    fit_kernel_param,
    fit_kernel_param_pooled,
    fit_kernel_param_pooled_nd,
    log_marginal_likelihood,
    log_marginal_likelihood_nd,
)
from bq_splat.kernels import MaternKernel, ProductKernel, RBFKernel


def test_lml_prefers_true_generating_bandwidth_over_far_off_ones():
    """Not a strict recovery test (LML surfaces for a single lengthscale can
    be flat near the optimum), but the true bandwidth used to generate the
    data should beat bandwidths that are off by an order of magnitude."""
    rng = np.random.default_rng(0)
    true_sigma = 0.4
    nodes = np.sort(rng.uniform(0, 10, size=25))
    kxx = RBFKernel(sigma=true_sigma).k(nodes.reshape(-1, 1), nodes.reshape(1, -1))
    kxx = kxx + 1e-6 * np.eye(len(nodes))
    values = rng.multivariate_normal(np.zeros(len(nodes)), kxx)

    lml_true = log_marginal_likelihood(nodes, values, RBFKernel(sigma=true_sigma))
    lml_too_small = log_marginal_likelihood(nodes, values, RBFKernel(sigma=0.01))
    lml_too_large = log_marginal_likelihood(nodes, values, RBFKernel(sigma=5.0))

    assert lml_true > lml_too_small
    assert lml_true > lml_too_large


def test_fit_kernel_param_recovers_reasonable_bandwidth_for_gp_samples():
    rng = np.random.default_rng(1)
    true_sigma = 0.5
    nodes = np.sort(rng.uniform(0, 10, size=40))
    kxx = RBFKernel(sigma=true_sigma).k(nodes.reshape(-1, 1), nodes.reshape(1, -1))
    kxx = kxx + 1e-6 * np.eye(len(nodes))
    values = rng.multivariate_normal(np.zeros(len(nodes)), kxx)

    fit = fit_kernel_param(nodes, values, lambda s: RBFKernel(sigma=s), bounds=(0.05, 3.0))

    assert 0.2 < fit.param < 1.2


def test_fit_kernel_param_pooled_recovers_shared_bandwidth_across_datasets():
    rng = np.random.default_rng(3)
    true_sigma = 0.6
    datasets = []
    for _ in range(5):
        nodes = np.sort(rng.uniform(0, 10, size=25))
        kxx = RBFKernel(sigma=true_sigma).k(nodes.reshape(-1, 1), nodes.reshape(1, -1))
        kxx = kxx + 1e-6 * np.eye(len(nodes))
        values = rng.multivariate_normal(np.zeros(len(nodes)), kxx)
        datasets.append((nodes, values))

    fit = fit_kernel_param_pooled(datasets, lambda s: RBFKernel(sigma=s), bounds=(0.05, 3.0))
    assert 0.3 < fit.param < 1.1


def test_fit_kernel_param_works_for_matern_too():
    rng = np.random.default_rng(2)

    def g_true(t):
        return np.sin(t) + 0.3 * np.sin(5 * t)

    nodes = np.sort(rng.uniform(0, 10, size=30))
    values = g_true(nodes)

    fit = fit_kernel_param(nodes, values, lambda r: MaternKernel(rho=r), bounds=(1e-2, 3.0))
    assert fit.param > 0
    assert np.isfinite(fit.log_marginal_likelihood)


def _product_rbf(sigma, d=3):
    return ProductKernel([RBFKernel(sigma=sigma)] * d)


def test_lml_nd_matches_1d_lml_for_a_single_axis_product_kernel():
    """A ProductKernel with exactly one 1D RBF factor computes the exact
    same Gram matrix as RBFKernel directly (ProductKernel.k reduces to its
    single factor) -- a direct correctness check that
    log_marginal_likelihood_nd isn't a different formula, just a different
    input convention (RBFKernel isn't normalized to k(x,x)=1, so this must
    be checked via a genuine D=1 product kernel, not by collapsing extra
    axes to a shared constant, which would rescale the Gram matrix by that
    constant's self-similarity and change the LML for a reason that has
    nothing to do with correctness)."""
    rng = np.random.default_rng(4)
    true_sigma = 0.4
    x = np.sort(rng.uniform(0, 10, size=20))
    kxx_1d = RBFKernel(sigma=true_sigma).k(x.reshape(-1, 1), x.reshape(1, -1))
    kxx_1d = kxx_1d + 1e-6 * np.eye(len(x))
    values = rng.multivariate_normal(np.zeros(len(x)), kxx_1d)

    lml_1d = log_marginal_likelihood(x, values, RBFKernel(sigma=true_sigma))

    nodes_1d_as_nd = x.reshape(-1, 1)
    lml_nd = log_marginal_likelihood_nd(nodes_1d_as_nd, values, ProductKernel([RBFKernel(sigma=true_sigma)]))

    assert np.isclose(lml_1d, lml_nd, atol=1e-9)


def test_fit_kernel_param_pooled_nd_recovers_shared_bandwidth_across_3d_windows():
    rng = np.random.default_rng(5)
    true_sigma = 0.3
    datasets = []
    for _ in range(6):
        nodes = rng.uniform(0, 2, size=(20, 3))
        kxx = _product_rbf(true_sigma).k(nodes, nodes) + 1e-6 * np.eye(len(nodes))
        values = rng.multivariate_normal(np.zeros(len(nodes)), kxx)
        datasets.append((nodes, values))

    fit = fit_kernel_param_pooled_nd(datasets, lambda s: _product_rbf(s), bounds=(0.03, 2.0))
    assert 0.1 < fit.param < 0.8
