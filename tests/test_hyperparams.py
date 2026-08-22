import numpy as np

from bq_splat.hyperparams import fit_kernel_param, fit_kernel_param_pooled, log_marginal_likelihood
from bq_splat.kernels import MaternKernel, RBFKernel


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
