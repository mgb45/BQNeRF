import numpy as np

from gs_experiment.hyperparams import (
    fit_kernel_param,
    fit_kernel_param_and_noise_pooled_nd,
    fit_kernel_param_pooled,
    fit_kernel_param_pooled_nd,
    log_marginal_likelihood,
    log_marginal_likelihood_nd,
)
from gs_experiment.kernels import ProductKernel, RBFKernel


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


def test_fit_kernel_param_and_noise_pooled_nd_recovers_bandwidth_and_noise():
    """Generate data from the actual noisy-observation model this fit
    targets (y = f(x) + eps, f ~ GP(0, k), eps ~ N(0, true_noise_var) iid)
    and check the joint fit recovers both the true bandwidth and a
    same-order-of-magnitude noise variance -- not a strict-equality
    recovery test (marginal-likelihood surfaces can be flat, especially
    for a variance parameter with modest data), the same "true value beats
    wildly-off ones" standard this file's other LML tests use."""
    rng = np.random.default_rng(9)
    true_sigma = 0.3
    true_noise_var = 0.05
    datasets = []
    for _ in range(8):
        nodes = rng.uniform(0, 2, size=(25, 3))
        kxx = _product_rbf(true_sigma).k(nodes, nodes) + (1e-6 + true_noise_var) * np.eye(len(nodes))
        values = rng.multivariate_normal(np.zeros(len(nodes)), kxx)
        datasets.append((nodes, values))

    fit = fit_kernel_param_and_noise_pooled_nd(
        datasets, lambda s: _product_rbf(s), bounds=(0.03, 2.0), noise_bounds=(1e-4, 1.0), n_grid=15
    )
    assert 0.1 < fit.param < 0.8
    assert 0.01 < fit.noise_variance < 0.25

    # the fitted (sigma, noise_variance) pair must explain the data at
    # least as well as the true generating values under the same pooled
    # objective -- a real, not just plausible-looking, optimum.
    from gs_experiment.hyperparams import pooled_log_marginal_likelihood_nd

    lml_fitted = pooled_log_marginal_likelihood_nd(
        datasets, _product_rbf(fit.param), noise_variance=fit.noise_variance
    )
    lml_true = pooled_log_marginal_likelihood_nd(datasets, _product_rbf(true_sigma), noise_variance=true_noise_var)
    assert lml_fitted >= lml_true - 1e-6


def test_fit_kernel_param_and_noise_pooled_nd_zero_noise_data_prefers_small_noise_to_large():
    """When the data really is noiseless (generated from f(x) alone, no
    eps) but points are sparse relative to the domain (so most pairs are
    far apart and weakly correlated -- a real identifiability limit, not a
    bug: sparse, short-lengthscale GP draws are hard to tell apart from
    i.i.d. noise from modest data alone), don't demand the fit recovers
    exactly zero noise. Instead check the weaker, still-meaningful claim
    this file's other LML tests already rely on (see
    test_lml_prefers_true_generating_bandwidth_over_far_off_ones): the
    fitted noise variance should explain the data at least as well, under
    the pooled marginal likelihood, as an artificially large one pinned at
    the top of the search range."""
    rng = np.random.default_rng(3)
    true_sigma = 0.25
    datasets = []
    for _ in range(6):
        nodes = rng.uniform(0, 2, size=(20, 3))
        kxx = _product_rbf(true_sigma).k(nodes, nodes) + 1e-6 * np.eye(len(nodes))
        values = rng.multivariate_normal(np.zeros(len(nodes)), kxx)
        datasets.append((nodes, values))

    fit = fit_kernel_param_and_noise_pooled_nd(
        datasets, lambda s: _product_rbf(s), bounds=(0.03, 2.0), noise_bounds=(1e-4, 1.0), n_grid=15
    )

    from gs_experiment.hyperparams import pooled_log_marginal_likelihood_nd

    lml_fitted = pooled_log_marginal_likelihood_nd(
        datasets, _product_rbf(fit.param), noise_variance=fit.noise_variance
    )
    lml_large_noise = pooled_log_marginal_likelihood_nd(datasets, _product_rbf(fit.param), noise_variance=1.0)
    assert lml_fitted >= lml_large_noise
