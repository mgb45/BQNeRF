import numpy as np

from bq_splat.kernels import DirectionalKernel, ProductKernel, RBFKernel
from bq_splat.quadrature import (
    bayesian_quadrature_nd,
    bayesian_quadrature_directional,
    directional_posterior_variance,
)


def angle_to_unit_vector(theta):
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    return np.stack([np.cos(theta), np.sin(theta)], axis=1)


def test_directional_kernel_self_similarity_is_one():
    kernel = DirectionalKernel(kappa=3.0)
    for theta in [0.0, 1.0, -2.5, np.pi]:
        w = angle_to_unit_vector(theta)
        assert abs(kernel.k(w, w)[0, 0] - 1.0) < 1e-12


def test_directional_kernel_gram_is_positive_semidefinite():
    rng = np.random.default_rng(0)
    kernel = DirectionalKernel(kappa=2.0)
    thetas = rng.uniform(0, 2 * np.pi, size=15)
    w = angle_to_unit_vector(thetas)
    K = kernel.k(w, w)
    eigvals = np.linalg.eigvalsh(K)
    assert eigvals.min() > -1e-8


def test_directional_kernel_decreases_with_angular_separation():
    kernel = DirectionalKernel(kappa=2.0)
    w0 = angle_to_unit_vector(0.0)
    seps = [0.0, 0.3, 1.0, 2.0, np.pi]
    values = [float(kernel.k(w0, angle_to_unit_vector(s))[0, 0]) for s in seps]
    assert all(v1 >= v2 - 1e-12 for v1, v2 in zip(values, values[1:]))


def test_bayesian_quadrature_directional_reduces_exactly_to_nd_when_kappa_zero():
    """With kappa=0, dir_kernel.k(.,.) == 1 identically for every pair,
    regardless of the actual directions -- so the joint computation must
    reduce EXACTLY (not approximately) to the pure-position result,
    whatever (nonsense) directions are passed in.

    bayesian_quadrature_directional expects the ProductKernel-style
    v(x, bounds)/vv(bounds) interface (bounds as a list of (a, b) pairs,
    even in 1D), not plain Kernel's v(x, a, b)/vv(a, b) -- so pos_kernel is
    wrapped in a single-kernel ProductKernel here to match.
    """
    rng = np.random.default_rng(1)
    a, b = 0.0, 10.0
    positions = np.sort(rng.uniform(a, b, size=12))
    values = np.sin(positions) + 1.0
    random_directions = angle_to_unit_vector(rng.uniform(0, 2 * np.pi, size=12))
    random_query_direction = angle_to_unit_vector(rng.uniform(0, 2 * np.pi))

    pos_kernel = ProductKernel([RBFKernel(sigma=0.5)])
    dir_kernel = DirectionalKernel(kappa=0.0)
    bounds = [(a, b)]

    expected = bayesian_quadrature_nd(positions, values, pos_kernel, bounds)
    got = bayesian_quadrature_directional(
        positions, random_directions, values, pos_kernel, dir_kernel, bounds, random_query_direction
    )

    assert abs(got.mean - expected.mean) < 1e-10
    assert abs(got.variance - expected.variance) < 1e-10


def test_directional_posterior_variance_lower_inside_observed_cone_than_outside():
    rng = np.random.default_rng(2)
    dir_kernel = DirectionalKernel(kappa=4.0)

    # observations clustered in a narrow cone around theta=0
    cone_thetas = rng.uniform(-0.3, 0.3, size=10)
    directions = angle_to_unit_vector(cone_thetas)
    values = rng.normal(size=10)

    inside_query = angle_to_unit_vector(0.1)
    outside_query = angle_to_unit_vector(np.pi)  # directly opposite the cone

    inside_result = directional_posterior_variance(directions, values, dir_kernel, inside_query)
    outside_result = directional_posterior_variance(directions, values, dir_kernel, outside_query)

    assert inside_result.variance < outside_result.variance


def test_directional_posterior_variance_shrinks_as_angular_coverage_widens():
    rng = np.random.default_rng(3)
    dir_kernel = DirectionalKernel(kappa=4.0)
    query = angle_to_unit_vector(np.pi)  # a fixed, hard-to-reach direction

    variances = []
    for half_width in [0.2, 1.0, 2.0, np.pi]:
        thetas = rng.uniform(-half_width, half_width, size=15)
        directions = angle_to_unit_vector(thetas)
        values = rng.normal(size=15)
        result = directional_posterior_variance(directions, values, dir_kernel, query)
        variances.append(result.variance)

    assert all(v1 >= v2 - 1e-9 for v1, v2 in zip(variances, variances[1:]))
