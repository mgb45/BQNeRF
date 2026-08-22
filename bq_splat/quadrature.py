"""Bayesian quadrature: posterior mean/variance of an integral given noiseless
point evaluations of the integrand, under a GP prior with a chosen kernel.

Formulation note (important, and the reason this module exists separately
from a direct port of models/nerf.py): the observations passed in here must
be point evaluations g(t_i) of the *integrand* of the rendering integral
C = integral of g(t) dt, not pre-integrated per-bin contributions like
weight_i * color_i (which already bake in an implicit bin width). Treating
an already-integrated quantity as a further point evaluation to integrate
again silently double-counts the bin width — this repo's own git history
has a "fixed bug in bq quadrature, was doing double quad" commit, which is
exactly this trap. See bq_splat/reference.py and bq_splat/toy_scene.py for
how nodes/values are generated to avoid it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bq_splat.kernels import DirectionalKernel, Kernel, ProductKernel


@dataclass
class BQResult:
    mean: float
    variance: float


def bayesian_quadrature_directional(
    positions,
    directions,
    values,
    pos_kernel,
    dir_kernel: DirectionalKernel,
    pos_bounds,
    query_direction,
    rel_jitter: float = 1e-4,
) -> BQResult:
    """BQ over a joint (position, direction) domain, where position is
    integrated over (as in bayesian_quadrature_nd -- a genuine quadrature
    integral, e.g. a pixel footprint) but direction is evaluated at one
    query direction, not integrated -- a rendered pixel looks in one
    specific outgoing direction, it doesn't average over a range of them.
    See DirectionalKernel's docstring for why this asymmetry is the correct
    generalization, not an approximation of a "properly" symmetric one.

    `positions`/`directions`/`values` are parallel arrays: `values[i]` is an
    observation at `positions[i]` from `directions[i]` (e.g. one training
    view's contribution to one splat). `pos_kernel` must use the
    ProductKernel-style interface -- `v(x, bounds)`/`vv(bounds)` with
    `bounds` a list of (a, b) pairs, even in 1D (wrap a plain Kernel like
    `ProductKernel([RBFKernel(sigma)])`) -- rather than plain Kernel's
    `v(x, a, b)`/`vv(a, b)`, so callers don't need to special-case
    dimensionality. `dir_kernel` is a DirectionalKernel (no v/vv, evaluated
    pointwise).

    K_ij = pos_kernel.k(x_i, x_j) * dir_kernel.k(w_i, w_j)
    v_i   = pos_kernel.v(x_i, pos_bounds) * dir_kernel.k(w_i, w_query)
    vv    = pos_kernel.vv(pos_bounds) * dir_kernel.k(w_query, w_query)
          = pos_kernel.vv(pos_bounds)               (self-similarity is 1)
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim == 1:
        positions = positions.reshape(-1, 1)
    directions = np.asarray(directions, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)
    n = positions.shape[0]

    vv = float(pos_kernel.vv(pos_bounds))  # dir_kernel.k(q, q) == 1, omitted
    if n == 0:
        return BQResult(mean=0.0, variance=vv)

    kxx = pos_kernel.k(positions, positions) * dir_kernel.k(directions, directions)
    jitter = rel_jitter * np.mean(np.diag(kxx))
    kxx = kxx + jitter * np.eye(n)

    v_pos = np.asarray(pos_kernel.v(positions, pos_bounds)).reshape(-1)
    v_dir = dir_kernel.k(directions, query_direction).reshape(-1)
    v = v_pos * v_dir

    solved = np.linalg.solve(kxx, values)
    mean = float(v @ solved)

    solved_v = np.linalg.solve(kxx, v)
    variance = float(vv - v @ solved_v)

    return BQResult(mean=mean, variance=max(variance, 0.0))


def directional_posterior_variance(
    directions, values, dir_kernel: DirectionalKernel, query_direction, rel_jitter: float = 1e-4
) -> BQResult:
    """Pure-directional special case of bayesian_quadrature_directional,
    for a single fixed spatial location (position integration dropped
    entirely rather than degenerated into it) -- standard GP regression
    posterior mean/variance at one query direction, given observations from
    other directions. Used to validate DirectionalKernel's behavior in
    isolation, without conflating it with the position-integration
    machinery bayesian_quadrature_directional also does.
    """
    directions = np.asarray(directions, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)
    n = directions.shape[0]

    prior_variance = 1.0  # dir_kernel.k(q, q) == 1 always
    if n == 0:
        return BQResult(mean=0.0, variance=prior_variance)

    kxx = dir_kernel.k(directions, directions)
    jitter = rel_jitter * np.mean(np.diag(kxx))
    kxx = kxx + jitter * np.eye(n)

    k_query = dir_kernel.k(directions, query_direction).reshape(-1)

    solved = np.linalg.solve(kxx, values)
    mean = float(k_query @ solved)

    solved_k = np.linalg.solve(kxx, k_query)
    variance = float(prior_variance - k_query @ solved_k)

    return BQResult(mean=mean, variance=max(variance, 0.0))


def bayesian_quadrature_nd(nodes, values, kernel: ProductKernel, bounds, rel_jitter: float = 1e-4) -> BQResult:
    """Same as `bayesian_quadrature`, generalized to a D-dimensional domain
    via a `ProductKernel` and a per-axis `bounds` list of (a_d, b_d) pairs.
    Kept as a separate function (rather than folding the 1D case into this
    one) so the already-tested 1D `bayesian_quadrature` path is untouched.
    """
    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim == 1:
        nodes = nodes.reshape(-1, 1)
    values = np.asarray(values, dtype=float).reshape(-1)
    n = nodes.shape[0]
    if n == 0:
        return BQResult(mean=0.0, variance=float(kernel.vv(bounds)))

    kxx = kernel.k(nodes, nodes)
    jitter = rel_jitter * np.mean(np.diag(kxx))
    kxx = kxx + jitter * np.eye(n)
    v = kernel.v(nodes, bounds).reshape(-1)

    solved = np.linalg.solve(kxx, values)
    mean = float(v @ solved)

    solved_v = np.linalg.solve(kxx, v)
    variance = float(kernel.vv(bounds) - v @ solved_v)

    return BQResult(mean=mean, variance=max(variance, 0.0))


def bayesian_quadrature(nodes, values, kernel: Kernel, a: float, b: float, rel_jitter: float = 1e-4) -> BQResult:
    """Posterior mean/variance of integral_a^b g(t) dt given g(nodes) = values.

    `rel_jitter` scales the Gram matrix diagonal (jitter = rel_jitter *
    mean(diag(K))) rather than adding a fixed absolute constant. Node
    placements here can be irregular enough to produce near-duplicate nodes
    (unlike the original repo's regularly-spaced ray samples), which drives
    the Gram matrix condition number past 1e18 with a fixed-scale jitter of
    1e-8 -- confirmed empirically for n=40 random uniform nodes at sigma=0.35.
    Splats can be similarly near-collocated in a real GS scene, so this is a
    real stability requirement, not just a toy-script wrinkle.
    """
    nodes = np.asarray(nodes, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float).reshape(-1)
    n = nodes.shape[0]
    if n == 0:
        return BQResult(mean=0.0, variance=float(kernel.vv(a, b)))

    kxx = kernel.k(nodes.reshape(-1, 1), nodes.reshape(1, -1))
    jitter = rel_jitter * np.mean(np.diag(kxx))
    kxx = kxx + jitter * np.eye(n)
    v = kernel.v(nodes, a, b).reshape(-1)

    solved = np.linalg.solve(kxx, values)
    mean = float(v @ solved)

    solved_v = np.linalg.solve(kxx, v)
    variance = float(kernel.vv(a, b) - v @ solved_v)

    return BQResult(mean=mean, variance=max(variance, 0.0))
