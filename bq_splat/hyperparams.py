"""Fitting the kernel's bandwidth parameter (RBF sigma / Matern rho) to data
by maximizing the GP log marginal likelihood -- the standard way to "train"
a GP kernel hyperparameter (Rasmussen & Williams, GPML, ch. 5), and the
natural next step after finding (bq_splat/results/FINDINGS.md) that a single
hardcoded bandwidth loses to a naive Riemann sum: the original models/nerf.py
hardcodes `sig=0.25` and never adapts it, and bq_splat's own toy sweep used
one fixed sigma/rho across scenes whose true bump widths ranged from 0.05 to
0.6. This module tests whether fitting the bandwidth per scene closes that
gap.

No torch/autodiff here -- `bq_splat` stays pure numpy/scipy at this stage
(see ROADMAP.md). If/when this idea moves into the gsplat-integrated code,
the same quantity becomes a literal torch.nn.Parameter optimized jointly
with the rest of the pipeline; here it's a 1D scalar fit via scipy, which is
exactly what GP libraries (sklearn, GPy, GPflow) do for kernel hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import optimize

from bq_splat.kernels import Kernel, ProductKernel


def log_marginal_likelihood(nodes, values, kernel: Kernel, rel_jitter: float = 1e-4) -> float:
    """Standard zero-mean GP log marginal likelihood:
    -0.5 y^T K^-1 y - 0.5 log|K| - (n/2) log(2 pi).

    Uses a Cholesky factorization for both the quadratic form and the log
    determinant, and the same relative-jitter convention as
    bq_splat.quadrature.bayesian_quadrature (see that module's docstring for
    why a fixed jitter isn't safe with irregular node spacing).
    """
    nodes = np.asarray(nodes, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float).reshape(-1)
    n = nodes.shape[0]
    if n == 0:
        return 0.0

    kxx = kernel.k(nodes.reshape(-1, 1), nodes.reshape(1, -1))
    jitter = rel_jitter * np.mean(np.diag(kxx))
    kxx = kxx + jitter * np.eye(n)

    try:
        L = np.linalg.cholesky(kxx)
    except np.linalg.LinAlgError:
        return -np.inf

    alpha = np.linalg.solve(L.T, np.linalg.solve(L, values))
    quad_term = -0.5 * float(values @ alpha)
    logdet_term = -np.sum(np.log(np.diag(L)))  # -0.5 * log|K| = -sum(log(diag(L)))
    const_term = -0.5 * n * np.log(2 * np.pi)
    return quad_term + logdet_term + const_term


@dataclass
class FitResult:
    param: float
    log_marginal_likelihood: float


def pooled_log_marginal_likelihood(datasets, kernel: Kernel) -> float:
    """Sum of `log_marginal_likelihood` across multiple (nodes, values)
    pairs under one shared kernel -- the objective for fitting a single
    bandwidth across many scenes/regions rather than one per scene. Used to
    test whether a bandwidth fit once on a calibration set generalizes to
    unseen scenes (see scripts/validate_trainable_kernel_heldout.py) --
    which matters for deployment cost too: a bandwidth that needs
    refitting per query is a very different computational proposition at
    GS scale than one fit once and reused.
    """
    total = 0.0
    for nodes, values in datasets:
        lml = log_marginal_likelihood(nodes, values, kernel)
        if not np.isfinite(lml):
            return -np.inf
        total += lml
    return total


def _grid_search_then_refine(objective: Callable[[float], float], bounds, n_grid: int) -> FitResult:
    lo, hi = bounds
    grid = np.geomspace(lo, hi, n_grid)
    values = np.array([objective(p) for p in grid])
    best_idx = int(np.argmax(values))

    refine_lo = grid[max(best_idx - 1, 0)]
    refine_hi = grid[min(best_idx + 1, n_grid - 1)]

    result = optimize.minimize_scalar(lambda p: -objective(p), bounds=(refine_lo, refine_hi), method="bounded")
    best_param = float(result.x)
    best_value = -float(result.fun)

    if best_value < values[best_idx]:
        # bounded line search can, rarely, undershoot the grid optimum near
        # a boundary -- fall back to the grid point itself if so.
        return FitResult(param=float(grid[best_idx]), log_marginal_likelihood=float(values[best_idx]))

    return FitResult(param=best_param, log_marginal_likelihood=best_value)


def fit_kernel_param(
    nodes,
    values,
    kernel_factory: Callable[[float], Kernel],
    bounds=(1e-3, 5.0),
    n_grid: int = 25,
) -> FitResult:
    """Find the bandwidth maximizing the log marginal likelihood for a
    single (nodes, values) dataset.

    `kernel_factory` maps a scalar bandwidth to a Kernel instance, e.g.
    `lambda sig: RBFKernel(sigma=sig)`. Does a log-spaced grid pre-search
    (log marginal likelihood surfaces in a single lengthscale can have more
    than one local optimum) and then refines the best grid point with a
    bounded 1D line search, rather than trusting a single local optimizer
    call from one starting point.
    """
    objective = lambda p: log_marginal_likelihood(nodes, values, kernel_factory(p))
    return _grid_search_then_refine(objective, bounds, n_grid)


def fit_kernel_param_pooled(
    datasets,
    kernel_factory: Callable[[float], Kernel],
    bounds=(1e-3, 5.0),
    n_grid: int = 25,
) -> FitResult:
    """Same procedure as `fit_kernel_param`, but maximizing
    `pooled_log_marginal_likelihood` across many (nodes, values) datasets
    under one shared bandwidth."""
    objective = lambda p: pooled_log_marginal_likelihood(datasets, kernel_factory(p))
    return _grid_search_then_refine(objective, bounds, n_grid)


def log_marginal_likelihood_nd(nodes, values, kernel: ProductKernel, rel_jitter: float = 1e-4) -> float:
    """Same objective as `log_marginal_likelihood`, generalized to a
    `ProductKernel` over a D-dimensional domain (real splat positions, not
    a 1D ray-depth domain) -- what ROADMAP.md item 2 asks for: this module
    fit a bandwidth per *toy* scene, never against real splat data.

    Kept as a separate function rather than making `log_marginal_likelihood`
    branch on kernel type, mirroring how `bayesian_quadrature_nd` was kept
    separate from `bayesian_quadrature` (bq_splat/quadrature.py) so the
    already-tested 1D path stays untouched.

    `nodes`: (N, D), used directly with `kernel.k(nodes, nodes)` -- unlike
    the 1D functions above, no `.reshape(-1, 1)` convention, since
    `ProductKernel.k` already expects (N, D) inputs.
    """
    nodes = np.atleast_2d(np.asarray(nodes, dtype=float))
    values = np.asarray(values, dtype=float).reshape(-1)
    n = nodes.shape[0]
    if n == 0:
        return 0.0

    kxx = kernel.k(nodes, nodes)
    jitter = rel_jitter * np.mean(np.diag(kxx))
    kxx = kxx + jitter * np.eye(n)

    try:
        L = np.linalg.cholesky(kxx)
    except np.linalg.LinAlgError:
        return -np.inf

    alpha = np.linalg.solve(L.T, np.linalg.solve(L, values))
    quad_term = -0.5 * float(values @ alpha)
    logdet_term = -np.sum(np.log(np.diag(L)))
    const_term = -0.5 * n * np.log(2 * np.pi)
    return quad_term + logdet_term + const_term


def pooled_log_marginal_likelihood_nd(datasets, kernel: ProductKernel) -> float:
    """ND analogue of `pooled_log_marginal_likelihood`: sum of
    `log_marginal_likelihood_nd` across many (positions, colors) local
    windows sampled from one (or several) real checkpoints, under one
    shared bandwidth -- the practical fitting objective at GS scale, since
    a single window rarely has enough points on its own to pin down a
    bandwidth precisely."""
    total = 0.0
    for nodes, values in datasets:
        lml = log_marginal_likelihood_nd(nodes, values, kernel)
        if not np.isfinite(lml):
            return -np.inf
        total += lml
    return total


def fit_kernel_param_pooled_nd(
    datasets,
    kernel_factory: Callable[[float], ProductKernel],
    bounds=(1e-3, 5.0),
    n_grid: int = 25,
) -> FitResult:
    """ND analogue of `fit_kernel_param_pooled`: fit one shared bandwidth
    across many local (positions, colors) windows via a `ProductKernel`.
    `kernel_factory` maps a scalar bandwidth to e.g.
    `ProductKernel([RBFKernel(sig)] * 3)`."""
    objective = lambda p: pooled_log_marginal_likelihood_nd(datasets, kernel_factory(p))
    return _grid_search_then_refine(objective, bounds, n_grid)
