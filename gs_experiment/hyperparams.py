"""Fitting the kernel's bandwidth parameter (RBF sigma / Matern rho) to data
by maximizing the GP log marginal likelihood -- the standard way to "train"
a GP kernel hyperparameter (Rasmussen & Williams, GPML, ch. 5), and the
natural next step after finding (gs_experiment/results/FINDINGS.md) that a single
hardcoded bandwidth loses to a naive Riemann sum: an early toy sweep used
one fixed sigma/rho across scenes whose true bump widths ranged from 0.05 to
0.6. This module tests whether fitting the bandwidth per scene closes that
gap.

No torch/autodiff here -- this module stays pure numpy/scipy at this stage
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

from gs_experiment.kernels import Kernel, ProductKernel


def log_marginal_likelihood(nodes, values, kernel: Kernel, rel_jitter: float = 1e-4) -> float:
    """Standard zero-mean GP log marginal likelihood:
    -0.5 y^T K^-1 y - 0.5 log|K| - (n/2) log(2 pi).

    Uses a Cholesky factorization for both the quadratic form and the log
    determinant, and the same relative-jitter convention as
    gs_experiment.quadrature.bayesian_quadrature (see that module's docstring for
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
    unseen scenes -- which matters for deployment cost too: a bandwidth that needs
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


def log_marginal_likelihood_nd(
    nodes, values, kernel: ProductKernel, rel_jitter: float = 1e-4, noise_variance: float = 0.0
) -> float:
    """Same objective as `log_marginal_likelihood`, generalized to a
    `ProductKernel` over a D-dimensional domain (real splat positions, not
    a 1D ray-depth domain) -- what ROADMAP.md item 2 asks for: this module
    fit a bandwidth per *toy* scene, never against real splat data.

    Kept as a separate function rather than making `log_marginal_likelihood`
    branch on kernel type, the same separation-over-branching precedent
    `gs_experiment/quadrature.py` follows elsewhere, so the already-tested 1D
    path stays untouched.

    `nodes`: (N, D), used directly with `kernel.k(nodes, nodes)` -- unlike
    the 1D functions above, no `.reshape(-1, 1)` convention, since
    `ProductKernel.k` already expects (N, D) inputs.

    `noise_variance` (default 0.0): a real, homoscedastic observation-noise
    variance added to K's diagonal on top of (not instead of) `rel_jitter`'s
    numerical term -- the standard noisy-observation GP marginal likelihood,
    used to jointly fit it alongside the bandwidth (see
    `fit_kernel_param_and_noise_pooled_nd`) rather than leaving every splat
    color treated as an exact constraint. See
    `gs_experiment.quadrature._rendering_aware_moments`'s docstring for the
    full motivation (this is the same noise model, applied here to the
    fitting objective instead of the query-time posterior).
    """
    nodes = np.atleast_2d(np.asarray(nodes, dtype=float))
    values = np.asarray(values, dtype=float).reshape(-1)
    n = nodes.shape[0]
    if n == 0:
        return 0.0

    kxx = kernel.k(nodes, nodes)
    jitter = rel_jitter * np.mean(np.diag(kxx))
    kxx = kxx + (jitter + noise_variance) * np.eye(n)

    try:
        L = np.linalg.cholesky(kxx)
    except np.linalg.LinAlgError:
        return -np.inf

    alpha = np.linalg.solve(L.T, np.linalg.solve(L, values))
    quad_term = -0.5 * float(values @ alpha)
    logdet_term = -np.sum(np.log(np.diag(L)))
    const_term = -0.5 * n * np.log(2 * np.pi)
    return quad_term + logdet_term + const_term


def pooled_log_marginal_likelihood_nd(datasets, kernel: ProductKernel, noise_variance: float = 0.0) -> float:
    """ND analogue of `pooled_log_marginal_likelihood`: sum of
    `log_marginal_likelihood_nd` across many (positions, colors) local
    windows sampled from one (or several) real checkpoints, under one
    shared bandwidth -- the practical fitting objective at GS scale, since
    a single window rarely has enough points on its own to pin down a
    bandwidth precisely. `noise_variance`: see `log_marginal_likelihood_nd`."""
    total = 0.0
    for nodes, values in datasets:
        lml = log_marginal_likelihood_nd(nodes, values, kernel, noise_variance=noise_variance)
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


@dataclass
class NoisyFitResult:
    param: float
    noise_variance: float
    log_marginal_likelihood: float


def fit_kernel_param_and_noise_pooled_nd(
    datasets,
    kernel_factory: Callable[[float], ProductKernel],
    bounds=(1e-3, 5.0),
    noise_bounds=(1e-6, 1.0),
    n_grid: int = 15,
) -> NoisyFitResult:
    """Joint marginal-likelihood fit of the kernel bandwidth *and* a real
    homoscedastic observation-noise variance (see `log_marginal_likelihood_nd`'s
    `noise_variance` docstring for the model and motivation), instead of
    `fit_kernel_param_pooled_nd`'s bandwidth-only fit against an implicitly
    noiseless model. Two free parameters now, so this does a 2D log-spaced
    grid search (`n_grid` x `n_grid`, kept smaller than the 1D fitters'
    default since cost is quadratic in `n_grid`) followed by a local
    `scipy.optimize.minimize` (L-BFGS-B, bounded) refine from the best grid
    cell -- the natural 2D generalization of `_grid_search_then_refine`'s
    "grid pre-search, then local refine" discipline (a marginal-likelihood
    surface in a single lengthscale can already have more than one local
    optimum; two free parameters makes a pure local optimizer from one
    arbitrary starting point even less trustworthy on its own).

    `noise_bounds` defaults to a much smaller range than `bounds`: even a
    homoscedastic noise variance that is small relative to the kernel's own
    signal variance can meaningfully relax the near-duplicate-point
    ill-conditioning this exists to fix (see
    `gs_experiment.quadrature._rendering_aware_moments`) -- there is no a
    priori reason to search as wide a range for it as for the bandwidth
    itself.
    """
    log_lo, log_hi = np.log(bounds[0]), np.log(bounds[1])
    log_nlo, log_nhi = np.log(noise_bounds[0]), np.log(noise_bounds[1])
    param_grid = np.exp(np.linspace(log_lo, log_hi, n_grid))
    noise_grid = np.exp(np.linspace(log_nlo, log_nhi, n_grid))

    def objective(log_p, log_nv):
        p, nv = float(np.exp(log_p)), float(np.exp(log_nv))
        return pooled_log_marginal_likelihood_nd(datasets, kernel_factory(p), noise_variance=nv)

    best_val = -np.inf
    best_log_p, best_log_nv = np.log(param_grid[0]), np.log(noise_grid[0])
    for lp in np.log(param_grid):
        for lnv in np.log(noise_grid):
            val = objective(lp, lnv)
            if val > best_val:
                best_val, best_log_p, best_log_nv = val, lp, lnv

    result = optimize.minimize(
        lambda x: -objective(x[0], x[1]),
        x0=[best_log_p, best_log_nv],
        bounds=[(log_lo, log_hi), (log_nlo, log_nhi)],
        method="L-BFGS-B",
    )
    if -float(result.fun) >= best_val:
        final_log_p, final_log_nv, final_val = result.x[0], result.x[1], -float(result.fun)
    else:
        # local refine can, rarely, undershoot the grid optimum near a
        # boundary -- fall back to the grid point itself if so, same
        # convention as _grid_search_then_refine's 1D case.
        final_log_p, final_log_nv, final_val = best_log_p, best_log_nv, best_val

    return NoisyFitResult(
        param=float(np.exp(final_log_p)), noise_variance=float(np.exp(final_log_nv)), log_marginal_likelihood=final_val
    )
