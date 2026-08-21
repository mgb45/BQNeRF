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

from bq_splat.kernels import Kernel


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


def fit_kernel_param(
    nodes,
    values,
    kernel_factory: Callable[[float], Kernel],
    bounds=(1e-3, 5.0),
    n_grid: int = 25,
) -> FitResult:
    """Find the bandwidth maximizing the log marginal likelihood.

    `kernel_factory` maps a scalar bandwidth to a Kernel instance, e.g.
    `lambda sig: RBFKernel(sigma=sig)`. Does a log-spaced grid pre-search
    (log marginal likelihood surfaces in a single lengthscale can have more
    than one local optimum) and then refines the best grid point with a
    bounded 1D line search, rather than trusting a single local optimizer
    call from one starting point.
    """
    lo, hi = bounds
    grid = np.geomspace(lo, hi, n_grid)
    lmls = np.array([log_marginal_likelihood(nodes, values, kernel_factory(p)) for p in grid])
    best_idx = int(np.argmax(lmls))

    refine_lo = grid[max(best_idx - 1, 0)]
    refine_hi = grid[min(best_idx + 1, n_grid - 1)]

    def neg_lml(p):
        return -log_marginal_likelihood(nodes, values, kernel_factory(p))

    result = optimize.minimize_scalar(neg_lml, bounds=(refine_lo, refine_hi), method="bounded")
    best_param = float(result.x)
    best_lml = -float(result.fun)

    if best_lml < lmls[best_idx]:
        # bounded line search can, rarely, undershoot the grid optimum near
        # a boundary -- fall back to the grid point itself if so.
        return FitResult(param=float(grid[best_idx]), log_marginal_likelihood=float(lmls[best_idx]))

    return FitResult(param=best_param, log_marginal_likelihood=best_lml)
