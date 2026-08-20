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

from bq_splat.kernels import Kernel


@dataclass
class BQResult:
    mean: float
    variance: float


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
