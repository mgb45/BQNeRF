"""Reference estimators: the naive discrete (piecewise-constant / "alpha
compositing style") Riemann sum, and the numerically-exact ground-truth
integral, both used as baselines to check Bayesian quadrature against.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import integrate


def riemann_estimate(nodes, values, a: float, b: float) -> float:
    """Piecewise-constant Riemann sum from irregularly spaced samples: each
    node is assigned a "slab" of the domain out to the midpoint of its
    neighbors (clipped to [a, b]), exactly generalizing how NeRF/GS assign a
    bin width `dists[i]` to each ray sample."""
    nodes = np.asarray(nodes, dtype=float)
    values = np.asarray(values, dtype=float)
    order = np.argsort(nodes)
    t = nodes[order]
    g = values[order]
    edges = np.concatenate([[a], 0.5 * (t[1:] + t[:-1]), [b]])
    widths = np.diff(edges)
    return float(np.sum(widths * g))


def true_integral(g_true: Callable[[np.ndarray], np.ndarray], a: float, b: float) -> float:
    val, _ = integrate.quad(lambda t: float(g_true(np.array([t]))[0]), a, b, limit=200)
    return val
