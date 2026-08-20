"""Synthetic 1D ray/pixel signals and node-placement strategies.

`g_true` stands in for the continuous rendering integrand along a ray (the
thing splats or NeRF samples are noisy/sparse point evaluations of). Node
placement strategies mimic how splat coverage can be irregular in a way
uniform NeRF stratified sampling never was.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np


@dataclass
class ToyScene:
    domain: Tuple[float, float]
    g_true: Callable[[np.ndarray], np.ndarray]
    description: str


def make_mixture_scene(rng: np.random.Generator, domain=(0.0, 10.0), n_bumps=5, min_width=0.05, max_width=0.6) -> ToyScene:
    """g_true is a mixture of Gaussian bumps of varying width -- varying
    `min_width`/`max_width` controls how "high-frequency" the signal is."""
    a, b = domain
    centers = rng.uniform(a, b, size=n_bumps)
    widths = rng.uniform(min_width, max_width, size=n_bumps)
    heights = rng.uniform(0.3, 1.0, size=n_bumps)

    def g_true(t):
        t = np.asarray(t, dtype=float)
        out = np.zeros_like(t)
        for c, w, h in zip(centers, widths, heights):
            out = out + h * np.exp(-0.5 * ((t - c) / w) ** 2)
        return out

    return ToyScene(domain=domain, g_true=g_true, description=f"{n_bumps} bumps, widths in [{min_width}, {max_width}]")


def uniform_nodes(rng: np.random.Generator, domain, n) -> np.ndarray:
    a, b = domain
    return np.sort(rng.uniform(a, b, size=n))


def gap_nodes(rng: np.random.Generator, domain, n, gap_center_frac=0.5, gap_width_frac=0.15, thin_prob=0.9):
    """Uniform-ish coverage everywhere except a deliberate gap of sparse
    coverage placed inside the domain (not at the edges) -- a region that is
    fully "visible" (interior to [a, b], not occluded) but under-sampled.
    Returns (nodes, (gap_lo, gap_hi))."""
    a, b = domain
    span = b - a
    gap_center = a + gap_center_frac * span
    gap_half = 0.5 * gap_width_frac * span
    gap = (gap_center - gap_half, gap_center + gap_half)

    nodes = []
    attempts = 0
    while len(nodes) < n and attempts < 200 * n:
        attempts += 1
        t = rng.uniform(a, b)
        if gap[0] < t < gap[1] and rng.random() < thin_prob:
            continue
        nodes.append(t)
    return np.sort(np.array(nodes)), gap
