"""Synthetic 1D ray/pixel signals and node-placement strategies, plus their
2D image-plane analogues (merged in from the former toy_scene_2d.py).

`g_true` stands in for the continuous rendering integrand along a ray (the
thing splats or NeRF samples are noisy/sparse point evaluations of). Node
placement strategies mimic how splat coverage can be irregular in a way
uniform NeRF stratified sampling never was. The 2D variants scatter splat
centers over an image patch instead of along a ray's depth axis -- closer
to real GS geometry, where compositing happens per-pixel over whichever
splats' footprints overlap that pixel in the image plane.
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


# ---------------------------------------------------------------------------
# 2D image-plane analogue (formerly bq_splat/toy_scene_2d.py, merged here):
# splat centers scattered over an image patch (not along a ray's depth
# axis), and a continuous "true" 2D color function they're meant to
# represent. This is the closer-to-real-GS geometry: in actual 3DGS,
# compositing happens per-pixel over whichever splats' anisotropic
# footprints overlap that pixel in the image plane, not along a 1D depth
# integral.
# ---------------------------------------------------------------------------


@dataclass
class ToyScene2D:
    domain: Tuple[Tuple[float, float], Tuple[float, float]]  # ((x0,x1), (y0,y1))
    g_true: Callable[[np.ndarray], np.ndarray]  # (N,2) -> (N,)
    description: str


def make_mixture_scene_2d(rng: np.random.Generator, domain=((0.0, 10.0), (0.0, 10.0)), n_bumps=8, min_width=0.15, max_width=1.0) -> ToyScene2D:
    """g_true is a mixture of 2D Gaussian bumps (isotropic per-bump, varying
    widths) -- stands in for a continuous image/radiance function a finite
    set of splats is meant to reconstruct."""
    (x0, x1), (y0, y1) = domain
    centers = np.stack([rng.uniform(x0, x1, size=n_bumps), rng.uniform(y0, y1, size=n_bumps)], axis=1)
    widths = rng.uniform(min_width, max_width, size=n_bumps)
    heights = rng.uniform(0.3, 1.0, size=n_bumps)

    def g_true(points):
        points = np.atleast_2d(np.asarray(points, dtype=float))
        out = np.zeros(points.shape[0])
        for c, w, h in zip(centers, widths, heights):
            d2 = np.sum((points - c) ** 2, axis=1)
            out = out + h * np.exp(-0.5 * d2 / w**2)
        return out

    return ToyScene2D(domain=domain, g_true=g_true, description=f"{n_bumps} 2D bumps, widths in [{min_width}, {max_width}]")


def uniform_nodes_2d(rng: np.random.Generator, domain, n) -> np.ndarray:
    (x0, x1), (y0, y1) = domain
    return np.stack([rng.uniform(x0, x1, size=n), rng.uniform(y0, y1, size=n)], axis=1)


def gap_nodes_2d(rng: np.random.Generator, domain, n, gap_center_frac=(0.5, 0.5), gap_radius_frac=0.15, thin_prob=0.9):
    """Uniform-ish coverage over the image patch except a deliberate circular
    gap of sparse coverage placed in the interior -- a region that's fully
    "visible" (interior to the patch) but under-sampled by splat centers.
    Returns (nodes, (gap_center, gap_radius))."""
    (x0, x1), (y0, y1) = domain
    span_x, span_y = x1 - x0, y1 - y0
    gap_center = np.array([x0 + gap_center_frac[0] * span_x, y0 + gap_center_frac[1] * span_y])
    gap_radius = gap_radius_frac * min(span_x, span_y)

    nodes = []
    attempts = 0
    while len(nodes) < n and attempts < 400 * n:
        attempts += 1
        p = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
        if np.linalg.norm(p - gap_center) < gap_radius and rng.random() < thin_prob:
            continue
        nodes.append(p)
    return np.array(nodes), (gap_center, gap_radius)
