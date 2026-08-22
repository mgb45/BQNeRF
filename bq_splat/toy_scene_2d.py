"""2D image-plane analogue of toy_scene.py: splat centers scattered over an
image patch (not along a ray's depth axis), and a continuous "true" 2D color
function they're meant to represent. This is the closer-to-real-GS geometry:
in actual 3DGS, compositing happens per-pixel over whichever splats'
anisotropic footprints overlap that pixel in the image plane, not along a
1D depth integral.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np


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
