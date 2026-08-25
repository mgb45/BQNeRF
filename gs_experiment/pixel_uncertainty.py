"""Per-query-point local BQ uncertainty over a real (or mock) splat scene,
in 3D world space. Directly reuses bq_splat's validated kernel/quadrature
machinery and the two exact optimizations found in
scripts/benchmark_local_bq_scaling.py (bq_splat/results/FINDINGS.md
section 8): a KD-tree for neighbor lookup instead of brute force, and
caching the kernel's `vv` term per window size instead of recomputing it
per query (exact for a stationary kernel on a fixed-size, translated
window, not an approximation).

Neighbor-finding uses a ball query (efficient via scipy's cKDTree); the
integration domain for `v`/`vv` is then the axis-aligned bounding box of
that same nominal radius, matching what benchmark_local_bq_scaling.py
already validated (ball query for speed, box for the actual quadrature
domain) rather than a novel choice made here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from bq_splat.kernels import DirectionalKernel, MaternKernel, ProductKernel, RBFKernel
from bq_splat.quadrature import BQResult, bayesian_quadrature_directional, bayesian_quadrature_nd


def make_default_3d_position_kernel(sigma: float) -> ProductKernel:
    """3D generalization of the 2D ProductKernel used throughout bq_splat --
    no new kernel code needed, ProductKernel already supports arbitrary D."""
    return ProductKernel([RBFKernel(sigma=sigma), RBFKernel(sigma=sigma), RBFKernel(sigma=sigma)])


def make_default_3d_matern_kernel(rho: float) -> ProductKernel:
    """Matern-3/2 analogue of make_default_3d_position_kernel, for the
    RBF-vs-Matern kernel-choice comparison ROADMAP.md flags as unresolved
    (bq_splat/results/FINDINGS.md sections 5-7 validated it only at toy
    scale, never against a real trained GS checkpoint). `rho` plays the
    same "bandwidth" role sigma does for RBF, not an identical physical
    quantity -- passing the same numeric value to both is a like-for-like
    comparison of "same nominal length scale, different smoothness
    assumption," not a claim the two parameters are interchangeable in
    general.
    """
    return ProductKernel([MaternKernel(rho=rho), MaternKernel(rho=rho), MaternKernel(rho=rho)])


def box_bounds(center: np.ndarray, radius: float, scene_bounds) -> list:
    bounds = []
    for d in range(len(center)):
        lo, hi = scene_bounds[d]
        bounds.append((max(lo, center[d] - radius), min(hi, center[d] + radius)))
    return bounds


@dataclass
class LocalUncertaintyEngine:
    """Holds the spatial index and a vv-cache across many queries against
    one fixed splat scene -- building the KD-tree once and reusing the
    cache is the whole point of benchmark_local_bq_scaling.py's findings;
    a fresh engine per query would throw that away.
    """

    positions: np.ndarray
    values: np.ndarray
    pos_kernel: ProductKernel
    scene_bounds: Tuple[Tuple[float, float], ...]
    directions: Optional[np.ndarray] = None
    dir_kernel: Optional[DirectionalKernel] = None
    # Real gsplat checkpoints can pack thousands of (splat, camera)
    # observation rows into one query's window (found the hard way: an
    # angular_tol loosened enough for real occlusion attribution to work
    # sensibly on a real densely-packed scene let >10k rows into a single
    # window, and the BQ solve below is at least O(n^2)-O(n^3) in neighbor
    # count -- one such query pegged ~18 CPU cores for half an hour before
    # being killed). benchmark_local_bq_scaling.py
    # (bq_splat/results/FINDINGS.md section 8) validated the solve cost as
    # negligible up to "hundreds" of local neighbors, never thousands+, so
    # capping there rather than letting window contents grow unbounded
    # with real-data density is restoring the validated regime, not an ad
    # hoc shortcut. None disables the cap (the toy/mock-scene regime this
    # class was originally validated in stays exactly as before).
    max_neighbors: Optional[int] = 400
    seed: int = 0

    def __post_init__(self):
        self.positions = np.asarray(self.positions, dtype=float)
        self.values = np.asarray(self.values, dtype=float)
        self.tree = cKDTree(self.positions)
        self._vv_cache: Dict[tuple, float] = {}
        self._rng = np.random.default_rng(self.seed)

    def _cached_vv(self, bounds) -> float:
        # Interior queries share one window size (2*radius per axis); edge
        # queries clipped by scene_bounds get their own (smaller) shape --
        # still cached, just under a different key, per benchmark_local_bq_
        # scaling.py's caveat about clipped windows near boundaries.
        key = tuple(round(hi - lo, 9) for lo, hi in bounds)
        if key not in self._vv_cache:
            self._vv_cache[key] = float(self.pos_kernel.vv(bounds))
        return self._vv_cache[key]

    def local_neighbors(self, query_point: np.ndarray, radius: float) -> np.ndarray:
        idx = np.array(self.tree.query_ball_point(query_point, radius), dtype=int)
        if self.max_neighbors is not None and len(idx) > self.max_neighbors:
            idx = self._rng.choice(idx, size=self.max_neighbors, replace=False)
        return idx

    def spatial_only_variance(self, query_point: np.ndarray, radius: float) -> BQResult:
        idx = self.local_neighbors(query_point, radius)
        bounds = box_bounds(query_point, radius, self.scene_bounds)
        vv = self._cached_vv(bounds)
        local_positions = self.positions[idx]
        local_values = self.values[idx]
        return bayesian_quadrature_nd(local_positions, local_values, self.pos_kernel, bounds, precomputed_vv=vv)

    def directional_variance(self, query_point: np.ndarray, query_direction: np.ndarray, radius: float) -> BQResult:
        if self.directions is None or self.dir_kernel is None:
            raise ValueError("directions/dir_kernel not set on this engine -- construct with both to use this method")
        idx = self.local_neighbors(query_point, radius)
        bounds = box_bounds(query_point, radius, self.scene_bounds)
        vv = self._cached_vv(bounds)
        local_positions = self.positions[idx]
        local_directions = self.directions[idx]
        local_values = self.values[idx]
        return bayesian_quadrature_directional(
            local_positions, local_directions, local_values, self.pos_kernel, self.dir_kernel, bounds, query_direction,
            precomputed_pos_vv=vv,
        )
