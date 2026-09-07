"""Per-query-point local BQ uncertainty over a real (or mock) splat scene,
in 3D world space. Directly reuses bq_splat's validated kernel/quadrature
machinery and the two exact optimizations found in
bq_splat/validate.py --check scaling (bq_splat/results/FINDINGS.md
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
from bq_splat.quadrature import (
    BQResult,
    bayesian_quadrature_directional,
    bayesian_quadrature_nd,
    bayesian_quadrature_rendering_aware,
)
from bq_splat.render_weight import GaussianRenderWeight
from gs_experiment.camera import CameraPose, project_point_to_pixel, viewmat_from_camera_pose
from gs_experiment.visibility_attribution import project_to_camera_local, ray_transmittance_weights


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
    # Per-splat opacity, parallel to `positions`/`values` -- optional, only
    # needed for rendering_aware_variance's a_q amplitude (see that method).
    opacities: Optional[np.ndarray] = None
    # Per-splat scale (N, 3) and rotation quaternion (N, 4), parallel to
    # `positions` -- optional, only needed for
    # rendering_aware_variance_via_gsplat's real anisotropic footprint
    # (see that method); unused by every other method on this class.
    scales: Optional[np.ndarray] = None
    rotations: Optional[np.ndarray] = None
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
        if self.opacities is not None:
            self.opacities = np.asarray(self.opacities, dtype=float)
        if self.scales is not None:
            self.scales = np.asarray(self.scales, dtype=float)
        if self.rotations is not None:
            self.rotations = np.asarray(self.rotations, dtype=float)
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

    def local_neighbors(self, query_point: np.ndarray, radius: float, exclude_idx: Optional[int] = None) -> np.ndarray:
        idx = np.array(self.tree.query_ball_point(query_point, radius), dtype=int)
        if exclude_idx is not None:
            # for a leave-one-out calibration check (query at a real
            # splat's own position, exclude_idx=that splat's own index):
            # a ball query centered exactly on a splat always finds that
            # splat itself at distance 0, so without this the "prediction"
            # would trivially see its own held-out answer -- filtered
            # before the max_neighbors subsample below, not after, so
            # excluding self never reduces the *budget* of real neighbors
            # a capped query gets.
            idx = idx[idx != exclude_idx]
        if self.max_neighbors is not None and len(idx) > self.max_neighbors:
            idx = self._rng.choice(idx, size=self.max_neighbors, replace=False)
        return idx

    def spatial_only_variance(self, query_point: np.ndarray, radius: float, exclude_idx: Optional[int] = None) -> BQResult:
        idx = self.local_neighbors(query_point, radius, exclude_idx=exclude_idx)
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

    def rendering_aware_variance(
        self, query_point: np.ndarray, radius: float, exclude_idx: Optional[int] = None, sigma_rbf: Optional[float] = None
    ) -> BQResult:
        """spatial_only_variance's renderer-aware replacement: instead of
        integrating pos_kernel uniformly over an arbitrary box (see that
        method, and bq_splat/PROOF_alpha_compositing_equivalence.md section 7
        for why that's the thing this project's own docs already flag as
        unfinished), this builds a real per-query a_q
        (bq_splat/render_weight.py) from data this engine already has --
        real per-splat opacity as a_q's amplitude, and a Gaussian footprint
        tied to `radius` -- and calls bayesian_quadrature_rendering_aware
        instead of bayesian_quadrature_nd. A splat with low local opacity
        now contributes less to both mean and variance by construction,
        rather than by virtue of sitting outside an ad hoc box.

        What this does NOT model: genuine depth-ordered transmittance
        through occluders along one specific camera ray -- amplitude is a
        flat neighborhood-mean opacity, occlusion-blind. See
        rendering_aware_variance_along_ray below for the version that adds
        real per-ray occlusion ordering; this method stays as the simpler,
        camera-free baseline (no camera pose needed, useful when only a
        world-space query point is available).

        `sigma_rbf` defaults to `pos_kernel`'s own (shared, isotropic) RBF
        bandwidth, so the same lengthscale used for K is used for a_q's
        Gaussian envelope rather than an unrelated free parameter; only an
        RBF `pos_kernel` is supported (bayesian_quadrature_rendering_aware's
        closed form requires it -- pass sigma_rbf explicitly is not enough
        on its own since K itself is still built from `pos_kernel`, so a
        Matern pos_kernel would need the numerical mode, not wired up here).
        """
        if self.opacities is None:
            raise ValueError("opacities not set on this engine -- construct with opacities= to use this method")

        first_kernel = self.pos_kernel.kernels_per_axis[0]
        if not isinstance(first_kernel, RBFKernel):
            raise ValueError(
                "rendering_aware_variance requires an RBF pos_kernel (bayesian_quadrature_rendering_aware's "
                "closed form is RBF-only -- Matern would need its numerical mode, not wired up here)"
            )
        if sigma_rbf is None:
            sigma_rbf = first_kernel.sigma

        idx = self.local_neighbors(query_point, radius, exclude_idx=exclude_idx)
        local_positions = self.positions[idx]
        local_values = self.values[idx]

        amplitude = float(np.mean(self.opacities[idx])) if len(idx) > 0 else float(np.mean(self.opacities))
        d = np.asarray(query_point).shape[0]
        covariance = (radius / 2.0) ** 2 * np.eye(d)
        render_weight = GaussianRenderWeight(amplitude=amplitude, center=query_point, covariance=covariance)

        return bayesian_quadrature_rendering_aware(local_positions, local_values, render_weight, sigma_rbf=sigma_rbf)

    def rendering_aware_variance_along_ray(
        self,
        query_point: np.ndarray,
        camera: CameraPose,
        radius: float,
        exclude_idx: Optional[int] = None,
        angular_tol: float = 0.05,
        sigma_rbf: Optional[float] = None,
    ) -> BQResult:
        """rendering_aware_variance's occlusion-aware upgrade: a_q's shape
        is built from real, depth-ordered alpha-compositing transmittance
        weights along the specific ray from `camera` through `query_point`
        (gs_experiment.visibility_attribution.ray_transmittance_weights),
        instead of a flat neighborhood-mean opacity. A splat behind a
        closer, opaque splat *on this ray* now gets a small weight from
        real accumulated transmittance -- the gap rendering_aware_variance's
        own docstring names -- while a splat that's simply nearby in 3D but
        off to the side of this particular ray gets weight 0 regardless of
        its opacity, since it was never going to occlude or contribute to
        this pixel.

        Still not a live differentiable rasterizer: no antialiasing or
        sub-pixel footprint integration, no gradient path -- a real,
        geometric depth/occlusion proxy (project_to_camera_local + real
        per-splat opacity as alpha), not a claim of pixel-exact equivalence
        to an actual renderer.

        When every local neighbor is off-ray or behind the camera (a_q is
        identically 0 there), falls back to a near-zero amplitude rather
        than raising -- correctly reports "this query has ~nothing to see
        here": mean ~0, variance ~0 (the posterior collapses toward the
        near-vanishing prior, not toward a spurious nonzero guess).
        """
        if self.opacities is None:
            raise ValueError("opacities not set on this engine -- construct with opacities= to use this method")

        first_kernel = self.pos_kernel.kernels_per_axis[0]
        if not isinstance(first_kernel, RBFKernel):
            raise ValueError(
                "rendering_aware_variance_along_ray requires an RBF pos_kernel (bayesian_quadrature_rendering_aware's "
                "closed form is RBF-only -- Matern would need its numerical mode, not wired up here)"
            )
        if sigma_rbf is None:
            sigma_rbf = first_kernel.sigma

        query_point = np.asarray(query_point, dtype=float)
        d = query_point.shape[0]
        idx = self.local_neighbors(query_point, radius, exclude_idx=exclude_idx)
        local_positions = self.positions[idx]
        local_values = self.values[idx]

        if len(idx) == 0:
            weights = np.zeros(0)
        else:
            local_opacities = self.opacities[idx]
            ref_bearing_x, ref_bearing_y, _ = project_to_camera_local(query_point.reshape(1, -1), camera)
            reference_bearing = (float(ref_bearing_x[0]), float(ref_bearing_y[0]))
            weights = ray_transmittance_weights(
                local_positions, local_opacities, camera, reference_bearing, angular_tol=angular_tol
            )

        render_weight = self._render_weight_from_local_weights(local_positions, weights, query_point, radius, d)
        return bayesian_quadrature_rendering_aware(local_positions, local_values, render_weight, sigma_rbf=sigma_rbf)

    @staticmethod
    def _render_weight_from_local_weights(
        local_positions: np.ndarray, weights: np.ndarray, query_point: np.ndarray, radius: float, d: int
    ) -> GaussianRenderWeight:
        """Moment-match a set of nonnegative per-neighbor rendering weights
        (real alpha-compositing transmittance weights, from whichever
        source computed them) into a GaussianRenderWeight -- shared by
        rendering_aware_variance_along_ray and
        rendering_aware_variance_via_gsplat, which differ only in how
        `weights` itself is computed. Falls back to a near-zero-amplitude
        weight centered on `query_point` when every candidate has weight 0
        (nothing on-ray / nothing gsplat projects as visible here) --
        correctly reports "nothing to see" rather than dividing by zero.
        """
        fallback_covariance = (radius / 2.0) ** 2 * np.eye(d)
        total_weight = float(weights.sum()) if weights.size > 0 else 0.0
        if total_weight <= 0.0:
            return GaussianRenderWeight(amplitude=1e-12, center=query_point, covariance=fallback_covariance)

        mean = (weights[:, None] * local_positions).sum(axis=0) / total_weight
        diff = local_positions - mean
        covariance = (weights[:, None, None] * (diff[:, :, None] * diff[:, None, :])).sum(axis=0) / total_weight
        covariance = covariance + 1e-6 * np.eye(d)  # avoid a singular covariance from a single-splat weight mass
        return GaussianRenderWeight(amplitude=float(weights.max()), center=mean, covariance=covariance)

    def rendering_aware_variance_via_gsplat(
        self,
        query_point: np.ndarray,
        camera: CameraPose,
        K: np.ndarray,
        width: int,
        height: int,
        radius: float,
        exclude_idx: Optional[int] = None,
        sigma_rbf: Optional[float] = None,
        device: str = "cuda",
    ) -> BQResult:
        """rendering_aware_variance_along_ray's most faithful version: a_q's
        shape comes from gsplat's own differentiable EWA-splatting
        projection (gs_experiment.gsplat_rendering_weights, a real GPU
        computation) instead of an isotropic bearing-distance threshold --
        the real anisotropic 2D footprint (from each splat's actual scale
        and rotation) and real camera intrinsics `K` determine each local
        splat's per-pixel alpha, not a flat opacity gated by angular
        proximity. Needs torch + a CUDA-enabled gsplat build and a GPU
        (see ../requirements-gsplat.txt) -- imported lazily here so that
        this module and its default test suite stay importable without
        those installed.

        `K`: pinhole intrinsics (3, 3) for `camera`, e.g.
        gs_experiment.nerf_transforms.fov_x_to_intrinsics(fov_x_radians,
        width, height). `width`/`height`: the image `K` corresponds to.
        """
        if self.opacities is None or self.scales is None or self.rotations is None:
            raise ValueError(
                "opacities/scales/rotations not set on this engine -- construct with all three to use this method"
            )
        first_kernel = self.pos_kernel.kernels_per_axis[0]
        if not isinstance(first_kernel, RBFKernel):
            raise ValueError(
                "rendering_aware_variance_via_gsplat requires an RBF pos_kernel (bayesian_quadrature_rendering_aware's "
                "closed form is RBF-only -- Matern would need its numerical mode, not wired up here)"
            )
        if sigma_rbf is None:
            sigma_rbf = first_kernel.sigma

        from gs_experiment.gsplat_rendering_weights import gsplat_alpha_compositing_weights

        query_point = np.asarray(query_point, dtype=float)
        d = query_point.shape[0]
        idx = self.local_neighbors(query_point, radius, exclude_idx=exclude_idx)
        local_positions = self.positions[idx]
        local_values = self.values[idx]

        if len(idx) == 0:
            weights = np.zeros(0)
        else:
            pixel_xy = project_point_to_pixel(query_point, viewmat_from_camera_pose(camera), K)
            weights = gsplat_alpha_compositing_weights(
                local_positions, self.opacities[idx], self.scales[idx], self.rotations[idx],
                camera, K, width, height, pixel_xy, device=device,
            )

        render_weight = self._render_weight_from_local_weights(local_positions, weights, query_point, radius, d)
        return bayesian_quadrature_rendering_aware(local_positions, local_values, render_weight, sigma_rbf=sigma_rbf)
