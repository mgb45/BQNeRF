"""Per-query-point local, rendering-aware BQ uncertainty over a real (or
mock) splat scene, in 3D world space. Directly reuses this project's
validated kernel/quadrature machinery (`kernels.py`/`quadrature.py`),
plus a KD-tree-based neighbor lookup.

`LocalUncertaintyEngine.rendering_aware_alpha_risk_along_ray` builds a
query-specific renderer weight (`render_weight.py`'s GaussianRenderWeight:
transmittance x opacity x footprint) from the real, depth-ordered
alpha-compositing weights along one camera ray, then scores those real
weights' own RKHS risk under a position-only kernel -- this is
u_spatial_BQ(q) in the renderer-consistent sparse-GP decomposition (see
gs_experiment/sh_directional_uncertainty.py's module docstring for the
full picture).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from gs_experiment.kernels import ProductKernel, RBFKernel
from gs_experiment.quadrature import BQResult, bayesian_quadrature_rendering_aware, rendering_aware_alternative_weight_risk
from gs_experiment.render_weight import GaussianRenderWeight
from gs_experiment.camera import CameraPose
from gs_experiment.visibility_attribution import CameraSplatIndex, project_to_camera_local, ray_transmittance_weights


def make_default_3d_position_kernel(sigma: float) -> ProductKernel:
    """3D generalization of the 2D ProductKernel used elsewhere in this
    project -- no new kernel code needed, ProductKernel already supports
    arbitrary D."""
    return ProductKernel([RBFKernel(sigma=sigma), RBFKernel(sigma=sigma), RBFKernel(sigma=sigma)])


def quat_scale_to_covariance(quats: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Real 3D covariance Sigma_i = R_i diag(scale_i^2) R_i^T for each
    (quaternion, scale) pair -- quats as (w, x, y, z), the convention
    gsplat.quat_scale_to_covar_preci uses and this repo's identity
    quaternion [1, 0, 0, 0] already assumes.

    `quats`/`scales`: (N, 4) / (N, 3). Returns (N, 3, 3).
    """
    quats = np.asarray(quats, dtype=float)
    scales = np.asarray(scales, dtype=float)
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    n = quats.shape[0]
    rotation = np.empty((n, 3, 3))
    rotation[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    rotation[:, 0, 1] = 2 * (x * y - w * z)
    rotation[:, 0, 2] = 2 * (x * z + w * y)
    rotation[:, 1, 0] = 2 * (x * y + w * z)
    rotation[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    rotation[:, 1, 2] = 2 * (y * z - w * x)
    rotation[:, 2, 0] = 2 * (x * z - w * y)
    rotation[:, 2, 1] = 2 * (y * z + w * x)
    rotation[:, 2, 2] = 1 - 2 * (x**2 + y**2)
    scales_sq = scales**2  # (N, 3)
    return np.einsum("nik,nk,njk->nij", rotation, scales_sq, rotation)


@dataclass
class RenderingAwareAlphaRisk:
    """`LocalUncertaintyEngine.rendering_aware_alpha_risk_along_ray`'s
    result: the usual BQ-optimal (`mean`, `variance`) pair alongside the
    real local alpha-compositing quadrature rule's own (`alpha_mean`,
    `alpha_risk`) under the identical kernel/z/K/z0 geometry -- see that
    method's docstring and `gs_experiment.quadrature.
    rendering_aware_alternative_weight_risk` for what each quantity means.
    `alpha_risk` is u_spatial_BQ(q).
    """

    mean: float
    variance: float
    alpha_mean: float
    alpha_risk: float


@dataclass
class LocalUncertaintyEngine:
    """Holds the spatial index across many queries against one fixed splat
    scene -- building the KD-tree once and reusing it is the point."""

    positions: np.ndarray
    values: np.ndarray
    pos_kernel: ProductKernel
    scene_bounds: Tuple[Tuple[float, float], ...]
    # Per-splat opacity, parallel to `positions`/`values` -- needed for
    # every rendering_aware_* method's real a_q.
    opacities: Optional[np.ndarray] = None
    # Per-splat scale (N, 3) and rotation quaternion (N, 4), parallel to
    # `positions` -- optional, only needed for `covariances()`'s real
    # anisotropic per-candidate footprint.
    scales: Optional[np.ndarray] = None
    rotations: Optional[np.ndarray] = None
    # Real gsplat checkpoints can pack thousands of (splat, camera)
    # observation rows into one query's window (found the hard way: an
    # angular_tol loosened enough for real occlusion attribution to work
    # sensibly on a real densely-packed scene let >10k rows into a single
    # window, and the BQ solve below is at least O(n^2)-O(n^3) in neighbor
    # count -- one such query pegged ~18 CPU cores for half an hour before
    # being killed). Capping here rather than letting window contents grow
    # unbounded with real-data density is restoring a validated regime, not
    # an ad hoc shortcut. None disables the cap.
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
        self._covariance_cache: Optional[np.ndarray] = None
        self._rng = np.random.default_rng(self.seed)

    def covariances(self) -> np.ndarray:
        """Every splat's real 3D covariance (from its scale/rotation),
        computed once and cached -- `rendering_aware_alpha_risk_along_ray`
        just indexes into this (`covariances()[idx]`) instead of
        recomputing it fresh per query point, which used to mean one extra
        computation per query on top of the already-fixed candidate-
        selection cost (confirmed to matter: real-checkpoint queries
        dropped from ~215ms/point to single-digit ms/point once this was
        cached instead)."""
        if self.scales is None or self.rotations is None:
            raise ValueError("scales/rotations not set on this engine -- construct with both to use this method")
        if self._covariance_cache is None:
            self._covariance_cache = quat_scale_to_covariance(self.rotations, self.scales)
        return self._covariance_cache

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

    def build_bearing_index(self, camera: CameraPose) -> CameraSplatIndex:
        """Build once per camera, reuse across every
        `rendering_aware_alpha_risk_along_ray` call against that camera --
        see CameraSplatIndex's docstring for why a fresh 3D-ball-query
        candidate set per call is both wrong and wasteful on a real, dense
        checkpoint."""
        return CameraSplatIndex.build(self.positions, camera)

    def rendering_aware_alpha_risk_along_ray(
        self,
        query_point: np.ndarray,
        camera_index: CameraSplatIndex,
        radius: float,
        exclude_idx: Optional[int] = None,
        angular_tol: float = 0.05,
        max_candidates: int = 500,
        sigma_rbf: Optional[float] = None,
        noise_variance: float = 0.0,
    ) -> "RenderingAwareAlphaRisk":
        """Real, depth-ordered alpha-compositing transmittance weights
        along the specific ray through `query_point`
        (gs_experiment.visibility_attribution.ray_transmittance_weights)
        build a query-specific a_q (position-only, occlusion-aware), then
        `alpha_risk` scores those SAME real weights' own RKHS worst-case
        risk under that kernel (`rendering_aware_alternative_weight_risk`)
        -- this is u_spatial_BQ(q), the finite-spatial-representation term
        of the renderer-consistent sparse-GP decomposition (see
        gs_experiment/sh_directional_uncertainty.py). `mean`/`variance` is
        the separate BQ-*optimal* estimator under the same kernel, reported
        alongside for reference -- the sparse-GP decomposition's own mean
        is the real renderer output, not this.

        `camera_index`: a CameraSplatIndex built once via
        `build_bearing_index(camera)` and reused across every query against
        that camera -- candidates come from real bearing-space proximity,
        not a 3D-world-space ball query (a 3D radius wide enough to
        reliably contain the truly relevant splat can pull in a large
        fraction of the *entire* scene on a real, densely-packed
        checkpoint).

        `alpha_mean`/`alpha_risk` are local to this call's own candidate
        window (`radius`/`angular_tol`/`max_candidates`), so `alpha_mean`
        is *not* guaranteed to bit-match a full scene renderer's actual
        per-pixel output (which sees every splat along the ray, not just
        this window's candidates) -- compare against a real rendered pixel
        value directly if that gap matters for a given use, don't assume
        equality.

        `noise_variance` (default 0.0, unchanged behavior): a real
        homoscedastic observation-noise variance added to the underlying
        Gram matrix's diagonal -- see
        `gs_experiment.quadrature._rendering_aware_moments`'s docstring.
        """
        query_point = np.asarray(query_point, dtype=float)
        sigma_rbf = self._require_rbf_sigma("rendering_aware_alpha_risk_along_ray", sigma_rbf)
        idx, local_positions, local_values, render_weight, alpha_weights = self._along_ray_local_data(
            query_point, camera_index, radius, exclude_idx, angular_tol, max_candidates
        )
        bq_result = bayesian_quadrature_rendering_aware(
            local_positions, local_values, render_weight, sigma_rbf=sigma_rbf, noise_variance=noise_variance
        )
        alpha_mean, alpha_risk = rendering_aware_alternative_weight_risk(
            local_positions, local_values, render_weight, alpha_weights, sigma_rbf=sigma_rbf, noise_variance=noise_variance
        )
        return RenderingAwareAlphaRisk(
            mean=bq_result.mean, variance=bq_result.variance, alpha_mean=alpha_mean, alpha_risk=alpha_risk,
        )

    def _require_rbf_sigma(self, method_name: str, sigma_rbf: Optional[float]) -> float:
        first_kernel = self.pos_kernel.kernels_per_axis[0]
        if not isinstance(first_kernel, RBFKernel):
            raise ValueError(f"{method_name} requires an RBF pos_kernel (this closed form is RBF-only)")
        return first_kernel.sigma if sigma_rbf is None else sigma_rbf

    def _along_ray_local_data(
        self,
        query_point: np.ndarray,
        camera_index: CameraSplatIndex,
        radius: float,
        exclude_idx: Optional[int],
        angular_tol: float,
        max_candidates: Optional[int],
    ):
        """Shared candidate-gathering + render_weight construction for
        `rendering_aware_alpha_risk_along_ray`."""
        if self.opacities is None:
            raise ValueError("opacities not set on this engine -- construct with opacities= to use this method")

        d = query_point.shape[0]
        ref_bearing_x, ref_bearing_y, _ = project_to_camera_local(query_point.reshape(1, -1), camera_index.camera)
        reference_bearing = (float(ref_bearing_x[0]), float(ref_bearing_y[0]))
        idx = camera_index.query(reference_bearing, angular_tol, max_candidates=max_candidates)
        if exclude_idx is not None:
            idx = idx[idx != exclude_idx]
        # Canonical (ascending global-index) order before any further processing --
        # `idx` is a set of unique row indices, so this is a tie-free sort (safe,
        # deterministic) that fixes a common baseline order for whichever candidate
        # gathering path produced `idx` (this scalar path, or gpu_uncertainty.py's
        # batched equivalent, which canonicalizes the same way). Matters downstream
        # in ray_transmittance_weights: real candidate depths routinely tie exactly
        # (many rows here are the same physical splat observed by different training
        # cameras -- same position, same depth from any query point), and neither
        # implementation's depth-argsort is stable against its OWN arrival order --
        # without a shared canonical baseline first, tied candidates could get
        # composited in a different order (hence a different transmittance-weighted
        # color) between the scalar and batched paths even when both select the
        # identical candidate SET.
        idx = np.sort(idx)

        local_positions = self.positions[idx]
        local_values = self.values[idx]

        if len(idx) == 0:
            weights = np.zeros(0)
            local_covariances = None
        else:
            local_opacities = self.opacities[idx]
            weights = ray_transmittance_weights(
                local_positions, local_opacities, camera_index.camera, reference_bearing, angular_tol=angular_tol
            )
            local_covariances = self.covariances()[idx] if self.scales is not None and self.rotations is not None else None

        render_weight = self._render_weight_from_local_weights(
            local_positions, weights, query_point, radius, d, local_covariances=local_covariances
        )
        return idx, local_positions, local_values, render_weight, weights

    @staticmethod
    def _render_weight_from_local_weights(
        local_positions: np.ndarray,
        weights: np.ndarray,
        query_point: np.ndarray,
        radius: float,
        d: int,
        local_covariances: Optional[np.ndarray] = None,
    ) -> GaussianRenderWeight:
        """Moment-match a set of nonnegative per-neighbor rendering weights
        (real alpha-compositing transmittance weights) into a single
        GaussianRenderWeight. Falls back to a near-zero-amplitude weight
        centered on `query_point` when every candidate has weight 0
        (nothing on-ray) -- correctly reports "nothing to see" rather than
        dividing by zero.

        `local_covariances` (N, d, d), optional: each candidate's own real
        3D covariance (from its scale/rotation). Standard Gaussian-mixture
        moment matching (e.g. Bishop, *Pattern Recognition and Machine
        Learning*, eq. 2.44 applied to first two moments) says the single
        Gaussian matching the first two moments of a mixture with weights
        w_i, means mu_i, covariances Sigma_i has covariance
        `sum_i w_i (Sigma_i + (mu_i - mean)(mu_i - mean)^T)` -- the spread-
        of-centers term alone (omitting `Sigma_i`, i.e. treating each
        candidate as a literal point) silently collapses to a near-zero,
        jitter-only width whenever the real weights concentrate on one
        dominant splat, even though that splat's own real footprint is not,
        physically, a point. Passing each candidate's real covariance
        fixes this.

        The returned render weight's `total_mass` is pinned to
        `sum(weights)` (via GaussianRenderWeight.from_total_mass), not its
        peak amplitude -- `sum(weights)` is exactly `1 - T_final` for real
        alpha-compositing weights, a real, bounded (<= 1) quantity, unlike
        a peak-amplitude convention whose *mass* would otherwise depend on
        the arbitrary volume of whatever covariance moment-matching
        happens to produce.
        """
        fallback_covariance = (radius / 2.0) ** 2 * np.eye(d)
        total_weight = float(weights.sum()) if weights.size > 0 else 0.0
        if total_weight <= 0.0:
            return GaussianRenderWeight.from_total_mass(total_mass=1e-12, center=query_point, covariance=fallback_covariance)

        normalized = weights / total_weight
        mean = (normalized[:, None] * local_positions).sum(axis=0)
        diff = local_positions - mean
        spread = (normalized[:, None, None] * (diff[:, :, None] * diff[:, None, :])).sum(axis=0)
        within = (normalized[:, None, None] * local_covariances).sum(axis=0) if local_covariances is not None else 0.0
        covariance = spread + within + 1e-6 * np.eye(d)  # jitter avoids a singular covariance from a single point mass
        return GaussianRenderWeight.from_total_mass(total_mass=total_weight, center=mean, covariance=covariance)
