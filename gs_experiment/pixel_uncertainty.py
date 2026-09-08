"""Per-query-point local, rendering-aware BQ uncertainty over a real (or
mock) splat scene, in 3D world space. Directly reuses this project's
validated kernel/quadrature machinery (`kernels.py`/`quadrature.py`),
plus a KD-tree-based neighbor lookup (see `gs_experiment/results/FINDINGS.md`).

`LocalUncertaintyEngine.rendering_aware_variance` and its `_along_ray`/
`_via_gsplat` variants build a query-specific renderer weight
(`render_weight.py`'s GaussianRenderWeight: transmittance x opacity x
footprint) instead of integrating the base kernel uniformly over an
arbitrary box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from gs_experiment.kernels import DirectionalKernel, MaternKernel, ProductKernel, RBFKernel
from gs_experiment.quadrature import BQResult, bayesian_quadrature_rendering_aware, bayesian_quadrature_rendering_aware_directional
from gs_experiment.render_weight import GaussianRenderWeight
from gs_experiment.camera import CameraPose, project_point_to_pixel, viewmat_from_camera_pose
from gs_experiment.visibility_attribution import CameraSplatIndex, project_to_camera_local, ray_transmittance_weights

if TYPE_CHECKING:  # gs_experiment.gsplat_rendering_weights needs torch/gsplat -- not imported at runtime here
    from gs_experiment.gsplat_rendering_weights import GsplatCameraProjection


def make_default_3d_position_kernel(sigma: float) -> ProductKernel:
    """3D generalization of the 2D ProductKernel used elsewhere in this
    project -- no new kernel code needed, ProductKernel already supports
    arbitrary D."""
    return ProductKernel([RBFKernel(sigma=sigma), RBFKernel(sigma=sigma), RBFKernel(sigma=sigma)])


def make_default_3d_matern_kernel(rho: float) -> ProductKernel:
    """Matern-3/2 analogue of make_default_3d_position_kernel, for the
    RBF-vs-Matern kernel-choice comparison ROADMAP.md flags as unresolved
    (gs_experiment/results/FINDINGS.md sections 5-7 validated it only at toy
    scale, never against a real trained GS checkpoint). `rho` plays the
    same "bandwidth" role sigma does for RBF, not an identical physical
    quantity -- passing the same numeric value to both is a like-for-like
    comparison of "same nominal length scale, different smoothness
    assumption," not a claim the two parameters are interchangeable in
    general.
    """
    return ProductKernel([MaternKernel(rho=rho), MaternKernel(rho=rho), MaternKernel(rho=rho)])


def quat_scale_to_covariance(quats: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Real 3D covariance Sigma_i = R_i diag(scale_i^2) R_i^T for each
    (quaternion, scale) pair -- quats as (w, x, y, z), the convention
    gsplat.quat_scale_to_covar_preci uses and this repo's identity
    quaternion [1, 0, 0, 0] already assumes (see
    gs_experiment.splat_scene.make_mock_scene). Pure numpy, so
    rendering_aware_variance_along_ray can use it without a torch/gsplat
    dependency; rendering_aware_variance_via_gsplat instead calls
    gsplat.quat_scale_to_covar_preci directly (the real GPU implementation
    of this exact formula) for the same quantity -- cross-checked against
    each other in tests/test_gsplat_rendering_weights.py.

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
    # (gs_experiment/results/FINDINGS.md section 8) validated the solve cost as
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
        self._covariance_cache: Optional[np.ndarray] = None
        self._rng = np.random.default_rng(self.seed)

    def covariances(self, use_gsplat: bool = False, device: str = "cuda") -> np.ndarray:
        """Every splat's real 3D covariance (from its scale/rotation),
        computed once and cached -- rendering_aware_variance_along_ray and
        rendering_aware_variance_via_gsplat both just index into this
        (`covariances()[idx]`) instead of recomputing it fresh per query
        point, which used to mean one extra GPU kernel launch per query on
        top of the already-fixed candidate-selection cost (confirmed to
        matter: real-checkpoint queries dropped from ~215ms/point to
        single-digit ms/point once this was cached instead).

        `use_gsplat=True` computes it via gsplat's own real
        `quat_scale_to_covar_preci` (needs torch/gsplat + a GPU, lazily
        imported); otherwise the plain-numpy `quat_scale_to_covariance` --
        the same formula, cross-checked against gsplat's own in
        tests/test_gsplat_rendering_weights.py, and cheap enough (a single
        vectorized call over the whole scene, not per-query) that the GPU
        path is an optional consistency choice, not a necessity.
        """
        if self.scales is None or self.rotations is None:
            raise ValueError("scales/rotations not set on this engine -- construct with both to use this method")
        if self._covariance_cache is None:
            if use_gsplat:
                from gs_experiment.gsplat_rendering_weights import gsplat_covariances

                self._covariance_cache = gsplat_covariances(self.scales, self.rotations, device=device)
            else:
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

    def rendering_aware_variance(
        self, query_point: np.ndarray, radius: float, exclude_idx: Optional[int] = None, sigma_rbf: Optional[float] = None
    ) -> BQResult:
        """A renderer-aware local BQ variance: instead of integrating
        pos_kernel uniformly over an arbitrary box, this builds a real per-query a_q
        (gs_experiment/render_weight.py) from data this engine already has --
        real per-splat opacity as a_q's amplitude, and a Gaussian footprint
        tied to `radius` -- and calls bayesian_quadrature_rendering_aware.
        A splat with low local opacity now contributes less to both mean
        and variance by construction, rather than by virtue of sitting
        outside an ad hoc box.

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

        mean_opacity = float(np.mean(self.opacities[idx])) if len(idx) > 0 else float(np.mean(self.opacities))
        d = np.asarray(query_point).shape[0]
        covariance = (radius / 2.0) ** 2 * np.eye(d)
        # from_total_mass, not a direct amplitude=mean_opacity constructor
        # call: mean_opacity is a real, bounded quantity (a compositing
        # weight budget), and pinning it as a_q's *mass* (rather than its
        # peak height) is what keeps it comparable across different
        # `radius` choices -- see gs_experiment/render_weight.py's docstring.
        render_weight = GaussianRenderWeight.from_total_mass(total_mass=mean_opacity, center=query_point, covariance=covariance)

        return bayesian_quadrature_rendering_aware(local_positions, local_values, render_weight, sigma_rbf=sigma_rbf)

    def build_bearing_index(self, camera: CameraPose) -> CameraSplatIndex:
        """Build once per camera, reuse across every
        rendering_aware_variance_along_ray[_directional] call against that
        camera -- see CameraSplatIndex's docstring for why a fresh
        3D-ball-query candidate set per call is both wrong and wasteful on
        a real, dense checkpoint. Passes `self.directions` through (a
        no-op when unset) so a directional query can rank overflow
        candidates by directional alignment instead of bearing-distance
        ties -- see CameraSplatIndex's docstring for why that matters.
        """
        return CameraSplatIndex.build(self.positions, camera, directions=self.directions)

    def rendering_aware_variance_along_ray(
        self,
        query_point: np.ndarray,
        camera_index: CameraSplatIndex,
        radius: float,
        exclude_idx: Optional[int] = None,
        angular_tol: float = 0.05,
        max_candidates: int = 500,
        sigma_rbf: Optional[float] = None,
    ) -> BQResult:
        """rendering_aware_variance's occlusion-aware upgrade: a_q's shape
        is built from real, depth-ordered alpha-compositing transmittance
        weights along the specific ray through `query_point`
        (gs_experiment.visibility_attribution.ray_transmittance_weights),
        instead of a flat neighborhood-mean opacity. A splat behind a
        closer, opaque splat *on this ray* now gets a small weight from
        real accumulated transmittance -- the gap rendering_aware_variance's
        own docstring names -- while a splat that's simply nearby in 3D but
        off to the side of this particular ray gets weight 0 regardless of
        its opacity, since it was never going to occlude or contribute to
        this pixel.

        `camera_index`: a CameraSplatIndex built once via
        `build_bearing_index(camera)` and reused across every query against
        that camera -- candidates come from real bearing-space proximity,
        not a 3D-world-space ball query (see CameraSplatIndex's docstring
        for why that matters on a real, densely-packed checkpoint: a 3D
        radius wide enough to reliably contain the truly relevant splat can
        pull in a large fraction of the *entire* scene, and randomly
        subsampling down from there almost never keeps that splat).

        Still not a live differentiable rasterizer: no antialiasing or
        sub-pixel footprint integration, no gradient path -- a real,
        geometric depth/occlusion proxy (project_to_camera_local + real
        per-splat opacity as alpha), not a claim of pixel-exact equivalence
        to an actual renderer.

        When every candidate is off-ray or behind the camera (a_q is
        identically 0 there), falls back to a near-zero amplitude rather
        than raising -- correctly reports "this query has ~nothing to see
        here": mean ~0, variance ~0 (the posterior collapses toward the
        near-vanishing prior, not toward a spurious nonzero guess).
        """
        query_point = np.asarray(query_point, dtype=float)
        sigma_rbf = self._require_rbf_sigma("rendering_aware_variance_along_ray", sigma_rbf)
        idx, local_positions, local_values, render_weight = self._along_ray_local_data(
            query_point, camera_index, radius, exclude_idx, angular_tol, max_candidates
        )
        return bayesian_quadrature_rendering_aware(local_positions, local_values, render_weight, sigma_rbf=sigma_rbf)

    def rendering_aware_variance_along_ray_directional(
        self,
        query_point: np.ndarray,
        query_direction: np.ndarray,
        camera_index: CameraSplatIndex,
        radius: float,
        exclude_idx: Optional[int] = None,
        angular_tol: float = 0.05,
        max_candidates: int = 500,
        sigma_rbf: Optional[float] = None,
    ) -> BQResult:
        """rendering_aware_variance_along_ray, completed with the
        directional/epistemic term the original rendering-aware
        construction always specified but this module only partially
        implemented: a joint `k_base(xi, xi') = k_pos(x,x') * k_dir(d,d')`
        (see bayesian_quadrature_rendering_aware_directional), instead of
        a position-only kernel. `self.directions`/`self.dir_kernel` must
        be set -- one direction per row of `self.positions`, i.e. this
        engine must be built from observation-expanded arrays (one row
        per (splat, observing-camera) pair, as `splat_observations`
        produces): a splat needs to have been observed from *multiple*
        directions for this term to carry any signal.

        Without this, `rendering_aware_variance_along_ray` only answers
        "is this local color well-resolved, correctly down-weighting
        occluded/off-ray splats" -- a spatial/occlusion question. This
        answers the complementary one: "is *this specific viewing angle*
        well-constrained by the directions training actually observed
        this splat from," now combined with the same real occlusion-aware
        a_q rather than computed against an occlusion-blind box kernel.
        """
        if self.directions is None or self.dir_kernel is None:
            raise ValueError("directions/dir_kernel not set on this engine -- construct with both to use this method")
        query_point = np.asarray(query_point, dtype=float)
        sigma_rbf = self._require_rbf_sigma("rendering_aware_variance_along_ray_directional", sigma_rbf)
        idx, local_positions, local_values, render_weight = self._along_ray_local_data(
            query_point, camera_index, radius, exclude_idx, angular_tol, max_candidates, query_direction=query_direction
        )
        local_directions = self.directions[idx]
        return bayesian_quadrature_rendering_aware_directional(
            local_positions, local_directions, local_values, render_weight, self.dir_kernel, query_direction,
            sigma_rbf=sigma_rbf,
        )

    def _require_rbf_sigma(self, method_name: str, sigma_rbf: Optional[float]) -> float:
        first_kernel = self.pos_kernel.kernels_per_axis[0]
        if not isinstance(first_kernel, RBFKernel):
            raise ValueError(
                f"{method_name} requires an RBF pos_kernel (bayesian_quadrature_rendering_aware's "
                "closed form is RBF-only -- Matern would need its numerical mode, not wired up here)"
            )
        return first_kernel.sigma if sigma_rbf is None else sigma_rbf

    def _along_ray_local_data(
        self,
        query_point: np.ndarray,
        camera_index: CameraSplatIndex,
        radius: float,
        exclude_idx: Optional[int],
        angular_tol: float,
        max_candidates: Optional[int],
        query_direction: Optional[np.ndarray] = None,
    ):
        """Shared candidate-gathering + render_weight construction for
        rendering_aware_variance_along_ray and its directional variant --
        they differ only in which quadrature function they call with the
        result (and whether they also need local_directions).

        `query_direction`, passed only by the directional variant: ranks
        any overflow past `max_candidates` by directional alignment
        instead of bearing-distance ties -- see CameraSplatIndex.query's
        docstring for why the directional case needs that (bearing alone
        can't distinguish between a single physical splat's many
        observation-direction rows, which all share one bearing)."""
        if self.opacities is None:
            raise ValueError("opacities not set on this engine -- construct with opacities= to use this method")

        d = query_point.shape[0]
        ref_bearing_x, ref_bearing_y, _ = project_to_camera_local(query_point.reshape(1, -1), camera_index.camera)
        reference_bearing = (float(ref_bearing_x[0]), float(ref_bearing_y[0]))
        idx = camera_index.query(reference_bearing, angular_tol, max_candidates=max_candidates, query_direction=query_direction)
        if exclude_idx is not None:
            idx = idx[idx != exclude_idx]

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
        return idx, local_positions, local_values, render_weight

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
        (real alpha-compositing transmittance weights, from whichever
        source computed them) into a single GaussianRenderWeight -- shared
        by rendering_aware_variance_along_ray and
        rendering_aware_variance_via_gsplat, which differ only in how
        `weights` (and optionally `local_covariances`) are computed. Falls
        back to a near-zero-amplitude weight centered on `query_point` when
        every candidate has weight 0 (nothing on-ray / nothing gsplat
        projects as visible here) -- correctly reports "nothing to see"
        rather than dividing by zero.

        `local_covariances` (N, d, d), optional: each candidate's own real
        3D covariance (from its scale/rotation). Standard Gaussian-mixture
        moment matching (e.g. Bishop, *Pattern Recognition and Machine
        Learning*, eq. 2.44 applied to first two moments) says the single
        Gaussian matching the first two moments of a mixture with weights
        w_i, means mu_i, covariances Sigma_i has covariance
        `sum_i w_i (Sigma_i + (mu_i - mean)(mu_i - mean)^T)` -- the spread-
        of-centers term alone (omitting `Sigma_i`, i.e. treating each
        candidate as a literal point) is what this function computed
        before `local_covariances` existed, and silently collapses to a
        near-zero, jitter-only width whenever the real weights concentrate
        on one dominant splat (confirmed: this happened in practice for
        gsplat's real, much sharper anisotropic weights -- see git history)
        even though that splat's own real footprint is not, physically,
        a point. Passing each candidate's real covariance fixes this.

        The returned render weight's `total_mass` is pinned to
        `sum(weights)` (via GaussianRenderWeight.from_total_mass), not its
        peak amplitude -- `sum(weights)` is exactly `1 - T_final` for real
        alpha-compositing weights, a real, bounded (<= 1) quantity, unlike
        a peak-amplitude convention whose *mass* would otherwise depend on
        the arbitrary volume of whatever covariance moment-matching
        happens to produce (see gs_experiment/render_weight.py's docstring --
        this is what made rendering_aware_variance_via_gsplat's variance
        hit ~1e-100 on a real checkpoint before this fix).
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

    def build_gsplat_projection(
        self, camera: CameraPose, K: np.ndarray, width: int, height: int, device: str = "cuda"
    ) -> "GsplatCameraProjection":
        """Build once per camera (+ intrinsics), reuse across every
        rendering_aware_variance_via_gsplat call against that camera -- see
        GsplatCameraProjection's docstring for why a fresh per-call
        fully_fused_projection over a 3D-ball-queried candidate set is both
        wrong and wasteful on a real, dense checkpoint. Requires torch + a
        CUDA-enabled gsplat build and a GPU (see ../requirements-gsplat.txt)
        -- imported lazily so this module stays importable without those.
        """
        if self.opacities is None or self.scales is None or self.rotations is None:
            raise ValueError(
                "opacities/scales/rotations not set on this engine -- construct with all three to use this method"
            )
        from gs_experiment.gsplat_rendering_weights import GsplatCameraProjection

        return GsplatCameraProjection.build(
            self.positions, self.opacities, self.scales, self.rotations, camera, K, width, height, device=device,
            directions=self.directions,
        )

    def rendering_aware_variance_via_gsplat(
        self,
        query_point: np.ndarray,
        projection: "GsplatCameraProjection",
        radius: float,
        exclude_idx: Optional[int] = None,
        pixel_radius: float = 64.0,
        max_candidates: int = 300,
        sigma_rbf: Optional[float] = None,
        device: str = "cuda",
    ) -> BQResult:
        """rendering_aware_variance_along_ray's most faithful version: a_q's
        shape comes from gsplat's own differentiable EWA-splatting
        projection (gs_experiment.gsplat_rendering_weights, a real GPU
        computation) instead of an isotropic bearing-distance threshold --
        the real anisotropic 2D footprint (from each splat's actual scale
        and rotation) and real camera intrinsics determine each candidate
        splat's per-pixel alpha, not a flat opacity gated by angular
        proximity.

        `projection`: a GsplatCameraProjection built once via
        `build_gsplat_projection(camera, K, width, height)` and reused
        across every query against that camera + intrinsics -- candidates
        come from real pixel-space proximity (a cKDTree over each splat's
        own projected 2D mean), not a 3D-world-space ball query (see
        GsplatCameraProjection's docstring for why that matters on a real,
        densely-packed checkpoint: a 3D radius wide enough to reliably
        contain the truly relevant splat can pull in a large fraction of
        the *entire* scene, and randomly subsampling down from there almost
        never keeps that splat -- confirmed directly: reported variance as
        small as `1e-95` on a real checkpoint before this fix, for exactly
        that reason, not because there was genuinely no uncertainty there).

        `max_candidates` defaults to 300, not higher: unlike the old random
        subsample, candidates surviving this cap are the real nearest-in-
        pixel-space ones (see GsplatCameraProjection.query_pixel), so
        capping doesn't reintroduce the relevance problem -- but the BQ
        solve below is at least O(n^2)-O(n^3) in candidate count
        (gs_experiment/results/FINDINGS.md section 8's already-validated
        "negligible up to hundreds, not thousands" regime), and a dense
        real checkpoint can easily project thousands of splats within a
        generous `pixel_radius` (confirmed: ~200ms/query at 2000
        candidates vs. single-digit ms at 300, on an RTX 3090).
        """
        query_point = np.asarray(query_point, dtype=float)
        sigma_rbf = self._require_rbf_sigma("rendering_aware_variance_via_gsplat", sigma_rbf)
        idx, local_positions, local_values, render_weight = self._via_gsplat_local_data(
            query_point, projection, radius, exclude_idx, pixel_radius, max_candidates, device
        )
        return bayesian_quadrature_rendering_aware(local_positions, local_values, render_weight, sigma_rbf=sigma_rbf)

    def rendering_aware_variance_via_gsplat_directional(
        self,
        query_point: np.ndarray,
        query_direction: np.ndarray,
        projection: "GsplatCameraProjection",
        radius: float,
        exclude_idx: Optional[int] = None,
        pixel_radius: float = 64.0,
        max_candidates: int = 300,
        sigma_rbf: Optional[float] = None,
        device: str = "cuda",
    ) -> BQResult:
        """rendering_aware_variance_via_gsplat, completed with the
        directional/epistemic term (see
        rendering_aware_variance_along_ray_directional's docstring for the
        general argument, and
        bayesian_quadrature_rendering_aware_directional for the math):
        `k_base(xi, xi') = k_pos(x,x') * k_dir(d,d')` instead of a
        position-only kernel, combined with gsplat's real per-pixel a_q.
        `self.directions`/`self.dir_kernel` must be set -- one direction
        per row of `self.positions` (observation-expanded arrays, one row
        per (splat, observing-camera) pair), the same requirement
        `rendering_aware_variance_along_ray_directional` already has.
        """
        if self.directions is None or self.dir_kernel is None:
            raise ValueError("directions/dir_kernel not set on this engine -- construct with both to use this method")
        query_point = np.asarray(query_point, dtype=float)
        sigma_rbf = self._require_rbf_sigma("rendering_aware_variance_via_gsplat_directional", sigma_rbf)
        idx, local_positions, local_values, render_weight = self._via_gsplat_local_data(
            query_point, projection, radius, exclude_idx, pixel_radius, max_candidates, device, query_direction=query_direction
        )
        local_directions = self.directions[idx]
        return bayesian_quadrature_rendering_aware_directional(
            local_positions, local_directions, local_values, render_weight, self.dir_kernel, query_direction,
            sigma_rbf=sigma_rbf,
        )

    def _via_gsplat_local_data(
        self,
        query_point: np.ndarray,
        projection: "GsplatCameraProjection",
        radius: float,
        exclude_idx: Optional[int],
        pixel_radius: float,
        max_candidates: Optional[int],
        device: str,
        query_direction: Optional[np.ndarray] = None,
    ):
        """Shared candidate-gathering + render_weight construction for
        rendering_aware_variance_via_gsplat and its directional variant.

        `query_direction`, passed only by the directional variant: see
        GsplatCameraProjection.query_pixel's docstring for why an
        overflowing max_candidates cut needs to rank by directional
        alignment rather than pixel-distance ties when `positions`/
        `directions` are a camera-expanded observation array."""
        d = query_point.shape[0]
        pixel_xy = project_point_to_pixel(query_point, viewmat_from_camera_pose(projection.camera), projection.K)
        idx, weights = projection.query_pixel(
            pixel_xy, pixel_radius=pixel_radius, max_candidates=max_candidates, query_direction=query_direction
        )
        if exclude_idx is not None:
            keep = idx != exclude_idx
            idx, weights = idx[keep], weights[keep]

        local_positions = self.positions[idx]
        local_values = self.values[idx]
        # Each candidate's real 3D covariance, not just a point -- see
        # _render_weight_from_local_weights's docstring for why this
        # matters: gsplat's real per-pixel weights are often sharply
        # concentrated on very few splats, which collapses the spread-
        # of-centers term alone to a near-degenerate width. Cached across
        # the whole scene (see covariances()), not recomputed per query.
        local_covariances = self.covariances(use_gsplat=True, device=device)[idx] if len(idx) > 0 else None

        render_weight = self._render_weight_from_local_weights(
            local_positions, weights, query_point, radius, d, local_covariances=local_covariances
        )
        return idx, local_positions, local_values, render_weight
