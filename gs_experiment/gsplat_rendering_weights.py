"""Real, GPU-computed per-pixel alpha-compositing weights via gsplat's own
differentiable EWA-splatting projection (`gsplat.fully_fused_projection`) --
the actual anisotropic 2D covariance/opacity math the real CUDA rasterizer
uses for a given camera, in place of a geometric proxy.

Needs torch + a CUDA-enabled gsplat build and a real GPU (see
../requirements-gsplat.txt for the exact setup, including the gcc/nvcc
version gotcha already solved there). This is a strictly more faithful
replacement for
`gs_experiment.visibility_attribution.ray_transmittance_weights`, which
uses an isotropic bearing-distance threshold and flat per-splat opacity
instead of each splat's real anisotropic projected footprint.

Kept as its own module -- not imported at
`gs_experiment.pixel_uncertainty`'s top level -- so that module, and the
rest of this project's default GPU-free test suite
(`python -m pytest tests/`), stays importable without torch/gsplat
installed. See `LocalUncertaintyEngine.rendering_aware_variance_via_gsplat`
for the (lazily imported) caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import gsplat
import numpy as np
import torch
from scipy.spatial import cKDTree

from gs_experiment.camera import CameraPose, viewmat_from_camera_pose


def gsplat_alpha_compositing_weights(
    positions: np.ndarray,
    opacities: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    camera: CameraPose,
    K: np.ndarray,
    width: int,
    height: int,
    pixel_xy,
    device: str = "cuda",
) -> np.ndarray:
    """Real per-splat alpha-compositing transmittance weight at one
    specific pixel, `w_i = T_i * alpha_i(pixel)`, using gsplat's own
    projected 2D mean/conic/depth for `camera` -- the same EWA-splatting
    math the real renderer uses, not a geometric proxy:

        alpha_i(pixel) = opacities_i * exp(-sigma_i),
        sigma_i = 0.5 * d_i^T conic_i d_i,     d_i = pixel - means2d_i

    clipped to at most 0.999 and zeroed whenever `sigma_i < 0` or gsplat's
    own projection marks the Gaussian invalid (zero radius: behind the
    camera, culled by the near/far plane, or projecting off-screen) --
    exactly the formula gsplat's CUDA rasterizer evaluates per pixel, just
    run here in plain torch for one query pixel over a local neighborhood
    instead of the full fused kernel over a whole image.
    `T_i = prod_{j closer in depth} (1 - alpha_j(pixel))`: the same
    recursion as `visibility_attribution.ray_transmittance_weights`, now
    driven by a real anisotropic per-pixel alpha instead of a flat opacity
    constant gated by an isotropic bearing threshold.

    Returns weights as a numpy array, same length/order as `positions`.
    Runs under `torch.no_grad()` -- this is an evaluation-time query, not
    a training step; drop that context manager if a gradient through
    these weights is ever needed.

    For many queries against the *same* camera (e.g. a per-pixel sweep
    over one image), use `GsplatCameraProjection` instead: calling this
    function fresh for every query point re-runs `fully_fused_projection`
    over whatever candidate set the caller passed in, which is both
    expensive (repeated CUDA launches) and, if that candidate set came
    from a 3D-world-space neighbor search, likely *wrong* -- confirmed on
    a real 300k-splat checkpoint, where a 3D ball query wide enough to
    reliably contain the truly relevant splat pulled in ~half the entire
    scene, and randomly subsampling down from there almost always
    discarded that splat. `GsplatCameraProjection` projects the whole
    scene once and selects candidates by real pixel-space proximity
    instead.
    """
    n = positions.shape[0]
    if n == 0:
        return np.zeros(0)

    means = torch.as_tensor(positions, dtype=torch.float32, device=device)
    quats = torch.as_tensor(rotations, dtype=torch.float32, device=device)
    scales_t = torch.as_tensor(scales, dtype=torch.float32, device=device)
    opacities_t = torch.as_tensor(opacities, dtype=torch.float32, device=device)
    viewmat_t = torch.as_tensor(viewmat_from_camera_pose(camera), dtype=torch.float32, device=device)
    K_t = torch.as_tensor(K, dtype=torch.float32, device=device)
    pixel = torch.as_tensor(pixel_xy, dtype=torch.float32, device=device)

    with torch.no_grad():
        radii, means2d, depths, conics, _ = gsplat.fully_fused_projection(
            means, None, quats, scales_t, viewmat_t[None], K_t[None], width, height, opacities=opacities_t
        )
        radii = radii[0]  # (N, 2)
        means2d = means2d[0]  # (N, 2)
        depths = depths[0]  # (N,)
        conics = conics[0]  # (N, 3)

        valid = (radii[:, 0] > 0) & (radii[:, 1] > 0)
        d = pixel[None, :] - means2d
        c00, c01, c11 = conics[:, 0], conics[:, 1], conics[:, 2]
        sigma = 0.5 * (c00 * d[:, 0] ** 2 + c11 * d[:, 1] ** 2) + c01 * d[:, 0] * d[:, 1]
        alpha_pixel = torch.clamp(opacities_t * torch.exp(-sigma), max=0.999)
        alpha_pixel = torch.where(valid & (sigma >= 0), alpha_pixel, torch.zeros_like(alpha_pixel))

        depth_order = torch.argsort(depths).tolist()
        weights = torch.zeros(n, device=device)
        transmittance = 1.0
        for i in depth_order:
            alpha = float(alpha_pixel[i])
            if alpha <= 0.0:
                continue
            weights[i] = transmittance * alpha
            transmittance *= 1.0 - alpha

    return weights.cpu().numpy()


def gsplat_covariances(scales: np.ndarray, rotations: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Real 3D covariance Sigma_i = R_i diag(scale_i^2) R_i^T for each
    splat, via gsplat's own `quat_scale_to_covar_preci` -- the real GPU
    implementation of the same formula
    `gs_experiment.pixel_uncertainty.quat_scale_to_covariance` computes in
    plain numpy (cross-checked against each other in
    tests/test_gsplat_rendering_weights.py). Used by
    `LocalUncertaintyEngine.rendering_aware_variance_via_gsplat` to give
    the moment-matched GaussianRenderWeight each candidate splat's real
    physical width, not just its center point -- see
    `_render_weight_from_local_weights`'s docstring for why that matters.
    """
    quats = torch.as_tensor(rotations, dtype=torch.float32, device=device)
    scales_t = torch.as_tensor(scales, dtype=torch.float32, device=device)
    with torch.no_grad():
        covars, _ = gsplat.quat_scale_to_covar_preci(quats, scales_t, compute_covar=True, compute_preci=False)
    return covars.cpu().numpy()


@dataclass
class GsplatCameraProjection:
    """Real gsplat projection of an entire scene against ONE camera,
    computed once (a single batched `gsplat.fully_fused_projection` call)
    and reused for many per-pixel
    `LocalUncertaintyEngine.rendering_aware_variance_via_gsplat` queries
    against that camera.

    Replaces re-running `fully_fused_projection` over a fresh, 3D-world-
    space-ball-queried candidate set for every single query point (see
    `gsplat_alpha_compositing_weights`'s docstring, and
    `gs_experiment.visibility_attribution.CameraSplatIndex`'s docstring
    for the identical argument in the non-gsplat bearing-based method):
    that's both wrong -- 3D Euclidean proximity is not the same as
    pixel-space relevance, confirmed directly on a real 300k-splat
    checkpoint -- and wasteful, one CUDA launch per query point instead
    of one per camera. Candidates for a given pixel are selected here by
    real *pixel-space* proximity (a cKDTree over each splat's own
    projected 2D mean), matching how a real tile-based rasterizer bins
    splats into screen tiles.
    """

    indices: np.ndarray  # into the original positions/opacities/scales/rotations arrays; valid (radii > 0) splats only
    means2d: np.ndarray  # (M, 2)
    depths: np.ndarray  # (M,)
    conics: np.ndarray  # (M, 3)
    opacities: np.ndarray  # (M,)
    camera: CameraPose
    K: np.ndarray
    width: int
    height: int
    _tree: Optional[cKDTree] = field(default=None, repr=False)

    def __post_init__(self):
        self._tree = cKDTree(self.means2d) if len(self.indices) > 0 else None

    @classmethod
    def build(
        cls,
        positions: np.ndarray,
        opacities: np.ndarray,
        scales: np.ndarray,
        rotations: np.ndarray,
        camera: CameraPose,
        K: np.ndarray,
        width: int,
        height: int,
        device: str = "cuda",
    ) -> "GsplatCameraProjection":
        means = torch.as_tensor(positions, dtype=torch.float32, device=device)
        quats = torch.as_tensor(rotations, dtype=torch.float32, device=device)
        scales_t = torch.as_tensor(scales, dtype=torch.float32, device=device)
        opacities_t = torch.as_tensor(opacities, dtype=torch.float32, device=device)
        viewmat_t = torch.as_tensor(viewmat_from_camera_pose(camera), dtype=torch.float32, device=device)
        K_t = torch.as_tensor(K, dtype=torch.float32, device=device)

        with torch.no_grad():
            radii, means2d, depths, conics, _ = gsplat.fully_fused_projection(
                means, None, quats, scales_t, viewmat_t[None], K_t[None], width, height, opacities=opacities_t
            )
        radii_np = radii[0].cpu().numpy()
        valid = (radii_np[:, 0] > 0) & (radii_np[:, 1] > 0)
        idx = np.where(valid)[0]
        return cls(
            indices=idx,
            means2d=means2d[0].cpu().numpy()[valid],
            depths=depths[0].cpu().numpy()[valid],
            conics=conics[0].cpu().numpy()[valid],
            opacities=np.asarray(opacities)[valid],
            camera=camera,
            K=np.asarray(K, dtype=float),
            width=width,
            height=height,
        )

    def query_pixel(self, pixel_xy, pixel_radius: float = 64.0, max_candidates: Optional[int] = 2000):
        """Real, depth-ordered alpha-compositing transmittance weight
        `w_i = T_i * alpha_i(pixel)` for splats within `pixel_radius`
        screen-space pixels of `pixel_xy` -- the same analytic per-pixel
        alpha formula `gsplat_alpha_compositing_weights` uses (see that
        function's docstring), evaluated here in plain numpy against this
        cached projection instead of a fresh CUDA call. If more than
        `max_candidates` splats project within `pixel_radius`, keeps the
        ones nearest in pixel space -- a real relevance ranking, not a
        random subsample.

        Returns `(original_indices, weights)`: indices into the arrays
        `build` was called with, and their composited weights (zeros for
        any candidate a closer, opaque splat fully occludes).
        """
        if self._tree is None:
            return np.empty(0, dtype=int), np.empty(0)
        pixel_xy = np.asarray(pixel_xy, dtype=float)
        local = np.array(self._tree.query_ball_point(pixel_xy, pixel_radius), dtype=int)
        if local.size == 0:
            return np.empty(0, dtype=int), np.empty(0)
        if max_candidates is not None and local.size > max_candidates:
            dists = np.linalg.norm(self.means2d[local] - pixel_xy, axis=1)
            local = local[np.argsort(dists)[:max_candidates]]

        d = pixel_xy[None, :] - self.means2d[local]
        c00, c01, c11 = self.conics[local, 0], self.conics[local, 1], self.conics[local, 2]
        sigma = 0.5 * (c00 * d[:, 0] ** 2 + c11 * d[:, 1] ** 2) + c01 * d[:, 0] * d[:, 1]
        alpha = np.clip(self.opacities[local] * np.exp(-sigma), 0.0, 0.999)
        alpha = np.where(sigma >= 0, alpha, 0.0)

        depth_order = np.argsort(self.depths[local])
        weights_local = np.zeros(local.size)
        transmittance = 1.0
        for k in depth_order:
            a = float(alpha[k])
            if a <= 0.0:
                continue
            weights_local[k] = transmittance * a
            transmittance *= 1.0 - a

        return self.indices[local], weights_local
