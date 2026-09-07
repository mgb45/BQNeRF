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

import gsplat
import numpy as np
import torch

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
