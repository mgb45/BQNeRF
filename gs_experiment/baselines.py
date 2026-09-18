"""Competing uncertainty constructions, reimplemented in this project's own
harness so that every method is scored by the pre-registered protocol in
`evaluation.py` on IDENTICAL checkpoints and identical held-out views.

Reimplementation, not reuse of authors' code, is a deliberate compromise and
its direction matters: a weak reimplementation of a competitor turns a real
comparison into a strawman. Each baseline below is therefore built in its
strongest reasonable form, using the same rasterizer-native machinery our own
method uses, and the places where our version may still understate the
original are stated in each function's docstring.

Three constructions, chosen because each isolates ONE mechanism choice:

1. `fit_residual_supervised_sh` -- Galappaththige-style
   (`galappaththige2026predictive`). Same slot as ours (frozen map, per-splat
   SH-valued uncertainty channel) but FITTED, by regularised linear least
   squares against real photometric residuals. Isolates *closed-form vs
   fitted*. It is supervised on the very quantity the evaluation scores, so
   it should win; that is the point of including it.

2. `all_parameter_laplace_draws` -- FisherRF/OUGS-style (`jiang2024`,
   `li2026ougs`). Diagonal Fisher information over EVERY Gaussian parameter,
   not just the SH coefficients, sampled and rendered. Isolates *which part
   of the model gets the posterior*, and is the direct test of FINDINGS
   section 7 (where adding opacity to our posterior made calibration worse)
   against OUGS's premise (that geometry belongs in the uncertainty).

3. `uniform_coverage_sh` -- Han-style (`han2025viewdependent`) in ablation
   form. Identical to our construction except every observation is weighted
   equally instead of by its real compositing weight `beta^2`. Isolates
   *Fisher weighting vs plain angular coverage* -- i.e. whether the renderer's
   own contribution weights carry information that "how many directions was
   this splat seen from" does not.
"""

from __future__ import annotations

import numpy as np
import torch

from gs_experiment.coupled_sh_posterior import _camera_basis, _data_matvec, _viewmat
from gs_experiment.rasterized_sh_precision import _sh_basis_torch


def _adjoint(target_fn, means, quats, scales, opacities, frames, Ks, width, height,
             degree, n_rhs, device):
    """`Phi^T t`: for each training camera, backpropagate that camera's target
    image through the render to per-splat weights `sum_q beta_{q,i} t_q`, then
    lift to SH coefficients by multiplying by `phi(d_{i,c})`.

    `Phi` is the linear operator taking per-splat SH coefficients to rendered
    images, `(Phi v)(q) = sum_i beta_{q,i} phi(d_{i,c})^T v_i` -- the same
    operator `coupled_sh_posterior._data_matvec` applies `Phi^T Phi` for."""
    import gsplat

    n_splats = means.shape[0]
    n_coeffs = _sh_basis_torch(torch.zeros((1, 3), device=device), degree).shape[-1]
    out = torch.zeros((n_splats, n_rhs, n_coeffs), dtype=torch.float32, device=device)
    for p, (_, c2w) in enumerate(frames):
        target = target_fn(p)                     # (1, H, W, n_rhs)
        if target is None:
            continue
        phi = _camera_basis(means, c2w, degree)
        colors = torch.ones((n_splats, n_rhs), dtype=torch.float32, device=device,
                            requires_grad=True)
        viewmat = torch.tensor(_viewmat(c2w), dtype=torch.float32, device=device)[None]
        image, _, _ = gsplat.rasterization(means, quats, scales, opacities, colors,
                                           viewmat, Ks, width=width, height=height,
                                           sh_degree=None)
        (image * target).sum().backward()
        out += colors.grad[:, :, None] * phi[:, None, :]
    return out


def fit_residual_supervised_sh(checkpoint, frames, K, width, height, degree, image_dir,
                               lam=1.0, n_cg_iters=200, device="cuda",
                               background_color=(1.0, 1.0, 1.0), verbose=True):
    """Galappaththige-style per-splat SH uncertainty channel, fitted by
    regularised linear least squares against the real photometric residual on
    the TRAINING images:

        min_psi  || Phi psi - |I_gt - I_render| ||^2 + lam ||psi||^2

    solved as `(Phi^T Phi + lam I) psi = Phi^T t` by preconditioned CG, using
    the same rasterizer-native operator our own posterior uses. Returns
    `(n_splats, 3, n_coeffs)` coefficients whose render IS the predicted
    per-pixel uncertainty.

    Where this may understate the original: the published method fits a
    Bayesian-regularised model with its own hyperparameter schedule and may
    target a different residual statistic. We fit the absolute residual per
    channel (so the rendered field is a sigma up to one global factor, which
    the evaluation protocol's scale-free calibration absorbs anyway), and pick
    `lam` by the same empirical-Bayes logic used elsewhere here rather than by
    their schedule. The construction -- frozen map, per-splat SH channel,
    linear LS against photometric residuals -- is theirs.
    """
    import gsplat
    import os
    from PIL import Image

    t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    means, quats = t(checkpoint["positions"]), t(checkpoint["rotations"])
    scales, ops = t(checkpoint["scales"]), t(checkpoint["opacities"])
    sh = t(checkpoint["sh_coeffs"]).transpose(1, 2)
    Ks, bg = t(K)[None], t(background_color)
    n_splats = means.shape[0]

    targets = {}
    with torch.no_grad():
        for p, (file_path, c2w) in enumerate(frames):
            viewmat = t(_viewmat(c2w))[None]
            rendered, _, _ = gsplat.rasterization(means, quats, scales, ops, sh, viewmat, Ks,
                                                  width=width, height=height,
                                                  sh_degree=checkpoint["sh_degree"], backgrounds=bg)
            gt = np.asarray(Image.open(os.path.join(image_dir, file_path + ".png")).convert("RGB"),
                            dtype=np.float32) / 255.0
            targets[p] = (rendered[0].clamp(0, 1) - t(gt)).abs()[None]     # (1, H, W, 3)

    rhs = _adjoint(lambda p: targets[p], means, quats, scales, ops, frames, Ks,
                   width, height, degree, 3, device)

    def A(v):
        return _data_matvec(v, means, quats, scales, ops, frames, Ks,
                            width, height, degree, device) + lam * v

    x = torch.zeros_like(rhs)
    r = rhs.clone()
    p_dir = r.clone()
    rr = (r * r).sum(dim=(0, 2))
    rhs_norm = rhs.norm(dim=(0, 2)).clamp_min(1e-30)
    for it in range(n_cg_iters):
        active = (r.norm(dim=(0, 2)) / rhs_norm) > 1e-6
        if not bool(active.any()):
            break
        Ap = A(p_dir)
        pAp = (p_dir * Ap).sum(dim=(0, 2))
        if bool((active & ((pAp <= 0) | (rr <= 0))).any()):
            print(f"    LS CG breakdown at {it + 1}; stopping")
            break
        ones = torch.ones_like(rr)
        alpha = torch.where(active, rr / torch.where(active, pAp, ones), torch.zeros_like(rr))
        x += alpha[None, :, None] * p_dir
        r -= alpha[None, :, None] * Ap
        rr_new = (r * r).sum(dim=(0, 2))
        rel = (r.norm(dim=(0, 2)) / rhs_norm).max().item()
        if verbose and (it % 50 == 0 or it == n_cg_iters - 1):
            print(f"    LS CG {it + 1}/{n_cg_iters}: max rel residual {rel:.3e}")
        if rel < 1e-6:
            break
        beta = torch.where(active, rr_new / torch.where(active, rr, ones), torch.zeros_like(rr))
        p_dir = r + beta[None, :, None] * p_dir
        rr = rr_new
    return x        # (n_splats, 3, n_coeffs)


def render_sh_field(checkpoint, psi, c2w, K, width, height, degree, device="cuda"):
    """Render a per-splat SH-valued field through the real rasterizer, exactly
    linearly: `(Phi psi)(q) = sum_i beta_{q,i} phi(d_{i,c})^T psi_i`.

    Evaluated as an explicit per-splat scalar feature per channel rather than
    by handing `psi` to gsplat as SH coefficients, because gsplat's SH path
    adds 0.5 and clamps at zero -- fine for colours, but it would break the
    linearity this operator depends on."""
    import gsplat

    t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    means = t(checkpoint["positions"])
    phi = _camera_basis(means, c2w, degree)                    # (N, n_coeffs)
    colors = torch.einsum("nck,nk->nc", psi, phi).contiguous()  # (N, 3)
    with torch.no_grad():
        img, _, _ = gsplat.rasterization(
            means, t(checkpoint["rotations"]), t(checkpoint["scales"]), t(checkpoint["opacities"]),
            colors, t(_viewmat(c2w))[None], t(K)[None], width=width, height=height, sh_degree=None)
    return img[0].cpu().numpy()      # (H, W, 3), a predicted sigma up to one global factor


def uniform_coverage_sh(checkpoint, frames, K, width, height, degree, lam_diag,
                        device="cuda", camera_indices=None, visibility_eps=1e-6):
    """Han-style ablation: our own construction with every observation
    weighted EQUALLY instead of by its real compositing weight `beta^2`.

    `D_i = sum_p 1[splat i visible in p] phi(d_{i,p}) phi(d_{i,p})^T`, i.e. a
    pure angular-coverage statistic -- how many directions was this splat seen
    from, and how spread out were they -- with no information about how much
    it actually contributed to any pixel. Returns precision matrices
    `(n_splats, n_coeffs, n_coeffs)` with `lam_diag` already added, ready to
    sample from exactly like ours.
    """
    from gs_experiment.rasterized_sh_precision import probe_squared_footprint_weights

    t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    means, quats = t(checkpoint["positions"]), t(checkpoint["rotations"])
    scales, ops = t(checkpoint["scales"]), t(checkpoint["opacities"])
    Ks = t(K)[None]
    n_coeffs = lam_diag.shape[-1]
    precision = torch.zeros((means.shape[0], n_coeffs, n_coeffs), dtype=torch.float64, device=device)
    generator = torch.Generator(device=device)
    selected = range(len(frames)) if camera_indices is None else [int(c) for c in camera_indices]
    for p in selected:
        _, c2w = frames[p]
        generator.manual_seed(p)
        _, sum_beta = probe_squared_footprint_weights(
            means, quats, scales, ops, t(_viewmat(c2w))[None], Ks, width, height,
            n_probes=1, generator=generator)
        visible = sum_beta > visibility_eps          # the rasterizer's own visibility, weight discarded
        if not bool(visible.any()):
            continue
        phi = _camera_basis(means, c2w, degree)[visible].double()
        idx = torch.nonzero(visible, as_tuple=True)[0]
        precision.index_add_(0, idx, phi[:, :, None] * phi[:, None, :])
    return precision.cpu().numpy()
