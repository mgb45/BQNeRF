"""Cross-splat coupled SH posterior: sampling from the FULL joint posterior
over every splat's coefficients at once, without ever forming or inverting it.

Why this is needed. `rasterized_sh_precision` builds the block-diagonal
posterior -- every splat's coefficients treated as independent. That is the
wrong structure, and visibly so: the posterior draws it produces show
chromatic *speckle*, because each splat's colour wobbles on its own.
Photometric training only ever constrains the SUM of contributions along a
ray, so splats in an overlapping stack are jointly non-identifiable. The
block diagonal is an upper bound on precision, and therefore UNDERESTIMATES
variance exactly in the ambiguous regions the method exists to flag.

The true stacked precision over `theta = (theta_1, ..., theta_N)` is

    A = Lambda + (1/sigma_n^2) sum_p g_p g_p^T,
    g_p = vec_i( beta_{p,i} phi(d_{i,p}) )        (p ranges over TRAINING pixels)

which is enormous (N*K per channel; ~4.8M for a 300k-splat degree-3 scene)
but never has to be formed, because **every operation it needs is a render**:

* **Matvec.** `g_p^T v = sum_i beta_{p,i} (phi(d_{i,c})^T v_i)` is literally
  the rendered image when each splat is given the scalar feature
  `phi(d_{i,c})^T v_i`. And `sum_p g_p s_p` is the BACKWARD pass of that same
  render against the image `s`. Taking `s` to be the rendered image itself,
  the whole data term of `A v` is the gradient of `0.5 * ||render||^2` --
  one forward+backward per training camera, nothing else.
* **Sampling.** Drawing `x ~ N(0, A^-1)` needs no square root and no
  inversion. Draw `b ~ N(0, A)` in closed form, which is possible because
  every term of `A` is known:

      b = Lambda^{1/2} r_0 + (1/sigma_n) sum_p g_p eps_p,   r_0, eps_p ~ N(0, I)

  and `sum_p g_p eps_p` is again just a backward pass against a white-noise
  image. Then solve `A x = b`; since `Cov(b) = A`, `Cov(x) = A^-1 A A^-1 =
  A^-1` exactly. (This is the standard randomize-then-optimize / perturbation
  sampler for Gaussian posteriors.)

The solve is preconditioned conjugate gradients, and the natural
preconditioner is already built: the block-diagonal precision from
`rasterized_sh_precision` is exactly `diag(A)` in the block sense. All draws
and all three colour channels are solved simultaneously as multiple
right-hand sides sharing one matvec, so the cost of an ensemble is
`n_cg_iters` render-pairs per camera in total, not per draw.

Note `Lambda` differs per channel (empirical-Bayes prior precisions are fit
per band AND per channel) while the data term does not depend on channel at
all -- `beta` is geometric. So one shared render-based matvec serves every
right-hand side, and only the cheap diagonal `Lambda v` part varies.
"""

from __future__ import annotations

import numpy as np
import torch

from gs_experiment.rasterized_sh_precision import _sh_basis_torch


def _camera_basis(means: torch.Tensor, c2w, degree: int) -> torch.Tensor:
    """phi(d_{i,c}) for every splat, at gsplat's own SH direction for this
    camera (`normalize(means - campos)`)."""
    campos = torch.tensor(np.asarray(c2w)[:3, 3], dtype=means.dtype, device=means.device)
    dirs = means - campos
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return _sh_basis_torch(dirs, degree)


def _data_matvec(v, means, quats, scales, opacities, frames, Ks, width, height, degree, device):
    """`(1/sigma_n^2)`-free data part of `A v`: `sum_p g_p (g_p^T v)`,
    accumulated over every training camera as one forward+backward each.

    `v`: (N, R, K) -- R right-hand sides at once. Returns the same shape."""
    import gsplat

    out = torch.zeros_like(v)
    for _, c2w in frames:
        phi = _camera_basis(means, c2w, degree)                       # (N, K)
        colors = torch.einsum("nrk,nk->nr", v, phi).contiguous()      # (N, R)
        colors = colors.detach().requires_grad_(True)
        viewmat = torch.tensor(_viewmat(c2w), dtype=torch.float32, device=device)[None]
        image, _, _ = gsplat.rasterization(means, quats, scales, opacities, colors,
                                           viewmat, Ks, width=width, height=height, sh_degree=None)
        # d/dc of 0.5*||render||^2 is exactly sum_p beta_{p,i} (g_p^T v).
        (0.5 * (image ** 2).sum()).backward()
        out += colors.grad[:, :, None] * phi[:, None, :]
    return out


def _viewmat(c2w):
    from gs_experiment.nerf_transforms import opencv_viewmat_from_c2w
    return opencv_viewmat_from_c2w(c2w)


def _sample_b(means, quats, scales, opacities, frames, Ks, width, height, degree,
              lam, noise_std, n_rhs, generator, device):
    """`b ~ N(0, A)`, drawn in closed form: `Lambda^{1/2} r_0 + (1/sigma_n)
    sum_p g_p eps_p`, the second term being a backward pass against a
    white-noise image per training camera."""
    import gsplat

    n_splats, n_coeffs = means.shape[0], lam.shape[-1]
    b = torch.sqrt(lam)[None, :, :] * torch.randn(
        (n_splats, n_rhs, n_coeffs), generator=generator, dtype=torch.float32, device=device)

    ones = torch.ones((n_splats, n_rhs), dtype=torch.float32, device=device)
    for _, c2w in frames:
        phi = _camera_basis(means, c2w, degree)
        colors = ones.clone().detach().requires_grad_(True)
        viewmat = torch.tensor(_viewmat(c2w), dtype=torch.float32, device=device)[None]
        image, _, _ = gsplat.rasterization(means, quats, scales, opacities, colors,
                                           viewmat, Ks, width=width, height=height, sh_degree=None)
        eps = torch.randn((1, height, width, n_rhs), generator=generator,
                          dtype=torch.float32, device=device)
        (image * eps).sum().backward()   # -> colors.grad[i, r] = sum_q beta_{q,i} eps_{q,r}
        b += (colors.grad[:, :, None] / noise_std) * phi[:, None, :]
    return b


def sample_coupled_perturbations(
    checkpoint, frames, K, width, height, degree, band_precision, noise_var,
    block_precision, n_draws=8, seed=0, n_cg_iters=40, device="cuda", verbose=True,
):
    """`n_draws` samples from the FULL coupled posterior `N(0, A^-1)`, as
    `(n_draws, n_splats, 3, n_coeffs)` perturbations to add to the
    checkpoint's own SH coefficients.

    `band_precision` (3, n_coeffs): empirical-Bayes prior precisions, per
    channel (`render_posterior_ensemble.empirical_band_precision`).
    `block_precision` (n_splats, n_coeffs, n_coeffs): the block-diagonal
    precision from `rasterized_sh_precision`, already divided by
    `sigma_n^2` and with the prior added -- used ONLY as the CG
    preconditioner, where it is exactly the block diagonal of `A`.

    All `3 * n_draws` right-hand sides are solved at once against one shared
    render-based matvec."""
    n_splats = checkpoint["positions"].shape[0]
    n_coeffs = band_precision.shape[1]
    n_rhs = 3 * n_draws
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    means, quats = t(checkpoint["positions"]), t(checkpoint["rotations"])
    scales, opacities = t(checkpoint["scales"]), t(checkpoint["opacities"])
    Ks = t(K)[None]
    generator = torch.Generator(device=device).manual_seed(seed)
    noise_std = float(np.sqrt(noise_var))

    # Right-hand-side layout: rhs index r = draw * 3 + channel.
    lam = t(np.repeat(band_precision[None, :, :], n_draws, axis=0).reshape(n_rhs, n_coeffs))
    inv_noise_var = 1.0 / noise_var

    def A(v):
        data = _data_matvec(v, means, quats, scales, opacities, frames, Ks,
                            width, height, degree, device)
        return lam[None, :, :] * v + inv_noise_var * data

    # Block-diagonal preconditioner: one batched Cholesky per channel, reused
    # for every CG iteration and every draw of that channel.
    chol = torch.linalg.cholesky(t(block_precision))  # (N, K, K)

    def M_inv(v):
        # (N, R, K) -> (N, K, R) for the batched triangular solve and back.
        sol = torch.cholesky_solve(v.permute(0, 2, 1).contiguous(), chol)
        return sol.permute(0, 2, 1).contiguous()

    b = _sample_b(means, quats, scales, opacities, frames, Ks, width, height, degree,
                  lam, noise_std, n_rhs, generator, device)

    # Preconditioned CG, one independent solve per right-hand side sharing
    # every matvec (scalars are per-RHS, the expensive operator call is not).
    x = torch.zeros_like(b)
    r = b.clone()
    z = M_inv(r)
    p = z.clone()
    rz = (r * z).sum(dim=(0, 2))
    b_norm = b.norm(dim=(0, 2)).clamp_min(1e-30)
    for it in range(n_cg_iters):
        # A column whose residual has already converged legitimately drives rz
        # and pAp to zero. Freeze those columns outright rather than dividing
        # by a near-zero denominator: only an UNCONVERGED column with a
        # non-positive form is a real breakdown of the SPD assumption.
        active = (r.norm(dim=(0, 2)) / b_norm) > 1e-6
        if not bool(active.any()):
            break
        Ap = A(p)
        pAp = (p * Ap).sum(dim=(0, 2))
        if bool((active & ((pAp <= 0) | (rz <= 0))).any()):
            print(f"    CG BREAKDOWN at iteration {it + 1}: an unconverged right-hand side has "
                  f"p^T A p = {pAp.min().item():.3e} or r^T z = {rz.min().item():.3e}; both must "
                  f"be > 0 for SPD A. Stopping -- the iterate is not a valid solve.")
            break
        ones = torch.ones_like(rz)
        alpha = torch.where(active, rz / torch.where(active, pAp, ones), torch.zeros_like(rz))
        x += alpha[None, :, None] * p
        r -= alpha[None, :, None] * Ap
        rel = (r.norm(dim=(0, 2)) / b_norm).max().item()
        if verbose and (it % 25 == 0 or it == n_cg_iters - 1):
            print(f"    CG {it + 1}/{n_cg_iters}: max rel residual {rel:.3e}  "
                  f"min pAp {pAp.min().item():.3e}  min rz {rz.min().item():.3e}")
        if rel < 1e-6:
            break
        z = M_inv(r)
        rz_new = (r * z).sum(dim=(0, 2))
        beta = torch.where(active, rz_new / torch.where(active, rz, ones), torch.zeros_like(rz))
        p = z + beta[None, :, None] * p
        rz = rz_new
    else:
        if rel > 1e-3:
            print(f"    WARNING: CG did not converge (max relative residual {rel:.3e} after "
                  f"{n_cg_iters} iterations). A truncated CG solve started from x=0 UNDERSTATES "
                  f"the solution magnitude, so these draws understate the posterior covariance "
                  f"and must not be reported as samples from N(0, A^-1).")

    return x.reshape(n_splats, n_draws, 3, n_coeffs).permute(1, 0, 2, 3).contiguous()
