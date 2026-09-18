"""Per-splat OPACITY posterior, read off the real rasterizer the same way
`rasterized_sh_precision` reads the SH one.

Why. The posterior-ensemble uncertainty so far samples appearance only --
geometry (positions, scales, opacities) is frozen. So an error of geometric
origin, a floater sitting in the wrong place with a confidently-fit colour,
need not light up at all. That is the leading remaining suspect for the weak
object-pixel correlation with real held-out error (results/FINDINGS.md
section 5), now that cross-splat coupling has been implemented, validated
and found not to explain it (section 6).

Opacity is the cheapest piece of geometry to put back, and the one most
likely to matter: it is what distinguishes "this splat is really there" from
"this splat is a floater the training views never constrained".

Construction. Work in LOGIT space, `o_i = sigmoid(u_i)`. A Gaussian
posterior on `u` is unconstrained, so a draw can never leave `(0, 1)`, and
the prior can be fit by the same empirical-Bayes rule used for the SH bands
(`lambda_u = 1 / Var_i[logit(o_i)]`, read off the checkpoint). Linearizing
the render about the fitted opacities gives a Gaussian (Laplace) likelihood
whose per-splat Fisher information is

    F_i = sum_p sum_{q,c} ( dC_c(q) / du_i )^2,     P_i = lambda_u + F_i / sigma_n^2

and `dC_c(q)/du_i` is exactly what gsplat's backward returns when the
opacities are a leaf requiring grad, so -- as with the SH term -- the
quantity is read off the real renderer rather than modelled. Rademacher
probes on the image again turn "sum of squared derivatives over all pixels"
into a few backward passes: for `r ~ +-1`, `d/du_i sum_{q,c} r_{q,c} C_c(q)`
has expected square exactly `F_i`.

Unlike the SH case the probes CANNOT ride as extra channels: the forward
render has a fixed 3 channels here (it is the real image, not an arbitrary
feature), so each probe is its own backward pass over a retained graph.
That makes this term cost `n_probes` backwards per camera rather than one --
still seconds, not minutes.
"""

from __future__ import annotations

import numpy as np
import torch


def _logit(o: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    o = np.clip(np.asarray(o, dtype=np.float64), eps, 1.0 - eps)
    return np.log(o / (1.0 - o))


def empirical_opacity_precision(opacities: np.ndarray) -> float:
    """`lambda_u = 1 / Var_i[logit(o_i)]`, the empirical-Bayes prior
    precision on log-odds opacity, read off the checkpoint's own population.
    Same rule as the SH bands' `lambda_{l,c}`, and no hand-picked scalar."""
    return float(1.0 / max(np.var(_logit(opacities)), 1e-12))


def accumulate_opacity_information(
    checkpoint, frames, K, width, height, n_probes: int = 16, seed: int = 0,
    device: str = "cuda", camera_indices=None, background_color=(1.0, 1.0, 1.0),
    progress_every: int = 0,
):
    """`F_i = sum_p sum_{q,c} (dC_c(q)/du_i)^2` over the selected training
    cameras, by Rademacher probes backpropagated through the real rasterizer
    to logit-opacity. Returns `(n_splats,)` float64 -- raw Fisher
    information, with no prior and no `1/sigma_n^2` folded in.

    Each camera's probes are seeded from its own GLOBAL index, so a camera
    contributes the same estimated term in every subset it appears in (the
    same property `rasterized_sh_precision` relies on for nested-subset
    monotonicity)."""
    import gsplat
    from gs_experiment.nerf_transforms import opencv_viewmat_from_c2w

    t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    means, quats, scales = t(checkpoint["positions"]), t(checkpoint["rotations"]), t(checkpoint["scales"])
    sh = t(checkpoint["sh_coeffs"]).transpose(1, 2)
    Ks = t(K)[None]
    background = t(background_color)
    u0 = t(_logit(checkpoint["opacities"]))
    generator = torch.Generator(device=device)

    info = torch.zeros(means.shape[0], dtype=torch.float64, device=device)
    selected = range(len(frames)) if camera_indices is None else [int(c) for c in camera_indices]
    for n_done, p in enumerate(selected):
        _, c2w = frames[p]
        generator.manual_seed(seed * 1_000_003 + p)
        u = u0.clone().detach().requires_grad_(True)
        viewmat = t(opencv_viewmat_from_c2w(c2w))[None]
        image, _, _ = gsplat.rasterization(
            means, quats, scales, torch.sigmoid(u), sh, viewmat, Ks,
            width=width, height=height, sh_degree=checkpoint["sh_degree"], backgrounds=background,
        )
        for j in range(n_probes):
            u.grad = None
            r = torch.randint(0, 2, image.shape, generator=generator, device=device,
                              dtype=torch.float32) * 2 - 1
            (image * r).sum().backward(retain_graph=(j < n_probes - 1))
            info += u.grad.double() ** 2
        if progress_every and (n_done + 1) % progress_every == 0:
            print(f"    opacity camera {n_done + 1}/{len(selected)}")
    return (info / n_probes).cpu().numpy()


def sample_opacity_draws(opacities, opacity_info, lam_u, noise_var, n_draws, seed, device="cuda"):
    """`n_draws` opacity vectors drawn from the per-splat logit posterior
    `N(logit(o_i), 1/P_i)`, `P_i = lambda_u + F_i/sigma_n^2`, pushed back
    through the sigmoid. Returns `(n_draws, n_splats)` in `(0, 1)`.

    Centred on the checkpoint's own opacities, so the ensemble mean stays the
    real render and the spread is the posterior's."""
    generator = torch.Generator(device=device).manual_seed(seed)
    u0 = torch.tensor(_logit(opacities), dtype=torch.float32, device=device)
    precision = torch.tensor(lam_u + np.asarray(opacity_info) / noise_var,
                             dtype=torch.float32, device=device)
    std = precision.clamp_min(1e-12).rsqrt()
    noise = torch.randn((n_draws, u0.shape[0]), generator=generator, dtype=torch.float32, device=device)
    return torch.sigmoid(u0[None, :] + std[None, :] * noise)
