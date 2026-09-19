"""Diagonal Fisher information over EVERY Gaussian parameter, and the
posterior ensemble it implies -- a FisherRF/OUGS-style baseline
(`jiang2024`, `li2026ougs`) built with this project's own rasterizer-native
machinery so it is scored on identical footing.

Why this baseline specifically. Our construction puts a posterior on the SH
appearance coefficients only. FisherRF computes a per-primitive information
quantity over all parameters with a bespoke CUDA kernel; OUGS accumulates a
diagonal Fisher over position, scale, rotation, opacity AND appearance as an
EMA of squared gradients during training, then propagates it to pixels
through the rendering Jacobian. Both bet that geometry belongs in the
uncertainty. FINDINGS section 7 found the opposite for CALIBRATION -- adding
an opacity posterior to ours made it worse. This baseline is the controlled
test of that disagreement: identical checkpoints, identical probes, identical
scoring, the only difference being which parameters carry the posterior.

Two honest differences from the originals, both stated rather than hidden:

  * OUGS accumulates its Fisher DURING training as an EMA of squared
    gradients. That is unavailable post-hoc on a frozen checkpoint, so here
    the same diagonal is estimated directly on the finished map by Rademacher
    probes -- which is the cleaner estimator of the same quantity (an exact
    unbiased estimate of `sum_q (dC/dtheta_k)^2` rather than a decaying
    average over a moving parameter trajectory), but it is not identical.
  * FisherRF uses a custom kernel for the exact diagonal. The probe estimator
    here is unbiased but Monte-Carlo, so it carries variance a bespoke exact
    kernel would not.

Parameterisation matters and follows training, not the .ply: positions raw,
scales in LOG space, opacity in LOGIT space, quaternions perturbed then
renormalised. Perturbing a positive scale or a bounded opacity additively
would put mass outside the feasible set and flatter or wreck the baseline
for a reason that has nothing to do with its uncertainty model.
"""

from __future__ import annotations

import numpy as np
import torch

PARAM_KEYS = ("means", "log_scales", "quats", "logit_opacity", "sh")


def _unconstrained(checkpoint, device):
    """The parameters training actually optimises, as leaf tensors."""
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=device)  # noqa: E731
    o = np.clip(checkpoint["opacities"], 1e-6, 1 - 1e-6)
    return {
        "means": t(checkpoint["positions"]),
        "log_scales": t(np.log(np.maximum(checkpoint["scales"], 1e-12))),
        "quats": t(checkpoint["rotations"]),
        "logit_opacity": t(np.log(o / (1 - o))),
        "sh": t(checkpoint["sh_coeffs"]).transpose(1, 2).contiguous(),   # (N, K, 3) for gsplat
    }


def _render_from_unconstrained(params, viewmat, Ks, width, height, sh_degree, background):
    import gsplat

    quats = params["quats"] / params["quats"].norm(dim=1, keepdim=True).clamp_min(1e-12)
    image, _, _ = gsplat.rasterization(
        params["means"], quats, torch.exp(params["log_scales"]),
        torch.sigmoid(params["logit_opacity"]), params["sh"],
        viewmat, Ks, width=width, height=height, sh_degree=sh_degree, backgrounds=background)
    return image


def accumulate_parameter_fisher(checkpoint, frames, K, width, height, n_probes=16, seed=0,
                                device="cuda", background_color=(1.0, 1.0, 1.0),
                                camera_indices=None, progress_every=0):
    """`F_k = sum_p sum_{q,c} (dC_c(q)/dtheta_k)^2` for every parameter, by
    Rademacher image probes backpropagated through the real rasterizer.
    Returns a dict keyed by `PARAM_KEYS`, each the shape of that parameter.

    Unlike the SH-only accumulation the probes cannot ride as extra render
    channels here: the forward is the real 3-channel image, not an arbitrary
    feature, so each probe is its own backward over a retained graph."""
    from gs_experiment.nerf_transforms import opencv_viewmat_from_c2w

    params = _unconstrained(checkpoint, device)
    for v in params.values():
        v.requires_grad_(True)
    Ks_t = torch.tensor(K, dtype=torch.float32, device=device)[None]
    background = torch.tensor(background_color, dtype=torch.float32, device=device)
    generator = torch.Generator(device=device)
    fisher = {k: torch.zeros_like(v, dtype=torch.float64) for k, v in params.items()}

    selected = range(len(frames)) if camera_indices is None else [int(c) for c in camera_indices]
    for n_done, p in enumerate(selected):
        _, c2w = frames[p]
        generator.manual_seed(seed * 1_000_003 + p)
        viewmat = torch.tensor(opencv_viewmat_from_c2w(c2w), dtype=torch.float32, device=device)[None]
        image = _render_from_unconstrained(params, viewmat, Ks_t, width, height,
                                           checkpoint["sh_degree"], background)
        for j in range(n_probes):
            for v in params.values():
                v.grad = None
            r = torch.randint(0, 2, image.shape, generator=generator, device=device,
                              dtype=torch.float32) * 2 - 1
            (image * r).sum().backward(retain_graph=(j < n_probes - 1))
            for k, v in params.items():
                if v.grad is not None:
                    fisher[k] += v.grad.double() ** 2
        if progress_every and (n_done + 1) % progress_every == 0:
            print(f"    parameter-Fisher camera {n_done + 1}/{len(selected)}")
    return {k: (v / n_probes).cpu().numpy() for k, v in fisher.items()}


def empirical_parameter_precision(checkpoint):
    """Empirical-Bayes prior precision per parameter GROUP: the reciprocal of
    that group's population variance across splats, in the same unconstrained
    space the Fisher is computed in. Same rule already used for the SH bands
    and for logit-opacity, applied uniformly so no group gets a hand-picked
    prior the others do not."""
    o = np.clip(checkpoint["opacities"], 1e-6, 1 - 1e-6)
    groups = {
        "means": checkpoint["positions"],
        "log_scales": np.log(np.maximum(checkpoint["scales"], 1e-12)),
        "quats": checkpoint["rotations"],
        "logit_opacity": np.log(o / (1 - o)),
        "sh": checkpoint["sh_coeffs"],
    }
    return {k: float(1.0 / max(np.var(v), 1e-12)) for k, v in groups.items()}


def sample_parameter_draws(checkpoint, fisher, priors, noise_var, n_draws, seed=0, device="cuda"):
    """`n_draws` parameter sets drawn from the diagonal Gaussian posterior
    `N(theta_hat, (F/sigma_n^2 + lambda)^-1)`, returned ready for gsplat.

    Centred on the checkpoint, so the ensemble mean stays the real render.
    Scales come back through `exp`, opacity through `sigmoid`, quaternions
    renormalised -- every draw is a valid scene."""
    generator = torch.Generator(device=device).manual_seed(seed)
    base = _unconstrained(checkpoint, device)
    draws = []
    for _ in range(n_draws):
        pert = {}
        for k, v in base.items():
            f = torch.tensor(np.asarray(fisher[k]), dtype=torch.float32, device=device)
            std = (f / noise_var + priors[k]).clamp_min(1e-12).rsqrt()
            pert[k] = v + std * torch.randn(v.shape, generator=generator,
                                            dtype=torch.float32, device=device)
        quats = pert["quats"] / pert["quats"].norm(dim=1, keepdim=True).clamp_min(1e-12)
        draws.append({
            "means": pert["means"], "quats": quats,
            "scales": torch.exp(pert["log_scales"]),
            "opacities": torch.sigmoid(pert["logit_opacity"]),
            "sh": pert["sh"],
        })
    return draws


def render_draw(draw, viewmat, Ks, width, height, sh_degree, background):
    import gsplat

    with torch.no_grad():
        image, _, _ = gsplat.rasterization(
            draw["means"], draw["quats"], draw["scales"], draw["opacities"], draw["sh"],
            viewmat, Ks, width=width, height=height, sh_degree=sh_degree, backgrounds=background)
    return image[0].clamp(0, 1).cpu().numpy()
