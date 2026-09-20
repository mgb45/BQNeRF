"""Per-splat SH Fisher information, read straight off the real rasterizer.

This replaces `gpu_sh_directional_uncertainty.accumulate_sh_precision`'s
KNN surrogate, which was measuring essentially nothing. Two independent
defects compounded there, both confirmed directly against gsplat on the
real 300k-splat lego `wide` checkpoint:

1. **Centre-pixel-only.** `beta_{p,i}` was the splat's alpha-compositing
   weight at the single pixel its own projected centre landed on. But a
   training observation constrains splat i through EVERY pixel it touches,
   so the information it carries is `sum_q beta_{q,i}^2` over its whole
   real footprint, not one sample of it. A splat covering 400 pixels and
   one covering 1 were being given comparable weight.
2. **The bearing ball counted non-occluders as occluders.** Candidates came
   from a 0.05 rad (~2.9 deg) bearing ball -- vastly wider than a pixel --
   capped at the 500 nearest. Every one of those hundreds of splats then
   entered the depth-ordered transmittance product as though it occluded
   the query, collapsing `T` to ~0 for almost every splat.

Measured consequence: the surrogate's median `beta` over genuinely visible
splats was 1.1e-12 (numerically zero) where the real rasterizer gives those
same splats a footprint-summed `sum_q beta^2` of 6.5e-5 -- a median ratio of
**2.1e18**. Downstream, the accumulated data term came out ~1e-13 times the
prior precision in every SH band, for every splat: the "posterior" over SH
coefficients was the prior, exactly, and `u_SH` was measuring nothing but
query-side coverage. (That is also the real explanation for the
hand-picked-`lam` trouble recorded in `render_angle_sweep.py`'s LAM comment
-- no scalar prior precision can be balanced against a data term that is 13
orders of magnitude too small.)

The fix uses the rasterizer itself. For per-splat features `c_i`, gsplat
renders `I(q) = sum_i beta_{q,i} c_i`, so backpropagating an arbitrary image
`r` gives, exactly and for free,

    dL/dc_i = sum_q beta_{q,i} r_q            (L = sum_q r_q I(q))

Take `r` to be Rademacher (+-1) noise: `E[(dL/dc_i)^2] = sum_q beta_{q,i}^2`,
the quantity actually wanted, footprint-summed, under the REAL rasterizer's
own weights -- no bearing ball, no candidate cap, no surrogate compositing
model, and no approximation beyond Monte-Carlo error in the probe average.
Independent probes are rendered as independent CHANNELS of one render
(`beta` is geometric -- `T_i*alpha_i` -- so it is identical across
channels), making the whole thing one forward+backward pass per training
camera.

The block-diagonal Fisher information for splat i then follows directly
from the Gaussian likelihood `I(q) = sum_i beta_{q,i} phi(d_i)^T theta_i +
eps`, `eps ~ N(0, sigma_n^2)`:

    D_i = sum_p (sum_q beta_{q,i,p}^2) phi(d_{i,p}) phi(d_{i,p})^T
    P_i = Lambda + D_i / sigma_n^2

`sigma_n^2` is NOT optional here and was missing before: without it the data
term is expressed in squared-colour units while the prior is in coefficient
units, so the two cannot be balanced at all. `estimate_noise_variance`
fits it from the checkpoint's own real training residuals.

Direction convention: `phi` is evaluated at gsplat's own SH direction,
`normalize(means - campos)` (camera -> splat), NOT this project's usual
`camera.center - positions` (splat -> camera, `camera.
directions_from_positions_to_camera`). The two differ by a sign flip on the
odd SH bands, `phi(-d) = S phi(d)` with `S = diag(+-1)`, and since `S^2 = I`
that flip cancels exactly in `phi^T Sigma phi` whenever BOTH sides use the
same convention -- which is why the existing pipeline is self-consistent
and not buggy. It matters here only because these coefficients are sampled
and handed back to gsplat to render, so every step now uses the renderer's
own convention and nothing has to cancel.
"""

from __future__ import annotations

import numpy as np
import torch

from gs_experiment.spherical_harmonics import N_COEFFS_FOR_DEGREE


def _sh_basis_torch(directions: torch.Tensor, degree: int) -> torch.Tensor:
    """Same real SH basis as `sh_directional_uncertainty.sh_basis`, on GPU.
    `directions` (..., 3) must already be unit."""
    from gs_experiment.spherical_harmonics import SH_C0, SH_C1, SH_C2, SH_C3

    x, y, z = directions[..., 0], directions[..., 1], directions[..., 2]
    cols = [torch.full_like(x, SH_C0)]
    if degree > 0:
        cols += [-SH_C1 * y, SH_C1 * z, -SH_C1 * x]
        if degree > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            cols += [SH_C2[0] * xy, SH_C2[1] * yz, SH_C2[2] * (2.0 * zz - xx - yy),
                     SH_C2[3] * xz, SH_C2[4] * (xx - yy)]
            if degree > 2:
                cols += [SH_C3[0] * y * (3 * xx - yy), SH_C3[1] * xy * z,
                         SH_C3[2] * y * (4 * zz - xx - yy),
                         SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy),
                         SH_C3[4] * x * (4 * zz - xx - yy), SH_C3[5] * z * (xx - yy),
                         SH_C3[6] * x * (xx - 3 * yy)]
    return torch.stack(cols, dim=-1)


def probe_squared_footprint_weights(
    means, quats, scales, opacities, viewmat, Ks, width, height,
    n_probes: int = 32, generator=None,
):
    """`(sum_q beta_{q,i}^2, sum_q beta_{q,i})` per splat for one camera, via
    `n_probes` Rademacher image probes backpropagated through the real
    rasterizer (all probes carried as channels of a SINGLE render -- see
    module docstring).

    The second return, the plain footprint total `sum_q beta_{q,i}`, comes
    from the same backward with an all-ones image and is what "did this
    camera actually see this splat" should mean: the rasterizer's own
    answer, not a frustum-and-min-opacity proxy."""
    import gsplat

    n_splats = means.shape[0]
    device, dtype = means.device, means.dtype
    colors = torch.ones((n_splats, n_probes + 1), dtype=dtype, device=device, requires_grad=True)

    image, _, _ = gsplat.rasterization(
        means, quats, scales, opacities, colors, viewmat, Ks,
        width=width, height=height, sh_degree=None,
    )  # (1, H, W, n_probes + 1)

    probe = torch.randint(
        0, 2, (1, height, width, n_probes), generator=generator, device=device, dtype=dtype
    ) * 2 - 1
    ones = torch.ones((1, height, width, 1), dtype=dtype, device=device)
    (image * torch.cat([probe, ones], dim=-1)).sum().backward()

    grad = colors.grad.detach()
    sum_beta_sq = (grad[:, :n_probes].double() ** 2).mean(dim=1)
    sum_beta = grad[:, n_probes].double()
    return sum_beta_sq, sum_beta


def accumulate_sh_precision_rasterized(
    checkpoint, frames, K, width, height, degree: int,
    n_probes: int = 32, seed: int = 0, device: str = "cuda", visibility_eps: float = 1e-6,
    progress_every: int = 20, camera_indices=None,
):
    """`D_i = sum_p (sum_q beta_{q,i,p}^2) phi(d_{i,p}) phi(d_{i,p})^T` over
    every real training camera in `frames`, accumulated on GPU.

    Returns `(n_splats, n_coeffs, n_coeffs)` float64 -- the raw Fisher
    information, with NO prior and NO `1/sigma_n^2` folded in, so both can be
    chosen (and re-chosen) afterwards without re-paying for this.

    One forward+backward render per camera, against the real rasterizer --
    versus one KNN candidate search per camera in the surrogate this
    replaces, which was both slower and measuring the wrong thing.

    `camera_indices`: optional iterable of indices into `frames`; when given,
    only those training cameras are accumulated. Each camera's probe noise is
    seeded from its own GLOBAL index (`seed`, camera index), not from a
    generator consumed in loop order, so a camera draws the same probes no
    matter which subset it appears in. Nested camera subsets are then nested
    in the Loewner order -- `D_i` can only shrink as cameras are removed --
    which is what makes the frozen-map monotonicity check
    (`scripts/frozen_map_monotonicity_test.py`) a test of this function
    rather than a test of Monte-Carlo noise.

    That nesting is NOT bitwise exact, and the reason is worth knowing:
    gsplat's backward accumulates per-splat gradients with float32 atomics,
    so it is not deterministic even for identical inputs and an identical
    seed -- measured at ~7.7e-8 relative, run to run, on a fixed camera.
    Per-camera seeding removes the dominant source of disagreement (order-
    dependent probe draws, ~30% relative); what remains is that atomic
    noise floor. Monotonicity checks must therefore use a relative
    tolerance around 1e-6, not machine epsilon."""
    from gs_experiment.nerf_transforms import opencv_viewmat_from_c2w

    n_coeffs = N_COEFFS_FOR_DEGREE[degree]
    means = torch.tensor(checkpoint["positions"], dtype=torch.float32, device=device)
    quats = torch.tensor(checkpoint["rotations"], dtype=torch.float32, device=device)
    scales = torch.tensor(checkpoint["scales"], dtype=torch.float32, device=device)
    opacities = torch.tensor(checkpoint["opacities"], dtype=torch.float32, device=device)
    Ks_t = torch.tensor(K, dtype=torch.float32, device=device)[None]
    generator = torch.Generator(device=device)

    n_splats = means.shape[0]
    precision = torch.zeros((n_splats, n_coeffs, n_coeffs), dtype=torch.float64, device=device)

    selected = range(len(frames)) if camera_indices is None else [int(c) for c in camera_indices]
    for n_done, p in enumerate(selected):
        _, c2w = frames[p]
        # Per-camera seed from the camera's own global index -- see docstring.
        generator.manual_seed(seed * 1_000_003 + p)
        viewmat = torch.tensor(opencv_viewmat_from_c2w(c2w), dtype=torch.float32, device=device)[None]
        sum_beta_sq, sum_beta = probe_squared_footprint_weights(
            means, quats, scales, opacities, viewmat, Ks_t, width, height, n_probes, generator
        )
        visible = sum_beta > visibility_eps
        if not bool(visible.any()):
            continue

        # gsplat's OWN SH direction: normalize(means - campos). See module docstring.
        campos = torch.tensor(np.asarray(c2w)[:3, 3], dtype=torch.float32, device=device)
        dirs = means[visible] - campos
        dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-12)
        phi = _sh_basis_torch(dirs, degree).double()  # (n_vis, n_coeffs)

        w = sum_beta_sq[visible]
        precision.index_add_(0, torch.nonzero(visible, as_tuple=True)[0],
                             w[:, None, None] * (phi[:, :, None] * phi[:, None, :]))
        if progress_every and (n_done + 1) % progress_every == 0:
            print(f"    camera {n_done + 1}/{len(selected)}: {int(visible.sum())} visible splats")

    return precision.cpu().numpy()


def estimate_noise_variance(checkpoint, frames, K, width, height, image_dir,
                            n_views: int = 10, device: str = "cuda", background_color=(1.0, 1.0, 1.0)):
    """`sigma_n^2`, the per-pixel-per-channel observation noise the Gaussian
    likelihood above assumes, fit as the mean squared residual between the
    real render and the real training image over `n_views` evenly-spaced
    training views.

    This is what puts the data term and the prior into the same units. It is
    a *residual* variance, so it absorbs everything the model genuinely
    cannot explain (unmodelled view-dependence, sensor noise, the finite
    splat budget) -- which is exactly what should temper how much a training
    observation is allowed to sharpen a coefficient posterior."""
    import gsplat
    import os
    from PIL import Image
    from gs_experiment.nerf_transforms import opencv_viewmat_from_c2w

    means = torch.tensor(checkpoint["positions"], dtype=torch.float32, device=device)
    quats = torch.tensor(checkpoint["rotations"], dtype=torch.float32, device=device)
    scales = torch.tensor(checkpoint["scales"], dtype=torch.float32, device=device)
    opacities = torch.tensor(checkpoint["opacities"], dtype=torch.float32, device=device)
    sh = torch.tensor(checkpoint["sh_coeffs"], dtype=torch.float32, device=device).transpose(1, 2)
    Ks_t = torch.tensor(K, dtype=torch.float32, device=device)[None]
    background = torch.tensor(background_color, dtype=torch.float32, device=device)

    picks = np.linspace(0, len(frames) - 1, min(n_views, len(frames))).astype(int)
    sq = []
    with torch.no_grad():
        for i in picks:
            file_path, c2w = frames[i]
            viewmat = torch.tensor(opencv_viewmat_from_c2w(c2w), dtype=torch.float32, device=device)[None]
            rendered, _, _ = gsplat.rasterization(
                means, quats, scales, opacities, sh, viewmat, Ks_t,
                width=width, height=height, sh_degree=checkpoint["sh_degree"], backgrounds=background,
            )
            recon = rendered[0].clamp(0, 1).cpu().numpy()
            gt = np.asarray(Image.open(os.path.join(image_dir, file_path + ".png")).convert("RGB"),
                            dtype=np.float32) / 255.0
            sq.append(np.mean((recon - gt) ** 2))
    return float(np.mean(sq))


def pixel_weight_concentration(means, quats, scales, opacities, viewmat, Ks, width, height,
                               n_probes: int = 32, generator=None, sh=None, sh_degree=None):
    """`sum_i beta_{q,i}^2` per PIXEL -- the forward twin of
    `probe_squared_footprint_weights` (which gives the same quantity summed
    the other way, per splat).

    Rendering per-splat Rademacher features `r_i` gives
    `I(q) = sum_i beta_{q,i} r_i`, so `E[I(q)^2] = sum_i beta_{q,i}^2`. One
    render with the probes as channels; no backward pass needed.

    This is a resolution statistic, not an uncertainty: it is large where one
    splat dominates a pixel (the local representation is coarse relative to
    the detail there) and small where many splats blend smoothly. That makes
    it a natural regressor for the ALEATORIC part of pixel error -- the
    misspecification an epistemic posterior over fitted parameters cannot
    see, because the training views really do determine those parameters.
    """
    import gsplat

    n_splats = means.shape[0]
    probes = torch.randint(0, 2, (n_splats, n_probes), generator=generator,
                           device=means.device, dtype=means.dtype) * 2 - 1
    with torch.no_grad():
        image, _, _ = gsplat.rasterization(means, quats, scales, opacities, probes, viewmat, Ks,
                                           width=width, height=height, sh_degree=None)
    return (image[0] ** 2).mean(dim=-1)   # (H, W)


def fit_band_precision_evidence(sh_coeffs, data_precision, n_iters: int = 30,
                                device: str = "cuda", chunk: int = 200_000, verbose: bool = False):
    """Type-II maximum likelihood (MacKay evidence) fit of the per-band,
    per-channel prior precision, replacing the population-variance rule
    `empirical_band_precision`.

    Why this exists. The population rule sets `lambda_l = 1/Var_i[theta_i,l]`
    across all splats. On object-centric synthetic scenes that is a sensible
    scale, but it ignores the data entirely: the variance it measures includes
    all the spread the training views ALREADY explain. On a real unbounded
    capture -- 1.07M splats spanning foreground, room and far-field background
    -- that population variance is enormous, `lambda` comes out tiny, and the
    prior swamps the data term for essentially every splat. The symptom is a
    posterior that is nearly flat: on Mip-NeRF 360 bonsai our sigma had a
    2.3x dynamic range against U-3DGS's 7.2x, and was only 1.09x larger on the
    worst-5% error pixels against their 3.49x.

    MacKay's update fixes exactly that, by measuring how much of each
    coefficient the data actually determines:

        gamma_l   = sum_i sum_{k in band l} (1 - lambda_l * Sigma_i,kk)
        lambda_l <- gamma_l / sum_i sum_{k in band l} (theta_i,k - mu_l)^2

    `gamma_l` is the effective number of well-determined parameters in that
    band. Where the data constrains a coefficient, `Sigma_kk` is small,
    `gamma` approaches its count, and the prior is allowed to be loose; where
    it does not, `gamma` falls and the prior tightens. Coefficients are
    centred on their band/channel population mean, so the prior is
    hierarchical -- shrinkage is toward the population, not toward black.

    `data_precision` must ALREADY be divided by sigma_n^2, i.e. be the same
    matrix the posterior uses. Returns `(3, n_coeffs)` to drop straight into
    `sample_sh_draws` in place of `empirical_band_precision`.
    """
    theta = np.asarray(sh_coeffs, dtype=np.float64)          # (N, 3, K)
    n_splats, _, n_coeffs = theta.shape
    bands = [(lo, min(hi, n_coeffs)) for lo, hi in
             ((0, 1), (1, 4), (4, 9), (9, 16)) if lo < n_coeffs]
    out = np.empty((3, n_coeffs), dtype=np.float64)
    d_cpu = torch.tensor(np.asarray(data_precision), dtype=torch.float32)

    for c in range(3):
        lam = np.array([1.0 / max(np.var(theta[:, c, lo:hi]), 1e-12) for lo, hi in bands])
        mu = np.array([theta[:, c, lo:hi].mean() for lo, hi in bands])
        sq = np.array([float(((theta[:, c, lo:hi] - m) ** 2).sum())
                       for (lo, hi), m in zip(bands, mu)])
        counts = np.array([n_splats * (hi - lo) for lo, hi in bands], dtype=np.float64)

        for it in range(n_iters):
            diag_full = np.zeros(n_coeffs)
            lam_vec = np.zeros(n_coeffs)
            for b, (lo, hi) in enumerate(bands):
                lam_vec[lo:hi] = lam[b]
            lam_t = torch.tensor(lam_vec, dtype=torch.float32, device=device)
            for start in range(0, n_splats, chunk):                # chunked: (N,K,K) inverse
                blk = d_cpu[start:start + chunk].to(device)
                sigma_diag = torch.diagonal(
                    torch.linalg.inv(blk + torch.diag(lam_t)), dim1=-2, dim2=-1)
                diag_full += sigma_diag.double().sum(dim=0).cpu().numpy()
                del blk, sigma_diag
            gamma = np.array([counts[b] - lam[b] * diag_full[lo:hi].sum()
                              for b, (lo, hi) in enumerate(bands)])
            new = np.clip(gamma, 1e-6, None) / np.maximum(sq, 1e-12)
            if np.max(np.abs(np.log(new / lam))) < 1e-4:
                lam = new
                break
            lam = new
        if verbose:
            print(f"    channel {c}: lambda per band {np.round(lam, 4)} "
                  f"(population rule gave "
                  f"{np.round([1.0 / max(np.var(theta[:, c, lo:hi]), 1e-12) for lo, hi in bands], 4)})")
        for b, (lo, hi) in enumerate(bands):
            out[c, lo:hi] = lam[b]
    return out
