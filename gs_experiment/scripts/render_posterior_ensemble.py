"""Posterior-ensemble rendering: 3DGS as quadrature, uncertainty for the
price of k extra renders.

Alpha compositing IS a quadrature rule -- `C(q) = sum_i beta_{q,i} c_i(d_q)`,
nodes = splats, weights = `beta_{q,i} = T_i*alpha_i`, and the rasterizer
already computes the weights. So given a posterior over the splats' own SH
appearance coefficients, the per-pixel predictive variance is just that
posterior pushed through weights the renderer computes anyway. The cheapest
and most legible way to push it through is not a quadratic form at all --
it's sampling:

    draw theta^(s) ~ N(theta_hat, Sigma_theta),  render it with the REAL
    rasterizer, and take the per-pixel spread across draws.

No retraining, no model ensemble, one checkpoint, k renders. Regions the
training views actually constrain come out pixel-identical across draws;
regions they don't visibly disagree.

The posterior is the same per-splat Bayesian linear regression the rest of
this project uses (`sh_directional_uncertainty.py`):

    P_i^(c) = Lambda^(c) + sum_p beta_{p,i}^2 phi(d_p) phi(d_p)^T

with two changes from `render_sparse_gp_uncertainty.py`'s version:

1. `Lambda^(c)` is DIAGONAL-BY-SH-BAND AND PER-CHANNEL, fit by empirical
   Bayes from the checkpoint's own coefficient population
   (`empirical_band_precision` below), not a single hand-picked scalar
   `lam` shared across all 16 coefficients and all 3 channels. A single
   scalar is badly mis-specified here: the l=0 (DC colour) coefficients and
   the l=3 coefficients of a real checkpoint differ in natural scale by
   orders of magnitude, which is exactly why the hand-picked value had to
   be tuned between "dominates the data term, u_SH goes blind" (lam=10) and
   "prior saturates the image" (lam=1e-3) with nothing good in between --
   see `render_angle_sweep.py`'s own LAM comment. Empirical Bayes has no
   such knob: lambda_{l,c} = 1/Var_i[theta_{i,c,k}: k in band l], read
   straight off the checkpoint, per scene.
2. The data term `D_i = sum_p beta_{p,i}^2 phi phi^T` is accumulated ONCE
   (`accumulate_sh_precision` with `lam=0`) and cached to .npz, since it is
   the expensive part and is independent of the prior -- so re-fitting the
   prior, or re-drawing the ensemble, is free after the first run.

Run: .venv-gsplat/bin/python gs_experiment/scripts/render_posterior_ensemble.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from gs_experiment.rasterized_sh_precision import (
    accumulate_sh_precision_rasterized,
    estimate_noise_variance,
)
from gs_experiment.nerf_transforms import fov_x_to_intrinsics, load_transforms, opencv_viewmat_from_c2w
from gs_experiment.ply_io import read_3dgs_ply
from gs_experiment.scripts.render_reconstruction import EVAL_DIRS, LOCAL_RUNS, RESULTS_DIR

SCENE = "lego"
VIEW_IDX = 21  # this project's standard held-out lego view (render_sparse_gp_uncertainty.SCENE_VIEWS)
CHECKPOINT = "wide"
BACKGROUND_COLOR = (1.0, 1.0, 1.0)
ANGULAR_TOL = 0.05
MAX_CANDIDATES = 500
N_DRAWS = 8
N_PROBES = 32
SEED = 0
CACHE_DIR = LOCAL_RUNS / "posterior_cache"

# SH band layout for degree 3: coefficient index -> l. Bands are pooled
# within themselves (the 2l+1 coefficients of one band are rotations of
# each other, so they share a natural scale) but never across bands.
SH_BANDS = [(0, 1), (1, 4), (4, 9), (9, 16)]


def empirical_band_precision(sh_coeffs: np.ndarray, degree: int) -> np.ndarray:
    """lambda_{l,c} = 1 / Var_i[theta_{i,c,k} : k in band l], the empirical-
    Bayes prior precision, read off the checkpoint's own coefficient
    population. Returns `(3, n_coeffs)` of per-(channel, coefficient) prior
    precisions -- constant within an SH band, different across bands and
    across colour channels.

    `sh_coeffs`: (N, 3, n_coeffs), this project's on-disk convention.

    This replaces a hand-picked scalar `lam`. The variance is taken about
    the population MEAN, not about zero: a real checkpoint's DC band is not
    centred on black, and a zero-centred prior would call every splat's own
    base colour surprising. (Only the prior's covariance enters the
    posterior covariance, so the population mean never has to be formed --
    `np.var` handles it.)"""
    n_coeffs = sh_coeffs.shape[2]
    out = np.empty((3, n_coeffs), dtype=np.float64)
    for c in range(3):
        for lo, hi in SH_BANDS:
            if lo >= n_coeffs:
                break
            hi = min(hi, n_coeffs)
            var = float(np.var(sh_coeffs[:, c, lo:hi]))
            out[c, lo:hi] = 1.0 / max(var, 1e-12)
    return out


def load_or_build_data_precision(scene: str, checkpoint_name: str, checkpoint, frames, K, width, height):
    """`D_i = sum_p (sum_q beta_{q,i,p}^2) phi phi^T` plus `sigma_n^2`, both
    from the real rasterizer (`rasterized_sh_precision`) -- NOT the KNN
    surrogate `accumulate_sh_precision` uses, which measured the per-splat
    information low by a median factor of 2e18 on this very checkpoint (see
    that module's docstring). Cached together: neither depends on the prior,
    so refitting the prior or redrawing the ensemble never re-pays for them."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{scene}_{checkpoint_name}_rasterized_precision.npz"
    if cache_path.exists():
        print(f"loading cached rasterized data-term precision: {cache_path}")
        blob = np.load(cache_path)
        return blob["data_precision"], float(blob["noise_var"])

    train_dir = LOCAL_RUNS / f"{scene}_prepared" / checkpoint_name
    print(f"building rasterized data-term precision from {train_dir} ({len(frames)} training cameras)")
    t0 = time.time()
    data_precision = accumulate_sh_precision_rasterized(
        checkpoint, frames, K, width, height, checkpoint["sh_degree"],
        n_probes=N_PROBES, seed=SEED, device="cuda",
    )
    print(f"  accumulate_sh_precision_rasterized done in {time.time() - t0:.1f}s")
    noise_var = estimate_noise_variance(checkpoint, frames, K, width, height, str(train_dir),
                                        background_color=BACKGROUND_COLOR)
    print(f"  sigma_n^2 (mean squared training residual) = {noise_var:.6g}  (sigma_n = {np.sqrt(noise_var):.4g})")
    np.savez_compressed(cache_path, data_precision=data_precision, noise_var=noise_var)
    return data_precision, noise_var


def sample_sh_draws(sh_coeffs, data_precision, band_precision, n_draws, seed, device="cuda"):
    """`n_draws` samples from the per-splat, per-channel SH posterior
    `N(theta_hat, P^-1)` with `P^(c) = diag(lambda^(c)) + D`, drawn as
    `theta_hat + L^-T r` for `P = L L^T` and `r ~ N(0, I)` (so the draw has
    covariance `L^-T L^-1 = P^-1` exactly, with no explicit inversion).

    Returns `(n_draws, N, 3, n_coeffs)`, the stored coefficients perturbed --
    i.e. the ensemble is centred on the real checkpoint, so draw-to-draw
    spread is genuinely the posterior's own, and the mean render stays the
    real render."""
    generator = torch.Generator(device=device).manual_seed(seed)
    theta = torch.tensor(sh_coeffs, dtype=torch.float32, device=device)  # (N, 3, K)
    d_term = torch.tensor(data_precision, dtype=torch.float32, device=device)  # (N, K, K)
    n_splats, _, n_coeffs = theta.shape

    draws = torch.empty((n_draws, n_splats, 3, n_coeffs), dtype=torch.float32, device=device)
    for c in range(3):
        prior = torch.diag(torch.tensor(band_precision[c], dtype=torch.float32, device=device))
        chol = torch.linalg.cholesky(d_term + prior)  # (N, K, K), lower
        noise = torch.randn((n_draws, n_splats, n_coeffs, 1), generator=generator, dtype=torch.float32, device=device)
        # L^T x = r, solved per draw against the shared (N, K, K) factor
        perturbation = torch.linalg.solve_triangular(
            chol.transpose(-1, -2).unsqueeze(0).expand(n_draws, -1, -1, -1), noise, upper=True
        ).squeeze(-1)
        draws[:, :, c, :] = theta[:, c, :].unsqueeze(0) + perturbation
        del chol, noise, perturbation
        torch.cuda.empty_cache()
    return draws


def render_with_coefficients(checkpoint, sh_nkc, viewmat, Ks, width, height, background, device="cuda"):
    """One real gsplat render with a given (N, 3, K) SH coefficient tensor
    substituted for the checkpoint's own -- geometry, opacity and camera
    untouched. This is the ONLY place a draw ever enters: an ensemble member
    is a normal render, at normal render cost, not a modified renderer."""
    import gsplat

    with torch.no_grad():
        rendered, _, _ = gsplat.rasterization(
            torch.tensor(checkpoint["positions"], dtype=torch.float32, device=device),
            torch.tensor(checkpoint["rotations"], dtype=torch.float32, device=device),
            torch.tensor(checkpoint["scales"], dtype=torch.float32, device=device),
            torch.tensor(checkpoint["opacities"], dtype=torch.float32, device=device),
            sh_nkc.transpose(1, 2).contiguous(),  # (N, 3, K) -> (N, K, 3), gsplat's convention
            viewmat, Ks, width=width, height=height,
            sh_degree=checkpoint["sh_degree"], backgrounds=background,
        )
    return rendered[0].clamp(0, 1).cpu().numpy()


def run():
    ckpt_dir = LOCAL_RUNS / f"{SCENE}_prepared" / CHECKPOINT
    eval_dir = EVAL_DIRS[SCENE]
    checkpoint = read_3dgs_ply(str(ckpt_dir / "splats.ply"))
    sh_coeffs = checkpoint["sh_coeffs"]
    sh_degree = checkpoint["sh_degree"]

    # Training views (what the posterior is conditioned on) and the held-out
    # view (what it is evaluated at) are DIFFERENT transforms.json files.
    train_cax, train_frames = load_transforms(str(ckpt_dir / "transforms.json"))
    with Image.open(str(ckpt_dir / (train_frames[0][0] + ".png"))) as im:
        train_w, train_h = im.size
    train_K = fov_x_to_intrinsics(train_cax, train_w, train_h)

    data_precision, noise_var = load_or_build_data_precision(
        SCENE, CHECKPOINT, checkpoint, train_frames, train_K, train_w, train_h
    )
    band_precision = empirical_band_precision(sh_coeffs, sh_degree)
    print("\nempirical-Bayes prior precision by SH band (lambda = 1/Var over splats):")
    for l, (lo, hi) in enumerate(SH_BANDS):
        if lo >= sh_coeffs.shape[2]:
            break
        print(f"  l={l}: " + "  ".join(f"{ch}={band_precision[c, lo]:.4g}" for c, ch in enumerate("RGB")))

    # The data term only becomes comparable to the prior once divided by the
    # observation noise variance -- without it the two are in different units
    # entirely (see rasterized_sh_precision's docstring).
    data_precision = data_precision / noise_var
    print(f"\ndata term scaled by 1/sigma_n^2 = {1.0 / noise_var:.4g}")
    diag_data = np.einsum("nkk->nk", data_precision)
    print("data/prior precision ratio, per band (this is what was ~1e-13 under the KNN surrogate):")
    for l, (lo, hi) in enumerate(SH_BANDS):
        if lo >= sh_coeffs.shape[2]:
            break
        ratio = diag_data[:, lo:min(hi, sh_coeffs.shape[2])].mean(axis=1) / band_precision[0, lo]
        print(f"  l={l}: median {np.median(ratio):.3g}  90th pct {np.percentile(ratio, 90):.3g}  "
              f"frac>1 {np.mean(ratio > 1):.3f}")

    camera_angle_x, frames = load_transforms(str(eval_dir / "transforms.json"))
    file_path, c2w = frames[VIEW_IDX]
    with Image.open(str(eval_dir / (file_path + ".png"))) as im:
        width, height = im.size
    K = fov_x_to_intrinsics(camera_angle_x, width, height)
    viewmat = torch.tensor(opencv_viewmat_from_c2w(c2w), dtype=torch.float32, device="cuda")[None]
    Ks = torch.tensor(K, dtype=torch.float32, device="cuda")[None]
    background = torch.tensor(BACKGROUND_COLOR, dtype=torch.float32, device="cuda")
    gt = np.asarray(Image.open(str(eval_dir / (file_path + ".png"))).convert("RGB"), dtype=np.float32) / 255.0

    theta_hat = torch.tensor(sh_coeffs, dtype=torch.float32, device="cuda")
    # Warm up first -- the first gsplat call pays CUDA context/JIT setup and
    # is not a render time.
    mean_render = render_with_coefficients(checkpoint, theta_hat, viewmat, Ks, width, height, background)
    torch.cuda.synchronize()
    t0 = time.time()
    mean_render = render_with_coefficients(checkpoint, theta_hat, viewmat, Ks, width, height, background)
    torch.cuda.synchronize()
    single_render_s = time.time() - t0
    print(f"\nsingle real render (warmed): {single_render_s * 1000:.1f} ms")

    t0 = time.time()
    draws = sample_sh_draws(sh_coeffs, data_precision, band_precision, N_DRAWS, SEED)
    torch.cuda.synchronize()
    sample_s = time.time() - t0

    t0 = time.time()
    ensemble = np.stack([
        render_with_coefficients(checkpoint, draws[s], viewmat, Ks, width, height, background)
        for s in range(N_DRAWS)
    ], axis=0)
    torch.cuda.synchronize()
    ensemble_s = time.time() - t0
    print(f"posterior sampling ({N_DRAWS} draws, {sh_coeffs.shape[0]} splats): {sample_s * 1000:.0f} ms")
    print(f"{N_DRAWS} ensemble renders: {ensemble_s * 1000:.0f} ms ({ensemble_s / N_DRAWS * 1000:.1f} ms each)")
    print(f"=> uncertainty costs +{(sample_s + ensemble_s) * 1000:.0f} ms on a "
          f"{single_render_s * 1000:.1f} ms render (~{(sample_s + ensemble_s) / single_render_s:.1f}x one render)")

    std_map = ensemble.std(axis=0).mean(axis=2)
    abs_err = np.abs(mean_render - gt).mean(axis=2)

    # Correlation restricted to the OBJECT. Over the full frame it is
    # dominated by the object/white-background split -- both std and error
    # are ~0 on background, so a whole-frame correlation mostly reports
    # "found the silhouette", which is not a result.
    obj = gt.min(axis=2) < 0.99
    print(f"\nper-pixel posterior std: mean={std_map.mean():.5g} max={std_map.max():.5g} "
          f"99th pct={np.percentile(std_map, 99):.5g}")
    print(f"per-pixel |held-out error|: mean={abs_err.mean():.5g} max={abs_err.max():.5g}")
    for name, mask in (("whole frame", np.ones_like(obj)), ("object pixels only", obj)):
        a, b = std_map[mask], abs_err[mask]
        pear = np.corrcoef(a, b)[0, 1]
        spear = np.corrcoef(a.argsort().argsort(), b.argsort().argsort())[0, 1]
        print(f"  {name} ({mask.sum()} px): pearson={pear:.4f}  spearman={spear:.4f}")

    np.savez_compressed(
        CACHE_DIR / f"{SCENE}_{CHECKPOINT}_ensemble_view{VIEW_IDX}.npz",
        ensemble=ensemble, mean_render=mean_render, gt=gt, std_map=std_map, abs_err=abs_err,
    )

    n_show = min(4, N_DRAWS)
    fig, axes = plt.subplots(1, n_show + 2, figsize=(3.0 * (n_show + 2), 3.4))
    for s in range(n_show):
        axes[s].imshow(ensemble[s])
        axes[s].set_title(f"posterior draw {s + 1}", fontsize=11)
    im = axes[n_show].imshow(std_map, cmap="inferno")
    axes[n_show].set_title("per-pixel std across draws", fontsize=11)
    fig.colorbar(im, ax=axes[n_show], fraction=0.046)
    im2 = axes[n_show + 1].imshow(abs_err, cmap="inferno")
    axes[n_show + 1].set_title("|held-out error| (reference)", fontsize=11)
    fig.colorbar(im2, ax=axes[n_show + 1], fraction=0.046)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(
        f"{SCENE} / {CHECKPOINT} / held-out view {VIEW_IDX} -- uncertainty = {N_DRAWS} extra renders "
        f"(+{(sample_s + ensemble_s) * 1000:.0f} ms on a {single_render_s * 1000:.1f} ms render)",
        fontsize=12,
    )
    fig.tight_layout()
    out_path = RESULTS_DIR / "posterior_ensemble.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"\nwrote {out_path}")
    return ensemble, std_map, abs_err


if __name__ == "__main__":
    run()
