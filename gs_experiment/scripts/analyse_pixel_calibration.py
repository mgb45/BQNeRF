"""Is the per-pixel uncertainty actually miscalibrated, or was it scored
against an unreachable target?

FINDINGS sections 5-8 reported per-pixel Spearman of 0.25-0.31 between
predicted std and |held-out error| and read it as "modest". That reading
compares against an implicit ceiling of 1.0, which is wrong. Even with a
PERFECTLY calibrated sigma, the observable is one realization
`eps_q ~ N(0, sigma_q^2)`, i.e. `|eps_q| = sigma_q |z_q|` with `z_q ~ N(0,1)`
independent of everything. The nuisance factor `|z_q|` destroys rank
information on its own: `Var(log|z|) = pi^2/8 ~ 1.23`, so unless `log sigma`
varies by more than that across pixels, a high rank correlation is
unattainable no matter how good the model is. The same objection applies to
AUSE, whose usual oracle (sort by TRUE error) is likewise unreachable.

So this script scores the uncertainty against its own achievable ceiling,
obtained by simulating `eps* ~ N(0, sigma_pred^2)` and re-running the same
metric -- and separately checks the thing rank correlation cannot see, which
is whether the magnitudes are right:

  * **binned calibration**: bin pixels by predicted sigma, compare each bin's
    predicted sigma against the RMS error actually observed in it. This is
    the real per-pixel calibration question, and it averages away `|z|`.
  * **a fitted scale** `s` with an aleatoric floor `sigma_0`, so that
    `sigma_total^2 = s^2 sigma_pred^2 + sigma_0^2`, scored by Gaussian NLL
    before and after. Ranking is untouched by this; only calibration is.

Run: .venv-gsplat/bin/python gs_experiment/scripts/analyse_pixel_calibration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.optimize import minimize

from gs_experiment.nerf_transforms import fov_x_to_intrinsics, load_transforms, opencv_viewmat_from_c2w
from gs_experiment.ply_io import read_3dgs_ply
from gs_experiment.rasterized_sh_precision import (
    accumulate_sh_precision_rasterized,
    estimate_noise_variance,
    pixel_weight_concentration,
)
from gs_experiment.scripts.render_geometry_ensemble import ause, render, spearman
from gs_experiment.scripts.render_posterior_ensemble import (
    BACKGROUND_COLOR, N_PROBES, SEED, empirical_band_precision, sample_sh_draws,
)
from gs_experiment.scripts.render_reconstruction import LOCAL_RUNS, RESULTS_DIR

PREPARED = LOCAL_RUNS / "lego_prepared"
CONDITIONS = [("wide", "wide_eval", None), ("gap_4", "gap_4_eval", None)]
N_DRAWS = 32
N_CEILING_SIMS = 16
RNG = np.random.default_rng(0)


def collect(ckpt_name, eval_name, view_idxs):
    """Per-channel predicted sigma and signed residual on object pixels,
    pooled over the chosen held-out views of one checkpoint."""
    ckpt_dir, eval_dir = PREPARED / ckpt_name, PREPARED / eval_name
    checkpoint = read_3dgs_ply(str(ckpt_dir / "splats.ply"))
    sh_coeffs, degree = checkpoint["sh_coeffs"], checkpoint["sh_degree"]
    cax, train_frames = load_transforms(str(ckpt_dir / "transforms.json"))
    with Image.open(str(ckpt_dir / (train_frames[0][0] + ".png"))) as im:
        tw, th = im.size
    train_K = fov_x_to_intrinsics(cax, tw, th)

    noise_var = estimate_noise_variance(checkpoint, train_frames, train_K, tw, th,
                                        str(ckpt_dir), background_color=BACKGROUND_COLOR)
    band_precision = empirical_band_precision(sh_coeffs, degree)
    data = accumulate_sh_precision_rasterized(
        checkpoint, train_frames, train_K, tw, th, degree,
        n_probes=N_PROBES, seed=SEED, device="cuda", progress_every=0) / noise_var
    draws = sample_sh_draws(sh_coeffs, data, band_precision, N_DRAWS, SEED)
    theta_hat = torch.tensor(sh_coeffs, dtype=torch.float32, device="cuda")
    op_hat = torch.tensor(checkpoint["opacities"], dtype=torch.float32, device="cuda")

    ecax, eval_frames = load_transforms(str(eval_dir / "transforms.json"))
    with Image.open(str(eval_dir / (eval_frames[0][0] + ".png"))) as im:
        width, height = im.size
    K = fov_x_to_intrinsics(ecax, width, height)
    background = torch.tensor(BACKGROUND_COLOR, dtype=torch.float32, device="cuda")
    idxs = range(len(eval_frames)) if view_idxs is None else view_idxs

    groups = []
    for vi in idxs:
        file_path, c2w = eval_frames[vi]
        viewmat = torch.tensor(opencv_viewmat_from_c2w(c2w), dtype=torch.float32, device="cuda")[None]
        Ks = torch.tensor(K, dtype=torch.float32, device="cuda")[None]
        gt = np.asarray(Image.open(str(eval_dir / (file_path + ".png"))).convert("RGB"),
                        dtype=np.float32) / 255.0
        obj = gt.min(axis=2) < 0.99
        mean_render = render(checkpoint, theta_hat, op_hat, viewmat, Ks, width, height, background)
        ens = np.stack([render(checkpoint, draws[s], op_hat, viewmat, Ks, width, height, background)
                        for s in range(N_DRAWS)], axis=0)
        # Render-derived aleatoric regressors, both per-pixel and broadcast
        # over the 3 channels so they line up with the per-channel residuals.
        t = lambda a: torch.tensor(a, dtype=torch.float32, device="cuda")  # noqa: E731
        gen = torch.Generator(device="cuda").manual_seed(SEED + vi)
        conc = pixel_weight_concentration(
            t(checkpoint["positions"]), t(checkpoint["rotations"]), t(checkpoint["scales"]),
            op_hat, viewmat, Ks, width, height, n_probes=32, generator=gen).cpu().numpy()
        gy, gx = np.gradient(mean_render.mean(axis=2))
        grad = np.sqrt(gx ** 2 + gy ** 2)
        feats = np.stack([np.repeat(conc[:, :, None], 3, axis=2)[obj].ravel(),
                          np.repeat(grad[:, :, None], 3, axis=2)[obj].ravel()], axis=1)
        groups.append((ens.std(axis=0)[obj].ravel(), (mean_render - gt)[obj].ravel(), feats))
    return groups   # (sigma, residual, features) per held-out view, per channel


def ceiling(sigma, metric, n_sims=N_CEILING_SIMS):
    """The metric's own achievable value if `sigma` were exactly right:
    simulate eps* ~ N(0, sigma^2) and re-score."""
    vals = [metric(sigma, np.abs(RNG.normal(0.0, np.maximum(sigma, 1e-12)))) for _ in range(n_sims)]
    return float(np.mean(vals)), float(np.std(vals))


def binned_calibration(sigma, resid, n_bins=15):
    """Bin by predicted sigma; compare predicted sigma against the RMS
    residual actually observed in each bin. Averaging over a bin removes the
    |z| nuisance that caps rank correlation."""
    order = np.argsort(sigma)
    bins = np.array_split(order, n_bins)
    pred = np.array([sigma[b].mean() for b in bins])
    obs = np.array([np.sqrt((resid[b] ** 2).mean()) for b in bins])
    return pred, obs


def _nll(var, resid):
    return float(np.mean(0.5 * np.log(2 * np.pi * var) + resid ** 2 / (2 * var)))


def fit_variance_model(sigma, resid, feats=None):
    """Fit `sigma_total^2 = s^2 sigma^2 + a (+ sum_k b_k f_k)` by Gaussian
    NLL, all coefficients kept positive by exponentiation so the variance is
    always valid. With `feats=None` this is the plain two-parameter scale +
    constant aleatoric floor; with features it lets the aleatoric part vary
    spatially, which is what a misspecification term should do.

    Returns a callable mapping (sigma, feats) -> variance, plus the fitted
    parameters."""
    n_f = 0 if feats is None else feats.shape[1]

    def variance(p, sig, f):
        v = np.exp(p[0]) * sig ** 2 + np.exp(p[1])
        for k in range(n_f):
            v = v + np.exp(p[2 + k]) * f[:, k]
        return v

    x0 = [0.0, np.log(np.var(resid) + 1e-12)] + [-8.0] * n_f
    r = minimize(lambda p: _nll(variance(p, sigma, feats), resid), x0=x0,
                 method="Nelder-Mead", options={"maxiter": 4000, "fatol": 1e-10})
    return (lambda sig, f: variance(r.x, sig, f)), r.x


def run():
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(6.0 * len(CONDITIONS), 5.0), squeeze=False)
    for j, (ckpt, ev, views) in enumerate(CONDITIONS):
        groups = collect(ckpt, ev, views)
        # Fit the calibration on ALTERNATE views and score it on the rest --
        # a scale fitted and evaluated on the same pixels proves nothing.
        fit_idx = list(range(0, len(groups), 2))
        test_idx = [i for i in range(len(groups)) if i not in fit_idx]
        if not test_idx:                      # only one view available
            fit_idx, test_idx = [0], [0]
        sigma_fit = np.concatenate([groups[i][0] for i in fit_idx])
        resid_fit = np.concatenate([groups[i][1] for i in fit_idx])
        feats_fit = np.concatenate([groups[i][2] for i in fit_idx])
        sigma = np.concatenate([groups[i][0] for i in test_idx])
        resid = np.concatenate([groups[i][1] for i in test_idx])
        feats = np.concatenate([groups[i][2] for i in test_idx])
        err = np.abs(resid)
        print(f"  calibration fit on {len(fit_idx)} view(s), scored on {len(test_idx)} held-out view(s)")
        print(f"\n=== {ckpt} ({len(sigma)} object pixel-channels) ===")
        print(f"  spread of log sigma: std {np.std(np.log(np.maximum(sigma, 1e-12))):.3f}  "
              f"(|z| noise contributes std {np.sqrt(np.pi**2/8):.3f} -- this is what caps rank metrics)")

        obs_sp = spearman(sigma, err)
        ceil_sp, sd_sp = ceiling(sigma, lambda s, e: spearman(s, e))
        obs_au, _, _ = ause(sigma, err)
        ceil_au, sd_au = ceiling(sigma, lambda s, e: ause(s, e)[0])
        print(f"  spearman: observed {obs_sp:.3f}   achievable ceiling {ceil_sp:.3f} +- {sd_sp:.3f}"
              f"   -> {100 * obs_sp / ceil_sp:.0f}% of attainable")
        print(f"  AUSE:     observed {obs_au:.4f}  achievable floor  {ceil_au:.4f} +- {sd_au:.4f}")

        # Every model fitted on the held-in views ONLY, scored on the rest.
        f_const, p_const = fit_variance_model(sigma_fit, resid_fit, None)
        f_feat, p_feat = fit_variance_model(sigma_fit, resid_fit, feats_fit)
        f_noep, p_noep = fit_variance_model(np.zeros_like(sigma_fit), resid_fit, feats_fit)
        s, s0 = float(np.sqrt(np.exp(p_const[0]))), float(np.sqrt(np.exp(p_const[1])))
        nll_raw = _nll(np.maximum(sigma, 1e-12) ** 2, resid)
        nll_base = _nll(np.full_like(resid, np.var(resid_fit)), resid)
        nll_const = _nll(f_const(sigma, None), resid)
        nll_feat = _nll(f_feat(sigma, feats), resid)
        nll_noep = _nll(f_noep(np.zeros_like(sigma), feats), resid)
        print(f"  fitted on held-in views: scale s = {s:.3f}, constant floor sigma_0 = {s0:.5f}")
        print("  Gaussian NLL on HELD-OUT views (lower is better):")
        print(f"    raw, uncalibrated                        {nll_raw:>12.4g}")
        print(f"    constant variance (no uncertainty at all){nll_base:>12.4f}")
        print(f"    render features only, no posterior       {nll_noep:>12.4f}")
        print(f"    posterior + constant floor               {nll_const:>12.4f}"
              f"   ({nll_base - nll_const:+.4f} nats vs constant)")
        print(f"    posterior + render-derived floor         {nll_feat:>12.4f}"
              f"   ({nll_base - nll_feat:+.4f} nats vs constant)")

        pred, obs = binned_calibration(sigma, resid)
        ax = axes[0, j]
        ax.loglog(pred, obs, "o-", label="observed RMS error")
        lim = [min(pred.min(), obs.min()) * 0.7, max(pred.max(), obs.max()) * 1.4]
        ax.loglog(lim, lim, "k--", lw=1, label="perfect calibration")
        ax.loglog(pred, np.sqrt(s**2 * pred**2 + s0**2), "r:", lw=2,
                  label=f"fitted (s={s:.2f}, $\\sigma_0$={s0:.3f})")
        ax.set_xlabel("predicted $\\sigma$ (bin mean)")
        ax.set_ylabel("observed RMS residual")
        ax.set_title(f"{ckpt}: spearman {obs_sp:.2f} of {ceil_sp:.2f} attainable", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, which="both")
        print("  binned calibration (predicted -> observed):")
        for p_, o_ in zip(pred, obs):
            print(f"    {p_:.5f} -> {o_:.5f}   ratio {o_ / max(p_, 1e-12):6.2f}")

    fig.suptitle("lego: per-pixel calibration, scored against the attainable ceiling", fontsize=13)
    fig.tight_layout()
    path = RESULTS_DIR / "pixel_calibration.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    run()
