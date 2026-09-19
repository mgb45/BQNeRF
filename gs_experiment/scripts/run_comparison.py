"""Comparative evaluation against the post-hoc uncertainty cluster.

Every method here is scored by `gs_experiment/evaluation.py` -- the protocol
pre-registered and frozen before any of these baselines was written -- on
IDENTICAL checkpoints and identical held-out views. Nothing below chooses its
own metric.

Run: .venv-gsplat/bin/python gs_experiment/scripts/run_comparison.py [scene] [checkpoint]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from PIL import Image

from gs_experiment.baselines import fit_residual_supervised_sh, render_sh_field, uniform_coverage_sh
from gs_experiment.evaluation import Prediction, compare, format_table, object_mask
from gs_experiment.nerf_transforms import fov_x_to_intrinsics, load_transforms, opencv_viewmat_from_c2w
from gs_experiment.ply_io import read_3dgs_ply
from gs_experiment.rasterized_parameter_fisher import (
    accumulate_parameter_fisher,
    empirical_parameter_precision,
    render_draw,
    sample_parameter_draws,
)
from gs_experiment.rasterized_sh_precision import (
    accumulate_sh_precision_rasterized, estimate_noise_variance, pixel_weight_concentration,
)
from gs_experiment.scripts.render_geometry_ensemble import render
from gs_experiment.scripts.render_posterior_ensemble import (
    BACKGROUND_COLOR, N_PROBES, SEED, empirical_band_precision, sample_sh_draws,
)
from gs_experiment.scripts.render_reconstruction import LOCAL_RUNS, RESULTS_DIR

N_DRAWS = 16


def _ensemble_std(checkpoint, draws, op_hat, viewmat, Ks, w, h, bg):
    ens = np.stack([render(checkpoint, draws[s], op_hat, viewmat, Ks, w, h, bg)
                    for s in range(draws.shape[0])], axis=0)
    return ens.std(axis=0)                      # (H, W, 3)


def run(scene="lego", ckpt_name="wide", eval_name="eval"):
    ckpt_dir = LOCAL_RUNS / f"{scene}_prepared" / ckpt_name
    eval_dir = LOCAL_RUNS / f"{scene}_prepared" / eval_name
    checkpoint = read_3dgs_ply(str(ckpt_dir / "splats.ply"))
    sh_coeffs, degree = checkpoint["sh_coeffs"], checkpoint["sh_degree"]
    cax, train_frames = load_transforms(str(ckpt_dir / "transforms.json"))
    with Image.open(str(ckpt_dir / (train_frames[0][0] + ".png"))) as im:
        tw, th = im.size
    train_K = fov_x_to_intrinsics(cax, tw, th)
    print(f"{scene}/{ckpt_name}: {sh_coeffs.shape[0]} splats, {len(train_frames)} training views")

    sigma_n = float(np.sqrt(estimate_noise_variance(
        checkpoint, train_frames, train_K, tw, th, str(ckpt_dir),
        background_color=BACKGROUND_COLOR)))
    band_precision = empirical_band_precision(sh_coeffs, degree)
    t = lambda a: torch.tensor(a, dtype=torch.float32, device="cuda")  # noqa: E731
    theta_hat, op_hat = t(sh_coeffs), t(checkpoint["opacities"])

    # ---- prepare each method (cost recorded) -------------------------------
    prep = {}
    t0 = time.time()
    data = accumulate_sh_precision_rasterized(
        checkpoint, train_frames, train_K, tw, th, degree, n_probes=N_PROBES,
        seed=SEED, device="cuda", progress_every=0) / (sigma_n ** 2)
    ours_draws = sample_sh_draws(sh_coeffs, data, band_precision, N_DRAWS, SEED)
    prep["ours (SH posterior)"] = time.time() - t0

    t0 = time.time()
    psi = fit_residual_supervised_sh(checkpoint, train_frames, train_K, tw, th, degree,
                                     str(ckpt_dir), lam=1.0, n_cg_iters=150,
                                     background_color=BACKGROUND_COLOR, verbose=False)
    prep["residual-supervised SH"] = time.time() - t0

    t0 = time.time()
    param_fisher = accumulate_parameter_fisher(
        checkpoint, train_frames, train_K, tw, th, n_probes=16, seed=SEED,
        device="cuda", background_color=BACKGROUND_COLOR)
    param_draws = sample_parameter_draws(
        checkpoint, param_fisher, empirical_parameter_precision(checkpoint),
        sigma_n ** 2, N_DRAWS, seed=SEED)
    prep["all-parameter Fisher"] = time.time() - t0

    t0 = time.time()
    cov_prec = uniform_coverage_sh(checkpoint, train_frames, train_K, tw, th, degree,
                                   band_precision) + np.diag(band_precision[0])
    cov_draws = sample_sh_draws(sh_coeffs, cov_prec - np.diag(band_precision[0]),
                                band_precision, N_DRAWS, SEED)
    prep["uniform-coverage SH"] = time.time() - t0

    # ---- score on held-out views ------------------------------------------
    ecax, eval_frames = load_transforms(str(eval_dir / "transforms.json"))
    with Image.open(str(eval_dir / (eval_frames[0][0] + ".png"))) as im:
        w, h = im.size
    K = fov_x_to_intrinsics(ecax, w, h)
    bg = t(BACKGROUND_COLOR)
    acc = {k: {"sigma": [], "resid": [], "view": [], "secs": prep.get(k, 0.0)}
           for k in list(prep) + ["weight concentration", "render gradient"]}
    for k in ("weight concentration", "render gradient"):
        acc[k]["secs"] = 0.0

    for vi, (file_path, c2w) in enumerate(eval_frames):
        viewmat, Ks = t(opencv_viewmat_from_c2w(c2w))[None], t(K)[None]
        gt = np.asarray(Image.open(str(eval_dir / (file_path + ".png"))).convert("RGB"),
                        dtype=np.float32) / 255.0
        obj = object_mask(gt)
        mean_render = render(checkpoint, theta_hat, op_hat, viewmat, Ks, w, h, bg)
        resid = (mean_render - gt)[obj].ravel()

        maps, timings = {}, {}
        t0 = time.time()
        maps["ours (SH posterior)"] = _ensemble_std(checkpoint, ours_draws, op_hat, viewmat, Ks, w, h, bg)
        timings["ours (SH posterior)"] = time.time() - t0

        t0 = time.time()
        maps["residual-supervised SH"] = np.abs(
            render_sh_field(checkpoint, psi, c2w, K, w, h, degree))
        timings["residual-supervised SH"] = time.time() - t0

        t0 = time.time()
        maps["uniform-coverage SH"] = _ensemble_std(checkpoint, cov_draws, op_hat, viewmat, Ks, w, h, bg)
        timings["uniform-coverage SH"] = time.time() - t0

        t0 = time.time()
        pens = np.stack([render_draw(d, viewmat, Ks, w, h, degree, bg) for d in param_draws], axis=0)
        maps["all-parameter Fisher"] = pens.std(axis=0)
        timings["all-parameter Fisher"] = time.time() - t0

        t0 = time.time()
        gen = torch.Generator(device="cuda").manual_seed(SEED + vi)
        conc = pixel_weight_concentration(
            t(checkpoint["positions"]), t(checkpoint["rotations"]), t(checkpoint["scales"]),
            op_hat, viewmat, Ks, w, h, n_probes=32, generator=gen).cpu().numpy()
        maps["weight concentration"] = np.repeat(np.sqrt(conc)[:, :, None], 3, axis=2)
        timings["weight concentration"] = time.time() - t0

        t0 = time.time()
        gy, gx = np.gradient(mean_render.mean(axis=2))
        maps["render gradient"] = np.repeat(np.sqrt(gx ** 2 + gy ** 2)[:, :, None], 3, axis=2)
        timings["render gradient"] = time.time() - t0

        for name, m in maps.items():
            acc[name]["sigma"].append(np.maximum(m[obj].ravel(), 1e-12))
            acc[name]["resid"].append(resid)
            acc[name]["view"].append(np.full(resid.shape, vi))
            acc[name]["secs"] += timings[name]

    preds = [Prediction(sigma=np.concatenate(a["sigma"]), residual=np.concatenate(a["resid"]),
                        view_id=np.concatenate(a["view"]), sigma_n=sigma_n,
                        seconds=a["secs"], requires_retraining=False, name=name)
             for name, a in acc.items()]
    rows = compare(preds)
    print(f"\n=== {scene}/{ckpt_name}: {len(eval_frames)} held-out views, object pixels, "
          f"frozen protocol ===")
    print(format_table(rows))
    out = RESULTS_DIR / f"comparison_{scene}_{ckpt_name}.json"
    json.dump(rows, open(out, "w"), indent=1)
    print(f"\nwrote {out}")
    return rows


if __name__ == "__main__":
    run(*(sys.argv[1:] or []))
