"""Does putting geometry into the posterior fix the calibration?

The posterior-ensemble uncertainty correlates only weakly with real held-out
error on object pixels (Spearman ~0.24; see results/FINDINGS.md section 5).
Cross-splat coupling was implemented, validated and ruled out as the cause
(section 6). The remaining suspect is that the posterior samples appearance
only, with geometry frozen -- so this compares, on the same frozen
checkpoint and the same held-out views:

    1. SH only          -- the current method
    2. opacity only     -- geometry alone (rasterized_opacity_precision)
    3. SH + opacity     -- both

against real held-out error, by Spearman correlation AND by AUSE (the
sparsification metric, which is what a calibration claim should actually be
scored on). Object pixels only throughout: whole-frame numbers on
NeRF-Synthetic are dominated by the object/background split and report the
silhouette, not calibration.

Run: .venv-gsplat/bin/python gs_experiment/scripts/render_geometry_ensemble.py
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

from gs_experiment.nerf_transforms import fov_x_to_intrinsics, load_transforms, opencv_viewmat_from_c2w
from gs_experiment.ply_io import read_3dgs_ply
from gs_experiment.rasterized_opacity_precision import (
    accumulate_opacity_information, empirical_opacity_precision, sample_opacity_draws,
)
from gs_experiment.rasterized_sh_precision import accumulate_sh_precision_rasterized, estimate_noise_variance
from gs_experiment.scripts.render_posterior_ensemble import (
    BACKGROUND_COLOR, N_PROBES, SEED, empirical_band_precision, sample_sh_draws,
)
from gs_experiment.scripts.render_reconstruction import EVAL_DIRS, LOCAL_RUNS, RESULTS_DIR

SCENE, CHECKPOINT = "lego", "wide"
VIEW_IDXS = [21, 3, 12]   # three real held-out views, not one cherry-picked
N_DRAWS = 16
N_OPACITY_PROBES = 16


def spearman(a, b):
    return float(np.corrcoef(a.argsort().argsort(), b.argsort().argsort())[0, 1])


def ause(unc, err, n_steps=100):
    """Area under the sparsification error curve. Remove the most-uncertain
    pixels progressively and track the mean error of what remains, against
    the oracle ordering (by true error). Lower is better; 0 means the
    uncertainty ranks pixels exactly as well as the true error does."""
    n = len(err)
    fracs = np.linspace(0, 0.95, n_steps)
    by_unc = np.argsort(-unc)
    by_err = np.argsort(-err)
    curve_u = np.array([err[by_unc[int(f * n):]].mean() for f in fracs])
    curve_o = np.array([err[by_err[int(f * n):]].mean() for f in fracs])
    denom = curve_o[0] if curve_o[0] > 0 else 1.0
    return float(np.trapezoid((curve_u - curve_o) / denom, fracs)), curve_u / denom, curve_o / denom


def render(checkpoint, sh_nkc, opacities, viewmat, Ks, width, height, background, device="cuda"):
    import gsplat
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=device) if not torch.is_tensor(a) else a  # noqa: E731
    with torch.no_grad():
        out, _, _ = gsplat.rasterization(
            t(checkpoint["positions"]), t(checkpoint["rotations"]), t(checkpoint["scales"]),
            opacities, sh_nkc.transpose(1, 2).contiguous(), viewmat, Ks,
            width=width, height=height, sh_degree=checkpoint["sh_degree"], backgrounds=background,
        )
    return out[0].clamp(0, 1).cpu().numpy()


def run():
    ckpt_dir = LOCAL_RUNS / f"{SCENE}_prepared" / CHECKPOINT
    eval_dir = EVAL_DIRS[SCENE]
    checkpoint = read_3dgs_ply(str(ckpt_dir / "splats.ply"))
    sh_coeffs, degree = checkpoint["sh_coeffs"], checkpoint["sh_degree"]

    train_cax, train_frames = load_transforms(str(ckpt_dir / "transforms.json"))
    with Image.open(str(ckpt_dir / (train_frames[0][0] + ".png"))) as im:
        tw, th = im.size
    train_K = fov_x_to_intrinsics(train_cax, tw, th)

    noise_var = estimate_noise_variance(checkpoint, train_frames, train_K, tw, th,
                                        str(ckpt_dir), background_color=BACKGROUND_COLOR)
    band_precision = empirical_band_precision(sh_coeffs, degree)
    lam_u = empirical_opacity_precision(checkpoint["opacities"])
    print(f"sigma_n^2 = {noise_var:.6g}, lambda_u = {lam_u:.5g} "
          f"(prior std on logit-opacity = {1 / np.sqrt(lam_u):.3f})")

    t0 = time.time()
    sh_data = accumulate_sh_precision_rasterized(
        checkpoint, train_frames, train_K, tw, th, degree,
        n_probes=N_PROBES, seed=SEED, device="cuda", progress_every=0) / noise_var
    print(f"SH information: {time.time() - t0:.1f}s")

    t0 = time.time()
    op_info = accumulate_opacity_information(
        checkpoint, train_frames, train_K, tw, th, n_probes=N_OPACITY_PROBES,
        seed=SEED, device="cuda", background_color=BACKGROUND_COLOR)
    print(f"opacity information: {time.time() - t0:.1f}s")
    op_prec = lam_u + op_info / noise_var
    print(f"  opacity data/prior precision: median {np.median(op_info / noise_var / lam_u):.4g}  "
          f"frac>1 {np.mean(op_info / noise_var > lam_u):.3f}")
    print(f"  posterior std on logit-opacity: median {np.median(1 / np.sqrt(op_prec)):.4g}  "
          f"90th pct {np.percentile(1 / np.sqrt(op_prec), 90):.4g}")

    sh_draws = sample_sh_draws(sh_coeffs, sh_data, band_precision, N_DRAWS, SEED)
    op_draws = sample_opacity_draws(checkpoint["opacities"], op_info, lam_u, noise_var, N_DRAWS, SEED)
    theta_hat = torch.tensor(sh_coeffs, dtype=torch.float32, device="cuda")
    op_hat = torch.tensor(checkpoint["opacities"], dtype=torch.float32, device="cuda")

    conditions = {
        "SH only": (sh_draws, [op_hat] * N_DRAWS),
        "opacity only": ([theta_hat] * N_DRAWS, op_draws),
        "SH + opacity": (sh_draws, op_draws),
    }

    cax, frames = load_transforms(str(eval_dir / "transforms.json"))
    rows = {name: {"sp": [], "ause": [], "std": []} for name in conditions}
    per_view = []
    for view_idx in VIEW_IDXS:
        file_path, c2w = frames[view_idx]
        with Image.open(str(eval_dir / (file_path + ".png"))) as im:
            width, height = im.size
        K = fov_x_to_intrinsics(cax, width, height)
        viewmat = torch.tensor(opencv_viewmat_from_c2w(c2w), dtype=torch.float32, device="cuda")[None]
        Ks = torch.tensor(K, dtype=torch.float32, device="cuda")[None]
        background = torch.tensor(BACKGROUND_COLOR, dtype=torch.float32, device="cuda")
        gt = np.asarray(Image.open(str(eval_dir / (file_path + ".png"))).convert("RGB"),
                        dtype=np.float32) / 255.0
        obj = gt.min(axis=2) < 0.99
        mean_render = render(checkpoint, theta_hat, op_hat, viewmat, Ks, width, height, background)
        err = np.abs(mean_render - gt).mean(axis=2)

        view_maps = {}
        for name, (shs, ops) in conditions.items():
            ens = np.stack([render(checkpoint, shs[s], ops[s], viewmat, Ks, width, height, background)
                            for s in range(N_DRAWS)], axis=0)
            std = ens.std(axis=0).mean(axis=2)
            a, _, _ = ause(std[obj], err[obj])
            rows[name]["sp"].append(spearman(std[obj], err[obj]))
            rows[name]["ause"].append(a)
            rows[name]["std"].append(std[obj].mean())
            view_maps[name] = (ens, std)
        per_view.append((view_idx, gt, mean_render, err, obj, view_maps))
        print(f"view {view_idx}: " + "  ".join(
            f"{n} sp={rows[n]['sp'][-1]:.3f} ause={rows[n]['ause'][-1]:.4f}" for n in conditions))

    print("\n=== mean over 3 held-out views, object pixels only ===")
    print(f"{'condition':<16}{'spearman':>10}{'AUSE':>10}{'mean std':>12}")
    for name in conditions:
        print(f"{name:<16}{np.mean(rows[name]['sp']):>10.3f}{np.mean(rows[name]['ause']):>10.4f}"
              f"{np.mean(rows[name]['std']):>12.5g}")

    view_idx, gt, mean_render, err, obj, view_maps = per_view[0]
    names = list(conditions)
    fig, axes = plt.subplots(2, len(names) + 1, figsize=(4.0 * (len(names) + 1), 7.2))
    vmax = max(view_maps[n][1].max() for n in names)
    for j, name in enumerate(names):
        ens, std = view_maps[name]
        axes[0, j].imshow(ens[0]); axes[0, j].set_title(f"{name}: draw 1", fontsize=12)
        im = axes[1, j].imshow(std, cmap="inferno", vmin=0, vmax=vmax)
        axes[1, j].set_title(f"std  (sp={np.mean(rows[name]['sp']):.2f})", fontsize=12)
        fig.colorbar(im, ax=axes[1, j], fraction=0.046)
    axes[0, -1].imshow(mean_render); axes[0, -1].set_title("mean render", fontsize=12)
    im = axes[1, -1].imshow(err, cmap="inferno"); axes[1, -1].set_title("|held-out error|", fontsize=12)
    fig.colorbar(im, ax=axes[1, -1], fraction=0.046)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"{SCENE} view {view_idx}: does putting geometry in the posterior improve calibration?",
                 fontsize=14)
    fig.tight_layout()
    path = RESULTS_DIR / "geometry_ensemble.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    run()
