"""Block-diagonal vs. cross-splat-coupled posterior, same frozen checkpoint,
same conditioning set, same held-out view.

The block-diagonal posterior treats every splat's coefficients as
independent. Photometric training only constrains the SUM of contributions
along a ray, so that is structurally wrong, and `(A^-1)_ii >= (A_ii)^-1`
says it can only ever UNDERSTATE a splat's marginal variance. This script
measures how much, and whether the difference shows up where it should.

Run: .venv-gsplat/bin/python gs_experiment/scripts/render_coupled_comparison.py
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

from gs_experiment.coupled_sh_posterior import sample_coupled_perturbations
from gs_experiment.nerf_transforms import fov_x_to_intrinsics, load_transforms, opencv_viewmat_from_c2w
from gs_experiment.ply_io import read_3dgs_ply
from gs_experiment.rasterized_sh_precision import accumulate_sh_precision_rasterized, estimate_noise_variance
from gs_experiment.scripts.render_posterior_ensemble import (
    BACKGROUND_COLOR, N_PROBES, SEED, empirical_band_precision, render_with_coefficients, sample_sh_draws,
)
from gs_experiment.scripts.render_posterior_view_sweep import evenly_spaced
from gs_experiment.scripts.render_reconstruction import EVAL_DIRS, LOCAL_RUNS, RESULTS_DIR

SCENE, CHECKPOINT, VIEW_IDX = "lego", "wide", 21
N_VIEWS = 25
N_DRAWS = 8
N_CG = 400


def spearman(a, b):
    return float(np.corrcoef(a.argsort().argsort(), b.argsort().argsort())[0, 1])


def run():
    ckpt_dir = LOCAL_RUNS / f"{SCENE}_prepared" / CHECKPOINT
    eval_dir = EVAL_DIRS[SCENE]
    checkpoint = read_3dgs_ply(str(ckpt_dir / "splats.ply"))
    sh_coeffs, degree = checkpoint["sh_coeffs"], checkpoint["sh_degree"]

    train_cax, train_frames = load_transforms(str(ckpt_dir / "transforms.json"))
    with Image.open(str(ckpt_dir / (train_frames[0][0] + ".png"))) as im:
        tw, th = im.size
    train_K = fov_x_to_intrinsics(train_cax, tw, th)
    picks = evenly_spaced(len(train_frames), N_VIEWS)
    subset = [train_frames[i] for i in picks]
    print(f"{sh_coeffs.shape[0]} splats, conditioning on {len(subset)}/{len(train_frames)} training views")

    noise_var = estimate_noise_variance(checkpoint, train_frames, train_K, tw, th,
                                        str(ckpt_dir), background_color=BACKGROUND_COLOR)
    band_precision = empirical_band_precision(sh_coeffs, degree)
    data = accumulate_sh_precision_rasterized(
        checkpoint, train_frames, train_K, tw, th, degree,
        n_probes=N_PROBES, seed=SEED, device="cuda", progress_every=0, camera_indices=picks,
    ) / noise_var
    # Preconditioner uses the R-channel prior; a preconditioner need not be exact.
    block_precision = data + np.diag(band_precision[0])

    cax, frames = load_transforms(str(eval_dir / "transforms.json"))
    file_path, c2w = frames[VIEW_IDX]
    with Image.open(str(eval_dir / (file_path + ".png"))) as im:
        width, height = im.size
    K = fov_x_to_intrinsics(cax, width, height)
    viewmat = torch.tensor(opencv_viewmat_from_c2w(c2w), dtype=torch.float32, device="cuda")[None]
    Ks = torch.tensor(K, dtype=torch.float32, device="cuda")[None]
    background = torch.tensor(BACKGROUND_COLOR, dtype=torch.float32, device="cuda")
    gt = np.asarray(Image.open(str(eval_dir / (file_path + ".png"))).convert("RGB"), dtype=np.float32) / 255.0
    obj = gt.min(axis=2) < 0.99

    theta_hat = torch.tensor(sh_coeffs, dtype=torch.float32, device="cuda")
    mean_render = render_with_coefficients(checkpoint, theta_hat, viewmat, Ks, width, height, background)
    abs_err = np.abs(mean_render - gt).mean(axis=2)

    results = {}
    t0 = time.time()
    block_draws = sample_sh_draws(sh_coeffs, data, band_precision, N_DRAWS, SEED)
    results["block-diagonal"] = (block_draws, time.time() - t0)

    t0 = time.time()
    pert = sample_coupled_perturbations(
        checkpoint, subset, train_K, tw, th, degree, band_precision, noise_var,
        block_precision, n_draws=N_DRAWS, seed=SEED, n_cg_iters=N_CG, device="cuda",
    )
    results["coupled"] = (theta_hat[None] + pert, time.time() - t0)

    out = {}
    for name, (draws, secs) in results.items():
        ens = np.stack([render_with_coefficients(checkpoint, draws[s], viewmat, Ks, width, height, background)
                        for s in range(N_DRAWS)], axis=0)
        std = ens.std(axis=0).mean(axis=2)
        out[name] = (ens, std)
        print(f"\n{name} ({secs:.1f}s to sample):")
        print(f"  std: mean {std.mean():.5g}  on object {std[obj].mean():.5g}  max {std.max():.5g}")
        print(f"  spearman vs |held-out error|: whole frame {spearman(std.ravel(), abs_err.ravel()):.4f}  "
              f"object only {spearman(std[obj], abs_err[obj]):.4f}")

    b_std, c_std = out["block-diagonal"][1], out["coupled"][1]
    # Restrict the ratio to pixels where the block-diagonal std is actually
    # resolved -- near-zero denominators make the raw max meaningless.
    live = obj & (b_std > 0.1 * b_std[obj].mean())
    ratio = c_std[live] / b_std[live]
    print(f"\ncoupled/block std ratio on {live.sum()} resolved object pixels: "
          f"median {np.median(ratio):.3f}  10th pct {np.percentile(ratio, 10):.3f}  "
          f"90th pct {np.percentile(ratio, 90):.3f}")

    vmax = max(b_std.max(), c_std.max())
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 7))
    for r, name in enumerate(["block-diagonal", "coupled"]):
        ens, std = out[name]
        for s in range(2):
            axes[r, s].imshow(ens[s])
            if r == 0:
                axes[r, s].set_title(f"posterior draw {s + 1}", fontsize=12)
        im = axes[r, 2].imshow(std, cmap="inferno", vmin=0, vmax=vmax)
        if r == 0:
            axes[r, 2].set_title("per-pixel std", fontsize=12)
        fig.colorbar(im, ax=axes[r, 2], fraction=0.046)
        im2 = axes[r, 3].imshow(abs_err, cmap="inferno")
        if r == 0:
            axes[r, 3].set_title("|held-out error|", fontsize=12)
        fig.colorbar(im2, ax=axes[r, 3], fraction=0.046)
        axes[r, 0].set_ylabel(name, fontsize=13)
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        for s in range(1, 4):
            axes[r, s].axis("off")
    fig.suptitle(f"{SCENE}: block-diagonal vs cross-splat-coupled SH posterior "
                 f"({N_VIEWS} training views, frozen map)", fontsize=14)
    fig.tight_layout()
    path = RESULTS_DIR / "coupled_comparison.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    run()
