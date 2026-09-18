"""Frozen-map view-count sweep: the same scene, the same splats, the same
held-out view -- only the number of training views the appearance posterior
is CONDITIONED on changes.

This is the figure `render_posterior_ensemble.py` could not produce. On the
full 100-view `wide` checkpoint the SH posterior is genuinely tight (per-
pixel std ~0.002, four draws visually identical), which is the correct
answer and a good sanity result but shows nothing at a glance. Spread
appears when the data is thin -- so vary the data, not the model.

Everything except the conditioning set is frozen: ONE checkpoint, loaded
once. Geometry, opacities, the stored SH coefficients, the query camera and
the render are never touched. Only `D_i = sum_p (sum_q beta^2) phi phi^T` is
re-accumulated over a subset of the real training cameras. So this is a
strictly nested-conditioning experiment with NO retraining confound at all
-- unlike comparing independently trained checkpoints (`render_angle_
sweep.py`), where splat count, positions, opacities, learned coefficients
and the query-side weights all move at once and nothing can be attributed.

It is also cheap: re-accumulating over a camera subset is ~5s for the full
100 cameras (one forward+backward render per camera -- see
`rasterized_sh_precision`), so an entire sweep costs less than a single
run of the KNN-surrogate accumulation it replaces.

Run: .venv-gsplat/bin/python gs_experiment/scripts/render_posterior_view_sweep.py
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
from gs_experiment.rasterized_sh_precision import accumulate_sh_precision_rasterized, estimate_noise_variance
from gs_experiment.scripts.render_posterior_ensemble import (
    BACKGROUND_COLOR, N_PROBES, SEED, SH_BANDS,
    empirical_band_precision, render_with_coefficients, sample_sh_draws,
)
from gs_experiment.scripts.render_reconstruction import EVAL_DIRS, LOCAL_RUNS, RESULTS_DIR

SCENE = "lego"
CHECKPOINT = "wide"
VIEW_IDX = 21
N_VIEWS_SWEEP = [100, 25, 8, 3]
N_DRAWS = 8
N_SHOW = 3


def evenly_spaced(n_total: int, k: int) -> list[int]:
    """`k` training-camera indices spread as evenly as possible over the real
    capture order -- NOT the first k. The NeRF-Synthetic training cameras
    orbit the object, so the first k would all sit in one arc and confound
    "few views" with "one narrow arc"; evenly spaced keeps angular coverage
    as uniform as the subset size allows, so the sweep isolates view COUNT."""
    return sorted(set(np.linspace(0, n_total - 1, k).astype(int).tolist()))


def run():
    ckpt_dir = LOCAL_RUNS / f"{SCENE}_prepared" / CHECKPOINT
    eval_dir = EVAL_DIRS[SCENE]
    checkpoint = read_3dgs_ply(str(ckpt_dir / "splats.ply"))
    sh_coeffs, sh_degree = checkpoint["sh_coeffs"], checkpoint["sh_degree"]
    print(f"frozen checkpoint: {sh_coeffs.shape[0]} splats, sh_degree={sh_degree}")

    train_cax, train_frames = load_transforms(str(ckpt_dir / "transforms.json"))
    with Image.open(str(ckpt_dir / (train_frames[0][0] + ".png"))) as im:
        train_w, train_h = im.size
    train_K = fov_x_to_intrinsics(train_cax, train_w, train_h)

    noise_var = estimate_noise_variance(checkpoint, train_frames, train_K, train_w, train_h,
                                        str(ckpt_dir), background_color=BACKGROUND_COLOR)
    band_precision = empirical_band_precision(sh_coeffs, sh_degree)
    print(f"sigma_n^2 = {noise_var:.6g}, prior lambda_l0 (R) = {band_precision[0, 0]:.4g}")

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
    mean_render = render_with_coefficients(checkpoint, theta_hat, viewmat, Ks, width, height, background)
    obj = gt.min(axis=2) < 0.99

    rows = []
    for n_views in N_VIEWS_SWEEP:
        picks = evenly_spaced(len(train_frames), n_views)
        t0 = time.time()
        data_precision = accumulate_sh_precision_rasterized(
            checkpoint, train_frames, train_K, train_w, train_h, sh_degree,
            n_probes=N_PROBES, seed=SEED, device="cuda", progress_every=0,
            camera_indices=picks,
        ) / noise_var
        accum_s = time.time() - t0

        diag = np.einsum("nkk->nk", data_precision)
        ratio = diag[:, 0] / band_precision[0, 0]
        draws = sample_sh_draws(sh_coeffs, data_precision, band_precision, N_DRAWS, SEED)
        ensemble = np.stack([
            render_with_coefficients(checkpoint, draws[s], viewmat, Ks, width, height, background)
            for s in range(N_DRAWS)
        ], axis=0)
        std_map = ensemble.std(axis=0).mean(axis=2)
        rows.append({"n_views": n_views, "ensemble": ensemble, "std_map": std_map})
        print(f"{n_views:>4} views ({accum_s:.1f}s): l=0 data/prior median {np.median(ratio):8.3g}  "
              f"std mean {std_map.mean():.5g}  std on object {std_map[obj].mean():.5g}  "
              f"max {std_map.max():.5g}")

    vmax = max(r["std_map"].max() for r in rows)
    n_col = N_SHOW + 1
    fig, axes = plt.subplots(len(rows), n_col, figsize=(3.1 * n_col, 3.25 * len(rows)))
    for r, row in enumerate(rows):
        for s in range(N_SHOW):
            axes[r, s].imshow(row["ensemble"][s])
            if r == 0:
                axes[r, s].set_title(f"posterior draw {s + 1}", fontsize=13)
        im = axes[r, N_SHOW].imshow(row["std_map"], cmap="inferno", vmin=0, vmax=vmax)
        if r == 0:
            axes[r, N_SHOW].set_title("per-pixel std", fontsize=13)
        fig.colorbar(im, ax=axes[r, N_SHOW], fraction=0.046)
        axes[r, 0].set_ylabel(f"{row['n_views']} training views", fontsize=14)
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        for s in range(1, n_col):
            axes[r, s].axis("off")
    fig.suptitle(
        f"{SCENE}: one frozen checkpoint -- only the number of training views the appearance "
        f"posterior is conditioned on changes", fontsize=15)
    fig.tight_layout()
    out_path = RESULTS_DIR / "posterior_view_sweep.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nwrote {out_path}")
    return rows


if __name__ == "__main__":
    run()
