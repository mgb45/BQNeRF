"""Where does this uncertainty actually predict error?

Two hypotheses for the weak object-pixel correlation on lego `wide` have now
been tested and rejected: cross-splat coupling (FINDINGS section 6) and
geometry-in-the-posterior (section 7). Both made calibration WORSE, not
better. That points at the premise rather than the model.

A posterior over splat parameters measures EPISTEMIC uncertainty -- what the
training views failed to determine. On a 300k-splat checkpoint fit to 100
well-spread views, held-out error is not mostly epistemic: it is dominated
by model misspecification and resolution limits (thin geometry, aliasing at
edges), which no posterior over the fitted parameters can see, because the
data really does pin those parameters down. If that reading is right, the
uncertainty should predict error well exactly where epistemic error
dominates, and poorly where it does not -- and the fix is not a better
posterior but an honest statement of what the signal is for.

This tests it on the `gap_*` checkpoints, which were trained with a
deliberate angular hole carved out of the training views. For each, the
posterior is conditioned on that checkpoint's OWN real training cameras, and
every held-out eval view is scored. A view inside the hole has genuinely
epistemic error; a view outside it does not.

Run: .venv-gsplat/bin/python gs_experiment/scripts/render_epistemic_regime.py
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

from gs_experiment.nerf_transforms import fov_x_to_intrinsics, load_transforms, opencv_viewmat_from_c2w
from gs_experiment.ply_io import read_3dgs_ply
from gs_experiment.rasterized_sh_precision import accumulate_sh_precision_rasterized, estimate_noise_variance
from gs_experiment.scripts.render_geometry_ensemble import ause, render, spearman
from gs_experiment.scripts.render_posterior_ensemble import (
    BACKGROUND_COLOR, N_PROBES, SEED, empirical_band_precision, sample_sh_draws,
)
from gs_experiment.scripts.render_reconstruction import LOCAL_RUNS, RESULTS_DIR

PREPARED = LOCAL_RUNS / "lego_prepared"
GAPS = [("gap_0", 0), ("gap_2", 30), ("gap_4", 75)]
REFERENCE_IDX = 0
N_DRAWS = 16


def run():
    rows = []
    for ckpt_name, half_width in GAPS:
        ckpt_dir, eval_dir = PREPARED / ckpt_name, PREPARED / f"{ckpt_name}_eval"
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

        # Gap centre: the direction the training views were removed around.
        centres = np.array([c2w[:3, 3] for _, c2w in train_frames])
        _, wide_frames = load_transforms(str(PREPARED / "wide" / "transforms.json"))
        wide_c = np.array([c2w[:3, 3] for _, c2w in wide_frames])
        gap_dir = wide_c[REFERENCE_IDX] / np.linalg.norm(wide_c[REFERENCE_IDX])

        ecax, eval_frames = load_transforms(str(eval_dir / "transforms.json"))
        with Image.open(str(eval_dir / (eval_frames[0][0] + ".png"))) as im:
            width, height = im.size
        K = fov_x_to_intrinsics(ecax, width, height)
        background = torch.tensor(BACKGROUND_COLOR, dtype=torch.float32, device="cuda")

        per_view = []
        for vi, (file_path, c2w) in enumerate(eval_frames):
            d = c2w[:3, 3] / np.linalg.norm(c2w[:3, 3])
            ang = float(np.degrees(np.arccos(np.clip(d @ gap_dir, -1, 1))))
            viewmat = torch.tensor(opencv_viewmat_from_c2w(c2w), dtype=torch.float32, device="cuda")[None]
            Ks = torch.tensor(K, dtype=torch.float32, device="cuda")[None]
            gt = np.asarray(Image.open(str(eval_dir / (file_path + ".png"))).convert("RGB"),
                            dtype=np.float32) / 255.0
            obj = gt.min(axis=2) < 0.99
            mean_render = render(checkpoint, theta_hat, op_hat, viewmat, Ks, width, height, background)
            err = np.abs(mean_render - gt).mean(axis=2)
            ens = np.stack([render(checkpoint, draws[s], op_hat, viewmat, Ks, width, height, background)
                            for s in range(N_DRAWS)], axis=0)
            std = ens.std(axis=0).mean(axis=2)
            a, _, _ = ause(std[obj], err[obj])
            per_view.append({"ang": ang, "err": err[obj].mean(), "std": std[obj].mean(),
                             "sp": spearman(std[obj], err[obj]), "ause": a})

        inside = [v for v in per_view if v["ang"] < max(half_width, 1e-9)]
        outside = [v for v in per_view if v["ang"] >= max(half_width, 1e-9)]
        pv_sp = spearman(np.array([v["std"] for v in per_view]), np.array([v["err"] for v in per_view]))
        rows.append({"name": ckpt_name, "hw": half_width, "per_view": per_view,
                     "inside": inside, "outside": outside, "pv_sp": pv_sp})
        print(f"\n{ckpt_name} (gap half-width {half_width} deg, {len(train_frames)} training views)")
        print(f"  per-VIEW spearman, mean uncertainty vs mean error, over {len(per_view)} eval views: {pv_sp:.3f}")
        for label, grp in (("inside gap", inside), ("outside gap", outside)):
            if grp:
                print(f"  {label:<12} n={len(grp):>2}  mean err {np.mean([v['err'] for v in grp]):.5f}  "
                      f"mean std {np.mean([v['std'] for v in grp]):.5f}  "
                      f"per-pixel sp {np.mean([v['sp'] for v in grp]):.3f}  "
                      f"AUSE {np.mean([v['ause'] for v in grp]):.4f}")

    fig, axes = plt.subplots(1, len(rows), figsize=(5.0 * len(rows), 4.4), squeeze=False)
    max_ang = max(v["ang"] for row in rows for v in row["per_view"])
    norm = plt.Normalize(0.0, max_ang)
    sc = None
    for j, row in enumerate(rows):
        ax = axes[0, j]
        ang = np.array([v["ang"] for v in row["per_view"]])
        e = np.array([v["err"] for v in row["per_view"]])
        s_ = np.array([v["std"] for v in row["per_view"]])
        sc = ax.scatter(e, s_, c=ang, cmap="viridis", norm=norm, s=44)
        ax.set_xlabel("mean held-out error (object px)")
        ax.set_ylabel("mean predicted std")
        ax.set_title(f"{row['name']}: gap {row['hw']}deg\nper-view spearman {row['pv_sp']:.2f}", fontsize=12)
        if row["inside"]:
            ax.axvline(np.mean([v["err"] for v in row["inside"]]), ls=":", c="crimson", lw=1)
    fig.colorbar(sc, ax=axes[0, -1], label="angle from gap centre (deg)")
    fig.suptitle("lego: does predicted uncertainty track held-out error when the error is epistemic?",
                 fontsize=13)
    fig.tight_layout()
    path = RESULTS_DIR / "epistemic_regime.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    run()
