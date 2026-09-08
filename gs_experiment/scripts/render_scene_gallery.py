"""Cross-scene gallery figure: one genuinely trustworthy held-out view per NeRF-Synthetic
scene, ground truth / reconstruction / |error| / BQ uncertainty side by side -- the
"pixel-wise renders of complex scenes, uncertainty vs. reconstruction error" figure.

Reuses render_reconstruction.py's render_views/compute_uncertainty_maps; the per-scene view
index below was picked by hand after scanning every held-out view's PSNR, not the arbitrary
evenly-spaced views run_synthetic_pipeline.py's --n-examples uses. That scan mattered: several
of these checkpoints have per-view quality that swings enormously with viewing angle -- a real
held-out-view extrapolation effect (some eval camera poses land well outside the training-view
distribution), not a rendering bug. hotdog is the clearest example: its view 0 is a near-total
failure (a formless blur) while views 12/14 are excellent (>36dB) on the exact same checkpoint.
Picking the arbitrary view 0 for every scene, as a naive "first N held-out views" figure would,
silently mixes real BQ-uncertainty content with unrelated held-out-extrapolation failures --
this script picks each scene's best-quality held-out view instead, so what's on screen is a
genuine reconstruction difficulty (or a genuinely well-covered region), not a rendering
artifact. materials is excluded entirely: even its best held-out view (20.5dB) stays visibly
hazy, below the bar every other scene's best view clears (>=23dB) -- a real, honestly-reported
limitation of highly reflective/specular materials under this project's vanilla-3DGS training
recipe, not something to paper over by cherry-picking a still-bad frame.

Needs torch + gsplat (requirements-gsplat.txt).

Run: .venv-gsplat/bin/python gs_experiment/scripts/render_scene_gallery.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gs_experiment.nerf_transforms import load_transforms
from gs_experiment.scripts.render_reconstruction import RESULTS_DIR, compute_uncertainty_maps, render_views

# scene -> hand-picked best-quality held-out view index (see module docstring)
SCENE_BEST_VIEWS = {
    "chair": 29,
    "drums": 3,
    "ficus": 13,
    "hotdog": 12,
    "lego": 27,
    "mic": 11,
    "ship": 21,
}
BACKGROUND_COLOR = (1.0, 1.0, 1.0)
# Marginal-likelihood-fitted (gs_experiment/hyperparams.py::fit_kernel_param_pooled_nd), pooled
# across 9 real checkpoints spanning both this figure's scenes and the lego gap-sweep figure's
# checkpoints, not hand-picked -- held-out log marginal likelihood -8607 vs -11694 at the old
# hardcoded 0.05, and -35.8 MILLION at the coverage-sweep script's old 0.9 (not a typo: that
# value was catastrophically wrong for real data, not just suboptimal). The same value is used
# in render_coverage_uncertainty_sweep.py so results are comparable across both figures.
SIGMA = 0.0694
WINDOW_RADIUS = 0.08
DEPTH_RES = 64
PREPARED_ROOT = Path(__file__).resolve().parents[1] / "local_runs"


def build_rows(scene_views=SCENE_BEST_VIEWS, prepared_root=PREPARED_ROOT, sigma=SIGMA, window_radius=WINDOW_RADIUS, depth_res=DEPTH_RES):
    rows = []
    for scene, view_idx in scene_views.items():
        wide_dir = prepared_root / f"{scene}_prepared" / "wide"
        eval_dir = prepared_root / f"{scene}_prepared" / "eval"
        camera_angle_x, frames = load_transforms(str(eval_dir / "transforms.json"))

        results, checkpoint = render_views(str(eval_dir), [view_idx], checkpoint_dir=str(wide_dir), background_color=BACKGROUND_COLOR)
        _, gt, recon = results[0]
        height, width = gt.shape[:2]
        maps = compute_uncertainty_maps(
            str(eval_dir), [view_idx], frames, camera_angle_x, width, height, checkpoint,
            checkpoint_dir=str(wide_dir), sigma=sigma, window_radius=window_radius,
            depth_width=depth_res, depth_height=depth_res,
        )
        # maps[0] is (spatial_map, dir_map) -- spatial_map (position-only) is a reduced
        # ablation of the same kernel, not the complete rendering-aware construction; not
        # used in this figure (see module docstring / plot_gallery).
        _, dir_map = maps[0]
        err = np.abs(gt - recon).mean(axis=-1)
        psnr = -10.0 * np.log10(max(float(np.mean((gt - recon) ** 2)), 1e-10))
        print(f"{scene}: view {view_idx}, PSNR {psnr:.2f}dB, {checkpoint['positions'].shape[0]} splats")
        rows.append(dict(scene=scene, view_idx=view_idx, psnr=psnr, gt=gt, recon=recon, err=err, dir_map=dir_map))
    return rows


def plot_gallery(rows, out_path):
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3 * n))

    # |error| shares one scale across scenes -- its absolute magnitude is itself part of
    # the story (it tracks each scene's PSNR).
    err_vmax = float(np.percentile(np.stack([r["err"] for r in rows]), 95))
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad(color=(0.4, 0.4, 0.4))

    # dir_map is variance/prior_variance in [0,1] (see compute_uncertainty_maps' docstring):
    # 0 = fully informed by real data, 1 = as uncertain as a query with zero relevant
    # observations. A FIXED [0,1] scale, not a per-scene/per-panel percentile, is used
    # deliberately: a per-panel autoscale (e.g. vmax = that panel's own 95th percentile)
    # always paints *something* as "hot" regardless of whether the scene's real uncertainty
    # ever gets close to the noise floor vs. genuinely uninformed -- exactly the complaint
    # that motivated this ratio in the first place. This scale lets a reader compare the
    # *absolute* level across every panel and every figure directly.
    for row, r in enumerate(rows):
        axes[row, 0].imshow(r["gt"])
        axes[row, 0].set_title("ground truth" if row == 0 else "")
        axes[row, 1].imshow(r["recon"])
        axes[row, 1].set_title("gsplat reconstruction" if row == 0 else "")
        im_err = axes[row, 2].imshow(r["err"], cmap="inferno", vmin=0, vmax=err_vmax)
        axes[row, 2].set_title("|error| (mean over RGB)" if row == 0 else "")
        fig.colorbar(im_err, ax=axes[row, 2], fraction=0.046, pad=0.04)
        im_dir = axes[row, 3].imshow(r["dir_map"], cmap=cmap, vmin=0, vmax=1)
        axes[row, 3].set_title("uncertainty ratio\n(variance / prior variance)" if row == 0 else "")
        fig.colorbar(im_dir, ax=axes[row, 3], fraction=0.046, pad=0.04)

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel(f"{r['scene']}\n{r['psnr']:.1f}dB", fontsize=10)

    fig.suptitle(
        "NeRF-Synthetic: reconstruction error vs. BQ uncertainty across scenes (best held-out view per scene)\n"
        "uncertainty = complete position+direction rendering-aware variance / prior variance, fixed [0,1] scale",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def run(out_path=None, **kwargs):
    rows = build_rows(**kwargs)
    out_path = Path(out_path) if out_path else (RESULTS_DIR / "scene_gallery.png")
    plot_gallery(rows, out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None)
    parser.add_argument("--sigma", type=float, default=SIGMA)
    parser.add_argument("--window-radius", type=float, default=WINDOW_RADIUS)
    parser.add_argument("--depth-res", type=int, default=DEPTH_RES)
    args = parser.parse_args()
    run(out_path=args.out, sigma=args.sigma, window_radius=args.window_radius, depth_res=args.depth_res)


if __name__ == "__main__":
    main()
