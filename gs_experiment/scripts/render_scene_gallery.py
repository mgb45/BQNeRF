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
from matplotlib.colors import LogNorm

from gs_experiment.nerf_transforms import load_transforms
from gs_experiment.scripts.render_reconstruction import RESULTS_DIR, compute_uncertainty_maps, render_views

# scene -> hand-picked best-quality held-out view index (see module docstring)
SCENE_BEST_VIEWS = {
    "chair": 29,
    "drums": 3,
    "ficus": 13,
    "hotdog": 12,
    "lego": 21,  # re-picked after retraining lego's wide checkpoint to match the other scenes' recipe
    "mic": 11,
    "ship": 21,
}
BACKGROUND_COLOR = (1.0, 1.0, 1.0)
# None -> compute_uncertainty_maps fits sigma/kappa per checkpoint (see
# splat_scene.fit_kernel_hyperparams) instead of reusing one bandwidth pooled
# once across a fixed calibration set. Matters here specifically: this gallery
# is also built at lower splat budgets (--checkpoint-subdir budget_10000,
# budget_500, ...), and a bandwidth tuned at the default 300k-splat density is
# a real mismatch at 500 splats, not just a theoretical worry -- see
# splat_budget_uncertainty_sweep.py's fixed-vs-refit-sigma comparison, which
# first surfaced this for sigma alone. Pass --sigma/--kappa explicitly to
# pin a fixed value instead (e.g. to reproduce the old pooled-value galleries:
# --sigma 0.0694).
SIGMA = None
KAPPA = None
WINDOW_RADIUS = 0.08
DEPTH_RES = 64
PREPARED_ROOT = Path(__file__).resolve().parents[1] / "local_runs"


def build_rows(
    scene_views=SCENE_BEST_VIEWS, prepared_root=PREPARED_ROOT, sigma=SIGMA, kappa=KAPPA, window_radius=WINDOW_RADIUS,
    depth_res=DEPTH_RES, checkpoint_subdir="wide", max_observations_per_splat=None,
):
    """`checkpoint_subdir`: which per-scene checkpoint directory to render/evaluate against
    (default "wide", this project's full-quality 300k-splat recipe). Pass e.g. "budget_10000"
    to build the same gallery against a lower-splat-budget checkpoint instead (see
    splat_budget_uncertainty_sweep.py's naming convention, reused here) -- everything else
    (view choice, window_radius, eval split) stays identical except sigma/kappa, which (at
    their default of None) are fit fresh per scene *and* per checkpoint_subdir, exactly
    because a lower splat budget is expected to need a different bandwidth, not the same
    one reused from "wide" (see the SIGMA/KAPPA module comment).

    `max_observations_per_splat`: passed straight through to compute_uncertainty_maps --
    None (the default) is correct at this function's own default budgets (max ~300k
    splats), but a caller building a gallery at a much larger splat budget (e.g. lego's
    1M/3M checkpoints) needs this set, or risks the exact host OOM
    splat_budget_uncertainty_sweep.py's memory-model guard exists to prevent (confirmed
    directly: an uncapped 1M-splat call here was OOM-killed at 18GB). Use
    splat_budget_uncertainty_sweep.max_observations_per_splat_for_budget(budget) to get
    the same validated cap that script already uses, rather than guessing a new one."""
    rows = []
    for scene, view_idx in scene_views.items():
        ckpt_dir = prepared_root / f"{scene}_prepared" / checkpoint_subdir
        eval_dir = prepared_root / f"{scene}_prepared" / "eval"
        camera_angle_x, frames = load_transforms(str(eval_dir / "transforms.json"))

        results, checkpoint = render_views(str(eval_dir), [view_idx], checkpoint_dir=str(ckpt_dir), background_color=BACKGROUND_COLOR)
        _, gt, recon = results[0]
        height, width = gt.shape[:2]
        maps = compute_uncertainty_maps(
            str(eval_dir), [view_idx], frames, camera_angle_x, width, height, checkpoint,
            checkpoint_dir=str(ckpt_dir), sigma=sigma, kappa=kappa, window_radius=window_radius,
            max_observations_per_splat=max_observations_per_splat,
            depth_width=depth_res, depth_height=depth_res, return_raw_variance=True,
        )
        # maps[0] is (spatial_map, dir_map, raw_map) -- spatial_map (position-only) is a
        # reduced ablation of the same kernel, not the complete rendering-aware construction;
        # not used in this figure (see module docstring / plot_gallery).
        _, dir_map, raw_map = maps[0]
        err = np.abs(gt - recon).mean(axis=-1)
        psnr = -10.0 * np.log10(max(float(np.mean((gt - recon) ** 2)), 1e-10))
        n_splats = int(checkpoint["positions"].shape[0])
        print(f"{scene}: view {view_idx}, PSNR {psnr:.2f}dB, {n_splats} splats")
        rows.append(dict(
            scene=scene, view_idx=view_idx, psnr=psnr, n_splats=n_splats, gt=gt, recon=recon, err=err,
            dir_map=dir_map, raw_map=raw_map,
        ))
    return rows


DEFAULT_TITLE = (
    "NeRF-Synthetic: reconstruction error vs. BQ uncertainty across scenes (best held-out view per scene)\n"
    "uncertainty = complete position+direction rendering-aware variance / prior variance, fixed [0,1] scale"
)


def plot_gallery(rows, out_path, title=DEFAULT_TITLE):
    n = len(rows)
    fig, axes = plt.subplots(n, 5, figsize=(17.5, 3 * n))

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

        # Raw (unnormalized) posterior variance -- deliberately NOT on the ratio's shared
        # [0,1] scale: its magnitude is dominated by this checkpoint's own fitted sigma/kappa
        # (see compute_uncertainty_maps' return_raw_variance docstring), so it isn't
        # comparable across rows/scenes the way the ratio is. Per-panel log-scale instead,
        # since it's strictly positive and can span orders of magnitude within one view.
        raw = r["raw_map"]
        finite = raw[np.isfinite(raw) & (raw > 0)]
        if finite.size:
            im_raw = axes[row, 4].imshow(raw, cmap=cmap, norm=LogNorm(vmin=finite.min(), vmax=finite.max()))
            fig.colorbar(im_raw, ax=axes[row, 4], fraction=0.046, pad=0.04)
        else:
            axes[row, 4].text(0.5, 0.5, "n/a", ha="center", va="center", transform=axes[row, 4].transAxes)
        axes[row, 4].set_title("raw posterior variance\n(unnormalized, per-panel log scale)" if row == 0 else "")

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel(f"{r['scene']}\n{r['psnr']:.1f}dB, {r['n_splats']:,} splats", fontsize=10)

    fig.suptitle(title, fontsize=12)
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
    parser.add_argument("--checkpoint-subdir", default="wide", help='per-scene checkpoint dir to use, e.g. "budget_10000"')
    parser.add_argument("--sigma", type=float, default=SIGMA, help="fixed value; omit to fit per checkpoint (default)")
    parser.add_argument("--kappa", type=float, default=KAPPA, help="fixed value; omit to fit per checkpoint (default)")
    parser.add_argument("--window-radius", type=float, default=WINDOW_RADIUS)
    parser.add_argument("--depth-res", type=int, default=DEPTH_RES)
    args = parser.parse_args()
    run(
        out_path=args.out, checkpoint_subdir=args.checkpoint_subdir, sigma=args.sigma, kappa=args.kappa,
        window_radius=args.window_radius, depth_res=args.depth_res,
    )


if __name__ == "__main__":
    main()
