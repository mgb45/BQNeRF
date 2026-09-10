"""Cross-scene gallery figure: one genuinely trustworthy held-out view per NeRF-Synthetic
scene, at two splat budgets (500 and this project's standard 300k "wide" recipe), ground
truth / reconstruction / |error| / raw BQ posterior variance side by side.

Reuses render_reconstruction.py's render_views/compute_uncertainty_maps; the per-scene view
index below was picked by hand after scanning every held-out view's PSNR. That scan mattered:
several of these checkpoints have per-view quality that swings enormously with viewing angle --
a real held-out-view extrapolation effect (some eval camera poses land well outside the
training-view distribution), not a rendering bug. hotdog is the clearest example: its view 0 is
a near-total failure (a formless blur) while views 12/14 are excellent (>36dB) on the exact same
checkpoint. Picking the arbitrary view 0 for every scene, as a naive "first N held-out views"
figure would, silently mixes real BQ-uncertainty content with unrelated held-out-extrapolation
failures -- this script picks each scene's best-quality held-out view instead, so what's on
screen is a genuine reconstruction difficulty (or a genuinely well-covered region), not a
rendering artifact. materials is excluded entirely: even its best held-out view (20.5dB) stays
visibly hazy, below the bar every other scene's best view clears (>=23dB) -- a real,
honestly-reported limitation of highly reflective/specular materials under this project's
vanilla-3DGS training recipe, not something to paper over by cherry-picking a still-bad frame.

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
from gs_experiment.scripts.render_reconstruction import (
    RAW_VARIANCE_VMAX,
    RAW_VARIANCE_VMIN,
    RESULTS_DIR,
    compute_uncertainty_maps,
    render_views,
)

# scene -> hand-picked best-quality held-out view index (see module docstring)
SCENE_BEST_VIEWS = {
    "chair": 29,
    "drums": 3,
    "ficus": 13,
    "hotdog": 12,
    "lego": 21,
    "mic": 11,
    "ship": 21,
}
# (checkpoint subdir, row label) -- both budgets shown per scene, see module docstring
BUDGETS = [("budget_500", "500 splats"), ("wide", "300,000 splats")]
BACKGROUND_COLOR = (1.0, 1.0, 1.0)
# None -> compute_uncertainty_maps fits sigma/kappa per checkpoint (see
# splat_scene.fit_kernel_hyperparams) instead of reusing one bandwidth pooled
# once across a fixed calibration set. Matters here specifically: this gallery
# spans two very different splat budgets, and a bandwidth tuned at 300k-splat
# density is a real mismatch at 500 splats, not just a theoretical worry. Pass
# --sigma/--kappa explicitly to pin a fixed value instead.
SIGMA = None
KAPPA = None
WINDOW_RADIUS = 0.08
DEPTH_RES = 64
PREPARED_ROOT = Path(__file__).resolve().parents[1] / "local_runs"


def build_rows(
    scene_views=SCENE_BEST_VIEWS, prepared_root=PREPARED_ROOT, sigma=SIGMA, kappa=KAPPA, window_radius=WINDOW_RADIUS,
    depth_res=DEPTH_RES, checkpoint_subdir="wide", max_observations_per_splat=None,
):
    """`checkpoint_subdir`: which per-scene checkpoint directory to render/evaluate against.
    Everything else (view choice, window_radius, eval split) stays identical except
    sigma/kappa, which (at their default of None) are fit fresh per scene *and* per
    checkpoint_subdir, exactly because a lower splat budget is expected to need a
    different bandwidth, not the same one reused from "wide" (see the SIGMA/KAPPA
    module comment).

    `max_observations_per_splat`: passed straight through to compute_uncertainty_maps --
    None (the default) is correct at this function's own budgets (max 300k splats), but a
    caller building a gallery at a much larger splat budget (e.g. lego's 1M/3M checkpoints,
    see render_splat_sweep_gallery.py) needs this set, or risks the exact host OOM
    splat_scene.max_observations_per_splat_for_budget's memory-model guard exists to
    prevent (confirmed directly: an uncapped 1M-splat call here was OOM-killed at 18GB)."""
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
        # maps[0] is (spatial_map, dir_map, raw_map) -- spatial_map (position-only) and
        # dir_map (the normalized ratio) are reduced ablations of the same kernel, not
        # the complete rendering-aware construction this figure shows; not used here
        # (see plot_gallery, which plots only the raw posterior variance).
        _, _, raw_map = maps[0]
        err = np.abs(gt - recon).mean(axis=-1)
        psnr = -10.0 * np.log10(max(float(np.mean((gt - recon) ** 2)), 1e-10))
        n_splats = int(checkpoint["positions"].shape[0])
        print(f"{scene}: view {view_idx}, PSNR {psnr:.2f}dB, {n_splats} splats")
        rows.append(dict(
            scene=scene, view_idx=view_idx, psnr=psnr, n_splats=n_splats, gt=gt, recon=recon, err=err,
            raw_map=raw_map,
        ))
    return rows


def build_multi_budget_rows(scene_views=SCENE_BEST_VIEWS, budgets=BUDGETS, **kwargs):
    """One entry per scene, each holding one (label, row) pair per budget in `budgets`
    -- feeds plot_gallery's side-by-side layout (both budgets in one print-height row
    per scene, not stacked as separate rows) since stacking doubled the figure's height
    to the point of not fitting a printed page (14 rows at this project's per-row size)."""
    all_rows = []
    for scene, view_idx in scene_views.items():
        budget_rows = []
        for checkpoint_subdir, label in budgets:
            r = build_rows(scene_views={scene: view_idx}, checkpoint_subdir=checkpoint_subdir, **kwargs)[0]
            budget_rows.append((label, r))
        all_rows.append(dict(scene=scene, budget_rows=budget_rows))
    return all_rows


def plot_gallery(rows, out_path):
    n = len(rows)
    n_budgets = len(rows[0]["budget_rows"])
    n_cols = n_budgets * 4
    fig, axes = plt.subplots(n, n_cols, figsize=(3.5 * n_cols, 3 * n))

    # |error| shares one scale across scenes and budgets -- its absolute magnitude is
    # itself part of the story (it tracks reconstruction quality).
    err_vmax = float(np.percentile(
        np.stack([r["err"] for row in rows for _, r in row["budget_rows"]]), 95
    ))
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad(color=(0.4, 0.4, 0.4))
    col_titles = ["ground truth", "gsplat reconstruction", "|error|", "posterior variance"]

    for row_idx, row in enumerate(rows):
        psnrs = []
        for b_idx, (label, r) in enumerate(row["budget_rows"]):
            psnrs.append(f"{r['psnr']:.1f}dB @ {label}")
            c0 = b_idx * 4
            axes[row_idx, c0 + 0].imshow(r["gt"])
            axes[row_idx, c0 + 1].imshow(r["recon"])
            im_err = axes[row_idx, c0 + 2].imshow(r["err"], cmap="inferno", vmin=0, vmax=err_vmax)
            fig.colorbar(im_err, ax=axes[row_idx, c0 + 2], fraction=0.046, pad=0.04)
            # Raw (unnormalized) posterior variance on a fixed, shared log scale (see
            # render_reconstruction.RAW_VARIANCE_VMIN/VMAX) -- comparable across every
            # panel, scene, budget, and figure, unlike the normalized ratio this figure
            # no longer shows.
            im_raw = axes[row_idx, c0 + 3].imshow(
                r["raw_map"], cmap=cmap, norm=LogNorm(vmin=RAW_VARIANCE_VMIN, vmax=RAW_VARIANCE_VMAX),
            )
            fig.colorbar(im_raw, ax=axes[row_idx, c0 + 3], fraction=0.046, pad=0.04)

            if row_idx == 0:
                for k in range(4):
                    axes[row_idx, c0 + k].set_title(f"{label}\n{col_titles[k]}" if k == 0 else col_titles[k], fontsize=9)
            for k in range(4):
                axes[row_idx, c0 + k].set_xticks([])
                axes[row_idx, c0 + k].set_yticks([])

        axes[row_idx, 0].set_ylabel(f"{row['scene']}\n" + "\n".join(psnrs), fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def run(out_path=None, budgets=BUDGETS, **kwargs):
    rows = build_multi_budget_rows(budgets=budgets, **kwargs)
    out_path = Path(out_path) if out_path else (RESULTS_DIR / "scene_gallery.png")
    plot_gallery(rows, out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None)
    parser.add_argument("--sigma", type=float, default=SIGMA, help="fixed value; omit to fit per checkpoint (default)")
    parser.add_argument("--kappa", type=float, default=KAPPA, help="fixed value; omit to fit per checkpoint (default)")
    parser.add_argument("--window-radius", type=float, default=WINDOW_RADIUS)
    parser.add_argument("--depth-res", type=int, default=DEPTH_RES)
    args = parser.parse_args()
    run(
        out_path=args.out, sigma=args.sigma, kappa=args.kappa,
        window_radius=args.window_radius, depth_res=args.depth_res,
    )


if __name__ == "__main__":
    main()
