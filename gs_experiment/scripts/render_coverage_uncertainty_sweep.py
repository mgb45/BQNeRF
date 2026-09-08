"""Pixel-wise view-coverage sweep figure: the same held-out view, rendered
and BQ-queried against each of the real lego gap checkpoints already built
by real_directional_coverage_experiment.py (--dataset lego) -- gap
half-widths 0/15/30/50/75deg around one reference training-view direction,
every other training view left untouched (FINDINGS.md section 3's design).

That experiment only ever reported a scalar summary (variance at one fixed
query point/direction per condition, plotted as a line graph -- see
gap_directional_gradient.png). This renders the actual per-pixel picture
instead: pick the one held-out eval view closest to the gap center (so it's
genuinely inside every condition's gap, most severely in the widest one),
and for each condition show ground truth, reconstruction, |error|, and
position+direction BQ variance side by side -- so a reader can see, by eye,
both the reconstruction degrading and BQ's own uncertainty growing in
exactly the same missing-coverage region as the gap widens, without having
to trust a single summary number.

Needs the checkpoints already trained by real_directional_coverage_experiment.py
--dataset lego (gap_{i}/splats.ply and gap_{i}_eval/ for i in range(len(gap
half-widths))); does not retrain anything itself. Needs torch + gsplat
(requirements-gsplat.txt).

Run: .venv-gsplat/bin/python gs_experiment/scripts/render_coverage_uncertainty_sweep.py \\
    gs_experiment/local_runs/lego_prepared
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gs_experiment.nerf_transforms import load_transforms
from gs_experiment.scripts.real_directional_coverage_experiment import (
    LEGO_GAP_GATE_BACKGROUND_COLOR,
    LEGO_GAP_HALF_WIDTHS_DEG,
    LEGO_GAP_KAPPA,
    LEGO_GAP_SIGMA,
    LEGO_GAP_WINDOW_RADIUS,
    REFERENCE_IDX,
)
from gs_experiment.scripts.render_reconstruction import RESULTS_DIR, compute_uncertainty_maps, render_views

DEPTH_RES = 64  # square, matching NeRF-Synthetic's square images (see run_synthetic_pipeline.py)


def pick_gap_query_view(prepared_dir: str, condition_prefix: str):
    """The held-out eval view whose direction is closest to the gap center
    -- genuinely well-covered at gap_0 (no gap yet) and genuinely inside
    every wider condition's excluded zone, so the same view tells the whole
    coverage story across all columns."""
    _, wide_frames = load_transforms(os.path.join(prepared_dir, "wide", "transforms.json"))
    centers = np.array([c2w[:3, 3] for _, c2w in wide_frames])
    dirs = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    gap_center_direction = dirs[REFERENCE_IDX]

    eval_dir = os.path.join(prepared_dir, f"{condition_prefix}_0_eval")
    camera_angle_x, eval_frames = load_transforms(os.path.join(eval_dir, "transforms.json"))
    eval_centers = np.array([c2w[:3, 3] for _, c2w in eval_frames])
    eval_dirs = eval_centers / np.linalg.norm(eval_centers, axis=1, keepdims=True)
    angular_dist_deg = np.degrees(np.arccos(np.clip(eval_dirs @ gap_center_direction, -1.0, 1.0)))
    query_idx = int(np.argmin(angular_dist_deg))
    return query_idx, float(angular_dist_deg[query_idx]), camera_angle_x, eval_frames


def build_columns(
    prepared_dir: str,
    gap_half_widths=LEGO_GAP_HALF_WIDTHS_DEG,
    condition_prefix: str = "gap",
    depth_res: int = DEPTH_RES,
    sigma: float = LEGO_GAP_SIGMA,
    kappa: float = LEGO_GAP_KAPPA,
    window_radius: float = LEGO_GAP_WINDOW_RADIUS,
    background_color=LEGO_GAP_GATE_BACKGROUND_COLOR,
):
    query_idx, query_gap_dist_deg, camera_angle_x, eval_frames = pick_gap_query_view(prepared_dir, condition_prefix)
    print(f"query held-out view: index {query_idx}, {query_gap_dist_deg:.1f} deg from gap center direction")

    columns = []
    for i, hw in enumerate(gap_half_widths):
        zone_dir = os.path.join(prepared_dir, f"{condition_prefix}_{i}")
        zone_eval_dir = zone_dir + "_eval"
        _, train_frames = load_transforms(os.path.join(zone_dir, "transforms.json"))

        results, checkpoint = render_views(zone_eval_dir, [query_idx], background_color=background_color)
        _, gt, recon = results[0]
        height, width = gt.shape[:2]
        # checkpoint_dir=zone_dir is required, not optional: without it, compute_uncertainty_maps
        # falls back to attributing splat-observing-cameras against zone_eval_dir's transforms.json
        # (the held-out eval split, which spans the full orbit and is untouched by the gap) instead
        # of zone_dir's (the actual gap-restricted training pool this whole experiment manipulates)
        # -- silently substituting a full, ungapped camera pool for the directional-coverage
        # attribution and defeating the experiment's entire premise (confirmed directly: every
        # sampled pixel showed ratio~0.997, i.e. correctly near-maximal uncertainty, once this was
        # fixed and checked against a real per-pixel diagnostic -- the ~0.01 figures this bug
        # produced were a real, wrong result, not a colormap-scale artifact).
        maps = compute_uncertainty_maps(
            zone_eval_dir, [query_idx], eval_frames, camera_angle_x, width, height, checkpoint,
            checkpoint_dir=zone_dir, sigma=sigma, kappa=kappa, window_radius=window_radius,
            depth_width=depth_res, depth_height=depth_res,
        )
        _, dir_map = maps[0]
        err = np.abs(gt - recon).mean(axis=-1)
        psnr = -10.0 * np.log10(max(float(np.mean((gt - recon) ** 2)), 1e-10))
        print(
            f"gap {hw:.0f} deg: {len(train_frames)} train views, query-view PSNR {psnr:.2f}dB, "
            f"uncertainty ratio mean={np.nanmean(dir_map):.4f} median={np.nanmedian(dir_map):.4f} "
            f"p95={np.nanpercentile(dir_map, 95):.4f} max={np.nanmax(dir_map):.4f}"
        )

        columns.append(dict(
            gap_deg=hw, n_train_views=len(train_frames), psnr=psnr,
            gt=gt, recon=recon, err=err, dir_map=dir_map,
        ))
    return columns


def plot_coverage_sweep(columns, out_path):
    n = len(columns)
    # A shared color scale across the pixel-map columns is correct here (unlike the
    # cross-scene gallery) -- these are the same checkpoint family at the same view, so
    # absolute BQ-variance magnitude is directly comparable condition to condition. But
    # the mean shift across conditions turns out to be real and monotonic-ish yet modest
    # (~30-40%, see the printed per-condition stats) against a much larger pixel-to-pixel
    # spatial range within any single condition -- easy to miss by eye against a shared
    # scale. A 5th row makes the trend explicit as numbers, not just color.
    fig = plt.figure(figsize=(3 * n, 15.5))
    gs = fig.add_gridspec(5, n, height_ratios=[3, 3, 3, 3, 2])
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(n)] for r in range(4)])
    trend_ax = fig.add_subplot(gs[4, :])

    err_vmax = float(np.percentile(np.stack([c["err"] for c in columns]), 95))
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad(color=(0.4, 0.4, 0.4))

    # dir_map is variance/prior_variance in [0,1] (see compute_uncertainty_maps' docstring).
    # A fixed [0,1] scale, not a per-column percentile, matches render_scene_gallery.py and
    # is what actually lets the mean-value labels below mean something on sight, rather than
    # a colorbar that autoscales to whatever this specific sweep's own range happens to be.
    for col, c in enumerate(columns):
        mean_ratio = float(np.nanmean(c["dir_map"]))
        axes[0, col].imshow(c["gt"])
        axes[0, col].set_title(f"gap ±{c['gap_deg']:.0f}°\n{c['n_train_views']} train views, {c['psnr']:.1f}dB")
        axes[1, col].imshow(c["recon"])
        im_err = axes[2, col].imshow(c["err"], cmap="inferno", vmin=0, vmax=err_vmax)
        fig.colorbar(im_err, ax=axes[2, col], fraction=0.046, pad=0.04)
        im_dir = axes[3, col].imshow(c["dir_map"], cmap=cmap, vmin=0, vmax=1)
        fig.colorbar(im_dir, ax=axes[3, col], fraction=0.046, pad=0.04)
        axes[3, col].set_xlabel(f"mean ratio={mean_ratio:.3f}", fontsize=9)
        for ax in axes[:, col]:
            ax.set_xticks([])
            ax.set_yticks([])

    axes[0, 0].set_ylabel("ground truth\n(held-out view)", fontsize=10)
    axes[1, 0].set_ylabel("gsplat\nreconstruction", fontsize=10)
    axes[2, 0].set_ylabel("|error|\n(mean over RGB)", fontsize=10)
    axes[3, 0].set_ylabel("uncertainty ratio\n(variance / prior variance)", fontsize=10)

    gap_degs = [c["gap_deg"] for c in columns]
    mean_ratios = [float(np.nanmean(c["dir_map"])) for c in columns]
    median_ratios = [float(np.nanmedian(c["dir_map"])) for c in columns]
    trend_ax.plot(gap_degs, mean_ratios, "o-", color="tab:red", label="mean")
    trend_ax.plot(gap_degs, median_ratios, "s--", color="tab:orange", label="median")
    trend_ax.set_xlabel("gap half-width (deg)")
    trend_ax.set_ylabel("uncertainty ratio\n(variance / prior variance, averaged over the whole held-out view)")
    trend_ax.set_ylim(0, 1)
    trend_ax.set_title("same numbers as the row above, averaged over every pixel per condition")
    trend_ax.legend()
    trend_ax.grid(alpha=0.3)

    fig.suptitle(
        "Lego: same held-out view, decreasing training-view coverage (left→right) --\n"
        "reconstruction degrades and BQ directional uncertainty grows together in the missing-coverage region",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def run(prepared_dir: str, out_path=None, **kwargs):
    columns = build_columns(prepared_dir, **kwargs)
    out_path = Path(out_path) if out_path else (RESULTS_DIR / "coverage_uncertainty_sweep.png")
    plot_coverage_sweep(columns, out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prepared_dir", help="e.g. gs_experiment/local_runs/lego_prepared")
    parser.add_argument("--condition-prefix", default="gap")
    parser.add_argument("--out", default=None)
    parser.add_argument("--sigma", type=float, default=LEGO_GAP_SIGMA)
    parser.add_argument("--kappa", type=float, default=LEGO_GAP_KAPPA)
    parser.add_argument("--window-radius", type=float, default=LEGO_GAP_WINDOW_RADIUS)
    parser.add_argument("--depth-res", type=int, default=DEPTH_RES)
    args = parser.parse_args()

    run(
        args.prepared_dir, out_path=args.out, condition_prefix=args.condition_prefix,
        depth_res=args.depth_res, sigma=args.sigma, kappa=args.kappa, window_radius=args.window_radius,
    )


if __name__ == "__main__":
    main()
