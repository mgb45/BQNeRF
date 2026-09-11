"""Pixel-wise view-coverage sweep figure: the same held-out view, rendered
and BQ-queried against each of the real lego gap checkpoints already built
by real_directional_coverage_experiment.py (--dataset lego) -- gap
half-widths 0/15/30/50/75deg around one reference training-view direction,
every other training view left untouched (FINDINGS.md section 3's design).

That experiment only ever reported a scalar summary (variance at one fixed
query point/direction per condition, plotted as a line graph -- retired,
see git history). This renders the actual per-pixel picture instead:
pick the one held-out eval view closest to the gap center (so it's
genuinely inside every condition's gap, most severely in the widest one),
and for each condition show ground truth, reconstruction, |error|, and raw
position+direction BQ posterior variance side by side -- so a reader can see,
by eye, both the reconstruction degrading and BQ's own uncertainty growing in
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
from matplotlib.colors import LogNorm

from gs_experiment.nerf_transforms import load_transforms
from gs_experiment.scripts.real_directional_coverage_experiment import (
    LEGO_GAP_GATE_BACKGROUND_COLOR,
    LEGO_GAP_HALF_WIDTHS_DEG,
    LEGO_GAP_KAPPA,
    LEGO_GAP_NOISE_VARIANCE,
    LEGO_GAP_SIGMA,
    LEGO_GAP_WINDOW_RADIUS,
    REFERENCE_IDX,
)
from gs_experiment.scripts.render_reconstruction import (
    RAW_VARIANCE_VMAX,
    RAW_VARIANCE_VMIN,
    RESULTS_DIR,
    compute_uncertainty_maps,
    render_views,
)

DEPTH_RES = 64  # square, matching NeRF-Synthetic's square images


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
    noise_variance: float = LEGO_GAP_NOISE_VARIANCE,
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
            depth_width=depth_res, depth_height=depth_res, return_raw_variance=True,
            noise_variance=noise_variance, return_bq_mean=True,
        )
        # maps[0] is (spatial_map, dir_map, raw_map, bq_mean_map) -- see
        # render_reconstruction.compute_uncertainty_maps's return_bq_mean docstring and
        # render_scene_gallery.plot_gallery's docstring for why C_BQ/u_BQ (not C_alpha/u_BQ)
        # is the coherent pairing.
        _, _, raw_map, bq_mean_map = maps[0]
        err = np.abs(gt - recon).mean(axis=-1)
        # bq_mean_map is real RGB (see compute_uncertainty_maps's return_bq_mean
        # docstring) -- same mean-over-channels convention as `err` above, not a shared
        # column (see render_scene_gallery.plot_gallery's docstring for why these are
        # kept separate).
        err_bq = np.abs(gt - bq_mean_map).mean(axis=-1)
        psnr = -10.0 * np.log10(max(float(np.mean((gt - recon) ** 2)), 1e-10))
        print(
            f"gap {hw:.0f} deg: {len(train_frames)} train views, query-view PSNR {psnr:.2f}dB, "
            f"posterior variance mean={np.nanmean(raw_map):.4f} median={np.nanmedian(raw_map):.4f} "
            f"p95={np.nanpercentile(raw_map, 95):.4f} max={np.nanmax(raw_map):.4f}"
        )

        columns.append(dict(
            gap_deg=hw, n_train_views=len(train_frames), psnr=psnr,
            gt=gt, recon=recon, err=err, raw_map=raw_map, bq_mean_map=bq_mean_map, err_bq=err_bq,
        ))
    return columns


def plot_coverage_sweep(columns, out_path):
    """Rows: ground truth, alpha reconstruction (`C_alpha`), BQ mean
    (`C_BQ`, real RGB), `|error|` for each of those two against its own
    real target, raw posterior variance (`u_BQ`), and the trend line --
    see `render_scene_gallery.plot_gallery`'s docstring for why C_BQ/u_BQ
    (not C_alpha/u_BQ) is this project's coherent mean/variance pairing,
    and why the two error rows are kept separate rather than shared.

    A shared color scale across the pixel-map columns is still correct
    here (unlike render_scene_gallery.py's per-scene scale) -- these are
    the same checkpoint family at the same view, so absolute BQ-variance
    magnitude is directly comparable condition to condition. What changed:
    that shared scale is now fit to this figure's OWN actual data range,
    not the fixed `render_reconstruction.RAW_VARIANCE_VMIN/VMAX` constants
    -- those were calibrated under the old noiseless kernel fit and go
    stale the moment the fitting regime changes (confirmed directly on
    render_scene_gallery.py's figure: under this project's current
    noise-aware default fit, real raw-variance values can sit almost
    entirely below that fixed floor, silently clipping every column to
    the same bottom color). Printed range is also logged for the record.
    """
    n = len(columns)
    fig = plt.figure(figsize=(3 * n, 21.5))
    gs = fig.add_gridspec(7, n, height_ratios=[3, 3, 3, 3, 3, 3, 2])
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(n)] for r in range(6)])
    trend_ax = fig.add_subplot(gs[6, :])

    err_vmax = float(np.nanpercentile(
        np.concatenate([
            np.stack([c["err"] for c in columns]).ravel(),
            np.stack([c["err_bq"] for c in columns]).ravel(),
        ]),
        95,
    ))
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad(color=(0.4, 0.4, 0.4))

    raw_values = np.concatenate([c["raw_map"][np.isfinite(c["raw_map"])] for c in columns])
    raw_values = raw_values[raw_values > 0]
    if raw_values.size > 0:
        vmin = float(raw_values.min())
        vmax = max(float(raw_values.max()), vmin * 1.01)  # LogNorm needs vmin < vmax
    else:
        vmin, vmax = RAW_VARIANCE_VMIN, RAW_VARIANCE_VMAX
    print(f"posterior variance range this figure = [{vmin:.4g}, {vmax:.4g}]")

    for col, c in enumerate(columns):
        mean_var = float(np.nanmean(c["raw_map"]))
        axes[0, col].imshow(c["gt"])
        axes[0, col].set_title(f"gap ±{c['gap_deg']:.0f}°\n{c['n_train_views']} train views, {c['psnr']:.1f}dB")
        axes[1, col].imshow(c["recon"])
        # bq_mean_map is real RGB -- background/invalid pixels filled to white to match
        # the other panels' white background convention, not because they carry a real
        # BQ estimate. Clipped to [0,1] for display (see render_scene_gallery.
        # plot_gallery's own comment: a real, documented ~50% out-of-range rate, shown
        # clipped rather than left to matplotlib's own implicit clip-with-warning).
        bq_rgb = np.where(np.isfinite(c["bq_mean_map"]), c["bq_mean_map"], 1.0)
        axes[2, col].imshow(np.clip(bq_rgb, 0.0, 1.0))
        im_err = axes[3, col].imshow(c["err"], cmap="inferno", vmin=0, vmax=err_vmax)
        fig.colorbar(im_err, ax=axes[3, col], fraction=0.046, pad=0.04)
        im_err_bq = axes[4, col].imshow(c["err_bq"], cmap=cmap, vmin=0, vmax=err_vmax)
        fig.colorbar(im_err_bq, ax=axes[4, col], fraction=0.046, pad=0.04)
        im_raw = axes[5, col].imshow(c["raw_map"], cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
        fig.colorbar(im_raw, ax=axes[5, col], fraction=0.046, pad=0.04)
        axes[5, col].set_xlabel(f"mean variance={mean_var:.3f}", fontsize=9)
        for ax in axes[:, col]:
            ax.set_xticks([])
            ax.set_yticks([])

    axes[0, 0].set_ylabel("ground truth\n(held-out view)", fontsize=10)
    axes[1, 0].set_ylabel("alpha\nreconstruction", fontsize=10)
    axes[2, 0].set_ylabel("BQ mean", fontsize=10)
    axes[3, 0].set_ylabel("|error| (alpha)\n(mean over RGB)", fontsize=10)
    axes[4, 0].set_ylabel("|error|\n(BQ mean)", fontsize=10)
    axes[5, 0].set_ylabel("posterior variance\n(unnormalized)", fontsize=10)

    gap_degs = [c["gap_deg"] for c in columns]
    mean_vars = [float(np.nanmean(c["raw_map"])) for c in columns]
    median_vars = [float(np.nanmedian(c["raw_map"])) for c in columns]
    trend_ax.plot(gap_degs, mean_vars, "o-", color="tab:red", label="mean")
    trend_ax.plot(gap_degs, median_vars, "s--", color="tab:orange", label="median")
    trend_ax.set_xlabel("gap half-width (deg)")
    trend_ax.set_ylabel("posterior variance\n(averaged over the whole held-out view)")
    trend_ax.set_yscale("log")
    trend_ax.legend()
    trend_ax.grid(alpha=0.3)

    fig.tight_layout()
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
    parser.add_argument("--noise-variance", type=float, default=LEGO_GAP_NOISE_VARIANCE)
    args = parser.parse_args()

    run(
        args.prepared_dir, out_path=args.out, condition_prefix=args.condition_prefix,
        depth_res=args.depth_res, sigma=args.sigma, kappa=args.kappa, window_radius=args.window_radius,
        noise_variance=args.noise_variance,
    )


if __name__ == "__main__":
    main()
