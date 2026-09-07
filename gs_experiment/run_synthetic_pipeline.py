"""End-to-end pipeline for a NeRF-Synthetic scene family, per ROADMAP.md's
"render it and look, before reaching for statistics": train each named
scene, render its genuinely held-out test-split views (not training
views) plus real per-pixel BQ uncertainty on those same views, and plot
ground truth / reconstruction / error / uncertainty side by side for a
few examples per scene -- one glance to check both reconstruction
quality and whether BQ's uncertainty behaves sensibly, across scenes,
without reaching for correlation numbers first.

Thin orchestration only: download/prepare/train reuses
evaluate_checkpoint.py's multi-scene machinery unchanged, and
render/plot reuses render_reconstruction.py's per-view functions
(generalized with a `checkpoint_dir` split from the views being
rendered, exactly so a held-out split's images can be paired with a
different split's checkpoint). See both files' docstrings.

Needs torch + gsplat (requirements-gsplat.txt).

Run: .venv-gsplat/bin/python gs_experiment/run_synthetic_pipeline.py --scenes chair,drums,lego
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from gs_experiment.evaluate_checkpoint import prepare_and_train
from gs_experiment.nerf_transforms import load_transforms
from gs_experiment.render_directional_uncertainty_sweep import check_quality_gate
from gs_experiment.render_reconstruction import RESULTS_DIR, compute_uncertainty_maps, plot_comparisons, render_views

# NeRF-Synthetic scenes are trained against a white background
# (prepare_nerf_synthetic.py's default); sigma/window_radius match the
# scale evaluate_checkpoint.py's multi-scene TRAIN_KWARGS trains at
# (bounds +/-2.5, 2k-15k splats) -- render_reconstruction.py's own
# defaults were tuned for a different, larger-scale checkpoint.
BACKGROUND_COLOR = (1.0, 1.0, 1.0)
SIGMA = 0.05
WINDOW_RADIUS = 0.08
DEPTH_RES = 64  # square, matching NeRF-Synthetic's square images (unlike render_reconstruction's widescreen-tuned default)


def pick_example_indices(n_views: int, n_examples: int) -> list:
    return sorted(set(np.linspace(0, n_views - 1, min(n_examples, n_views)).round().astype(int).tolist()))


def run_scene(
    scene: str,
    n_examples: int,
    skip_download: bool,
    skip_prepare: bool,
    skip_train: bool,
    sigma: float,
    window_radius: float,
    depth_res: int,
    no_uncertainty: bool,
    min_psnr: float,
    force: bool,
):
    ply_path = prepare_and_train(scene, skip_download=skip_download, skip_prepare=skip_prepare, skip_train=skip_train)
    wide_dir = os.path.dirname(ply_path)
    eval_dir = os.path.join(os.path.dirname(wide_dir), "eval")

    print(f"[{scene}] checking reconstruction quality against the full held-out split before trusting anything else...")
    held_out_psnr = check_quality_gate(wide_dir, eval_dir, min_psnr, force, background_color=BACKGROUND_COLOR)

    camera_angle_x, frames = load_transforms(os.path.join(eval_dir, "transforms.json"))
    view_indices = pick_example_indices(len(frames), n_examples)

    results, checkpoint = render_views(eval_dir, view_indices, checkpoint_dir=wide_dir, background_color=BACKGROUND_COLOR)
    print(f"[{scene}] held-out PSNR (full split): {held_out_psnr:.2f}dB  ({checkpoint['positions'].shape[0]} splats)")

    uncertainty_maps = None
    if not no_uncertainty:
        height, width = results[0][1].shape[:2]
        uncertainty_maps = compute_uncertainty_maps(
            eval_dir, view_indices, frames, camera_angle_x, width, height, checkpoint,
            checkpoint_dir=wide_dir, sigma=sigma, window_radius=window_radius,
            depth_width=depth_res, depth_height=depth_res,
        )

    out_path = RESULTS_DIR / f"pipeline_{scene}.png"
    plot_comparisons(results, out_path, title=f"{scene}: held-out test views vs. reconstruction", uncertainty_maps=uncertainty_maps)

    return dict(scene=scene, held_out_psnr=held_out_psnr, n_splats=int(checkpoint["positions"].shape[0]), out_path=str(out_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", required=True, help="comma-separated NeRF-Synthetic scene names, e.g. chair,drums,lego")
    parser.add_argument("--n-examples", type=int, default=3, help="held-out example views to plot per scene")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-train", action="store_true", help="reuse an existing wide/splats.ply if present")
    parser.add_argument("--sigma", type=float, default=SIGMA)
    parser.add_argument("--window-radius", type=float, default=WINDOW_RADIUS)
    parser.add_argument("--depth-res", type=int, default=DEPTH_RES, help="square uncertainty-map resolution before upsampling")
    parser.add_argument("--no-uncertainty", action="store_true", help="skip the two BQ uncertainty columns (faster)")
    parser.add_argument("--min-psnr", type=float, default=20.0, help="held-out PSNR quality gate (ROADMAP.md's bar); see check_quality_gate")
    parser.add_argument("--force", action="store_true", help="plot anyway even if a scene fails the quality gate")
    args = parser.parse_args()

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    summary = [
        run_scene(
            scene, args.n_examples, args.skip_download, args.skip_prepare, args.skip_train,
            args.sigma, args.window_radius, args.depth_res, args.no_uncertainty, args.min_psnr, args.force,
        )
        for scene in scenes
    ]

    print(f"\n{'scene':<12}{'held-out PSNR':>15}{'n_splats':>10}  figure")
    for r in summary:
        print(f"{r['scene']:<12}{r['held_out_psnr']:>13.2f}dB{r['n_splats']:>10}  {r['out_path']}")


if __name__ == "__main__":
    main()
