"""The view-direction uncertainty gradient question on real geometry,
with the reconstruction-quality confound designed out from the start.

`real_directional_gradient_experiment.py` held total view *count* fixed
and varied the *width* of the window views were drawn from -- which
turned out to confound "spread" with "overall reconstruction
difficulty": held-out PSNR dropped as spread widened even after a 2.5x
training-budget increase (FINDINGS.md section 4's follow-up), because
thinning a fixed view count across a wider window also thins local view
*density* everywhere, not just near the region actually being tested.

This experiment instead starts from the full real 100-view lego pool and
removes a single deliberate angular *gap* of increasing half-width around
one reference view's direction (`prepare_nerf_synthetic.select_gap_subset`),
leaving every other view in the pool untouched. Density should stay high
everywhere except inside the gap itself, so overall reconstruction
quality shouldn't move much with gap size -- isolating the coverage-gap
manipulation instead of conflating it with global view thinning. Held-out
PSNR is reported both overall and restricted to eval views that actually
fall inside the gap, so a real local quality drop (expected, and not a
confound) is distinguishable from a global one (would be a confound).

Directional and position-only BQ variance are queried at the gap's own
center direction -- the specific viewing angle that's actually going
missing as the gap widens.

Needs torch + gsplat (requirements-gsplat.txt).

Run: .venv-gsplat/bin/python gs_experiment/gap_directional_experiment.py gs_experiment/local_runs/lego_prepared
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from bq_splat.kernels import DirectionalKernel
from gs_experiment.nerf_transforms import load_transforms
from gs_experiment.pixel_uncertainty import LocalUncertaintyEngine, make_default_3d_position_kernel
from gs_experiment.prepare_nerf_synthetic import select_gap_subset, write_condition
from gs_experiment.render_reconstruction import render_views
from gs_experiment.splat_scene import load_from_gsplat_checkpoint, splat_observations
from gs_experiment.train_minimal_gsplat import train

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GAP_HALF_WIDTHS_DEG = [0.0, 15.0, 30.0, 50.0, 75.0]
REFERENCE_IDX = 0

TRAIN_KWARGS = dict(
    n_splats=3000, bounds=((-2.5, 2.5), (-2.5, 2.5), (-2.5, 2.5)), sh_degree=1, n_iters=10000, seed=0,
    init_scale=0.05, opacity_reg_weight=0.01, densify=True, densify_interval=300, densify_start=300,
    min_opacity=0.005, max_splats=35000, log_every=1000,
)
SIGMA = 0.9
WINDOW_RADIUS = 1.6
KAPPA = 4.0
GATE_BACKGROUND_COLOR = (1.0, 1.0, 1.0)  # matches prepare_nerf_synthetic's own default compositing background


def build_conditions(prepared_dir: str, gap_half_widths=GAP_HALF_WIDTHS_DEG, condition_prefix: str = "gap"):
    wide_dir = os.path.join(prepared_dir, "wide")
    camera_angle_x, frames = load_transforms(os.path.join(wide_dir, "transforms.json"))

    centers = np.array([c2w[:3, 3] for _, c2w in frames])
    dirs = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    gap_center_direction = dirs[REFERENCE_IDX]

    zone_dirs = []
    n_views = []
    for i, hw in enumerate(gap_half_widths):
        idx = select_gap_subset(frames, gap_half_width_deg=hw, reference_idx=REFERENCE_IDX)
        zone_dir = write_condition(prepared_dir, camera_angle_x, frames, idx, f"{condition_prefix}_{i}", "train")
        zone_dirs.append(zone_dir)
        n_views.append(len(idx))

    return zone_dirs, np.array(gap_half_widths), np.array(n_views), gap_center_direction


def train_zones(zone_dirs, train_kwargs=TRAIN_KWARGS):
    for zone_dir in zone_dirs:
        ply_path = os.path.join(zone_dir, "splats.ply")
        if not os.path.exists(ply_path):
            print(f"training {zone_dir}...")
            train(zone_dir, ply_path, **train_kwargs)
        else:
            print(f"reusing existing checkpoint at {ply_path}")


def check_reconstruction_quality(zone_dirs, eval_dir: str, gap_half_widths, gap_center_direction, reference_idx=REFERENCE_IDX):
    """Overall held-out PSNR *and* PSNR restricted to eval views that fall
    inside each condition's own gap -- a real local quality drop inside
    the gap is expected and isn't a confound; a global drop would be."""
    _, eval_frames = load_transforms(os.path.join(eval_dir, "transforms.json"))
    n_eval = len(eval_frames)

    eval_centers = np.array([c2w[:3, 3] for _, c2w in eval_frames])
    eval_dirs = eval_centers / np.linalg.norm(eval_centers, axis=1, keepdims=True)
    eval_angular_dist_deg = np.degrees(np.arccos(np.clip(eval_dirs @ gap_center_direction, -1.0, 1.0)))

    overall_psnrs, gap_psnrs = [], []
    for zone_dir, hw in zip(zone_dirs, gap_half_widths):
        eval_copy_dir = zone_dir + "_eval"
        os.makedirs(eval_copy_dir, exist_ok=True)
        shutil.copy(os.path.join(eval_dir, "transforms.json"), os.path.join(eval_copy_dir, "transforms.json"))
        images_link = os.path.join(eval_copy_dir, "test")
        if not os.path.exists(images_link):
            os.symlink(os.path.abspath(os.path.join(eval_dir, "test")), images_link)
        shutil.copy(os.path.join(zone_dir, "splats.ply"), os.path.join(eval_copy_dir, "splats.ply"))

        results, _ = render_views(eval_copy_dir, list(range(n_eval)), background_color=GATE_BACKGROUND_COLOR)
        per_view_psnr = np.array(
            [-10.0 * np.log10(max(float(np.mean((gt - recon) ** 2)), 1e-10)) for _, gt, recon in results]
        )
        overall_psnrs.append(float(per_view_psnr.mean()))

        in_gap = eval_angular_dist_deg <= max(hw, 1e-6)
        gap_psnr = float(per_view_psnr[in_gap].mean()) if in_gap.any() else float("nan")
        gap_psnrs.append(gap_psnr)
        print(
            f"{zone_dir}: overall held-out PSNR = {overall_psnrs[-1]:.2f}dB over {n_eval} views; "
            f"in-gap PSNR = {gap_psnr:.2f}dB over {int(in_gap.sum())} views"
        )

    return np.array(overall_psnrs), np.array(gap_psnrs)


def analyze(zone_dirs, gap_half_widths, n_views, query_direction, overall_psnrs=None, gap_psnrs=None):
    directional_vars, spatial_vars = [], []
    query_point = np.zeros(3)

    for zone_dir in zone_dirs:
        scene = load_from_gsplat_checkpoint(zone_dir, attribution_angular_tol=0.01)
        positions, directions, values = splat_observations(scene)
        bounds = tuple((positions[:, d].min() - 1.0, positions[:, d].max() + 1.0) for d in range(3))

        pos_kernel = make_default_3d_position_kernel(sigma=SIGMA)
        dir_kernel = DirectionalKernel(kappa=KAPPA)
        engine = LocalUncertaintyEngine(
            positions=positions, values=values, pos_kernel=pos_kernel, scene_bounds=bounds,
            directions=directions, dir_kernel=dir_kernel,
        )

        dir_result = engine.directional_variance(query_point, query_direction, WINDOW_RADIUS)
        spatial_result = engine.spatial_only_variance(query_point, WINDOW_RADIUS)
        directional_vars.append(dir_result.variance)
        spatial_vars.append(spatial_result.variance)

    directional_vars = np.array(directional_vars)
    spatial_vars = np.array(spatial_vars)
    order = np.argsort(gap_half_widths)

    header = f"{'gap (deg)':>10}{'n_views':>9}"
    if overall_psnrs is not None:
        header += f"{'overall PSNR':>14}{'in-gap PSNR':>14}"
    header += f"{'directional var':>18}{'spatial-only var':>18}"
    print(f"\n{header}")
    for i in order:
        row = f"{gap_half_widths[i]:>10.1f}{n_views[i]:>9d}"
        if overall_psnrs is not None:
            row += f"{overall_psnrs[i]:>14.2f}{gap_psnrs[i]:>14.2f}"
        row += f"{directional_vars[i]:>18.5f}{spatial_vars[i]:>18.5f}"
        print(row)

    dir_sorted = directional_vars[order]
    is_monotonic = bool(np.all(np.diff(dir_sorted) >= -1e-12))
    rho = float(np.corrcoef(np.argsort(np.argsort(gap_half_widths)), np.argsort(np.argsort(directional_vars)))[0, 1])
    print(f"\ndirectional variance monotonically increasing with gap width: {is_monotonic}")
    print(f"rank correlation (gap width vs. directional variance): rho={rho:.3f}")
    print(f"directional variance range (widest/narrowest): {dir_sorted[-1] / max(dir_sorted[0], 1e-12):.2f}x")
    print(f"spatial-only variance range (control): {spatial_vars.max() / max(spatial_vars.min(), 1e-12):.2f}x")
    if overall_psnrs is not None:
        print(
            f"overall held-out PSNR range across conditions: {overall_psnrs.min():.2f}-{overall_psnrs.max():.2f}dB "
            "(should stay tight if the gap design avoids the earlier global-thinning confound)"
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(gap_half_widths[order], directional_vars[order], "o-", color="tab:red", label="directional variance")
    ax1.set_xlabel("gap half-width around query direction (deg)")
    ax1.set_ylabel("position+direction BQ variance", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.plot(gap_half_widths[order], spatial_vars[order], "s--", color="tab:blue", label="spatial-only variance (control)")
    ax2.set_ylabel("position-only BQ variance", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    fig.suptitle("Real NeRF-Synthetic (lego): directional BQ variance vs. a deliberate real coverage gap")
    fig.tight_layout()
    out_path = RESULTS_DIR / "gap_directional_gradient.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_path}")


def run(
    prepared_dir: str,
    gap_half_widths=GAP_HALF_WIDTHS_DEG,
    condition_prefix: str = "gap",
    train_kwargs=TRAIN_KWARGS,
    check_quality: bool = True,
):
    zone_dirs, gap_half_widths, n_views, query_direction = build_conditions(
        prepared_dir, gap_half_widths=gap_half_widths, condition_prefix=condition_prefix,
    )
    train_zones(zone_dirs, train_kwargs=train_kwargs)
    overall_psnrs = gap_psnrs = None
    if check_quality:
        eval_dir = os.path.join(prepared_dir, "eval")
        if os.path.exists(eval_dir):
            overall_psnrs, gap_psnrs = check_reconstruction_quality(
                zone_dirs, eval_dir, gap_half_widths, query_direction
            )
        else:
            print(f"no eval/ split found at {eval_dir}, skipping reconstruction-quality check")
    analyze(zone_dirs, gap_half_widths, n_views, query_direction, overall_psnrs=overall_psnrs, gap_psnrs=gap_psnrs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prepared_dir")
    parser.add_argument("--gap-half-widths", type=float, nargs="+", default=GAP_HALF_WIDTHS_DEG)
    parser.add_argument("--condition-prefix", default="gap")
    parser.add_argument("--n-iters", type=int, default=TRAIN_KWARGS["n_iters"])
    parser.add_argument("--max-splats", type=int, default=TRAIN_KWARGS["max_splats"])
    parser.add_argument("--n-splats", type=int, default=TRAIN_KWARGS["n_splats"])
    args = parser.parse_args()
    train_kwargs = dict(TRAIN_KWARGS, n_iters=args.n_iters, max_splats=args.max_splats, n_splats=args.n_splats)
    run(
        args.prepared_dir, gap_half_widths=args.gap_half_widths,
        condition_prefix=args.condition_prefix, train_kwargs=train_kwargs,
    )


if __name__ == "__main__":
    main()
