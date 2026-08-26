"""The view-direction uncertainty *gradient* experiment
(`directional_gradient_experiment.py`), on a real NeRF-Synthetic
benchmark scene instead of a hand-built thin-rod cluster.

That earlier experiment could place cameras anywhere (a hand-built
Blender scene), so it built a designed, continuous coverage gradient by
varying a turntable arc's half-width directly. A real benchmark's camera
poses are fixed by the dataset -- there's no "render a new arc" option --
so this approximates the same idea by subsampling the real 100-view lego
training pool into 5 conditions of *equal view count* but increasingly
wide angular spread around a shared reference view
(`prepare_nerf_synthetic.select_gradient_subset`, holding count fixed the
same way the toy version held rod-cluster geometry fixed across zones, so
the only thing that varies between conditions is genuinely angular
spread, not view count).

Each condition trains its own real gsplat checkpoint (same lego geometry,
different real 15-view subset of the same 100 real training photos).
Directional and position-only BQ variance are then queried at the same
fixed point (the world origin -- NeRF-Synthetic's own object-centering
convention) and the same fixed, real query direction (a real camera
direction from the full 100-view pool, chosen once for a direction that's
maximally dissimilar to the reference view -- the same "opposite side"
construction differentiation_experiment.py's real-scene builder uses, not
hand-derived spherical trigonometry) across all 5 conditions, so any
difference in reported variance reflects the coverage-spread manipulation
and nothing else.

Needs torch + gsplat (requirements-gsplat.txt).

Run: .venv-gsplat/bin/python gs_experiment/real_directional_gradient_experiment.py gs_experiment/local_runs/lego_prepared
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from bq_splat.kernels import DirectionalKernel
from gs_experiment.nerf_transforms import load_transforms
from gs_experiment.pixel_uncertainty import LocalUncertaintyEngine, make_default_3d_position_kernel
from gs_experiment.prepare_nerf_synthetic import select_gradient_subset, write_condition
from gs_experiment.render_reconstruction import render_views
from gs_experiment.splat_scene import load_from_gsplat_checkpoint, splat_observations
from gs_experiment.train_minimal_gsplat import train

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_FRACTIONS = [0.06, 0.15, 0.35, 0.6, 1.0]
N_PER_ZONE = 6
REFERENCE_IDX = 0

TRAIN_KWARGS = dict(
    n_splats=1500, bounds=((-2.5, 2.5), (-2.5, 2.5), (-2.5, 2.5)), sh_degree=1, n_iters=2500, seed=0,
    init_scale=0.05, opacity_reg_weight=0.01, densify=True, densify_interval=300, densify_start=300,
    min_opacity=0.005, max_splats=8000, log_every=500,
)
SIGMA = 0.05
WINDOW_RADIUS = 0.5
KAPPA = 4.0


def build_conditions(
    prepared_dir: str,
    n_per_zone: int = N_PER_ZONE,
    window_fractions=WINDOW_FRACTIONS,
    condition_prefix: str = "gradient",
):
    wide_dir = os.path.join(prepared_dir, "wide")
    camera_angle_x, frames = load_transforms(os.path.join(wide_dir, "transforms.json"))

    centers = np.array([c2w[:3, 3] for _, c2w in frames])
    dirs = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    ref_dir = dirs[REFERENCE_IDX]
    query_direction = dirs[np.argmin(dirs @ ref_dir)]  # real, most-dissimilar-to-reference view direction

    zone_dirs = []
    spreads = []
    for i, window_fraction in enumerate(window_fractions):
        idx = select_gradient_subset(frames, n_per_zone=n_per_zone, window_fraction=window_fraction, reference_idx=REFERENCE_IDX)
        zone_dir = write_condition(prepared_dir, camera_angle_x, frames, idx, f"{condition_prefix}_{i}", "train")
        zone_dirs.append(zone_dir)
        # real measured angular spread of this zone's selected views (not
        # the arbitrary window_fraction knob): min cosine similarity to
        # the reference view among the views actually selected.
        min_sim = float((dirs[idx] @ ref_dir).min())
        spreads.append(np.degrees(np.arccos(np.clip(min_sim, -1.0, 1.0))))

    return zone_dirs, np.array(spreads), query_direction


def train_zones(zone_dirs, train_kwargs=TRAIN_KWARGS):
    ply_paths = []
    for zone_dir in zone_dirs:
        ply_path = os.path.join(zone_dir, "splats.ply")
        if not os.path.exists(ply_path):
            print(f"training {zone_dir}...")
            train(zone_dir, ply_path, **train_kwargs)
        else:
            print(f"reusing existing checkpoint at {ply_path}")
        ply_paths.append(ply_path)
    return ply_paths


def check_reconstruction_quality(zone_dirs, eval_dir: str):
    """Real held-out PSNR per condition, against the shared official-
    test-split eval set every lego_prepared/bonsai_prepared directory
    already has -- computed and reported *before* trusting any BQ number
    built on top of these checkpoints, closing exactly the gap that made
    the first real-scene attempt's result uninterpretable (FINDINGS.md
    section 37): a BQ variance computed on a checkpoint nobody checked
    could actually reconstruct the scene."""
    import shutil

    from gs_experiment.nerf_transforms import load_transforms as _load_transforms

    _, eval_frames = _load_transforms(os.path.join(eval_dir, "transforms.json"))
    n_eval = len(eval_frames)

    psnrs = []
    for zone_dir in zone_dirs:
        eval_copy_dir = zone_dir + "_eval"
        os.makedirs(eval_copy_dir, exist_ok=True)
        shutil.copy(os.path.join(eval_dir, "transforms.json"), os.path.join(eval_copy_dir, "transforms.json"))
        images_link = os.path.join(eval_copy_dir, "test")
        if not os.path.exists(images_link):
            os.symlink(os.path.abspath(os.path.join(eval_dir, "test")), images_link)
        shutil.copy(os.path.join(zone_dir, "splats.ply"), os.path.join(eval_copy_dir, "splats.ply"))

        results, _ = render_views(eval_copy_dir, list(range(n_eval)))
        psnr = float(np.mean([-10.0 * np.log10(max(float(np.mean((gt - recon) ** 2)), 1e-10)) for _, gt, recon in results]))
        psnrs.append(psnr)
        print(f"{zone_dir}: held-out PSNR over {n_eval} eval views = {psnr:.2f}dB")

    return np.array(psnrs)


def analyze(zone_dirs, spreads_deg, query_direction, psnrs=None):
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

    order = np.argsort(spreads_deg)
    if psnrs is not None:
        print(f"\n{'zone':>4}{'spread (deg)':>14}{'held-out PSNR':>15}{'directional var':>18}{'spatial-only var':>18}")
        for i in order:
            print(f"{i:>4}{spreads_deg[i]:>14.1f}{psnrs[i]:>15.2f}{directional_vars[i]:>18.5f}{spatial_vars[i]:>18.5f}")
    else:
        print(f"\n{'zone':>4}{'spread (deg)':>14}{'directional var':>18}{'spatial-only var':>18}")
        for i in order:
            print(f"{i:>4}{spreads_deg[i]:>14.1f}{directional_vars[i]:>18.5f}{spatial_vars[i]:>18.5f}")

    dir_sorted = directional_vars[order]
    is_monotonic = bool(np.all(np.diff(dir_sorted) <= 1e-12))
    rho = float(np.corrcoef(np.argsort(np.argsort(spreads_deg)), np.argsort(np.argsort(-directional_vars)))[0, 1])
    print(f"\ndirectional variance monotonically decreasing as spread widens: {is_monotonic}")
    print(f"rank correlation (spread vs. directional variance): rho={rho:.3f}")
    print(f"directional variance range (narrowest/widest): {dir_sorted[0] / max(dir_sorted[-1], 1e-12):.2f}x")
    print(f"spatial-only variance range (control): {spatial_vars.max() / max(spatial_vars.min(), 1e-12):.2f}x")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(spreads_deg[order], directional_vars[order], "o-", color="tab:red", label="directional variance")
    ax1.set_xlabel("real angular spread of selected views (deg, min similarity to reference)")
    ax1.set_ylabel("position+direction BQ variance", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.plot(spreads_deg[order], spatial_vars[order], "s--", color="tab:blue", label="spatial-only variance (control)")
    ax2.set_ylabel("position-only BQ variance", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    fig.suptitle("Real NeRF-Synthetic (lego): directional BQ variance vs. real view-coverage spread")
    fig.tight_layout()
    out_path = RESULTS_DIR / "real_directional_gradient.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_path}")


def run(
    prepared_dir: str,
    n_per_zone: int = N_PER_ZONE,
    window_fractions=WINDOW_FRACTIONS,
    condition_prefix: str = "gradient",
    train_kwargs=TRAIN_KWARGS,
    check_quality: bool = True,
):
    zone_dirs, spreads_deg, query_direction = build_conditions(
        prepared_dir, n_per_zone=n_per_zone, window_fractions=window_fractions, condition_prefix=condition_prefix,
    )
    train_zones(zone_dirs, train_kwargs=train_kwargs)
    psnrs = None
    if check_quality:
        eval_dir = os.path.join(prepared_dir, "eval")
        if os.path.exists(eval_dir):
            psnrs = check_reconstruction_quality(zone_dirs, eval_dir)
        else:
            print(f"no eval/ split found at {eval_dir}, skipping reconstruction-quality check")
    analyze(zone_dirs, spreads_deg, query_direction, psnrs=psnrs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prepared_dir")
    parser.add_argument("--n-per-zone", type=int, default=N_PER_ZONE)
    parser.add_argument("--window-fractions", type=float, nargs="+", default=WINDOW_FRACTIONS)
    parser.add_argument("--condition-prefix", default="gradient")
    parser.add_argument("--n-iters", type=int, default=TRAIN_KWARGS["n_iters"])
    parser.add_argument("--max-splats", type=int, default=TRAIN_KWARGS["max_splats"])
    parser.add_argument("--n-splats", type=int, default=TRAIN_KWARGS["n_splats"])
    args = parser.parse_args()
    train_kwargs = dict(TRAIN_KWARGS, n_iters=args.n_iters, max_splats=args.max_splats, n_splats=args.n_splats)
    run(
        args.prepared_dir, n_per_zone=args.n_per_zone, window_fractions=args.window_fractions,
        condition_prefix=args.condition_prefix, train_kwargs=train_kwargs,
    )


if __name__ == "__main__":
    main()
