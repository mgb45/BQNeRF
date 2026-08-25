"""ROADMAP.md milestone 4: active-view / next-best-view (NBV) combination
experiment. "Use BQ variance alongside a visibility proxy for
candidate-view scoring; check whether the combined signal selects views
that improve reconstruction in under-resolved regions faster than either
signal alone."

Uses scene_spec.nbv_test_scene: a single thin-rod cluster observed from a
narrow training arc, a discrete pool of candidate next-view poses, and a
disjoint held-out evaluation ring (never a candidate, never trained on
until it's used for evaluation).

Pipeline:
  1. Train a baseline checkpoint on the training arc alone.
  2. Score every candidate view two ways, using the baseline checkpoint's
     real splat positions/observed directions (no retraining needed for
     scoring itself -- this is BQ's actual practical advantage, "closed-
     form, essentially free to compute", exercised for real here):
       (a) BQ: position+direction variance at the cluster center, queried
           at the candidate's viewing direction -- high variance means
           that direction is under-covered by the training arc.
       (b) visibility: how much adding the candidate's direction would
           reduce the mean resultant length of the already-observed
           direction set (bigger reduction = more angular-diversity
           gain) -- a genuinely different, non-BQ mechanism, computed
           with visibility_baseline.resultant_length.
       (c) combined: normalized sum of (a) and (b).
  3. Retrain two more checkpoints -- training arc + the top-combined
     candidate, training arc + the worst-combined (most redundant)
     candidate -- and evaluate all three (baseline, +best, +worst) on the
     held-out eval ring via PSNR, to check whether the BQ+visibility
     combination actually picks a view that helps more than a poor one.

Needs torch + gsplat (requirements-gsplat.txt).

Run: .venv-gsplat/bin/python gs_experiment/nbv_experiment.py <scene_dir> <info_npz>
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
from gs_experiment.camera import directions_from_positions_to_camera, turntable_camera
from gs_experiment.nerf_transforms import load_transforms, write_transforms_json
from gs_experiment.pixel_uncertainty import LocalUncertaintyEngine, make_default_3d_position_kernel
from gs_experiment.render_reconstruction import render_views
from gs_experiment.splat_scene import load_from_gsplat_checkpoint, splat_observations
from gs_experiment.train_minimal_gsplat import train
from gs_experiment.visibility_baseline import resultant_length

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def make_subset_scene_dir(source_dir: str, frame_indices, out_dir: str):
    camera_angle_x, frames = load_transforms(os.path.join(source_dir, "transforms.json"))
    subset = [{"file_path": frames[i][0], "transform_matrix": frames[i][1]} for i in frame_indices]
    os.makedirs(out_dir, exist_ok=True)
    write_transforms_json(os.path.join(out_dir, "transforms.json"), camera_angle_x, subset)
    images_link = os.path.join(out_dir, "images")
    if not os.path.exists(images_link):
        os.symlink(os.path.abspath(os.path.join(source_dir, "images")), images_link)


def score_candidates(baseline_dir: str, radius: float, window_radius: float = 1.6, angular_tol: float = 0.01):
    scene = load_from_gsplat_checkpoint(baseline_dir, attribution_angular_tol=angular_tol)
    positions, directions, values = splat_observations(scene)

    pos_margin = 1.0
    bounds = tuple((positions[:, d].min() - pos_margin, positions[:, d].max() + pos_margin) for d in range(3))
    pos_kernel = make_default_3d_position_kernel(sigma=0.9)
    dir_kernel = DirectionalKernel(kappa=4.0)
    engine = LocalUncertaintyEngine(
        positions=positions, values=values, pos_kernel=pos_kernel, scene_bounds=bounds,
        directions=directions, dir_kernel=dir_kernel,
    )

    center = np.zeros(3)
    existing_idx = engine.local_neighbors(center, window_radius)
    existing_dirs = directions[existing_idx]
    base_resultant = resultant_length(existing_dirs)
    print(f"baseline: {len(existing_idx)} local (splat,camera) observations, resultant_length={base_resultant:.4f}")

    return engine, center, existing_dirs, base_resultant


def run(nbv_dir: str, info_path: str, radius: float = 6.5, n_iters: int = 3000, seed: int = 0):
    info = dict(np.load(info_path))
    train_idx = info["train_idx"].tolist()
    candidate_idx = info["candidate_idx"].tolist()
    eval_idx = info["eval_idx"].tolist()
    candidate_thetas = info["candidate_thetas"]

    baseline_dir = os.path.join(nbv_dir, "baseline")
    make_subset_scene_dir(nbv_dir, train_idx, baseline_dir)

    print("=== training baseline (train arc alone) ===")
    train(
        baseline_dir, os.path.join(baseline_dir, "splats.ply"),
        n_splats=1500, bounds=((-2.5, 2.5), (-2.5, 2.5), (-2.5, 2.5)), sh_degree=1, n_iters=n_iters, seed=seed,
        init_scale=0.1, opacity_reg_weight=0.003, densify=True, densify_interval=300, densify_start=300,
        densify_grad_percentile=80.0, min_opacity=0.005, max_splats=6000, log_every=1000,
    )

    engine, center, existing_dirs, base_resultant = score_candidates(baseline_dir, radius=radius)

    bq_scores, vis_scores = [], []
    for theta in candidate_thetas:
        cam = turntable_camera(radius, 35.0, float(theta))
        cand_dir = directions_from_positions_to_camera(center.reshape(1, -1), cam)[0]
        bq_scores.append(engine.directional_variance(center, cand_dir, 1.6).variance)
        new_resultant = resultant_length(np.vstack([existing_dirs, cand_dir]))
        vis_scores.append(base_resultant - new_resultant)  # positive = diversity gain

    bq_scores = np.array(bq_scores)
    vis_scores = np.array(vis_scores)
    bq_norm = (bq_scores - bq_scores.min()) / (np.ptp(bq_scores) + 1e-12)
    vis_norm = (vis_scores - vis_scores.min()) / (np.ptp(vis_scores) + 1e-12)
    combined = bq_norm + vis_norm

    print("\ntheta   bq_var   vis_gain   combined")
    for t, b, v, c in sorted(zip(candidate_thetas, bq_scores, vis_scores, combined), key=lambda r: -r[3]):
        print(f"{t:6.1f}  {b:7.3f}  {v:8.4f}  {c:7.3f}")

    rank_corr = np.corrcoef(np.argsort(np.argsort(bq_scores)), np.argsort(np.argsort(vis_scores)))[0, 1]
    print(f"\nBQ vs visibility candidate-ranking correlation: {rank_corr:.3f}")

    best_local = int(np.argmax(combined))
    worst_local = int(np.argmin(combined))
    best_global = candidate_idx[best_local]
    worst_global = candidate_idx[worst_local]
    print(
        f"best candidate: theta={candidate_thetas[best_local]:.1f} (combined={combined[best_local]:.3f})  "
        f"worst candidate: theta={candidate_thetas[worst_local]:.1f} (combined={combined[worst_local]:.3f})"
    )

    results = {}
    for label, extra_idx in [("baseline", []), ("plus_best", [best_global]), ("plus_worst", [worst_global])]:
        scene_dir = os.path.join(nbv_dir, label)
        if extra_idx:
            make_subset_scene_dir(nbv_dir, train_idx + extra_idx, scene_dir)
            print(f"\n=== training {label} ===")
            train(
                scene_dir, os.path.join(scene_dir, "splats.ply"),
                n_splats=1500, bounds=((-2.5, 2.5), (-2.5, 2.5), (-2.5, 2.5)), sh_degree=1, n_iters=n_iters, seed=seed,
                init_scale=0.1, opacity_reg_weight=0.003, densify=True, densify_interval=300, densify_start=300,
                densify_grad_percentile=80.0, min_opacity=0.005, max_splats=6000, log_every=1000,
            )
        else:
            scene_dir = baseline_dir  # already trained above

        eval_scene_dir = os.path.join(nbv_dir, f"{label}_eval")
        make_subset_scene_dir(nbv_dir, eval_idx, eval_scene_dir)
        shutil.copy(os.path.join(scene_dir, "splats.ply"), os.path.join(eval_scene_dir, "splats.ply"))

        eval_results, _ = render_views(eval_scene_dir, list(range(len(eval_idx))))
        psnrs = [-10.0 * np.log10(max(float(np.mean((gt - recon) ** 2)), 1e-10)) for _, gt, recon in eval_results]
        results[label] = float(np.mean(psnrs))
        print(f"{label}: mean held-out PSNR over {len(eval_idx)} eval views = {results[label]:.2f}dB")

    print("\n=== summary ===")
    print(f"baseline (train arc alone):        {results['baseline']:.2f}dB")
    print(f"+ best (BQ+visibility combined):   {results['plus_best']:.2f}dB  (delta {results['plus_best']-results['baseline']:+.2f}dB)")
    print(f"+ worst (most redundant candidate): {results['plus_worst']:.2f}dB  (delta {results['plus_worst']-results['baseline']:+.2f}dB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("nbv_dir")
    parser.add_argument("info_path")
    parser.add_argument("--radius", type=float, default=6.5)
    parser.add_argument("--n-iters", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.nbv_dir, args.info_path, radius=args.radius, n_iters=args.n_iters, seed=args.seed)


if __name__ == "__main__":
    main()
