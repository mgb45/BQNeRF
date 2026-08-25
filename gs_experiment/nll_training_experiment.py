"""ROADMAP.md item 3: does training under the likelihood -- an
uncertainty-weighted Gaussian-NLL auxiliary loss and/or BQ-variance-driven
densification, both added to train_minimal_gsplat.py -- actually beat
standard photometric-loss + gradient-densification at a matched splat
budget? The comparison this project's own honesty norm requires before
either mechanism gets used anywhere else: neither is assumed to help just
because it's more "principled."

Four variants, same scene, seed, and every other hyperparameter (matching
nbv_experiment.py's exact training call, the established convention for
this scene family), only densify_criterion and nll_weight differing:
  - baseline:       gradient densification,   no NLL term (today's default)
  - bq_densify:      bq_variance densification, no NLL term
  - nll_loss:        gradient densification,   NLL term on
  - bq_densify+nll:  bq_variance densification, NLL term on

Trained on gs_experiment/local_runs/nbv_out/baseline (a real, already-used
narrow 10-view training arc), evaluated on both the training views
themselves and the genuinely disjoint held-out ring in
nbv_out/baseline_eval -- generalization, not just training-view fit, is
the claim that matters.

Needs torch + gsplat (requirements-gsplat.txt).

Run: .venv-gsplat/bin/python gs_experiment/nll_training_experiment.py gs_experiment/local_runs/nbv_out
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from gs_experiment.nerf_transforms import load_transforms
from gs_experiment.render_reconstruction import render_views
from gs_experiment.train_minimal_gsplat import train

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

COMMON_KWARGS = dict(
    n_splats=1500, bounds=((-2.5, 2.5), (-2.5, 2.5), (-2.5, 2.5)), sh_degree=1,
    init_scale=0.1, opacity_reg_weight=0.003, densify=True, densify_interval=300, densify_start=300,
    densify_grad_percentile=80.0, min_opacity=0.005, max_splats=6000, log_every=1000,
    bq_sigma=0.9, bq_window_radius=1.6,
)

VARIANTS = {
    "baseline": dict(densify_criterion="gradient", nll_weight=0.0),
    "bq_densify": dict(densify_criterion="bq_variance", nll_weight=0.0),
    "nll_loss": dict(densify_criterion="gradient", nll_weight=0.02),
    "bq_densify+nll": dict(densify_criterion="bq_variance", nll_weight=0.02),
}


def mean_psnr(scene_dir: str, n_views: int) -> float:
    results, _ = render_views(scene_dir, list(range(n_views)))
    psnrs = [-10.0 * np.log10(max(float(np.mean((gt - recon) ** 2)), 1e-10)) for _, gt, recon in results]
    return float(np.mean(psnrs))


def run(nbv_dir: str, n_iters: int = 3000, seed: int = 0, nll_interval: int = 50, nll_grid_res: int = 12):
    train_dir = os.path.join(nbv_dir, "baseline")
    eval_source_dir = os.path.join(nbv_dir, "baseline_eval")
    out_root = os.path.join(nbv_dir, "nll_experiment")
    os.makedirs(out_root, exist_ok=True)

    # frame count comes from transforms.json, not the images/ directory --
    # images/ is a shared symlinked pool of renders, transforms.json is
    # what actually subsets it into this scene's train/eval views.
    _, train_frames = load_transforms(os.path.join(train_dir, "transforms.json"))
    _, eval_frames = load_transforms(os.path.join(eval_source_dir, "transforms.json"))
    n_train_views = len(train_frames)
    n_eval_views = len(eval_frames)

    results = {}
    for name, overrides in VARIANTS.items():
        print(f"\n=== training variant: {name} ({overrides}) ===")
        variant_dir = os.path.join(out_root, name)
        os.makedirs(variant_dir, exist_ok=True)
        ply_path = os.path.join(variant_dir, "splats.ply")
        train(
            train_dir, ply_path, n_iters=n_iters, seed=seed,
            nll_interval=nll_interval, nll_grid_res=nll_grid_res,
            **COMMON_KWARGS, **overrides,
        )

        # train-view PSNR: evaluate directly against the training scene_dir
        # with this variant's checkpoint swapped in.
        eval_train_dir = os.path.join(variant_dir, "eval_on_train")
        os.makedirs(eval_train_dir, exist_ok=True)
        for fname in ("transforms.json",):
            shutil.copy(os.path.join(train_dir, fname), os.path.join(eval_train_dir, fname))
        images_link = os.path.join(eval_train_dir, "images")
        if not os.path.exists(images_link):
            os.symlink(os.path.abspath(os.path.join(train_dir, "images")), images_link)
        shutil.copy(ply_path, os.path.join(eval_train_dir, "splats.ply"))
        train_psnr = mean_psnr(eval_train_dir, n_train_views)

        # held-out PSNR: same pattern against the disjoint eval ring.
        eval_heldout_dir = os.path.join(variant_dir, "eval_on_heldout")
        os.makedirs(eval_heldout_dir, exist_ok=True)
        for fname in ("transforms.json",):
            shutil.copy(os.path.join(eval_source_dir, fname), os.path.join(eval_heldout_dir, fname))
        images_link = os.path.join(eval_heldout_dir, "images")
        if not os.path.exists(images_link):
            os.symlink(os.path.abspath(os.path.join(eval_source_dir, "images")), images_link)
        shutil.copy(ply_path, os.path.join(eval_heldout_dir, "splats.ply"))
        heldout_psnr = mean_psnr(eval_heldout_dir, n_eval_views)

        import gs_experiment.ply_io as ply_io
        n_splats_final = len(ply_io.read_3dgs_ply(ply_path)["positions"])

        results[name] = dict(train_psnr=train_psnr, heldout_psnr=heldout_psnr, n_splats=n_splats_final)
        print(f"{name}: n_splats={n_splats_final}  train PSNR={train_psnr:.2f}dB  held-out PSNR={heldout_psnr:.2f}dB")

    print("\n=== summary ===")
    print(f"{'variant':<18}{'n_splats':>10}{'train PSNR':>14}{'held-out PSNR':>16}")
    for name, r in results.items():
        print(f"{name:<18}{r['n_splats']:>10}{r['train_psnr']:>14.2f}{r['heldout_psnr']:>16.2f}")

    base = results["baseline"]
    print("\ndeltas vs. baseline (train arc, gradient densify, no NLL):")
    for name, r in results.items():
        if name == "baseline":
            continue
        print(
            f"  {name:<18} train {r['train_psnr']-base['train_psnr']:+.2f}dB   "
            f"held-out {r['heldout_psnr']-base['heldout_psnr']:+.2f}dB   "
            f"n_splats {r['n_splats']-base['n_splats']:+d}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("nbv_dir")
    parser.add_argument("--n-iters", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.nbv_dir, n_iters=args.n_iters, seed=args.seed)


if __name__ == "__main__":
    main()
