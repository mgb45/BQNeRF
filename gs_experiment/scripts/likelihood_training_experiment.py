"""ROADMAP.md item 1: "Train directly under the likelihood."

Every result so far (the coverage/floater/splat-budget headline figures)
computes BQ variance *after* training, read off a checkpoint trained by
ordinary photometric loss and gradient-triggered densification. This
script tests whether the BQ posterior helps *during* training, not just
diagnoses a finished checkpoint, by training matched variants of
`train_minimal_gsplat.train()` on the same real scene -- same seed, same
every other hyperparameter -- differing only in the two knobs `train()`
already exposes for this:

- `densify_criterion`: `"gradient"` (today's default) vs. `"bq_variance"`
  (closed-form BQ position-only variance drives clone/split instead of
  gsplat's view-space positional gradient).
- `nll_weight`: 0 (off) vs. an uncertainty-weighted Gaussian-NLL auxiliary
  loss term on real ray-surface points.

Trained on `gs_experiment/local_runs/lego_prepared/narrow` (a real,
already-prepared 12-view training pool) and evaluated on both those
training views and a genuinely disjoint subsample of
`gs_experiment/local_runs/lego_prepared/eval` (the official NeRF-Synthetic
lego test split) -- generalization, not just training-view fit, is the
claim that matters. This is a fresh, self-contained script: it does not
depend on the retired `run_nll_experiment`/`NLL_EXPERIMENT_VARIANTS` data
layout (`<nbv_dir>/baseline` + `baseline_eval`), which no longer exists in
this repo -- it reads `lego_prepared`'s actual on-disk layout instead
(`transforms.json` file_paths are already scene_dir-relative, e.g.
`./train/r_0`, so no `images/` symlink juggling is needed; a small
`train`/`test` symlink into the shared image pool, matching how
`narrow/` and `eval/` already do it, is enough).

Needs torch + gsplat (requirements-gsplat.txt). Run:
    .venv-gsplat/bin/python gs_experiment/scripts/likelihood_training_experiment.py \\
        gs_experiment/local_runs/lego_prepared/narrow \\
        gs_experiment/local_runs/lego_prepared/eval \\
        gs_experiment/local_runs/likelihood_experiment
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gs_experiment.ply_io as ply_io
from gs_experiment.nerf_transforms import load_transforms
from gs_experiment.scripts.train_minimal_gsplat import mean_psnr, train

# Real lego-scale BQ hyperparameters (marginal-likelihood-fitted, see
# real_directional_coverage_experiment.py's LEGO_GAP_SIGMA/LEGO_GAP_WINDOW_RADIUS
# comment for the fitting story, including the 2025-09 refit after a real
# colors bug -- gs_experiment/results/FINDINGS.md section 4) -- NOT train()'s
# own bq_sigma=0.9/bq_window_radius=1.6 defaults, which train()'s docstring
# explicitly warns match a different (thin-rod/cylinder) scene family's
# scale, not lego's.
LEGO_BQ_SIGMA = 0.13926
LEGO_BQ_WINDOW_RADIUS = 0.08

# Common recipe: matches real_directional_coverage_experiment.py's established
# lego convention (white background, position-LR decay, SSIM loss term) scaled
# down in n_splats/max_splats/n_iters to actually finish in one session on a
# 12-view scene, per this experiment's own scope (a real, honest, fast
# comparison -- not the reference 3DGS recipe at full iteration count).
COMMON_KWARGS = dict(
    n_splats=2000,
    bounds=((-2.5, 2.5), (-2.5, 2.5), (-2.5, 2.5)),
    sh_degree=3,
    seed=0,
    init_scale=0.05,
    opacity_reg_weight=0.01,
    densify=True,
    densify_interval=200,
    densify_start=300,
    min_opacity=0.005,
    max_splats=8000,
    log_every=500,
    bq_sigma=LEGO_BQ_SIGMA,
    bq_window_radius=LEGO_BQ_WINDOW_RADIUS,
    background_color=(1.0, 1.0, 1.0),
    position_lr_final=2e-5,
    ssim_weight=0.2,
)

# The four variants ROADMAP.md item 1 asks for, plus a fifth exploring the
# open bq_densify_min_opacity design question (does flooring the BQ-variance
# densify score by opacity change the outcome vs. leaving it unset?).
VARIANTS = {
    "baseline": dict(densify_criterion="gradient", nll_weight=0.0),
    "bq_densify": dict(densify_criterion="bq_variance", nll_weight=0.0),
    "nll_loss": dict(densify_criterion="gradient", nll_weight=0.02),
    "bq_densify+nll": dict(densify_criterion="bq_variance", nll_weight=0.02),
    "bq_densify_floor": dict(densify_criterion="bq_variance", nll_weight=0.0, bq_densify_min_opacity=0.05),
}


def _make_eval_scene_dir(out_dir: str, source_dir: str, image_subdir: str, ply_path: str, n_views=None):
    """Builds a self-contained scene_dir `mean_psnr`/`render_views` can be
    pointed at directly: a (possibly frame-subsampled) copy of
    `source_dir`'s transforms.json, a symlink to its real image directory
    (matching how lego_prepared/narrow and /eval already symlink `train`/
    `test` into the shared image pool, rather than copying image data), and
    this variant's checkpoint copied in as splats.ply (mean_psnr/render_views
    read splats.ply from the scene_dir itself, not a separate checkpoint_dir
    argument). Returns (scene_dir, n_frames_actually_used)."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(source_dir, "transforms.json")) as f:
        data = json.load(f)
    if n_views is not None:
        data["frames"] = data["frames"][:n_views]
    with open(os.path.join(out_dir, "transforms.json"), "w") as f:
        json.dump(data, f)

    link_path = os.path.join(out_dir, image_subdir)
    if not os.path.exists(link_path):
        os.symlink(os.path.abspath(os.path.join(source_dir, image_subdir)), link_path)

    shutil.copy(ply_path, os.path.join(out_dir, "splats.ply"))
    return out_dir, len(data["frames"])


def run_experiment(
    train_dir: str,
    eval_dir: str,
    out_root: str,
    n_iters: int = 3000,
    n_eval_views: int = 15,
    variants=None,
    common_kwargs=None,
):
    """Trains every variant in `variants` (default `VARIANTS`) on
    `train_dir`, evaluates each on its own training views and a
    `n_eval_views`-frame subsample of the disjoint `eval_dir`, and prints a
    comparison table + deltas vs. baseline. Returns the results dict."""
    variants = VARIANTS if variants is None else variants
    common_kwargs = COMMON_KWARGS if common_kwargs is None else common_kwargs
    os.makedirs(out_root, exist_ok=True)

    _, train_frames = load_transforms(os.path.join(train_dir, "transforms.json"))
    n_train_views = len(train_frames)

    results = {}
    for name, overrides in variants.items():
        print(f"\n=== training variant: {name} ({overrides}) ===")
        variant_dir = os.path.join(out_root, name)
        os.makedirs(variant_dir, exist_ok=True)
        ply_path = os.path.join(variant_dir, "splats.ply")

        train(train_dir, ply_path, n_iters=n_iters, **common_kwargs, **overrides)

        # train-view PSNR: this variant's checkpoint against its own training scene.
        eval_train_dir, n_tv = _make_eval_scene_dir(
            os.path.join(variant_dir, "eval_on_train"), train_dir, "train", ply_path,
        )
        # background_color must match common_kwargs["background_color"] (white,
        # matching lego_prepared's white-composited images) -- mean_psnr's own
        # default is a dark background matching train()'s default for a
        # different scene family; passing the wrong one silently reports PSNR
        # tens of dB too low without the reconstruction itself being wrong
        # (caught exactly this way while building this script -- see mean_psnr's
        # docstring).
        bg = common_kwargs.get("background_color", (0.05, 0.05, 0.05))
        train_psnr = mean_psnr(eval_train_dir, n_tv, background_color=bg)

        # held-out PSNR: same checkpoint against the genuinely disjoint eval split.
        eval_heldout_dir, n_hv = _make_eval_scene_dir(
            os.path.join(variant_dir, "eval_on_heldout"), eval_dir, "test", ply_path, n_views=n_eval_views,
        )
        heldout_psnr = mean_psnr(eval_heldout_dir, n_hv, background_color=bg)

        n_splats_final = len(ply_io.read_3dgs_ply(ply_path)["positions"])
        results[name] = dict(train_psnr=train_psnr, heldout_psnr=heldout_psnr, n_splats=n_splats_final)
        print(f"{name}: n_splats={n_splats_final}  train PSNR={train_psnr:.2f}dB  held-out PSNR={heldout_psnr:.2f}dB")

    print(f"\n=== summary (train_dir={train_dir}, n_iters={n_iters}, n_train_views={n_train_views}, n_eval_views={n_eval_views}) ===")
    print(f"{'variant':<20}{'n_splats':>10}{'train PSNR':>14}{'held-out PSNR':>16}")
    for name, r in results.items():
        print(f"{name:<20}{r['n_splats']:>10}{r['train_psnr']:>14.2f}{r['heldout_psnr']:>16.2f}")

    if "baseline" in results:
        base = results["baseline"]
        print("\ndeltas vs. baseline (gradient densify, no NLL):")
        for name, r in results.items():
            if name == "baseline":
                continue
            print(
                f"  {name:<20} train {r['train_psnr']-base['train_psnr']:+.2f}dB   "
                f"held-out {r['heldout_psnr']-base['heldout_psnr']:+.2f}dB   "
                f"n_splats {r['n_splats']-base['n_splats']:+d}"
            )

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("train_dir", help="e.g. gs_experiment/local_runs/lego_prepared/narrow")
    parser.add_argument("eval_dir", help="e.g. gs_experiment/local_runs/lego_prepared/eval")
    parser.add_argument("out_root", help="output directory for per-variant checkpoints + eval scene_dirs")
    parser.add_argument("--n-iters", type=int, default=3000)
    parser.add_argument("--n-eval-views", type=int, default=15, help="subsample of eval_dir's frames to render (speed)")
    parser.add_argument(
        "--variants", nargs="+", default=None,
        help=f"subset of {list(VARIANTS.keys())} to run (default: all)",
    )
    args = parser.parse_args()

    variants = VARIANTS if args.variants is None else {k: VARIANTS[k] for k in args.variants}
    run_experiment(
        args.train_dir, args.eval_dir, args.out_root,
        n_iters=args.n_iters, n_eval_views=args.n_eval_views, variants=variants,
    )


if __name__ == "__main__":
    main()
