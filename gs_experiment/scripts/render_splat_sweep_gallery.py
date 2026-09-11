"""Single-scene splat-budget sweep, reusing render_scene_gallery.py's row-building
and plotting exactly (same columns: ground truth / gsplat reconstruction / |error| /
raw posterior variance, sigma+kappa fit per checkpoint) -- rows are increasing splat
budgets for ONE scene instead of different scenes, so growing splat count's effect
on quality and BQ uncertainty can be seen on one figure instead of flipping between
separate scene_gallery*.png files.

Every budget above render_scene_gallery.py's own ~300k default is passed through
build_rows' `max_observations_per_splat`, sized per budget by
splat_scene.max_observations_per_splat_for_budget -- the same validated memory model
splat_scene.py already uses for load_from_gsplat_checkpoint, not a new guess.
Confirmed necessary the hard way: an early version of this script called build_rows
with no cap at all, and rendering the 1,000,000-splat checkpoint uncapped OOM-killed
the host at 18GB (an explicit gc.collect()/torch.cuda.empty_cache() between budget
iterations was tried first and did NOT fix it -- the unbuffered log showed the
crash happened *during* the first, single 1M-splat call, not from cross-iteration
accumulation, so a per-call cap was the actual fix needed, not per-iteration
cleanup). BUDGETS stops at 1,000,000, not the 3,000,000 checkpoint also on disk for
lego, purely out of caution -- the cap makes 3M plausible too, just not verified
here yet.

Needs torch + gsplat (requirements-gsplat.txt). Needs each budget already trained
into <prepared_root>/<scene>_prepared/budget_<n>/ (or "wide" for the project's
standard 300k recipe) -- does not train anything itself.

Run: .venv-gsplat/bin/python gs_experiment/scripts/render_splat_sweep_gallery.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gs_experiment.scripts.render_scene_gallery import RESULTS_DIR, build_rows, plot_gallery
from gs_experiment.splat_scene import max_observations_per_splat_for_budget

SCENE = "lego"
VIEW_IDX = 21
BUDGETS = [500, 10_000, 30_000, 100_000, "wide", 1_000_000]
WIDE_BUDGET_EQUIVALENT = 300_000  # "wide" recipe's actual splat count, for the memory-model lookup below


def budget_label(b) -> str:
    return "wide (300,000)" if b == "wide" else f"{b:,}"


def run(scene: str = SCENE, view_idx: int = VIEW_IDX, budgets=BUDGETS, out_path=None, fit_noise_variance=True):
    import gc

    import torch

    rows = []
    for b in budgets:
        checkpoint_subdir = b if b == "wide" else f"budget_{b}"
        budget_int = WIDE_BUDGET_EQUIVALENT if b == "wide" else b
        max_obs = max_observations_per_splat_for_budget(budget_int)
        print(f"=== budget={b} (max_observations_per_splat={max_obs}) ===", flush=True)
        scene_rows = build_rows(
            scene_views={scene: view_idx}, checkpoint_subdir=checkpoint_subdir, max_observations_per_splat=max_obs,
            fit_noise_variance=fit_noise_variance,
        )
        label = budget_label(b)
        for r in scene_rows:
            r["scene"] = label  # row label becomes the budget, not the scene name
            # plot_gallery (render_scene_gallery.py) was refactored to the multi-budget
            # side-by-side layout and now expects each row to carry a "budget_rows" list
            # of (label, row) pairs rather than being plotted flat -- wrap each single-budget
            # row as its own one-entry "budget_rows" list so it renders as one row per budget
            # (this script's actual intent), not side by side.
            rows.append(dict(scene=label, budget_rows=[(label, r)]))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_path = Path(out_path) if out_path else (RESULTS_DIR / f"{scene}_splat_sweep.png")
    plot_gallery(rows, out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default=SCENE)
    parser.add_argument("--view-idx", type=int, default=VIEW_IDX)
    parser.add_argument("--budgets", nargs="+", default=BUDGETS, help='e.g. --budgets 500 10000 wide 1000000')
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--no-fit-noise-variance", action="store_true",
        help="disable the noise-aware joint sigma/noise_variance fit (default: enabled) -- "
        "reproduces the old noiseless-fit figure.",
    )
    args = parser.parse_args()
    budgets = [int(b) if b != "wide" else b for b in args.budgets]
    run(
        scene=args.scene, view_idx=args.view_idx, budgets=budgets, out_path=args.out,
        fit_noise_variance=not args.no_fit_noise_variance,
    )


if __name__ == "__main__":
    main()
