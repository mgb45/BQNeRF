"""ROADMAP.md item 4 ("full NeRF-Synthetic, all 8 scenes -- lego is the
only one attempted") and item 7 (multi-scene statistics): extends the
headline sparsity-correlation and calibration checks
(sparsity_correlation_experiment.py, calibration_experiment.py) across the
complete standard 8-scene NeRF-Synthetic benchmark, not just lego.

Scope note, stated up front rather than left implicit: this trains each
new scene at a deliberately *lighter* budget than lego's original 80,000-
splat-cap run (`n_splats=2000`, `max_splats=15000`, `n_iters=2500`) --
enough for a real, densified checkpoint with genuine view-coverage-
dependent splat density (the property these two checks actually need),
not a publication-quality reconstruction. This is a scope choice for
feasible wall-clock across 7 new scenes in one sitting, not a hidden
shortcut -- reconstruction quality is not what either check measures.

Dataset: `pablovela5620/nerf-synthetic-mirror` on Hugging Face -- unlike
the mirror used for lego originally (`phuckstnk63/nerf-synthetic`, which
turned out to contain only lego at all, and an incomplete test split even
for that one scene -- gs_experiment/results/FINDINGS.md section 20), this
one is verified complete for all 8 standard scenes (100 train / 100 val /
200 real color test images each, checked file-by-file before trusting it,
not assumed from the repo name).

Needs torch + gsplat (training) and huggingface_hub (download).

Run: .venv-gsplat/bin/python gs_experiment/multi_scene_experiment.py chair drums ficus hotdog materials mic ship
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scipy.stats import pearsonr

from gs_experiment.kernel_family_ablation import build_engine, calibration  # reuses the same checkpoint-loading/calibration convention
from gs_experiment.prepare_nerf_synthetic import run as prepare_scene
from gs_experiment.train_minimal_gsplat import train

RAW_ROOT = "gs_experiment/local_runs/nerf_synthetic_raw"
PREPARED_ROOT = "gs_experiment/local_runs"

TRAIN_KWARGS = dict(
    n_splats=2000, bounds=((-2.5, 2.5), (-2.5, 2.5), (-2.5, 2.5)), sh_degree=1, n_iters=2500, seed=0,
    init_scale=0.05, opacity_reg_weight=0.01, densify=True, densify_interval=300, densify_start=300,
    max_splats=15000, log_every=500,
)
SIGMA = 0.05
WINDOW_RADIUS = 0.08


def download_scene(scene: str):
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="pablovela5620/nerf-synthetic-mirror", repo_type="dataset",
        allow_patterns=[f"{scene}/*"], local_dir=RAW_ROOT,
    )


def prepare_and_train(scene: str, skip_download: bool = False, skip_prepare: bool = False, skip_train: bool = False):
    raw_dir = os.path.join(RAW_ROOT, scene)
    prepared_dir = os.path.join(PREPARED_ROOT, f"{scene}_prepared")
    ply_path = os.path.join(prepared_dir, "wide", "splats.ply")

    if not skip_download and not os.path.exists(raw_dir):
        print(f"[{scene}] downloading...")
        download_scene(scene)

    if not skip_prepare and not os.path.exists(os.path.join(prepared_dir, "wide", "transforms.json")):
        print(f"[{scene}] preparing...")
        prepare_scene(raw_dir, prepared_dir)

    if not skip_train and not os.path.exists(ply_path):
        print(f"[{scene}] training (lighter budget than lego's original)...")
        train(os.path.join(prepared_dir, "wide"), ply_path, **TRAIN_KWARGS)

    return ply_path


def evaluate(scene: str, ply_path: str, n_samples: int = 150, seed: int = 0):
    from scipy.spatial import cKDTree

    engine = build_engine(ply_path, "rbf", SIGMA)
    n_splats = len(engine.positions)

    import numpy as np

    rng = np.random.default_rng(seed)
    query_idx = rng.choice(n_splats, size=min(n_samples, n_splats), replace=False)
    query_points = engine.positions[query_idx]
    local_counts = np.array([engine.tree.query_ball_point(p, WINDOW_RADIUS, return_length=True) for p in query_points])
    bq_variances = np.array([engine.spatial_only_variance(p, WINDOW_RADIUS).variance for p in query_points])
    sparsity_r, sparsity_p = pearsonr(np.log1p(local_counts), bq_variances)

    calib_r, calib_p, nll_bq, nll_const = calibration(engine, WINDOW_RADIUS, n_samples=n_samples, seed=seed)

    return dict(
        scene=scene, n_splats=n_splats, median_local_count=float(np.median(local_counts)),
        sparsity_r=sparsity_r, sparsity_p=sparsity_p, calib_r=calib_r, calib_p=calib_p,
        nll_bq=nll_bq, nll_const=nll_const,
    )


def run(scenes, skip_download=False, skip_prepare=False, skip_train=False):
    results = []
    for scene in scenes:
        ply_path = prepare_and_train(scene, skip_download=skip_download, skip_prepare=skip_prepare, skip_train=skip_train)
        result = evaluate(scene, ply_path)
        results.append(result)
        print(
            f"[{scene}] n_splats={result['n_splats']}  median_local_count={result['median_local_count']:.1f}  "
            f"sparsity_r={result['sparsity_r']:.3f} (p={result['sparsity_p']:.1e})  "
            f"calib_r={result['calib_r']:.3f} (p={result['calib_p']:.1e})  "
            f"NLL(bq)={result['nll_bq']:.2f}  NLL(const)={result['nll_const']:.2f}"
        )

    print(f"\n{'scene':<12}{'n_splats':>10}{'sparsity_r':>13}{'calib_r':>10}{'NLL(bq)':>12}{'NLL(const)':>12}")
    for r in results:
        print(f"{r['scene']:<12}{r['n_splats']:>10}{r['sparsity_r']:>13.3f}{r['calib_r']:>10.3f}{r['nll_bq']:>12.2f}{r['nll_const']:>12.2f}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenes", nargs="+")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()
    run(args.scenes, skip_download=args.skip_download, skip_prepare=args.skip_prepare, skip_train=args.skip_train)


if __name__ == "__main__":
    main()
