"""ROADMAP.md item 6 (systematic ablations): how sensitive is the headline
sparsity-correlation claim (BQ variance vs. local splat density) to the
`window_radius` hyperparameter? Every result reported so far
(sparsity_correlation_experiment.py, §24/§26/§28/§29 in
gs_experiment/results/FINDINGS.md) used one hand-picked value per scene
family. §28 already found this matters a lot in one specific case --
reusing a `window_radius` picked for a *different* experiment on the same
scene flipped the correlation's sign entirely (r=+0.22 at window=1.6 vs.
r=-0.96 at window=0.15) -- so this sweeps the knob explicitly and reports
the whole curve, rather than trusting that the one value used elsewhere
was a lucky pick.

Fixes `sigma` at each checkpoint's already-established value (so this
isolates window_radius specifically, not a joint sigma/window_radius
search) and sweeps window_radius across a wide multiplicative range
(0.2x-8x the established value) on the same three real checkpoints used
throughout items 4-5: lego wide, and the two thin-rod checkpoints
(from-scratch trainer, gsplat reference-strategy trainer) from the
cross-trainer validation.

Needs torch + gsplat only insofar as the checkpoints were already trained;
this script itself is pure numpy/scipy.

Run: .venv-gsplat/bin/python gs_experiment/window_radius_ablation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from gs_experiment.pixel_uncertainty import LocalUncertaintyEngine, make_default_3d_position_kernel
from gs_experiment.ply_io import read_3dgs_ply

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = [
    dict(label="lego_wide", path="gs_experiment/local_runs/lego_prepared/wide/splats.ply", sigma=0.05, base_window=0.08),
    dict(label="thinrod_fromscratch", path="gs_experiment/local_runs/nbv_out/nll_experiment/baseline/splats.ply", sigma=0.05, base_window=0.15),
    dict(label="thinrod_referencestrategy", path="gs_experiment/local_runs/nbv_out/reference_strategy/splats.ply", sigma=0.05, base_window=0.15),
]

MULTIPLIERS = [0.2, 0.5, 1.0, 2.0, 4.0, 8.0]


def sweep_one_checkpoint(path: str, sigma: float, base_window: float, n_samples: int = 150, min_opacity: float = 0.1, seed: int = 0):
    ck = read_3dgs_ply(path)
    keep = ck["opacities"] > min_opacity
    positions = ck["positions"][keep]
    colors = ck["sh_coeffs"][keep, :, 0].mean(axis=1)

    bounds = tuple((positions[:, d].min() - 0.3, positions[:, d].max() + 0.3) for d in range(3))
    pos_kernel = make_default_3d_position_kernel(sigma=sigma)
    engine = LocalUncertaintyEngine(positions=positions, values=colors, pos_kernel=pos_kernel, scene_bounds=bounds)

    rng = np.random.default_rng(seed)
    query_idx = rng.choice(len(positions), size=min(n_samples, len(positions)), replace=False)
    query_points = positions[query_idx]

    rows = []
    for mult in MULTIPLIERS:
        window_radius = base_window * mult
        local_counts = np.array(
            [engine.tree.query_ball_point(p, window_radius, return_length=True) for p in query_points]
        )
        bq_variances = np.array([engine.spatial_only_variance(p, window_radius).variance for p in query_points])
        r, p = pearsonr(np.log1p(local_counts), bq_variances)
        rows.append(dict(mult=mult, window_radius=window_radius, r=r, p=p, median_count=float(np.median(local_counts))))
    return len(positions), rows


def run():
    fig, ax = plt.subplots(figsize=(7, 5))
    summary_rows = []

    for ckpt in CHECKPOINTS:
        n_splats, rows = sweep_one_checkpoint(ckpt["path"], ckpt["sigma"], ckpt["base_window"])
        print(f"\n=== {ckpt['label']} ({n_splats} splats, sigma={ckpt['sigma']}, base_window={ckpt['base_window']}) ===")
        print(f"{'mult':>6}{'window_radius':>16}{'median_count':>14}{'r':>10}{'p':>12}")
        for row in rows:
            print(f"{row['mult']:>6.1f}{row['window_radius']:>16.3f}{row['median_count']:>14.1f}{row['r']:>10.3f}{row['p']:>12.2e}")
            summary_rows.append(dict(label=ckpt["label"], **row))

        mults = [row["mult"] for row in rows]
        rs = [row["r"] for row in rows]
        ax.plot(mults, rs, marker="o", label=ckpt["label"])

    ax.axhline(0.0, color="gray", linewidth=1, linestyle=":")
    ax.axvline(1.0, color="gray", linewidth=1, linestyle=":", label="each checkpoint's established window_radius")
    ax.set_xscale("log")
    ax.set_xlabel("window_radius / established value (log scale)")
    ax.set_ylabel("Pearson r(log(1+local count), BQ variance)")
    ax.set_title("Sensitivity of the sparsity-correlation claim to window_radius")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = RESULTS_DIR / "window_radius_ablation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out}")

    print("\n=== summary: does the correlation survive at 0.2x-8x the established window_radius? ===")
    for ckpt in CHECKPOINTS:
        label = ckpt["label"]
        label_rows = [r for r in summary_rows if r["label"] == label]
        signs = set(np.sign(r["r"]) for r in label_rows if r["p"] < 0.05)
        strong = [r for r in label_rows if r["p"] < 0.05 and abs(r["r"]) > 0.3]
        print(
            f"{label}: significant-r sign(s) across the sweep = {signs}; "
            f"{len(strong)}/{len(label_rows)} multipliers give |r|>0.3 and p<0.05"
        )


if __name__ == "__main__":
    run()
