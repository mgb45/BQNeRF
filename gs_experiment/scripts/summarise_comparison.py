"""Aggregate `run_comparison.py`'s per-scene JSON into the cross-scene table
and figure. Reads only what the frozen protocol wrote; computes no new metric.

Run: .venv-gsplat/bin/python gs_experiment/scripts/summarise_comparison.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gs_experiment.scripts.render_reconstruction import RESULTS_DIR

METRICS = [("spearman", "per-pixel Spearman", "hi"),
           ("per_view_spearman", "per-view Spearman", "hi"),
           ("ause", "AUSE", "lo"),
           ("gain_over_constant", "NLL gain vs constant", "hi")]
SHORT = {"ours (SH posterior)": "ours", "residual-supervised SH": "resid-sup",
         "uniform-coverage SH": "coverage", "weight concentration": "weight-conc",
         "render gradient": "grad"}


def load():
    files = sorted(glob.glob(str(RESULTS_DIR / "comparison_*_wide.json")))
    real = RESULTS_DIR / "comparison_bonsai_gap_0.json"
    if real.exists():
        files.append(str(real))       # real capture last, kept visually distinct
    scenes, data = [], {}
    for f in files:
        scene = Path(f).stem.replace("comparison_", "").replace("_wide", "").replace("_gap_0", "")
        scenes.append(scene)
        for row in json.load(open(f)):
            data.setdefault(row["name"], {})[scene] = row
    return scenes, data


def run():
    scenes, data = load()
    methods = list(data)
    print(f"{len(scenes)} scenes: {', '.join(scenes)}  (bonsai = real Mip-NeRF 360 capture)\n")
    for key, label, better in METRICS:
        vals = {m: np.array([data[m][s][key] for s in scenes]) for m in methods}
        best = (np.argmax if better == "hi" else np.argmin)(
            np.stack([vals[m] for m in methods]), axis=0)
        print(f"--- {label} ({'higher' if better == 'hi' else 'lower'} better) ---")
        print(f"{'method':<24}" + "".join(f"{s[:7]:>9}" for s in scenes) + f"{'mean':>9}{'wins':>7}")
        for i, m in enumerate(methods):
            print(f"{m:<24}" + "".join(f"{v:>9.3f}" for v in vals[m]) +
                  f"{vals[m].mean():>9.3f}{int((best == i).sum()):>7}")
        print()

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.2))
    x = np.arange(len(scenes))
    width = 0.16
    for ax, (key, label, _) in zip(axes, [METRICS[0], METRICS[1]]):
        for i, m in enumerate(methods):
            v = [data[m][s][key] for s in scenes]
            ax.bar(x + (i - 2) * width, v, width, label=SHORT.get(m, m),
                   zorder=3, lw=1.6 if m.startswith("ours") else 0,
                   edgecolor="black" if m.startswith("ours") else "none")
        ax.axhline(0, color="k", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([s + ("\n(real)" if s == "bonsai" else "") for s in scenes], fontsize=10)
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=13)
        ax.grid(axis="y", alpha=0.3, zorder=0)
    axes[0].legend(ncol=5, fontsize=9, loc="upper center", bbox_to_anchor=(1.05, -0.12))
    fig.suptitle("Uncertainty methods on identical checkpoints, frozen protocol, object pixels "
                 "-- within a view (left) vs across views (right)", fontsize=13)
    fig.tight_layout()
    path = RESULTS_DIR / "comparison_summary.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    run()
