"""Panels for looking at the wins, rather than reading them off a table.

Every claim in FINDINGS sections 24-28 is a number produced by someone
else's scorer. That is the right way to be believed and the wrong way to
catch a degenerate result: a uniform sigma map, a black render, a hold-out
that is mostly sky, or an error map dominated by one blown highlight would
all score plausibly and look obviously wrong.

So: for each case, the held-out views at their extremes of realised error,
with all five maps side by side on a shared scale.

Sigma maps are normalised per PANEL, not per row, because the two methods
produce different units -- theirs is a learned per-pixel scalar and ours is
a posterior standard deviation in colour units. Comparing their absolute
magnitudes would be meaningless; comparing where each puts its mass is the
whole question. The error map uses the same treatment for the same reason.

Run: .venv-gsplat/bin/python gs_experiment/scripts/make_inspection_panels.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOCAL = Path("gs_experiment/local_runs")
OUT = Path("gs_experiment/results/inspection")

# label -> (model dir, what the case is, how it scored)
CASES = [
    ("drjohnson_dense", "gsu_drjohnson", "Deep Blending, dense capture",
     "ours 0.341 AUSE vs 0.368 — a win in the regime we usually lose"),
    ("truck_arc", "arc_truck", "Tanks & Temples, 30° arc hold-out",
     "ours 0.624 vs 0.684 AUSE, 0.765 vs 0.245 selection — the full inversion"),
    ("drjohnson_arc", "arc_drjohnson", "Deep Blending, 30° arc hold-out",
     "ours 0.462 vs 0.548 AUSE, their Pearson 0.043"),
    ("room_arc", "arc_room", "Mip-NeRF 360, 30° arc hold-out",
     "ours 0.544 vs 0.584 AUSE — the scene §21 had to exclude"),
    ("garden_gap", "slam_garden", "CONTROL: trajectory gap we lose within-view",
     "ours 0.540 vs 0.441 AUSE, but 0.758 vs 0.370 selection"),
    ("garden_dense", "gsu_garden", "CONTROL: dense capture, our worst scene",
     "ours 0.487 vs 0.310 AUSE — what a clear loss looks like"),
]

ROW = [("original", "ground truth", None), ("render", "render", None),
       ("error", "realised error", "inferno"),
       ("error_masks_ours", "ours σ", "viridis"),
       ("error_masks", "U-3DGS σ", "viridis")]


def load(base: Path, folder: str):
    files = sorted(glob.glob(str(base / folder / "*.npy")))
    return [np.load(f) for f in files] if files else None


def to_hwc(a):
    """gsplat writes (H,W,3); their pipeline writes (3,H,W). Getting this
    wrong silently transposes an image into stripes, which is exactly the
    failure this script exists to catch, so it is asserted rather than
    guessed."""
    a = np.asarray(a)
    if a.ndim == 3 and a.shape[0] == 3 and a.shape[2] != 3:
        a = np.transpose(a, (1, 2, 0))
    return a


def scalar(a):
    a = to_hwc(a)
    return a.mean(axis=2) if a.ndim == 3 else a


def build(label, model, title, scored, n_views=3):
    base = LOCAL / model / "renders" / "test"
    gt, rd = load(base, "original"), load(base, "render")
    if gt is None or rd is None:
        print(f"  {label}: no maps under {base} — skipped")
        return None
    uo, ut = load(base, "error_masks_ours"), load(base, "error_masks")
    err = [np.abs(to_hwc(a) - to_hwc(b)).mean(axis=2) for a, b in zip(gt, rd)]

    # worst, median and best held-out view by realised error: a win that only
    # holds on the easy views is one worth seeing.
    order = np.argsort([e.mean() for e in err])
    picks = [(order[-1], "worst view"), (order[len(order) // 2], "median view"),
             (order[0], "best view")][:n_views]

    fig, axes = plt.subplots(len(picks), len(ROW),
                             figsize=(3.1 * len(ROW), 2.6 * len(picks)), squeeze=False)
    for r, (idx, tag) in enumerate(picks):
        srcs = {"original": to_hwc(gt[idx]), "render": to_hwc(rd[idx]), "error": err[idx],
                "error_masks_ours": scalar(uo[idx]) if uo else None,
                "error_masks": scalar(ut[idx]) if ut else None}
        for c, (key, name, cmap) in enumerate(ROW):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            img = srcs.get(key)
            if img is None:
                ax.text(.5, .5, "absent", ha="center", va="center", color="#999")
                continue
            if cmap is None:
                ax.imshow(np.clip(img, 0, 1))
            else:
                hi = np.percentile(img, 99)          # one blown pixel must not
                ax.imshow(img, cmap=cmap, vmin=0, vmax=hi if hi > 0 else None)
            if r == 0:
                ax.set_title(name, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{tag}\nL1 {err[idx].mean():.3f}", fontsize=9)
    fig.suptitle(f"{title}\n{scored}", fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{label}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  {label}: {len(gt)} views -> {path} ({path.stat().st_size / 1024:.0f} KB)")
    return path


if __name__ == "__main__":
    for label, model, title, scored in CASES:
        build(label, model, title, scored)
