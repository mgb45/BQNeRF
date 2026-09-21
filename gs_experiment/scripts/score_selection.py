"""Per-view selection efficiency: the decision an agent actually makes.

Aggregate AUSE answers "within this view, where is the error". An agent
choosing where to observe next asks something different and per-VIEW: of the
poses it has not occupied, which should it distrust? FINDINGS section 21
scored that with a rank correlation and lost to a baseline that needs no
model at all -- angular distance to the nearest training view, 0.767 against
our 0.716. If a k-d tree reproduces the headline there is no headline.

Rank correlation is the wrong question. It rewards ordering the whole list,
including the middle no one acts on. Score the DECISION instead:

    of N held-out poses, take the K ranked most uncertain and credit the
    method with the realised error in that selection; normalise so random
    selection is 0 and an oracle ordering by realised error is 1; average
    over K = 1 .. N-1 so the number does not depend on a budget we chose.

Under that metric the ordering reverses and the free baseline falls away
(0.446 against 0.736, FINDINGS section 27), because it ranks isolation
faithfully -- that is what it measures -- while the most isolated pose is
often a blank wall that renders fine.

This is the standard normalised selection curve from active learning with
error-caught as the utility, not a metric invented for this paper. That
mattered: winning on a metric of our own devising would be worth little.

Inputs are the per-view scalars written by `reduce_per_view.py`. Angular
isolation, where the scene is a constructed hold-out, comes from the
`gap_manifest.json` written at construction time.

Run: .venv-gsplat/bin/python gs_experiment/scripts/score_selection.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

PER_VIEW = Path("gs_experiment/results/per_view")
GAPSCENES = Path("gs_experiment/local_runs/gapscenes")
RESULTS = Path("gs_experiment/results")

DATASET = {**{s: "Mip-NeRF 360" for s in
              ["bicycle", "flowers", "garden", "stump", "treehill",
               "room", "counter", "kitchen", "bonsai"]},
           **{s: "Tanks & Temples" for s in ["truck", "train"]},
           **{s: "Deep Blending" for s in ["drjohnson", "playroom"]}}

# model-dir prefix -> (label, gap-scene suffix or None for the dense benchmark)
FAMILIES = {"gsu_": ("dense capture", None),
            "slam_": ("trajectory gap", "_trajectory"),
            "arc_": ("30 deg arc gap", "_cone")}


def selection_efficiency(score, err) -> float:
    """Mean over K of (error caught - random) / (oracle - random).

    Ties in `score` are broken by the order the views arrive in, which is
    the acquisition order. That is deliberate: an arbitrary tie-break would
    make the number depend on sort implementation, and no method here
    produces exact ties in practice.
    """
    err = np.asarray(err, dtype=float)
    n = len(err)
    if n < 3:
        return float("nan")
    caught = np.cumsum(err[np.argsort(-np.asarray(score, dtype=float), kind="stable")])
    oracle = np.cumsum(np.sort(err)[::-1])
    chance = np.cumsum(np.full(n, err.mean()))
    num, den = caught[:-1] - chance[:-1], oracle[:-1] - chance[:-1]
    good = den > 0
    return float(np.mean(num[good] / den[good])) if good.any() else float("nan")


def isolation_for(scene: str, suffix: str | None):
    """Degrees to the nearest training view, recorded before any training."""
    if suffix is None:
        return None
    man = GAPSCENES / f"{scene}{suffix}" / "gap_manifest.json"
    if not man.exists():
        return None
    return json.load(open(man))["test_isolation_deg"]


def collect(prefix: str, suffix: str | None) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(PER_VIEW / f"{prefix}*.json"))):
        name = os.path.basename(path)[:-5]
        if "__" in name:                      # the scorer's own output, not ours
            continue
        scene = name[len(prefix):]
        d = json.load(open(path))
        row = {"scene": scene, "dataset": DATASET.get(scene, "?"), "n": len(d["err"]),
               "U-3DGS": selection_efficiency(d["unc_theirs"], d["err"]),
               "ours": selection_efficiency(d["unc_ours"], d["err"])}
        iso = isolation_for(scene, suffix)
        if iso is not None and len(iso) == len(d["err"]):
            row["farthest-point"] = selection_efficiency(iso, d["err"])
            row["median_iso_deg"] = float(np.median(iso))
        rows.append(row)
    return rows


def table(title: str, rows: list[dict]) -> None:
    if not rows:
        print(f"\n{title}: no scenes found")
        return
    cols = ["farthest-point", "U-3DGS", "ours"]
    cols = [c for c in cols if any(c in r for r in rows)]
    print(f"\n{title}  (selection efficiency: random = 0, oracle = 1)")
    print(f"  {'scene':11}{'n':>4}{'iso°':>7}" + "".join(f"{c:>16}" for c in cols))
    for r in sorted(rows, key=lambda r: r["scene"]):
        iso = f"{r['median_iso_deg']:.1f}" if "median_iso_deg" in r else "—"
        cells = "".join(f"{r.get(c, float('nan')):>16.3f}" for c in cols)
        win = "  ours" if r["ours"] > r["U-3DGS"] else ""
        print(f"  {r['scene']:11}{r['n']:>4}{iso:>7}{cells}{win}")
    print(f"  {'MEAN':11}{len(rows):>4}{'':>7}" +
          "".join(f"{np.nanmean([r.get(c, np.nan) for r in rows]):>16.3f}" for c in cols))
    wins = sum(r["ours"] > r["U-3DGS"] for r in rows)
    print(f"  ours wins {wins}/{len(rows)}")

    by_ds: dict[str, list] = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    if len(by_ds) > 1:
        print(f"  {'-- by dataset':11}")
        for ds, rs in by_ds.items():
            w = sum(r["ours"] > r["U-3DGS"] for r in rs)
            print(f"  {ds:22} n={len(rs)}  U-3DGS {np.mean([r['U-3DGS'] for r in rs]):.3f}"
                  f"   ours {np.mean([r['ours'] for r in rs]):.3f}   ours wins {w}/{len(rs)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=str(RESULTS / "selection_efficiency.json"),
                    help="where to write the machine-readable copy")
    a = ap.parse_args()

    out = {}
    for prefix, (label, suffix) in FAMILIES.items():
        rows = collect(prefix, suffix)
        table(label, rows)
        out[label] = rows

    Path(a.json).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {a.json}")
