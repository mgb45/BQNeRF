"""Reduce a scored scene's per-view maps to scalars, before they are deleted.

`scene_pipeline.sh` ends with `rm -rf $M/renders/test` -- a disk-saving step
that also threw away the only record of PER-VIEW behaviour on all thirteen
benchmark scenes. The aggregate JSONs it kept answer "within a view, where
is the error"; they cannot answer "which view should an agent distrust",
which is the empty cell of the 2x2 in `EPISTEMIC_PLAN.md`.

The fix is not to keep the maps. Four folders x 33 views x 4.7 MB is 616 MB
a scene and 8 GB across the benchmark, to support a metric that consumes one
number per view. This writes those numbers -- a few KB that can be kept
forever -- and the maps stay deletable.

Run: .venv-gsplat/bin/python gs_experiment/scripts/reduce_per_view.py MODELDIR [...]
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

FOLDERS = {"gt": "original", "render": "render",
           "unc_theirs": "error_masks", "unc_ours": "error_masks_ours"}


def reduce_scene(model: Path) -> dict:
    base = model / "renders" / "test"
    out: dict[str, list] = {}
    for key, folder in FOLDERS.items():
        files = sorted(glob.glob(str(base / folder / "*.npy")))
        if not files:
            raise SystemExit(f"{model.name}: {folder} is missing -- rerun the fit stages first")
        out[key] = [float(np.load(f).mean()) for f in files]
        out[f"{key}_n"] = len(files)

    gt = [np.load(f) for f in sorted(glob.glob(str(base / "original" / "*.npy")))]
    rd = [np.load(f) for f in sorted(glob.glob(str(base / "render" / "*.npy")))]
    # L1 per view, matching what their scorer correlates against within a view.
    out["err"] = [float(np.abs(a - b).mean()) for a, b in zip(gt, rd)]
    out["names"] = [Path(f).stem for f in sorted(glob.glob(str(base / "original" / "*.npy")))]
    return out


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        model = Path(arg)
        rec = reduce_scene(model)
        dest = Path("gs_experiment/results/per_view") / f"{model.name}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(rec))
        print(f"{model.name}: {len(rec['err'])} views -> {dest} ({dest.stat().st_size/1024:.1f} KB)")
