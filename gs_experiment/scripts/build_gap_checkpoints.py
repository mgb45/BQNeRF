"""Build epistemic-regime checkpoints for every scene, at full capacity.

FINDINGS section 8 established that the epistemic regime cannot be simulated
on a frozen map: conditioning the posterior on fewer cameras does not change
the reconstruction's ERROR, because the map was still fit to all of them, so
there is no epistemic error to predict. Measuring calibration where error is
genuinely epistemic therefore requires retraining with views removed.

FINDINGS section 16 then established that capacity is a real factor and that
a reduced budget reports conclusions which do not survive at the operating
point anyone uses. So these are trained at the project's full `wide` recipe,
not the calibrated cheap one.

Construction matches lego's existing `gap_4` exactly: remove every training
view whose camera direction lies within 75 degrees of one reference
direction, keep the rest untouched. Coverage stays dense everywhere except
inside the cone, which isolates the coverage gap from overall view count.

Run: .venv-gsplat/bin/python gs_experiment/scripts/build_gap_checkpoints.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from gs_experiment.nerf_transforms import load_transforms
from gs_experiment.scripts.render_posterior_ensemble import BACKGROUND_COLOR
from gs_experiment.scripts.render_reconstruction import LOCAL_RUNS, SCENES

GAP_HALF_WIDTH_DEG = 75.0
REFERENCE_IDX = 0
GAP_NAME = "gap75"


def build(scene: str) -> tuple[Path, int, int]:
    pool = LOCAL_RUNS / f"{scene}_prepared" / "wide"
    out = LOCAL_RUNS / f"{scene}_prepared" / GAP_NAME
    cax, frames = load_transforms(str(pool / "transforms.json"))
    centres = np.array([np.asarray(c2w)[:3, 3] for _, c2w in frames])
    dirs = centres / np.linalg.norm(centres, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip(dirs @ dirs[REFERENCE_IDX], -1.0, 1.0)))
    keep = np.where(ang > GAP_HALF_WIDTH_DEG)[0]

    out.mkdir(parents=True, exist_ok=True)
    json.dump({"camera_angle_x": cax,
               "frames": [{"file_path": frames[i][0],
                           "transform_matrix": np.asarray(frames[i][1]).tolist()} for i in keep]},
              open(out / "transforms.json", "w"))
    for i in keep:
        rel = frames[i][0] + ".png"
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        if not (out / rel).exists():
            os.symlink(os.path.abspath(pool / rel), out / rel)
    return out, len(keep), len(frames)


def run():
    from gs_experiment.scripts.train_minimal_gsplat import DEFAULT_TRAIN_KWARGS, train

    for scene in SCENES:
        out, n_keep, n_total = build(scene)
        ply = out / "splats.ply"
        if ply.exists():
            print(f"{scene}: {n_keep}/{n_total} views kept, checkpoint already present", flush=True)
            continue
        kw = dict(DEFAULT_TRAIN_KWARGS)
        kw["log_every"] = 10 ** 9
        assert kw["background_color"] == BACKGROUND_COLOR, "train/eval background mismatch"
        print(f"{scene}: {n_keep}/{n_total} views kept ({GAP_HALF_WIDTH_DEG:.0f} deg cone removed), "
              f"training at full capacity...", flush=True)
        t0 = time.time()
        train(str(out), str(ply), **kw)
        print(f"  done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    run()
