"""Score the SLAM-style trajectory hold-outs: does the uncertainty know which
viewpoints the agent never visited?

The aggregate metrics (AUSE, Pearson) answer "within a view, where is the
error", which FINDINGS section 13 showed is the question a residual-supervised
baseline is good at. The question that matters for an agent deciding where to
go next is different and per-VIEW: given a pose it has not occupied, can it
tell in advance how much to trust the render there?

So the headline here is the correlation between each held-out view's ANGULAR
ISOLATION -- degrees to the nearest training view, recorded by
`build_colmap_gap_scene.py` at construction time -- and (a) the error actually
incurred there, (b) each method's predicted uncertainty. A method whose
per-view uncertainty tracks isolation is one an agent can plan with; one that
does not is a within-view error map wearing the wrong hat.

Run: .venv-gsplat/bin/python gs_experiment/scripts/analyse_slam_gap.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from gs_experiment.scripts.render_reconstruction import LOCAL_RUNS, RESULTS_DIR

SCENES = ["kitchen", "counter", "bonsai"]
METHODS = [("error_masks", "U-3DGS (their code)"), ("error_masks_ours", "ours (post-hoc)")]


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a.argsort().argsort(), b.argsort().argsort())[0, 1])


def per_view_error_from_checkpoint(model, source, images_dir, n_views):
    """Recompute per-view error directly, rather than from their saved .npy
    renders. Those are large and get deleted to keep disk in budget; the
    numbers are equivalent because gsplat was verified against their
    rasterizer to 0.03 dB PSNR (57.5 dB agreement) on their own checkpoint."""
    import torch
    from PIL import Image

    from gs_experiment.ply_io import read_3dgs_ply
    from gs_experiment.scripts.our_uncertainty_for_3dgs_model import (
        intrinsics, load_cameras, load_gt, render, view_matrix)

    ck = read_3dgs_ply(str(model / "point_cloud" / "iteration_30000" / "point_cloud.ply"))
    test_cams, _ = load_cameras(model, n_views)
    im0 = Image.open(next((source / images_dir).glob("*")))
    width, height = im0.size
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device="cuda")  # noqa: E731
    ck_t = {"means": t(ck["positions"]), "quats": t(ck["rotations"]),
            "scales": t(ck["scales"]), "opacities": t(ck["opacities"])}
    sh = t(ck["sh_coeffs"]).transpose(1, 2).contiguous()
    K, bg = t(intrinsics(test_cams[0], width, height))[None], t([0.0, 0.0, 0.0])
    out = []
    for cam in test_cams:
        rec = render(ck_t, sh, t(view_matrix(cam))[None], K, width, height,
                     ck["sh_degree"], bg).cpu().numpy()
        gt = load_gt(source, images_dir, cam["img_name"], width, height)
        out.append(float(np.abs(rec - gt).mean()))
    return np.array(out)


def load_scene(scene):
    model = LOCAL_RUNS / f"slam_{scene}"
    gapdir = LOCAL_RUNS / "gapscenes" / f"{scene}_trajectory"
    manifest = json.load(open(gapdir / "gap_manifest.json"))
    base = model / "renders" / "test"
    if not (base / "error_masks_ours").is_dir():
        return None
    iso = np.array(manifest["test_isolation_deg"])
    n_views = len(sorted(glob.glob(str(base / "error_masks_ours" / "*.npy"))))
    gt = [np.load(f) for f in sorted(glob.glob(str(base / "original" / "*.npy")))]
    rd = [np.load(f) for f in sorted(glob.glob(str(base / "render" / "*.npy")))]
    if gt and len(gt) == n_views:
        err = np.array([np.abs(g - r).mean() for g, r in zip(gt, rd)])
    else:
        err = per_view_error_from_checkpoint(model, gapdir, "images_2", n_views)
    unc = {}
    for folder, label in METHODS:
        fs = sorted(glob.glob(str(base / folder / "*.npy")))
        if len(fs) == len(err):
            unc[label] = np.array([np.load(f).mean() for f in fs])
    return {"scene": scene, "iso": iso[:len(err)], "err": err, "unc": unc,
            "n_train": manifest["n_train"]}


def run():
    rows = [load_scene(s) for s in SCENES]
    rows = [r for r in rows if r]
    if not rows:
        print("no SLAM gap results yet -- the queued runs have not produced renders")
        return
    print("Per-VIEW behaviour on trajectory hold-outs (unvisited part of the capture).")
    print("Isolation = degrees from a held-out view to the NEAREST training view.\n")
    hdr = f"{'scene':<9}{'train':>6}{'views':>6}{'iso med':>9}{'err~iso':>9}"
    for _, label in METHODS:
        hdr += f"{label.split()[0] + ' ~iso':>14}{label.split()[0] + ' ~err':>14}"
    print(hdr)
    print("-" * len(hdr))
    agg = {label: {"iso": [], "err": []} for _, label in METHODS}
    for r in rows:
        line = (f"{r['scene']:<9}{r['n_train']:>6}{len(r['err']):>6}"
                f"{np.median(r['iso']):>9.1f}{spearman(r['iso'], r['err']):>9.3f}")
        for _, label in METHODS:
            if label in r["unc"]:
                si, se = spearman(r["iso"], r["unc"][label]), spearman(r["err"], r["unc"][label])
                agg[label]["iso"].append(si); agg[label]["err"].append(se)
                line += f"{si:>14.3f}{se:>14.3f}"
            else:
                line += f"{'--':>14}{'--':>14}"
        print(line)
    print("-" * len(hdr))
    line = f"{'mean':<9}{'':>6}{'':>6}{'':>9}{'':>9}"
    for _, label in METHODS:
        v = agg[label]
        line += (f"{np.mean(v['iso']):>14.3f}" if v["iso"] else f"{'--':>14}")
        line += (f"{np.mean(v['err']):>14.3f}" if v["err"] else f"{'--':>14}")
    print(line)
    print("\n'~iso' = does per-view uncertainty know which poses were never visited;")
    print("'~err' = does it rank views by the error actually incurred there.")
    json.dump({r["scene"]: {"iso": r["iso"].tolist(), "err": r["err"].tolist(),
                            "unc": {k: v.tolist() for k, v in r["unc"].items()}} for r in rows},
              open(RESULTS_DIR / "slam_gap_perview.json", "w"), indent=1)


if __name__ == "__main__":
    run()
