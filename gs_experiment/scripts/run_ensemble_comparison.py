"""Deep-ensemble baseline, at matched capacity.

The deep ensemble is the reference epistemic baseline -- it is what
`zhao2026posterior` benchmarks against, and it is the honest cost ceiling,
since it needs N full trainings where every other method here needs none.

Matching capacity is the whole difficulty. An ensemble's members are
DIFFERENT checkpoints, so it cannot simply be dropped into the comparison
that runs every post-hoc method on one shared 300k-splat `wide` map: the
ensemble's mean render, and therefore its residuals, would differ from every
other method's, and the protocol requires an identical target. So this script
runs the entire comparison one capacity down, where the arithmetic works:

  * train `N_MEMBERS` checkpoints on the SAME training views, differing only
    in seed, using the project's canonical recipe at the calibrated budget
    (28.4 dB held-out at 30 lego views in 80 s -- see ROADMAP item 1);
  * member 0 is the shared reference checkpoint. Every post-hoc method,
    ours included, is built on it and scored against its residuals;
  * the ensemble's uncertainty is the per-pixel standard deviation across
    all N members.

This slightly disadvantages the ensemble -- its spread is centred on the
ensemble mean, not on member 0 -- so the supplementary row
`deep ensemble (own mean)` scores it against its own mean render as well,
which is its natural form. Both are reported; neither is chosen after the
fact.

Run: .venv-gsplat/bin/python gs_experiment/scripts/run_ensemble_comparison.py [scene]
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from PIL import Image

from gs_experiment.evaluation import Prediction, compare, format_table, object_mask
from gs_experiment.nerf_transforms import fov_x_to_intrinsics, load_transforms, opencv_viewmat_from_c2w
from gs_experiment.ply_io import read_3dgs_ply
from gs_experiment.rasterized_sh_precision import accumulate_sh_precision_rasterized, estimate_noise_variance
from gs_experiment.scripts.render_geometry_ensemble import render
from gs_experiment.scripts.render_posterior_ensemble import (
    BACKGROUND_COLOR, N_PROBES, SEED, empirical_band_precision, sample_sh_draws,
)
from gs_experiment.scripts.render_reconstruction import LOCAL_RUNS, RESULTS_DIR

N_MEMBERS = 5
N_DRAWS = 16
TRAIN_OVERRIDES = dict(n_iters=6000, max_splats=100000, densify_end=3000, log_every=10**9)


def ensemble_dir(scene):
    return LOCAL_RUNS / f"{scene}_prepared" / "ensemble_configA"


def train_members(scene, pool_dir):
    """N checkpoints on identical views, differing only in seed. Cached: each
    is ~80 s and nothing about them changes between runs."""
    from gs_experiment.scripts.train_minimal_gsplat import DEFAULT_TRAIN_KWARGS, train

    out = ensemble_dir(scene)
    out.mkdir(parents=True, exist_ok=True)
    # One scene directory, shared by every member -- identical views by construction.
    member_scene = out / "scene"
    if not (member_scene / "transforms.json").exists():
        member_scene.mkdir(parents=True, exist_ok=True)
        cax, frames = load_transforms(str(pool_dir / "transforms.json"))
        json.dump({"camera_angle_x": cax,
                   "frames": [{"file_path": fp, "transform_matrix": np.asarray(c2w).tolist()}
                              for fp, c2w in frames]}, open(member_scene / "transforms.json", "w"))
        for fp, _ in frames:
            rel = fp + ".png"
            (member_scene / rel).parent.mkdir(parents=True, exist_ok=True)
            if not (member_scene / rel).exists():
                os.symlink(os.path.abspath(pool_dir / rel), member_scene / rel)

    paths, secs = [], 0.0
    for m in range(N_MEMBERS):
        ply = out / f"member_{m}.ply"
        if not ply.exists():
            kw = dict(DEFAULT_TRAIN_KWARGS)
            kw.update(TRAIN_OVERRIDES)
            kw["seed"] = 1000 + m
            assert kw["background_color"] == BACKGROUND_COLOR, "train/eval background mismatch"
            t0 = time.time()
            train(str(member_scene), str(ply), **kw)
            secs += time.time() - t0
            print(f"  trained member {m} ({time.time() - t0:.0f}s)", flush=True)
        paths.append(ply)
    return member_scene, paths, secs


def run(scene="lego", pool="wide", eval_name="eval"):
    pool_dir = LOCAL_RUNS / f"{scene}_prepared" / pool
    eval_dir = LOCAL_RUNS / f"{scene}_prepared" / eval_name
    member_scene, plies, train_secs = train_members(scene, pool_dir)
    members = [read_3dgs_ply(str(p)) for p in plies]
    reference = members[0]
    sh_coeffs, degree = reference["sh_coeffs"], reference["sh_degree"]
    cax, train_frames = load_transforms(str(member_scene / "transforms.json"))
    with Image.open(str(member_scene / (train_frames[0][0] + ".png"))) as im:
        tw, th = im.size
    train_K = fov_x_to_intrinsics(cax, tw, th)
    print(f"{scene}: {N_MEMBERS} members, reference has {sh_coeffs.shape[0]} splats, "
          f"{len(train_frames)} training views, {train_secs:.0f}s of training")

    sigma_n = float(np.sqrt(estimate_noise_variance(
        reference, train_frames, train_K, tw, th, str(member_scene),
        background_color=BACKGROUND_COLOR)))
    t = lambda a: torch.tensor(a, dtype=torch.float32, device="cuda")  # noqa: E731

    t0 = time.time()
    data = accumulate_sh_precision_rasterized(
        reference, train_frames, train_K, tw, th, degree, n_probes=N_PROBES,
        seed=SEED, device="cuda", progress_every=0) / (sigma_n ** 2)
    ours_draws = sample_sh_draws(sh_coeffs, data, empirical_band_precision(sh_coeffs, degree),
                                 N_DRAWS, SEED)
    ours_prep = time.time() - t0

    ecax, eval_frames = load_transforms(str(eval_dir / "transforms.json"))
    with Image.open(str(eval_dir / (eval_frames[0][0] + ".png"))) as im:
        w, h = im.size
    K = fov_x_to_intrinsics(ecax, w, h)
    bg = t(BACKGROUND_COLOR)
    theta_hat, op_hat = t(sh_coeffs), t(reference["opacities"])
    names = ["ours (SH posterior)", f"deep ensemble ({N_MEMBERS}x)",
             f"deep ensemble ({N_MEMBERS}x, own mean)"]
    acc = {n: {"sigma": [], "resid": [], "view": [], "secs": 0.0} for n in names}
    acc["ours (SH posterior)"]["secs"] = ours_prep

    for vi, (file_path, c2w) in enumerate(eval_frames):
        viewmat, Ks = t(opencv_viewmat_from_c2w(c2w))[None], t(K)[None]
        gt = np.asarray(Image.open(str(eval_dir / (file_path + ".png"))).convert("RGB"),
                        dtype=np.float32) / 255.0
        obj = object_mask(gt)
        ref_render = render(reference, theta_hat, op_hat, viewmat, Ks, w, h, bg)
        ref_resid = (ref_render - gt)[obj].ravel()

        t0 = time.time()
        ours_std = np.stack([render(reference, ours_draws[s], op_hat, viewmat, Ks, w, h, bg)
                             for s in range(N_DRAWS)], axis=0).std(axis=0)
        acc["ours (SH posterior)"]["secs"] += time.time() - t0

        t0 = time.time()
        member_renders = np.stack([
            render(mem, t(mem["sh_coeffs"]), t(mem["opacities"]), viewmat, Ks, w, h, bg)
            for mem in members], axis=0)
        ens_std = member_renders.std(axis=0)
        ens_mean = member_renders.mean(axis=0)
        ens_secs = time.time() - t0

        for name, sig, res in (
            ("ours (SH posterior)", ours_std, ref_resid),
            (f"deep ensemble ({N_MEMBERS}x)", ens_std, ref_resid),
            (f"deep ensemble ({N_MEMBERS}x, own mean)", ens_std, (ens_mean - gt)[obj].ravel()),
        ):
            acc[name]["sigma"].append(np.maximum(sig[obj].ravel(), 1e-12))
            acc[name]["resid"].append(res)
            acc[name]["view"].append(np.full(res.shape, vi))
        for n in names[1:]:
            acc[n]["secs"] += ens_secs / 2.0

    preds = [Prediction(sigma=np.concatenate(a["sigma"]), residual=np.concatenate(a["resid"]),
                        view_id=np.concatenate(a["view"]), sigma_n=sigma_n,
                        seconds=a["secs"] + (train_secs if "ensemble" in n else 0.0),
                        requires_retraining="ensemble" in n, name=n)
             for n, a in acc.items()]
    rows = compare(preds)
    print(f"\n=== {scene}: deep ensemble vs ours, matched capacity, frozen protocol ===")
    print(format_table(rows))
    out = RESULTS_DIR / f"ensemble_{scene}.json"
    json.dump(rows, open(out, "w"), indent=1)
    print(f"\nwrote {out}")
    return rows


if __name__ == "__main__":
    run(*(sys.argv[1:] or []))
