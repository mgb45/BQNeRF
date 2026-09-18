"""Next-best-view selection driven by the posterior-ensemble uncertainty.

This is the "so what". FINDINGS section 8 established that per-VIEW mean
uncertainty tracks per-view held-out error at Spearman 0.97 when the error is
epistemic -- and a per-view aggregate is exactly what next-best-view
selection consumes. So: does choosing the next training views by uncertainty
actually reduce held-out error faster than not?

Protocol. Start from a small seed set of real training views, and repeatedly
(a) train a checkpoint on the current set, (b) score every held-out
CANDIDATE pose by rendering the current model there and averaging the
posterior-ensemble std over its foreground, (c) add the top-k. Held-out PSNR
is measured on the scene's own 30-view eval split, which is never selected
from and never trained on.

Three strategies, identical budget and identical training recipe:

  * **uncertainty**  -- the method under test.
  * **farthest-point** -- the honest baseline. "Pick the camera furthest
    from the ones you already have" is the obvious cheap heuristic for view
    selection, needs no model at all, and captures most of what naive view
    planning does. If uncertainty cannot beat this, it is not adding value,
    and comparing only against random would hide that.
  * **random** -- several seeds, for the error bar.

Candidate scoring never looks at a candidate's IMAGE, only its pose: the
model is rendered there and the ensemble spread is measured. Using the image
would be using the very data acquisition is supposed to be deciding whether
to collect.

Run: .venv-gsplat/bin/python gs_experiment/scripts/run_next_best_view.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from gs_experiment.nerf_transforms import fov_x_to_intrinsics, load_transforms, opencv_viewmat_from_c2w
from gs_experiment.ply_io import read_3dgs_ply
from gs_experiment.rasterized_sh_precision import accumulate_sh_precision_rasterized, estimate_noise_variance
from gs_experiment.scripts.render_posterior_ensemble import (
    BACKGROUND_COLOR, N_PROBES, SEED, empirical_band_precision, sample_sh_draws,
)
from gs_experiment.scripts.render_reconstruction import LOCAL_RUNS, RESULTS_DIR

SCENE = "lego"
POOL_DIR = LOCAL_RUNS / f"{SCENE}_prepared" / "wide"
EVAL_DIR = LOCAL_RUNS / f"{SCENE}_prepared" / "eval"
WORK = LOCAL_RUNS / "nbv_out" / "uncertainty_vs_baselines"
N_SEED_VIEWS = 6
N_PER_ROUND = 6
N_ROUNDS = 4
N_DRAWS = 8
# Overrides on the project's OWN canonical recipe (DEFAULT_TRAIN_KWARGS).
# Do NOT drive training through the CLI here: its defaults are
# background_color=(0.05,0.05,0.05) and init_scale=None, and it exposes no
# flag for either, so a CLI-trained checkpoint is fit against a dark
# background and then scored against white-composited NeRF-Synthetic ground
# truth. That reads every uncovered pixel as a huge error and cost ~17 dB of
# held-out PSNR (11.6 dB against 28.4 dB for the identical budget) -- the
# exact failure train_minimal_gsplat.py's own comments warn about. Measured
# on 30 lego views: this config 28.4 dB / 80 s, doubling it to
# 12k iters + 200k splats buys 0.6 dB for 2x the time.
TRAIN_OVERRIDES = dict(n_iters=6000, max_splats=100000, densify_end=3000, log_every=10**9)
RANDOM_SEEDS = [0, 1, 2]


def build_scene_dir(dst: Path, frames_all, cam_angle_x, keep_idx):
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    frames = [frames_all[i] for i in keep_idx]
    json.dump({"camera_angle_x": cam_angle_x,
               "frames": [{"file_path": fp, "transform_matrix": np.asarray(c2w).tolist()}
                          for fp, c2w in frames]}, open(dst / "transforms.json", "w"))
    for fp, _ in frames:
        rel = fp + ".png"
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        os.symlink(os.path.abspath(POOL_DIR / rel), dst / rel)
    return dst


def train_and_eval(scene_dir: Path):
    from gs_experiment.scripts.render_reconstruction import render_views
    from gs_experiment.scripts.train_minimal_gsplat import DEFAULT_TRAIN_KWARGS, train

    kwargs = dict(DEFAULT_TRAIN_KWARGS)
    kwargs.update(TRAIN_OVERRIDES)
    assert kwargs["background_color"] == BACKGROUND_COLOR, (
        "training and evaluation backgrounds must match, or held-out PSNR is meaningless")
    train(str(scene_dir), str(scene_dir / "splats.ply"), **kwargs)
    _, eval_frames = load_transforms(str(EVAL_DIR / "transforms.json"))
    results, _ = render_views(str(EVAL_DIR), list(range(len(eval_frames))),
                              checkpoint_dir=str(scene_dir), background_color=BACKGROUND_COLOR)
    psnrs = [-10.0 * np.log10(max(float(np.mean((gt - rec) ** 2)), 1e-10)) for _, gt, rec in results]
    return float(np.mean(psnrs))


def score_candidates(scene_dir: Path, frames_all, cam_angle_x, candidate_idx):
    """Mean posterior-ensemble std over each candidate view's foreground.
    Uses the candidate's POSE only -- never its image."""
    import gsplat

    checkpoint = read_3dgs_ply(str(scene_dir / "splats.ply"))
    sh_coeffs, degree = checkpoint["sh_coeffs"], checkpoint["sh_degree"]
    _, train_frames = load_transforms(str(scene_dir / "transforms.json"))
    with Image.open(str(scene_dir / (train_frames[0][0] + ".png"))) as im:
        w, h = im.size
    K = fov_x_to_intrinsics(cam_angle_x, w, h)
    noise_var = estimate_noise_variance(checkpoint, train_frames, K, w, h,
                                        str(scene_dir), background_color=BACKGROUND_COLOR)
    data = accumulate_sh_precision_rasterized(
        checkpoint, train_frames, K, w, h, degree, n_probes=N_PROBES, seed=SEED,
        device="cuda", progress_every=0) / noise_var
    draws = sample_sh_draws(sh_coeffs, data, empirical_band_precision(sh_coeffs, degree),
                            N_DRAWS, SEED)

    t = lambda a: torch.tensor(a, dtype=torch.float32, device="cuda")  # noqa: E731
    means, quats = t(checkpoint["positions"]), t(checkpoint["rotations"])
    scales, ops = t(checkpoint["scales"]), t(checkpoint["opacities"])
    Ks, bg = t(K)[None], t(BACKGROUND_COLOR)

    scores = {}
    for ci in candidate_idx:
        viewmat = t(opencv_viewmat_from_c2w(frames_all[ci][1]))[None]
        ens, alpha = [], None
        with torch.no_grad():
            for s in range(N_DRAWS):
                img, a, _ = gsplat.rasterization(
                    means, quats, scales, ops, draws[s].transpose(1, 2).contiguous(),
                    viewmat, Ks, width=w, height=h, sh_degree=degree, backgrounds=bg)
                ens.append(img[0].clamp(0, 1).cpu().numpy())
                alpha = a[0, :, :, 0].cpu().numpy()
        std = np.stack(ens, 0).std(axis=0).mean(axis=2)
        fg = alpha > 0.1
        scores[ci] = float(std[fg].mean()) if fg.any() else float(std.mean())
    return scores


def farthest_point(frames_all, chosen, candidates, k):
    """Greedy farthest-point on camera centres -- the model-free baseline."""
    centres = np.array([np.asarray(c2w)[:3, 3] for _, c2w in frames_all])
    sel, pool = list(chosen), list(candidates)
    out = []
    for _ in range(k):
        d = np.array([min(np.linalg.norm(centres[c] - centres[s]) for s in sel) for c in pool])
        pick = pool[int(np.argmax(d))]
        out.append(pick); sel.append(pick); pool.remove(pick)
    return out


def run_strategy(name, frames_all, cam_angle_x, seed_views, picker, tag):
    chosen = list(seed_views)
    curve = []
    for rnd in range(N_ROUNDS + 1):
        sd = build_scene_dir(WORK / f"{tag}_r{rnd}", frames_all, cam_angle_x, chosen)
        t0 = time.time()
        psnr = train_and_eval(sd)
        curve.append((len(chosen), psnr))
        print(f"  [{name}] {len(chosen):>3} views -> held-out PSNR {psnr:.3f} dB  ({time.time() - t0:.0f}s)")
        if rnd == N_ROUNDS:
            break
        candidates = [i for i in range(len(frames_all)) if i not in chosen]
        chosen = chosen + picker(sd, chosen, candidates)
    return curve


def run():
    WORK.mkdir(parents=True, exist_ok=True)
    cam_angle_x, frames_all = load_transforms(str(POOL_DIR / "transforms.json"))
    seed_views = sorted(set(np.linspace(0, len(frames_all) - 1, N_SEED_VIEWS).astype(int).tolist()))
    print(f"pool {len(frames_all)} views, seed {seed_views}, +{N_PER_ROUND}/round x {N_ROUNDS} rounds")

    curves = {}
    curves["uncertainty"] = run_strategy(
        "uncertainty", frames_all, cam_angle_x, seed_views,
        lambda sd, chosen, cand: [c for c, _ in sorted(
            score_candidates(sd, frames_all, cam_angle_x, cand).items(),
            key=lambda kv: -kv[1])[:N_PER_ROUND]],
        "unc")
    curves["farthest-point"] = run_strategy(
        "farthest-point", frames_all, cam_angle_x, seed_views,
        lambda sd, chosen, cand: farthest_point(frames_all, chosen, cand, N_PER_ROUND),
        "fps")
    for rs in RANDOM_SEEDS:
        rng = np.random.default_rng(rs)
        curves[f"random-{rs}"] = run_strategy(
            f"random-{rs}", frames_all, cam_angle_x, seed_views,
            lambda sd, chosen, cand, rng=rng: list(rng.choice(cand, N_PER_ROUND, replace=False)),
            f"rnd{rs}")

    rand = np.array([[p for _, p in curves[f"random-{rs}"]] for rs in RANDOM_SEEDS])
    xs = [n for n, _ in curves["uncertainty"]]
    print(f"\n{'views':>6}{'uncertainty':>13}{'farthest-pt':>13}{'random mean':>13}{'random sd':>11}")
    for i, n in enumerate(xs):
        print(f"{n:>6}{curves['uncertainty'][i][1]:>13.3f}{curves['farthest-point'][i][1]:>13.3f}"
              f"{rand[:, i].mean():>13.3f}{rand[:, i].std():>11.3f}")
    u = np.array([p for _, p in curves["uncertainty"]])
    f = np.array([p for _, p in curves["farthest-point"]])
    print(f"\nmean over acquisition rounds (excluding the shared seed state):")
    print(f"  uncertainty   - random       : {np.mean(u[1:] - rand.mean(0)[1:]):+.3f} dB")
    print(f"  uncertainty   - farthest-pt  : {np.mean(u[1:] - f[1:]):+.3f} dB")
    print(f"  farthest-pt   - random       : {np.mean(f[1:] - rand.mean(0)[1:]):+.3f} dB")

    json.dump({k: v for k, v in curves.items()}, open(WORK / "curves.json", "w"), indent=1)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.errorbar(xs, rand.mean(0), yerr=rand.std(0), marker="s", capsize=3,
                label=f"random (n={len(RANDOM_SEEDS)})", color="gray")
    ax.plot(xs, f, marker="^", label="farthest-point", color="tab:orange")
    ax.plot(xs, u, marker="o", label="uncertainty (ours)", color="tab:blue", lw=2)
    ax.set_xlabel("training views"); ax.set_ylabel("held-out PSNR (dB)")
    ax.set_title(f"{SCENE}: next-best-view selection", fontsize=13)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    path = RESULTS_DIR / "next_best_view.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    run()
