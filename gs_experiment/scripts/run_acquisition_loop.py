"""The SLAM-shaped experiment: an agent chooses where to look next.

Everything else in this project is a correlation scored by a metric, one of
which we chose ourselves. This one is not. Each arm acquires views under its
own uncertainty, and the verdict is a full-capacity retrain on whatever it
assembled, scored on a test set none of them could touch. If the signal is
useful, the retrain is better; if it is not, no correlation rescues it.

Three decisions, from `EPISTEMIC_PLAN.md` section 2, all made before running:

* **Fixed poses from one COLMAP solve.** Re-running COLMAP inside the loop
  would give each arm its own reconstruction, its own frame and its own
  registration failures -- and registration failure is a function of which
  views were chosen. The final number would then mix "reduces render error"
  with "makes COLMAP happy". The arms must differ ONLY in their choices.
* **A trajectory constraint.** At each step the candidates are the `REACH`
  nearest unacquired poses to the agent's current position, so an arm builds
  a path rather than teleporting to a set. A real agent cannot jump across
  the room, and the most uncertain pose is often exactly where it cannot go.
* **A cheap map in the loop, full capacity only at the end.** This separates
  "did the uncertainty choose well" from "was the cheap map any good", and
  it is the cost argument made concrete: a method needing its own training
  run cannot be in this loop at all.

Arms: ours, farthest-point (free, model-free), and random over >=3 seeds --
a discarded run in this project once had a random arm go 11.56 -> 10.84 ->
10.90 dB, which is why one random seed is not a control.

Run: .venv-gsplat/bin/python gs_experiment/scripts/run_acquisition_loop.py SCENE
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party" / "GS-U"))

import numpy as np
import torch
from PIL import Image

from scene.colmap_loader import qvec2rotmat, read_extrinsics_binary, read_intrinsics_binary

from gs_experiment.ply_io import read_3dgs_ply
from gs_experiment.rasterized_sh_precision import accumulate_sh_precision_rasterized
from gs_experiment.scripts.render_posterior_ensemble import empirical_band_precision
from gs_experiment.scripts.build_colmap_gap_scene import write_model
from gs_experiment.scripts.our_uncertainty_for_3dgs_model import draw_one, render

ROOT = Path(__file__).resolve().parents[2]
GSU = ROOT / "third_party" / "GS-U"
PY = str(ROOT / ".venv-gsplat" / "bin" / "python")
N_DRAWS = 8
SEED = 0


# --------------------------------------------------------------------------- scene

def load_scene(src: Path):
    extr = read_extrinsics_binary(src / "sparse" / "0" / "images.bin")
    intr = read_intrinsics_binary(src / "sparse" / "0" / "cameras.bin")
    ids = sorted(extr, key=lambda k: extr[k].name)
    names = [extr[k].name for k in ids]
    centres = np.array([-qvec2rotmat(extr[k].qvec).T @ extr[k].tvec for k in ids])
    return extr, intr, ids, names, centres


def bearings(centres):
    c = centres - centres.mean(axis=0)
    return c / np.linalg.norm(c, axis=1, keepdims=True).clip(1e-12)


def write_train_only(out_dir: Path, src: Path, image_dirs, extr, intr, ids, names, idx):
    """A COLMAP model containing just the acquired views, for the in-loop map.

    `write_model` exists to make 3DGS's positional split land on a chosen test
    set, which requires the training set to be exactly seven times the test
    set. The in-loop map is never evaluated -- it only has to be fitted to what
    the agent holds -- so that constraint does not apply and must not be
    imposed, or the agent could not hold an arbitrary number of views.
    """
    sparse = out_dir / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    with open(sparse / "images.txt", "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n#\n")
        for o, i in enumerate(sorted(idx)):
            im = extr[ids[i]]
            q, tv = im.qvec, im.tvec
            f.write(f"{o + 1} {q[0]} {q[1]} {q[2]} {q[3]} {tv[0]} {tv[1]} {tv[2]} "
                    f"{im.camera_id} {names[i]}\n")
            f.write(" ".join(f"{x:.2f} {y:.2f} {pid}"
                             for (x, y), pid in zip(im.xys, im.point3D_ids)) + "\n")
    with open(sparse / "cameras.txt", "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n#\n")
        for cid, cam in intr.items():
            f.write(f"{cid} {cam.model} {cam.width} {cam.height} "
                    + " ".join(str(x) for x in cam.params) + "\n")
    dst = sparse / "points3D.bin"
    if not dst.exists():
        os.symlink(os.path.abspath(src / "sparse" / "0" / "points3D.bin"), dst)
    for d in image_dirs:
        (out_dir / d).mkdir(parents=True, exist_ok=True)
        for i in sorted(idx):
            link = out_dir / d / names[i]
            if link.exists():
                continue
            target = src / d / names[i]
            if not target.exists():
                cands = list((src / d).glob(Path(names[i]).stem + ".*"))
                if not cands:
                    raise FileNotFoundError(target)
                target = cands[0]
            link.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(os.path.abspath(target), link)


# --------------------------------------------------------------------------- training

def train(src: Path, model: Path, images: str, iterations: int, grad_thresh: float | None,
          log: Path):
    if model.exists():
        shutil.rmtree(model)
    cmd = [PY, "train.py", "-s", str(src), "-i", images, "-m", str(model), "--eval",
           "--iterations", str(iterations), "--save_iterations", str(iterations), "--quiet"]
    if grad_thresh is not None:
        cmd += ["--densify_grad_threshold", str(grad_thresh)]
    env = dict(os.environ, PYTHONPATH=".")
    with open(log, "w") as f:
        rc = subprocess.run(cmd, cwd=GSU, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    ply = model / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    if rc != 0 or not ply.exists():
        raise RuntimeError(f"training failed (rc={rc}); see {log}")
    return ply


# --------------------------------------------------------------------------- scoring

def camera_matrices(intr, cam_id, width, height):
    cam = intr[cam_id]
    sx, sy = width / cam.width, height / cam.height
    if cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
        f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
        fx = fy = f
    else:
        fx, fy, cx, cy = cam.params[0], cam.params[1], cam.params[2], cam.params[3]
    return np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]], dtype=np.float32)


_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


def c2w_opencv(extr, key):
    R = qvec2rotmat(extr[key].qvec)
    m = np.eye(4)
    m[:3, :3] = R.T
    m[:3, 3] = -R.T @ extr[key].tvec
    return m


def our_scores(ply, extr, intr, ids, acquired, candidates, width, height, device="cuda"):
    """Mean posterior spread at each candidate pose, from a map fitted to the
    acquired views only. No ground truth is needed or used -- which is the
    point: the agent has not been to these poses."""
    ck = read_3dgs_ply(str(ply))
    for k in ("positions", "scales", "rotations", "opacities", "sh_coeffs"):
        ck[k] = np.asarray(ck[k], dtype=np.float32)
    degree = ck["sh_degree"]
    K_np = camera_matrices(intr, extr[ids[0]].camera_id, width, height)

    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=device)  # noqa: E731
    ck_t = {"means": t(ck["positions"]), "quats": t(ck["rotations"]),
            "scales": t(ck["scales"]), "opacities": t(ck["opacities"])}
    K = t(K_np)[None]
    bg = t([0.0, 0.0, 0.0])

    # sigma_n from the acquired views' own residuals would need their images;
    # the loop only needs a RANKING over candidates, and a global scale factor
    # cannot change a ranking, so a fixed nominal value is used and recorded.
    sigma_n = 0.05
    frames = [(extr[ids[i]].name, c2w_opencv(extr, ids[i]) @ _FLIP) for i in acquired]
    data = accumulate_sh_precision_rasterized(
        ck, frames, K_np, width, height, degree, n_probes=8, seed=SEED,
        device=device, progress_every=0, out_dtype=np.float32) / np.float32(sigma_n ** 2)
    band = empirical_band_precision(ck["sh_coeffs"], degree)

    theta = t(ck["sh_coeffs"])
    mean = [np.zeros((height, width, 3), np.float64) for _ in candidates]
    m2 = [np.zeros((height, width, 3), np.float64) for _ in candidates]
    for s_i in range(N_DRAWS):
        draw = draw_one(theta, data, band, SEED + s_i, device)
        sh_draw = draw.transpose(1, 2).contiguous()
        for v, i in enumerate(candidates):
            vm = np.linalg.inv(c2w_opencv(extr, ids[i]))
            img = render(ck_t, sh_draw, t(vm)[None], K, width, height,
                         degree, bg).cpu().numpy().astype(np.float64)
            d = img - mean[v]
            mean[v] += d / (s_i + 1)
            m2[v] += d * (img - mean[v])
        del draw, sh_draw
        torch.cuda.empty_cache()
    return np.array([float(np.sqrt(m / max(N_DRAWS - 1, 1)).mean()) for m in m2])


def farthest_scores(bear, acquired, candidates):
    cos = bear[candidates] @ bear[list(acquired)].T
    return np.degrees(np.arccos(np.clip(cos.max(axis=1), -1, 1)))


# --------------------------------------------------------------------------- loop

def reachable(centres, current, pool, reach):
    pool = np.asarray(sorted(pool))
    d = np.linalg.norm(centres[pool] - centres[current], axis=1)
    return pool[np.argsort(d)[:reach]]


def run_arm(arm, seed, scene, src, images, extr, intr, ids, names, centres, bear,
            test_idx, seed_idx, pool, out_root, rounds, batch, reach,
            cheap_iters, cheap_thresh, image_dirs):
    rng = np.random.default_rng(seed)
    acquired = list(seed_idx)
    current = acquired[-1]
    work = out_root / f"{arm}{'' if seed is None else f'_s{seed}'}"
    work.mkdir(parents=True, exist_ok=True)
    remaining = set(pool)
    trace = []

    im0 = Image.open(next((src / images).glob("*")))
    width, height = im0.size
    ply = None

    for r in range(rounds):
        picked = []
        ply = None                      # refit once per round, on what is held so far
        for _ in range(batch):
            if not remaining:
                break
            cands = reachable(centres, current, remaining, reach)
            if arm == "random":
                choice = int(rng.choice(cands))
            elif arm == "farthest":
                choice = int(cands[np.argmax(farthest_scores(bear, acquired, cands))])
            else:                                          # ours -- needs a map
                if ply is None:
                    scene_dir = work / f"round{r}_src"
                    scene_dir.mkdir(parents=True, exist_ok=True)
                    write_train_only(scene_dir, src, image_dirs, extr, intr, ids,
                                     names, sorted(acquired))
                    ply = train(scene_dir, work / f"round{r}_map", images,
                                cheap_iters, cheap_thresh, work / f"round{r}_train.log")
                s = our_scores(ply, extr, intr, ids, acquired, list(cands), width, height)
                choice = int(cands[int(np.argmax(s))])
            picked.append(choice)
            acquired.append(choice)
            remaining.discard(choice)
            current = choice
        trace.append({"round": r, "picked": [names[i] for i in picked],
                      "n_acquired": len(acquired)})
        print(f"  [{arm}{'' if seed is None else f'/s{seed}'}] round {r}: "
              f"+{len(picked)} -> {len(acquired)} views")
        for p in work.glob(f"round{r}_*"):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
    return acquired, trace


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--source", required=True)
    ap.add_argument("-i", "--images", default="images")
    ap.add_argument("--image_dirs", default="images")
    ap.add_argument("--seed_fraction", type=float, default=0.35)
    ap.add_argument("--n_test", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--reach", type=int, default=20)
    ap.add_argument("--cheap_iters", type=int, default=3000)
    ap.add_argument("--cheap_thresh", type=float, default=0.0008)
    ap.add_argument("--full_iters", type=int, default=30000)
    ap.add_argument("--random_seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    src = Path(a.source)
    out_root = Path(a.out or ROOT / "gs_experiment" / "local_runs" / f"acq_{a.scene}")
    out_root.mkdir(parents=True, exist_ok=True)
    image_dirs = tuple(a.image_dirs.split(","))

    extr, intr, ids, names, centres = load_scene(src)
    bear = bearings(centres)
    n = len(ids)

    # Fixed test set, spread uniformly over the capture, acquirable by no arm.
    #
    # Its SIZE is constrained, not chosen. write_model controls 3DGS's split by
    # naming images so that sorted(names)[::8] is exactly the test set, which
    # needs seven training views between consecutive test views. The smallest
    # training set here is the seed prefix, so n_test <= n_seed // 7, and
    # asking for more silently produces a split that is not the intended one.
    # The final models' sizes are dictated by their loader, not chosen by us.
    # write_model controls 3DGS's split by naming images so sorted(names)[::8]
    # is exactly the test set, which holds only when the written model has
    # exactly 8x as many images as test views -- not merely at least. The
    # budget is fixed, so the seed prefix is what has to give.
    n_test = a.n_test
    budget = a.rounds * a.batch
    n_seed = 7 * n_test - budget
    if n_seed < 8:
        raise SystemExit(f"budget {budget} leaves a seed of {n_seed} at n_test={n_test}; "
                         f"raise --n_test or lower the budget")
    test_idx = [int(x) for x in np.linspace(0, n - 1, n_test).round()]
    rest = [i for i in range(n) if i not in set(test_idx)]
    if len(rest) < n_seed + budget:
        raise SystemExit(f"{a.scene}: {len(rest)} acquirable views, need {n_seed + budget}")
    seed_idx = rest[:n_seed]                # a contiguous prefix: where the agent has been
    pool = rest[n_seed:]
    print(f"{a.scene}: {n} views -> {len(test_idx)} test, {len(seed_idx)} seed, "
          f"{len(pool)} pool; budget {a.rounds * a.batch} over {a.rounds} rounds")

    arms = [("ours", None), ("farthest", None)] + [("random", s) for s in a.random_seeds]
    results = {}
    for arm, seed in arms:
        tag = f"{arm}{'' if seed is None else f'_s{seed}'}"
        t0 = time.time()
        acquired, trace = run_arm(arm, seed, a.scene, src, a.images, extr, intr, ids, names,
                                  centres, bear, test_idx, seed_idx, pool, out_root,
                                  a.rounds, a.batch, a.reach, a.cheap_iters, a.cheap_thresh,
                                  image_dirs)
        results[tag] = {"acquired": [names[i] for i in acquired], "trace": trace,
                        "loop_seconds": time.time() - t0}
        json.dump(results, open(out_root / "acquisition.json", "w"), indent=1)

    # A seed-only reference would need its own test split under the naming
    # constraint above, and a different test set is not a comparison. The
    # arms are compared against each other on identical views instead.

    print("\n=== full-capacity retrains ===")
    for tag, rec in results.items():
        idx = sorted(names.index(x) for x in rec["acquired"])
        final_src = out_root / f"final_{tag}_src"
        if not (final_src / "sparse" / "0" / "images.txt").exists():
            final_src.mkdir(parents=True, exist_ok=True)
            write_model(final_src, src, image_dirs, extr, intr, ids, names, list(test_idx), idx)
        model = out_root / f"final_{tag}"
        try:
            train(final_src, model, a.images, a.full_iters, None,
                  out_root / f"final_{tag}.log")
        except RuntimeError as e:
            print(f"  {tag}: {e}")
            continue
        env = dict(os.environ, PYTHONPATH=".")
        subprocess.run([PY, "render.py", "-m", str(model), "--skip_train", "--quiet"],
                       cwd=GSU, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        subprocess.run([PY, "metrics.py", "-m", str(model)], cwd=GSU, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        rp = model / "results.json"
        if rp.exists():
            rec["final"] = json.load(open(rp))
            print(f"  {tag}: {rec['final']}")
        json.dump(results, open(out_root / "acquisition.json", "w"), indent=1)
        shutil.rmtree(model / "point_cloud", ignore_errors=True)

    json.dump(results, open(out_root / "acquisition.json", "w"), indent=1)
    print(f"\nwrote {out_root / 'acquisition.json'}")
