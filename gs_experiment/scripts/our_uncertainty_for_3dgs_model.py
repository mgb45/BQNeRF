"""Run our posterior-ensemble uncertainty on a stock 3DGS model directory and
write the result in U-3DGS's own output layout, so that THEIR
`uncertainty_metrics.py` scores it.

This is the cleanest available answer to "you reimplemented prior work and
invented your own protocol". Nothing of theirs is reimplemented here: their
`train.py` produces the map, their `train_errors.py` produces their
uncertainty, their `uncertainty_metrics.py` produces every number, and this
script only adds one more `error_masks_*` folder for our sigma. Our method can
do this at all because it is post-hoc -- it consumes a finished checkpoint and
never touches training.

Conventions that are easy to get wrong, all verified against their released
code and their own renders on bonsai:

* `cameras.json` lists **test cameras first, then train** (`scene/__init__.py`
  extends `camlist` with `test_cameras` before `train_cameras`). Neither
  "every 8th" nor "the tail" is right, and both fail quietly -- every-8th
  scored 12.5 dB while still looking plausible on view 0, because index 0
  happens to be a real test camera.
* `cameras.json` stores COLMAP's FULL-resolution intrinsics while training
  runs on a downscaled folder, so `fx` must be scaled by `render_width /
  cameras.json width`.
* `rotation` is the camera-to-world rotation and `position` the camera
  centre, so the view matrix is `inv([R | pos])`.
* COLMAP scenes render on a BLACK background.
* `accumulate_sh_precision_rasterized` takes c2w matrices and converts them
  with `nerf_transforms.opencv_viewmat_from_c2w`, which applies an
  OpenGL->OpenCV axis flip because NeRF-Synthetic's c2w is OpenGL. COLMAP
  rotations are ALREADY OpenCV, so a 3DGS c2w must be pre-multiplied by the
  same flip to survive that conversion unchanged (the flip is its own
  inverse). Getting this wrong does not crash: it points every accumulation
  camera backwards, so 87% of splats record zero observations, the posterior
  collapses to the prior everywhere, and the only symptom is a suspiciously
  flat sigma.

With those right, gsplat reproduces their renders to 0.03 dB PSNR (57.5 dB
agreement between the two rasterizers) on their own bonsai checkpoint, which
is what licenses using our renderer for our row of their table.

Run: .venv-gsplat/bin/python gs_experiment/scripts/our_uncertainty_for_3dgs_model.py \
        -m <model_dir> -s <colmap_source> -i images_2 [--iteration 30000]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from PIL import Image

from gs_experiment.ply_io import read_3dgs_ply
from gs_experiment.rasterized_sh_precision import (
    accumulate_sh_precision_rasterized,
    fit_band_precision_evidence,
)
from gs_experiment.scripts.render_posterior_ensemble import empirical_band_precision, sample_sh_draws

N_DRAWS = 16
N_PROBES = 32
SEED = 0


def load_cameras(model_dir: Path, n_test: int):
    """(test_cams, train_cams) -- test first, per their Scene writer."""
    cams = json.load(open(model_dir / "cameras.json"))
    return cams[:n_test], cams[n_test:]


# OpenGL<->OpenCV axis flip; its own inverse.
_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


def c2w_opencv(cam) -> np.ndarray:
    """Camera-to-world in OpenCV convention, straight from cameras.json."""
    c2w = np.eye(4)
    c2w[:3, :3] = np.array(cam["rotation"])
    c2w[:3, 3] = np.array(cam["position"])
    return c2w


def view_matrix(cam) -> np.ndarray:
    return np.linalg.inv(c2w_opencv(cam))


def c2w_for_helper(cam) -> np.ndarray:
    """What `accumulate_sh_precision_rasterized` must be handed so that its
    internal `opencv_viewmat_from_c2w` (which flips OpenGL->OpenCV) yields the
    correct OpenCV view matrix. See the module docstring."""
    return c2w_opencv(cam) @ _FLIP


def intrinsics(cam, width, height) -> np.ndarray:
    s = width / cam["width"]
    return np.array([[cam["fx"] * s, 0.0, width / 2.0],
                     [0.0, cam["fy"] * s, height / 2.0], [0.0, 0.0, 1.0]])


def render(ck_t, sh, viewmat, K, width, height, degree, bg):
    import gsplat

    with torch.no_grad():
        img, _, _ = gsplat.rasterization(
            ck_t["means"], ck_t["quats"], ck_t["scales"], ck_t["opacities"], sh,
            viewmat, K, width=width, height=height, sh_degree=degree, backgrounds=bg)
    return img[0].clamp(0, 1)


def load_gt(source: Path, images_dir: str, name: str, width, height) -> np.ndarray:
    path = source / images_dir / name
    if not path.exists():                       # COLMAP names may carry a different suffix
        stem = Path(name).stem
        cands = list((source / images_dir).glob(stem + ".*"))
        if not cands:
            raise FileNotFoundError(path)
        path = cands[0]
    im = Image.open(path).convert("RGB")
    if im.size != (width, height):
        im = im.resize((width, height), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0


def run(model_dir, source, images_dir, iteration=30000, out_name="error_masks_ours",
        split="test", renders_folder="renders", device="cuda", prior="evidence"):
    model_dir, source = Path(model_dir), Path(source)
    ply = model_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    ck = read_3dgs_ply(str(ply))
    degree = ck["sh_degree"]

    # Their own saved originals define the test set, its order and its size.
    orig_dir = model_dir / renders_folder / split / "original"
    originals = sorted(orig_dir.glob("*.npy"))
    if not originals:
        raise FileNotFoundError(
            f"{orig_dir} is empty -- run their train_errors.py first; it writes the "
            "original/render/error_masks .npy files this script matches.")
    gt0 = np.load(originals[0])                       # (3, H, W)
    height, width = gt0.shape[1], gt0.shape[2]
    test_cams, train_cams = load_cameras(model_dir, len(originals))
    print(f"{model_dir.name}: {ck['positions'].shape[0]} splats, {len(train_cams)} train / "
          f"{len(test_cams)} test cams, {width}x{height}, sh_degree {degree}")

    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=device)  # noqa: E731
    ck_t = {"means": t(ck["positions"]), "quats": t(ck["rotations"]),
            "scales": t(ck["scales"]), "opacities": t(ck["opacities"])}
    sh_hat = t(ck["sh_coeffs"]).transpose(1, 2).contiguous()
    bg = t([0.0, 0.0, 0.0])                            # COLMAP scenes: black

    fx = {round(c["fx"], 3) for c in test_cams + train_cams}
    if len(fx) != 1:
        raise ValueError(f"per-camera intrinsics differ ({len(fx)} distinct fx); "
                         "this script assumes one COLMAP camera model per scene")
    K = t(intrinsics(test_cams[0], width, height))[None]

    # sigma_n from TRAINING residuals, subsampled -- these scenes have 200+
    # training views and the estimate is converged long before all of them.
    picks = np.linspace(0, len(train_cams) - 1, min(15, len(train_cams))).astype(int)
    sq = []
    for i in picks:
        cam = train_cams[i]
        vm = t(view_matrix(cam))[None]
        rec = render(ck_t, sh_hat, vm, K, width, height, degree, bg).cpu().numpy()
        gt = load_gt(source, images_dir, cam["img_name"], width, height)
        sq.append(float(np.mean((rec - gt) ** 2)))
    sigma_n = float(np.sqrt(np.mean(sq)))
    print(f"  sigma_n = {sigma_n:.5f} (from {len(picks)} training views)")

    t0 = time.time()
    frames = [(c["img_name"], c2w_for_helper(c)) for c in train_cams]
    data = accumulate_sh_precision_rasterized(
        ck, frames, np.array(intrinsics(test_cams[0], width, height)), width, height, degree,
        n_probes=N_PROBES, seed=SEED, device=device, progress_every=0) / (sigma_n ** 2)
    if prior == "evidence":
        band = fit_band_precision_evidence(ck["sh_coeffs"], data, device=device, verbose=True)
    elif prior == "population":
        band = empirical_band_precision(ck["sh_coeffs"], degree)
    else:
        raise ValueError(f"unknown prior mode {prior!r}")
    draws = sample_sh_draws(ck["sh_coeffs"], data, band, N_DRAWS, SEED)
    fit_secs = time.time() - t0
    print(f"  posterior fitted in {fit_secs:.1f}s")

    out_dir = model_dir / renders_folder / split / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, cam in enumerate(test_cams):
        vm = t(view_matrix(cam))[None]
        ens = np.stack([render(ck_t, draws[s].transpose(1, 2).contiguous(), vm, K,
                               width, height, degree, bg).cpu().numpy()
                        for s in range(N_DRAWS)], axis=0)
        # Their maps are one scalar per pixel, so collapse CHANNELS as they do.
        # gsplat returns (H, W, 3) while their images are (3, H, W), so the
        # channel axis is -1 here, not 0 -- averaging axis 0 silently produces
        # a (W, 3) array that only fails downstream on a shape check.
        np.save(out_dir / f"{i:05d}.npy", ens.std(axis=0).mean(axis=-1))
    infer_secs = time.time() - t0
    print(f"  wrote {len(test_cams)} maps to {out_dir} ({infer_secs:.1f}s)")
    json.dump({"sigma_n": sigma_n, "fit_seconds": fit_secs, "infer_seconds": infer_secs,
               "n_draws": N_DRAWS, "n_test": len(test_cams), "n_train": len(train_cams)},
              open(model_dir / renders_folder / split / f"{out_name}_meta.json", "w"), indent=1)
    return fit_secs, infer_secs


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-m", "--model_dir", required=True)
    ap.add_argument("-s", "--source", required=True)
    ap.add_argument("-i", "--images_dir", default="images_2")
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--out_name", default="error_masks_ours")
    ap.add_argument("--prior", default="evidence", choices=["evidence", "population"],
                    help="evidence = MacKay type-II ML fit; population = 1/Var_i[theta] rule")
    a = ap.parse_args()
    run(a.model_dir, a.source, a.images_dir, a.iteration, a.out_name, prior=a.prior)
