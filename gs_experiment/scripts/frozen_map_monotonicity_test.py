"""Frozen-map nested-camera-removal monotonicity test for the SH posterior.

The exact linear-algebra invariant: each splat's precision is

    P_i = Lambda + (1/sigma_n^2) sum_p w_{i,p} phi(d_{i,p}) phi(d_{i,p})^T,
    w_{i,p} = sum_q beta_{q,i,p}^2 >= 0

so removing training cameras can only ever DROP positive-semi-definite
terms. `P_i` can only shrink in the Loewner order, hence `Sigma_i = P_i^-1`
and every quadratic form `phi^T Sigma_i phi` can only GROW. A violation is
unambiguous evidence of an implementation bug: it does not depend on
retraining, on the prior, on the kernel, or on any modelling assumption.

Two properties of `accumulate_sh_precision_rasterized` make this a real test
rather than a test of Monte-Carlo noise:

  * `camera_indices` selects cameras from ONE frozen checkpoint -- geometry,
    opacities, stored coefficients and the query camera are never touched,
    so there is no retraining confound of the kind that makes comparing
    independently-trained `gap_*` checkpoints uninterpretable.
  * each camera's Rademacher probes are seeded from its own GLOBAL index, so
    a camera draws the same probes in every subset it appears in. Without
    that, each subset would re-draw every camera's probes and the invariant
    would be swamped by ~30% Monte-Carlo noise.

The check is a RELATIVE one, not machine-precision, and deliberately so:
gsplat's backward accumulates per-splat gradients with float32 atomics and
is not deterministic even for identical inputs and seed (~7.7e-8 relative,
measured run to run on a fixed camera). That noise floor, not float64
round-off, sets the tolerance below.

This test is also what caught the defect it now guards against: the retired
`gpu_sh_directional_uncertainty` path was cross-validated only against
another implementation of the same surrogate, so both agreed and both were
wrong (see results/FINDINGS.md section 0).

Run: .venv-gsplat/bin/python gs_experiment/scripts/frozen_map_monotonicity_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image

from gs_experiment.nerf_transforms import fov_x_to_intrinsics, load_transforms
from gs_experiment.ply_io import read_3dgs_ply
from gs_experiment.rasterized_sh_precision import accumulate_sh_precision_rasterized
from gs_experiment.scripts.render_posterior_ensemble import N_PROBES, SEED, empirical_band_precision
from gs_experiment.scripts.render_reconstruction import LOCAL_RUNS
from gs_experiment.sh_directional_uncertainty import sh_basis

SCENE = "lego"
CHECKPOINT = "wide"
# Strictly nested by construction: each subset is a prefix-preserving
# subsample of the one before it.
N_VIEWS_SWEEP = [100, 50, 25, 8, 3]
NOISE_VAR = 0.000167059  # lego/wide, see estimate_noise_variance
REL_TOL = 1e-6  # set by gsplat's float32-atomic backward, not by float64 round-off


def nested_subsets(n_total: int, counts: list[int]) -> list[list[int]]:
    subsets = [sorted(set(np.linspace(0, n_total - 1, counts[0]).astype(int).tolist()))]
    for k in counts[1:]:
        prev = subsets[-1]
        picks = sorted(set(np.linspace(0, len(prev) - 1, k).astype(int).tolist()))
        subsets.append([prev[i] for i in picks])
    for i in range(1, len(subsets)):
        assert set(subsets[i]) <= set(subsets[i - 1]), "subsets not nested -- test-construction bug"
    return subsets


def run():
    ckpt_dir = LOCAL_RUNS / f"{SCENE}_prepared" / CHECKPOINT
    checkpoint = read_3dgs_ply(str(ckpt_dir / "splats.ply"))
    sh_coeffs, degree = checkpoint["sh_coeffs"], checkpoint["sh_degree"]
    cax, frames = load_transforms(str(ckpt_dir / "transforms.json"))
    with Image.open(str(ckpt_dir / (frames[0][0] + ".png"))) as im:
        width, height = im.size
    K = fov_x_to_intrinsics(cax, width, height)
    print(f"frozen checkpoint: {sh_coeffs.shape[0]} splats, {len(frames)} cameras, sh_degree={degree}")

    band_precision = empirical_band_precision(sh_coeffs, degree)
    prior = np.diag(band_precision[0])  # R channel; the invariant is per-channel
    subsets = nested_subsets(len(frames), N_VIEWS_SWEEP)

    # One fixed query direction, shared by every condition -- the cleanest
    # per-splat instance of the theorem, independent of which pixel a splat
    # happens to be a candidate for.
    d_query = np.array([0.0, 0.0, 1.0])
    phi_q = sh_basis(d_query[None, :], degree)[0]

    quads = []
    for n_views, subset in zip(N_VIEWS_SWEEP, subsets):
        data = accumulate_sh_precision_rasterized(
            checkpoint, frames, K, width, height, degree,
            n_probes=N_PROBES, seed=SEED, device="cuda", progress_every=0, camera_indices=subset,
        ) / NOISE_VAR
        sigma = np.linalg.inv(data + prior)
        quad = np.einsum("i,nij,j->n", phi_q, sigma, phi_q)
        quads.append(quad)
        print(f"{n_views:>4} views: phi^T Sigma phi -- mean {quad.mean():.6g}  "
              f"median {np.median(quad):.6g}  max {quad.max():.6g}")

    stack = np.stack(quads, axis=0)  # (n_conditions, n_splats), conditioning shrinking down the rows
    diffs = np.diff(stack, axis=0)   # must be >= 0 everywhere
    scale = np.maximum(stack[:-1], 1e-30)
    violations = diffs < -REL_TOL * scale
    n_bad = int(violations.any(axis=0).sum())
    print(f"\n--- Monotonicity: phi^T Sigma_i phi non-decreasing as cameras are removed ---")
    print(f"{n_bad}/{stack.shape[1]} splats violate monotonicity (rel tol={REL_TOL})")
    if n_bad:
        print(f"  worst violation magnitude: {diffs.min():.6g} "
              f"(worst relative: {(diffs / scale).min():.3g})")
        print("FAIL: unambiguous implementation bug in accumulate_sh_precision_rasterized.")
    else:
        print("PASS: Loewner monotonicity holds under nested camera removal, "
              "to within the rasterizer's own float32-atomic noise floor.")
    return stack


if __name__ == "__main__":
    run()
