"""Tests for rasterized_sh_precision.py.

The defect these guard against (results/FINDINGS.md section 0) survived a
full cross-validation suite because that suite compared a batched
implementation of a surrogate against a scalar implementation of the SAME
surrogate: both agreed, both were wrong by a median factor of 2.1e18. So
the tests here deliberately validate against the **real rasterizer** --
the only external reference that exists for a quantity defined as "what the
renderer does" -- by rendering one-hot per-splat features to recover each
beta_{q,i} exactly, and checking the probe estimator reproduces it.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("rasterized_sh_precision needs a real GPU", allow_module_level=True)
gsplat = pytest.importorskip("gsplat")

from gs_experiment.rasterized_sh_precision import (  # noqa: E402
    _sh_basis_torch,
    accumulate_sh_precision_rasterized,
    probe_squared_footprint_weights,
)
from gs_experiment.sh_directional_uncertainty import sh_basis  # noqa: E402

DEV = "cuda"
RES = 48
N_SPLATS = 12


def _tiny_scene(seed=0):
    """A small real splat scene in front of a camera at the origin looking
    down +z (OpenCV convention), with splats deliberately overlapping in
    depth so transmittance actually matters."""
    rng = np.random.default_rng(seed)
    means = np.stack([
        rng.uniform(-0.5, 0.5, N_SPLATS),
        rng.uniform(-0.5, 0.5, N_SPLATS),
        rng.uniform(2.5, 4.0, N_SPLATS),
    ], axis=1)
    quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (N_SPLATS, 1))
    scales = np.full((N_SPLATS, 3), 0.12)
    opacities = rng.uniform(0.3, 0.9, N_SPLATS)
    t = lambda a, d=torch.float32: torch.tensor(a, dtype=d, device=DEV)  # noqa: E731
    f = 60.0
    Ks = t([[[f, 0.0, RES / 2], [0.0, f, RES / 2], [0.0, 0.0, 1.0]]])
    viewmat = t(np.eye(4)[None])
    return t(means), t(quats), t(scales), t(opacities), viewmat, Ks


def _exact_beta(means, quats, scales, opacities, viewmat, Ks):
    """beta_{q,i} for EVERY pixel and splat, exactly, by rendering a one-hot
    feature per splat: gsplat renders I_c(q) = sum_i beta_{q,i} c_i, so with
    c = e_i the c-th channel IS beta_{.,i}."""
    onehot = torch.eye(means.shape[0], dtype=torch.float32, device=DEV)
    with torch.no_grad():
        img, _, _ = gsplat.rasterization(means, quats, scales, opacities, onehot,
                                         viewmat, Ks, width=RES, height=RES, sh_degree=None)
    return img[0].double()  # (H, W, N), [..., i] == beta_{q,i}


def test_probe_estimator_matches_exact_rasterizer_beta():
    means, quats, scales, opacities, viewmat, Ks = _tiny_scene()
    beta = _exact_beta(means, quats, scales, opacities, viewmat, Ks)
    exact_sum_sq = (beta**2).sum(dim=(0, 1)).cpu().numpy()
    exact_sum = beta.sum(dim=(0, 1)).cpu().numpy()

    gen = torch.Generator(device=DEV).manual_seed(0)
    # Average many independent probe batches: the estimator is unbiased, so
    # the only disagreement allowed is Monte-Carlo error.
    acc = np.zeros(N_SPLATS)
    n_batches = 40
    for _ in range(n_batches):
        s2, s1 = probe_squared_footprint_weights(means, quats, scales, opacities, viewmat, Ks,
                                                 RES, RES, n_probes=32, generator=gen)
        acc += s2.cpu().numpy()
    est_sum_sq = acc / n_batches

    assert np.all(exact_sum_sq > 0), "test scene degenerate -- no splat contributes"
    # The estimator is unbiased but stochastic, so the honest test is a
    # z-score against its OWN standard error, not a fixed percentage. For
    # Rademacher probes, Var[(sum_q beta_q r_q)^2] <= 2 (sum_q beta_q^2)^2,
    # so the relative standard error over n probes is at most sqrt(2/n).
    # A fixed 5% tolerance would be a flake generator: at 1280 probes the
    # 1-sigma relative error is already 3.7%.
    n_total = 32 * n_batches
    se = exact_sum_sq * np.sqrt(2.0 / n_total)
    z = np.abs(est_sum_sq - exact_sum_sq) / se
    assert z.max() < 5.0, f"probe estimator inconsistent with exact beta: max |z| = {z.max():.2f}"

    # The plain footprint total rides an all-ones channel, so it is EXACT.
    np.testing.assert_allclose(s1.cpu().numpy(), exact_sum, rtol=1e-5, atol=1e-8)


def test_probe_estimator_is_not_the_centre_pixel_value():
    """Regression guard for defect (1) of FINDINGS section 0: the accumulated
    weight must be the footprint SUM, which for a real multi-pixel splat is
    strictly larger than any single pixel's beta^2."""
    means, quats, scales, opacities, viewmat, Ks = _tiny_scene()
    beta = _exact_beta(means, quats, scales, opacities, viewmat, Ks)
    exact_sum_sq = (beta**2).sum(dim=(0, 1))
    per_pixel_max = (beta**2).amax(dim=(0, 1))
    covers_many_pixels = (beta > 1e-6).sum(dim=(0, 1)) > 4
    assert bool(covers_many_pixels.any()), "test scene degenerate -- no multi-pixel splat"
    assert torch.all(exact_sum_sq[covers_many_pixels] > 1.5 * per_pixel_max[covers_many_pixels])


def test_nested_camera_removal_is_exactly_monotone():
    """Removing cameras can only drop PSD terms from D_i, so D_i shrinks in
    the Loewner order. Exact (not statistical) because each camera's probes
    are seeded from its own global index -- see the module docstring."""
    rng = np.random.default_rng(1)
    n_cams = 6
    frames = []
    for i in range(n_cams):
        ang = 2 * np.pi * i / n_cams
        c2w = np.eye(4)
        c2w[:3, 3] = [2.5 * np.sin(ang), 0.3 * np.cos(ang), 0.0]
        frames.append((f"cam{i}", c2w))
    checkpoint = {
        "positions": rng.uniform(-0.4, 0.4, (N_SPLATS, 3)),
        "rotations": np.tile([1.0, 0.0, 0.0, 0.0], (N_SPLATS, 1)),
        "scales": np.full((N_SPLATS, 3), 0.15),
        "opacities": rng.uniform(0.3, 0.9, N_SPLATS),
        "sh_degree": 2,
    }
    f = 60.0
    K = np.array([[f, 0.0, RES / 2], [0.0, f, RES / 2], [0.0, 0.0, 1.0]])

    subsets = [list(range(n_cams)), [0, 1, 2, 3], [0, 2], [0]]
    phi = sh_basis(np.array([[0.0, 0.0, 1.0]]), 2)[0]
    prior = np.eye(phi.shape[0])
    quads = []
    for subset in subsets:
        D = accumulate_sh_precision_rasterized(
            checkpoint, frames, K, RES, RES, 2, n_probes=16, seed=0, device=DEV,
            progress_every=0, camera_indices=subset,
        )
        sigma = np.linalg.inv(D + prior)
        quads.append(np.einsum("i,nij,j->n", phi, sigma, phi))
    stack = np.stack(quads, axis=0)
    diffs = np.diff(stack, axis=0)
    # Relative, not machine-precision: gsplat's backward uses float32 atomics
    # and is non-deterministic at ~7.7e-8 relative even for a fixed seed.
    rel = diffs / np.maximum(stack[:-1], 1e-30)
    assert rel.min() > -1e-6, f"Loewner monotonicity violated by {rel.min():.3g} (relative)"


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_sh_basis_torch_matches_numpy_reference(degree):
    rng = np.random.default_rng(0)
    d = rng.normal(size=(64, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    ref = sh_basis(d, degree)
    got = _sh_basis_torch(torch.tensor(d, dtype=torch.float64, device=DEV), degree).cpu().numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)
