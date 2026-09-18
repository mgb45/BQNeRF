"""Tests for coupled_sh_posterior.py.

The sampler never forms `A` and never inverts it, so the test does both --
on a scene small enough to make that possible -- and checks that the draws
it produces really are distributed as `N(0, A^-1)`. This is the only
validation that actually pins the construction down: a matvec that is
subtly wrong, a `b` whose covariance is not `A`, or a CG that has not
converged would all still produce plausible-looking samples.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("coupled_sh_posterior needs a real GPU", allow_module_level=True)
pytest.importorskip("gsplat")

from gs_experiment.coupled_sh_posterior import _data_matvec, sample_coupled_perturbations  # noqa: E402
from gs_experiment.rasterized_sh_precision import accumulate_sh_precision_rasterized  # noqa: E402

DEV = "cuda"
RES = 32
N_SPLATS = 8
DEGREE = 1
N_COEFFS = 4
NOISE_VAR = 0.01


def _tiny_problem(seed=0):
    rng = np.random.default_rng(seed)
    checkpoint = {
        "positions": rng.uniform(-0.4, 0.4, (N_SPLATS, 3)),
        "rotations": np.tile([1.0, 0.0, 0.0, 0.0], (N_SPLATS, 1)),
        "scales": np.full((N_SPLATS, 3), 0.18),
        "opacities": rng.uniform(0.4, 0.9, N_SPLATS),
        "sh_degree": DEGREE,
    }
    frames = []
    for i in range(4):
        ang = 2 * np.pi * i / 4
        c2w = np.eye(4)
        c2w[:3, 3] = [2.2 * np.sin(ang), 0.4 * np.cos(ang), 0.0]
        frames.append((f"cam{i}", c2w))
    f = 40.0
    K = np.array([[f, 0.0, RES / 2], [0.0, f, RES / 2], [0.0, 0.0, 1.0]])
    band_precision = np.ones((3, N_COEFFS))
    return checkpoint, frames, K, band_precision


def _dense_A(checkpoint, frames, K, band_precision):
    """`A` as an explicit (N*K, N*K) matrix, by applying the operator to every
    unit basis vector at once. Only possible because the scene is tiny; the
    real path never does this."""
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEV)  # noqa: E731
    means, quats = t(checkpoint["positions"]), t(checkpoint["rotations"])
    scales, opacities, Ks = t(checkpoint["scales"]), t(checkpoint["opacities"]), t(K)[None]
    dim = N_SPLATS * N_COEFFS
    basis = torch.zeros((N_SPLATS, dim, N_COEFFS), dtype=torch.float32, device=DEV)
    for i in range(N_SPLATS):
        for k in range(N_COEFFS):
            basis[i, i * N_COEFFS + k, k] = 1.0
    data = _data_matvec(basis, means, quats, scales, opacities, frames, Ks, RES, RES, DEGREE, DEV)
    lam = t(band_precision[0])
    out = lam[None, None, :] * basis + data / NOISE_VAR    # (N, dim, K)
    return out.permute(0, 2, 1).reshape(dim, dim).cpu().numpy()


def test_dense_A_is_symmetric_positive_definite():
    checkpoint, frames, K, band_precision = _tiny_problem()
    A = _dense_A(checkpoint, frames, K, band_precision)
    assert np.abs(A - A.T).max() < 1e-3 * np.abs(A).max(), "matvec is not a symmetric operator"
    assert np.linalg.eigvalsh(0.5 * (A + A.T)).min() > 0, "A is not positive definite"


def test_samples_have_the_posterior_covariance():
    checkpoint, frames, K, band_precision = _tiny_problem()
    A = _dense_A(checkpoint, frames, K, band_precision)
    A_inv = np.linalg.inv(0.5 * (A + A.T))

    block = accumulate_sh_precision_rasterized(
        checkpoint, frames, K, RES, RES, DEGREE, n_probes=32, seed=0, device=DEV, progress_every=0,
    ) / NOISE_VAR + np.diag(band_precision[0])

    n_draws = 400
    draws = sample_coupled_perturbations(
        checkpoint, frames, K, RES, RES, DEGREE, band_precision, NOISE_VAR, block,
        n_draws=n_draws, seed=0, n_cg_iters=60, device=DEV, verbose=False,
    ).cpu().numpy()                                   # (n_draws, N, 3, K)
    # With an identical prior on all three channels, each channel is an
    # independent draw from the same posterior.
    samples = draws.transpose(0, 2, 1, 3).reshape(n_draws * 3, N_SPLATS * N_COEFFS)
    emp = np.cov(samples, rowvar=False)

    rel = np.linalg.norm(emp - A_inv, "fro") / np.linalg.norm(A_inv, "fro")
    assert rel < 0.15, f"sample covariance does not match A^-1: relative Frobenius error {rel:.3f}"
    d_emp, d_ref = np.diag(emp), np.diag(A_inv)
    assert np.abs(d_emp - d_ref).max() / d_ref.max() < 0.15, "marginal variances off"


def test_coupling_never_reduces_marginal_variance_below_block_diagonal():
    """For PSD `A`, `(A^-1)_ii >= (A_ii)^-1` (Schur complement). So accounting
    for cross-splat coupling can only ever INCREASE a splat's marginal
    posterior variance relative to the block-diagonal approximation -- which
    is the whole reason the block diagonal is the wrong thing to report."""
    checkpoint, frames, K, band_precision = _tiny_problem()
    # Symmetrize ONCE and use the same matrix for both sides: `_dense_A` is
    # built from float32 renders whose backward uses non-deterministic
    # atomics, so the raw operator is symmetric only to ~1e-6 relative.
    A = _dense_A(checkpoint, frames, K, band_precision)
    A = 0.5 * (A + A.T)
    A_inv = np.linalg.inv(A)
    tol = 1e-6 * np.abs(np.diag(A_inv)).max()
    for i in range(N_SPLATS):
        sl = slice(i * N_COEFFS, (i + 1) * N_COEFFS)
        gap = A_inv[sl, sl] - np.linalg.inv(A[sl, sl])
        assert np.linalg.eigvalsh(0.5 * (gap + gap.T)).min() > -tol, f"splat {i} violates the bound"
