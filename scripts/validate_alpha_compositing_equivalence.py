"""Numerical verification for `bq_splat/PROOF_alpha_compositing_equivalence.md`
(ROADMAP.md item 1: "formal statement and proof that BQ posterior mean
recovers alpha compositing").

Three independent checks, corresponding to the proof's three main claims:

1. **Theorem A (exact reduction).** Under a piecewise-constant density/color
   model, the continuous rendering integral has a closed form that is
   *exactly* the standard discrete alpha-compositing sum -- a floating-point
   identity, no kernel or BQ machinery involved. This just confirms the
   algebra in the proof document is right.
2. **Theorem B (RKHS worst-case-error bound).** For any g in the kernel's
   RKHS, |true integral - BQ mean| <= ||g||_H * sqrt(BQ variance). Checked
   two ways: (a) never violated across many random test functions built as
   finite kernel-section combinations (where ||g||_H is exactly computable
   in closed form), and (b) approached with equality for a test function
   built to approximate the quadrature-error representer itself -- the
   bound's contents are quantitatively real, not just directionally
   plausible.
3. **Discontinuity sensitivity.** g = T*sigma*c is, in general, only
   piecewise smooth: c can jump between adjacent bins/splats, meaning the
   worst-case bound's constant ||g||_H is large (formally infinite in the
   RBF RKHS for a literal jump) exactly where colors change sharply. This
   checks how BQ mean error and variance behave near a genuine color jump
   for RBF vs. Matern-3/2 at matched nominal lengthscale, and connects to
   the same RBF-vs-Matern divergence already found empirically in
   bq_splat/results/FINDINGS.md sections 5-7 and gs_experiment's real-data
   150x scale gap.

Run: .venv/bin/python scripts/validate_alpha_compositing_equivalence.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy import integrate

from bq_splat.kernels import MaternKernel, RBFKernel

RESULTS_DIR = Path(__file__).resolve().parents[1] / "bq_splat" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Theorem A: alpha compositing is the exact integral under the
#    piecewise-constant density/color model.
# ---------------------------------------------------------------------------

def check_theorem_a(rng, domain=(0.0, 10.0), n_bins=6, n_trials=20):
    a, b = domain
    max_abs_err = 0.0
    for _ in range(n_trials):
        interior = np.sort(rng.uniform(a, b, size=n_bins - 1))
        edges = np.concatenate([[a], interior, [b]])
        sigmas = rng.uniform(0.1, 3.0, size=n_bins)
        colors = rng.uniform(0.0, 1.0, size=n_bins)

        deltas = np.diff(edges)
        alphas = 1.0 - np.exp(-sigmas * deltas)
        T = np.concatenate([[1.0], np.cumprod(1.0 - alphas)[:-1]])
        w = alphas * T
        alpha_compositing_value = float(np.sum(w * colors))

        def g(t, edges=edges, sigmas=sigmas, colors=colors, T=T):
            for i in range(n_bins):
                lo, hi = edges[i], edges[i + 1]
                if lo <= t <= hi:
                    return float(T[i] * math.exp(-sigmas[i] * (t - lo)) * sigmas[i] * colors[i])
            return 0.0

        true_val, _ = integrate.quad(g, a, b, points=list(edges[1:-1]), limit=200)
        max_abs_err = max(max_abs_err, abs(alpha_compositing_value - true_val))

    return max_abs_err


# ---------------------------------------------------------------------------
# 2. Theorem B: RKHS worst-case-error bound, |I[g] - BQ_mean[g]| <=
#    ||g||_H * sqrt(BQ_variance).
# ---------------------------------------------------------------------------

def _bq_weights_and_variance(kernel, nodes, a, b, rel_jitter=1e-8):
    nodes = np.asarray(nodes, dtype=float)
    n = nodes.shape[0]
    K = kernel.k(nodes.reshape(-1, 1), nodes.reshape(1, -1))
    K = K + rel_jitter * np.mean(np.diag(K)) * np.eye(n)
    v = np.asarray(kernel.v(nodes, a, b)).reshape(-1)
    vv = float(kernel.vv(a, b))
    w = np.linalg.solve(K, v)
    variance = vv - v @ w
    return K, v, vv, w, max(variance, 0.0)


def check_theorem_b(rng, kernel, domain=(0.0, 10.0), n_nodes=15, n_trials=40, n_extra=6):
    a, b = domain
    nodes = np.sort(rng.uniform(a, b, size=n_nodes))
    K, v, vv, w, var_bq = _bq_weights_and_variance(kernel, nodes, a, b)
    sqrt_var = math.sqrt(var_bq)

    ratios = []
    violated = 0
    for _ in range(n_trials):
        z = rng.uniform(a, b, size=n_extra)
        c = rng.normal(size=n_extra)

        Kzz = kernel.k(z.reshape(-1, 1), z.reshape(1, -1))
        norm_g = math.sqrt(max(float(c @ Kzz @ c), 0.0))
        vz = np.asarray(kernel.v(z, a, b)).reshape(-1)
        I_g = float(c @ vz)

        Kxz = kernel.k(nodes.reshape(-1, 1), z.reshape(1, -1))  # (n_nodes, n_extra)
        g_at_nodes = Kxz @ c
        bq_mean_g = float(v @ np.linalg.solve(K, g_at_nodes))

        error = abs(I_g - bq_mean_g)
        bound = norm_g * sqrt_var
        if bound > 1e-12:
            ratios.append(error / bound)
        if error > bound * (1.0 + 1e-6) + 1e-10:
            violated += 1

    # Tight case: build g to approximate the quadrature-error representer
    # r(t) = v_kernel(t) - k(t, nodes) @ K^-1 @ v, whose RKHS norm is
    # defined to equal sqrt(var_bq) -- fit a finite kernel-section
    # combination to match r on a fine auxiliary grid, then check the ratio
    # approaches 1 as that grid gets denser (rather than trusting the
    # identity by construction, which would be circular).
    tight_ratios = []
    for n_fit in (10, 30, 80):
        z = np.linspace(a, b, n_fit)
        r_z = np.asarray(kernel.v(z, a, b)).reshape(-1) - kernel.k(z.reshape(-1, 1), nodes.reshape(1, -1)) @ w
        Kzz = kernel.k(z.reshape(-1, 1), z.reshape(1, -1))
        Kzz = Kzz + 1e-8 * np.mean(np.diag(Kzz)) * np.eye(n_fit)
        c = np.linalg.solve(Kzz, r_z)

        norm_g = math.sqrt(max(float(c @ Kzz @ c), 0.0))
        vz = np.asarray(kernel.v(z, a, b)).reshape(-1)
        I_g = float(c @ vz)
        Kxz = kernel.k(nodes.reshape(-1, 1), z.reshape(1, -1))
        g_at_nodes = Kxz @ c
        bq_mean_g = float(v @ np.linalg.solve(K, g_at_nodes))
        error = abs(I_g - bq_mean_g)
        bound = norm_g * sqrt_var
        tight_ratios.append(error / bound if bound > 1e-12 else float("nan"))

    return dict(
        var_bq=var_bq, violated=violated, n_trials=n_trials,
        ratio_min=min(ratios), ratio_max=max(ratios), ratio_mean=float(np.mean(ratios)),
        tight_ratios=tight_ratios,
    )


# ---------------------------------------------------------------------------
# 3. Discontinuity sensitivity: RBF vs. Matern near a genuine color jump.
# ---------------------------------------------------------------------------

def _step_scene(domain, jump_at, low, high):
    a, b = domain

    def g(t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return np.where(t < jump_at, low, high)

    return g


def check_discontinuity(rng, domain=(0.0, 10.0), jump_at=5.0, low=0.2, high=1.0, node_counts=(10, 20, 40, 80), sigma=0.3, rho=0.3):
    a, b = domain
    g = _step_scene(domain, jump_at, low, high)
    true_val, _ = integrate.quad(lambda t: float(g(np.array([t]))[0]), a, b, points=[jump_at], limit=200)

    rbf = RBFKernel(sigma=sigma)
    matern = MaternKernel(rho=rho)

    rows = []
    for n in node_counts:
        # deliberately no node exactly at the jump -- evenly spaced avoids
        # any single trial getting lucky/unlucky by chance node placement.
        nodes = np.linspace(a + 0.5 * (b - a) / n, b - 0.5 * (b - a) / n, n)
        values = g(nodes)
        row = {"n": n}
        for name, kernel in (("rbf", rbf), ("matern", matern)):
            K, v, vv, w, var_bq = _bq_weights_and_variance(kernel, nodes, a, b)
            mean = float(v @ np.linalg.solve(K, values))
            row[f"{name}_error"] = abs(mean - true_val)
            row[f"{name}_variance"] = var_bq
        rows.append(row)
    return true_val, rows


def main():
    rng = np.random.default_rng(0)

    print("=== 1. Theorem A: exact reduction to alpha compositing ===")
    max_err = check_theorem_a(rng, n_trials=20)
    print(f"max |alpha_compositing - true_integral| over 20 random piecewise-constant "
          f"scenes (6 bins each): {max_err:.3e}\n")

    print("=== 2. Theorem B: RKHS worst-case-error bound ===")
    for name, kernel in (("RBF (sigma=0.6)", RBFKernel(sigma=0.6)), ("Matern-3/2 (rho=0.6)", MaternKernel(rho=0.6))):
        result = check_theorem_b(rng, kernel, n_trials=40)
        print(f"[{name}]")
        print(f"  BQ variance at these nodes: {result['var_bq']:.5f}")
        print(f"  bound violated: {result['violated']}/{result['n_trials']} random test functions")
        print(f"  error/bound ratio over random test functions: "
              f"min={result['ratio_min']:.4f} mean={result['ratio_mean']:.4f} max={result['ratio_max']:.4f}")
        print(f"  error/bound ratio for the representer-fitting test function, "
              f"as the fitting grid densifies (10/30/80 points): "
              f"{[f'{r:.4f}' for r in result['tight_ratios']]}\n")

    print("=== 3. Discontinuity sensitivity: RBF vs. Matern near a genuine color jump ===")
    true_val, rows = check_discontinuity(rng)
    print(f"true integral of the step scene: {true_val:.5f}")
    print(f"{'n':>4}  {'rbf_error':>10}  {'rbf_var':>10}  {'matern_error':>13}  {'matern_var':>11}")
    for row in rows:
        print(f"{row['n']:>4}  {row['rbf_error']:>10.5f}  {row['rbf_variance']:>10.5f}  "
              f"{row['matern_error']:>13.5f}  {row['matern_variance']:>11.5f}")


if __name__ == "__main__":
    main()
