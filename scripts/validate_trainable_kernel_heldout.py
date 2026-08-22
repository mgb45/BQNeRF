"""Held-out generalization test for bandwidth fitting (follow-up to
FINDINGS.md section 5's caveat: that result fit and evaluated the bandwidth
on the same scene's nodes, so it couldn't rule out the fitted bandwidth
just being an unfair in-sample advantage over Riemann rather than a real
generalizing improvement).

This splits scenes into a "calibration" set and a disjoint "test" set,
fits ONE shared bandwidth by maximizing pooled log marginal likelihood over
the calibration set only, and evaluates it on test-set scenes it never saw.
Compared against: the old hardcoded bandwidth, per-scene in-sample fitting
(an oracle upper bound -- it gets to see each test scene's own data), and
Riemann sum.

This also matters for scripts/benchmark_local_bq_scaling.py's question: if
one bandwidth fit once on a calibration set generalizes well to unseen
scenes, that means a real GS deployment would NOT need to refit a bandwidth
per pixel/per local neighborhood -- fit once, reuse everywhere -- which is
a much cheaper computational story than fitting per query.

Run: .venv/bin/python scripts/validate_trainable_kernel_heldout.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from bq_splat.hyperparams import fit_kernel_param, fit_kernel_param_pooled
from bq_splat.kernels import MaternKernel, RBFKernel
from bq_splat.quadrature import bayesian_quadrature
from bq_splat.reference import riemann_estimate, true_integral
from bq_splat.toy_scene import make_mixture_scene, uniform_nodes


def make_dataset(rng, n_scenes, node_counts, domain=(0.0, 10.0)):
    """One (nodes, values, truth) triple per (scene, node_count) combination."""
    rows = []
    for n in node_counts:
        for _ in range(n_scenes):
            scene = make_mixture_scene(rng, domain=domain, n_bumps=rng.integers(3, 8))
            nodes = uniform_nodes(rng, domain, n)
            values = scene.g_true(nodes)
            truth = true_integral(scene.g_true, *domain)
            rows.append(dict(n=n, nodes=nodes, values=values, truth=truth))
    return rows


def run(n_calib_scenes=30, n_test_scenes=150, node_counts=(10, 20, 40), seed=0, fit_bounds=(0.02, 8.0)):
    domain = (0.0, 10.0)
    rng_calib = np.random.default_rng(seed)
    rng_test = np.random.default_rng(seed + 1)  # disjoint stream, not just a later slice of the same one

    calib = make_dataset(rng_calib, n_calib_scenes, node_counts, domain)
    test = make_dataset(rng_test, n_test_scenes, node_counts, domain)

    calib_datasets = [(r["nodes"], r["values"]) for r in calib]
    print(f"Fitting global bandwidth on {len(calib_datasets)} calibration (scene, n) draws...")
    global_rbf = fit_kernel_param_pooled(calib_datasets, lambda s: RBFKernel(sigma=s), bounds=fit_bounds)
    global_matern = fit_kernel_param_pooled(calib_datasets, lambda r_: MaternKernel(rho=r_), bounds=fit_bounds)
    print(f"Global fitted RBF sigma:    {global_rbf.param:.3f}")
    print(f"Global fitted Matern rho:   {global_matern.param:.3f}")

    for n in node_counts:
        subset = [r for r in test if r["n"] == n]
        riemann_errs, fixed_rbf_errs, fixed_matern_errs = [], [], []
        global_rbf_errs, global_matern_errs = [], []
        oracle_rbf_errs, oracle_matern_errs = [], []

        for r in subset:
            nodes, values, truth = r["nodes"], r["values"], r["truth"]
            riemann_errs.append(abs(riemann_estimate(nodes, values, *domain) - truth))
            fixed_rbf_errs.append(abs(bayesian_quadrature(nodes, values, RBFKernel(sigma=0.35), *domain).mean - truth))
            fixed_matern_errs.append(abs(bayesian_quadrature(nodes, values, MaternKernel(rho=0.5), *domain).mean - truth))
            global_rbf_errs.append(abs(bayesian_quadrature(nodes, values, RBFKernel(sigma=global_rbf.param), *domain).mean - truth))
            global_matern_errs.append(abs(bayesian_quadrature(nodes, values, MaternKernel(rho=global_matern.param), *domain).mean - truth))

            oracle_rbf = fit_kernel_param(nodes, values, lambda s: RBFKernel(sigma=s), bounds=fit_bounds)
            oracle_matern = fit_kernel_param(nodes, values, lambda r_: MaternKernel(rho=r_), bounds=fit_bounds)
            oracle_rbf_errs.append(abs(bayesian_quadrature(nodes, values, RBFKernel(sigma=oracle_rbf.param), *domain).mean - truth))
            oracle_matern_errs.append(abs(bayesian_quadrature(nodes, values, MaternKernel(rho=oracle_matern.param), *domain).mean - truth))

        print(f"\n--- n={n} (test-set MAE, {len(subset)} held-out scenes) ---")
        print(f"Riemann:                  {np.mean(riemann_errs):.4f}")
        print(f"BQ-RBF fixed (sig=0.35):  {np.mean(fixed_rbf_errs):.4f}")
        print(f"BQ-RBF global-fit:        {np.mean(global_rbf_errs):.4f}   (fit once on calibration set)")
        print(f"BQ-RBF per-scene oracle:  {np.mean(oracle_rbf_errs):.4f}   (fit on the test scene itself)")
        print(f"BQ-Matern fixed (rho=0.5):{np.mean(fixed_matern_errs):.4f}")
        print(f"BQ-Matern global-fit:     {np.mean(global_matern_errs):.4f}   (fit once on calibration set)")
        print(f"BQ-Matern per-scene oracle:{np.mean(oracle_matern_errs):.4f}   (fit on the test scene itself)")


if __name__ == "__main__":
    run()
