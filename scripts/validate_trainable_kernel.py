"""Follow-up to milestone 1 (see bq_splat/results/FINDINGS.md, section 1):
BQ with a hardcoded kernel bandwidth loses to a naive Riemann sum, and the
gap shrinks as node count rises relative to the fixed bandwidth -- pointing
at bandwidth mismatch, not a fundamental flaw in BQ, as the cause. This
script re-runs the same accuracy sweep from validate_milestone1.py, but
fits the RBF/Matern bandwidth per-trial via marginal-likelihood
optimization (bq_splat.hyperparams) instead of using one fixed value, and
checks whether that closes the gap.

Run: .venv/bin/python scripts/validate_trainable_kernel.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from bq_splat.hyperparams import fit_kernel_param
from bq_splat.kernels import MaternKernel, RBFKernel
from bq_splat.quadrature import bayesian_quadrature
from bq_splat.reference import riemann_estimate, true_integral
from bq_splat.toy_scene import make_mixture_scene, uniform_nodes


def sweep(n_trials=200, node_counts=(5, 10, 20, 40), seed=0, fit_bounds=(0.02, 8.0)):
    rng = np.random.default_rng(seed)
    rows = []
    t0 = time.time()
    for n in node_counts:
        for _ in range(n_trials):
            domain = (0.0, 10.0)
            scene = make_mixture_scene(rng, domain=domain, n_bumps=rng.integers(3, 8))
            nodes = uniform_nodes(rng, domain, n)
            values = scene.g_true(nodes)
            truth = true_integral(scene.g_true, *domain)

            riemann = riemann_estimate(nodes, values, *domain)
            riemann_err = abs(riemann - truth)

            row = dict(n=n, truth=truth, riemann_err=riemann_err)

            # fixed-bandwidth baselines, same as validate_milestone1.py
            fixed_rbf = bayesian_quadrature(nodes, values, RBFKernel(sigma=0.35), *domain)
            fixed_matern = bayesian_quadrature(nodes, values, MaternKernel(rho=0.5), *domain)
            row["fixed_rbf_err"] = abs(fixed_rbf.mean - truth)
            row["fixed_matern_err"] = abs(fixed_matern.mean - truth)

            # per-trial fitted bandwidth
            if n >= 3:  # marginal likelihood on 1-2 points is uninformative about lengthscale
                fit_rbf = fit_kernel_param(nodes, values, lambda s: RBFKernel(sigma=s), bounds=fit_bounds)
                fit_matern = fit_kernel_param(nodes, values, lambda r: MaternKernel(rho=r), bounds=fit_bounds)
                fitted_rbf_result = bayesian_quadrature(nodes, values, RBFKernel(sigma=fit_rbf.param), *domain)
                fitted_matern_result = bayesian_quadrature(nodes, values, MaternKernel(rho=fit_matern.param), *domain)
                row["fitted_rbf_err"] = abs(fitted_rbf_result.mean - truth)
                row["fitted_matern_err"] = abs(fitted_matern_result.mean - truth)
                row["fitted_rbf_sigma"] = fit_rbf.param
                row["fitted_matern_rho"] = fit_matern.param
            else:
                row["fitted_rbf_err"] = row["fixed_rbf_err"]
                row["fitted_matern_err"] = row["fixed_matern_err"]
                row["fitted_rbf_sigma"] = np.nan
                row["fitted_matern_rho"] = np.nan

            rows.append(row)
    print(f"(sweep took {time.time() - t0:.1f}s)")
    return rows


def summarize(rows):
    print("\n=== Fixed-bandwidth vs. fitted-bandwidth BQ vs. Riemann sum ===")
    node_counts = sorted(set(r["n"] for r in rows))
    header = f"{'n':>4}  {'riemann':>8}  {'rbf-fixed':>10}  {'rbf-fit':>8}  {'matern-fixed':>13}  {'matern-fit':>11}"
    print(header)
    for n in node_counts:
        subset = [r for r in rows if r["n"] == n]
        riemann_mae = np.mean([r["riemann_err"] for r in subset])
        rbf_fixed_mae = np.mean([r["fixed_rbf_err"] for r in subset])
        rbf_fit_mae = np.mean([r["fitted_rbf_err"] for r in subset])
        matern_fixed_mae = np.mean([r["fixed_matern_err"] for r in subset])
        matern_fit_mae = np.mean([r["fitted_matern_err"] for r in subset])
        print(
            f"{n:>4}  {riemann_mae:>8.4f}  {rbf_fixed_mae:>10.4f}  {rbf_fit_mae:>8.4f}  "
            f"{matern_fixed_mae:>13.4f}  {matern_fit_mae:>11.4f}"
        )

    fitted_sigmas = [r["fitted_rbf_sigma"] for r in rows if not np.isnan(r["fitted_rbf_sigma"])]
    fitted_rhos = [r["fitted_matern_rho"] for r in rows if not np.isnan(r["fitted_matern_rho"])]
    print(f"\nFitted RBF sigma:    median={np.median(fitted_sigmas):.3f}  "
          f"[{np.percentile(fitted_sigmas, 10):.3f}, {np.percentile(fitted_sigmas, 90):.3f}] (10-90th pctile)")
    print(f"Fitted Matern rho:   median={np.median(fitted_rhos):.3f}  "
          f"[{np.percentile(fitted_rhos, 10):.3f}, {np.percentile(fitted_rhos, 90):.3f}] (10-90th pctile)")
    print("(fixed baselines used sigma=0.35, rho=0.5 -- for comparison)")


if __name__ == "__main__":
    rows = sweep()
    summarize(rows)
