"""Toy-scale (no GPU, pure numpy/scipy) validation CLI for bq_splat.

Consolidates what used to be 8 separate one-off scripts under scripts/ into
one flag-driven tool. Each `--check` below is basically a verbatim copy of
one of those scripts' core logic (same math, same defaults, same print/plot
output) -- only the dispatch is new. See each section's docstring for the
original script's rationale.

Run: .venv/bin/python bq_splat/validate.py --check <name>
  accuracy                   (was scripts/validate_milestone1.py)
  trainable-kernel            (was scripts/validate_trainable_kernel.py)
  trainable-kernel-heldout    (was scripts/validate_trainable_kernel_heldout.py)
  2d-gap                      (was scripts/validate_2d_gap_experiment.py)
  alpha-compositing           (was scripts/validate_alpha_compositing_equivalence.py)
  directional-isolation       (was scripts/validate_directional_isolation.py)
  directional-combined        (was scripts/validate_directional_combined.py)
  scaling                     (was scripts/benchmark_local_bq_scaling.py)
  rendering-aware              (new: rendering-aware BQ vs. box-style BQ)
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy.spatial import cKDTree

from bq_splat.hyperparams import fit_kernel_param, fit_kernel_param_pooled
from bq_splat.kernels import DirectionalKernel, MaternKernel, ProductKernel, RBFKernel
from bq_splat.quadrature import (
    BQResult,
    bayesian_quadrature,
    bayesian_quadrature_directional,
    bayesian_quadrature_nd,
    bayesian_quadrature_rendering_aware,
    directional_posterior_variance,
)
from bq_splat.reference import riemann_estimate, true_integral
from bq_splat.render_weight import GaussianRenderWeight
from bq_splat.toy_scene import (
    gap_nodes,
    gap_nodes_2d,
    make_mixture_scene,
    make_mixture_scene_2d,
    uniform_nodes,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _angle_to_unit_vector(theta):
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    return np.stack([np.cos(theta), np.sin(theta)], axis=1)


# =============================================================================
# --check accuracy  (was scripts/validate_milestone1.py)
#
# Milestone 1 (see ROADMAP.md): does BQ improve on naive discrete
# (Riemann-sum / "alpha compositing style") integral estimates on irregularly
# sampled 1D synthetic rays, does its posterior variance track true error, and
# does a deliberate coverage gap raise BQ variance the way the paper's
# differentiation-experiment claim expects?
#
# Outputs: printed summary + PNGs under bq_splat/results/
# =============================================================================


def accuracy_sweep(n_trials=200, node_counts=(5, 10, 20, 40), seed=0):
    """For random scenes and random sparse node placements, compare BQ mean
    error and Riemann-sum error against the true integral, and check whether
    BQ posterior variance correlates with BQ's actual error."""
    rng = np.random.default_rng(seed)
    kernels = {
        "rbf": RBFKernel(sigma=0.35),
        "matern32": MaternKernel(rho=0.5),
    }

    rows = []
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
            for name, kernel in kernels.items():
                result = bayesian_quadrature(nodes, values, kernel, *domain)
                row[f"{name}_err"] = abs(result.mean - truth)
                row[f"{name}_var"] = result.variance
            rows.append(row)

    return rows


def accuracy_summarize(rows):
    print("\n=== Accuracy sweep: BQ vs. Riemann-sum, by node count ===")
    node_counts = sorted(set(r["n"] for r in rows))
    for n in node_counts:
        subset = [r for r in rows if r["n"] == n]
        riemann_mae = np.mean([r["riemann_err"] for r in subset])
        rbf_mae = np.mean([r["rbf_err"] for r in subset])
        matern_mae = np.mean([r["matern32_err"] for r in subset])
        print(
            f"n={n:3d}  riemann MAE={riemann_mae:.4f}  "
            f"BQ-rbf MAE={rbf_mae:.4f}  BQ-matern32 MAE={matern_mae:.4f}"
        )

    for kname in ["rbf", "matern32"]:
        errs = np.array([r[f"{kname}_err"] for r in rows])
        variances = np.array([r[f"{kname}_var"] for r in rows])
        corr = np.corrcoef(errs, np.sqrt(variances))[0, 1]
        print(f"\nCorrelation(|BQ-{kname} error|, BQ-{kname} posterior std) = {corr:.3f}")


def accuracy_plot(rows):
    node_counts = sorted(set(r["n"] for r in rows))
    riemann_mae = [np.mean([r["riemann_err"] for r in rows if r["n"] == n]) for n in node_counts]
    rbf_mae = [np.mean([r["rbf_err"] for r in rows if r["n"] == n]) for n in node_counts]
    matern_mae = [np.mean([r["matern32_err"] for r in rows if r["n"] == n]) for n in node_counts]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(node_counts, riemann_mae, "o-", label="Riemann sum")
    ax.plot(node_counts, rbf_mae, "o-", label="BQ (RBF)")
    ax.plot(node_counts, matern_mae, "o-", label="BQ (Matern-3/2)")
    ax.set_xlabel("number of nodes")
    ax.set_ylabel("mean abs. error vs. true integral")
    ax.set_title("Milestone 1: BQ vs. Riemann sum, synthetic 1D rays")
    ax.legend()
    fig.tight_layout()
    out = RESULTS_DIR / "accuracy_vs_node_count.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out}")


def accuracy_gap_experiment(seed=1):
    """The toy proxy for the paper's differentiation experiment: a region
    that's fully interior to the domain (not occluded/out of range) but has
    deliberately sparse local node coverage, containing real signal
    structure. Check that BQ variance rises there even though the rest of
    the domain is well covered."""
    rng = np.random.default_rng(seed)
    domain = (0.0, 10.0)
    kernel = RBFKernel(sigma=0.35)

    scene = make_mixture_scene(rng, domain=domain, n_bumps=6, min_width=0.08, max_width=0.25)
    nodes, gap = gap_nodes(rng, domain, n=40, gap_center_frac=0.5, gap_width_frac=0.2, thin_prob=0.92)
    values = scene.g_true(nodes)

    query_points = np.linspace(domain[0] + 0.3, domain[1] - 0.3, 60)
    local_variances = []
    for q in query_points:
        lo, hi = max(domain[0], q - 0.75), min(domain[1], q + 0.75)
        local_nodes_mask = (nodes >= lo) & (nodes <= hi)
        local_nodes = nodes[local_nodes_mask]
        local_values = values[local_nodes_mask]
        result = bayesian_quadrature(local_nodes, local_values, kernel, lo, hi)
        local_variances.append(result.variance)

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    t_fine = np.linspace(*domain, 500)
    axes[0].plot(t_fine, scene.g_true(t_fine), label="g_true(t)")
    axes[0].scatter(nodes, values, color="black", s=15, zorder=3, label="node observations")
    axes[0].axvspan(*gap, color="orange", alpha=0.2, label="sparse-coverage gap (visible, not occluded)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("signal")
    axes[0].set_title("Milestone 1 gap experiment: sparse-but-visible region")

    axes[1].plot(query_points, local_variances, color="crimson")
    axes[1].axvspan(*gap, color="orange", alpha=0.2)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("local BQ posterior variance")

    fig.tight_layout()
    out = RESULTS_DIR / "gap_experiment.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")

    in_gap = [(q, v) for q, v in zip(query_points, local_variances) if gap[0] <= q <= gap[1] and not np.isnan(v)]
    out_gap = [(q, v) for q, v in zip(query_points, local_variances) if not (gap[0] <= q <= gap[1]) and not np.isnan(v)]
    mean_in = np.mean([v for _, v in in_gap]) if in_gap else float("nan")
    mean_out = np.mean([v for _, v in out_gap]) if out_gap else float("nan")
    print(f"\nMean local BQ variance inside gap:  {mean_in:.5f}")
    print(f"Mean local BQ variance outside gap: {mean_out:.5f}")
    print(f"Ratio (inside / outside): {mean_in / mean_out:.2f}x")


def check_accuracy():
    rows = accuracy_sweep()
    accuracy_summarize(rows)
    accuracy_plot(rows)
    accuracy_gap_experiment()


# =============================================================================
# --check trainable-kernel  (was scripts/validate_trainable_kernel.py)
#
# Follow-up to milestone 1 (see bq_splat/results/FINDINGS.md, section 1):
# BQ with a hardcoded kernel bandwidth loses to a naive Riemann sum, and the
# gap shrinks as node count rises relative to the fixed bandwidth -- pointing
# at bandwidth mismatch, not a fundamental flaw in BQ, as the cause. This
# re-runs the same accuracy sweep as --check accuracy, but fits the
# RBF/Matern bandwidth per-trial via marginal-likelihood optimization
# (bq_splat.hyperparams) instead of using one fixed value, and checks
# whether that closes the gap.
# =============================================================================


def trainable_kernel_sweep(n_trials=200, node_counts=(5, 10, 20, 40), seed=0, fit_bounds=(0.02, 8.0)):
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

            # fixed-bandwidth baselines, same as --check accuracy
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


def trainable_kernel_summarize(rows):
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


def check_trainable_kernel():
    rows = trainable_kernel_sweep()
    trainable_kernel_summarize(rows)


# =============================================================================
# --check trainable-kernel-heldout  (was scripts/validate_trainable_kernel_heldout.py)
#
# Held-out generalization test for bandwidth fitting (follow-up to
# FINDINGS.md section 5's caveat: that result fit and evaluated the bandwidth
# on the same scene's nodes, so it couldn't rule out the fitted bandwidth
# just being an unfair in-sample advantage over Riemann rather than a real
# generalizing improvement).
#
# This splits scenes into a "calibration" set and a disjoint "test" set,
# fits ONE shared bandwidth by maximizing pooled log marginal likelihood over
# the calibration set only, and evaluates it on test-set scenes it never saw.
# Compared against: the old hardcoded bandwidth, per-scene in-sample fitting
# (an oracle upper bound -- it gets to see each test scene's own data), and
# Riemann sum.
#
# This also matters for --check scaling's question: if one bandwidth fit
# once on a calibration set generalizes well to unseen scenes, that means a
# real GS deployment would NOT need to refit a bandwidth per pixel/per local
# neighborhood -- fit once, reuse everywhere -- which is a much cheaper
# computational story than fitting per query.
# =============================================================================


def trainable_kernel_heldout_make_dataset(rng, n_scenes, node_counts, domain=(0.0, 10.0)):
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


def trainable_kernel_heldout_run(n_calib_scenes=30, n_test_scenes=150, node_counts=(10, 20, 40), seed=0, fit_bounds=(0.02, 8.0)):
    domain = (0.0, 10.0)
    rng_calib = np.random.default_rng(seed)
    rng_test = np.random.default_rng(seed + 1)  # disjoint stream, not just a later slice of the same one

    calib = trainable_kernel_heldout_make_dataset(rng_calib, n_calib_scenes, node_counts, domain)
    test = trainable_kernel_heldout_make_dataset(rng_test, n_test_scenes, node_counts, domain)

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


def check_trainable_kernel_heldout():
    trainable_kernel_heldout_run()


# =============================================================================
# --check 2d-gap  (was scripts/validate_2d_gap_experiment.py)
#
# 2D bridge experiment (see conversation / ROADMAP.md): rerun the toy
# differentiation experiment from --check accuracy's gap experiment over an
# image-plane domain instead of a 1D ray, using splat centers with
# GS-realistic 2D scatter instead of samples along a depth axis. Checks
# whether the same "high variance in a well-observed-but-under-resolved
# region" signal survives the move to 2D before ever touching gsplat.
#
# Output: bq_splat/results/gap_experiment_2d.png
# =============================================================================


def gap_2d_run(seed=1, n_nodes=250, grid_res=45, window_radius=1.4):
    rng = np.random.default_rng(seed)
    domain = ((0.0, 10.0), (0.0, 10.0))
    (x0, x1), (y0, y1) = domain
    kernel = ProductKernel([RBFKernel(sigma=0.5), RBFKernel(sigma=0.5)])

    scene = make_mixture_scene_2d(rng, domain=domain, n_bumps=10, min_width=0.2, max_width=0.5)
    nodes, (gap_center, gap_radius) = gap_nodes_2d(
        rng, domain, n=n_nodes, gap_center_frac=(0.5, 0.5), gap_radius_frac=0.18, thin_prob=0.9
    )
    values = scene.g_true(nodes)

    margin = 0.3
    xs = np.linspace(x0 + margin, x1 - margin, grid_res)
    ys = np.linspace(y0 + margin, y1 - margin, grid_res)
    variance_grid = np.full((grid_res, grid_res), np.nan)

    for i, qx in enumerate(xs):
        for j, qy in enumerate(ys):
            lo_x, hi_x = max(x0, qx - window_radius), min(x1, qx + window_radius)
            lo_y, hi_y = max(y0, qy - window_radius), min(y1, qy + window_radius)
            mask = (nodes[:, 0] >= lo_x) & (nodes[:, 0] <= hi_x) & (nodes[:, 1] >= lo_y) & (nodes[:, 1] <= hi_y)
            local_nodes = nodes[mask]
            local_values = values[mask]
            result = bayesian_quadrature_nd(local_nodes, local_values, kernel, [(lo_x, hi_x), (lo_y, hi_y)])
            variance_grid[j, i] = result.variance  # row=y, col=x for imshow

    # true image, for reference
    img_res = 120
    img_xs = np.linspace(x0, x1, img_res)
    img_ys = np.linspace(y0, y1, img_res)
    grid_x, grid_y = np.meshgrid(img_xs, img_ys)
    true_img = scene.g_true(np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)).reshape(img_res, img_res)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].imshow(true_img, extent=[x0, x1, y0, y1], origin="lower", cmap="viridis")
    axes[0].scatter(nodes[:, 0], nodes[:, 1], s=6, color="white", edgecolor="black", linewidth=0.3, label="splat centers")
    gap_circle = plt.Circle(gap_center, gap_radius, fill=False, color="orange", linewidth=2, label="sparse-coverage gap")
    axes[0].add_patch(gap_circle)
    axes[0].set_title("true signal g(x,y) + splat centers")
    axes[0].legend(loc="upper right", fontsize=7)
    axes[0].set_xlim(x0, x1)
    axes[0].set_ylim(y0, y1)

    im = axes[1].imshow(variance_grid, extent=[x0, x1, y0, y1], origin="lower", cmap="inferno")
    gap_circle2 = plt.Circle(gap_center, gap_radius, fill=False, color="cyan", linewidth=2)
    axes[1].add_patch(gap_circle2)
    axes[1].set_title("local BQ posterior variance")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle("2D bridge experiment: sparse-but-visible region, image-plane domain")
    fig.tight_layout()
    out = RESULTS_DIR / "gap_experiment_2d.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")

    xx, yy = np.meshgrid(xs, ys)
    in_gap = np.sqrt((xx - gap_center[0]) ** 2 + (yy - gap_center[1]) ** 2) < gap_radius
    mean_in = np.nanmean(variance_grid[in_gap])
    mean_out = np.nanmean(variance_grid[~in_gap])
    print(f"Mean local BQ variance inside gap:  {mean_in:.5f}")
    print(f"Mean local BQ variance outside gap: {mean_out:.5f}")
    print(f"Ratio (inside / outside): {mean_in / mean_out:.2f}x")
    print(f"NaN fraction in variance grid: {np.mean(np.isnan(variance_grid)):.3f}")


def check_2d_gap():
    gap_2d_run()


# =============================================================================
# --check alpha-compositing  (was scripts/validate_alpha_compositing_equivalence.py)
#
# Numerical verification for `bq_splat/PROOF_alpha_compositing_equivalence.md`
# (ROADMAP.md item 1: "formal statement and proof that BQ posterior mean
# recovers alpha compositing").
#
# Three independent checks, corresponding to the proof's three main claims:
#
# 1. **Theorem A (exact reduction).** Under a piecewise-constant density/color
#    model, the continuous rendering integral has a closed form that is
#    *exactly* the standard discrete alpha-compositing sum -- a floating-point
#    identity, no kernel or BQ machinery involved. This just confirms the
#    algebra in the proof document is right.
# 2. **Theorem B (RKHS worst-case-error bound).** For any g in the kernel's
#    RKHS, |true integral - BQ mean| <= ||g||_H * sqrt(BQ variance). Checked
#    two ways: (a) never violated across many random test functions built as
#    finite kernel-section combinations (where ||g||_H is exactly computable
#    in closed form), and (b) approached with equality for a test function
#    built to approximate the quadrature-error representer itself -- the
#    bound's contents are quantitatively real, not just directionally
#    plausible.
# 3. **Discontinuity sensitivity.** g = T*sigma*c is, in general, only
#    piecewise smooth: c can jump between adjacent bins/splats, meaning the
#    worst-case bound's constant ||g||_H is large (formally infinite in the
#    RBF RKHS for a literal jump) exactly where colors change sharply. This
#    checks how BQ mean error and variance behave near a genuine color jump
#    for RBF vs. Matern-3/2 at matched nominal lengthscale, and connects to
#    the same RBF-vs-Matern divergence already found empirically in
#    bq_splat/results/FINDINGS.md sections 5-7 and gs_experiment's real-data
#    150x scale gap.
# =============================================================================


def alpha_compositing_check_theorem_a(rng, domain=(0.0, 10.0), n_bins=6, n_trials=20):
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


def _alpha_compositing_bq_weights_and_variance(kernel, nodes, a, b, rel_jitter=1e-8):
    nodes = np.asarray(nodes, dtype=float)
    n = nodes.shape[0]
    K = kernel.k(nodes.reshape(-1, 1), nodes.reshape(1, -1))
    K = K + rel_jitter * np.mean(np.diag(K)) * np.eye(n)
    v = np.asarray(kernel.v(nodes, a, b)).reshape(-1)
    vv = float(kernel.vv(a, b))
    w = np.linalg.solve(K, v)
    variance = vv - v @ w
    return K, v, vv, w, max(variance, 0.0)


def alpha_compositing_check_theorem_b(rng, kernel, domain=(0.0, 10.0), n_nodes=15, n_trials=40, n_extra=6):
    a, b = domain
    nodes = np.sort(rng.uniform(a, b, size=n_nodes))
    K, v, vv, w, var_bq = _alpha_compositing_bq_weights_and_variance(kernel, nodes, a, b)
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


def _alpha_compositing_step_scene(domain, jump_at, low, high):
    a, b = domain

    def g(t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return np.where(t < jump_at, low, high)

    return g


def alpha_compositing_check_discontinuity(rng, domain=(0.0, 10.0), jump_at=5.0, low=0.2, high=1.0, node_counts=(10, 20, 40, 80), sigma=0.3, rho=0.3):
    a, b = domain
    g = _alpha_compositing_step_scene(domain, jump_at, low, high)
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
            K, v, vv, w, var_bq = _alpha_compositing_bq_weights_and_variance(kernel, nodes, a, b)
            mean = float(v @ np.linalg.solve(K, values))
            row[f"{name}_error"] = abs(mean - true_val)
            row[f"{name}_variance"] = var_bq
        rows.append(row)
    return true_val, rows


def check_alpha_compositing():
    rng = np.random.default_rng(0)

    print("=== 1. Theorem A: exact reduction to alpha compositing ===")
    max_err = alpha_compositing_check_theorem_a(rng, n_trials=20)
    print(f"max |alpha_compositing - true_integral| over 20 random piecewise-constant "
          f"scenes (6 bins each): {max_err:.3e}\n")

    print("=== 2. Theorem B: RKHS worst-case-error bound ===")
    for name, kernel in (("RBF (sigma=0.6)", RBFKernel(sigma=0.6)), ("Matern-3/2 (rho=0.6)", MaternKernel(rho=0.6))):
        result = alpha_compositing_check_theorem_b(rng, kernel, n_trials=40)
        print(f"[{name}]")
        print(f"  BQ variance at these nodes: {result['var_bq']:.5f}")
        print(f"  bound violated: {result['violated']}/{result['n_trials']} random test functions")
        print(f"  error/bound ratio over random test functions: "
              f"min={result['ratio_min']:.4f} mean={result['ratio_mean']:.4f} max={result['ratio_max']:.4f}")
        print(f"  error/bound ratio for the representer-fitting test function, "
              f"as the fitting grid densifies (10/30/80 points): "
              f"{[f'{r:.4f}' for r in result['tight_ratios']]}\n")

    print("=== 3. Discontinuity sensitivity: RBF vs. Matern near a genuine color jump ===")
    true_val, rows = alpha_compositing_check_discontinuity(rng)
    print(f"true integral of the step scene: {true_val:.5f}")
    print(f"{'n':>4}  {'rbf_error':>10}  {'rbf_var':>10}  {'matern_error':>13}  {'matern_var':>11}")
    for row in rows:
        print(f"{row['n']:>4}  {row['rbf_error']:>10.5f}  {row['rbf_variance']:>10.5f}  "
              f"{row['matern_error']:>13.5f}  {row['matern_variance']:>11.5f}")


# =============================================================================
# --check directional-isolation  (was scripts/validate_directional_isolation.py)
#
# Isolation validation for DirectionalKernel / directional_posterior_variance,
# before combining direction with position (see --check directional-combined)
# or trusting either anywhere near a real pipeline. Same discipline as the 1D
# and 2D gap experiments: validate the new mechanism on its own first.
#
# Two synthetic cases, single fixed spatial location:
#   - "wide": training-view directions spread across most of the circle --
#     simulates a splat seen from many angles (e.g. orbited by the camera).
#   - "narrow": training-view directions clustered in a tight cone --
#     simulates a splat seen only briefly, from nearly the same angle each
#     time (a common SLAM situation: a surface glimpsed along a short
#     stretch of trajectory).
#
# For each, plot the true view-dependent signal, the observations, and the
# posterior mean/variance as a function of query angle -- check that variance
# stays low across the whole circle for "wide" but spikes for query angles
# far from the narrow cone in "narrow".
#
# Output: bq_splat/results/directional_isolation.png
# =============================================================================


def directional_isolation_g_true(theta, peak=0.6, height=1.0, base=0.3):
    """A simple synthetic view-dependent signal: a diffuse base plus one
    specular-like lobe peaked at `peak` -- stands in for a real BRDF-ish
    appearance function, not meant to be physically exact."""
    theta = np.asarray(theta, dtype=float)
    return base + height * np.exp(2.5 * (np.cos(theta - peak) - 1.0))


def directional_isolation_run(seed=0, n_obs=12, kappa=4.0):
    rng = np.random.default_rng(seed)
    dir_kernel = DirectionalKernel(kappa=kappa)
    query_thetas = np.linspace(-np.pi, np.pi, 200)

    cases = {
        "wide coverage": rng.uniform(-np.pi, np.pi, size=n_obs),
        "narrow cone": rng.uniform(-0.35, 0.35, size=n_obs),
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for col, (label, obs_thetas) in enumerate(cases.items()):
        directions = _angle_to_unit_vector(obs_thetas)
        values = directional_isolation_g_true(obs_thetas) + rng.normal(scale=0.02, size=n_obs)

        means, variances = [], []
        for q_theta in query_thetas:
            q_dir = _angle_to_unit_vector(q_theta)
            result = directional_posterior_variance(directions, values, dir_kernel, q_dir)
            means.append(result.mean)
            variances.append(result.variance)
        means, variances = np.array(means), np.array(variances)

        ax_top = axes[0, col]
        ax_top.plot(query_thetas, directional_isolation_g_true(query_thetas), label="true g(theta)", color="black", linewidth=1)
        ax_top.plot(query_thetas, means, label="posterior mean", color="tab:blue")
        ax_top.fill_between(
            query_thetas, means - np.sqrt(variances), means + np.sqrt(variances),
            color="tab:blue", alpha=0.2, label="+/- 1 posterior std",
        )
        ax_top.scatter(obs_thetas, values, color="black", s=20, zorder=5, label="observations")
        ax_top.set_title(f"{label}: signal + posterior")
        ax_top.set_xlabel("query angle (rad)")
        if col == 0:
            ax_top.set_ylabel("appearance")
        ax_top.legend(fontsize=7, loc="upper right")

        ax_bot = axes[1, col]
        ax_bot.plot(query_thetas, variances, color="crimson")
        for t in obs_thetas:
            ax_bot.axvline(t, color="gray", alpha=0.3, linewidth=0.8)
        ax_bot.set_title(f"{label}: posterior variance vs. query angle")
        ax_bot.set_xlabel("query angle (rad)")
        if col == 0:
            ax_bot.set_ylabel("posterior variance")

    fig.suptitle("Directional isolation experiment: single spatial point, varying angular coverage")
    fig.tight_layout()
    out = RESULTS_DIR / "directional_isolation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")

    for label, obs_thetas in cases.items():
        directions = _angle_to_unit_vector(obs_thetas)
        values = directional_isolation_g_true(obs_thetas)
        near_query = _angle_to_unit_vector(np.median(obs_thetas))
        far_query = _angle_to_unit_vector(np.median(obs_thetas) + np.pi)
        near = directional_posterior_variance(directions, values, dir_kernel, near_query)
        far = directional_posterior_variance(directions, values, dir_kernel, far_query)
        print(f"{label:>14}: variance near observed directions = {near.variance:.4f}, "
              f"variance at opposite direction = {far.variance:.4f}, ratio = {far.variance/max(near.variance,1e-9):.1f}x")


def check_directional_isolation():
    directional_isolation_run()


# =============================================================================
# --check directional-combined  (was scripts/validate_directional_combined.py)
#
# The actual claim from the conversation, tested directly: can two regions
# with IDENTICAL spatial splat density be told apart by whether they were
# observed from a wide spread of directions or a narrow cone -- something a
# position-only kernel is blind to by construction, but a position+direction
# kernel should catch.
#
# Setup: splat positions scattered at roughly uniform density over the whole
# 2D domain (no spatial "gap" this time -- spatial coverage is deliberately
# matched everywhere, unlike --check 2d-gap). Two equal-size circular zones
# get different angular treatment: "wide" zone splats are each observed from
# directions spread across most of the circle; "narrow" zone splats are each
# observed from a tight cone. Every other splat gets wide coverage too. Then:
#
#   (a) pure spatial-only BQ variance (bayesian_quadrature_nd, direction
#       ignored) over the domain -- should look similar in both zones, since
#       spatial density is matched.
#   (b) position+direction BQ variance (bayesian_quadrature_directional),
#       queried at one fixed "novel" direction chosen to lie outside the
#       narrow zone's cone -- should spike specifically over the narrow zone.
#
# Output: bq_splat/results/directional_combined.png
# =============================================================================


def directional_combined_build_scene(rng, domain, n_background=150, n_per_zone=18, n_dirs_per_splat=6, narrow_half_width=0.3, zone_radius=1.4, window_radius=1.6):
    """Two attempts at "spatial density matched" zones failed before this
    one, and it's worth recording why: independently random placement
    (even with equal *counts* per zone) can still differ in how *spread
    out* those points are within the disk -- 18 points is small enough for
    that clumping-by-chance to matter, and it showed up as a 0.32x
    position-only-variance ratio between zones meant to be identical.

    The fix used here is exact rather than statistical: both zones get the
    IDENTICAL set of relative offsets from their own center (one offset
    pattern, generated once, translated twice). Since a stationary kernel's
    behavior only depends on relative positions (the same fact that made
    caching `vv` by window size exact in --check scaling), this guarantees
    position-only BQ variance is equal between zones up to floating point,
    not just on average. Background splats are kept outside both zones'
    (zone_radius + window_radius) so they can't leak into either zone's
    local windows and reintroduce an asymmetry.
    """
    (x0, x1), (y0, y1) = domain
    wide_center = np.array([3.0, 3.0])
    narrow_center = np.array([7.0, 7.0])
    narrow_cone_center_theta = 0.0  # narrow zone's splats are all seen from near theta=0
    exclusion_radius = zone_radius + window_radius

    raw_background = np.stack([rng.uniform(x0, x1, size=n_background * 3), rng.uniform(y0, y1, size=n_background * 3)], axis=1)
    far_enough = (
        (np.linalg.norm(raw_background - wide_center, axis=1) > exclusion_radius)
        & (np.linalg.norm(raw_background - narrow_center, axis=1) > exclusion_radius)
    )
    background_positions = raw_background[far_enough][:n_background]

    offsets = []
    while len(offsets) < n_per_zone:
        o = rng.uniform(-zone_radius, zone_radius, size=2)
        if np.linalg.norm(o) < zone_radius:
            offsets.append(o)
    offsets = np.array(offsets)

    all_positions, all_directions, all_values = [], [], []
    scene = make_mixture_scene_2d(rng, domain=domain, n_bumps=8, min_width=0.4, max_width=0.9)

    def add_splat(p, thetas):
        value = float(scene.g_true(p.reshape(1, -1))[0])
        for theta in thetas:
            all_positions.append(p)
            all_directions.append(theta)
            all_values.append(value)

    for p in background_positions:
        add_splat(p, rng.uniform(-np.pi, np.pi, size=n_dirs_per_splat))
    for offset in offsets:
        add_splat(wide_center + offset, rng.uniform(-np.pi, np.pi, size=n_dirs_per_splat))
    for offset in offsets:
        thetas = rng.uniform(
            narrow_cone_center_theta - narrow_half_width, narrow_cone_center_theta + narrow_half_width,
            size=n_dirs_per_splat,
        )
        add_splat(narrow_center + offset, thetas)

    return (
        np.array(all_positions),
        _angle_to_unit_vector(np.array(all_directions)),
        np.array(all_values),
        scene,
        dict(wide_center=wide_center, wide_radius=zone_radius, narrow_center=narrow_center, narrow_radius=zone_radius),
    )


def directional_combined_run(seed=0, grid_res=40, window_radius=1.6, kappa=4.0):
    rng = np.random.default_rng(seed)
    domain = ((0.0, 10.0), (0.0, 10.0))
    (x0, x1), (y0, y1) = domain

    positions, directions, values, scene, zones = directional_combined_build_scene(rng, domain)

    pos_kernel = ProductKernel([RBFKernel(sigma=0.6), RBFKernel(sigma=0.6)])
    dir_kernel = DirectionalKernel(kappa=kappa)
    query_direction = _angle_to_unit_vector(np.pi)  # deliberately outside the narrow zone's cone (centered at 0)

    margin = 0.3
    xs = np.linspace(x0 + margin, x1 - margin, grid_res)
    ys = np.linspace(y0 + margin, y1 - margin, grid_res)
    spatial_only_grid = np.full((grid_res, grid_res), np.nan)
    directional_grid = np.full((grid_res, grid_res), np.nan)

    for i, qx in enumerate(xs):
        for j, qy in enumerate(ys):
            lo_x, hi_x = max(x0, qx - window_radius), min(x1, qx + window_radius)
            lo_y, hi_y = max(y0, qy - window_radius), min(y1, qy + window_radius)
            mask = (
                (positions[:, 0] >= lo_x) & (positions[:, 0] <= hi_x)
                & (positions[:, 1] >= lo_y) & (positions[:, 1] <= hi_y)
            )
            local_positions = positions[mask]
            local_directions = directions[mask]
            local_values = values[mask]
            bounds = [(lo_x, hi_x), (lo_y, hi_y)]

            spatial_result = bayesian_quadrature_nd(local_positions, local_values, pos_kernel, bounds)
            spatial_only_grid[j, i] = spatial_result.variance

            dir_result = bayesian_quadrature_directional(
                local_positions, local_directions, local_values, pos_kernel, dir_kernel, bounds, query_direction
            )
            directional_grid[j, i] = dir_result.variance

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    img_res = 100
    img_x, img_y = np.meshgrid(np.linspace(x0, x1, img_res), np.linspace(y0, y1, img_res))
    true_img = scene.g_true(np.stack([img_x.ravel(), img_y.ravel()], axis=1)).reshape(img_res, img_res)
    axes[0].imshow(true_img, extent=[x0, x1, y0, y1], origin="lower", cmap="viridis")
    unique_positions = np.unique(positions, axis=0)
    axes[0].scatter(unique_positions[:, 0], unique_positions[:, 1], s=4, color="white", edgecolor="black", linewidth=0.2)
    for center, radius, color, label in [
        (zones["wide_center"], zones["wide_radius"], "lime", "wide-angle zone"),
        (zones["narrow_center"], zones["narrow_radius"], "orange", "narrow-cone zone"),
    ]:
        axes[0].add_patch(plt.Circle(center, radius, fill=False, color=color, linewidth=2, label=label))
    axes[0].set_title("true signal + splat positions\n(spatial density matched everywhere)")
    axes[0].legend(loc="upper left", fontsize=7)
    axes[0].set_xlim(x0, x1)
    axes[0].set_ylim(y0, y1)

    im1 = axes[1].imshow(spatial_only_grid, extent=[x0, x1, y0, y1], origin="lower", cmap="inferno")
    for center, radius, color in [(zones["wide_center"], zones["wide_radius"], "lime"), (zones["narrow_center"], zones["narrow_radius"], "cyan")]:
        axes[1].add_patch(plt.Circle(center, radius, fill=False, color=color, linewidth=2))
    axes[1].set_title("(a) position-only BQ variance\n(blind to direction)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(directional_grid, extent=[x0, x1, y0, y1], origin="lower", cmap="inferno")
    for center, radius, color in [(zones["wide_center"], zones["wide_radius"], "lime"), (zones["narrow_center"], zones["narrow_radius"], "cyan")]:
        axes[2].add_patch(plt.Circle(center, radius, fill=False, color=color, linewidth=2))
    axes[2].set_title(f"(b) position+direction BQ variance\n(queried from theta=pi, outside narrow cone)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle("Combined experiment: does directionality catch what position-only BQ misses?")
    fig.tight_layout()
    out = RESULTS_DIR / "directional_combined.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")

    xx, yy = np.meshgrid(xs, ys)
    in_wide = np.linalg.norm(np.stack([xx, yy], axis=-1) - zones["wide_center"], axis=-1) < zones["wide_radius"]
    in_narrow = np.linalg.norm(np.stack([xx, yy], axis=-1) - zones["narrow_center"], axis=-1) < zones["narrow_radius"]

    print("\n--- position-only (spatial) BQ variance ---")
    print(f"wide zone:   mean={np.nanmean(spatial_only_grid[in_wide]):.5f}")
    print(f"narrow zone: mean={np.nanmean(spatial_only_grid[in_narrow]):.5f}")
    print(f"ratio (narrow/wide): {np.nanmean(spatial_only_grid[in_narrow]) / np.nanmean(spatial_only_grid[in_wide]):.2f}x")

    print("\n--- position+direction BQ variance (queried at theta=pi) ---")
    print(f"wide zone:   mean={np.nanmean(directional_grid[in_wide]):.5f}")
    print(f"narrow zone: mean={np.nanmean(directional_grid[in_narrow]):.5f}")
    print(f"ratio (narrow/wide): {np.nanmean(directional_grid[in_narrow]) / np.nanmean(directional_grid[in_wide]):.2f}x")


def check_directional_combined():
    directional_combined_run()


# =============================================================================
# --check scaling  (was scripts/benchmark_local_bq_scaling.py)
#
# Computational-scaling design work for ROADMAP.md's flagged engineering
# risk: "Batched closed-form BQ posterior-variance computation compatible
# with GS's typical splat counts (hundreds of thousands to millions) ...
# Naive per-pixel GP regression will not scale."
#
# This benchmarks the two things that actually determine feasibility, purely
# on CPU, before any gsplat/GPU code gets written:
#
# 1. Neighbor lookup: every local BQ computation (as in --check 2d-gap) needs
#    "which splats are near this query point" -- that check finds them by
#    brute-force masking over ALL nodes, which is O(N) per query and fine at
#    N~250 but not at GS scale (N up to ~10^6). A KD-tree
#    (scipy.spatial.cKDTree, already a scipy dependency -- no new library
#    needed) turns this into an O(log N + k) query after a one-time
#    O(N log N) build.
# 2. The BQ linear solve itself, as a function of local neighborhood size k
#    (not total scene size N -- per section 8's point, only nearby splats
#    matter for a local query, so the relevant scaling variable for the
#    expensive part is k, not N).
#
# Combining measured per-query neighbor-lookup cost and measured per-k BQ
# solve cost gives a concrete, non-hand-wavy estimate of full-image wall-clock
# cost on CPU, informing whether the real gsplat integration needs GPU
# batching from day one or can prototype further on CPU first.
# =============================================================================


def scaling_bq_with_cached_vv(nodes, values, kernel, bounds, cached_vv, rel_jitter=1e-4):
    """Same computation as bayesian_quadrature_nd, but skips recomputing
    kernel.vv(bounds) -- exact, not an approximation, for a translation-
    invariant (stationary) kernel evaluated on same-size, differently-
    centered windows: vv only depends on the window's shape/size, not its
    position (confirmed numerically to ~1e-13 relative difference for both
    RBFKernel and MaternKernel). Real per-pixel local windows are always the
    same size, just recentered, so this applies directly.
    """
    nodes = np.asarray(nodes, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)
    n = nodes.shape[0]
    if n == 0:
        return BQResult(mean=0.0, variance=float(cached_vv))
    kxx = kernel.k(nodes, nodes)
    jitter = rel_jitter * np.mean(np.diag(kxx))
    kxx = kxx + jitter * np.eye(n)
    v = kernel.v(nodes, bounds).reshape(-1)
    solved = np.linalg.solve(kxx, values)
    mean = float(v @ solved)
    solved_v = np.linalg.solve(kxx, v)
    variance = float(cached_vv - v @ solved_v)
    return BQResult(mean=mean, variance=max(variance, 0.0))


def scaling_benchmark_neighbor_queries(Ns, n_queries=100, radius=0.5, domain_size=100.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for N in Ns:
        nodes = rng.uniform(0, domain_size, size=(N, 2))
        queries = rng.uniform(0, domain_size, size=(n_queries, 2))

        t0 = time.perf_counter()
        neighbor_counts = []
        for q in queries:
            d2 = np.sum((nodes - q) ** 2, axis=1)
            idx = np.where(d2 <= radius**2)[0]
            neighbor_counts.append(idx.shape[0])
        brute_total = time.perf_counter() - t0

        t0 = time.perf_counter()
        tree = cKDTree(nodes)
        build_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        for q in queries:
            tree.query_ball_point(q, radius)
        kdtree_query_total = time.perf_counter() - t0

        rows.append(
            dict(
                N=N,
                mean_neighbors=np.mean(neighbor_counts),
                brute_per_query_ms=1000 * brute_total / n_queries,
                kdtree_build_ms=1000 * build_time,
                kdtree_per_query_ms=1000 * kdtree_query_total / n_queries,
            )
        )
    return rows


def scaling_benchmark_bq_solve_cost(ks, n_trials=30, seed=1, cache_vv=False):
    rng = np.random.default_rng(seed)
    kernel = ProductKernel([RBFKernel(sigma=0.5), RBFKernel(sigma=0.5)])
    bounds = [(0.0, 3.0), (0.0, 3.0)]
    cached_vv = kernel.vv(bounds) if cache_vv else None
    rows = []
    for k in ks:
        times = []
        for _ in range(n_trials):
            nodes = rng.uniform(0, 3, size=(max(k, 1), 2))
            values = rng.normal(size=max(k, 1))
            t0 = time.perf_counter()
            if cache_vv:
                scaling_bq_with_cached_vv(nodes, values, kernel, bounds, cached_vv)
            else:
                bayesian_quadrature_nd(nodes, values, kernel, bounds)
            times.append(time.perf_counter() - t0)
        rows.append(dict(k=k, mean_ms=1000 * np.mean(times), std_ms=1000 * np.std(times)))
    return rows


def scaling_extrapolate_full_image(neighbor_rows, solve_rows, image_res=(800, 800), label=""):
    n_pixels = image_res[0] * image_res[1]
    print(f"\n=== Extrapolated full-image cost{label} ({image_res[0]}x{image_res[1]} = {n_pixels:,} pixels) ===")
    for nrow in neighbor_rows:
        k = nrow["mean_neighbors"]
        # nearest solve-cost measurement to this neighbor count
        closest = min(solve_rows, key=lambda r: abs(r["k"] - k))
        per_pixel_ms = nrow["kdtree_per_query_ms"] + closest["mean_ms"]
        total_s = per_pixel_ms * n_pixels / 1000
        print(
            f"N={nrow['N']:>9,}  avg {k:5.1f} local neighbors/query  "
            f"lookup={nrow['kdtree_per_query_ms']:.4f}ms  solve={closest['mean_ms']:.4f}ms  "
            f"-> ~{total_s:,.1f}s single-threaded for the full image"
        )


def check_scaling():
    print("=== Neighbor lookup: brute-force vs. KD-tree ===")
    neighbor_rows = scaling_benchmark_neighbor_queries(Ns=[1_000, 10_000, 100_000, 500_000, 1_000_000])
    print(f"{'N':>10}  {'avg neighbors':>13}  {'brute ms/query':>15}  {'kdtree build ms':>16}  {'kdtree ms/query':>16}")
    for r in neighbor_rows:
        print(
            f"{r['N']:>10,}  {r['mean_neighbors']:>13.1f}  {r['brute_per_query_ms']:>15.4f}  "
            f"{r['kdtree_build_ms']:>16.2f}  {r['kdtree_per_query_ms']:>16.4f}"
        )

    print("\n=== BQ local solve cost vs. local neighborhood size k (vv recomputed every query) ===")
    solve_rows = scaling_benchmark_bq_solve_cost(ks=[5, 10, 20, 30, 50, 100, 200], cache_vv=False)
    print(f"{'k':>5}  {'mean ms':>10}  {'std ms':>10}")
    for r in solve_rows:
        print(f"{r['k']:>5}  {r['mean_ms']:>10.4f}  {r['std_ms']:>10.4f}")

    print("\n=== Same, but with vv cached once (exact for a fixed-size, translated window) ===")
    solve_rows_cached = scaling_benchmark_bq_solve_cost(ks=[5, 10, 20, 30, 50, 100, 200], cache_vv=True)
    print(f"{'k':>5}  {'mean ms':>10}  {'std ms':>10}")
    for r in solve_rows_cached:
        print(f"{r['k']:>5}  {r['mean_ms']:>10.4f}  {r['std_ms']:>10.4f}")

    speedup = np.mean([a["mean_ms"] / b["mean_ms"] for a, b in zip(solve_rows, solve_rows_cached)])
    print(f"\nAverage speedup from caching vv: {speedup:.1f}x")

    scaling_extrapolate_full_image(neighbor_rows, solve_rows, image_res=(800, 800), label=" -- naive (vv recomputed per pixel)")
    scaling_extrapolate_full_image(neighbor_rows, solve_rows_cached, image_res=(800, 800), label=" -- vv cached once per window size")


# =============================================================================
# --check rendering-aware  (new: rendering-aware Bayesian quadrature)
#
# Validates the rendering-aware BQ construction (bq_splat/render_weight.py,
# bayesian_quadrature_rendering_aware in bq_splat/quadrature.py) against the
# thing it replaces: bayesian_quadrature fed raw splat colors over a domain
# that doesn't know which part of it the renderer actually cares about --
# structurally the same mistake gs_experiment/pixel_uncertainty.py's
# LocalUncertaintyEngine makes with a generic 3D spatial window (see
# bq_splat/PROOF_alpha_compositing_equivalence.md section 7).
#
# Builds a genuine ray/pixel rendering functional a_q(t) = T(t) sigma(t) from
# an explicit density sigma(t) (a narrow bump -- a "hard" surface), not an
# arbitrary made-up weight, then moment-matches it to a GaussianRenderWeight
# for the closed-form path (a_q need not be exactly Gaussian in general --
# this is a cheap, honest approximation, not claimed exact). Splats scatter
# across the whole ray depth, including behind the surface (where a_q ~ 0,
# i.e. occluded); one such splat is given a deliberately wrong/extreme color
# to make the fix visible: the true rendered value and the rendering-aware BQ
# estimate barely move when it's added, while the old box-style estimate (fed
# the same raw colors) shifts noticeably, since it has no way to see that
# splat is irrelevant to this particular query.
#
# Output: bq_splat/results/rendering_aware.png
# =============================================================================


def rendering_aware_build_ray(domain=(0.0, 10.0), t_surface=4.0, density_amp=8.0, density_width=0.15, grid_res=4000):
    """A real transmittance-weighted rendering functional a_q(t) = T(t)
    sigma(t), derived from an explicit density sigma(t) (one narrow Gaussian
    bump -- a hard surface at t_surface), not an arbitrary made-up weight.
    T(t) = exp(-integral_a^t sigma(s) ds), computed by cumulative trapezoidal
    integration on a fine grid."""
    a, b = domain
    grid = np.linspace(a, b, grid_res)

    def density(t):
        t = np.asarray(t, dtype=float)
        return density_amp * np.exp(-0.5 * ((t - t_surface) / density_width) ** 2)

    dens_grid = density(grid)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (dens_grid[1:] + dens_grid[:-1]) * np.diff(grid))])
    transmittance_grid = np.exp(-cum)
    a_q_grid = transmittance_grid * dens_grid

    def a_q(t):
        return np.interp(t, grid, a_q_grid)

    return grid, a_q_grid, a_q


def rendering_aware_moment_match(grid, a_q_grid) -> GaussianRenderWeight:
    """Moment-match a_q (an arbitrary nonnegative bump T(t)*sigma(t)) to a
    GaussianRenderWeight: same total-mass location/spread as the real a_q,
    peak value as the amplitude."""
    mass = np.trapezoid(a_q_grid, grid)
    mean = np.trapezoid(grid * a_q_grid, grid) / mass
    var = np.trapezoid((grid - mean) ** 2 * a_q_grid, grid) / mass
    amplitude = float(np.max(a_q_grid))
    return GaussianRenderWeight(amplitude=amplitude, center=[mean], covariance=[[max(var, 1e-6)]])


def _mean_of(x):
    return x.mean if isinstance(x, BQResult) else float(x)


def _variance_suffix(x):
    return f" variance={x.variance:.5f}" if isinstance(x, BQResult) else ""


def rendering_aware_run(seed=0, domain=(0.0, 10.0), t_surface=4.0, n_nodes=40, sigma_rbf=0.3, occluder_t=8.0, occluder_color=5.0):
    rng = np.random.default_rng(seed)
    a, b = domain
    grid, a_q_grid, a_q = rendering_aware_build_ray(domain=domain, t_surface=t_surface)

    scene = make_mixture_scene(rng, domain=domain, n_bumps=5, min_width=0.3, max_width=0.8)
    c_true = scene.g_true

    render_weight = rendering_aware_moment_match(grid, a_q_grid)

    true_val, _ = integrate.quad(
        lambda t: float(a_q(np.array([t]))[0] * c_true(np.array([t]))[0]),
        a, b, points=[t_surface], limit=400, epsabs=1e-6, epsrel=1e-6,
    )

    nodes = uniform_nodes(rng, domain, n_nodes)
    colors = c_true(nodes)

    def estimates(nodes, colors):
        old_result = bayesian_quadrature(nodes, colors, RBFKernel(sigma=sigma_rbf), a, b)
        new_result = bayesian_quadrature_rendering_aware(
            nodes.reshape(-1, 1), colors, render_weight, sigma_rbf=sigma_rbf
        )
        riemann = riemann_estimate(nodes, a_q(nodes) * colors, a, b)
        return old_result, new_result, riemann

    old_before, new_before, riemann_before = estimates(nodes, colors)

    nodes_with_occ = np.append(nodes, occluder_t)
    colors_with_occ = np.append(colors, occluder_color)
    old_after, new_after, riemann_after = estimates(nodes_with_occ, colors_with_occ)

    return dict(
        grid=grid, a_q_grid=a_q_grid, c_true=c_true, nodes=nodes, colors=colors,
        occluder_t=occluder_t, occluder_color=occluder_color, true_val=true_val,
        old_before=old_before, new_before=new_before, riemann_before=riemann_before,
        old_after=old_after, new_after=new_after, riemann_after=riemann_after,
    )


def check_rendering_aware():
    result = rendering_aware_run()
    true_val = result["true_val"]

    print("=== Rendering-aware BQ vs. box-style BQ, ray/pixel domain with an occluded splat ===")
    print(f"true rendered value C_q = integral a_q(t) c(t) dt: {true_val:.5f}\n")

    def report(label, before, after):
        print(f"[{label}]")
        print(f"  before adding occluded splat: mean={_mean_of(before):.5f}{_variance_suffix(before)}")
        print(f"  after adding occluded splat:  mean={_mean_of(after):.5f}{_variance_suffix(after)}")
        print(f"  shift from one occluded (a_q~0) splat with a wrong color: {abs(_mean_of(after) - _mean_of(before)):.5f}\n")

    report("old box-style BQ (bayesian_quadrature, fed raw colors)", result["old_before"], result["old_after"])
    report("rendering-aware BQ (bayesian_quadrature_rendering_aware)", result["new_before"], result["new_after"])
    report("Riemann / alpha-compositing baseline (correctly a_q-weighted)", result["riemann_before"], result["riemann_after"])

    grid, a_q_grid, c_true = result["grid"], result["a_q_grid"], result["c_true"]
    nodes, colors = result["nodes"], result["colors"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(grid, c_true(grid), color="tab:blue", label="true color field c(t)")
    ax2 = ax.twinx()
    ax2.plot(grid, a_q_grid, color="tab:red", label="rendering weight a_q(t) = T(t) sigma(t)")
    ax.scatter(nodes, colors, s=18, color="black", zorder=5, label="splat observations (color)")
    ax.scatter(
        [result["occluder_t"]], [result["occluder_color"]], s=80, color="tab:orange", marker="*", zorder=6,
        label="occluded splat, wrong color",
    )
    ax.set_xlabel("ray depth t")
    ax.set_ylabel("color", color="tab:blue")
    ax2.set_ylabel("a_q(t)", color="tab:red")
    ax.set_title("ray profile: color field, rendering weight, splat observations")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7)

    ax = axes[1]
    labels = ["true", "riemann\n(alpha-comp.)", "old box BQ", "rendering-aware BQ"]
    before_vals = [true_val, result["riemann_before"], result["old_before"].mean, result["new_before"].mean]
    after_vals = [true_val, result["riemann_after"], result["old_after"].mean, result["new_after"].mean]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, before_vals, width, label="without occluded splat", color="tab:gray")
    ax.bar(x + width / 2, after_vals, width, label="with occluded splat (wrong color)", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(true_val, color="black", linestyle="--", linewidth=1, label="true value")
    ax.set_title("estimate shift from one occluded, wrong-colored splat")
    ax.legend(fontsize=7)

    fig.suptitle("Rendering-aware BQ vs. box-style BQ: does an occluded splat corrupt the estimate?")
    fig.tight_layout()
    out = RESULTS_DIR / "rendering_aware.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# =============================================================================
# CLI dispatch
# =============================================================================

CHECKS = {
    "accuracy": check_accuracy,
    "trainable-kernel": check_trainable_kernel,
    "trainable-kernel-heldout": check_trainable_kernel_heldout,
    "2d-gap": check_2d_gap,
    "alpha-compositing": check_alpha_compositing,
    "directional-isolation": check_directional_isolation,
    "directional-combined": check_directional_combined,
    "scaling": check_scaling,
    "rendering-aware": check_rendering_aware,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", required=True, choices=sorted(CHECKS.keys()), help="which validation to run")
    args = parser.parse_args()
    CHECKS[args.check]()


if __name__ == "__main__":
    main()
