"""Milestone 1 (see ROADMAP.md): does BQ improve on naive discrete
(Riemann-sum / "alpha compositing style") integral estimates on irregularly
sampled 1D synthetic rays, does its posterior variance track true error, and
does a deliberate coverage gap raise BQ variance the way the paper's
differentiation-experiment claim expects?

Run: .venv/bin/python scripts/validate_milestone1.py
Outputs: printed summary + PNGs under bq_splat/results/
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bq_splat.kernels import MaternKernel, RBFKernel
from bq_splat.quadrature import bayesian_quadrature
from bq_splat.reference import riemann_estimate, true_integral
from bq_splat.toy_scene import gap_nodes, make_mixture_scene, uniform_nodes

RESULTS_DIR = Path(__file__).resolve().parents[1] / "bq_splat" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def sweep_accuracy(n_trials=200, node_counts=(5, 10, 20, 40), seed=0):
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


def summarize_accuracy(rows):
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


def plot_accuracy(rows):
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


def gap_experiment(seed=1):
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


if __name__ == "__main__":
    rows = sweep_accuracy()
    summarize_accuracy(rows)
    plot_accuracy(rows)
    gap_experiment()
