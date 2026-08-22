"""Isolation validation for DirectionalKernel / directional_posterior_variance,
before combining direction with position (see validate_directional_combined.py)
or trusting either anywhere near a real pipeline. Same discipline as the 1D
and 2D gap experiments: validate the new mechanism on its own first.

Two synthetic cases, single fixed spatial location:
  - "wide": training-view directions spread across most of the circle --
    simulates a splat seen from many angles (e.g. orbited by the camera).
  - "narrow": training-view directions clustered in a tight cone --
    simulates a splat seen only briefly, from nearly the same angle each
    time (a common SLAM situation: a surface glimpsed along a short
    stretch of trajectory).

For each, plot the true view-dependent signal, the observations, and the
posterior mean/variance as a function of query angle -- check that variance
stays low across the whole circle for "wide" but spikes for query angles
far from the narrow cone in "narrow".

Run: .venv/bin/python scripts/validate_directional_isolation.py
Output: bq_splat/results/directional_isolation.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bq_splat.kernels import DirectionalKernel
from bq_splat.quadrature import directional_posterior_variance

RESULTS_DIR = Path(__file__).resolve().parents[1] / "bq_splat" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def angle_to_unit_vector(theta):
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    return np.stack([np.cos(theta), np.sin(theta)], axis=1)


def g_true(theta, peak=0.6, height=1.0, base=0.3):
    """A simple synthetic view-dependent signal: a diffuse base plus one
    specular-like lobe peaked at `peak` -- stands in for a real BRDF-ish
    appearance function, not meant to be physically exact."""
    theta = np.asarray(theta, dtype=float)
    return base + height * np.exp(2.5 * (np.cos(theta - peak) - 1.0))


def run(seed=0, n_obs=12, kappa=4.0):
    rng = np.random.default_rng(seed)
    dir_kernel = DirectionalKernel(kappa=kappa)
    query_thetas = np.linspace(-np.pi, np.pi, 200)

    cases = {
        "wide coverage": rng.uniform(-np.pi, np.pi, size=n_obs),
        "narrow cone": rng.uniform(-0.35, 0.35, size=n_obs),
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for col, (label, obs_thetas) in enumerate(cases.items()):
        directions = angle_to_unit_vector(obs_thetas)
        values = g_true(obs_thetas) + rng.normal(scale=0.02, size=n_obs)

        means, variances = [], []
        for q_theta in query_thetas:
            q_dir = angle_to_unit_vector(q_theta)
            result = directional_posterior_variance(directions, values, dir_kernel, q_dir)
            means.append(result.mean)
            variances.append(result.variance)
        means, variances = np.array(means), np.array(variances)

        ax_top = axes[0, col]
        ax_top.plot(query_thetas, g_true(query_thetas), label="true g(theta)", color="black", linewidth=1)
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
        directions = angle_to_unit_vector(obs_thetas)
        values = g_true(obs_thetas)
        near_query = angle_to_unit_vector(np.median(obs_thetas))
        far_query = angle_to_unit_vector(np.median(obs_thetas) + np.pi)
        near = directional_posterior_variance(directions, values, dir_kernel, near_query)
        far = directional_posterior_variance(directions, values, dir_kernel, far_query)
        print(f"{label:>14}: variance near observed directions = {near.variance:.4f}, "
              f"variance at opposite direction = {far.variance:.4f}, ratio = {far.variance/max(near.variance,1e-9):.1f}x")


if __name__ == "__main__":
    run()
