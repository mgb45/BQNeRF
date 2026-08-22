"""Computational-scaling design work for ROADMAP.md's flagged engineering
risk: "Batched closed-form BQ posterior-variance computation compatible
with GS's typical splat counts (hundreds of thousands to millions) ...
Naive per-pixel GP regression will not scale."

This benchmarks the two things that actually determine feasibility, purely
on CPU, before any gsplat/GPU code gets written:

1. Neighbor lookup: every local BQ computation (as in
   scripts/validate_2d_gap_experiment.py) needs "which splats are near this
   query point" -- that script finds them by brute-force masking over ALL
   nodes, which is O(N) per query and fine at N~250 but not at GS scale
   (N up to ~10^6). A KD-tree (scipy.spatial.cKDTree, already a scipy
   dependency -- no new library needed) turns this into an O(log N + k)
   query after a one-time O(N log N) build.
2. The BQ linear solve itself, as a function of local neighborhood size k
   (not total scene size N -- per section 8's point, only nearby splats
   matter for a local query, so the relevant scaling variable for the
   expensive part is k, not N).

Combining measured per-query neighbor-lookup cost and measured per-k BQ
solve cost gives a concrete, non-hand-wavy estimate of full-image wall-clock
cost on CPU, informing whether the real gsplat integration needs GPU
batching from day one or can prototype further on CPU first.

Run: .venv/bin/python scripts/benchmark_local_bq_scaling.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.spatial import cKDTree

from bq_splat.kernels import ProductKernel, RBFKernel
from bq_splat.quadrature import BQResult, bayesian_quadrature_nd


def bq_with_cached_vv(nodes, values, kernel, bounds, cached_vv, rel_jitter=1e-4):
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


def benchmark_neighbor_queries(Ns, n_queries=100, radius=0.5, domain_size=100.0, seed=0):
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


def benchmark_bq_solve_cost(ks, n_trials=30, seed=1, cache_vv=False):
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
                bq_with_cached_vv(nodes, values, kernel, bounds, cached_vv)
            else:
                bayesian_quadrature_nd(nodes, values, kernel, bounds)
            times.append(time.perf_counter() - t0)
        rows.append(dict(k=k, mean_ms=1000 * np.mean(times), std_ms=1000 * np.std(times)))
    return rows


def extrapolate_full_image(neighbor_rows, solve_rows, image_res=(800, 800), label=""):
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


if __name__ == "__main__":
    print("=== Neighbor lookup: brute-force vs. KD-tree ===")
    neighbor_rows = benchmark_neighbor_queries(Ns=[1_000, 10_000, 100_000, 500_000, 1_000_000])
    print(f"{'N':>10}  {'avg neighbors':>13}  {'brute ms/query':>15}  {'kdtree build ms':>16}  {'kdtree ms/query':>16}")
    for r in neighbor_rows:
        print(
            f"{r['N']:>10,}  {r['mean_neighbors']:>13.1f}  {r['brute_per_query_ms']:>15.4f}  "
            f"{r['kdtree_build_ms']:>16.2f}  {r['kdtree_per_query_ms']:>16.4f}"
        )

    print("\n=== BQ local solve cost vs. local neighborhood size k (vv recomputed every query) ===")
    solve_rows = benchmark_bq_solve_cost(ks=[5, 10, 20, 30, 50, 100, 200], cache_vv=False)
    print(f"{'k':>5}  {'mean ms':>10}  {'std ms':>10}")
    for r in solve_rows:
        print(f"{r['k']:>5}  {r['mean_ms']:>10.4f}  {r['std_ms']:>10.4f}")

    print("\n=== Same, but with vv cached once (exact for a fixed-size, translated window) ===")
    solve_rows_cached = benchmark_bq_solve_cost(ks=[5, 10, 20, 30, 50, 100, 200], cache_vv=True)
    print(f"{'k':>5}  {'mean ms':>10}  {'std ms':>10}")
    for r in solve_rows_cached:
        print(f"{r['k']:>5}  {r['mean_ms']:>10.4f}  {r['std_ms']:>10.4f}")

    speedup = np.mean([a["mean_ms"] / b["mean_ms"] for a, b in zip(solve_rows, solve_rows_cached)])
    print(f"\nAverage speedup from caching vv: {speedup:.1f}x")

    extrapolate_full_image(neighbor_rows, solve_rows, image_res=(800, 800), label=" -- naive (vv recomputed per pixel)")
    extrapolate_full_image(neighbor_rows, solve_rows_cached, image_res=(800, 800), label=" -- vv cached once per window size")
