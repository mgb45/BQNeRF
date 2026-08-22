# bq_splat — milestone 1: BQ math validated on toy 1D rays

Implements and validates the Bayesian-quadrature math this project's
[ROADMAP.md](../ROADMAP.md) calls for, at cheap 1D-synthetic-ray scale,
before touching gsplat. No torch/gsplat dependency — pure numpy/scipy.

## Layout

- `kernels.py` — RBF and Matérn-3/2 kernels, each with a closed-form or
  numerically-integrated mean embedding `v(x, a, b)` and double integral
  `vv(a, b)`. The RBF `v` is cross-checked against `models/nerf.py`'s exact
  erf-based formula in `tests/test_kernels.py`.
- `quadrature.py` — `bayesian_quadrature(nodes, values, kernel, a, b)`:
  posterior mean/variance of the integral given point evaluations. Uses a
  *relative* jitter (scaled to the Gram matrix's own diagonal) rather than a
  fixed constant — see "Numerical conditioning" below for why that matters.
- `reference.py` — the naive piecewise-constant ("alpha compositing style")
  Riemann-sum estimator, and a numerically-exact ground-truth integral, both
  used as baselines.
- `toy_scene.py` — synthetic 1D signals (mixtures of Gaussian bumps) and
  node-placement strategies, including a deliberate sparse-coverage "gap"
  that's interior to the domain (visible, not occluded) but under-sampled.
- `hyperparams.py` — fits the kernel bandwidth (RBF sigma / Matern rho) to
  data by maximizing the GP log marginal likelihood, instead of the
  hardcoded bandwidth `models/nerf.py` and the rest of this package default
  to. No torch/autodiff — a log-spaced grid search plus a bounded 1D
  refinement, same as classic GP-library hyperparameter fitting.
- `toy_scene_2d.py` / `ProductKernel` (in `kernels.py`) / `bayesian_quadrature_nd`
  (in `quadrature.py`) — the 2D, image-plane generalization: splat centers
  scattered over a patch rather than along a ray's depth axis, which is the
  geometry a real GS scene actually has. `ProductKernel` builds a D-D kernel
  as a product of 1D kernels per axis — exact for RBF, so `v`/`vv` reduce to
  products of the already-tested 1D formulas with no new integration code.
  `fit_kernel_param_pooled` (in `hyperparams.py`) fits one shared bandwidth
  across many datasets, for testing whether a single fitted bandwidth
  generalizes across scenes instead of needing to be refit per instance.

## Running it

```
python -m pytest tests/ -v                                  # 21 correctness/sanity tests
python scripts/validate_milestone1.py                        # the milestone-1 experiment (1D)
python scripts/validate_trainable_kernel.py                   # fixed vs. fitted bandwidth (1D)
python scripts/validate_trainable_kernel_heldout.py            # is the fitted bandwidth held-out-valid?
python scripts/validate_2d_gap_experiment.py                  # the 2D image-plane bridge experiment
python scripts/benchmark_local_bq_scaling.py                  # GS-scale computational feasibility
```

`validate_milestone1.py` prints an accuracy/calibration summary and writes
two plots to `bq_splat/results/`. `validate_trainable_kernel.py` and
`validate_trainable_kernel_heldout.py` print comparison tables of
fixed/fitted/held-out-fitted BQ vs. Riemann sum. `validate_2d_gap_experiment.py`
writes a heatmap comparing the true 2D signal/splat placement against local
BQ variance. `benchmark_local_bq_scaling.py` prints neighbor-lookup and
local-solve timing at up to 10^6 synthetic splats, with no plot output.

## Findings so far

See [`results/FINDINGS.md`](results/FINDINGS.md) for the full write-up.
Short version: with a hardcoded kernel bandwidth, BQ's posterior mean loses
to a plain Riemann sum on raw integral accuracy at every node count tested —
consistent with the original NeRF-BQ result recorded in this repo's history.
But that turns out to be substantially a fixed-bandwidth mismatch, not a
fundamental limitation: fitting the bandwidth per scene via marginal
likelihood (`hyperparams.py`) closes most of the gap, and fitted Matern
actually beats Riemann at n=20 and n=40 nodes. Separately, BQ's posterior
variance is reasonably well correlated with its own actual error for both
kernels (~0.7), and a deliberately under-sampled-but-visible region shows
~3.9x higher local BQ variance than well-covered regions on average —
peaking (~1.2) right at the region's leading edge, where sparse coverage
first meets real signal structure — a toy-scale replication of the paper's
central differentiation claim. One implementation lesson worth carrying into
the gsplat port: irregular (as opposed to evenly stratified) node placement
can push the Gram matrix condition number past 1e18 with a naive fixed
jitter; a jitter relative to the kernel's own scale fixes it and materially
changes downstream numbers (see FINDINGS.md).

Separately, the differentiation effect survives moving from a 1D ray-depth
domain to a 2D image-plane domain with scattered splat-center placement —
the geometry a real GS scene actually has — with a 4.85x inside/outside
variance ratio and the same "peaks near the coverage boundary" shape found
in 1D (see FINDINGS.md §6 and `results/gap_experiment_2d.png`).

A held-out check (§7) refines the bandwidth-fitting story: it's real and
generalizes for Matern (a bandwidth fit once on a calibration set nearly
matches an in-sample oracle on unseen scenes) but not for RBF (the
population-optimal RBF bandwidth turned out to be almost exactly the
original hardcoded 0.35 — RBF's earlier per-scene gains were mostly
overfitting to each scene's specific sample layout). And a computational
scaling check (§8) found the bottleneck ROADMAP.md worried about — an
expensive linear solve at GS scale — wasn't the real one: profiling found
94% of per-query cost was a numerically-integrated `vv` term, fixed exactly
(not approximated) by caching it per window size, since it's provably
position-independent for a fixed-size window under a stationary kernel.
That plus a KD-tree for neighbor lookup takes a naive ~2,400-3,000s
single-threaded per-800x800-image estimate down to ~140-420s, on CPU, with
up to a million synthetic splats — before any GPU code is written.
