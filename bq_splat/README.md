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

## Running it

```
python -m pytest tests/ -v                        # 11 correctness/sanity tests
python scripts/validate_milestone1.py              # the actual milestone-1 experiment
```

The script prints an accuracy/calibration summary and writes two plots to
`bq_splat/results/`.

## Findings so far

See [`results/FINDINGS.md`](results/FINDINGS.md) for the full write-up.
Short version: BQ's posterior mean still loses to a plain Riemann sum on raw
integral accuracy at every node count tested — consistent with the original
NeRF-BQ result recorded in this repo's history, and further evidence that
"beat accuracy" is the wrong frame for this paper (per ROADMAP.md). But BQ's
posterior variance is reasonably well correlated with its own actual error
for both kernels (~0.7), and a deliberately under-sampled-but-visible region
shows ~3.9x higher local BQ variance than well-covered regions on average —
peaking (~1.2) right at the region's leading edge, where sparse coverage
first meets real signal structure — a toy-scale replication of the paper's
central differentiation claim. One implementation
lesson worth carrying into the gsplat port: irregular (as opposed to evenly
stratified) node placement can push the Gram matrix condition number past
1e18 with a naive fixed jitter; a jitter relative to the kernel's own scale
fixes it and materially changes downstream numbers (see FINDINGS.md).
