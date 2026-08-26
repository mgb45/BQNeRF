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
- `DirectionalKernel` (in `kernels.py`) / `bayesian_quadrature_directional`
  / `directional_posterior_variance` (in `quadrature.py`) — a von
  Mises-Fisher kernel on viewing direction, combined multiplicatively with
  a position kernel (position integrated as usual, direction evaluated
  pointwise at a query direction, since a rendered pixel looks in one
  direction, it doesn't integrate over a range of them). Lets the same BQ
  formalism catch "seen from a narrow cone of angles" epistemic
  uncertainty, not just spatial under-sampling — see FINDINGS.md §9.

## Running it

```
python -m pytest tests/ -v                                  # 27 correctness/sanity tests
python scripts/validate_milestone1.py                        # the milestone-1 experiment (1D)
python scripts/validate_trainable_kernel.py                   # fixed vs. fitted bandwidth (1D)
python scripts/validate_trainable_kernel_heldout.py            # is the fitted bandwidth held-out-valid?
python scripts/validate_2d_gap_experiment.py                  # the 2D image-plane bridge experiment
python scripts/benchmark_local_bq_scaling.py                  # GS-scale computational feasibility
python scripts/validate_directional_isolation.py              # directional kernel, isolated
python scripts/validate_directional_combined.py                # position+direction vs. position-only
```

`validate_milestone1.py` prints an accuracy/calibration summary and writes
two plots to `bq_splat/results/`. `validate_trainable_kernel.py` and
`validate_trainable_kernel_heldout.py` print comparison tables of
fixed/fitted/held-out-fitted BQ vs. Riemann sum. `validate_2d_gap_experiment.py`
writes a heatmap comparing the true 2D signal/splat placement against local
BQ variance. `benchmark_local_bq_scaling.py` prints neighbor-lookup and
local-solve timing at up to 10^6 synthetic splats, with no plot output.
`validate_directional_isolation.py` and `validate_directional_combined.py`
write plots showing directional-coverage effects on posterior variance,
alone and combined with the spatial signal.

## Findings so far

See [`results/FINDINGS.md`](results/FINDINGS.md) for the current-conclusions
summary — short version: the ported math is correct, the raw-accuracy gap
against naive Riemann summation is understood and fixable (a bandwidth-
fitting issue, not a fundamental limitation), posterior variance is
reasonably calibrated and rises in genuinely under-resolved regions, the
computational-scaling concern that motivated an early GPU-rewrite worry
was resolved on CPU alone, and the directional-kernel extension (does the
same formalism catch viewing-angle coverage, not just spatial coverage?)
works at toy scale. A formal proof that the BQ posterior mean recovers
alpha compositing exactly, with the posterior variance as a *provable*
error bound, is in
[`PROOF_alpha_compositing_equivalence.md`](PROOF_alpha_compositing_equivalence.md).

For the complete chronological account — every bug, every intermediate
number — see [`results/ARCHIVE_FULL_LOG.md`](results/ARCHIVE_FULL_LOG.md).
For how these toy-scale results held up on real Gaussian-Splatting data,
see [`../gs_experiment/results/FINDINGS.md`](../gs_experiment/results/FINDINGS.md).
