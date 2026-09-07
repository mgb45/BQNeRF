# bq_splat — milestone 1: BQ math validated on toy 1D rays

Implements and validates the Bayesian-quadrature math this project's
[ROADMAP.md](../ROADMAP.md) calls for, at cheap 1D/2D-synthetic-ray
scale, before touching gsplat. No torch/gsplat dependency — pure
numpy/scipy.

## Layout

- `kernels.py` — RBF and Matérn-3/2 kernels, each with a closed-form or
  numerically-integrated mean embedding `v(x, a, b)` and double integral
  `vv(a, b)`. `ProductKernel` builds a D-D kernel as a product of 1D
  kernels per axis (exact for RBF). `DirectionalKernel` adds a von
  Mises-Fisher factor over viewing direction, combined multiplicatively
  with a position kernel.
- `quadrature.py` — `bayesian_quadrature(nodes, values, kernel, a, b)`:
  posterior mean/variance of the integral given point evaluations, plus
  the `_nd` (2D image-plane) and `_directional` generalizations. Uses a
  *relative* jitter (scaled to the Gram matrix's own diagonal) rather than
  a fixed constant, for numerical conditioning. Also
  `bayesian_quadrature_rendering_aware` / `renderer_centered_residual_variance`:
  the rendering-aware construction (see `render_weight.py` below) --
  `k_q(xi, xi') = a_q(xi) k_base(xi, xi') a_q(xi')` for a query-specific
  renderer weight `a_q`, in place of `_nd`'s uniform-box integration
  domain.
- `render_weight.py` — `GaussianRenderWeight`: `a_q = T_q sigma G_q`
  modeled as an unnormalized Gaussian bump (amplitude, center,
  covariance), for the closed-form rendering-aware quadrature above.
- `reference.py` — the naive piecewise-constant ("alpha compositing style")
  Riemann-sum estimator, and a numerically-exact ground-truth integral, both
  used as baselines.
- `toy_scene.py` — synthetic 1D and 2D signals (mixtures of Gaussian
  bumps) and node-placement strategies, including a deliberate
  sparse-coverage "gap" that's interior to the domain (visible, not
  occluded) but under-sampled.
- `hyperparams.py` — fits the kernel bandwidth (RBF sigma / Matern rho) to
  data by maximizing the GP log marginal likelihood, instead of the
  hardcoded bandwidth the rest of this package defaults to. No
  torch/autodiff — a log-spaced grid search plus a bounded 1D refinement,
  same as classic GP-library hyperparameter fitting. `fit_kernel_param_pooled`
  fits one shared bandwidth across many datasets, for testing whether a
  single fitted bandwidth generalizes across scenes.
- `validate.py` — the CLI that runs all of the checks below against
  `toy_scene.py`'s synthetic signals and writes the plots referenced in
  `results/FINDINGS.md`.

## Running it

```
python -m pytest tests/ -v                     # correctness/sanity tests
python -m bq_splat.validate --check accuracy                  # milestone-1 experiment (1D)
python -m bq_splat.validate --check trainable-kernel           # fixed vs. fitted bandwidth (1D)
python -m bq_splat.validate --check trainable-kernel-heldout   # is the fitted bandwidth held-out-valid?
python -m bq_splat.validate --check 2d-gap                     # 2D image-plane bridge experiment
python -m bq_splat.validate --check alpha-compositing          # BQ posterior mean == alpha compositing, empirically
python -m bq_splat.validate --check directional-isolation      # directional kernel, isolated
python -m bq_splat.validate --check directional-combined       # position+direction vs. position-only
python -m bq_splat.validate --check scaling                    # GS-scale computational feasibility
python -m bq_splat.validate --check rendering-aware             # rendering-aware BQ vs. box-style BQ
```

`--check accuracy` prints an accuracy/calibration summary and writes two
plots to `bq_splat/results/`. `--check trainable-kernel[-heldout]` prints
comparison tables of fixed/fitted/held-out-fitted BQ vs. Riemann sum.
`--check 2d-gap` writes a heatmap comparing the true 2D signal/splat
placement against local BQ variance. `--check scaling` prints
neighbor-lookup and local-solve timing at up to 10^6 synthetic splats,
with no plot output. `--check directional-isolation`/`directional-combined`
write plots showing directional-coverage effects on posterior variance,
alone and combined with the spatial signal. `--check rendering-aware`
builds a genuine ray/pixel scene with a real transmittance-weighted
rendering functional and shows that an occluded splat with a wrong color
barely moves the new rendering-aware BQ mean while it substantially moves
the old box-style BQ mean fed the same raw colors -- see
`PROOF_alpha_compositing_equivalence.md` section 7 for why the old
approach has this gap.

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

For how these toy-scale results held up on real Gaussian-Splatting data,
see [`../gs_experiment/results/FINDINGS.md`](../gs_experiment/results/FINDINGS.md).
