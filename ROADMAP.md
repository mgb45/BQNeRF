# ROADMAP

Forward experiment plan. See [`README.md`](README.md) for the claim and
[`gs_experiment/results/FINDINGS.md`](gs_experiment/results/FINDINGS.md) for
what has been shown and what has been retracted. Ordered by priority.

## 0. Rewrite the paper to match the current method (Abstract/Intro left)

Related Work, Method, "Using the Uncertainty" and Experiments now state the
actual construction: the SH-coefficient posterior of
`rasterized_sh_precision.py` (Eq. `sh-precision`/`posterior`), the
Rademacher-probe computation of `sum_q beta_{q,i}^2`, posterior-ensemble
rendering (`render_posterior_ensemble.py`/`render_posterior_view_sweep.py`),
the calibration fit and cross-scene anchoring transfer of FINDINGS sections
9-10, and the two rejected extensions (cross-splat coupling, opacity-in-
posterior) as an ablations subsection -- all with real figures copied from
`gs_experiment/results/` and framed per FINDINGS section 11's verdict: a
specific closed form and its efficient computation for a specific part of
the model, not a category claim.

What is still open: the Abstract and Introduction are empty placeholders
and were out of scope for the Method rewrite -- they need to be written
against the now-current Related Work/Method/Experiments rather than the
retired construction. The draft has not been compiled (no LaTeX toolchain
in this environment); only static checks were run (label/ref/cite
resolution, brace and environment balance, figure paths) -- compile it with
`latexmk -pdf main.tex` and check page count against RA-L's 8-page limit
before submission, since six new tables/figures were added. Do this before
any submission.

## 1. Next-best-view selection

Read `arXiv:2511.09397` (OUGS: object-aware active view selection in 3DGS via
Gaussian-parameter-covariance-to-Fisher-information) before starting --
noted in FINDINGS section 11 as active competition for exactly this
direction, and it may already answer the question below.

Now the best-supported next step. Per-VIEW uncertainty tracks per-view
held-out error at Spearman 0.97 in the epistemic regime and 0.61 even on a
fully-observed checkpoint (FINDINGS section 8) -- and a per-view aggregate
is exactly what NBV consumes. Aggregate per-pixel uncertainty over each
candidate view, pick the next training view with it, and check whether
held-out error drops faster than a random/round-robin schedule. There is an
`nbv_out/` from an earlier attempt to build on.

## 2. `mic`, and the limits of a scalar `c`

Item 2 as originally posed is answered (FINDINGS section 10): `s` transfers,
`sigma_0` does not, and anchoring the floor to each scene's own training
residual recovers 66% of the calibration gain with no held-out views.

What is left is the one scene where that fails. `mic` has
`sigma_0/sigma_n = 2.86` against a mean of 1.77 -- a largely specular object,
where held-out error is much worse than the training residual implies,
because view-dependent appearance is exactly what a training residual cannot
see. `ficus` (2.89) is the same story with fine thin structure. A single
scalar `c` cannot know this; a cheap per-scene statistic that CAN might
(e.g. the fitted SH energy in the higher bands, or the spread of training
residuals across views rather than pooled over them). Worth one experiment
before concluding that a validation split is genuinely required for
specular scenes.

## 3. The angular-gap figure (partly done)

"The view nobody trained on". The measurement exists already
(`render_epistemic_regime.py`, FINDINGS section 8); what is missing is the
*figure*: a row of renders at increasing gap width with the uncertainty map
beneath, and a small inset of the camera sphere with the hole growing, so
the point lands without reading.

Which experiment to use depends on what is being claimed, and the two are
not interchangeable:

- **Claims about the uncertainty's own behaviour** (does it rise when you
  condition on less?) must be **frozen-map** -- nested camera removal from
  ONE checkpoint, as `render_posterior_view_sweep.py` does for view count.
  Comparing independently-trained checkpoints would confound the effect with
  splat count, positions, opacities, learned coefficients and the query-side
  weights all moving at once.
- **Claims about calibration against real error** must use **retrained**
  checkpoints (`gap_0..gap_4`). On a frozen map the reconstruction error does
  not change when you condition the posterior on fewer cameras -- the map was
  still fit to all of them -- so there is no epistemic error to predict and
  the question cannot be asked. This is why FINDINGS section 8 uses the
  retrained gap checkpoints deliberately, and it is not a lapse from the
  frozen-map discipline.

Worth keeping as a correctness check in its own right: nested camera removal
can only ever drop positive-semi-definite terms from `D_i`, so posterior
variance must be non-decreasing as the conditioning set shrinks
(`scripts/frozen_map_monotonicity_test.py`; passes on all 300k splats).

## 4. `u_spatial_BQ`, the finite-representation term

`gpu_uncertainty.compute_alpha_risk_batched` -- the real alpha weights' RKHS
worst-case risk under a position-only kernel -- is unaffected by the
FINDINGS section 0 defect and still implemented, but is currently orphaned:
the posterior-ensemble story does not use it. It answers a different
question (is the node set adequate?) than the coefficient posterior (are the
node values determined?). Note it carries no GP amplitude, so it is a
*relative* risk in units of length^-3, not a variance in colour^2, and
cannot simply be added to the ensemble variance without fitting a signal
amplitude `sigma_f^2`. Decide deliberately whether to fit that and combine,
or to drop the term.
