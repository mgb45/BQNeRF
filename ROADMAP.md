# ROADMAP

Forward experiment plan. See [`README.md`](README.md) for the claim and
[`gs_experiment/results/FINDINGS.md`](gs_experiment/results/FINDINGS.md) for
what has been shown and what has been retracted. Ordered by priority.

## 1. Next-best-view selection

Now the best-supported next step. Per-VIEW uncertainty tracks per-view
held-out error at Spearman 0.97 in the epistemic regime and 0.61 even on a
fully-observed checkpoint (FINDINGS section 8) -- and a per-view aggregate
is exactly what NBV consumes. Aggregate per-pixel uncertainty over each
candidate view, pick the next training view with it, and check whether
held-out error drops faster than a random/round-robin schedule. There is an
`nbv_out/` from an earlier attempt to build on.

## 2. Does the calibration transfer?

The two calibration constants (`s ~ 5.5-6.4`, `sigma_0 ~ 0.018-0.043`) are
currently fitted per checkpoint on held-out views (FINDINGS section 9). They
came out close on two very different lego checkpoints, which hints they may
transfer -- but that is not established, and it is the difference between
"calibrated with a validation split" and "calibrated out of the box". Fit on
one scene, score on the other six; if `s` is stable, report it as a constant
of the construction rather than a fitted parameter.

Do NOT reach for a richer posterior. Three have now been implemented,
validated and measured, and all three cost more and calibrate worse:
cross-splat coupling (section 6, 1000x), opacity in the posterior (section
7, 2x), and a render-derived spatially-varying aleatoric floor (section 9,
free but no gain).

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
