# ROADMAP

Forward experiment plan. See [`README.md`](README.md) for the claim and
[`gs_experiment/results/FINDINGS.md`](gs_experiment/results/FINDINGS.md) for
what has been shown and what has been retracted. Ordered by priority.

## 1. Geometry in the posterior

Positions, scales and opacities are held fixed; only appearance is sampled.
An error of geometric origin -- a floater in the wrong place with a
confidently-fit colour -- need not light up. This is now the leading suspect
for the weak object-pixel error correlation (FINDINGS section 5), since
cross-splat coupling has been implemented, validated and found NOT to
explain it (FINDINGS section 6).

Opacity is the cheapest extension and the one most likely to buy the floater
story: it enters the render with a computable derivative, and sampling it
costs nothing extra at render time (a draw is still one render).

## 2. Calibration against real held-out error

Report AUSE (rank-based, robust to any residual scale error) alongside
Gaussian NLL (not robust, and therefore the real test of whether
`sigma_n^2` and `Lambda` are right). **Always restricted to object pixels** --
whole-frame correlations on NeRF-Synthetic are dominated by the
object/background split and report the silhouette, not calibration. Use the
block-diagonal posterior: coupling costs 1000x and does not improve the
correlation (FINDINGS section 6).

## 3. The angular-gap figure

"The view nobody trained on": remove training cameras within a growing
angular cone around one direction and show uncertainty rising specifically
there. Do this **frozen-map** (nested camera removal from one checkpoint,
as `render_posterior_view_sweep.py` does for view count), not by comparing
the independently-trained `gap_0..gap_4` checkpoints -- those confound the
effect with splat count, positions, opacities, learned coefficients and the
query-side weights all moving at once.

Worth keeping as a correctness check in its own right: nested camera removal
can only ever drop positive-semi-definite terms from `D_i`, so posterior
variance must be non-decreasing as the conditioning set shrinks. A violation
is unambiguous evidence of an implementation bug, independent of retraining,
kernel choice or function-class limitations.

## 4. Next-best-view selection

Aggregate per-pixel uncertainty per candidate view and pick the next
training view with it; check whether held-out error drops faster than a
random/round-robin schedule. This is the "so what" -- it converts the work
from a diagnostic into a tool. Depends on 2: a signal not yet shown to
correlate with real error is a weak basis for choosing views.

## 5. `u_spatial_BQ`, the finite-representation term

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
