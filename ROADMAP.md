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

## 1. Comparative baselines, calibration-led

The evaluation protocol is pre-registered and frozen in
`gs_experiment/evaluation.py` (FINDINGS section 12.1) -- committed before any
baseline exists, and every method scored by it and no other. Lead with
calibration: OUGS reports no calibration numbers at all and
`zhao2026posterior` reports variance collapse inside the object, so that is
the unoccupied ground (FINDINGS section 12.2).

Reimplemented in our own harness, on IDENTICAL checkpoints:

- **Galappaththige-style** (`galappaththige2026predictive`) -- frozen map, per-
  splat SH channel fit by regularised linear least squares against real
  photometric residuals. The hard test: it is supervised on the very error
  we are scored against, so it *should* win, and the linear-LS normal
  equations are solvable with the CG/matvec operator already in
  `coupled_sh_posterior.py`. Implement it in its strongest form; losing to a
  strong implementation is worth more than beating a strawman.
- **FisherRF-style** (`jiang2024`) -- per-primitive diagonal Fisher over ALL
  Gaussian parameters, via the same Rademacher-probe backward. Tests directly
  whether restricting the posterior to appearance is a win or a loss.
- **Han-style coverage field** (`han2025viewdependent`) -- tests whether the
  Fisher weighting beats a hand-designed angular-coverage weighting.
- **Deep ensemble** -- N independent trainings, per-pixel variance. Not in the
  bibliography but universal, and `zhao2026posterior` benchmarks against it.
  At the calibrated recipe (below) this is ~1 h for 5 members x 7 scenes.
- **Model-free floors** -- angular coverage count, nearest-training-view angle,
  render gradient magnitude. The "is any of this needed" bar.

Published-numbers-only, stated as such: `jia2026rendering` and
`wu2026horseshoe` are in-the-loop methods whose fair reimplementation means
adopting their whole training recipe; `zhao2026posterior` and
`wu2026perturbed` are X-ray modality. Cite, do not reimplement badly.

Scenes: 7 NeRF-Synthetic plus **bonsai** (`bonsai_prepared/gap_0`, a real
Mip-NeRF 360 capture, 262 training views + 30-view eval, checkpoint already
trained). Every number this project has is synthetic; one real capture is
needed for external validity, and Mip-NeRF 360 overlaps OUGS's own dataset.

Training recipe for any arm needing retraining: the project's own
`DEFAULT_TRAIN_KWARGS` with `n_iters=6000, max_splats=100000,
densify_end=3000` -- 28.4 dB held-out at 30 lego views in 80 s. Drive
training through the Python API, NEVER the CLI: the CLI defaults to
`background_color=(0.05,0.05,0.05)` with no flag to change it, so a
CLI-trained checkpoint is fit against dark grey and then scored against
white-composited ground truth, costing ~17 dB of held-out PSNR (11.6 dB
against 28.4 dB for an identical budget). `run_next_best_view.py` now
asserts the two backgrounds match.

Capacity is a FACTOR, not a confound: run the post-hoc arms at both this
recipe and the 300k-splat `wide` checkpoints. If conclusions agree that is
robustness; if they differ that is a finding. Re-running the section 9-10
calibration at the smaller capacity is then a second transfer axis (does
(s, c) transfer across capacity as well as across scene?), not a chore.

## 2. Next-best-view selection (demoted)

Demoted from item 1 -- see FINDINGS section 12.2/12.3. The field is crowded
(ActiveNeRF, FisherRF, Bayes' Rays, GauSS-MI, OUGS, `xue2026`), competing
head-on means reimplementing four strong baselines on three datasets this
project does not use, and none of them measures calibration.

What remains genuinely novel here is one sharp question neither paper can
answer alone: section 7 found that adding geometry (opacity) to the posterior
made CALIBRATION worse, while OUGS includes all geometry parameters and wins
at VIEW SELECTION. Those are different questions, not a contradiction. Run
both uncertainties through the same acquisition loop on a common harness and
find out whether geometry helps view selection even where it hurts
calibration.

Per-VIEW uncertainty tracks per-view held-out error at Spearman 0.97 in the
epistemic regime and 0.61 on a fully-observed checkpoint (FINDINGS section
8), and a per-view aggregate is exactly what NBV consumes, so the signal is
there. Baselines: farthest-point (model-free, and the honest bar -- random
alone would flatter any method) and FisherRF-style acquisition. >=3 seeds:
a discarded run had a random arm go 11.56 -> 10.84 -> 10.90 dB.

## 3. `mic`, and the limits of a scalar `c`

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

## 4. The angular-gap figure (partly done)

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
