# Roadmap

For the theory, see [`README.md`](README.md). For results, see
`gs_experiment/results/FINDINGS.md`. This document is the forward
experiment plan: what to try next, in priority order. Training directly
under the likelihood and alternative kernels are next up — everything
else in README's "What's been tested" list follows behind them.

## 1. Train directly under the likelihood

Every result so far computes BQ variance *after* training, read off a
checkpoint trained by ordinary photometric loss and gradient-triggered
densification. The open question: does the BQ posterior help *during*
training, not just diagnose a finished checkpoint?

`gs_experiment/scripts/train_minimal_gsplat.py` already exposes the hooks
needed to test this without new trainer code:

- `densify_criterion="bq_variance"` — swap the densification trigger from
  gsplat's view-space positional gradient to closed-form BQ position-only
  variance, queried at each splat's own position via
  `compute_per_splat_bq_variance`.
- `nll_weight` — an uncertainty-weighted Gaussian-NLL auxiliary loss,
  evaluated on a grid of real ray-surface points at `nll_interval`
  cadence.
- `bq_densify_min_opacity` — floors the BQ-variance densification score so
  splats in empty, low-opacity space don't outscore splats near real,
  under-resolved geometry.

**Plan**: a new `gs_experiment/scripts/likelihood_training_experiment.py`
that trains matched variants on one real scene (same seed, same other
hyperparameters, reusing `train_minimal_gsplat.train()`'s kwargs) and
evaluates on both training views and a genuinely disjoint held-out set:

- baseline: gradient densify, no NLL term
- BQ-variance densify, NLL off
- gradient densify, NLL on
- BQ-variance densify, NLL on

Design questions to resolve empirically rather than assume:

- Does `bq_densify_min_opacity` need tuning per-scene, and does it trade
  off splat-count growth against held-out quality (a floor that's too
  loose can let densification grow unboundedly; too tight starves it)?
- Is the NLL grid resolution/frequency (`nll_grid_res`, `nll_interval`)
  fine/frequent enough to actually move training, or does it need to be
  denser at real compute cost?
- Compare against `train_with_reference_strategy`'s gsplat-native
  densification as a stronger baseline than this project's from-scratch
  gradient path, so a win or loss isn't an artifact of a weak baseline.

Report whichever way this comes out — improvement, no effect, or
regression — in `gs_experiment/results/FINDINGS.md`, with the concrete
next untested step named explicitly rather than left as a vague "needs
more work."

**Status**: done, first installment — see `gs_experiment/results/FINDINGS.md`
section 1. Mixed, honest result on the lego `narrow` (12-view) pool at
matched splat budget: `bq_variance` densification is a real but modest win
(+0.83dB train / +0.29dB held-out PSNR vs. gradient densification), the
`nll_weight` auxiliary loss term alone is a no-op-to-mild-negative
(-0.14dB / -0.21dB), and `bq_densify_min_opacity` is a genuine efficiency
lever (62% fewer splats, no held-out quality cost). Next untested step:
differentiate the NLL term's variance through the BQ posterior itself
(currently detached) rather than iterating further on the auxiliary-loss
weighting as-is.

**Status update (bug fix + re-run)**: this experiment's own `LEGO_BQ_SIGMA`
constant was one of the stale, pre-`SplatScene.colors`-fix bandwidths
(0.0694 -> 0.13926, `FINDINGS.md` section 4) and had not yet been re-run
against. Now re-run in full (`FINDINGS.md` section 1's updated tables).
Two of three headline claims hold essentially unchanged: BQ-variance
densification's win is confirmed (+0.85dB train / +0.24dB held-out, was
+0.83/+0.29), and the NLL loss term's held-out mild-negative effect is
confirmed (-0.14dB held-out, was -0.21dB — same sign, similar size). The
opacity floor's claim is narrowed: "no held-out quality cost" still holds
(62.4% fewer splats, essentially flat held-out PSNR), but the previously
reported small held-out *gain* (+0.16dB) is now -0.02dB (flat, not a
measured win) — that specific framing should not be repeated. Next
untested step unchanged.

## 2. Alternative kernels

`gs_experiment/kernels.py` currently has two families — `RBFKernel` and
`MaternKernel` — behind the same `Kernel` interface, with bandwidths fit
by GP log marginal likelihood (`hyperparams.py`). The method's posterior/
variance machinery is kernel-agnostic by design; the open question is
which kernel families are actually worth offering and what each one buys.

**Plan**:

1. Add at least one more kernel family as a new `Kernel` subclass — e.g.
   a rational-quadratic kernel (a continuous mixture of RBF bandwidths,
   which may handle scenes with mixed fine/coarse structure better than a
   single-bandwidth RBF) or a periodic kernel (relevant for any scene
   content with repeating structure). Fit its bandwidth the same way as
   RBF/Matérn.
2. Build a small `gs_experiment/kernel_family_ablation.py` that, per
   checkpoint (the lego wide/500 and coverage-gap checkpoints already
   used by the kept results), fits every kernel family's hyperparameters
   and computes a clearly-defined comparison metric — sparsity
   correlation (does variance track deliberately sparse regions) and/or
   calibration against real held-out rendering error. This metric code
   doesn't exist in the current repo and needs to be written fresh, kept
   small and single-purpose rather than resurrecting a large multi-check
   eval script.
3. Report per-checkpoint winners honestly — expect a trade-off (different
   kernels may be better for different properties or scene geometry)
   rather than assuming a single universally-best kernel, and say so
   plainly if that's what the data shows.
4. Keep kernel/bandwidth choice a pluggable, exposed parameter throughout
   (already true via `pixel_uncertainty.LocalUncertaintyEngine` and
   `splat_scene.fit_kernel_hyperparams`) — this is a strength of the
   method, not a loose end to resolve into one hardcoded default.

**Status**: done — see `gs_experiment/results/FINDINGS.md` section 2 (and
its 2b addendum). Added `RationalQuadraticKernel` (alpha fixed at 1.0) and
ran the ablation on the real `wide`/`budget_500` checkpoints of all 7
NeRF-Synthetic scenes this project's other kept results use (chair,
drums, ficus, hotdog, lego, mic, ship — `materials` excluded, same
documented reason as `scripts/render_scene_gallery.py`), not just lego.
Genuine trade-off, partially universal: RBF is dramatically better
calibrated (Gaussian-NLL score) in *all* 14 scene/checkpoint combinations,
not just lego's 2 — a fully universal result. RationalQuadratic gives the
best sparsity-tracking signal (amplitude-normalized `variance/prior_variance`
ratio) on the sparse checkpoint in 6 of 7 scenes, but the dense-checkpoint
raw-variance confound that motivated the ratio metric in the first place
turns out to be scene-dependent, not universal (present on drums/hotdog/
lego/mic, absent on chair/ship, mixed on ficus, tracking each scene's own
real local splat density relative to the engine's `max_neighbors=60`
cap) — so the dense-checkpoint sparsity-ratio winner is more mixed across
scenes than the lego-only result suggested. Full per-scene numbers in
`gs_experiment/results/kernel_family_ablation_results.json` and
`paper/main.tex`'s appendix (Tables II-VIII).
Matern-3/2 did not win outright on any metric/checkpoint.

**Status update (bug fix + re-run)**: a real bug (`SplatScene.colors` was
raw SH coefficients, not real color — see `FINDINGS.md` section 4) was
fixed and the whole ablation was re-run. Fitted bandwidths shifted up
1.45x-2.00x (RBF sigma) to as much as 18.2x (Matern/RQ in one case); the
"RBF wins NLL in all 14/14 checkpoints" claim is no longer strictly true
(now 12/14, still the large majority, often by orders of magnitude) and
"RQ wins the sparse-checkpoint sparsity-ratio in 6/7 scenes" weakened to
4/7 (RBF now wins 3/7). Both general recommendations (RBF for NLL-safety,
RQ for sparsity-tracking) still hold as the dominant pattern, just less
cleanly than originally reported. `kernel_family_ablation_results.json`
has been overwritten with the corrected numbers; `paper/main.tex`'s
Tables II-VIII have **not** been updated (need review with the user
first — see `FINDINGS.md` section 4a for full before/after deltas).

## 3. Floater-flagging follow-up experiment

README already confirms the signal flags GS-training floaters but marks
this "needs an experiment." The floater mechanism is: a floater is, by
construction, a splat whose local render-weight spread (`Sigma_q`) is
anomalously large relative to its neighbors. Test whether that spread can
be used *during* training as a targeted regularizer or pruning criterion
— narrower in scope than item 1's general likelihood-training question,
since it targets one specific known failure mode rather than training
quality broadly.

## 4. Calibration experiment

README also marks "is the number calibrated" as tested-but-"needs an
experiment" — quantify whether posterior variance is calibrated against
real held-out rendering error (not just correlated with sparsity), across
the standard NeRF-Synthetic scenes already used elsewhere in this
project. Reuse whatever calibration metric gets built for item 2's kernel
ablation rather than writing a second, separate metric.

**Status update**: item 2's calibration metric (and every number in
`paper/main.tex`'s Tables II-VIII) turned out to rest on an incoherent
pairing — `u_BQ` (the variance around the BQ posterior mean `C_BQ`)
scored against the squared error of a *different* quantity, the real
alpha-compositing renderer's `C_alpha`. See
`gs_experiment/results/FINDINGS.md` section 3 for the fix (a proper RKHS
risk formulation, `rendering_aware_alternative_weight_risk`, applied to
5 coherent mean/uncertainty pairings across all 7 scenes × 2 checkpoints)
and its result: none of the real (non-null-baseline) variants beat a
trivial constant-variance model on Gaussian NLL, but pairing the real
`C_alpha` with `R_alpha = u_BQ + (C_BQ-C_alpha)^2` (variant 3) is
measurably more robust and better-calibrated (coverage sense) than the
existing practice, and `C_BQ` itself does not render well enough (Phase
B: ~1.6dB PSNR behind real alpha compositing on average, ~50% of its raw
predictions out of `[0,1]` range, ~49% negative BQ weights) to replace
alpha compositing as the deployed mean. Tables II-VIII have NOT yet been
updated to reflect this — that edit needs review with the user first.

**Status update 2 (bug fix + re-run)**: the same `SplatScene.colors` bug
from item 2's update above also affected every number in this section.
Re-run (`FINDINGS.md` section 4b) shows the core recommendation not only
holds but is demonstrated more starkly: variant 1 (existing post-hoc)'s
NLL on dense (`wide`) checkpoints got ~10-20x *worse* after the fix (a
real, mechanistically-understood consequence of `u_BQ` shrinking 5x-16x
at real query points while the real rendering error it's scored against
stayed unchanged — see FINDINGS.md for the full explanation, this is not
a regression from the fix), while variant 3 (`R_alpha`) held flat or
improved on 6/7 of those same checkpoints and now even beats the trivial
constant-variance baseline outright in 2/14 cases (previously 0/14) and
wins AUSE in 8/14 (previously 3/14, now a majority). Separately, Phase
B's "`C_BQ` does not render competitively" verdict is substantially
overturned by the fix: the ~56% out-of-range rate drops to ~5% (confirming
it was largely a units-bug artifact — a near-0 raw-SH prediction reads as
real color ~0.5, not black), and `C_BQ` now *beats* real alpha compositing
on PSNR in 8/8 checked views (was 1/8), though SSIM is only roughly on par
(not a clean win). The ~49% negative-BQ-weight rate is unchanged — a real,
values-independent structural property, not a units artifact. Tables
II-VIII still have NOT been updated — needs review with the user, now
with a larger, more nuanced set of deltas than before.

**Status update (Tier 2 risk, now resolved by actually regenerating all
three figures)**: all three headline PNGs have been regenerated with the
corrected sigma/kappa/colors and measured directly — see `FINDINGS.md`
section 6 (section 4c above is left in place as the original estimate,
marked superseded). `scene_gallery.png` and `lego_splat_sweep.png`
self-corrected on a bare re-run as expected (sigma now ~0.09-0.24 across
checkpoints, PSNR numbers unchanged and confirmed to exactly match
`paper/main.tex`'s quoted spatial-coverage figures). A real,
pre-existing bug unrelated to the colors fix was found and fixed while
regenerating `lego_splat_sweep.png` (`KeyError: 'budget_rows'` — the
script's row-building was never updated to match a later refactor of
`plot_gallery`'s row format; fixed in
`gs_experiment/scripts/render_splat_sweep_gallery.py`).
`coverage_uncertainty_sweep.png` picked up the corrected `LEGO_GAP_SIGMA`
(0.13926) automatically (confirmed: it imports the constant directly from
`real_directional_coverage_experiment.py`, no separate hardcoded value of
its own) — its PSNR numbers also match the paper exactly
(35.12/29.19/22.91/15.13/16.09dB vs. quoted 35.1/29.2/22.9/15.1/16.1dB),
but **its mean raw-variance numbers moved in the opposite direction from
the section-4c estimate**: measured $0.084 \to 1.254 \to 3.343 \to 8.886
\to 11.769$ (was estimated to grow ~6-8x to something like
$5$-$7 \to \dots \to 500$-$600$; instead it *shrank* ~6.6x-9.8x from the
paper's currently-quoted $0.82 \to 8.54 \to 22.12 \to 59.45 \to 78.85$).
Traced to a real, verified mechanism: `RBFKernel` is a normalized Gaussian
*density* (`gs_experiment/kernels.py`), whose own self-covariance
`k(x,x)=1/(sigma*sqrt(2*pi))` *shrinks* as sigma grows, and the 3D
position kernel is a product of three of these — so self-covariance scales
as `1/sigma^3`; the ~2.0x sigma correction predicts almost exactly the
measured ~6.6x-9.8x variance decrease (`2.0^3≈8`). The qualitative
"grows monotonically, roughly two orders of magnitude with the gap"
framing in the paper is unaffected and, if anything, slightly
strengthened (new max/min ratio ~140x vs. old ~96x). All four candidate
PNGs (`scene_gallery.png`, `scene_gallery_500.png`, `lego_splat_sweep.png`,
`coverage_uncertainty_sweep.png`) are git-tracked, not gitignored;
three were regenerated and are now modified in the working tree (not
committed, per this task's own scope);
`scene_gallery_500.png` was left untouched — it is a stale, orphaned file
from a superseded pipeline stage, produced by no current script path and
not referenced anywhere in `paper/main.tex` (only `scene_gallery.png` is).
`paper/main.tex`'s text has **not** been edited — the coverage-sweep's
quoted absolute numbers need the user's own review given they moved in the
opposite direction from what was previously estimated.

## 5. Next-best-view selection evaluation

README lists this as Todo. Use posterior variance to pick the next
training view (greedily query variance across a candidate view pool, add
the highest-variance view, retrain/fine-tune) and check whether it
improves held-out reconstruction faster than a round-robin or random view
schedule, on one or more of the standard NeRF-Synthetic scenes.
