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

## 5. Next-best-view selection evaluation

README lists this as Todo. Use posterior variance to pick the next
training view (greedily query variance across a candidate view pool, add
the highest-variance view, retrain/fine-tune) and check whether it
improves held-out reconstruction faster than a round-robin or random view
schedule, on one or more of the standard NeRF-Synthetic scenes.
