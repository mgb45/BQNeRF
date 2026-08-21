# Milestone 1 findings: BQ math on toy 1D rays

Ran via `scripts/validate_milestone1.py`. Numbers below are from that
script's default settings (200 trials per node count, domain [0, 10],
RBF sigma=0.35, Matérn-3/2 rho=0.5) and are reproducible with the seeds
hardcoded in the script.

## 1. Accuracy: BQ vs. naive Riemann sum

| nodes | Riemann MAE | BQ-RBF MAE | BQ-Matérn32 MAE |
|-------|-------------|------------|------------------|
| 5     | 1.173       | 1.701      | 1.524            |
| 10    | 0.701       | 1.134      | 0.979            |
| 20    | 0.358       | 0.494      | 0.434            |
| 40    | 0.171       | 0.241      | 0.161            |

BQ loses to the naive piecewise-constant Riemann-sum estimator at every
node count except a near-tie at n=40 for Matérn. This reproduces, at toy
scale and with a completely independent implementation, the same qualitative
result already on record for this project (the original NeRF-BQ PSNR
experiment in `inspect_model.ipynb`/`figs/Ns_*.png`, where BQ also lost to
standard quadrature). Two independent implementations losing the same way is
reasonably strong evidence this isn't a codebase-specific bug — it's a real
property of this formulation, at least with these test signals and a single
fixed kernel bandwidth. This is consistent with ROADMAP.md's framing:
raw accuracy is not the claim to defend.

## 2. Calibration: does BQ variance track BQ's actual error?

Correlation between `|BQ error|` and BQ posterior std, pooled across all
node counts:

- RBF: **0.706**
- Matérn-3/2: **0.722**

Both are reasonably well calibrated and close to each other. This is worth
flagging explicitly because an earlier run of this same experiment, before
a numerical-conditioning fix (see §4), showed RBF at only 0.082 — which
would have read as "Matérn is dramatically better calibrated than RBF," a
tempting and clean-sounding result. It wasn't real. Fixing the actual bug
made the two kernels look similar. This is recorded here as a caution against
over-reading small-scale results before checking for numerical artifacts —
and as the reason the codebase's jitter is now relative rather than fixed.

A real kernel-choice comparison (varying bandwidth/smoothness relative to
the signal's own structure, more kernels, statistical testing across many
more trials) is still open and is exactly the "cleanest, least-contested
novelty claim" ROADMAP.md points to — this run doesn't settle it, it mostly
rules out one artifact that would have produced a false positive.

## 3. Gap experiment: does variance rise in an under-resolved-but-visible region?

A region interior to the domain (not occluded, not near an edge) was given
deliberately sparse local node coverage while containing real high-frequency
signal structure (two overlapping narrow bumps).

- Mean local BQ variance **inside** the gap: 0.546
- Mean local BQ variance **outside** the gap: 0.142
- Ratio: **3.85x**

(An earlier version of this script had a plotting bug — a local window with
fewer than 2 nodes was skipped and recorded as `NaN` instead of computing
the variance `bayesian_quadrature` already supports for those cases. That
silently broke the plotted curve inside the gap, at exactly the points that
should matter most, and also biased the "inside"/"outside" means reported
in an earlier draft of this file — 0.456/0.070/6.5x — since points nearest
the gap's sparsest coverage were the ones being dropped. The numbers above
are from the corrected script.)

See `gap_experiment.png`. The curve is more specific than "rises uniformly
inside the sparse region": variance peaks (~1.2) right at the gap's leading
edge, where sparse coverage first meets the two-bump high-frequency
structure, then stays moderately elevated (~0.4-0.6) through the gap's
interior before dropping off outside it. That's still consistent with the
paper's differentiation claim — variance is responding to under-sampling
*relative to local signal structure*, not to node count in isolation — but
it's a more specific and more interesting shape than a flat elevated
plateau, and is worth designing the eventual real (GS-based) differentiation
experiment around: expect the effect to be strongest at the boundary where
coverage drops off against real structure, not necessarily uniform across
an entire under-covered region.

## 4. Numerical conditioning: a real implementation lesson

Random (irregular) node placement occasionally produces near-duplicate
nodes. At n=40 nodes uniform on [0, 10] with RBF sigma=0.35, worst-case Gram
matrix condition number over 200 draws was measured at **1.5e18** — beyond
double-precision's reliable range — causing one specific n=40 trial's BQ-RBF
error to spike to 1.38 (vs. a smooth 0.57 at n=20) in the first version of
this script, which used a fixed jitter of 1e-8. Switching to jitter scaled
to the Gram matrix's own diagonal (`rel_jitter=1e-4`, i.e.
`jitter = 1e-4 * mean(diag(K))`) brought the worst-case condition number
under 1e5 across the same test and fixed both the accuracy spike and the
RBF calibration-correlation artifact in §2.

This matters beyond the toy script: real Gaussian splats can be near-
collocated in a scene (dense regions after densification), so the eventual
gsplat-integrated BQ computation needs the same relative-jitter treatment,
not a fixed constant — noted in `bq_splat/quadrature.py`'s docstring and in
ROADMAP.md's engineering plan.

## 5. Trainable kernel bandwidth: does fitting it close the accuracy gap?

Section 1's diagnosis was that BQ loses to Riemann because both kernels use
one hardcoded bandwidth (RBF sigma=0.35, Matern rho=0.5) across scenes whose
true bump widths range from 0.05 to 0.6 -- the same fixed-bandwidth choice
`models/nerf.py` makes (`sig=0.25`, never adapted). `bq_splat/hyperparams.py`
adds standard GP kernel-hyperparameter fitting (maximize the log marginal
likelihood via a log-spaced grid search + bounded refinement -- no
torch/autodiff needed for this numpy/scipy stage) and
`scripts/validate_trainable_kernel.py` reruns the milestone-1 accuracy sweep
with the bandwidth fit per-trial instead of fixed.

| nodes | Riemann | BQ-RBF fixed | BQ-RBF fit | BQ-Matern fixed | BQ-Matern fit |
|-------|---------|--------------|------------|-------------------|----------------|
| 5     | 1.173   | 1.701        | 1.065      | 1.524             | 1.155          |
| 10    | 0.701   | 1.134        | 0.778      | 0.979             | 0.749          |
| 20    | 0.358   | 0.494        | 0.431      | 0.434             | **0.344**      |
| 40    | 0.171   | 0.241        | 0.232      | 0.161             | **0.148**      |

Fitting closes most of the gap at every node count, and **fitted Matern
beats Riemann outright at n=20 and n=40** (fitted RBF beats Riemann at n=5
but not at higher n). This is a meaningful confirmation of the section-1
diagnosis: the earlier loss to Riemann was substantially a fixed-bandwidth
mismatch problem, not a fundamental limitation of Bayesian quadrature
itself, at least for this class of signals.

Fitted bandwidths vary a lot by scene (RBF sigma: median 0.69, 10-90th
percentile [0.38, 1.57] -- more than 4x the hardcoded 0.35; Matern rho:
median 2.15, [1.09, 5.67] -- an order of magnitude above the hardcoded 0.5),
which is itself informative: no single fixed bandwidth could have served
this whole scene distribution well, and per-scene fitting is doing real
work, not just adding noise. One methodological note: an initial run capped
the fitting search at rho <= 3.0 and 12% of n=20 trials hit that boundary;
widening the bounds to 8.0 changed the reported MAE numbers by only
~0.001-0.0004 (noise-level) while moving the 90th-percentile fitted rho from
a clipped 3.0 to 5.67 -- so the result isn't an artifact of an overly tight
search range, but it's still worth using generous bounds by default.

Caveat: this fits one bandwidth per full node set (i.e., "per ray"), which
is the natural analogue for a per-pixel BQ computation in the eventual GS
setting, but it does mean the fit sees the same data it's then evaluated on
-- there's no held-out validation here, just marginal-likelihood fit
quality. That's standard practice for this kind of hyperparameter fitting,
but worth keeping in mind before treating "fitted Matern beats Riemann" as
a claim about generalization rather than about in-sample fit quality.

## Bottom line for the go/no-go gate

Milestone 1 (derivation + small-scale validation) is a qualified pass:
the ported math is correct (kernel/BQ unit tests all pass, RBF matches
`models/nerf.py`'s formula exactly), variance is reasonably calibrated for
both kernels, and the gap experiment supports the core differentiation
claim at toy scale. The accuracy gap vs. naive Riemann summation — which,
per ROADMAP.md, the paper isn't trying to close as its main claim — is now
understood precisely (§5): it was substantially a fixed-kernel-bandwidth
mismatch, and fitting the bandwidth per scene closes most of it, with fitted
Matern beating Riemann at n=20 and n=40. That both strengthens the accuracy
story (it's not a dead end, it's an unfitted hyperparameter) and reinforces
why the paper's real claim should be about calibrated uncertainty and the
differentiation experiment rather than accuracy alone. Proceeding to
ROADMAP.md's milestone 2 (the real, GS-based differentiation experiment) is
reasonable; if bandwidth fitting carries into that setting (fitting per-ray
or per-pixel, jointly with or alongside splat optimization), it should use
the same marginal-likelihood approach validated here rather than a
hardcoded value.
