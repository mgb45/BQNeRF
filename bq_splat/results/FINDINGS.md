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

- Mean local BQ variance **inside** the gap: 0.456
- Mean local BQ variance **outside** the gap: 0.070
- Ratio: **6.5x**

See `gap_experiment.png`. This is the toy-scale version of the paper's
central differentiation claim (quadrature uncertainty flags well-observed-
but-under-resolved geometry, distinct from occlusion/visibility-based
uncertainty) and it holds up here. One bonus observation from the plot:
variance also spikes near a tall, narrow bump around t≈9.5 that's *outside*
the designated gap and reasonably well-covered by node count — but the bump
is narrow enough relative to local node spacing that it's still locally
under-resolved. That's a good sign: the mechanism is responding to genuine
local under-sampling-relative-to-structure in general, not just to the one
region it was specifically constructed to flag.

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

## Bottom line for the go/no-go gate

Milestone 1 (derivation + small-scale validation) is a qualified pass:
the ported math is correct (kernel/BQ unit tests all pass, RBF matches
`models/nerf.py`'s formula exactly), variance is reasonably calibrated for
both kernels, and the gap experiment supports the core differentiation
claim at toy scale. The open item is the accuracy gap vs. naive Riemann
summation, which — per ROADMAP.md — the paper isn't trying to close, but
should be understood and stated precisely (likely a fixed-kernel-bandwidth
mismatch to signal structure) before writing it up. Proceeding to
ROADMAP.md's milestone 2 (the real, GS-based differentiation experiment) is
reasonable; the kernel-choice ablation from §2 should happen in parallel,
since it doesn't require gsplat.
