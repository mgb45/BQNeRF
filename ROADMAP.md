# Roadmap: Bayesian quadrature as a unified uncertainty signal for Gaussian Splatting

## The thesis

Volume rendering in Gaussian Splatting is a weighted sum over kernel nodes —
which is exactly a **Bayesian quadrature (BQ)** estimate of an integral. That
recognition is not just a reframing: putting a GP prior over the scene, with
a kernel built from the same primitives already used to represent it
(splat position, covariance, and viewing direction), gives a **closed-form
posterior** whose mean recovers the rendered value and whose **variance is a
principled, first-class uncertainty**, derived rather than engineered.

The kernel doesn't have to be a single object. Structured as a **product
kernel** — a spatial component integrated over the region a splat occupies,
times a directional (von Mises-Fisher) component evaluated pointwise at a
query viewing direction — a *single* GP posterior yields two uncertainty
signals that the existing literature treats as unrelated problems solved by
unrelated machinery:

- **Marginalizing over position** (integrating out direction) gives
  **quadrature/discretization uncertainty**: how poorly the *finite,
  currently-placed set of splats* resolves the integral at a point,
  independent of how many views saw it. This flags thin structure,
  high-frequency detail, and sparse local splat coverage — regions that can
  be well-observed and still numerically under-resolved.
- **Evaluating at a query direction** (holding position fixed) gives
  **directional/epistemic uncertainty**: how much the observed viewing
  directions at that point actually constrain the current query direction.
  This is a Bayesian, closed-form analogue of what dedicated visibility
  fields (GAVIS) and sensitivity-pruning methods (PUP) compute by other
  means — same category of signal, different mechanism.

**The claim worth defending in a paper is not "BQ finds a failure mode
others miss."** It's stronger and cleaner than that: **one coherent
probabilistic object, fit with the same effort already spent representing
the scene, subsumes signals that the field currently treats as requiring
separate machinery** — a Hessian/Fisher sensitivity computation for one, an
anisotropic visibility field with its own renderer for the other. If that
holds up under the plan below, it's a unification result, not a niche
add-on.

This has toy-scale (`bq_splat/`) and small real-scale (`gs_experiment/`)
support already — summarized under **Status so far**, below — but this
document does not let that work set the ceiling on the plan. What follows is
the research program needed to make the unification claim rigorously,
against real baselines, at a scale a reviewer will take seriously.

## What's still missing, and why each piece matters for the paper

Ordered roughly by how load-bearing each gap is for a reviewer.

### 1. Formal statement and proof: BQ posterior mean = alpha compositing

**First installment done.** `bq_splat/PROOF_alpha_compositing_equivalence.md`
proves this in two parts: (A) under the standard piecewise-constant
density/color model every NeRF/3DGS renderer already assumes, alpha
compositing is the *exact* value of the continuous rendering integral, not
an approximation of it (a known result, re-derived precisely in this
project's notation); (B) the classical Bayes-Hermite/kernel-quadrature
worst-case-error theorem (O'Hagan 1991), connected directly to this
project's own `v`/`vv`/`K` code, gives
`|true_integral - BQ_mean| <= ||g||_H * sqrt(BQ_variance)` — a *provable*
bound, not a heuristic correlation, verified numerically (never violated
across 40 random test functions per kernel, driven to a ~0.999 tightness
ratio for a test function built to approximate the theorem's own error
representer). The same theorem, applied to different linear functionals on
one product-kernel RKHS, is shown to be what produces both quadrature and
directional/epistemic uncertainty — the rigorous form of the unification
claim. One hypothesis this raised and directly tested was refuted rather
than massaged: RBF was *not* worse-behaved than Matern near a genuine
color-jump discontinuity as hypothesized (opposite result, likely a
lengthscale-matching artifact, left open). See
`bq_splat/results/FINDINGS.md` §10 for the full account, including what
this does and does not yet establish — notably, the formal connection
between these theorems (stated for the ray-depth domain) and what
`gs_experiment`'s actual production `LocalUncertaintyEngine` computes (a
3D spatial-window formulation) is flagged as still open, not assumed.

### 2. Kernel hyperparameters as fitted, first-class quantities — not hardcoded

`bq_splat/hyperparams.py` already fits bandwidth via marginal likelihood at
toy scale, with a genuine finding: it helps for Matérn, not for RBF (the
population-optimal RBF bandwidth turned out to match the original hardcoded
value almost exactly — see `bq_splat/FINDINGS.md` §7). Every `gs_experiment/`
result so far still hardcodes `sigma` per scene. This needs to move to real
GS scale as a first-class part of the pipeline, not an appendix footnote.

**First installment done.** `bq_splat/hyperparams.py` gained
`log_marginal_likelihood_nd`/`fit_kernel_param_pooled_nd`, working directly
with `ProductKernel` over a real 3D position domain, and
`scripts/fit_hyperparameters_real_checkpoint.py` fits a shared bandwidth
against real local windows from the lego "wide" checkpoint, with a
held-out split in the spirit of `validate_trainable_kernel_heldout.py`.
Result: fitted bandwidths differ from the hardcoded `0.05` in *opposite*
directions per kernel (RBF `0.0624`, Matern-3/2 `0.0234`), and generalize
decisively to held-out windows (held-out log marginal likelihood far
better at the fitted value than at the hardcoded one for both kernels,
especially Matern) — real evidence the hardcoded value was leaving
marginal likelihood on the table. But the fitted bandwidth did *not*
materially move the headline sparsity-correlation finding (`r=-0.616`
fitted vs. `-0.612` hardcoded) — good news for that claim's robustness,
and independent motivation for fitting anyway since calibration (item 5)
is a different claim from correlation. Full account, including an open
question about whether Matern's much larger correction is scene-specific,
in `gs_experiment/results/FINDINGS.md` §26.

**Still needed:**
- Joint fitting of the position *and* directional bandwidths together
  (kappa), not just the position term done so far.
- **Per-splat covariance as the kernel bandwidth**, rather than one shared
  scalar — flagged as a deliberate, real extension since the very first
  version of this plan and never attempted. This is likely to matter more
  than global fitting, since splats already carry anisotropic, learned
  covariance that a shared-bandwidth kernel throws away. Run as an explicit
  ablation against shared-bandwidth (see the ablation matrix below), not
  assumed to help.
- Repeat the fit/held-out check on the thin-rod checkpoint and other
  scenes, to test whether Matern's much larger correction on lego is a
  general kernel-family property or specific to that scene's geometry
  (flagged as open in §26).
- A cleaner rerun with matched RNG seeds between the fitting script and
  `sparsity_correlation_experiment.py`, so the correlation comparison is a
  literal replication, not just an internally-consistent one (§26 caveat).

### 3. Training under the likelihood, not just post-hoc readout

Every experiment so far computes BQ variance *after* training, from a
checkpoint trained by ordinary photometric loss. The original NeRF-BQ
prototype in this repo *did* train with a Gaussian-NLL loss
(`models/nerf.py`) — that idea needs to come back at GS scale, and go
further.

**First installment done — a negative result, reported honestly, not
softened.** `train_minimal_gsplat.py` gained both mechanisms:
`densify_criterion="bq_variance"` (swaps the densification trigger from
gsplat's view-space gradient to real closed-form BQ position-only variance,
queried at every splat's own position) and `nll_weight` (an
uncertainty-weighted Gaussian-NLL auxiliary loss term at a sparse grid of
real ray-surface points, `var` from a detached BQ-variance snapshot, not
itself differentiated through — an honest first approximation, not a fully
closed loop). `nll_training_experiment.py` compared 4 variants
(gradient/BQ-variance densify × NLL on/off) on a real scene, matched
seed/hyperparameters, evaluated on both training views and a genuinely
disjoint held-out ring. Result: **BQ-variance-driven densification is a
real regression** — fewer splats *and* worse quality than gradient-based
densification (train PSNR -7.27dB, held-out -0.49dB, not a favorable
efficiency trade-off), hypothesized to be the same problem already found
and fixed in the pruning experiment (§15-16): BQ variance is high in empty
space too, and densification's variance query had no opacity floor to
guard against that. The NLL term alone is close to a no-op (-0.15dB train,
-0.22dB held-out), explainably so given how sparse/coarse and detached its
first-installment signal is.

**The opacity-floor hypothesis was tested directly, same session — a
three-way result, not a clean fix.** Adding `bq_densify_min_opacity=0.3`
(v1: zero the score for low-opacity splats, but still count them in the
percentile threshold) recovered train PSNR almost to baseline (45.95dB),
**confirming the mechanism is real** — but did so via explosive,
uncontrolled growth (hit `max_splats` by iteration 900, since nearly the
entire initial population starts below the 0.3 floor, collapsing the
percentile toward 0), and held-out PSNR got *worse* (19.81dB). The
properly-scoped fix (v2: also exclude low-opacity splats from the
percentile computation, mirroring exactly how the gradient path already
excludes "never received a gradient" splats) gives a sane, non-degenerate
threshold — but lands train PSNR (38.46dB) right back near the original
regression, on an even smaller population (1347 splats). **Reading**: the
mismatch is more structural than a single missing floor — the
percentile-threshold densification scheme was designed around
gradient-magnitude signals, and BQ variance doesn't sit well inside that
same mechanism regardless of eligibility scoping. Full three-way table and
discussion in `gs_experiment/results/FINDINGS.md` §27's addendum.

**Still needed:**
- The more promising untested direction the addendum identifies:
  combine BQ variance *additively* with the existing gradient signal
  (matching the pattern that already worked for pruning) rather than
  replacing the densification trigger outright.
- A weight sweep and/or a finer/more frequent NLL grid, to check whether
  the near-no-op result is a real ceiling or just this installment's
  sparsity (the honest hypothesis in FINDINGS.md §27, not yet tested).
- Differentiating through the BQ posterior itself, rather than treating
  its variance as a periodically-refreshed, detached reweighting — the
  fully closed version of "training under the likelihood," a real further
  step beyond what this installment attempted.
- Gaussian-NLL (or a proper scoring rule) as the *primary* training loss
  rather than an auxiliary term, once the auxiliary version is shown to
  help at all.
- Repeating the comparison on more than one scene/seed before treating
  either result (the regression or the no-op) as general.

### 4. Experiments on the same settings baselines actually use

Nothing so far runs on a benchmark or protocol a reviewer can directly
compare against a cited number. Needed:

- **Full NeRF-Synthetic** (all 8 scenes — lego is the only one attempted),
  standard train/val/test split, standard resolution and view counts, not
  a single scene picked for convenience.
- **Mip-NeRF360** and **Tanks & Temples** — the scenes PUP 3D-GS and GAVIS
  actually report numbers on. Reproducing (or closely approximating) their
  splat-count-vs-quality and NBV-quality-vs-view-budget curves on the same
  scenes is what makes "combining beats either alone" a comparison against
  the literature rather than against an in-house visibility-proxy stand-in.
- Where full reproduction of PUP/GAVIS's own pipeline isn't practical,
  cite their published numbers on these exact scenes and report this
  project's numbers on the same scenes/splits, so the comparison is
  apples-to-apples even without re-running their code.
- **Validate against a real reference 3DGS/gsplat trainer** — **first
  installment done.** `train_with_reference_strategy`
  (`train_minimal_gsplat.py`) trains using `gsplat.strategy.DefaultStrategy`
  (gsplat's own official duplicate/split/prune/opacity-reset
  implementation) in place of this project's `densify_and_prune`, holding
  everything else identical. Re-running the sparsity-correlation check on
  a checkpoint from this real, external, this-project-didn't-write
  implementation gives `r=-0.915` (p=2.7e-60), matching sign and strength
  with the from-scratch trainer's `r=-0.956` on the same scene — direct
  evidence the central claim isn't an artifact of the simplified training
  loop. Caught a real integration bug along the way (a `packed=True`/
  `False` default mismatch between `gsplat.rasterization` and
  `strategy.step_post_backward`) and a real methodological pitfall in this
  project's own process (reusing a `sigma`/`window_radius` pair tuned for
  a *different*, zone-level experiment on the same scene produced a
  wrong-signed `r=+0.219` until rescaled to the checkpoint's actual splat
  spacing) — both documented in `gs_experiment/results/FINDINGS.md` §28,
  including a quality caveat (the reference checkpoint's PSNR trails the
  from-scratch one at this *matched-iteration-count* comparison, plausibly
  because `DefaultStrategy`'s defaults are tuned for far longer training
  budgets than this project's 3000-iteration convention — not yet
  investigated further, flagged rather than glossed over). Not yet done:
  re-running the *other* core experiments (differentiation, pruning, NBV)
  against a reference-strategy checkpoint, and repeating this specific
  check on lego rather than only the thin-rod scene family.

### 5. Calibration, not just correlation

Every real-data result so far is a correlation or ratio (BQ variance vs.
sparsity, wide-zone vs. narrow-zone ratio). None of it answers "is this
variance *calibrated*" — i.e., does a claimed 2x higher variance actually
correspond to 2x the squared error, on average, on genuinely held-out data?
Needed before any uncertainty number is used to justify a downstream
decision in a paper.

**First installment done — a real, nuanced gap, not a clean pass.**
`calibration_experiment.py` runs leave-one-out cross-validation on real
splat colors (`LocalUncertaintyEngine` gained an `exclude_idx` parameter:
a ball query centered on a real splat always finds itself at distance 0,
so predicting its own held-out color needs it explicitly excluded from its
own neighborhood), across three checkpoints (lego wide, and the two
thin-rod checkpoints from item 4's cross-trainer check). Three metrics,
three different readings: **direct Pearson correlation between variance
and squared error is weak everywhere (`|r|<0.21`) and wrong-signed on one
checkpoint** — far weaker than the `r=-0.74` to `-0.96` sparsity
correlations the headline claim rests on; **AUSE (sparsification-curve
ranking) is more encouraging, meaningfully beating random ordering on two
of three checkpoints**, consistent with `pruning_experiment.py`'s
already-positive ranking-based result; **held-out Gaussian NLL is worse
than a flat constant-variance baseline on all three checkpoints** — the
clearest negative result, meaning leave-one-out BQ variance's *absolute
scale* isn't yet a trustworthy per-point confidence value. Full numbers
and discussion, including why this doesn't contradict the sparsity-
correlation claim so much as sharpen exactly what that claim does and
doesn't establish (sparsity and leave-one-out prediction error are related
but not identical — this is the first place they were checked against
each other directly), in `gs_experiment/results/FINDINGS.md` §29.

**Still needed:**
- **Sparsification curves**: done (§29), but only leave-one-out on splat
  colors, not yet on genuinely held-out *test views* (the more standard
  NeRF/GS calibration protocol, closer to what a reviewer expects).
- **AUSE**: computed (§29); repeat on more checkpoints/scenes before
  treating "meaningfully beats random on 2 of 3" as a stable result rather
  than a 3-checkpoint sample.
- **NLL on held-out test views**, using the fitted (not hardcoded) variance
  from item 2, and — given §29's negative NLL result — a recalibration
  step (a monotonic rescaling fit against held-out error, standard
  practice for a miscalibrated-but-informative uncertainty estimate) tried
  explicitly, not assumed unnecessary.
- Test whether per-splat-covariance-as-bandwidth (item 2, not yet tried)
  closes the absolute-scale calibration gap §29 found, since it replaces
  one shared scalar bandwidth with each splat's own learned anisotropic
  covariance — a concrete, motivated next step, not a vague hope.

### 6. Systematic ablations, extended to match the plan's new pieces

The project's existing ablation instincts are good (RBF vs. Matérn,
position-only vs. position+direction, densification on/off, opacity floor
for pruning) but ad hoc and one-off. Consolidate into one ablation matrix,
run consistently across every main experiment rather than once each:

- Kernel family: RBF vs. Matérn-3/2 vs. a compactly-supported kernel
  matched to splat covariance (flagged as a candidate back in the original
  pivot, never tried).
- Bandwidth: fixed/hardcoded vs. marginal-likelihood-fit vs. per-splat
  covariance (item 2).
- Densification: gradient-triggered (current) vs. variance-triggered
  (item 3) vs. off.
- Window radius / von Mises-Fisher concentration (kappa): sensitivity
  sweep — results so far pick one value per experiment without checking
  how much the finding depends on it.
- Opacity floor for pruning: extend the existing two-point sweep
  (`pruning_experiment.py`, §15-16) to a real sweep with confidence
  intervals, not two budget points on one checkpoint.

### 7. Multi-scene, multi-seed statistics

Every headline number so far (18.7x, 0.33x-0.46x, r=-0.74, +1.89dB) comes
from one scene, sometimes replicated across two training seeds at most. A
paper needs distributions, not point estimates: repeat the core
differentiation and combination experiments across the full multi-scene
benchmark set from item 4, with enough seeds to report a confidence
interval or a Wilcoxon/paired test against baselines, not a single ratio.

### 8. Realistic, multi-round next-best-view prediction

The current NBV experiment (`nbv_experiment.py`, §17-19) is a real but
narrow result: single-shot (one view added, one retrain), a small
azimuth-only candidate pool where BQ and the visibility proxy happened to
rank every candidate identically (correlation 1.000) — meaning the
headline "guided beats poor" result doesn't yet demonstrate the more
specific "combination beats either signal alone" the milestone was meant
to show. Needed:

- **Sequential/multi-round** NBV: pick a view, retrain, re-evaluate
  uncertainty, pick the next view, repeat — the actual active-mapping loop
  GAVIS and similar systems run, not a single add-and-compare.
- A **realistic candidate-pose set** (informed by robot reachability or at
  minimum a dense, non-degenerate 3D pose sampling), not a discrete
  azimuth-only ring.
- Scenes **deliberately constructed so BQ and visibility diverge** — the
  single-cluster scene used so far structurally can't separate them, since
  every candidate that reduces one reduces the other identically. A scene
  with both an occluded-but-simple region and an unoccluded-but-fine-
  structure region is needed to actually test whether the combined signal
  beats either alone on view selection, which is the milestone's real
  claim and not yet demonstrated.

### 9. SLAM / incremental-mapping integration

Motivated by the original design question that led to the directional
kernel in the first place (`bq_splat/FINDINGS.md` §9): splats accumulate
incrementally in a SLAM setting, and the pipeline already has a real
likelihood (the BQ posterior) to plug into a mapping loop, rather than a
NeRF-BQ-style ad hoc score. Concretely:

- Wire the incremental/online case: as new views arrive and splats are
  added or updated, update the BQ posterior (both position and directional
  terms) incrementally instead of recomputing from scratch — this is a
  real systems question, not just a re-run of the batch experiments on a
  streaming schedule.
- Use the combined signal to drive online loop-closure or re-visitation
  decisions (go back to a region because directional uncertainty there is
  high, not just because it hasn't been seen recently) — a concrete robot-
  facing deliverable that ties back to the RA-L framing from the original
  pivot.
- This is the most exploratory item in this document and the one most
  likely to need its own scoping pass once items 1-8 are further along —
  listed here as a real target, not a promise of a specific result.

### 10. Robustness and failure-mode analysis

Nothing so far stress-tests the method against conditions where it should
plausibly *fail* or degrade — a paper is stronger for characterizing this
directly rather than a reviewer finding it first. Needed: behavior under
sparse/noisy input views, under a badly-initialized or under-trained
checkpoint, under scenes with reflective/transparent material (a known
hard case for both 3DGS itself and for uncertainty methods built on top of
it), and under adversarially-placed cameras designed to fool the
directional term specifically.

### 11. Runtime / cost benchmarking of the "nearly free" claim

"Nearly free" is currently a qualitative claim resting on one CPU scaling
benchmark (`benchmark_local_bq_scaling.py`, `bq_splat/FINDINGS.md` §8) plus
one measured per-query GPU-adjacent timing during GIF rendering
(2-3ms/solve at `max_neighbors=150`). Needed for the paper to make this
claim with a number instead of an adjective: wall-clock and memory
overhead of computing BQ variance (both terms) relative to plain rendering,
at real checkpoint scale, on GPU, reported alongside the accuracy/
calibration results — ideally as an actual table next to PUP's and GAVIS's
reported throughput numbers, since both of those papers lead with speed.

### 12. Real captured data

Every scene used so far, including the "real-benchmark" lego run, is
Blender-synthetic (NeRF-Synthetic is a recognized standard benchmark, but
still rendered, not photographed). A genuine photograph + COLMAP/SfM
capture — even one scene — is the standard sanity check a reviewer will
ask for before trusting that any of this survives real sensor noise, real
calibration error, and real (non-uniform, non-turntable) view distributions.

## Technical core (unchanged, now explicitly the unification derivation)

Reformulate GS rendering along a camera ray/pixel as a kernel-quadrature
estimate: each contributing splat is a weighted kernel node (its learned
covariance plays the role of kernel bandwidth/shape; its opacity and color
the role of quadrature weight and function value). Under a GP prior built
from a **product kernel** — spatial component (integrated over the domain)
times directional component (von Mises-Fisher, evaluated pointwise at a
query direction):

- The posterior mean over the pixel integral should recover (or closely
  approximate) standard alpha-compositing — item 1 above makes this
  precise.
- **Marginal (position-only) posterior variance** gives quadrature/
  discretization uncertainty: how well the current, finite set of splats
  covers the integral, independent of view count.
- **Position+direction posterior variance, evaluated at the actual query
  viewing direction**, gives directional/epistemic uncertainty: how well
  the observed viewing directions constrain that specific query direction.
- Both come from **one GP posterior**, not two separate models — the
  central unification claim, already shown mathematically real at toy
  scale (`bq_splat/FINDINGS.md` §9: 0.97x vs. 2.46x, isolating the
  directional term's real effect against a confound-controlled baseline)
  and replicated once at real GS scale (`gs_experiment/FINDINGS.md` §22,
  §25).

## Status so far (condensed — full detail in FINDINGS.md files)

This section is a summary of what's been validated, kept short
deliberately so it doesn't set the ceiling for the plan above. Full
narrative detail, including every bug found and fixed along the way, lives
in `bq_splat/results/FINDINGS.md` and `gs_experiment/results/FINDINGS.md`.

**Toy scale (`bq_splat/`, pure numpy/scipy, CPU):**
- RBF and Matérn-3/2 BQ math derived and unit-tested against
  `models/nerf.py` and numerical integration.
- Marginal-likelihood bandwidth fitting implemented; helps Matérn, doesn't
  help RBF (a held-out test showed the RBF gain was overfitting — §7).
- CPU scaling bottleneck identified (a `vv` term, not the linear solve) and
  fixed via caching + KD-tree, ~2,400-3,000s -> ~140-420s per 800x800
  image up to 10^6 synthetic splats (§8).
- `ProductKernel` + `DirectionalKernel` (von Mises-Fisher) implemented; a
  confound-controlled toy experiment isolates the directional term's real
  effect (0.97x position-only vs. 2.46x position+direction — §9).

**Real GS scale (`gs_experiment/`, gsplat + a from-scratch minimal
trainer, RTX 3090):**
- Real checkpoint loading, SH color evaluation, and geometric visibility
  attribution implemented and tested against a synthetic-occluder case.
- On a hand-built thin-rod scene: position+direction BQ variance shows an
  18.7x wide/narrow-view-count ratio; the core position-only
  quadrature-uncertainty claim (wide zone flagged as *more* uncertain than
  narrow, opposite the visibility proxy's ranking) required real
  gradient-triggered densification to appear at all, then replicated
  across two seeds and both kernel families, and survived four independent
  artifact checks — including a controlled test that refuted the leading
  "redundancy" hypothesis for the mechanism, which remains genuinely open
  (§9-14).
- Pruning combination (BQ + opacity, opacity-floored) beat opacity-only at
  a tight splat budget, no-op at looser ones (§15-16).
- Single-shot NBV combination picked a view that improved held-out PSNR
  ~3x more than a poor choice, but BQ and the visibility proxy ranked
  every candidate identically in this scene (correlation 1.000) — item 8
  above is what's needed to actually test signal combination for NBV
  (§17-19).
- On NeRF-Synthetic "lego" (real benchmark, not hand-built): BQ variance
  correlates with local splat sparsity (r=-0.74, p=8e-27) and responds to
  genuine angular coverage gaps rather than raw view count (flat across
  100->12 random-subset views, 2.75x higher for the same 12-view count
  when clustered instead of random) — direct support for the project's
  "uncertainty nearly for free" claim that doesn't route through a proxy
  (§20-25).

## Literature to read before drafting

This space moves roughly monthly. At minimum, get an abstract-level read on
these:
- **PUP 3D-GS** (CVPR 2025) — Hessian/Fisher sensitivity pruning.
- **GAVIS** (CVPR 2026) — anisotropic visibility field, Bayesian-network
  renderer, active mapping/NBV.
- **"Rendering-Aware Bayesian 3D Gaussian Splatting with Native Uncertainty
  and Adaptive Complexity Control"** (arXiv, July 2026) — NIW/Dirichlet-
  process posteriors, native uncertainty + complexity control + active
  view selection in one paper.
- Variational Bayes Gaussian Splatting (ICLR 2025)
- Variational Multi-Scale Representation for Estimating Uncertainty in 3DGS
  (NeurIPS 2024)
- WarpRF: Multi-View Consistency for Training-Free UQ in Radiance Fields
  (2025)
- Active3D: Active High-Fidelity 3D Reconstruction via Hierarchical
  Uncertainty Quantification (AAAI 2026)
- Predictive Photometric Uncertainty in Gaussian Splatting for Novel View
  Synthesis (2026)
- Uncertainty-Aware Gaussian Splatting with View-Dependent Regularization
  (Eurographics)

None of the above frames rendering uncertainty as quadrature/discretization
error, and none unifies it with the directional/epistemic term via one
posterior — that combination is still, as far as this search found, this
project's actual opening. Re-check before drafting; a year is a long time
in this literature.

## Write-up plan

Primer appendix (Bayesian quadrature basics, from the tutorial notebook);
the formal alpha-compositing equivalence (item 1) as the core derivation;
the unification argument (one posterior, two projections) as the
paper's central theoretical contribution; calibration results (item 5) and
matched-baseline comparisons (item 4) as the main empirical section;
training-under-the-likelihood (item 3) and kernel-fitting (item 2) as a
secondary "closing the loop" section if results support it; NBV (item 8)
and pruning as downstream-task validation; robustness (item 10) and
runtime cost (item 11) as a "practicality" section; SLAM integration
(item 9) either as a full section or scoped down to a discussion/future-
work note depending on how far it gets. Honest pilot-study section
covering the original NeRF-PSNR negative result and the refuted redundancy
hypothesis — both are evidence of a careful process, not embarrassments to
hide.

## Verification gates

- Item 1 (formal equivalence) gates everything downstream — if the
  posterior mean doesn't track alpha-compositing closely enough, every
  later "variance means X" claim needs re-deriving against whatever the
  mean actually is.
- Item 5 (calibration) gates any claim that a variance number is
  meaningful in an absolute sense, not just directionally correlated with
  error.
- Item 8 (multi-round, divergence-capable NBV) gates the "combination
  beats either alone" claim specifically — the current single-shot,
  non-divergent result doesn't support it yet, however good it looks.
- Item 4 (matched baselines) gates any comparison claim against PUP/GAVIS
  — cite their numbers only on scenes/splits this project actually
  reproduces the protocol for.
