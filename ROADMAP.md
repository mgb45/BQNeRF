# Roadmap: Bayesian quadrature as a unified uncertainty signal for Gaussian Splatting

For the plain-language version of the thesis below, see the top-level
[`README.md`](README.md). This document is the working research plan:
what's done, what's still open, ordered by how load-bearing each gap is
for a paper. Status blurbs here are intentionally short — the full
numbers and reasoning live in
[`bq_splat/results/FINDINGS.md`](bq_splat/results/FINDINGS.md) and
[`gs_experiment/results/FINDINGS.md`](gs_experiment/results/FINDINGS.md)
(current-conclusions summaries) and their `ARCHIVE_FULL_LOG.md`
companions (complete process logs).

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
  independent of how many views saw it.
- **Evaluating at a query direction** (holding position fixed) gives
  **directional/epistemic uncertainty**: how much the observed viewing
  directions at that point actually constrain the current query direction
  — a Bayesian, closed-form analogue of what dedicated visibility fields
  (GAVIS) and sensitivity-pruning methods (PUP) compute by other means.

**The claim worth defending in a paper is not "BQ finds a failure mode
others miss."** It's stronger: **one coherent probabilistic object, fit
with the same effort already spent representing the scene, subsumes
signals that the field currently treats as requiring separate
machinery.** If that holds up under the plan below, it's a unification
result, not a niche add-on. This document does not let the work done so
far set the ceiling on the plan — what follows is the research program
needed to make the unification claim rigorously, against real baselines,
at a scale a reviewer will take seriously.

## What's still missing, and why each piece matters for the paper

Ordered roughly by how load-bearing each gap is for a reviewer.

### 1. Formal statement and proof: BQ posterior mean = alpha compositing

**Done.** `bq_splat/PROOF_alpha_compositing_equivalence.md` proves this in
two parts: alpha compositing is the *exact* value of the continuous
rendering integral under the standard piecewise-constant model (not an
approximation of it), and the classical Bayes-Hermite worst-case-error
theorem gives a *provable* bound (not a heuristic correlation) on the BQ
mean's error in terms of its own reported variance — verified
numerically, including a tightness check, and shown to be the same
theorem behind the directional-uncertainty unification claim. One
hypothesis this raised was tested and refuted rather than massaged: RBF
was *not* found to be worse-behaved than Matérn near a discontinuity as
expected. Still open: connecting this ray-domain proof formally to what
`gs_experiment`'s production `LocalUncertaintyEngine` actually computes
(a 3D spatial-window formulation).

### 2. Kernel hyperparameters as fitted, first-class quantities — not hardcoded

**Done, at real checkpoint scale.** `bq_splat/hyperparams.py`'s toy-scale
marginal-likelihood bandwidth fitting was extended to real 3D checkpoints
(`fit_kernel_param_pooled_nd`, `scripts/fit_hyperparameters_real_checkpoint.py`).
Fitted bandwidths differ substantially from the hardcoded default and
generalize decisively to held-out windows, but which kernel needs the
larger correction turned out to be scene-specific, not a general rule
(opposite directions on lego vs. a thin-rod scene) — and fitting didn't
materially move the headline sparsity-correlation number, good news for
that claim's robustness. Still needed: joint fitting of the directional
bandwidth (kappa) together with position; **per-splat covariance as the
kernel bandwidth**, replacing one shared scalar with each splat's own
learned anisotropic covariance — flagged since the first version of this
plan, likely to matter more than global fitting, still not attempted.

### 3. Training under the likelihood, not just post-hoc readout

**Done — a real negative result, not softened.** `train_minimal_gsplat.py`
gained both an uncertainty-weighted Gaussian-NLL loss term and a
BQ-variance-driven densification trigger, compared against standard
gradient-based training on the same scene/seed. Neither helped:
variance-driven densification is a real regression (worse quality *and*
fewer splats, not a favorable trade-off), diagnosed and partially fixed
(an opacity floor recovers quality but only via uncontrolled growth,
pointing to a more structural mismatch between BQ variance and a
percentile-threshold densification scheme built for gradient signals);
the NLL term alone was close to a no-op. Still needed: combining BQ
variance *additively* with the existing gradient signal (the pattern
that worked for pruning) rather than replacing the trigger outright;
differentiating through the BQ posterior itself rather than treating it
as a periodically-refreshed detached reweighting; repeating on more than
one scene/seed.

### 4. Experiments on the same settings baselines actually use

**Full NeRF-Synthetic (all 8 scenes) — done for the sparsity/calibration
checks.** The sparsity-correlation claim replicates on all 8 standard
scenes (`r` between -0.74 and -0.97, every one significant) — the
strongest multi-scene evidence for the central claim so far, though at a
lighter training budget than the original lego run (flagged, with a
like-for-like replication as the natural follow-up). **Validating against
a real reference `gsplat` trainer — done.** Training with `gsplat`'s own
official reference densification strategy instead of this project's own
code reproduces the sparsity-correlation result closely — direct evidence
it isn't an artifact of a simplified training loop. **Still needed:**
Mip-NeRF360 and Tanks & Temples — the scenes PUP 3D-GS and GAVIS actually
report numbers on; without reproducing their protocol on these exact
scenes, any comparison against their published numbers isn't apples-to-
apples yet.

### 5. Calibration, not just correlation

**Done — a real, nuanced gap, not a clean pass.** Leave-one-out
cross-validation on real splat colors across several checkpoints finds
direct correlation between variance and squared error weak (`|r|<0.21`,
wrong-signed on one checkpoint) — much weaker than the sparsity-
correlation numbers the headline claim rests on. Ranking-based
calibration (AUSE) is more encouraging. Held-out Gaussian NLL is worse
than a flat constant-variance baseline on every checkpoint tested — the
clearest negative result: absolute-scale variance isn't yet a trustworthy
per-point confidence value, even though its relative ranking is useful
(consistent with pruning's already-positive ranking-based result). Still
needed: calibration on genuinely held-out *test views* (not just
leave-one-out on splat colors); whether per-splat-covariance-as-bandwidth
(item 2) closes the absolute-scale gap.

### 6. Systematic ablations

**Kernel family (RBF vs. Matérn) done, with fitted bandwidths, on three
real checkpoints — a real trade-off, not a winner.** RBF wins the
sparsity-correlation claim and absolute (NLL) calibration on every
checkpoint; Matérn wins ranking-based calibration on every checkpoint. No
paper claim of one universally-superior kernel is supported by this data.
**Window-radius sensitivity done.** The sparsity-correlation claim is
robust across a full order of magnitude (0.2x-2x an established value) on
every checkpoint tested, then degrades and flips sign past ~4x — a
general, predictable pattern (a large window increasingly measures
position relative to the point cloud's edge, not genuine local density),
giving grounded practical guidance for picking the parameter. Still
needed: a compactly-supported kernel option; kappa (directional
concentration) sensitivity; a real opacity-floor sweep for pruning with
confidence intervals, not two budget points.

### 7. Multi-scene, multi-seed statistics

**Partially done, for the sparsity-correlation claim specifically**: the
8-scene NeRF-Synthetic run (item 4) gives 8 independent `r` values, a
real range rather than one number, though still one seed per scene and
not a like-for-like density match across scenes. **Still needed**: the
core differentiation and downstream-combination experiments (pruning,
NBV) have not been repeated multi-scene — still one hand-built scene for
those claims, with only two training seeds.

### 8. Realistic, multi-round next-best-view prediction

The current NBV result (single-shot, a candidate pool where BQ and a
visibility proxy happened to rank every candidate identically) shows
"guided beats a poor choice" but not yet the more specific "combination
beats either signal alone." **A prerequisite piece is done**: a scene
with a designed, continuous (5-level) view-coverage gradient shows
directional BQ variance recovers it exactly (monotonic across all 5
zones, a `12.97x` range, a geometry-matched control staying flat) — real
evidence BQ tracks *degree* of coverage, not just a high/low threshold.
**Tested on real geometry, including a real photographed scene, with an
honestly open result** (see item 12). Still needed for the NBV claim
itself: sequential/multi-round view selection (not single-shot), a
realistic candidate-pose set, and a scene deliberately constructed so BQ
and a visibility proxy can diverge — none of which the coverage-gradient
work above addresses on its own.

### 9. SLAM / incremental-mapping integration

Not started. Splats accumulate incrementally in a SLAM setting, and this
project's pipeline already has a real likelihood (the BQ posterior) that
could drive online re-visitation decisions rather than an ad hoc score —
the original motivation for the directional kernel. The most exploratory
item here, likely needing its own scoping pass once items 1-8 are
further along.

### 10. Robustness and failure-mode analysis

Not started. Needed: behavior under sparse/noisy input views, a
badly-initialized or under-trained checkpoint, reflective/transparent
material (a known hard case for 3DGS itself), and adversarially-placed
cameras designed to fool the directional term specifically.

### 11. Runtime / cost benchmarking of the "nearly free" claim

Currently qualitative, resting on one CPU scaling benchmark and one
per-query GPU-adjacent timing measurement. Needed for a paper: wall-clock
and memory overhead of computing BQ variance relative to plain rendering,
at real checkpoint scale, on GPU, ideally as a table next to PUP's and
GAVIS's own reported throughput numbers.

### 12. Real captured data

**Done — a first real photographed scene, and an honestly open result,
not a clean pass in either direction.** `colmap_loader.py` (a genuinely
new capability: every prior scene had exactly-known poses, not an SfM
estimate) applied to a real Mip-NeRF360 scene repeats the coverage-
gradient test on real photographs. The first attempt looked like a clean
null but was built on a reconstruction too poor (8 total views per
condition) to trust either way — corrected explicitly rather than left
standing. A properly-resourced retest has been run on lego (30 views/
condition, held-out PSNR checked first): meaningfully better
reconstruction, but held-out quality is still mediocre and confounded
with the spread variable itself, leaving a real, moderately-well-grounded
null with one specific confound still unresolved — not yet a clean
paper-worthy result either way. The same properly-resourced retest hasn't
been run on the real photographed scene yet. Full scene reconstruction
quality at a competitive splat budget, and the sparsity-correlation/
calibration checks (items 4-6), also haven't been repeated on real
captured data yet.

## Technical core (the unification derivation)

Reformulate GS rendering along a camera ray/pixel as a kernel-quadrature
estimate: each contributing splat is a weighted kernel node (its learned
covariance plays the role of kernel bandwidth/shape; its opacity and color
the role of quadrature weight and function value). Under a GP prior built
from a **product kernel** — spatial component (integrated over the domain)
times directional component (von Mises-Fisher, evaluated pointwise at a
query direction):

- The posterior mean over the pixel integral recovers standard
  alpha-compositing — item 1 makes this precise and provable.
- **Marginal (position-only) posterior variance** gives quadrature/
  discretization uncertainty.
- **Position+direction posterior variance, evaluated at the actual query
  viewing direction**, gives directional/epistemic uncertainty.
- Both come from **one GP posterior**, not two separate models — shown
  mathematically real at toy scale, replicated at real GS scale on
  designed geometry, and honestly still open on real geometry (item 12).

## Literature to read before drafting

This space moves roughly monthly. At minimum, get an abstract-level read on
these before finalizing framing:
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

Primer appendix (Bayesian quadrature basics); the formal alpha-
compositing equivalence (item 1) as the core derivation; the unification
argument (one posterior, two projections) as the paper's central
theoretical contribution; calibration results (item 5) and matched-
baseline comparisons (item 4) as the main empirical section; training-
under-the-likelihood (item 3) and kernel-fitting (item 2) as a secondary
section; NBV (item 8) and pruning as downstream-task validation;
robustness (item 10) and runtime cost (item 11) as a "practicality"
section; SLAM integration (item 9) scoped to however far it gets. Honest
sections throughout covering what didn't work (the original NeRF-BQ pilot
in `archive/original_nerf_prototype/`, the refuted redundancy hypothesis,
the training-under-the-likelihood negative result, the still-open real-
geometry directional question) — evidence of a careful process, not
embarrassments to hide.

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
- Item 12 (real captured data) gates any claim that the directional/
  epistemic half of the unification result holds beyond designed,
  occlusion-free geometry — currently an open question, not a passed gate.
