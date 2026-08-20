# Roadmap: from tiny-NeRF prototype to a paper

## Where this project stands

This repo started as a principled take on uncertainty quantification for
NeRF rendering: instead of an ad hoc confidence heuristic, treat the
per-ray volume-rendering integral as a **Bayesian quadrature (BQ)** problem.
The weighted-color signal along a ray is modeled as a zero-mean GP with an
RBF kernel (`models/nerf.py`); the posterior mean of the ray integral gives
the rendered color and the posterior *variance* gives a closed-form,
per-pixel uncertainty, used directly in a Gaussian-NLL training loss. A
Matérn kernel variant was derived in `tutorial/Intro to Bayesian
quadrature.ipynb` but never wired into the live model, so the original goal
of comparing multiple kernels stalled at one.

The one experiment in the repo (`inspect_model.ipynb`, `figs/Ns_*.png`) —
PSNR vs. sample count, BQ vs. standard quadrature, on a single held-out
test image — shows BQ matching or losing to standard quadrature at every
sample count. And by the time the repo went dormant (Feb 2023), 3D Gaussian
Splatting had made pure ray-marching NeRF pipelines largely uncompetitive on
the metric (novel-view PSNR) this prototype was implicitly being judged on.

## First pivot attempt, and what a literature check found

The first version of this plan proposed porting the BQ derivation onto
Gaussian Splatting and using it for two downstream tasks — uncertainty-
guided densification/pruning, and active-view/next-best-view (NBV)
selection — targeting IEEE RA-L. A literature search turned up recent, strong
published work occupying almost exactly that ground:

- **PUP 3D-GS** (CVPR 2025) already does uncertainty-guided GS pruning, via
  Hessian/Fisher sensitivity — prunes 90% of splats while improving quality.
- **GAVIS** (CVPR 2026) already does uncertainty-driven active mapping/NBV
  for GS, via an anisotropic visibility field (spherical harmonics) through
  a Bayesian-network volume renderer — real-time at 200 FPS, claiming 500×
  faster visibility-field construction than the prior SOTA (NVF).
- **"Rendering-Aware Bayesian 3D Gaussian Splatting with Native Uncertainty
  and Adaptive Complexity Control"** (arXiv, July 2026) combines native
  uncertainty, complexity control, and active view selection under
  sparse-view budgets in one paper — via Normal-Inverse-Wishart /
  Dirichlet-process posteriors over Gaussian geometry, not kernel
  quadrature, but covering almost the same combined scope.

Competing with these on their own metrics (pruning ratio, FPS, NBV quality)
as a first attempt would be a weak position. But the same search surfaced
the actual opening: every method found — Hessian/Fisher sensitivity,
visibility fields, NIW/Dirichlet-process posteriors, variational weight
posteriors, Laplace approximation, dropout, training-free multi-view
consistency — targets **epistemic** uncertainty (missing views, occlusion,
model sensitivity). None of them frame rendering uncertainty as
**quadrature/discretization error**: the residual error from approximating
a continuous integral with a finite set of weighted kernels. One search
summary put it plainly — discretization is "treated as an inherent
approximation rather than a distinct, quantifiable source of uncertainty"
in this literature. The RBF-vs-Matérn kernel-choice question this repo
stalled on also has no hits anywhere in the rendering-uncertainty
literature.

## The revised pivot

Quadrature uncertainty and epistemic/visibility uncertainty are different
failure modes. Quadrature uncertainty flags regions that are
**well-observed but numerically under-resolved** — thin structures,
high-frequency detail, sparse local splat coverage — which visibility-based
and Hessian-based methods structurally cannot see, since those regions
*were* seen from enough views. The contribution is this distinction plus
the mechanism (closed-form via kernel choice, essentially free to compute),
not besting PUP or GAVIS at their own game. Positioned this way, BQ
uncertainty is complementary to, and combinable with, the current SOTA —
a more defensible reviewer story than a head-on comparison, and still a
credible target for a robotics venue (RA-L) given the map-building framing.

**Revised thesis:** BQ-derived uncertainty, computed directly from the same
kernel structure used to represent a Gaussian-splat scene, is a cheap
signal for a failure mode existing GS uncertainty methods miss —
under-resolved-but-visible geometry — and combining it with an existing
visibility/epistemic signal gives better densification or view-selection
decisions than either alone.

## What carries forward vs. what doesn't

**Keep and adapt:**
- The closed-form kernel-mean-embedding derivation (`rbf_vf`, `rbf_vff` /
  `rbf_vvf_part` in `models/nerf.py`) — generalizes from "GP over samples
  along a ray" to "GP over a mixture of anisotropic Gaussian kernels
  (splats) contributing to a pixel." This is the technical core of the
  paper.
- The tutorial notebook, as the basis for a "Bayesian quadrature primer"
  appendix.
- The Matérn kernel derivation — finish it and compare against RBF (and
  consider a third, e.g. a compactly-supported kernel matching splat
  covariance more directly). This kernel-choice study is the piece with
  the cleanest, least-contested novelty claim, independent of how the
  downstream-combination story lands.
- The Gaussian-NLL training idea, as a possible secondary (training-time)
  use of the uncertainty.
- The negative NeRF-PSNR result — report it explicitly as the pilot study
  that motivated pivoting away from pointwise-PSNR evaluation toward
  downstream-task evaluation, not as something to hide.

**Discard:**
- `train.py`'s single-scene training loop and the Std-vs-BQ PSNR comparison
  as a headline result — pilot/appendix material only.
- Further investment in the from-scratch NeRF MLP itself — the new work is
  GS-based.
- Framing either downstream track as a head-to-head win over PUP 3D-GS or
  GAVIS — reframed as combination, below.

## Technical core: BQ for Gaussian Splatting

Reformulate GS rendering along a camera ray/pixel as a kernel-quadrature
estimate: each contributing splat is a weighted kernel node (its learned
covariance plays the role of kernel bandwidth/shape; its opacity and color
the role of quadrature weight and function value). Under a GP prior tied to
the splats' own kernel family:
- The posterior mean over the pixel integral should recover (or closely
  approximate) standard alpha-compositing — verify this formally and
  empirically before building anything downstream.
- The posterior variance gives a closed-form per-pixel/per-splat
  uncertainty reflecting how well the *current, finite set of splats*
  covers the integral.

## The differentiation experiment (load-bearing — do this first)

Construct or find a scene/setup where a region is well-visible (low
visibility-uncertainty by a GAVIS-style measure) but poorly resolved
(high-frequency or thin geometry, sparse local splat density). Show BQ
quadrature uncertainty flags it while a visibility-based signal doesn't.
This is what makes "orthogonal, not redundant" a demonstrated claim rather
than an assertion — treat it as the actual go/no-go gate for the rest of
the plan, ahead of both downstream tracks below.

## Downstream evaluation, as combination rather than competition

Rather than separate "beat PUP on pruning" / "beat GAVIS on NBV" tracks,
test whether **combining** BQ quadrature uncertainty with an existing
visibility/epistemic signal (a simple visibility-count or entropy proxy
standing in for GAVIS/PUP, where full reproduction isn't in scope) gives
better densification or view-selection decisions than either signal alone:

1. **Densification/pruning.** Use BQ variance alongside a
   visibility/gradient-based criterion; check whether the combination
   reaches equal quality at fewer splats, or catches failure cases
   (under-resolved-but-visible regions) that the heuristic-only or
   Hessian-only baseline misses.
2. **Active view planning / NBV.** Use BQ variance alongside a visibility
   proxy for candidate-view scoring; check whether the combined signal
   selects views that improve reconstruction in under-resolved regions
   faster than either signal alone.

Cite PUP/GAVIS reported numbers rather than trying to outperform their
optimized, real-time implementations outright.

## Engineering plan

- Build on `gsplat` (Python/PyTorch-hackable) rather than the original
  Inria CUDA 3DGS repo, so a custom densification criterion and per-splat
  uncertainty computation don't require hand-writing CUDA.
- New components needed (none exist yet):
  1. Batched closed-form BQ posterior-variance computation compatible with
     GS's typical splat counts (hundreds of thousands to millions) — the
     main engineering risk; naive per-pixel GP regression won't scale.
     Validate first on small synthetic scenes.
  2. A validation harness with two modes: (a) BQ posterior variance vs. a
     brute-force baseline (leave-one-splat-out variance, or a GS-model
     ensemble) on a small scene, to sanity-check the closed-form
     derivation; (b) BQ variance vs. a visibility/epistemic proxy on the
     differentiation-experiment scene, to check the two signals actually
     diverge where expected.
  3. Modified densification/pruning logic combining BQ variance with an
     existing criterion.
  4. An NBV/active-view selection loop combining BQ variance with a
     visibility proxy, plus a candidate-pose evaluation harness — start on
     Blender synthetic scenes with a discrete candidate-pose set before
     considering real robot data.
- Datasets: standard GS synthetic benchmark scenes, plus whatever scene
  construction the differentiation experiment requires (likely a scene
  with both open, well-covered regions and thin/high-frequency structure).

## Literature to read before drafting

This space moves roughly monthly. At minimum, get an abstract-level read on
these before finalizing framing (beyond PUP 3D-GS, GAVIS, and the July 2026
Native Uncertainty preprint, already summarized above):
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

## Milestones

1. **Done** — derivation + small-scale validation, in `bq_splat/` (pure
   numpy/scipy, no gsplat/torch dependency yet). RBF and Matérn-3/2 kernel
   math is ported/derived and unit-tested against `models/nerf.py`'s exact
   formula and against numerical integration. A 1D toy-ray Monte Carlo sweep
   and a deliberate sparse-but-visible "gap" experiment are implemented and
   run; see `bq_splat/results/FINDINGS.md` for the qualified-pass write-up
   (BQ mean still loses to naive Riemann summation on raw accuracy, matching
   the original NeRF-BQ result, but posterior variance is reasonably
   calibrated for both kernels and the gap experiment shows ~6.5x higher
   local variance in an under-sampled-but-visible region — supporting the
   differentiation claim at toy scale). Also surfaced a real numerical-
   conditioning issue (irregular node spacing can push the Gram matrix
   condition number past 1e18 with a fixed jitter) and fixed it with a
   relative jitter — a lesson that carries into the eventual gsplat port.
2. The differentiation experiment — the real go/no-go gate, now to be run
   on an actual GS scene rather than a 1D toy signal.
3. Densification/pruning combination experiment.
4. NBV combination experiment.
5. Write-up: primer appendix, honest pilot-study section, main derivation,
   differentiation experiment, and the two combination experiments.

## Verification gates

- Step 1's brute-force validation is necessary but not sufficient — passing
  it doesn't rescue the paper if step 2 shows the uncertainty is redundant
  with visibility.
- Step 2 (differentiation experiment) is the actual gate before investing
  in either downstream track.
- Steps 3–4 are checked against cited PUP/GAVIS numbers and against the
  signal-alone baselines run in the same harness — the claim to defend is
  "combination beats either alone," not "beats PUP/GAVIS outright."
