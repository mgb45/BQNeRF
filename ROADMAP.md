# Roadmap: from tiny-NeRF prototype to a robotics paper

## Where this project stands

This repo started as a principled take on uncertainty quantification for
NeRF rendering: instead of an ad hoc confidence heuristic, treat the
per-ray volume-rendering integral as a **Bayesian quadrature (BQ)** problem.
The weighted-color signal along a ray is modeled as a zero-mean GP with an
RBF kernel (`models/nerf.py`); the posterior mean of the ray integral gives
the rendered color and the posterior *variance* gives a closed-form,
per-ray uncertainty, used directly in a Gaussian-NLL training loss. A
Matérn kernel variant was derived in `tutorial/Intro to Bayesian
quadrature.ipynb` but never wired into the live model, so the original goal
of comparing multiple kernels stalled at one.

The one experiment in the repo (`inspect_model.ipynb`, `figs/Ns_*.png`) —
PSNR vs. sample count, BQ vs. standard quadrature, on a single held-out
test image — shows BQ matching or losing to standard quadrature at every
sample count. And by the time the repo went dormant (Feb 2023), 3D Gaussian
Splatting had made pure ray-marching NeRF pipelines largely uncompetitive on
the metric (novel-view PSNR) this prototype was implicitly being judged on.

## The pivot

Competing on PSNR against NeRF/GS baselines is not a fight worth having.
But GS's rise is not actually a dead end for the core idea — it's arguably
a better home for it. GS represents a scene as a set of weighted anisotropic
Gaussian kernels (nodes + covariances), which is exactly the object a
kernel-quadrature rule operates over. Bayesian quadrature is a principled
theory of *where to place weighted kernel evaluation points to minimize
integration error* — which is what GS densification/pruning does today via
a crude gradient-magnitude heuristic. The uncertainty BQ yields is also
cheap: closed-form, no ensembles, no Fisher-information backward pass —
which matters if the eventual consumer is a robot that needs uncertainty
at map-building time, not just at evaluation time.

**Target:** IEEE RA-L (continuous submission — no fixed deadline, optional
fast-track to ICRA/IROS on early acceptance).

**Thesis:** BQ-derived uncertainty, computed directly from the same kernel
structure already used to represent a Gaussian-splat scene, is a cheap and
calibrated signal for two robot-relevant downstream tasks: (a) guiding
densification/pruning during map construction, and (b) driving active view /
next-best-view (NBV) selection. The claim is not "better PSNR" — it's
"uncertainty you get almost for free, that a robot can act on."

## What carries forward vs. what doesn't

**Keep and adapt:**
- The closed-form kernel-mean-embedding derivation (`rbf_vf`, `rbf_vff` /
  `rbf_vvf_part` in `models/nerf.py`) — generalizes from "GP over samples
  along a ray" to "GP over a mixture of anisotropic Gaussian kernels
  (splats) contributing to a pixel." This is the technical core of the
  paper.
- The tutorial notebook, as the basis for a "Bayesian quadrature primer"
  appendix.
- The Matérn kernel derivation — finish it and compare against RBF; its
  tunable smoothness may match compact-support, anisotropic splats better
  than RBF's infinite support.
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

## New technical core: BQ for Gaussian Splatting

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
  covers the integral — discretization/quadrature error, distinct from
  (and cheaper than) what ensemble/dropout/Fisher-information UQ methods
  target. This distinction is the novelty claim.

## Two downstream evaluation tracks

1. **Uncertainty-aware densification/pruning** (do first, de-risks track 2).
   Replace vanilla 3DGS's gradient-magnitude densification criterion with
   the BQ posterior-variance criterion. Evaluate on standard GS benchmark
   scenes: does BQ-guided density control reach equal quality at fewer
   splats, or converge faster, than the heuristic baseline?

2. **Active view planning / next-best-view** (flagship, robot-relevant).
   Use per-candidate-view aggregated BQ uncertainty to select the next
   viewpoint(s) for a robot incrementally building a GS map.
   - Baselines: random/uniform selection; a simple heuristic (entropy of
     accumulated opacity); and, scope permitting, an existing
     uncertainty-based NBV method (e.g. FisherRF) — confirm the current
     strongest comparator via a literature check first, since this area
     moves fast.
   - Metric: final reconstruction quality (and/or downstream localization
     accuracy — ties back to this repo's original README note about
     "localisation down the line") after a fixed number of acquired views,
     vs. baselines.

## Engineering plan

- Build on `gsplat` (Python/PyTorch-hackable) rather than the original
  Inria CUDA 3DGS repo, so a custom densification criterion and per-splat
  uncertainty computation don't require hand-writing CUDA.
- New components needed (none exist yet):
  1. Batched closed-form BQ posterior-variance computation compatible with
     GS's typical splat counts (hundreds of thousands to millions) — the
     main engineering risk; naive per-pixel GP regression won't scale.
     Validate first on small synthetic scenes.
  2. A validation harness comparing BQ posterior variance against a
     brute-force baseline (leave-one-splat-out variance, or a GS-model
     ensemble) on a small scene.
  3. Modified densification/pruning logic (Track 1).
  4. An NBV/active-view selection loop plus a candidate-pose evaluation
     harness (Track 2) — start on Blender synthetic scenes with a discrete
     candidate-pose set before considering real robot data.
- Datasets: standard GS synthetic benchmark scenes for Track 1; a
  synthetic multi-view dataset with a defined candidate-pose set for
  Track 2 initially (real-robot data as a stretch goal).

## Literature check (before drafting)

Uncertainty for GS/NeRF active mapping/NBV has moved fast since this repo
went dormant (FisherRF, Bayes' Rays, ActiveNeRF, and others, 2023–2026).
Before finalizing framing:
- Confirm current SOTA NBV/active-GS baselines and their reported numbers.
- Confirm no existing paper already stakes out "kernel-quadrature /
  Bayesian-quadrature uncertainty for GS" specifically.
- Use this to decide which baseline(s) get reproduced vs. cited.

## Milestones

1. Derivation + small-scale validation (posterior mean ≈ alpha compositing;
   posterior variance vs. brute-force baseline on a toy scene).
2. Track 1 (densification/pruning) experiment on standard GS benchmark
   scenes.
3. Track 2 (NBV) experiment, including baselines from the literature check.
4. Write-up: primer appendix, honest pilot-study section, main derivation,
   and the two experiment sections.

## Verification gates

- Step 1's validation harness is the go/no-go gate before any downstream
  experiment — if closed-form variance doesn't track a brute-force
  estimate, the plan needs rethinking.
- Track 1 is checked against vanilla gsplat's own benchmark numbers
  (splat count and PSNR/SSIM at convergence).
- Track 2 is checked against the random/heuristic baselines run in the same
  harness, plus the strongest existing NBV baseline from the literature
  check.
