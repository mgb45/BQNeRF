# BQ-Splat: uncertainty for Gaussian Splatting, nearly for free

**The idea in one sentence**: rendering a Gaussian-Splat scene is already
a weighted sum over kernels — which is exactly what a Bayesian-quadrature
estimate of an integral looks like — so treating it as one gives a
closed-form, per-region uncertainty from the same math already used to
render, without bolting on a separate uncertainty model.

This repo is a research project, not a finished tool. It includes real
negative and inconclusive results alongside positive ones, reported the
same way — see [What's been tested](#whats-been-tested) below.

## The theory, in plain language

A Gaussian-Splat renderer computes a pixel's color as a weighted sum of
nearby splats' colors — closer, more opaque splats contribute more. That
weighted sum is a *quadrature rule*: a way of approximating an integral
(here, the true light arriving along a ray) from a finite set of samples
(here, the splats). **Bayesian quadrature** is the standard, decades-old
way to reason about exactly this situation: put a probabilistic prior
over the thing you're integrating, and the *same* observations that give
you a point estimate of the integral also give you a principled variance
around it — how much that estimate could plausibly be wrong, given only
the finite, imperfect set of samples you actually have.

Applied to Gaussian Splatting, this means building a query-specific
renderer weight `a_q(xi) = T_q(xi) sigma(xi) G_q(xi)` (transmittance x
opacity x footprint) directly into the kernel:
`k_q(xi, xi') = a_q(xi) k_base(xi, xi') a_q(xi')` — rather than
integrating a generic base kernel uniformly over an arbitrary window. The
posterior *variance* under this kernel is a closed-form number that's
large when a region is thinly covered by splats (fine detail, sparse
reconstruction, occluded) and small when it's well-covered — no separate
learned uncertainty head, no ensemble, no dropout.

**The unifying idea** goes one step further. Build the base kernel as a
*product* of two parts — one over 3D position, one over viewing
direction — and the same posterior answers two different questions
depending on what you ask it:

- *Integrate over position, ignore direction* → **quadrature
  uncertainty**: is this region numerically well-resolved by the current
  splats, regardless of how many camera views actually saw it?
- *Evaluate at one specific query direction* → **directional/epistemic
  uncertainty**: is *this particular viewing angle* well-constrained by
  the directions training actually observed it from?

## What's been tested

Every result below is backed by a real experiment against real trained
checkpoints — see [`gs_experiment/results/FINDINGS.md`](gs_experiment/results/FINDINGS.md)
for the numbers and the full reasoning behind them.

- **Does the uncertainty signal track real sparse or missing coverage?**
  Tested across all 8 standard NeRF-Synthetic benchmark scenes (not one
  scene picked for convenience) — yes, robustly: local splat density and
  BQ variance correlate significantly on every single scene (`r` between
  `-0.30` and `-0.56`).
- **Does the directional/viewing-angle-coverage signal work on real
  geometry?** Yes, cleanly: a real lego checkpoint with a deliberate,
  carefully-controlled angular coverage gap now shows the per-pixel
  uncertainty ratio sweeping from 0.03 (full coverage) to 0.97 (widest
  gap), strictly increasing across all 5 conditions — the project's
  cleanest result. On a genuinely photographed scene (real COLMAP-estimated
  camera poses) — still an open question, one condition run so far.
- **Does the signal also flag GS-training floaters?** Yes, and this is
  distinct from view coverage: two checkpoints of the same scene with
  identical real camera coverage differed 35x in mean uncertainty purely
  because one had floaters (stray splats displaced outside the training
  volume, a normal optimization artifact) and the other didn't. Whether
  this is a confound or a genuinely useful second signal depends on the
  use case — see `ROADMAP.md`.
- **Is the number *calibrated* against rendering error, not just
  correlated with coverage?** An earlier "essentially zero correlation"
  finding does not survive a clean re-test: with the pipeline bugs fixed
  and floater-free checkpoints, per-pixel uncertainty correlates with
  real `|error|` at `r=0.69` pooled (up to `r=0.64` within a single
  coverage condition), strongly significant. Correlation is weak under
  full coverage and grows sharply as coverage worsens — see
  `gs_experiment/results/FINDINGS.md`'s session-update section for the
  full breakdown and the one-scene caveat.
- **Training directly under the likelihood** (as a loss weight, and as a
  densification trigger): tried once, got a discouraging result — but
  under an implementation from the same pre-fix era as the calibration
  finding above, which didn't hold up on re-test. Not re-run since, so
  treated as an open hypothesis, not a settled negative. A more targeted,
  floater-specific version of the idea is proposed in `ROADMAP.md`.

## Repo layout

- [`gs_experiment/`](gs_experiment/) — the whole project: the BQ math
  (kernels, quadrature, render weights, hyperparameter fitting) and the
  real Gaussian-Splatting experiments built on it (needs a GPU +
  `gsplat`). See [`gs_experiment/README.md`](gs_experiment/README.md) for
  the module/tool list.
- [`ROADMAP.md`](ROADMAP.md) — the forward-looking research plan: what a
  strong paper still needs, ordered by how load-bearing each gap is.
- `tests/` — the active test suite (`pytest tests/`).

A handful of library modules (kernels, quadrature, render weights, camera/
scene I/O) plus a small number of general, flag-driven CLI tools under
`gs_experiment/scripts/` — each tool's flags select among what used to be
separate one-off scripts, with the underlying math and results unchanged.

## Getting started

```
pip install -r requirements.txt
python -m pytest tests/ -v
```

That runs everything that doesn't need a GPU. For the real experiments,
set up `gsplat` (see [`requirements-gsplat.txt`](requirements-gsplat.txt)
for a from-scratch setup, including a couple of real CUDA/compiler
gotchas already solved there) and, for example, run the
sparsity-correlation check against a real checkpoint:

```
.venv-gsplat/bin/python gs_experiment/scripts/evaluate_checkpoint.py sparsity <path/to/splats.ply>
```

`gs_experiment/README.md` has the full list of tools, what each one
tests, and which real datasets they expect.

## Where to read more

- [`ROADMAP.md`](ROADMAP.md) — the honest state of the research plan: what's
  done, what's still open, ordered by priority for a paper.
- [`gs_experiment/results/FINDINGS.md`](gs_experiment/results/FINDINGS.md)
  — the current-conclusions summary and the primary results document.
