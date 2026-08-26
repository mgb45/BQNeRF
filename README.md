# BQ-Splat: uncertainty for Gaussian Splatting, nearly for free

**The idea in one sentence**: rendering a Gaussian-Splat scene is already
a weighted sum over kernels — which is exactly what a Bayesian-quadrature
estimate of an integral looks like — so treating it as one gives a
closed-form, per-region uncertainty from the same math already used to
render, without bolting on a separate uncertainty model.

This repo is a research project, not a finished tool. It includes real
negative and inconclusive results alongside positive ones, reported the
same way — see [What's been tested](#whats-been-tested-real-scenes-first)
below.

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

Applied to Gaussian Splatting, this means: model the splats' contribution
to a query as noiseless observations of a Gaussian process. The
posterior *mean* of that GP, integrated over the relevant region, is (up
to the model's own assumptions) the same thing standard alpha-compositing
computes — proven directly, not just checked empirically, in
[`bq_splat/PROOF_alpha_compositing_equivalence.md`](bq_splat/PROOF_alpha_compositing_equivalence.md).
The posterior *variance* is new: a closed-form number that's large when a
region is thinly covered by splats (fine detail, sparse reconstruction)
and small when it's well-covered — no separate learned uncertainty head,
no ensemble, no dropout.

**The unifying idea** goes one step further. Build the kernel as a
*product* of two parts — one over 3D position, one over viewing
direction — and the same posterior answers two different questions
depending on what you ask it:

- *Integrate over position, ignore direction* → **quadrature
  uncertainty**: is this region numerically well-resolved by the current
  splats, regardless of how many camera views actually saw it?
- *Evaluate at one specific query direction* → **directional/epistemic
  uncertainty**: is *this particular viewing angle* well-constrained by
  the directions training actually observed it from?

The literature on Gaussian-Splatting uncertainty treats these as two
different problems needing two different mechanisms (a Hessian/Fisher
sensitivity computation for one, a dedicated visibility field for the
other). This project's claim is that they're the same worst-case-error
theorem applied to different questions about one posterior — proven
formally, not just asserted, in the proof document linked above.

## What's been tested, real scenes first

Every result below is backed by a real experiment against real trained
checkpoints — see each package's `results/FINDINGS.md` for the numbers,
and `results/ARCHIVE_FULL_LOG.md` for the complete, warts-and-all process
behind them (every bug, every dead end, every correction).

- **Does the uncertainty signal track real sparse or missing coverage?**
  Tested across all 8 standard NeRF-Synthetic benchmark scenes (not one
  scene picked for convenience) — yes, robustly: local splat density and
  BQ variance correlate strongly and significantly on every single scene.
  It also responds specifically to genuine angular coverage gaps, not
  just raw view count.
- **Does that hold up against a real reference trainer, not just this
  project's own code?** Yes — the same result replicates on a checkpoint
  trained by `gsplat`'s own official densification strategy, code this
  project didn't write.
- **Is the number *calibrated*, not just correlated?** A more honest,
  nuanced answer: it's useful for *ranking* regions (which is what a
  pruning or active-view-selection policy actually needs, and where it
  measurably helps), but not yet trustworthy as an absolute confidence
  value — held-out likelihood is currently worse than a flat baseline.
  Reported as a real, unresolved gap, not glossed over.
- **Does kernel choice matter?** Yes, and there's no free lunch: one
  kernel family is more robust for the sparsity-tracking claim and for
  absolute calibration, a different one is more robust for ranking-based
  calibration. No universally-better choice found.
- **Does the directional/viewing-angle-coverage signal work on real
  geometry, including an actual photograph?** On a scene designed to
  have a clean, continuous coverage gradient — yes, cleanly. On real
  geometry, including a genuinely photographed scene with real
  COLMAP-estimated camera poses — an honestly open question, not a clean
  result either way. An early attempt looked like a negative result but
  turned out to be built on reconstructions too poor to trust; a
  properly-resourced retest is a real, moderately-well-grounded null with
  one specific, named confound still unresolved. This project would
  rather report "we don't know yet, and here's exactly why" than round
  an inconclusive result to a clean answer in either direction.
- **What didn't work**: training a model directly under this
  uncertainty (as a loss weight, and as a densification trigger) — tried
  directly, found not to help, kept in as a real negative result rather
  than left untested or quietly dropped.
- **Toy-scale foundations** (`bq_splat/`): before any of the above, the
  core math (does the closed-form variance behave sensibly, is it
  calibrated, does it rise in genuinely under-resolved regions, what does
  it cost at real Gaussian-Splatting scale) was validated on cheap,
  synthetic 1D/2D signals — no GPU needed. This is where a real
  computational-scaling concern was resolved (the assumed bottleneck
  wasn't the one that mattered) before any GPU time was spent, and where
  the directional-kernel idea was first shown to work at all.

## Repo layout

- [`bq_splat/`](bq_splat/) — toy-scale math validation (pure numpy/scipy,
  no GPU) and the formal proof. Start here to understand the theory.
- [`gs_experiment/`](gs_experiment/) — real Gaussian-Splatting experiments
  (needs a GPU + `gsplat`). The results above live here.
- [`archive/original_nerf_prototype/`](archive/original_nerf_prototype/)
  — where this project actually started (Bayesian quadrature for a
  from-scratch NeRF), kept for the record. Not part of the current work.
- [`ROADMAP.md`](ROADMAP.md) — the forward-looking research plan: what a
  strong paper still needs, ordered by how load-bearing each gap is.
- `tests/` — the active test suite (`pytest tests/`).

## Getting started

```
pip install -r requirements.txt
python -m pytest tests/ -v
```

That runs everything that doesn't need a GPU, including the full toy
validation suite. For the real experiments, set up `gsplat` (see
[`requirements-gsplat.txt`](requirements-gsplat.txt) for a from-scratch
setup, including a couple of real CUDA/compiler gotchas already solved
there) and, for example, run the sparsity-correlation check against a
real checkpoint:

```
.venv-gsplat/bin/python gs_experiment/sparsity_correlation_experiment.py <path/to/splats.ply>
```

`gs_experiment/README.md` has the full list of experiment scripts, what
each one tests, and which real datasets they expect.

## Where to read more

- [`ROADMAP.md`](ROADMAP.md) — the honest state of the research plan: what's
  done, what's still open, ordered by priority for a paper.
- [`bq_splat/results/FINDINGS.md`](bq_splat/results/FINDINGS.md) /
  [`gs_experiment/results/FINDINGS.md`](gs_experiment/results/FINDINGS.md)
  — current-conclusions summaries.
- Each has a companion `ARCHIVE_FULL_LOG.md` — the complete process log,
  every bug and every intermediate number, for anyone who wants the whole
  story.
