# BQ-Splat: uncertainty for Gaussian Splatting, nearly for free

**The idea in one sentence**: rendering a Gaussian-Splat scene is already a weighted sum over kernels or a quadrature rule. This means we can get uncertainty for "free" using Bayesian-quadrature. This gives
closed-form, per-region uncertainty from the same math already used to
render, without bolting on a separate uncertainty model.

## The theory, in plain language

A Gaussian-Splat renderer computes a pixel's color as a weighted sum of
nearby splats' colors — closer, more opaque splats contribute more. That weighted sum is a *quadrature rule*: a way of approximating an integral (here, the true light arriving along a ray) from a finite set of samples (here, the splats). **Bayesian quadrature** is a standard, decades-old way to reason about exactly this situation: put a probabilistic prior over the thing you're integrating, and the *same* observations that give you a point estimate of the integral also give you a principled variance around it, how much that estimate could plausibly be wrong, given only the finite, imperfect set of samples you actually have.

Applied to Gaussian Splatting, this means building a query-specific
renderer weight `a_q(xi) = T_q(xi) sigma(xi) G_q(xi)` (transmittance x
opacity x footprint) directly into the kernel:
`k_q(xi, xi') = a_q(xi) k_base(xi, xi') a_q(xi')` — rather than
integrating a generic base kernel uniformly over an arbitrary window. The posterior *variance* under this kernel is a closed-form number that's large when a region is thinly covered by splats (fine detail, sparse reconstruction, occluded) and small when it's well-covered — no separate learned uncertainty head, no ensemble, no dropout.

**The unifying idea** goes one step further. Build the base kernel as a
*product* of two parts — one over 3D position, one over viewing
direction — and the same posterior answers two different questions
depending on what you ask it:

- *Integrate over position, ignore direction* → **quadrature
  uncertainty**: is this region numerically well-resolved by the current splats, regardless of how many camera views actually saw it?
- *Evaluate at one specific query direction* → **directional/epistemic
  uncertainty**: is *this particular viewing angle* well-constrained by
  the directions training actually observed it from?

## What's been tested


- **Does the uncertainty signal track real sparse or missing coverage?**
  Yes, tested across all 8 standard NeRF-Synthetic benchmark scenes. 
- **Does the directional/viewing-angle-coverage signal work on real
  geometry?** Yes.
- **Does the signal also flag GS-training floaters?** Yes, needs an experiment
- **Is the number *calibrated** Yes, needs an experiment
- **Training directly under the likelihood** Todo
- **Next best view selection evaluation** Todo

## Repo layout

- [`gs_experiment/`](gs_experiment/) — the whole project: the BQ math
  (kernels, quadrature, render weights, hyperparameter fitting) and the
  real Gaussian-Splatting experiments built on it (needs a GPU +
  `gsplat`). See [`gs_experiment/README.md`](gs_experiment/README.md) for
  the module/tool list.
- [`ROADMAP.md`](ROADMAP.md) — the forward-looking research plan: what a strong paper still needs, ordered by how load-bearing each gap is.
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
gotchas already solved there) and, for example, build the cross-scene
uncertainty gallery against already-trained checkpoints:

```
.venv-gsplat/bin/python gs_experiment/scripts/render_scene_gallery.py
```

`gs_experiment/README.md` has the full list of tools, what each one
tests, and which real datasets they expect.

## Where to read more

- [`ROADMAP.md`](ROADMAP.md) — the honest state of the research plan: what's
  — the current-conclusions summary and the primary results document.
