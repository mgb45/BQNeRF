# BQ-Splat: 3D Gaussian Splatting is a quadrature rule, so uncertainty is nearly free

**The idea in one sentence**: alpha compositing is already a weighted
quadrature sum whose weights the rasterizer computes anyway, so a posterior
over the splats' appearance pushes forward to a per-pixel predictive
variance at the cost of a few extra renders.

## The theory, in plain language

A rendered pixel is

    C(q) = sum_i beta_{q,i} c_i(d_q),    beta_{q,i} = T_i * alpha_i

-- nodes = splats, weights = `beta_{q,i}`, i.e. a quadrature rule, and the
renderer already evaluates it. Nothing about the render has to be
approximated, replaced or re-derived to get uncertainty out of it; the
weights are right there.

So put a posterior on the thing being summed. Each splat's stored SH
coefficients are a learned, localized basis function -- the representation
photometric training actually optimizes -- and a Gaussian likelihood over
real training pixels gives each splat a closed-form coefficient posterior:

    D_i = sum_p (sum_q beta_{q,i,p}^2) phi(d_{i,p}) phi(d_{i,p})^T
    P_i = Lambda + D_i / sigma_n^2,   Sigma_theta_i = P_i^-1

The cheapest and most legible way to push that through the quadrature rule
is not a quadratic form -- it is sampling:

> **Render the scene a few times with the splat colours drawn from their
> posterior. The pixel-wise spread is the uncertainty.**

No retraining, no model ensemble, one checkpoint, k renders. The mean of the
ensemble is the real render, untouched, so this is strictly post-hoc and
cannot corrupt the reconstruction.

## Everything comes from the real rasterizer

`sum_q beta_{q,i}^2` -- how much information training view p carries about
splat i -- is read off gsplat itself. For arbitrary per-splat features
`c_i`, gsplat renders `I(q) = sum_i beta_{q,i} c_i`, so backpropagating an
image `r` gives `dL/dc_i = sum_q beta_{q,i} r_q` exactly. With `r`
Rademacher, `E[(dL/dc_i)^2] = sum_q beta_{q,i}^2`: the wanted quantity,
summed over the splat's whole real footprint, under the renderer's own
weights. Independent probes ride as independent channels of a single render,
so the whole accumulation is one forward+backward pass per training camera
(5 s for a 300k-splat, 100-camera scene).

This matters more than it sounds. The previous version of this project
reconstructed `beta` with a surrogate compositing model and was wrong by a
median factor of 2.1e18, which made the SH posterior exactly equal to its
prior -- see [`gs_experiment/results/FINDINGS.md`](gs_experiment/results/FINDINGS.md)
section 0, which records the defect, how it evaded the test suite, and the
correction.

## What's been shown

- **The mean is the real render**, by construction -- the ensemble is
  centred on the checkpoint's own coefficients.
- **On a fully-observed scene the posterior is tight.** 300k splats, 100
  training views: per-pixel std 0.002, draws visually identical. Correct,
  and a useful sanity result.
- **Spread tracks how much data you conditioned on.** On ONE frozen
  checkpoint, varying only the number of training cameras the posterior is
  conditioned on (100/25/8/3), per-pixel std on the object rises
  0.0057 -> 0.0100 -> 0.0373 -> 0.0430 with no retraining confound at all
  (`gs_experiment/results/posterior_view_sweep.png`).
- **Cost**: 8 draws = 323 ms of renders on a 21.5 ms render.
- **Per-VIEW calibration is excellent where the error is epistemic.** On
  checkpoints trained with a deliberate angular hole in their training
  views, per-view predicted uncertainty tracks per-view held-out error at
  Spearman **0.97** (75 deg hole), 0.82 (30 deg), 0.61 (no hole) -- the
  signal gets better precisely as the error becomes more epistemic, which is
  what a posterior over fitted parameters should do
  (`gs_experiment/results/epistemic_regime.png`).
- **Per-pixel uncertainty is calibrated.** Scored against the ceiling a
  perfectly calibrated sigma could attain (rank metrics are capped well below
  1 by the single-realization noise `|eps| = sigma|z|`), the per-pixel
  ranking reaches **98% of attainable** in the epistemic regime and 52% on a
  fully-observed checkpoint. A two-parameter fit (scale + aleatoric floor),
  fitted on half the held-out views and scored on the other half, beats a
  constant-variance baseline by **0.68 nats** in the epistemic regime and
  0.08 nats when the model is fully constrained. Whole-frame correlations on
  NeRF-Synthetic run ~0.95 but that is the object/background silhouette, not
  calibration, and is not quoted as such anywhere here.
- **Two richer posteriors were tried and made it worse**: cross-splat
  coupling (1000x cost, slightly worse correlation) and
  opacity-in-the-posterior (2x cost, worse). Both are implemented,
  validated, and recorded as negative results in FINDINGS sections 6-7.

## Repo layout

- [`gs_experiment/`](gs_experiment/) -- the whole project. `rasterized_sh_precision.py`
  is the current method; `scripts/render_posterior_ensemble.py` and
  `scripts/render_posterior_view_sweep.py` are the figures. See
  [`gs_experiment/README.md`](gs_experiment/README.md) for the module list.
- [`ROADMAP.md`](ROADMAP.md) -- the forward research plan.
- `tests/` -- the active test suite (`pytest tests/`).

## Getting started

```
pip install -r requirements.txt
python -m pytest tests/ -v
```

That runs everything that doesn't need a GPU. For the real experiments, set
up `gsplat` (see [`requirements-gsplat.txt`](requirements-gsplat.txt)) and
render the headline figure against already-trained checkpoints:

```
.venv-gsplat/bin/python gs_experiment/scripts/render_posterior_view_sweep.py
```
