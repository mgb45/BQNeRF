# BQ-Splat: a renderer-consistent sparse-GP view of 3D Gaussian Splatting

**The idea in one sentence**: treat 3D Gaussian Splatting as a sparse
interdomain Gaussian process whose alpha-composited image *is* its
posterior mean, so uncertainty falls out of the same representation
without ever touching or approximating the render itself.

## The theory, in plain language

An earlier version of this project built a Bayesian-quadrature posterior
*over* the rendering integral and asked its own solved-for weights
`w* = Kxx^-1 z` to explain real alpha compositing `w_alpha = T_i*alpha_i`.
They don't, in general — once a directional kernel is added, `w*` has no
structural relationship to `w_alpha` at all. The fix is not a better
kernel; it's a different picture of what the splats *are*.

Fix the splat geometry and treat each splat's stored SH coefficients as
an **interdomain inducing variable** of an underlying radiance-field GP,
chosen so that the GP's posterior mean, under the real renderer weights
`b_q`, is *exactly* the real image:

    mu_q = C_alpha(q) = b_q^T theta_hat

Conditioning a GP on inducing variables gives a standard, closed-form
predictive variance that decomposes into two independent terms:

    u_q = u_spatial_BQ(q) + b_q^T Sigma_theta b_q

- **`u_spatial_BQ(q)`** — the finite-spatial-representation term: the
  real alpha-compositing weights' own RKHS worst-case risk, scored (not
  solved for) under a position-only kernel. Large where splats are
  spatially sparse or the local footprint poorly resolves the query.
- **`b_q^T Sigma_theta b_q`** — parameter uncertainty in the *learned* SH
  coefficients: each splat's own Bayesian linear regression posterior
  over its SH coefficients, `Sigma_theta_i`, built from how much its real
  alpha-compositing weight contributed to each real training camera it
  was observed from. Large where a splat was seen from few or
  narrowly-clustered directions; this is `u_SH(q)` once propagated
  through the real per-pixel alpha weights.

Because the mean is pinned to the real renderer output by construction,
this is a strictly post-hoc, renderer-consistent uncertainty: it can
never corrupt the reconstruction, and it decomposes into "is this region
spatially under-resolved" vs. "is this viewing angle under-constrained,"
answerable independently.

See [`gs_experiment/sh_directional_uncertainty.py`](gs_experiment/sh_directional_uncertainty.py)
and [`gs_experiment/gpu_sh_directional_uncertainty.py`](gs_experiment/gpu_sh_directional_uncertainty.py)
for the full derivation and implementation.

## What's been tested

- **Does the mean stay exactly the real renderer's output?** Yes, by
  construction — `C_alpha(q)` is never solved for, only real alpha
  compositing.
- **Do the two uncertainty terms show visibly distinct spatial
  patterns?** Yes: `u_spatial_BQ` is sharp and structure-following;
  `u_SH` is smoother and tracks real training-view angular coverage
  (see `gs_experiment/results/sparse_gp_uncertainty.png`).
- Calibration against real held-out error, and a fitting procedure for
  the SH-coefficient prior precision `lam`, are open — see
  [`ROADMAP.md`](ROADMAP.md).

## Repo layout

- [`gs_experiment/`](gs_experiment/) — the whole project: the BQ math
  (kernels, quadrature, render weights, hyperparameter fitting), the
  SH-coefficient directional uncertainty, and the real Gaussian-Splatting
  experiments built on them (needs a GPU + `gsplat`). See
  [`gs_experiment/README.md`](gs_experiment/README.md) for the module/tool
  list.
- [`ROADMAP.md`](ROADMAP.md) — the forward-looking research plan.
- `tests/` — the active test suite (`pytest tests/`).

## Getting started

```
pip install -r requirements.txt
python -m pytest tests/ -v
```

That runs everything that doesn't need a GPU. For the real experiments,
set up `gsplat` (see [`requirements-gsplat.txt`](requirements-gsplat.txt))
and render the headline figure against already-trained checkpoints:

```
.venv-gsplat/bin/python gs_experiment/scripts/render_sparse_gp_uncertainty.py
```

`gs_experiment/README.md` has the full list of tools, what each one
tests, and which real datasets they expect.
