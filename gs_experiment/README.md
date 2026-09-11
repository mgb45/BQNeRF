# gs_experiment — a renderer-consistent sparse-GP view of 3D Gaussian Splatting

Real experiments against real trained `gsplat` checkpoints, implementing
the decomposition described in the top-level [`README.md`](../README.md):

    mu_q = C_alpha(q)                          (real alpha compositing, untouched)
    u_q  = u_spatial_BQ(q) + u_SH(q)

Needs a GPU and `gsplat` for anything that trains or loads a checkpoint
(see [`../requirements-gsplat.txt`](../requirements-gsplat.txt)); the pure
math/geometry modules (`kernels.py`, `quadrature.py`, `render_weight.py`,
`hyperparams.py`, `camera.py`, `visibility_attribution.py`,
`spherical_harmonics.py`, `sh_directional_uncertainty.py`) run on
`numpy`/`scipy` alone and are covered by the main test suite
(`pytest tests/`).

## Core library modules

- **`kernels.py`** — `RBFKernel` (closed-form mean embedding/double
  integral) and `ProductKernel` (a D-D kernel as a product of 1D kernels
  per axis, exact for RBF).
- **`quadrature.py`** — `bayesian_quadrature_rendering_aware` (the
  BQ-optimal position-only estimator under a query-specific renderer
  weight `a_q`, see `render_weight.py`) and
  `rendering_aware_alternative_weight_risk` (the general RKHS
  worst-case-risk formula, scored at the REAL alpha-compositing weights
  rather than solved for) -- this is `u_spatial_BQ(q)`'s scalar reference.
- **`render_weight.py`** — `GaussianRenderWeight`: `a_q = T_q sigma G_q`
  modeled as an unnormalized Gaussian bump (amplitude, center,
  covariance).
- **`hyperparams.py`** — fits the RBF bandwidth (and, jointly, a real
  homoscedastic observation-noise variance) to data by maximizing the GP
  log marginal likelihood, instead of a hardcoded bandwidth.
- **`pixel_uncertainty.py`** — `LocalUncertaintyEngine.
  rendering_aware_alpha_risk_along_ray`: real, depth-ordered
  alpha-compositing transmittance weights along the specific camera ray
  through a query point (`visibility_attribution.ray_transmittance_weights`)
  build a position-only `a_q`, then `alpha_risk` scores those SAME real
  weights' own RKHS risk under that kernel -- `u_spatial_BQ(q)`.
- **`gpu_uncertainty.py`** — `compute_alpha_risk_batched`: the whole-image
  batched-GPU equivalent of the scalar method above.
- **`sh_directional_uncertainty.py`** — the directional term: `sh_basis`
  (the real SH basis 3DGS's renderer evaluates against, pulled out as a
  design matrix) and the per-splat Bayesian linear regression posterior
  covariance `Sigma_theta_i` this project's SH-coefficient uncertainty is
  built from.
- **`gpu_sh_directional_uncertainty.py`** — `accumulate_sh_precision`:
  builds `Sigma_theta_i` for every splat by literally rerendering every
  real training camera (`compute_own_alpha_weight_batched`, each splat's
  real alpha-compositing weight at its own projected bearing); and
  `compute_sh_directional_uncertainty_batched`: the query-side batched
  evaluation of `u_SH(q) = sum_i beta_{q,i}^2 * phi(d_q)^T Sigma_theta_i
  phi(d_q)`.
- **`splat_scene.py`** — `load_from_gsplat_checkpoint` (reads a real
  `.ply` + `transforms.json`), `fit_kernel_hyperparams`/
  `fit_kernel_hyperparams_with_noise` (per-checkpoint bandwidth fitting).
- **`camera.py`** — camera pose representation, turntable pose
  generation, and per-splat viewing-direction geometry.
- **`visibility_attribution.py`** — frustum + soft-z-buffer occlusion
  proxy for "which cameras plausibly saw this splat" (real training
  pipelines don't record this), and `ray_transmittance_weights`: real
  per-splat opacity turned into a genuine alpha-compositing transmittance
  weight along one ray. `CameraSplatIndex` is the bearing-space candidate
  index queries share across many calls against one camera.
- **`gpu_visibility_attribution.py`** — batched-GPU equivalent of the
  attribution above, ~100x faster on a real checkpoint.
- **`spherical_harmonics.py`** — `eval_sh`, matching the standard
  3DGS/gsplat SH color convention.
- **`ply_io.py`** / **`nerf_transforms.py`** — the standard 3DGS `.ply`
  schema and NeRF-style `transforms.json` I/O.
- **`colmap_loader.py`** — reads real COLMAP camera poses (the format
  real photographed datasets like Mip-NeRF360 ship), for scenes where
  poses are an SfM *estimate*, not exactly known.
- **`train_minimal_gsplat.py`** — a minimal from-scratch `gsplat` trainer
  with real densify/prune, used to produce the checkpoints everything
  else runs against.

## Entry-point tools

- **`prepare_nerf_synthetic.py`** — downloads/prepares a standard
  NeRF-Synthetic scene (100 real training views + an official held-out
  test split).
- **`scripts/render_reconstruction.py`** — shared rendering library:
  `render_views` (GT vs. real gsplat reconstruction) and
  `_render_and_unproject` (real depth-unprojection into world-space query
  points), plus the per-scene checkpoint/eval-dir registry
  (`CHECKPOINTS`/`EVAL_DIRS`).
- **`scripts/render_sparse_gp_uncertainty.py`** — the current headline
  result: renders `mu_q = C_alpha(q)`, `u_spatial_BQ(q)`, `u_SH(q)`, and
  their sum, for real held-out views on real checkpoints.

## Running it

```
pytest ../tests/gs_experiment -v
```

runs everything that doesn't need a GPU. For a real experiment end to
end, with a `gsplat` environment set up (`../requirements-gsplat.txt`):

```
.venv-gsplat/bin/python gs_experiment/scripts/prepare_nerf_synthetic.py <raw_scene_dir> <out_dir>
.venv-gsplat/bin/python -m gs_experiment.scripts.train_minimal_gsplat <out_dir>/wide <out_dir>/wide/splats.ply --densify
.venv-gsplat/bin/python gs_experiment/scripts/render_sparse_gp_uncertainty.py
```
