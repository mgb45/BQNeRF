# gs_experiment — Bayesian quadrature on real Gaussian Splatting

Real experiments against real trained `gsplat` checkpoints: the BQ math
(kernels, quadrature, the directional extension) and its application to
real 3D Gaussian Splatting scenes — real cameras, real training, real
densification — live in this one package. Needs a GPU and `gsplat` for
anything that trains or loads a checkpoint (see
[`../requirements-gsplat.txt`](../requirements-gsplat.txt) for setup,
including two real CUDA/compiler gotchas already solved there); the pure
math/geometry modules (`kernels.py`, `quadrature.py`, `render_weight.py`,
`hyperparams.py`, `camera.py`, `visibility_attribution.py`,
`spherical_harmonics.py`) run on `numpy`/`scipy` alone and are covered by
the main test suite (`pytest tests/`).


## Core library modules

- **`kernels.py`** — `RBFKernel`/`MaternKernel` (each with a closed-form
  or numerically-integrated mean embedding and double integral),
  `ProductKernel` (a D-D kernel as a product of 1D kernels per axis,
  exact for RBF), `DirectionalKernel` (a von Mises-Fisher factor over
  viewing direction, combined multiplicatively with a position kernel).
- **`quadrature.py`** — the rendering-aware BQ family:
  `bayesian_quadrature_rendering_aware` /
  `bayesian_quadrature_rendering_aware_directional` /
  `renderer_centered_residual_variance`. `k_q(xi, xi') = a_q(xi)
  k_base(xi, xi') a_q(xi')` for a query-specific renderer weight `a_q`
  (see `render_weight.py`), in place of a uniform-box integration domain.
  An earlier multi-D/directional box-quadrature family
  (`bayesian_quadrature_nd`/`bayesian_quadrature_directional`/
  `directional_posterior_variance`) that predates this is retired -- see
  git history.
- **`render_weight.py`** — `GaussianRenderWeight`: `a_q = T_q sigma G_q`
  modeled as an unnormalized Gaussian bump (amplitude, center,
  covariance), for the closed-form rendering-aware quadrature above.
- **`hyperparams.py`** — fits the kernel bandwidth (RBF sigma / Matern
  rho) to data by maximizing the GP log marginal likelihood, instead of a
  hardcoded bandwidth. `fit_kernel_param_pooled_nd` fits one shared
  bandwidth across many datasets/windows, for testing whether a single
  fitted bandwidth generalizes.
- **`pixel_uncertainty.py`** — `LocalUncertaintyEngine`: the main entry
  point for querying rendering-aware BQ variance against a real
  checkpoint. Builds a KD-tree once for candidate lookup and caps
  local-neighbor count for tractability. `rendering_aware_variance`
  builds a real per-query `a_q` (see `render_weight.py`) from actual
  per-splat opacity and a Gaussian footprint tied to the query radius, so
  a low-opacity splat contributes less to both mean and variance by
  construction -- but the amplitude is occlusion-blind (a flat
  neighborhood-mean opacity). `rendering_aware_variance_along_ray` closes
  that: real, depth-ordered alpha-compositing transmittance weights
  along the specific ray from a given camera through the query point
  (`visibility_attribution.ray_transmittance_weights`), so a splat behind
  a closer, opaque splat *on that ray* gets a small weight from real
  accumulated transmittance rather than a uniform average -- but still
  via an isotropic bearing threshold, not each splat's real projected
  shape. `rendering_aware_variance_via_gsplat` is the most faithful
  version: real gsplat GPU projection (`gsplat_rendering_weights.py`)
  gives each local splat its actual anisotropic 2D footprint and real
  per-pixel alpha for a given camera + intrinsics, in place of the
  isotropic-bearing proxy. Needs `scales`/`rotations` on the engine and a
  GPU + gsplat env (see `../requirements-gsplat.txt`); still not a live
  differentiable rasterizer in the full sense (no antialiasing/sub-pixel
  footprint integration, no gradient path -- runs under
  `torch.no_grad()`) -- see that method's docstring for exactly what is
  and isn't modeled. Each method has a `_directional` variant completing
  the joint position+direction kernel
  (`bayesian_quadrature_rendering_aware_directional`) for the
  complementary "is this specific viewing angle well-constrained"
  question.
- **`splat_scene.py`** — `load_from_gsplat_checkpoint` (reads a real
  `.ply` + `transforms.json`), `splat_observations` (expands a scene into
  the (position, direction, value) rows the directional kernel needs;
  `include_render_attrs=True` also returns per-row opacity/scale/rotation,
  for the rendering-aware directional methods above).
- **`camera.py`** — camera pose representation, turntable pose
  generation, and per-splat viewing-direction geometry, including
  `viewmat_from_camera_pose`/`project_point_to_pixel` (the pure-numpy
  camera-to-gsplat-rasterization-boundary conversion `gsplat_rendering_weights.py`
  uses).
- **`gsplat_rendering_weights.py`** — `gsplat_alpha_compositing_weights`:
  real per-pixel alpha-compositing weights via gsplat's own differentiable
  EWA-splatting projection (`gsplat.fully_fused_projection`) -- needs
  torch + a CUDA-enabled gsplat build and a GPU; kept out of
  `pixel_uncertainty.py`'s top-level imports (lazily imported by
  `rendering_aware_variance_via_gsplat`) so that module and the default
  `pytest tests/` suite stay importable without torch/gsplat installed.
- **`visibility_attribution.py`** — frustum + soft-z-buffer occlusion
  proxy for "which cameras plausibly saw this splat" (real training
  pipelines don't record this). `ray_transmittance_weights` is the
  continuous analogue for one specific ray: real per-splat opacity as
  alpha, depth-ordered into a genuine alpha-compositing transmittance
  weight per splat, instead of `occlusion_mask`'s binary yes/no --
  what `pixel_uncertainty.rendering_aware_variance_along_ray` uses.
  `CameraSplatIndex` is the bearing-space candidate index those queries
  share across many calls against one camera; its `directions=`/
  `query_direction=` option ranks overflow candidates by directional
  alignment rather than bearing-distance ties, which matters whenever
  `positions` is a camera-expanded observation array (one row per
  (splat, observing-camera) pair) -- see its docstring.
- **`spherical_harmonics.py`** — `eval_sh`, matching the standard
  3DGS/gsplat SH color convention.
- **`ply_io.py`** / **`nerf_transforms.py`** — the standard 3DGS `.ply`
  schema and NeRF-style `transforms.json` I/O, including the OpenCV/OpenGL
  convention conversions `colmap_loader.py` also uses.
- **`colmap_loader.py`** — reads real COLMAP camera poses (the format
  real photographed datasets like Mip-NeRF360 ship), for scenes where
  poses are an SfM *estimate*, not exactly known.
- **`train_minimal_gsplat.py`** — a minimal from-scratch `gsplat` trainer

## Entry-point tools

These are the tools that actually produce this project's current results — one file per figure, reusing
`render_reconstruction.py`'s shared `render_views`/`compute_uncertainty_maps`
(no CLI of its own) rather than duplicating rendering logic per script.

- **`prepare_nerf_synthetic.py`** — downloads/prepares a standard
  NeRF-Synthetic scene (100 real training views + an official held-out
  test split); also builds a "narrow" (angularly clustered) real-view
  subset and graded-spread/gap conditions used by the tools below.
- **`render_scene_gallery.py`** — the cross-scene qualitative result:
  one held-out view per scene (7 of the 8 standard NeRF-Synthetic
  scenes; `materials` excluded, see the script's own docstring), at two
  splat budgets (500 and this project's standard 300k `wide` recipe)
  side by side, showing ground truth / reconstruction / |error| / raw
  rendering-aware posterior variance. `sigma`/`kappa` default to fitting
  per checkpoint (`splat_scene.fit_kernel_hyperparams`) rather than
  reusing one bandwidth pooled across a fixed calibration set.
- **`render_splat_sweep_gallery.py`** — same columns as
  `render_scene_gallery.py`, but rows are increasing splat budgets (500
  to 1,000,000 by default) for one scene (lego), showing reconstruction
  quality and BQ uncertainty improving and saturating together. Large
  budgets are automatically capped via
  `splat_scene.max_observations_per_splat_for_budget` to stay within a
  validated host-memory ceiling.
- **`real_directional_coverage_experiment.py`** — trains the lego
  coverage-gap checkpoints `render_coverage_uncertainty_sweep.py` needs:
  removes a deliberate angular gap of increasing half-width from the
  100-view training pool, leaving every other view untouched (confound-
  free relative to an earlier, retired "subsample" design that thinned
  the pool globally; see git history).
- **`render_coverage_uncertainty_sweep.py`** — the directional-coverage
  result: the same held-out view rendered against each gap condition's
  own checkpoint, showing reconstruction degrading and raw posterior
  variance growing together in the missing-coverage region.

## Running it

```
pytest ../tests/gs_experiment -v
```

runs everything that doesn't need a GPU. For a real experiment end to
end, with a `gsplat` environment set up (`../requirements-gsplat.txt`):

```
.venv-gsplat/bin/python gs_experiment/scripts/prepare_nerf_synthetic.py <raw_scene_dir> <out_dir>
.venv-gsplat/bin/python -m gs_experiment.scripts.train_minimal_gsplat <out_dir>/wide <out_dir>/wide/splats.ply --densify
.venv-gsplat/bin/python gs_experiment/scripts/render_scene_gallery.py
```
