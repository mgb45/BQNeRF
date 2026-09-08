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

**Results**: see [`results/FINDINGS.md`](results/FINDINGS.md) for the
current-conclusions summary (real scenes first).

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
  with real gradient-triggered densification (`train`), the reference
  strategy variant (`train_with_reference_strategy`), and an
  `--nll-experiment` mode: training directly under the BQ likelihood, as
  a loss term and as a densification trigger — a real negative result,
  kept in (`results/FINDINGS.md` §4).
- **`fit_hyperparameters.py`** — extends `hyperparams.py`'s
  marginal-likelihood bandwidth fitting to a real checkpoint's local
  windows, with a held-out check.

## Entry-point tools

These are the general, flag-driven tools that replace what used to be
many one-off scripts — one file per *kind* of question, not one file per
run.

- **`prepare_nerf_synthetic.py`** — downloads/prepares a standard
  NeRF-Synthetic scene (100 real training views + an official held-out
  test split); also builds a "narrow" (angularly clustered) real-view
  subset and graded-spread/gap conditions used by the tools below.
- **`run_synthetic_pipeline.py --scenes chair,drums,lego,...`** — the
  one-command version of "train, then render it and look": downloads/
  prepares/trains each named scene (reusing `evaluate_checkpoint.py`'s
  multi-scene machinery), checks each against the same held-out-PSNR
  quality gate as `render_directional_uncertainty_sweep.py`, then plots
  ground truth / reconstruction / error / BQ uncertainty for a few
  held-out (never trained-on) example views per scene, one figure per
  scene under `results/pipeline_<scene>.png`. `--n-examples`,
  `--sigma`/`--window-radius`, `--min-psnr`/`--force`, and
  `--skip-download`/`--skip-prepare`/`--skip-train` (to reuse whatever's
  already on disk) are all exposed.
- **`render_reconstruction.py`** — renders ground-truth vs. reconstruction
  comparisons; worth running before trusting any uncertainty number off a
  new checkpoint (a real past incident: a degenerate, blank reconstruction
  still reported a deceptively reasonable PSNR — see `FINDINGS.md`).
- **`render_directional_uncertainty_sweep.py`** — the general entry
  point for looking at BQ uncertainty on an arbitrary real checkpoint:
  `--mode directional` (default) renders a per-pixel, per-frame animated
  sweep showing spatial (quadrature) and directional (epistemic)
  uncertainty side by side with the reconstruction; `--mode
  position-only` renders the same sweep without the directional panel;
  `--mode view-projection` projects real splat positions into specific
  dataset camera views instead of an orbit. Auto-frames the camera from
  the checkpoint's own splat extent and refuses to compute anything on a
  checkpoint that fails a mandatory held-out-PSNR quality gate
  (`--min-psnr`, `--force` to override). RBF-only for now (`--kernel-
  family` accepts only `rbf` -- `rendering_aware_variance_via_gsplat`'s
  closed form doesn't support other base kernels yet). This is the
  primary way this project validates a new result: render it and look,
  rather than reach for statistics first (see `ROADMAP.md`).
- **`evaluate_checkpoint.py {sparsity,calibration,kernel-ablation,
  wide-vs-narrow,multi-scene}`** — the statistical checks behind
  `FINDINGS.md` §1-2, §5: does local splat density correlate with BQ
  variance (`sparsity`); is the variance calibrated via leave-one-out
  cross-validation, AUSE, and held-out NLL (`calibration`); sparsity +
  calibration at each checkpoint's own fitted RBF bandwidth across three
  fixed real checkpoints (`kernel-ablation` -- narrowed to RBF-only, see
  above); wide vs. narrow view-pool comparison (`wide-vs-narrow`); and the
  full 8-scene NeRF-Synthetic benchmark run via `--scenes` (`multi-scene`).
  (`window-ablation`/`visibility-trend` modes were retired -- each already
  has a concluded, summarized result in `FINDINGS.md`; see git history.)
- **`real_directional_coverage_experiment.py --dataset {lego,bonsai}`**
  — the directional/viewing-angle-coverage question on real geometry
  (`FINDINGS.md` §3): removes a deliberate angular gap from an otherwise-
  dense real view pool (confound-free, the current best-trusted design);
  `--dataset` selects lego or the real photographed Mip-NeRF360 "bonsai"
  scene. (An earlier "subsample" design -- thin the view pool, vary
  spread at fixed count -- confounded spread with global thinning and is
  retired; see git history.)

The designed hand-built-scene track this section used to also cover
(`scene_spec.py`, `blender_render.py`, `visibility_baseline.py`,
`designed_scene_experiments.py` -- pruning, NBV, differentiation,
declustering-isolation) predates the real-checkpoint path above and is
retired; its results are still documented, as historical record, in
`results/FINDINGS.md` §6. See git history to resurrect the code.

## Running it

```
pytest ../tests/gs_experiment -v
```

runs everything that doesn't need a GPU. For a real experiment end to
end, with a `gsplat` environment set up (`../requirements-gsplat.txt`):

```
.venv-gsplat/bin/python gs_experiment/scripts/prepare_nerf_synthetic.py <raw_scene_dir> <out_dir>
.venv-gsplat/bin/python -m gs_experiment.scripts.train_minimal_gsplat <out_dir>/wide <out_dir>/wide/splats.ply --densify
.venv-gsplat/bin/python gs_experiment/scripts/render_reconstruction.py <out_dir>/wide   # sanity-check before trusting anything below
.venv-gsplat/bin/python gs_experiment/scripts/evaluate_checkpoint.py sparsity <out_dir>/wide/splats.ply
```

or, for the full 8-scene benchmark run behind `FINDINGS.md`'s headline
multi-scene result:

```
.venv-gsplat/bin/python gs_experiment/scripts/evaluate_checkpoint.py multi-scene \
  --scenes chair,drums,ficus,hotdog,lego,materials,mic,ship
```
