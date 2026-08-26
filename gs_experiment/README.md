# gs_experiment — Bayesian quadrature on real Gaussian Splatting

Real experiments against real trained `gsplat` checkpoints. This is where
`bq_splat`'s toy-scale validated math (kernels, quadrature, the
directional extension) gets applied to real 3D Gaussian Splatting scenes
— real cameras, real training, real densification. Needs a GPU and
`gsplat` for anything that trains or loads a checkpoint (see
[`../requirements-gsplat.txt`](../requirements-gsplat.txt) for setup,
including two real CUDA/compiler gotchas already solved there); a few
pure-geometry modules (`camera.py`, `scene_spec.py`, `visibility_attribution.py`,
`spherical_harmonics.py`) run on `numpy`/`scipy` alone and are covered by
the main test suite (`pytest tests/`).

**Results**: see [`results/FINDINGS.md`](results/FINDINGS.md) for the
current-conclusions summary (real scenes first), and
[`results/ARCHIVE_FULL_LOG.md`](results/ARCHIVE_FULL_LOG.md) for the
complete process log.

## Real-scene experiments (the main results)

These are the scripts behind `FINDINGS.md`'s headline claims — run
against real trained checkpoints from standard benchmarks, not hand-built
scenes.

- **`prepare_nerf_synthetic.py`** — downloads/prepares a standard
  NeRF-Synthetic scene (100 real training views + an official held-out
  test split) into this project's pipeline; also builds a "narrow"
  (angularly clustered) real-view subset and `select_gradient_subset`
  for graded-spread conditions.
- **`multi_scene_experiment.py`** — the full-benchmark run: downloads,
  trains, and evaluates the sparsity-correlation and calibration checks
  across all 8 standard NeRF-Synthetic scenes.
- **`sparsity_correlation_experiment.py`** — the direct "uncertainty
  nearly for free" check on one real checkpoint: does local splat density
  correlate with BQ position-only variance?
- **`calibration_experiment.py`** — leave-one-out cross-validation on
  real splat colors: is the variance *calibrated* (Pearson r, AUSE
  sparsification curves, held-out NLL vs. a flat baseline), not just
  correlated with sparsity?
- **`kernel_family_ablation.py`** — RBF vs. Matérn-3/2 on real
  checkpoints, at *fitted* (not arbitrary) bandwidths for each.
- **`window_radius_ablation.py`** — sensitivity of the sparsity-
  correlation claim to the local-window-size hyperparameter, swept
  across real checkpoints.
- **`visibility_trend_experiment.py`** — does BQ variance respond to
  genuine angular coverage gaps, or just raw view count? (checkpoints at
  matched view counts, varying only how clustered the views are.)
- **`real_benchmark_experiment.py`** — wide (100-view) vs. narrow
  (12-view, angularly clustered) cross-checkpoint comparison on lego.
- **`../scripts/fit_hyperparameters_real_checkpoint.py`** — extends
  `bq_splat/hyperparams.py`'s marginal-likelihood bandwidth fitting from
  toy scenes to a real checkpoint's local windows, with a held-out check.
- **`train_minimal_gsplat.py`** (`train_with_reference_strategy`) —
  validates results aren't an artifact of this project's own from-scratch
  trainer by training with `gsplat`'s own official reference
  densification strategy instead.
- **`nll_training_experiment.py`** — training directly under the BQ
  likelihood (as a loss term and as a densification trigger): a real
  negative result, kept in.
- **`colmap_loader.py`** — reads real COLMAP camera poses (the format
  real photographed datasets like Mip-NeRF360 ship), for scenes where
  poses are an SfM *estimate*, not exactly known.
- **`real_directional_gradient_experiment.py`** /
  **`real_capture_gradient_experiment.py`** — the directional/viewing-
  angle-coverage question on real geometry: subsampling a real dataset's
  fixed camera pool into equal-count, increasing-spread conditions, on
  lego and on a real photographed scene (Mip-NeRF360 "bonsai")
  respectively. `real_directional_gradient_experiment.py` also has the
  properly-resourced version (`--n-per-zone 30`, held-out PSNR checked
  before trusting any BQ number) that produced this project's current,
  most-trustworthy real-geometry result.
- **`render_directional_uncertainty_sweep.py`** — replaces a handful of
  point samples with a full per-pixel, per-frame animated sweep: real
  depth-unprojected 3D points, queried at the real direction to the
  *current* frame's camera as it orbits. Used to visually stress-test the
  real-geometry results above rather than trust five numbers alone.

## Foundational / designed-scene experiments

Real `gsplat` training throughout, but on hand-built (not standard-
benchmark) scenes — used to de-risk and cleanly isolate an effect before
testing whether it survives real, messier geometry.

- **`scene_spec.py`** — builds the hand-built scenes: `differentiation_scene`
  (two identical thin-rod clusters, one widely observed, one narrowly),
  `nbv_test_scene` (a candidate-view pool for active-view selection), and
  `gradient_scene` (five identical zones with camera-arc coverage width
  increasing linearly — a designed, continuous coverage gradient, not a
  binary split). `blender_render.py` renders them (needs `bpy`).
- **`differentiation_experiment.py`** — the original go/no-go test: can
  position-only BQ variance flag a region that's well-*observed* but
  poorly *resolved*, something a visibility-only signal structurally
  can't see? Demonstrated once real densification was added (see
  `FINDINGS.md`); the mechanism behind *why* remains genuinely open.
- **`pruning_experiment.py`** — combining BQ variance with opacity-based
  pruning.
- **`nbv_experiment.py`** — combining BQ variance with a visibility
  proxy for next-best-view candidate scoring.
- **`directional_gradient_experiment.py`** — the designed-scene version
  of the coverage-gradient test (`gradient_scene`): BQ directional
  variance recovers a real, continuous, designed gradient cleanly. Its
  real-geometry counterparts are `real_directional_gradient_experiment.py`
  and `real_capture_gradient_experiment.py` above.
- **`render_sweep_gif.py`** / **`render_uncertainty_views.py`** — earlier
  visualization scripts (position-only variance, per-splat and per-pixel
  respectively); superseded for the directional question by
  `render_directional_uncertainty_sweep.py` above, still useful for the
  spatial/sparsity signal.
- **`kernel_comparison.py`** — RBF vs. Matérn-3/2 on a real checkpoint
  (position-only variance, hardcoded bandwidths) — the real-data
  predecessor to `kernel_family_ablation.py`'s fitted-bandwidth version.
- **`validate_declustering_isolation.py`** — the controlled test that
  refuted the leading hypothesis for the differentiation experiment's
  mechanism (redundant clustering from densification) — see
  `FINDINGS.md`.

## Core library modules

- **`pixel_uncertainty.py`** — `LocalUncertaintyEngine`: the main
  entry point for querying BQ variance against a real checkpoint. Builds
  a KD-tree once, caches the kernel's `vv` term per window size (the two
  exact optimizations `bq_splat` validated for GS-scale cost), and caps
  local-neighbor count for tractability.
- **`splat_scene.py`** — `load_from_gsplat_checkpoint` (reads a real
  `.ply` + `transforms.json`), `splat_observations` (expands a scene into
  the (position, direction, value) rows the directional kernel needs).
- **`camera.py`** — camera pose representation, turntable pose
  generation, and per-splat viewing-direction geometry.
- **`visibility_attribution.py`** — frustum + soft-z-buffer occlusion
  proxy for "which cameras plausibly saw this splat" (real training
  pipelines don't record this).
- **`visibility_baseline.py`** — a simple, deliberately non-BQ visibility
  proxy (mean resultant length of observation directions), standing in
  for a dedicated visibility field in the combination experiments.
- **`spherical_harmonics.py`** — `eval_sh`, matching the standard
  3DGS/gsplat SH color convention.
- **`ply_io.py`** / **`nerf_transforms.py`** — the standard 3DGS `.ply`
  schema and NeRF-style `transforms.json` I/O, including the OpenCV/OpenGL
  convention conversions `colmap_loader.py` also uses.
- **`train_minimal_gsplat.py`** — a minimal from-scratch `gsplat` trainer
  with real gradient-triggered densification, plus the reference-strategy
  and training-under-the-likelihood variants used above.
- **`render_reconstruction.py`** — renders ground-truth vs. reconstruction
  comparisons; worth running before trusting any uncertainty number off a
  new checkpoint (a real past incident: a degenerate, blank reconstruction
  still reported a deceptively reasonable PSNR — see `FINDINGS.md`).

## Running it

```
pytest ../tests/test_gs_camera.py ../tests/test_gs_splat_scene.py \
  ../tests/test_gs_pixel_uncertainty.py ../tests/test_gs_visibility_attribution.py \
  ../tests/test_spherical_harmonics.py ../tests/test_gs_colmap_loader.py -v
```

runs everything that doesn't need a GPU. For a real experiment end to
end, with a `gsplat` environment set up (`../requirements-gsplat.txt`):

```
.venv-gsplat/bin/python gs_experiment/prepare_nerf_synthetic.py <raw_scene_dir> <out_dir>
.venv-gsplat/bin/python -m gs_experiment.train_minimal_gsplat <out_dir>/wide <out_dir>/wide/splats.ply --densify
.venv-gsplat/bin/python gs_experiment/render_reconstruction.py <out_dir>/wide   # sanity-check before trusting anything below
.venv-gsplat/bin/python gs_experiment/sparsity_correlation_experiment.py <out_dir>/wide/splats.ply
```

or, for the full 8-scene benchmark run behind `FINDINGS.md`'s headline
multi-scene result:

```
.venv-gsplat/bin/python gs_experiment/multi_scene_experiment.py chair drums ficus hotdog lego materials mic ship
```
