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
current-conclusions summary (real scenes first).

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
- **`colmap_loader.py`** — reads real COLMAP camera poses (the format
  real photographed datasets like Mip-NeRF360 ship), for scenes where
  poses are an SfM *estimate*, not exactly known.
- **`scene_spec.py`** — builds hand-built scenes for the designed-scene
  experiments below: `differentiation_scene` (two identical thin-rod
  clusters, one widely observed, one narrowly), `nbv_test_scene` (a
  candidate-view pool for active-view selection), and `gradient_scene`
  (five identical zones with camera-arc coverage width increasing
  linearly). `blender_render.py` renders them (needs `bpy`).
- **`train_minimal_gsplat.py`** — a minimal from-scratch `gsplat` trainer
  with real gradient-triggered densification (`train`), the reference
  strategy variant (`train_with_reference_strategy`), and an
  `--nll-experiment` mode: training directly under the BQ likelihood, as
  a loss term and as a densification trigger — a real negative result,
  kept in (`results/FINDINGS.md` §5).
- **`fit_hyperparameters.py`** — extends `bq_splat/hyperparams.py`'s
  marginal-likelihood bandwidth fitting from toy scenes to a real
  checkpoint's local windows, with a held-out check.

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
  the checkpoint's own splat extent, exposes kernel family (`--kernel-
  family rbf|matern`) and bandwidth as parameters, and refuses to compute
  anything on a checkpoint that fails a mandatory held-out-PSNR quality
  gate (`--min-psnr`, `--force` to override). This is the primary way
  this project validates a new result: render it and look, rather than
  reach for statistics first (see `ROADMAP.md`).
- **`evaluate_checkpoint.py {sparsity,calibration,kernel-ablation,
  window-ablation,visibility-trend,wide-vs-narrow,multi-scene}`** — the
  statistical checks behind `FINDINGS.md` §1-3, §6: does local
  splat density correlate with BQ variance (`sparsity`); is the variance
  calibrated via leave-one-out cross-validation, AUSE, and held-out NLL
  (`calibration`); RBF vs. Matérn at fitted bandwidths (`kernel-
  ablation`); sensitivity to the local-window-size hyperparameter
  (`window-ablation`); does variance respond to genuine angular coverage
  gaps vs. raw view count (`visibility-trend`); wide vs. narrow view-pool
  comparison (`wide-vs-narrow`); and the full 8-scene NeRF-Synthetic
  benchmark run via `--scenes` (`multi-scene`).
- **`real_directional_coverage_experiment.py --design {subsample,gap}
  --dataset {lego,bonsai}`** — the directional/viewing-angle-coverage
  question on real geometry (`FINDINGS.md` §4): `subsample` thins a real
  view pool into equal-count, increasing-spread conditions; `gap` removes
  a deliberate angular gap from an otherwise-dense real view pool
  (confound-free, the current best-trusted design); `--dataset` selects
  lego or the real photographed Mip-NeRF360 "bonsai" scene.
- **`designed_scene_experiments.py {differentiation,
  declustering-isolation,pruning,nbv,directional-gradient}`** — the
  hand-built-scene experiments (`FINDINGS.md` §7-8): `differentiation` is
  the original go/no-go test (can position-only BQ variance flag a region
  that's well-observed but poorly resolved); `declustering-isolation` is
  the controlled follow-up that refuted the leading hypothesis for its
  mechanism; `pruning` combines BQ variance with opacity-based pruning;
  `nbv` combines it with a visibility proxy for next-best-view candidate
  scoring; `directional-gradient` is the designed-scene version of the
  coverage-gradient test (its real-geometry counterpart is
  `real_directional_coverage_experiment.py` above).

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
.venv-gsplat/bin/python gs_experiment/evaluate_checkpoint.py sparsity <out_dir>/wide/splats.ply
```

or, for the full 8-scene benchmark run behind `FINDINGS.md`'s headline
multi-scene result:

```
.venv-gsplat/bin/python gs_experiment/evaluate_checkpoint.py multi-scene \
  --scenes chair,drums,ficus,hotdog,lego,materials,mic,ship
```
