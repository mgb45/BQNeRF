# gs_experiment — scaffold for the real GS differentiation experiment

ROADMAP.md milestone 2 ("the differentiation experiment — the real go/no-go
gate ... the first milestone that needs a GPU"). This package is everything
that can be built and tested *before* GPU/gsplat access is available: the
glue between `bq_splat`'s validated kernel/quadrature math and real 3D
Gaussian Splatting data structures, plus a mock scene that exercises the
whole pipeline end-to-end right now.

## What's real vs. mocked

**Real, tested, no dependency on GPU/torch/gsplat:**
- `camera.py` — camera pose representation and turntable pose generation
  (ported from `models/pose.py`'s spherical convention to plain numpy),
  plus computing the viewing direction from any 3D point to a camera.
- `splat_scene.py` — the `SplatScene` data structure, a `make_mock_scene`
  generator for testing, and `splat_observations` to expand a scene into
  the (position, direction, value) rows `bayesian_quadrature_directional`
  expects.
- `pixel_uncertainty.py` — `LocalUncertaintyEngine`: builds a KD-tree once
  over splat positions and caches the kernel's `vv` term per window size,
  directly applying the two exact optimizations validated in
  `benchmark_local_bq_scaling.py` (bq_splat/results/FINDINGS.md section 8)
  to real query geometry instead of synthetic timing data. Reuses
  `bayesian_quadrature_nd`/`bayesian_quadrature_directional` from
  `bq_splat` unmodified (that package gained an optional
  `precomputed_vv`/`precomputed_pos_vv` parameter to make the caching a
  first-class, reusable feature rather than something reimplemented here).
- `visibility_baseline.py` — a simple, deliberately non-Bayesian-quadrature
  visibility-coverage proxy (mean resultant length of local observation
  directions, standard circular statistics), standing in for a real
  visibility field (GAVIS-style) or Hessian sensitivity (PUP-style) per
  ROADMAP.md's "combination not competition" framing.
- `spherical_harmonics.py` — `eval_sh`, matching the standard 3DGS/gsplat
  SH color convention (the widely-reused Plenoxels/instant-ngp `eval_sh`
  utility) exactly, including its "+0.5" DC-offset convention. The
  hardcoded normalization constants (`SH_C0`, `SH_C1`, `SH_C2`) are checked
  against their closed-form analytic values (e.g. `SH_C0 == 1/sqrt(4*pi)`)
  in tests, not just trusted as literals. Real splats store view-dependent
  color this way, not as the flat scalar `colors` field used to; a splat's
  color now genuinely depends on viewing direction when `sh_coeffs` is set
  on its `SplatScene` (`splat_observations` uses `eval_sh` per observation
  when available, falling back to flat `colors` otherwise).
- `visibility_attribution.py` — frustum + occlusion attribution: which
  cameras plausibly saw each splat, from pure geometry (no rendering).
  `in_frustum` checks field-of-view and being in front of the camera;
  `occlusion_mask` is a soft z-buffer (project splats to camera-local
  angular bearing via a KD-tree, flag a splat as occluded if a
  meaningfully-closer splat shares its bearing). Real training pipelines
  don't record "which views actually constrained this splat," so this is a
  genuine proxy, validated against a synthetic occluder (one splat directly
  behind another, same bearing, gets flagged; a splat at the same depth but
  a different bearing doesn't) rather than assumed correct.
- `splat_scene.make_occluder_scene` — the integration test that SH color
  and real visibility attribution actually compose with the rest of the
  pipeline, not just that each works alone: a "wall" of splats occludes a
  cluster of "target" splats behind it from cameras in front, while cameras
  behind the targets see them directly — `observed_camera_idx` comes from
  `attribute_observations`, not an assignment rule, and colors come from
  random SH coefficients. Confirmed front cameras see none of the targets
  and back cameras see most of them.
- `differentiation_experiment.py` — wires the original (non-occluder)
  pipeline together and runs end-to-end on a mock scene right now (see
  below); doesn't yet use `make_occluder_scene`/SH colors, since that
  script's zone-based mock predates them — a reasonable next follow-up.

**Real, GPU-validated (see `gs_experiment/results/FINDINGS.md` for the full account):**
- `splat_scene.load_from_gsplat_checkpoint` is implemented: reads a
  standard 3DGS `.ply` (`ply_io.py`, schema matched against the reference
  implementation's property order) plus `transforms.json`
  (`nerf_transforms.py`), and derives `observed_camera_idx` via real
  frustum + occlusion attribution. `blender_render.py` (a Blender
  headless script, needs `bpy`) and `train_minimal_gsplat.py` (a minimal
  from-scratch gsplat trainer, needs `requirements-gsplat.txt`) produce
  the synthetic multi-view data and trained checkpoints this loader reads
  — together the first fully real (not mocked) path through this
  package, run on a 3090. `render_reconstruction.py` renders ground-truth
  vs. reconstruction comparisons for visually sanity-checking a trained
  checkpoint — worth running before trusting any uncertainty numbers off
  of it (`FINDINGS.md` §6 is a concrete case where skipping this step
  would have meant trusting a degenerate, blank reconstruction that still
  reported a deceptively reasonable PSNR).
- `kernel_comparison.py` runs the RBF-vs-Matérn question
  (`bq_splat/results/FINDINGS.md` §5-7, previously toy-scale only)
  against a real checkpoint.

**Still not attempted:**
- Everything currently operates on 3D world-space query points, not 2D
  image-plane pixels. Mapping world-space uncertainty to a specific
  camera's per-pixel image is a reprojection step that a live gsplat
  renderer's own projection/ray logic should provide directly — deferred
  rather than reimplemented here.
- `visibility_attribution.py`'s occlusion test is a bearing-based soft
  z-buffer, not what the actual training/rendering pipeline used to decide
  which views constrained which splat. Good enough to unblock building the
  rest of the pipeline; revisit if real-data results look sensitive to it.

**Deliberately not attempted (documented decision, see module docstrings):**
per-splat heterogeneous covariance as the BQ kernel bandwidth. Real splats
each have their own learned covariance; the validated `bq_splat` machinery
uses one shared or pooled-fit bandwidth instead (FINDINGS.md sections 5, 7).
Using each splat's own covariance as its own kernel bandwidth is a real,
mathematically plausible extension, but it's a second, unvalidated change —
stacking it on top of the GPU/gsplat integration at the same time would
make it hard to tell which change caused which result. `scales`/`rotations`
are carried as scene metadata for this reason, not consumed by the kernel.

## Running it now

```
python -m pytest tests/test_gs_camera.py tests/test_gs_splat_scene.py tests/test_gs_pixel_uncertainty.py tests/test_gs_visibility_attribution.py tests/test_spherical_harmonics.py -v
python gs_experiment/differentiation_experiment.py
```

The experiment script builds a mock scene (350 splats, two zones with
spatial density *not* explicitly controlled this time, unlike
`scripts/validate_directional_combined.py`'s exact-offset construction —
see below), computes all three signals over a 2D slice, and writes
`gs_experiment/results/differentiation_experiment_mock.png`.

**Result on the mock scene:** position-only variance ratio (narrow/wide
zone) = 0.57x, position+direction ratio = 4.08x, visibility-proxy ratio =
2.07x. The direction is right and the effect size is large, but 0.57x is
not the clean ~1.0 the toy experiment achieved (`validate_directional_
combined.py`, 0.97x) — that script explicitly gave both zones the
*identical* relative offset pattern from their own center to hold spatial
density exactly equal; this scaffold's mock scene instead scatters
splats uniformly at random everywhere, so the two zones' spatial density
isn't controlled the same way. Worth knowing before reading too much into
the exact ratios here: this script demonstrates the *pipeline* works
end-to-end, not a repeat of the controlled statistical claim already
established at toy scale. Tightening the mock scene to match the toy
experiment's exact-offset construction would be a small, worthwhile
follow-up if this script's numbers themselves are ever going to be quoted
rather than just used to confirm the code runs.

## GPU access obtained — status and what's left

Steps 1-4 of this section's original plan (install deps, implement the
loader, swap it into `differentiation_experiment.py`, construct a real
differentiation scene) are done — see `gs_experiment/results/FINDINGS.md`
for the complete account, including four real bugs found getting from
"the pipeline runs" to "the numbers are trustworthy": an occlusion-
attribution default far too aggressive for dense real geometry
(`--angular-tol`), a query-direction construction that broke when both
camera rigs share an elevation, an uncapped local-neighbor count that
briefly pegged the whole machine's CPU (`LocalUncertaintyEngine.
max_neighbors`), and a scale-initialization bug that produced a blank
reconstruction behind a deceptively reasonable PSNR (`train_minimal_
gsplat`'s `init_scale`).

Run it:
```
.venv-gsplat/bin/python -m gs_experiment.train_minimal_gsplat <scene_dir> <out.ply> --init-scale 0.1 --n-iters 8000
.venv-gsplat/bin/python gs_experiment/render_reconstruction.py <scene_dir>   # sanity-check before trusting anything below
.venv-gsplat/bin/python gs_experiment/differentiation_experiment.py --checkpoint <scene_dir> --angular-tol 0.01
.venv-gsplat/bin/python gs_experiment/kernel_comparison.py <scene_dir> --angular-tol 0.01
```

**What's not resolved:** the core go/no-go claim (position-only BQ
variance flagging a well-observed-but-poorly-resolved region) isn't
demonstrated yet — position-only variance comes back statistically equal
between the wide and narrow zones regardless of kernel, because
`train_minimal_gsplat` has no densification, so splat density near a
region doesn't depend on view coverage the way a real 3DGS trainer's
would. See `FINDINGS.md` §7 for the full reasoning and the two options
for actually closing this (real densification, or a scene where a
deliberately-undersized splat budget forces a resolution gap on its own).

Still open, unrelated to the above:
5. Add the real visibility-field/Hessian-sensitivity comparison ROADMAP.md
   calls for (reproducing or citing PUP/GAVIS numbers) — `visibility_
   baseline.py`'s proxy is intentionally simple and not meant to stand in
   for that comparison.
6. Real captured data (photographs + COLMAP or similar SfM pose
   estimation) rather than synthetic Blender renders with known ground-
   truth poses — not started.
