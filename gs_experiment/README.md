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
- `differentiation_experiment.py` — wires all of the above together and
  runs end-to-end on a mock scene right now (see below).

**Stubbed, needs GPU/real data:**
- `splat_scene.load_from_gsplat_checkpoint` raises `NotImplementedError`
  with the intended real implementation documented in its docstring: read
  a 3DGS `.ply` checkpoint (needs the optional `plyfile`/`torch`/`gsplat`
  dependencies in `requirements-gsplat.txt`, not installed in this
  environment), extract positions/scales/rotations/opacities/colors,
  and load training camera poses from `transforms.json` or COLMAP output.
- Everything currently operates on 3D world-space query points, not 2D
  image-plane pixels. Mapping world-space uncertainty to a specific
  camera's per-pixel image is a reprojection step that a live gsplat
  renderer's own projection/ray logic should provide directly — deferred
  rather than reimplemented here.

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
python -m pytest tests/test_gs_camera.py tests/test_gs_splat_scene.py tests/test_gs_pixel_uncertainty.py -v
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

## Once GPU access is available

1. `pip install -r requirements-gsplat.txt` (torch, gsplat, plyfile).
2. Implement `load_from_gsplat_checkpoint` per its docstring, against an
   actual trained scene and its `transforms.json`/COLMAP cameras.
3. Swap `make_mock_scene(...)` for `load_from_gsplat_checkpoint(...)` in
   `differentiation_experiment.py` — nothing else in the pipeline should
   need to change, since `SplatScene`/`splat_observations` are the
   interface boundary.
4. Construct a real differentiation scene: a region well-observed from a
   wide spread of training views but containing fine/thin structure
   (spatial under-resolution), and a region observed from a narrow range
   of viewpoints (directional under-resolution) — the real-data analogue
   of this scaffold's two mock zones.
5. Add the real visibility-field/Hessian-sensitivity comparison ROADMAP.md
   calls for (reproducing or citing PUP/GAVIS numbers) once there's a real
   scene to compute them on — `visibility_baseline.py`'s proxy is
   intentionally simple and not meant to stand in for that comparison.
