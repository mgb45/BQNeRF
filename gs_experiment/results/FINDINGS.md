# Milestone 2 findings: the real differentiation experiment on a trained gsplat checkpoint

ROADMAP.md milestone 2 ("the differentiation experiment — the real go/no-go
gate ... the first milestone that needs a GPU"). This is the first run of
that experiment against an actual trained gsplat checkpoint rather than
`make_mock_scene`'s fiat camera-index assignment — on a 3090, via a new
`.venv-gsplat` environment (`requirements-gsplat.txt`), a Blender-based
synthetic-scene renderer (`blender_render.py`), a minimal from-scratch
gsplat trainer (`train_minimal_gsplat.py`), and the previously-stubbed
`splat_scene.load_from_gsplat_checkpoint`, now implemented for real.

Four real bugs turned up getting from "the pipeline runs" to "the numbers
are trustworthy" — recorded below in the order they were found, since each
one's fix changed the next result.

## 1. Environment: gsplat's CUDA kernels JIT-compile, and CUDA 12.3's nvcc rejects Ubuntu 24.04's default gcc

`pip install gsplat` installs a pure-Python wrapper; its CUDA kernels
compile on first import via `torch.utils.cpp_extension`, which needs nvcc
plus a host C++ compiler nvcc accepts. CUDA 12.3 rejects gcc/g++ newer than
12, but Ubuntu 24.04 ships gcc 13 as the default `gcc`/`g++`. Fix: nvcc
reads the `NVCC_APPEND_FLAGS` environment variable automatically (a real
CUDA-11+ nvcc feature), so `NVCC_APPEND_FLAGS="-ccbin /usr/bin/g++-11"`
(g++-11 already present system-wide) works with no changes to system
compiler alternatives. Full setup recorded in `requirements-gsplat.txt`.
Blender's own Cycles CUDA kernels hit the identical error for the same
reason; not worth solving twice for a scene this small, so
`blender_render.py` renders on CPU instead (fast enough at dozens of
400x400 images) and spends the GPU on gsplat training, where it matters.

## 2. Real checkpoint loading validated end-to-end

`load_from_gsplat_checkpoint` reads a standard 3DGS `.ply` (position,
log-scale, opacity-logit, channel-major SH coefficients — schema in
`ply_io.py`, matched against the reference implementation's property
order, not just internally consistent) plus `transforms.json`
(`nerf_transforms.py`), and derives `observed_camera_idx` via real frustum
+ occlusion attribution (`visibility_attribution.attribute_observations`),
the same mechanism `make_occluder_scene` already validated — a real
checkpoint has no recorded "which views actually constrained this splat,"
so this is a geometric proxy, not ground truth. A synthetic round-trip
test (`tests/test_gs_splat_scene.py`) and a first real trained checkpoint
(`quick_validation_scene`, 4 simple objects, 24 turntable views, 4000
splats, ~29-31dB PSNR) both load correctly: positions/opacities/scales/SH
match to float precision, and `splat_observations` produces genuinely
view-dependent color from the loaded SH coefficients.

## 3. Bug: the default occlusion `angular_tol` over-triggers by two orders of magnitude on dense real geometry

`visibility_attribution.py`'s soft z-buffer occlusion test defaults to
`angular_tol=0.05`, validated only against `make_occluder_scene`'s single
isolated occluder/target pair. Against the real differentiation scene (14
closely-packed thin rods per cluster), it flagged effectively everything as
occluded:

| `angular_tol` | (splat, camera) rows near wide-zone center | near narrow-zone center |
|---|---|---|
| 0.05 (old default) | 49 | 3 |
| 0.02 | 3,475 | 978 |
| 0.01 | 11,018 | 2,684 |
| 0.005 | 15,736 | 3,619 |

At the default, 99.95%+ of otherwise-valid observations were discarded —
tight object packing triggering the bearing-based occlusion test far more
than intended, not a bug in the test's logic itself. Fixed by exposing
`attribution_angular_tol` as a parameter (`load_from_gsplat_checkpoint`,
threaded through `differentiation_experiment.py --angular-tol`, default
0.01 for this scene). The wide/narrow row-count ratio stays roughly
constant (~4x) across tolerance values, consistent with the real 40-vs-10
camera-count difference rather than being an artifact of the threshold
itself.

## 4. Bug: negating a direction vector for a "discriminating query direction" silently breaks when both camera rigs share an elevation

The mock-scene experiment picks a query direction by negating a direction
actually observed inside the narrow zone's camera cone — reasoning "this
points somewhere the narrow arc doesn't cover." That only works when
negation is equivalent to rotating azimuth by 180 degrees. It isn't in
general: negating a 3D unit vector flips elevation too. Both camera rigs in
`scene_spec.differentiation_scene` share the same `phi_deg=35` elevation
(by construction — same turntable convention, different azimuthal ranges),
so the negated vector landed on an elevation band *neither* rig's cameras
actually observe. Symptom: position+direction BQ variance came back nearly
identical between wide and narrow zones (~11.6 both, ratio 1.00x-1.01x)
regardless of how much real observation data was available (§3's fix
changed row counts by 20x with no effect on this ratio) — a flat,
uninformative result masquerading as "no differentiation," rather than
what it actually was, a query direction that matched *nothing* in either
zone.

Fixed in `differentiation_experiment._build_real_scene` by searching every
camera's direction-as-seen-from-the-wide-cluster for the one *least*
similar (by dot product) to any camera's direction-as-seen-from-the-
narrow-cluster — robust to whatever elevation/azimuth convention the rigs
actually use, rather than assuming negation is equivalent to a 180-degree
azimuth rotation. With both this fix and §3's, `differentiation_experiment_
real.png`:

- Position+direction BQ variance: wide=0.62, narrow=11.67 — **ratio 18.7x**
  (stronger than the toy-scale result in `bq_splat/results/FINDINGS.md`
  §9, which found 2.46x under a more tightly controlled construction).
- Visibility proxy (non-BQ): wide=0.57, narrow=0.98 — **ratio 1.7x**,
  agreeing in direction.

## 5. Engineering incident: an uncapped local neighbor count pegged 18 CPU cores for half an hour

§3's fix increased local-window observation density by up to 20x. Bayesian
quadrature's per-query solve cost grows at least quadratically in neighbor
count, and `LocalUncertaintyEngine` had no cap — one run at
`angular_tol=0.01` ran for ~29 minutes at ~1800% CPU (18 cores) before
being killed, and the resulting system-wide load was disruptive enough to
visibly affect an unrelated desktop session sharing the machine. Root
cause: `benchmark_local_bq_scaling.py`
(`bq_splat/results/FINDINGS.md` §8) validated the solve as cheap up to
"hundreds" of local neighbors — never thousands. Fixed by adding
`max_neighbors` (default 400) to `LocalUncertaintyEngine`, randomly
subsampling down to that cap when a query's window contains more —
restoring the actually-validated regime rather than letting window
contents grow unbounded with real-data density. All results in this
document use this cap; re-running with `max_neighbors=None` on this scene
is not recommended without a hard wall-clock timeout.

## 6. Bug: bounding-box-derived scale initialization produced a degenerate reconstruction with a deceptively reasonable PSNR

`train_minimal_gsplat.init_splats` originally set every splat's initial
scale to `scene_extent / n_splats^(1/3)` — a heuristic that assumes
objects fill a roughly constant fraction of the scene's bounding volume.
For the differentiation scene (two small rod clusters inside a 22-unit-long
bounding box), this produced an initial scale of ~1.1 — about 25x the
actual 0.04-unit rod radius. 6000 iterations of single-random-view SGD
never shrank it enough: the trained checkpoint's high-opacity splats
(opacity > 0.5: 45 of 8000) had a *median scale of 1.1*, essentially
unchanged from initialization. The reconstruction was visually blank
(background-only) at every rendered view, yet reported ~21dB PSNR — background
occupies most of a 400x400 frame when the actual foreground is a handful
of thin rods, so "predict background everywhere" scores deceptively well
on a whole-image PSNR metric. This was caught by actually rendering a
reconstruction and looking at it (`render_reconstruction.py`), not by the
training-loop PSNR log — worth remembering before trusting an aggregate
metric on a scene with a small foreground fraction.

Fixed by adding an explicit `init_scale` parameter (set to 0.1 for this
scene, vs. the rod's 0.04 radius) so scale starts near the right order of
magnitude rather than needing gradient descent to correct a 25x error.
Retrained with `init_scale=0.1`, `opacity_reg_weight=0.003` (down from
0.01, since the tighter initial scale meant less needed to be pruned) and
8000 iterations: PSNR rose to ~33dB, and `reconstruction_diff_out.png`
shows recognizably reconstructed rods (soft-edged, as expected from a
trainer with no densification, but genuinely there) rather than a blank
frame.

Notably, **this fix left every uncertainty number in §4 numerically
unchanged** (0.99x/18.71x/1.72x before vs. 0.99x/18.71x/1.72x after,
computed independently and matching to within re-run noise). This is
expected, not a coincidence worth chasing: `splat_observations` builds BQ's
input purely from splat position and SH color, never opacity or scale
(documented in `splat_scene.py` as a deliberate scope limit). The scale bug
made the *visual* reconstruction degenerate without touching what BQ
actually consumes — a useful case study in why "the reconstruction looks
right" and "the uncertainty numbers are meaningful" are separate claims
that both need checking.

## 7. The core go/no-go claim: still open, for a specific, now well-understood reason

ROADMAP.md's actual differentiation claim is about *position-only*
variance: does it flag a region that's well-observed (many views) but
poorly resolved (fine/thin geometry), a failure mode a visibility-only
proxy structurally can't see? Across every configuration tried — the
broken checkpoint and the fixed one (§6), RBF and Matérn (§8) — position-
only variance came back statistically indistinguishable between the wide
(40-view) and narrow (10-view) zones (ratio 0.98x-1.02x). Both zones use
identical rod geometry, so this isn't necessarily wrong; the question is
whether it's evidence against the claim or evidence the experiment hasn't
actually stressed it yet.

The latter, most likely: `train_minimal_gsplat` has no densification. A
real 3DGS trainer clones/splits splats where photometric gradient is
strong and *consistent across many views* — exactly the mechanism that
would make splat *density* near a region depend on how well-observed it
is. Without it, splat count near a cluster is set once at random
initialization and never grows with view coverage, so position-only BQ
variance (which only sees where splats' positions/colors ended up, not how
they got there) has no reason to differ between a well- and poorly-
observed-but-equally-dense region. Closing this gate for real needs either
(a) real densification in the trainer, so density becomes view-coverage-
dependent, or (b) a scene redesigned so pure geometric fineness (not
densification) forces a resolution gap regardless of trainer sophistication
— e.g. a fixed, deliberately-undersized splat budget relative to the
finest structure present. Neither has been attempted yet.

## 8. Kernel comparison (RBF vs. Matérn-3/2) on real data: same conclusion, very different scale

`kernel_comparison.py` reruns the same real-scene position-only variance
computation with both kernels, everything else (scene, cameras, zones,
query direction, neighbor cap) held fixed — the first time this project's
RBF-vs-Matérn question (open since `bq_splat/results/FINDINGS.md` §5-7,
validated only at toy scale) has been run against a real trained GS
checkpoint:

- RBF: wide=0.0094, narrow=0.0097, ratio **1.02x**
- Matérn-3/2: wide=1.406, narrow=1.382, ratio **0.98x**
- Pearson correlation between the two kernels' variance grids: **0.984**

The two kernels agree almost perfectly on spatial *pattern* (correlation
0.98) but disagree by roughly 150x on absolute *scale* — expected, since
Matérn-3/2's rougher smoothness assumption shrinks posterior variance less
per nearby observation than RBF's infinitely-smooth prior does, for the
same nominal bandwidth. At this scale and for this scene, kernel choice
looks like a rescaling, not a source of different qualitative conclusions
— though this is one scene, one bandwidth value, not a swept comparison,
so "no qualitative difference" shouldn't be over-read as a settled result
the way §5-7's toy-scale, many-trial comparisons can be.

## Bottom line for the go/no-go gate

The real-checkpoint pipeline (loader, trainer, both kernels, visibility
attribution) is validated end-to-end on GPU-trained data, not mock data —
that part of milestone 2's scope is done. The directional/visibility
comparison replicates the toy-scale finding and more strongly (18.7x here
vs. 2.46x at toy scale in `bq_splat/results/FINDINGS.md` §9). Kernel choice
doesn't change that conclusion, only its scale (§8). But the actual gate —
does position-only BQ variance catch a failure mode visibility-based
methods miss — is not yet demonstrated, and now for a specific,
mechanistic reason rather than an open mystery: this trainer's lack of
densification means splat density isn't view-coverage-dependent, which is
exactly the property the claim needs. Proceeding to ROADMAP.md's milestones
3-4 (the densification/NBV combination experiments) before resolving this
would mean building on an unverified premise; per ROADMAP.md's own
verification gates, this is still the thing to resolve first.
