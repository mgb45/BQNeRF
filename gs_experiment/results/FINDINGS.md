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

(This run used the pre-densification checkpoint. §11 reruns the same
comparison on the post-densification checkpoint and finds the same
kernel-agreement pattern holds for the demonstrated go/no-go result.)

§7's diagnosis motivated adding real densification, below — §9-12 update
the status from "open" to "demonstrated."

## 9. Real densification, implemented and calibrated against measured gradients rather than a borrowed constant

`train_minimal_gsplat.densify_and_prune` (enabled via `densify=True`)
implements standard gradient-triggered clone/split plus opacity pruning
(Kerbl et al. 2023's mechanism), against gsplat's own view-space
positional gradient (`meta["means2d"].grad`, accumulated via
`meta["gaussian_ids"]` — the same signal the reference implementation
uses, retained via `retain_grad()` since it's a non-leaf autograd tensor).

First calibration attempt used the original 3DGS paper's threshold
(0.0002) and produced zero clones or splits over 900 iterations — only
pruning ever fired. Measuring the actual distribution (not assuming it
transfers): this project's gsplat gradient magnitudes run **~1e-6 to
~1e-5**, two to three orders of magnitude below the paper's value, which
is in a different normalized-image-space convention. Fixed by making the
threshold relative: each densify cycle uses the `densify_grad_percentile`
-th (default 80th) percentile of *that cycle's own* observed gradient
distribution, sidestepping the need to re-derive an absolute constant per
scene/loss/resolution combination. `max_splats` (default 30000, used
15000 for the differentiation scene) caps growth as a hard safety net,
given this project's one prior incident (§5) from unbounded growth left
unattended.

On the differentiation scene (starting from 2000 splats instead of the
previous fixed 8000, letting density grow on its own): count grew
steadily to the 15000 cap by iteration ~4500/10000, and reconstruction
quality rose sharply as a result — mean PSNR over 4 held-out-angle views
**33.0dB → 43.3dB**, with `reconstruction_diff_out.png` showing crisply
resolved rods rather than the softer edges of the fixed-count run.

## 10. A clone-adjacency numerical concern, checked and found not to be the driver

3DGS clones start a duplicate at the parent's exact position (the
reference implementation nudges it along the triggering gradient
direction to speed separation; this implementation initially didn't).
Measured consequence: in the wide zone, median nearest-neighbor distance
among splats was 0.009 (RBF bandwidth sigma=0.9 — three orders of
magnitude larger), with 217 pairs still under 0.001 apart after
thousands of post-clone iterations, and the local Gram matrix's condition
number was 1.6e30 unjittered, 2.2e6 after the existing `rel_jitter=1e-4`
fix (`bq_splat/results/FINDINGS.md` §4). This mattered enough to check
directly before trusting anything downstream — an ill-conditioned solve
producing an inflated "variance" would be numerical noise, not signal,
exactly the kind of false positive §2 of the bq_splat findings already
warns about.

Two things resolved this: (a) at 2.2e6, the *jittered* solve's relative
error is bounded by roughly `condition_number * machine_epsilon` ≈
2.2e6 * 2.2e-16 ≈ 5e-10 — negligible, nowhere near the ~1e18 regime that
made the original bug real; (b) `densify_and_prune` was changed anyway to
offset clones by a small (0.5x parent scale) random jitter rather than
leaving them exactly coincident, on the reasoning that it's the more
correct thing to do regardless. Retraining with the fix changed the
near-duplicate statistics only marginally (median NN distance 0.0090 →
0.0090, same order) — most of the tight clustering turned out to reflect
genuine convergence around thin geometry (many splats legitimately
needed to tile a 0.04-radius rod's surface), not an artifact of unoffset
cloning. The downstream result (§11) is essentially unchanged with or
without the offset, which is itself evidence it wasn't a clone-adjacency
artifact driving it.

## 11. The core go/no-go claim: demonstrated, and replicated across two seeds and two kernels

With real densification, position-only BQ variance **does** differentiate
between the wide and narrow zones — in the direction the claim needs, not
the direction visibility ranks them:

| run | position-only ratio (narrow/wide) | wide | narrow |
|---|---|---|---|
| seed=0, before clone-offset fix | 0.32x | 0.828 | 0.264 |
| seed=0, after clone-offset fix | 0.33x | 0.770 | 0.252 |
| seed=1 (independent retrain) | 0.44x | 0.837 | 0.369 |
| RBF, final checkpoint | 0.33x | 0.777 | 0.254 |
| Matérn-3/2, final checkpoint | 0.46x | 21.22 | 9.79 |
| RBF, deduplicated positions (§13) | 0.37x | 0.794 | 0.295 |
| Matérn-3/2, deduplicated positions (§13) | 0.55x | 21.39 | 11.68 |

Every configuration agrees: **wide-zone position-only variance is higher
than narrow-zone position-only variance** (ratio consistently well below
1x) — meanwhile the visibility proxy consistently ranks the *opposite*
direction (wide=0.58, narrow=0.99 — wide is the *better*-observed,
*lower*-uncertainty zone by visibility). This is exactly ROADMAP.md's
differentiation claim: a region a visibility-only signal calls
well-observed, that position-only BQ variance flags as more uncertain
than a region visibility calls poorly-observed. Position+direction
variance and the visibility proxy both still agree with each other on the
narrow-vs-wide *directional*-coverage story (ratios 5.6x-5.9x and
1.7x, consistent with §4's original result) — it's specifically the
position-only signal that diverges from visibility, which is the
mechanism, not a side effect of the directional kernel also being active.
Two independent training seeds and two kernel families (RBF, Matérn-3/2,
correlation 0.995 between their variance grids) all agree on direction,
which is enough replication to treat this as a real, not seed-specific,
effect — see `differentiation_experiment_real.png` and
`kernel_comparison_real.png`.

## 12. Leading hypothesis for the mechanism (not yet fully settled)

Raw splat count near each zone is similar (wide 7378, narrow 6020 — only
1.2x apart), but *high-opacity* (>0.1) splat count differs sharply (5355
vs. 1013 — 5.3x), and the wide zone's splats are measurably more spatially
clustered: median nearest-neighbor distance 0.009 (wide) vs. 0.013
(narrow), and nearly double the fraction of pairs closer than 0.005 apart
(19.6% vs. 10.4%). The most likely explanation ties these together: the
wide zone's stronger, more *consistent* (across 40 views) photometric
gradient triggered more densification cycles, and this trainer's
clone/split placement puts children near their parent by construction —
so more aggressive densification produces a *more spatially redundant*
splat population, not just a denser one. Position-only BQ variance
(`vv - v^T K^-1 v`) generally drops as data accumulates, but redundant,
near-duplicate observations contribute less marginal reduction per point
than well-separated ones would; if the wide zone's extra splats are
disproportionately redundant rather than independently informative, a
smaller, better-separated narrow-zone population could plausibly leave a
*lower* reported variance despite representing genuinely sparser real
coverage.

This is offered as the leading hypothesis, not a settled mechanism — it's
consistent with every measurement taken so far (opacity-weighted density,
clustering statistics, robustness to the clone-offset fix in §10) but
hasn't been isolated with a controlled test (e.g. deliberately equalizing
clustering between zones while varying only view count, the same "hold
one variable exactly equal" discipline `validate_directional_combined.py`
used for the directional result in `bq_splat/results/FINDINGS.md` §9).
Worth flagging for anyone citing §11's result: it may be closer to "BQ
variance is sensitive to how efficiently a finite splat budget tiles the
domain" than to "BQ variance directly measures view-coverage-driven
resolution" — a genuine, interesting distinction, and arguably still
consistent with ROADMAP.md's original framing of BQ uncertainty as
residual quadrature/discretization error (a redundant cluster of nodes is,
in a real sense, inefficiently tiling the domain), but not identical to
the naive reading of the claim either.

## 13. A second numerical concern, checked and also found not to be the driver: camera-count leaking into "position-only" via row duplication

`splat_observations` expands one row per (splat, camera) observation --
correct input for the directional kernel, but a real conceptual problem
for position-only variance if reused unchanged: "position-only, blind to
direction" should mean the signal doesn't depend on how many cameras saw
a splat, yet feeding camera-duplicated rows into `spatial_only_variance`
means it implicitly does. This turned out to matter a lot in absolute
terms: wide-zone splats were attributed to a mean of **30.1 cameras
each** (median 31, out of 40 possible) vs. the narrow zone's **1.5**
(median 1, out of 10) -- a ~20x difference in row-duplication factor
between the two zones being compared.

Two things were checked before concluding this invalidated §11's result.
First, whether `LocalUncertaintyEngine`'s `max_neighbors=400` random
subsampling (capping a query's candidate pool before the BQ solve) ends
up representing far fewer *distinct* splats in the heavily-duplicated
wide zone than the lightly-duplicated narrow zone, which would bias the
comparison directly: it does not, in practice -- 389/400 and 384/400
sampled rows were distinct positions in the wide and narrow zones
respectively (expected under random sampling without replacement when
the candidate pool, thousands of distinct splats each duplicated many
times, is much larger than the 400-sample: collisions stay rare by a
birthday-paradox argument, confirmed here rather than assumed). Second,
and more directly: recomputing position-only variance from scratch on
strictly deduplicated splat positions (`scene.positions`/`scene.colors`,
one row per splat, no camera expansion at all, bypassing
`splat_observations` entirely for this signal) reproduced the same
result closely -- wide=0.794, narrow=0.295, ratio 0.37x, squarely inside
the 0.32x-0.46x range from every other configuration in §11's table, and
the RBF/Matérn agreement (correlation 0.995) held too.

Both `differentiation_experiment.py` and `kernel_comparison.py` (and
`render_uncertainty_views.py`) were changed to use two separate
`LocalUncertaintyEngine` instances regardless -- one built on deduplicated
positions for `spatial_only_variance`, one on `splat_observations`'
camera-expanded rows for `directional_variance` -- since the deduplicated
version is the conceptually correct one even though it didn't change
§11's conclusion here. Worth remembering if this pipeline is ever run on
a scene with a more extreme camera-count imbalance than 40-vs-10, where
the effect (currently confirmed negligible) might not stay negligible.

## Bottom line for the go/no-go gate

The real-checkpoint pipeline (loader, trainer with real densification,
both kernels, visibility attribution) is validated end-to-end on
GPU-trained data. The directional/visibility comparison replicates the
toy-scale finding and more strongly (5.6-18.7x here vs. 2.46x at toy scale
in `bq_splat/results/FINDINGS.md` §9). Kernel choice doesn't change any
conclusion, only absolute scale (§8, §11). And the actual gate — does
position-only BQ variance catch a failure mode visibility-based methods
miss — is now **demonstrated**, replicated across two independent training
seeds, two kernel families, and two different position arrays (camera-
expanded and strictly deduplicated, §13): position-only variance ranks the
wide and narrow zones in the *opposite* order from the visibility proxy in
every configuration tried. Two distinct numerical/methodological concerns
that could each plausibly have explained the result away as an artifact
(clone-position adjacency, §10; camera-count leaking into a signal meant
to be blind to it, §13) were checked directly rather than assumed benign,
and neither was the driver. The mechanism actually driving it (§12) is a
well-supported leading hypothesis, not yet
a fully isolated causal test — the natural next step before treating this
as paper-ready is a controlled experiment that varies view count while
holding splat clustering/redundancy fixed (or vice versa), the same
one-variable-at-a-time discipline that made the toy-scale directional
result (`bq_splat/results/FINDINGS.md` §9) solid. With that caveat,
proceeding to ROADMAP.md's milestones 3-4 (the densification/NBV
combination experiments) is now reasonable — the premise they depend on
has real, replicated (if not yet fully mechanistically isolated) support,
rather than being unverified.
