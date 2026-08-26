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

## 12. Leading hypothesis for the mechanism (later tested and refuted — see §14)

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

**Update (§14): this hypothesis was tested directly and refuted.**
Equalizing splat count and equalizing spacing between the zones each left
the effect essentially unchanged — redundant clustering was a reasonable,
testable hypothesis, and testing it (rather than treating it as settled)
was the right call, but it wasn't the mechanism.

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

## 14. The controlled isolation test §12 called for: redundancy refuted, effect survives, mechanism now genuinely open

`validate_declustering_isolation.py` runs the actual controlled
experiment §12 flagged as missing, post-hoc on the existing checkpoint
(no retraining): three conditions, same two query points (zone centers),
same kernel/window/neighbor-cap throughout, only the *wide* zone's splat
population changes between them.

| condition | n (wide) | wide variance | narrow variance | ratio (narrow/wide) |
|---|---|---|---|---|
| (1) original wide | 7378 | 0.693 | 0.260 | 0.375x |
| (2) wide random-subsampled to narrow's count (6020) | 6020 | 0.670 | 0.294 | 0.439x |
| (3) wide greedily declustered to match narrow's median spacing (0.0132) | 3861 | 0.706 | 0.251 | 0.356x |

Condition (2) isolates raw count (spacing/redundancy pattern untouched);
condition (3) isolates spacing (greedy minimum-distance rejection
subsampling, Poisson-disk style, until the wide zone's own median
nearest-neighbor distance matches the narrow zone's). **Neither moved the
ratio meaningfully** — 0.36x-0.44x across all three, the same range as
every configuration in §11's table. §12's redundant-clustering hypothesis
predicted condition (3) specifically should move the ratio toward 1x; it
didn't. The hypothesis is refuted, cleanly, by the test it called for
existing.

This is a good outcome for the result even though the explanation was
wrong: the effect just survived a direct, well-motivated attempt to
explain it away as a trainer artifact (redundant points from
densification), on top of already surviving the clone-adjacency (§10) and
camera-duplication (§13) checks. Three independent "maybe this is just an
artifact" hypotheses tested, three refuted, the effect unchanged every
time — stronger evidence it's measuring something real about the two
zones than any of the confirmatory checks in §11 alone would give.

A supplementary observation, offered as a new candidate hypothesis rather
than a tested one: the two zones' local point clouds differ in *shape*
now that spacing/count are ruled out — the wide zone's local
neighborhood is measurably more anisotropic (covariance eigenvalue ratio
2.10, elongated along z, plausibly the rods' own axis) than the narrow
zone's (1.43, closer to isotropic), which is the opposite of the naive
"narrow-baseline reconstruction smears points along the viewing axis"
intuition. Both zones' bounding-box extents (~2.4-3.2 units) are larger
than the true rod cluster's own footprint (spread parameter 0.8, rod
length 0.5), meaning both windows contain some splats that aren't tightly
bound to real geometry -- but not necessarily in equal proportion, and a
higher fraction of loosely-bound, off-rod splats in one zone's window
would move BQ variance without changing median spacing or count. This is
speculative and untested -- flagged as the next thing to check (e.g.
opacity-weighted or on-rod-vs-off-rod partitioned variants of the same
declustering-isolation methodology used here), not a replacement
explanation being asserted with the same confidence §12's was before
testing it.

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
and neither was the driver. §14 then ran the actual controlled isolation
test §12 called for (matching splat count, then matching spacing, between
the two zones) and **refuted §12's redundant-clustering hypothesis
directly** — the effect didn't move. That makes four independent "maybe
this is an artifact" checks the result has survived (clone-adjacency,
camera-duplication, count-matching, spacing-matching), which is
considerably stronger standing than "one well-supported hypothesis,
untested" was. The cost is that the *mechanism* is now genuinely open
rather than pending confirmation of a specific leading candidate — §14's
anisotropy/off-rod-fraction observation is a new lead, not yet tested the
way the redundancy hypothesis was. Practically: the effect itself is
about as well-verified as post-hoc analysis on one scene can make it, and
is safe to build on; the mechanism needs another isolation test (the
opacity-weighted or on-rod-fraction variant §14 suggests) before anyone
should claim to *understand why*, as opposed to *that*, position-only BQ
variance diverges from visibility here. Proceeding to ROADMAP.md's
milestones 3-4 (the densification/NBV combination experiments) is
reasonable on that basis — the premise they depend on (the effect exists
and replicates) has real, thoroughly-checked support, even with the
mechanism still open.

# Milestone 3 findings: densification/pruning combination experiment

ROADMAP.md milestone 3: "Use BQ variance alongside a visibility/
gradient-based criterion; check whether the combination reaches equal
quality at fewer splats... that the heuristic-only baseline misses." This
tests that directly, post-hoc on the already-densified, already-trained
differentiation-scene checkpoint (`pruning_experiment.py`, no retraining
needed): prune to a target splat count two ways — opacity-only (the
standard 3DGS heuristic) vs. opacity combined with BQ position-only
variance (read from `differentiation_experiment.py`'s cached 2D
uncertainty grid via interpolation, orders of magnitude cheaper than
re-running fresh BQ solves for up to 15000 splats and precise enough at
this splat-count-level granularity) — and compare reconstruction PSNR at
matched, reduced splat counts.

## 15. First attempt: a real win at tight budgets, a real regression at loose ones

Unweighted combination (opacity rank + BQ-variance rank, no floor):

| keep count (of 15000) | opacity-only PSNR | BQ-combined PSNR | delta |
|---|---|---|---|
| 4000 | 15.52dB | 17.98dB | **+2.45dB** |
| 6000 | 33.66dB | 21.50dB | **-12.17dB** |
| 9000 | 38.73dB | 28.01dB | **-10.72dB** |

Not a simple "BQ helps" or "BQ hurts" result — genuinely mixed, and worth
understanding why rather than picking the favorable number. At
keep_count=6000/9000, the splats the BQ-combined criterion *saved* from
pruning (that opacity-only would have kept out) had mean opacity 0.07 and
0.02 respectively — barely-visible splats. The reason: BQ position-only
variance is high in genuinely *empty* space too (little/no local data,
correctly reflecting real uncertainty about that region, but not usefully
for a pruning decision), so an unweighted combination spends keep-budget
protecting near-transparent splats sitting in the gap between the two rod
clusters, at the direct expense of dropping real, higher-opacity content
elsewhere. At the tightest budget (4000), the BQ-saved splats had mean
opacity 0.605 — genuinely substantial content that opacity-only's cutoff
happened to just miss — which is why that condition won.

## 16. Fix: floor the BQ term at a minimum opacity, calibrated empirically

Restricting the BQ-variance boost to splats already above a minimum
opacity (`min_opacity_for_bq`) keeps BQ voting among plausible
reconstruction content rather than among obvious empty-space junk.
Swept two floors:

| floor | keep=4000 delta | keep=6000 delta | keep=9000 delta |
|---|---|---|---|
| none (§15) | +2.45dB | -12.17dB | -10.72dB |
| 0.05 | +2.45dB | -10.89dB | 0.00dB (no-op) |
| 0.3 | +2.32dB | 0.00dB (no-op) | 0.00dB (inferred no-op — a stricter floor can only shrink an already-empty affected set) |

`min_opacity_for_bq=0.3` gives the cleanest behavior: a real win
(+2.3dB) at the tightest budget, where the tradeoff BQ is making
actually matters, and a strict no-op (never worse than the baseline) at
looser budgets, where opacity-only alone already retains everything with
non-trivial opacity and there's nothing left for BQ to usefully
re-prioritize. This is the "combination not competition" story ROADMAP.md
asks for in its cleanest form: BQ variance doesn't need to *always* help
to be worth including — it needs to never meaningfully hurt, and to help
specifically in the regime (tight splat budgets) where getting the
allocation right matters most. Set as the script's default.

Caveats worth being explicit about: (a) this is one scene, one checkpoint,
one keep-count sweep, not a systematic study across budgets/scenes;
(b) the floor value (0.3) was picked by trying two values and observing
which behaved best, not derived from first principles or swept
exhaustively — a more thorough calibration (e.g. a finer sweep, or a
floor set relative to the checkpoint's own opacity distribution rather
than an absolute constant) is a reasonable follow-up before citing the
exact number; (c) PSNR is averaged over 8 held-out views spanning both
zones, not a full held-out test set.

# Milestone 4 findings: active-view / NBV combination experiment

ROADMAP.md milestone 4: "Use BQ variance alongside a visibility proxy for
candidate-view scoring; check whether the combined signal selects views
that improve reconstruction in under-resolved regions faster than either
signal alone." `nbv_experiment.py` + `scene_spec.nbv_test_scene` build a
self-contained testbed: one thin-rod cluster (the same construction as
`differentiation_scene`'s zones, via the now-factored-out
`thin_rod_cluster`) observed from a narrow 10-view training arc, with a
discrete pool of 15 candidate next-view poses (a turntable ring at the
training radius, excluding angles already covered by the arc) and a
disjoint, never-a-candidate 16-view held-out evaluation ring.

## 17. Scoring is genuinely free — no retraining needed to rank candidates

Every candidate is scored two ways using only the *baseline* checkpoint
(trained on the 10-view arc alone), no retraining per candidate: BQ
position+direction variance at the cluster center, queried at the
candidate's own viewing direction (high variance = under-covered
direction), and a visibility-proxy score (how much adding the candidate's
direction would reduce the mean resultant length of the already-observed
direction set — bigger reduction = more angular-diversity gain, using
`visibility_baseline.resultant_length`, a genuinely different mechanism
from BQ). Both are closed-form given the baseline checkpoint's real splat
positions and observed directions — exercising, for the first time on
real data, the "essentially free to compute" property ROADMAP.md's
engineering plan cites as BQ's advantage for this kind of candidate
evaluation loop.

## 18. The combined signal picks a genuinely better next-view than a poor one

| candidate | θ (deg) | combined score | held-out PSNR after adding | delta vs. baseline |
|---|---|---|---|---|
| baseline (arc alone) | — | — | 21.02dB | — |
| best (top combined) | 22.5 | 2.000 | 22.91dB | **+1.89dB** |
| worst (bottom combined) | 180.0 | 0.000 | 21.68dB | +0.65dB |

The training arc is centered at θ=200° (±12°); the best-scored candidate
(22.5°) is close to angularly opposite it, the worst-scored (180.0°) is
the candidate nearest the arc's own coverage among those available. Both
additions help some (adding any 11th view to a 10-view arc should), but
the guided pick helps **nearly 3x more** than the poor one — a real,
positive answer to "does candidate scoring pick views that actually
improve reconstruction," not just a plausible-sounding ranking that was
never checked against retrained, held-out-evaluated ground truth.

## 19. Caveat: this scene doesn't yet distinguish "combined" from "either signal alone"

BQ and visibility candidate rankings had a **Pearson correlation of
1.000** on this scene — every candidate ranked identically by both
signals. That means §18 demonstrates "guided selection beats a poor
choice," but *not yet* the more specific "combination beats either
signal alone" claim the milestone actually asks for — with only one
geometric cluster and candidates that vary purely in azimuth at a fixed
radius/elevation, both signals reduce to essentially the same thing
("how angularly far is this direction from the training arc"), so there
was no case for a combination to add value over. This isn't a negative
result — it's the same shape as milestone 2's own trajectory: the first,
simplest scene establishes the pipeline and a real, checked positive
result, and a follow-up scene specifically designed so the signals *can*
diverge (analogous to how the differentiation experiment needed real
thin-rod geometry, not just camera-count differences, to separate
position-only variance from visibility) is the natural next step before
claiming the full milestone-4 story. A candidate along those lines: mix
azimuth-only candidates (where the signals should keep agreeing) with
candidates that are directionally redundant but reveal fine geometry a
current radius/elevation can't resolve (where BQ's position-integrated
term might diverge from a purely angular visibility measure).

# Real-benchmark validation: NeRF-Synthetic "lego"

Everything up to this point ran on a hand-built thin-rod scene
(`scene_spec.differentiation_scene`/`nbv_test_scene`) -- useful for
controlled comparisons (geometry held exactly equal, only camera coverage
varied), but not a real, standardized benchmark. This section repeats the
core question -- does BQ variance flag something a visibility-only signal
misses? -- on the classic NeRF-Synthetic "lego" scan (the Technic
bulldozer/excavator model, 100 standard training views + a held-out
official test split), the same benchmark the original NeRF, 3DGS, and
most follow-up papers report numbers on.

## 20. Getting real data into the pipeline: one real bug, one real dataset-quality issue

`prepare_nerf_synthetic.py` adapts a standard NeRF-Synthetic scene
directory to this project's pipeline. Two things had to be gotten right
before trusting anything downstream:

- NeRF-Synthetic images are RGBA with a genuine alpha channel (transparent
  background), not opaque RGB like every scene used so far.
  `train_minimal_gsplat.load_dataset` / `render_reconstruction.
  render_views` both did `Image.open(path).convert("RGB")`, which on an
  RGBA source *drops* the alpha channel rather than compositing it,
  silently leaving raw (often black) pixel values under transparent
  regions -- ground truth that wouldn't match what the model actually
  renders against its own `background_color`. Fixed by alpha-compositing
  onto a chosen background (white, matching the field-standard convention
  for this dataset) once, up front, so the existing, already-tested
  loaders keep working unchanged on the output.
- The public Hugging Face mirror used here (`phuckstnk63/nerf-synthetic`)
  ships an incomplete `test` split: `transforms_test.json` lists 200
  frames, but only 36 have an actual plain-color PNG (the rest are
  missing, or present only as depth/normal debug renders that aren't
  ground-truth color at all). Caught by checking file existence rather
  than trusting the JSON's frame count; `prepare_nerf_synthetic.py` skips
  frames without a real color image rather than erroring, and 36 held-out
  views is still a fine evaluation set for this project's purposes.

Two conditions built from the real, standard 100-view training split (no
synthetic geometry, no fabricated camera rig): **wide** uses all 100
views; **narrow** uses the 12 whose camera positions are most angularly
clustered (by real 3D position similarity, not an assumed up-axis
convention). Both trained with `train_minimal_gsplat.py`'s real
densification, to 80,000 splats (the `max_splats` cap).

## 21. Reconstruction quality: a real, expected generalization gap

Evaluated on the 36 held-out official test views (never used in training
by either condition):

| condition | train views | held-out PSNR (mean over 30 sampled eval views) |
|---|---|---|
| wide | 100 | **27.17dB** (range 23.66-30.06) |
| narrow | 12 | **19.80dB** (range 13.97-32.19) |

Narrow's *training*-view PSNR was actually higher than wide's (33-35dB
vs. 26-29dB) -- the classic sparse-view overfitting signature: with only
12 views to fit, the model can memorize them well while generalizing
poorly to anything else, exactly what the held-out numbers confirm. This
is a real, standard-benchmark sanity check that both checkpoints are
legitimately trained, not a BQ result -- see `reconstruction_lego_wide.png`
/ `reconstruction_lego_narrow.png` for the visual comparison, which shows
recognizable Technic-model detail (thin support struts, wheels, gear
teeth) with the expected edge-concentrated error pattern.

## 22. Cross-checkpoint BQ differentiation replicates cleanly on real geometry

`real_benchmark_experiment.py` queries BQ position-only variance at the
*same* real 3D points (from the wide checkpoint's own fine-detail splats,
see §23) in both the wide and narrow checkpoints:

- wide checkpoint BQ variance at thin-structure points: 0.00008
- narrow checkpoint BQ variance at the same points: 0.00037
- **ratio (narrow/wide): 4.54x**

This is the "observation-count matters" claim (already established at toy
scale, `bq_splat/results/FINDINGS.md` §9, and at hand-built-scene scale,
this file's §11) replicated on a real, standardized benchmark object for
the first time -- consistent direction, and a *stronger* effect than the
toy-scale result, matching the pattern of every real-data replication in
this project so far.

## 23. The core same-checkpoint claim: not demonstrated with this query methodology -- an open question, not a positive result forced

The harder, more specific question -- does position-only BQ variance flag
genuinely *thin* real structure as more uncertain than *thick* real
structure, within the *same*, well-observed (wide) checkpoint -- did not
show a clear effect: thin-region mean variance 0.00008 vs. thick-region
0.00009, ratio 0.89x, no meaningful differentiation.

"Thin" and "thick" here are defined automatically from each splat's own
converged scale (bottom/top 20% by per-splat median scale) rather than
hand-annotated, since there's no ground-truth part labeling for an
off-the-shelf benchmark object the way the rod scene could just declare
geometry by construction. This generalizes to any real scene, which is
valuable, but it may also be why the signal didn't show up: a single
splat's own scale is a local, somewhat noisy statistic, and the query
point's *neighborhood* (what actually enters the BQ window) can contain a
mix of scales regardless of the query splat's own classification --
diluting whatever local contrast exists. This is reported as a genuine
open question, not spun into a positive result: the toy-scale and
hand-built-scene differentiation experiments controlled geometry by
construction specifically to isolate this effect cleanly (`bq_splat/
results/FINDINGS.md` §3, `differentiation_scene`'s matched-geometry
design), and reproducing that same cleanliness automatically on an
unannotated real object is evidently harder than the cross-checkpoint
comparison in §22. Reasonable next steps, not yet attempted: a
neighborhood-averaged (not per-splat) scale statistic for the thin/thick
classification; restricting query points to splats whose *local
neighborhood* is scale-homogeneous (avoiding windows straddling a
thin-to-thick transition); or a coarser, part-level classification (e.g.
via a rough manual or automated segmentation of the model into
"thin-part" vs. "thick-part" regions) rather than a per-splat scale
quantile.

**Reframing after §23**, prompted directly by feedback on this draft: the
same-checkpoint thin-vs-thick claim (does BQ flag genuinely fine
*geometric* detail as more uncertain than simple detail) is a harder,
more specific question than the one that actually matters most for this
project's central pitch -- that recognizing rendering as Bayesian
quadrature gives you real uncertainty essentially for free, closed-form,
from the same kernel structure already used to represent the scene. That
pitch doesn't need the thin-vs-thick claim; it needs two more fundamental
things demonstrated on real data: that BQ variance grows as visibility is
lost, and that it's elevated in sparsely-covered splat regions --
essentially definitional properties of a GP-quadrature posterior variance,
but worth actually checking on a real trained checkpoint rather than
asserting from the closed-form math alone. §24-25 test exactly those two
things, directly, without routing through a geometric-fineness proxy.

## 24. BQ variance tracks local splat sparsity directly, strongly, on real geometry

`sparsity_correlation_experiment.py` samples 150 real query points from
the wide (100-view) checkpoint's own splats and measures two independent
quantities at each: true local splat count within a fixed window (a
trivial KD-tree ball query, not a BQ computation) and BQ position-only
variance at that same point:

- Pearson correlation (log local count vs. BQ variance): **r = -0.736**
  (p = 8.3e-27)
- Spearman rank correlation: **rho = -0.578** (p = 9.8e-15)
- bottom-20%-density regions vs. top-20%-density regions: **3.49x** higher
  mean BQ variance in the sparse regions

See `sparsity_correlation.png`: a clean, monotonic-looking decay curve --
high variance concentrated at low local splat counts, dropping and
flattening as density increases, exactly the shape a GP posterior
variance should have as a function of local data density. This is the
direct, minimal version of the "uncertainty nearly for free" claim: no
geometric classification, no thin/thick proxy, just local quadrature-node
density vs. the closed-form variance computed from that same density --
strongly, significantly correlated on a real trained checkpoint.

## 25. BQ variance responds to angular coverage gaps, not raw view count -- a sharper finding than a monotonic trend

Five checkpoints of the same real object: wide/rand50/rand25/rand12 at
100/50/25/12 *randomly subsampled* training views (full angular spread
preserved, just sparser), plus narrow (12 *angularly clustered* views).
Same 150 fixed query points, same window, in every condition:

| condition | views | angular spread | mean BQ variance |
|---|---|---|---|
| wide | 100 | full | 0.0000806 |
| rand50 | 50 | full | 0.0000806 |
| rand25 | 25 | full | 0.0000897 |
| rand12 | 12 | full | 0.0000772 |
| narrow | 12 | clustered | 0.0002116 |

The first pass (before `rand12` existed) found something more interesting
than a clean monotonic decay: variance was essentially *flat* from
100 down to 25 random views, and only jumped at narrow's 12 clustered
views -- raising the question of whether the effect was really about
*count* or about *clustering*. `rand12` isolates it directly, holding
count fixed at 12 while varying only whether those 12 views are random or
clustered: rand12 (**0.0000772**) is statistically indistinguishable from
wide/rand50 (**0.0000806**, actually marginally *lower*), while narrow
(**0.0002116**, same count) is **2.75x higher**. See
`visibility_trend.png`.

Reducing total view count, on its own, barely moves position-only BQ
variance here -- as long as the reduced set still spans the full angular
range, the reconstruction stays essentially complete, and local splat
density (§24's actual driver) doesn't meaningfully drop. It's specifically
*coverage gaps* -- directions genuinely unobserved, not merely observed
fewer times -- that BQ variance responds to. This is a sharper, more
useful finding than a naive "more views always means less uncertainty"
result would have been: it says the signal isn't fooled by frame count
alone, which is exactly the property something like active-view planning
needs (ROADMAP.md milestone 4) -- a policy that fires on any view
reduction regardless of whether it actually left a gap would be a much
less useful signal than one that fires specifically when a real gap
exists.

## 26. Fitting the kernel bandwidth to a real checkpoint (ROADMAP.md item 2)

Every result above (§20-25) used a hardcoded `sigma=0.05`, picked once and
never checked against the data — the same gap `bq_splat/hyperparams.py`
found and fixed at toy scale (`bq_splat/results/FINDINGS.md` §5, §7), never
closed at real GS scale. `bq_splat/hyperparams.py` gained an ND version of
its marginal-likelihood machinery (`log_marginal_likelihood_nd`,
`fit_kernel_param_pooled_nd`, working directly with `ProductKernel` over a
real 3D position domain rather than the 1D ray-depth reshape convention)
and `scripts/fit_hyperparameters_real_checkpoint.py` uses it against the
real lego "wide" checkpoint (35,819 splats above opacity 0.1): 25 local
windows (same ball-query convention as `LocalUncertaintyEngine`, capped at
60 points/window) for fitting, 25 disjoint windows held out — the same
fit/held-out split spirit as `validate_trainable_kernel_heldout.py`
(§7), now against real splat data instead of a toy scene.

**Fitted bandwidths differ from the hardcoded value, in different
directions per kernel.** RBF: `0.0624` vs. the hardcoded `0.05` (a modest,
~25% correction). Matern-3/2: `0.0234`, less than half the hardcoded
value — a much bigger correction, and in the *opposite* direction from
RBF's.

**Fitting generalizes, decisively, to held-out windows it never saw.**
Pooled log marginal likelihood on the held-out set: RBF fitted `-1946.00`
vs. hardcoded `-2328.61`; Matern fitted `-1022.37` vs. hardcoded
`-3425.76` — the fitted bandwidth isn't just better on the windows it was
fit on (that would be unsurprising), it's substantially better on windows
it never saw, for both kernels, with the gap especially large for Matern.
This is real evidence the hardcoded value has been leaving marginal
likelihood on the table this whole project, not just at toy scale.

**But the fitted bandwidth does not materially change the headline
sparsity-correlation finding.** Re-running the §24-style check (150 query
points, same checkpoint) with the fitted RBF bandwidth instead of the
hardcoded one: `r=-0.616` (fitted) vs. `r=-0.612` (hardcoded) — both
strongly significant, both essentially the same effect size. (These don't
exactly reproduce §24's `r=-0.74` because the query-point sample here comes
from an RNG stream that had already been advanced by the window-sampling
steps earlier in the same script, not a fresh seed — an internally
consistent comparison between the two sigma values in the same run, not a
literal replication of §24's exact number; worth a clean re-run with a
matched seed before citing both figures together in a paper.)

**Reading together:** the hardcoded bandwidth was leaving real marginal
likelihood on the table (a genuine calibration gap, sizeable for Matern
specifically), but the project's central, most-cited claim — BQ variance
tracks local sparsity — turns out to be robust to that gap rather than an
artifact of it. Good news for the robustness of the headline result;
independent motivation for fitting anyway, since calibration (ROADMAP.md
item 5 — AUSE, sparsification, held-out NLL) is a separate claim from
correlation and this is direct evidence the hardcoded value was
miscalibrated even where it wasn't obviously *wrong*.

**What this doesn't yet do:** one checkpoint, one scene region split into
fit/held-out windows rather than genuinely disjoint scenes; no per-splat
covariance-as-bandwidth comparison (a separate, likely larger, ROADMAP item
2 sub-piece); Matern's much larger correction and much larger held-out gain
relative to RBF's raises a specific new question — is Matern's nominal
`rho=0.05` further from *its* population-optimal value than RBF's
`sigma=0.05` is from RBF's, structurally, or is this specific to lego's
geometry — not yet answered, and worth checking against the thin-rod
checkpoint too before treating it as a general kernel-family property.

## 27. Training under the likelihood (ROADMAP.md item 3): a negative result, reported honestly

Every result above computes BQ variance *after* training, read off a
checkpoint trained by ordinary photometric loss and gradient-triggered
densification. ROADMAP.md item 3 asked whether closing that loop —
training with an uncertainty-weighted Gaussian-NLL term, and/or swapping
densification's trigger from view-space gradient to real closed-form BQ
variance — actually helps, rather than assuming a more "principled"
mechanism must be better.

**Mechanism (`train_minimal_gsplat.py`).** `densify_criterion="bq_variance"`
swaps the densification trigger from `meta["means2d"].grad` to
`compute_per_splat_bq_variance`: at each densify cycle, build a
`LocalUncertaintyEngine` from the current (detached) splat positions/colors
and query BQ position-only variance at every splat's own position, same
percentile-threshold split/clone/prune logic either way. `nll_weight > 0`
adds `0.5 * ((pred-gt)^2/var + log(var))`, averaged over a sparse
`nll_grid_res`-by-`nll_grid_res` grid of real ray-surface points (gsplat's
own expected-depth output, unprojected -- the same construction as
`render_sweep_gif.py`), every `nll_interval` iterations. Honest scope note
written directly into the trainer's docstring: `var` comes from a detached
numpy snapshot (`LocalUncertaintyEngine` isn't a torch object) and is not
itself differentiated through, so the term's practical effect is an
uncertainty-*weighted* photometric reweighting via the `(pred-gt)^2/var`
term, not a fully closed training loop through the BQ posterior itself.

**Experiment (`nll_training_experiment.py`).** Four variants, identical
scene/seed/every other hyperparameter (matching `nbv_experiment.py`'s
established training call for this scene family exactly), trained on the
real 10-view `nbv_out/baseline` arc, evaluated on both training views and
the genuinely disjoint `nbv_out/baseline_eval` held-out ring, 3000
iterations each:

| variant | final n_splats | train PSNR | held-out PSNR |
|---|---|---|---|
| baseline (gradient densify, no NLL) | 5462 | 46.13dB | 21.14dB |
| bq_densify (BQ-variance densify, no NLL) | 3303 | 38.86dB | 20.66dB |
| nll_loss (gradient densify + NLL) | 5485 | 45.98dB | 20.92dB |
| bq_densify+nll | 3248 | 38.19dB | 20.52dB |

**Neither mechanism helped. BQ-variance-driven densification is a real
regression, not a neutral or favorable trade-off**: `bq_densify` ends up
with *both* fewer splats (3303 vs. 5462) *and* substantially worse quality
(-7.27dB train, -0.49dB held-out) than gradient-based densification --
not "fewer splats for slightly worse quality" (a defensible efficiency
trade), a straightforward loss on both axes. Its splat count even
*shrank* in the final densify cycle (3469 -> 3303, pruning outpacing
densification) where every other variant grew monotonically. The NLL loss
term, in isolation (`nll_loss` vs. `baseline`), is close to a no-op,
marginally negative (-0.15dB train, -0.22dB held-out).

**A concrete, honest hypothesis for the BQ-densify regression, not yet
tested**: this project already found and fixed the identical-shaped
problem once before, in the pruning experiment (§15-16) -- BQ variance is
high in genuinely empty space too (correct, but not useful as a signal),
and an *unweighted* combination there spent budget protecting
near-transparent junk until an opacity floor fixed it. Densification's
`compute_per_splat_bq_variance` has no such floor: it queries variance at
every splat regardless of opacity, so a splat sitting in a sparse but
low-opacity, physically-irrelevant region can outscore a splat near real,
under-resolved geometry. The natural next step is the same fix that
worked for pruning -- floor the BQ-variance densification signal by
opacity, the way `pruning_experiment.py`'s `min_opacity_for_bq` already
does -- not yet implemented or tested here, reported as the specific next
step rather than a vague "needs more work."

**Why the NLL term looks like a near-no-op is also explainable rather than
mysterious**: it's computed on a coarse 12x12 grid every 50 iterations
(174,000 real per-pixel-resolution photometric updates happen between
consecutive NLL evaluations at this scene's resolution), and `var` is
detached -- by the trainer's own explicit design (see its docstring), this
first installment reweights an already-small fraction of the training
signal rather than closing a full differentiable loop through the BQ
posterior. A near-zero effect at `nll_weight=0.02` and this sparsity is
the expected outcome of that scope, not evidence the underlying idea is
wrong -- a weight sweep and/or a much finer/more frequent grid (with the
added compute cost that implies) are the natural next checks before
concluding the mechanism itself doesn't help.

**Recorded as a real negative result, not softened**: "training under the
likelihood" was ROADMAP.md's own framing for what a strong paper needs,
and the first honest test of the most direct version of it (swap the
densification trigger, add the NLL term) did not deliver the improvement
the framing implicitly hoped for. This is exactly the kind of finding this
project's process is supposed to surface rather than avoid running the
experiment that might produce it.

**Addendum: testing the opacity-floor hypothesis directly (asked, and
answered, in the same session).** The hypothesis above -- that
`bq_densify`'s regression is the same "BQ variance is high in empty space
too" problem `pruning_experiment.py` already found and fixed (§15-16) --
was tested, not left as a guess. `train_minimal_gsplat.py` gained
`bq_densify_min_opacity`, and the result is a genuinely three-way story,
not a clean confirmation:

| variant | n_splats | train PSNR | held-out PSNR |
|---|---|---|---|
| baseline (gradient densify) | 5462 | 46.13dB | 21.14dB |
| bq_densify, no floor (original regression) | 3303 | 38.86dB | 20.66dB |
| bq_densify, floor v1 (zeroed, but still counted in the percentile) | 6000 (capped) | 45.95dB | 19.81dB |
| bq_densify, floor v2 (zeroed *and* excluded from the percentile) | 1347 | 38.46dB | 20.27dB |

**v1** (`bq_densify_min_opacity=0.3`, matching `pruning_experiment.py`'s
`min_opacity_for_bq` value, zeroing the BQ-variance score for low-opacity
splats but still including them when computing the percentile threshold)
**did confirm the mechanism is real**: train PSNR recovered from 38.86dB
to 45.95dB, nearly matching baseline. But it did so by growing explosively
-- hitting `max_splats=6000` by iteration 900 and staying capped -- because
almost the *entire* initial population starts opacity-ineligible
(`sigmoid(-1.0) ≈ 0.269`, just under the `0.3` floor, is this trainer's
default initial opacity), so a percentile computed over a population
that's mostly zeroed collapses toward 0, making nearly any eligible
splat's variance clear the bar. Held-out PSNR got *worse* (19.81dB, below
even the unfixed regression's 20.66dB) -- more splats overfit to the 10
train views, not better generalization.

**v2** fixed that specific artifact the principled way -- excluding
low-opacity splats from the percentile computation too (`has_data`),
exactly mirroring how the gradient path already excludes "never received a
gradient" splats, rather than leaving them in and just zeroing their
score. This gives a sane, non-degenerate threshold (0.04-0.09 across
cycles, not 0). But it does **not** fix the regression: train PSNR
(38.46dB) lands right back near the original unfixed number, on an even
*smaller* population (1347 splats -- net shrinkage in the very first
densify cycle, opacity-based pruning outpacing the now much more
conservative densification).

**Honest reading**: the opacity-floor hypothesis was directionally right
-- v1 proves quality recovery is achievable by loosening the eligibility
gate -- but the properly-scoped version of the same fix (v2) trades one
failure mode (too many low-quality splats growing) for another (too few
splats growing at all), landing back at essentially the original
regression. This says the mismatch is more structural than a single
missing floor: the percentile-threshold densification scheme itself was
designed and calibrated around gradient-magnitude signals, and BQ variance
doesn't sit well inside that same mechanism regardless of how its
eligibility is scoped. The more promising untested direction, given this:
combine BQ variance *additively* with the existing gradient signal (the
same pattern that already worked for pruning -- opacity-floored BQ term
added to, not swapping out, the existing criterion) rather than replacing
the densification trigger outright.

## 28. Validating against gsplat's real reference trainer (ROADMAP.md item 4), and a real methodological pitfall caught along the way

Every real-data result so far (§9-27) used this project's own from-scratch
trainer (`train_minimal_gsplat.py`), including its own from-scratch
densification (`densify_and_prune`) — a standing, explicitly-acknowledged
reviewer concern (ROADMAP.md item 4): is the central "BQ variance tracks
local sparsity" finding a real property of Bayesian-quadrature-on-splats,
or an artifact of this project's simplified training loop?

**Mechanism.** `train_with_reference_strategy` (`train_minimal_gsplat.py`)
trains using `gsplat.strategy.DefaultStrategy` — gsplat's own official
implementation of the standard 3DGS duplicate/split/prune/opacity-reset
algorithm — instead of this project's `densify_and_prune`, keeping loss,
initialization, and every other hyperparameter identical to `train` so any
downstream difference can be attributed to the densification mechanism
specifically. Deliberately not a full reproduction of gsplat's official
example script (which also uses SSIM loss and a heavier example-only
dependency set — fused-ssim, viser, nerfview — this module's docstring
already explains why those were avoided from the start). One real
integration gotcha, caught by a crash rather than silently wrong output:
`gsplat.rasterization` defaults to `packed=True`, but
`strategy.step_post_backward` defaults to `packed=False` and indexes its
internal state tensors differently depending on which was actually used —
confirmed by reading `gsplat/strategy/default.py`'s source, not assumed,
after the mismatched default threw `IndexError: tuple index out of range`.

**A real methodological pitfall, caught before it could produce a false
negative.** The first cross-trainer check reused this scene family's
existing `sigma=0.9`/`window_radius=1.6` (`nbv_experiment.py`,
`differentiation_experiment.py`) for a `sparsity_correlation_experiment.py`
-style check — and got `r=+0.219` (weak, barely significant, *wrong sign*
relative to every prior sparsity-correlation result). Before concluding
the effect doesn't hold on this scene family, the actual node spacing was
checked directly: median nearest-neighbor distance between splats here is
`~0.023`, while `window_radius=1.6` (picked for *zone-level* directional
experiments on this same scene, radius-1.2-1.5 clusters — not for a local
density measurement) is ~70x that spacing — large enough that nearly every
query point's "local window" swallows most of the scene, saturating the
sparse/dense contrast into noise rather than measuring anything local.
Re-run with `sigma=0.05`/`window_radius=0.15` (scaled to the checkpoint's
actual splat spacing, the way lego's `0.05`/`0.08` was scaled to *its*
spacing) on the exact same checkpoint: **`r=-0.956`, p=1.3e-80** — strong,
correctly signed, stronger than lego's `r=-0.74`. The lesson, worth
stating plainly for anyone reusing this codebase's kernel parameters
across experiments: **`sigma`/`window_radius` must be scaled to the
phenomenon and checkpoint being measured, not carried over from a
different experiment on the same scene** — direct, concrete motivation for
finishing ROADMAP.md item 2's per-checkpoint bandwidth fitting rather than
continuing to hand-pick these per script.

**The actual cross-trainer comparison, with correctly-scaled parameters.**
`train_with_reference_strategy` trained on the identical scene/seed/
hyperparameters as item 3's `baseline` variant (`nbv_out/baseline`, 3000
iterations), reaching 18,529 splats above opacity 0.1 (vs. the from-scratch
trainer's 2,043 — gsplat's real reference strategy grows far more
aggressively at this budget, expected since its default thresholds are
tuned for the ~30k-iteration budgets gsplat's own examples typically use,
not this project's 3000-iteration convention). Sparsity-correlation check
(`sigma=0.05`, `window_radius=0.15`, both checkpoints):

| checkpoint | n_splats | Pearson r | sparse/dense ratio |
|---|---|---|---|
| from-scratch trainer (`densify_and_prune`) | 2,043 | -0.956 (p=1.3e-80) | 1.61x |
| gsplat reference strategy (`DefaultStrategy`) | 18,529 | -0.915 (p=2.7e-60) | 2.05x |

**Same sign, comparable strength, on a checkpoint from a completely
independent, official implementation this project didn't write.** This is
direct evidence the central claim isn't an artifact of this project's
simplified densification — the strongest form of external validation
attempted so far, since gsplat's `DefaultStrategy` is code this project
has no influence over.

**Quality caveat, reported rather than glossed over**: at this *matched*
3000-iteration budget, the reference-strategy checkpoint's PSNR is lower
than the from-scratch trainer's despite having ~9x more splats (train
42.93dB / held-out 20.31dB vs. the from-scratch baseline's 46.13dB /
21.14dB from §27's table) — plausibly because `DefaultStrategy`'s
refine/reset cadence (`reset_every=3000` by default) is calibrated for a
much longer training run than this comparison used, not evidence the
reference algorithm is worse in general. Not investigated further here;
a matched *convergence* comparison (not matched iteration count) is the
correct next check before drawing any quality conclusion, but was out of
scope for what this installment set out to answer (does the sparsity
finding survive a real reference trainer — yes).

## 29. Calibration, not just correlation (ROADMAP.md item 5): a real, nuanced gap

Every real-data result so far is a correlation or a ratio (sparsity vs.
variance, wide-zone vs. narrow-zone). None of it checks whether a claimed
"2x higher variance" region actually has ~2x the squared error, which is
what "calibrated" means and what any downstream use of the number (a
sparsification policy, an active-view budget, an NLL training loss, a
paper claim about confidence) actually needs. This is a different,
stricter question than the sparsity correlation already established, and
this section's honest answer is: **not really, in absolute terms; modestly
yes in ranking terms; substantially weaker than the sparsity-correlation
numbers might suggest.**

**Protocol (`calibration_experiment.py`).** Leave-one-out cross-validation
on real splat colors: for many real splats, remove that splat from its own
local BQ neighborhood (`LocalUncertaintyEngine` gained an `exclude_idx`
parameter for exactly this — a ball query centered on a real splat's own
position always finds that splat at distance 0, so without excluding it
the "prediction" would trivially see its own held-out answer), predict its
color from its real neighbors alone, and compare the BQ posterior
mean/variance against the splat's own real, never-seen-by-the-prediction
color. Three metrics, on three checkpoints (lego "wide", the thin-rod
from-scratch-trainer checkpoint, the thin-rod gsplat-reference-strategy
checkpoint from §28 — the same three used for the cross-trainer check,
reused here for a second, independent purpose):

| checkpoint | Pearson r(var, sq.err) | AUSE (BQ / random, lower better) | held-out NLL (BQ / constant-var, lower better) |
|---|---|---|---|
| lego wide | +0.147 (p=0.011) | 0.308 / 0.405 | 6022.70 / 4040.22 |
| thin-rod, from-scratch trainer | **-0.202** (p=4.4e-4, wrong sign) | 0.176 / 0.180 | 9.77 / 8.65 |
| thin-rod, reference-strategy trainer | +0.134 (p=0.021) | 0.048 / 0.071 | 4.78 / 4.56 |

**Three separate, genuinely different readings, not one verdict:**

1. **Direct correlation is weak everywhere, and wrong-signed on one
   checkpoint.** All three `|r|` are under 0.21 — far weaker than the
   sparsity-vs-variance correlations (`r=-0.74` to `-0.96` across §24, §26,
   §28) that this project's headline claim rests on. The from-scratch
   thin-rod checkpoint's correlation is even negative (higher variance,
   *lower* error) — a real miscalibration signal on that specific
   checkpoint, not a typo or a sign-convention artifact (re-checked
   directly).
2. **AUSE (ranking-based) is more encouraging: BQ ordering beats random on
   two of three checkpoints by a real margin**, and is roughly tied with
   random on the third (from-scratch thin-rod, consistent with that
   checkpoint's weak/wrong-signed correlation). This matters because
   ranking, not absolute value, is what a sparsification or pruning policy
   actually consumes — `pruning_experiment.py`'s already-positive result
   (§15-16) is a ranking-based use, consistent with AUSE being the more
   favorable of these three metrics.
3. **Held-out Gaussian NLL is worse than a flat, constant-variance
   baseline on all three checkpoints, in every case.** This is the
   strictest test and the one where the finding is most clearly negative:
   the *absolute scale* of leave-one-out BQ variance is not a trustworthy
   per-point confidence value as measured here — using the SAME variance
   for every point would give a better-calibrated NLL than using BQ's own
   per-point number. lego's gap is especially large (6023 vs. 4040),
   likely because a few points with very small predicted variance and even
   modest real error blow up the `squared_error / variance` term
   disproportionately -- a classic failure mode for NLL-style scoring
   under an uncalibrated small-variance tail, not evidence the *ranking* is
   equally bad (AUSE for lego was the best of the three checkpoints).

**Why this doesn't contradict the sparsity-correlation claim (§20-26, §28)
-- it sharpens what that claim actually is.** "BQ variance tracks local
splat sparsity" and "BQ variance is a calibrated estimate of a splat's own
leave-one-out prediction error" are different claims. A sparse-but-locally-
consistent region (few neighbors, but the ones present agree closely in
color, e.g. a smooth low-frequency surface patch) can have high variance
by construction (correctly reflecting thin quadrature coverage) while
still being *predicted accurately* by those few neighbors -- sparsity and
leave-one-out error are related but not identical, and this section is the
first place in this project's real-data results where they're checked
against each other directly rather than conflated. The honest, sharpened
claim going forward: **the sparsity-correlation result is real and robust
(now checked across two trainers, two scene families, and one dataset,
consistently strong); the stronger claim that BQ variance is a directly
calibrated, absolute per-point error bound is not yet supported by this
data**, and should not be asserted in a paper without further work --
either a recalibration step (a monotonic rescaling fit against held-out
error, standard practice for miscalibrated uncertainty estimates) or the
per-splat-covariance-as-bandwidth extension from ROADMAP.md item 2, not
yet tried, which could plausibly tighten the absolute scale by using each
splat's own learned anisotropic covariance instead of one shared scalar
bandwidth.

Full sparsification-curve plots for all three checkpoints in
`gs_experiment/results/calibration_sparsification_*.png`.

## 30. Window-radius sensitivity ablation (ROADMAP.md item 6): the §28 sign flip is a general, predictable pattern

§28 found that reusing a `window_radius` picked for a different experiment
on the same scene flipped the sparsity-correlation's sign entirely
(`r=+0.22` at `window=1.6` vs. `r=-0.96` at `window=0.15`). Left open
there was whether that was a one-off quirk of that specific checkpoint or
a real, general property of the method. `window_radius_ablation.py`
answers this directly: sweep `window_radius` from 0.2x to 8x each
checkpoint's already-established value (sigma held fixed, isolating this
one knob), on all three real checkpoints used throughout items 4-5 (lego
wide, and both thin-rod trainers).

**The pattern is the same, clean, and monotonic on every checkpoint:**

| checkpoint | 0.2x | 0.5x | 1x (established) | 2x | 4x | 8x |
|---|---|---|---|---|---|---|
| lego wide | -0.88 | -0.83 | -0.74 | -0.32 | -0.27 | **+0.90** |
| thin-rod, from-scratch | -0.85 | -0.97 | -0.96 | -0.89 | -0.68 | **+0.70** |
| thin-rod, reference-strategy | -0.94 | -0.94 | -0.92 | -0.75 | -0.13 (n.s.) | **+0.54** |

(all cells `p<0.05` except the one marked n.s.)

**The correlation is strongly negative and robust across a wide, natural
range (0.2x-2x the established value) on all three checkpoints** — the
headline claim does not depend on a lucky hyperparameter pick, it holds
across a full order of magnitude of window sizes smaller than or near what
was actually used. **Past roughly 4x, it degrades and then flips sign on
every checkpoint** — the exact failure mode §28 found is not a one-off
artifact of that specific checkpoint/parameter combination, it is a
general, predictable property of the local-density measurement, and now
it's characterized with a curve instead of one data point.

**A plausible mechanism, stated as a hypothesis, not asserted as
fact**: once `window_radius` grows large enough that a "local" window
covers a large fraction of the whole scene, query points stop
differing meaningfully in *how much* of the scene they see and start
differing mainly in *where they sit relative to the point cloud's overall
extent* — a query near the bounding volume's center systematically sees
more neighbors within a huge radius than one near its edge, for reasons
having nothing to do with genuine local quadrature coverage. If BQ
variance and this edge-vs-center geometry happen to correlate positively
for unrelated reasons, that would produce exactly the observed large-
window sign flip. Not verified directly here (would need a synthetic
scene with density and edge-distance deliberately decorrelated) — flagged
as the natural follow-up rather than claimed as confirmed.

**Practical guidance this gives, now grounded in three checkpoints rather
than one:** pick `window_radius` well below the point where a typical
query's window captures a large fraction of total scene splats (roughly,
keep the median local count well under 10% of the checkpoint's total
splat count) — both established values used throughout this project
(lego's `0.08`, the thin-rod family's `0.15`) sit safely inside the
strongly-negative, robust region of this curve, not near the transition.
Full sweep table and plot in `gs_experiment/results/window_radius_ablation.png`.

## 31. Kernel-family ablation with fitted bandwidths (ROADMAP.md item 6): RBF and Matern trade wins, neither dominates

§22 checked RBF vs. Matern-3/2 on one real checkpoint and found they agree
on spatial *pattern* (correlation 0.98) while differing ~150x in absolute
*scale* — never whether the two kernels differ on the actual claims this
project's other results are built on (sparsity correlation, calibration),
and never with a properly *fitted* bandwidth for each rather than an
arbitrary shared numeric value. This closes both gaps at once, and along
the way closes an open question from item 2 (`gs_experiment/results/
FINDINGS.md` §26): "is Matern's much larger correction on lego a general
kernel-family property, or specific to that scene's geometry?" Freshly fit
on both thin-rod checkpoints this session (`scripts/
fit_hyperparameters_real_checkpoint.py --window-radius 0.15`): **the
opposite pattern from lego** — on lego, Matern needed the bigger
correction (`0.05 -> 0.0234`, more than half); on both thin-rod
checkpoints, *RBF* needed the bigger correction (`0.05 -> 0.11-0.12`,
more than double) while Matern stayed close to the hardcoded value. Answer:
**scene-specific, not a general kernel-family property** — which kernel
needs the larger correction depends on the scene, not a fixed rule.

One more real nuance surfaced while fitting: on the reference-strategy
thin-rod checkpoint, Matern's *fitted* bandwidth generalized *worse* to
held-out windows than the *hardcoded* one (pooled held-out log marginal
likelihood: fitted `-1032.01` vs. hardcoded `-560.36`) — a genuine
overfitting signal, the same shape as the toy-scale RBF overfitting found
in `bq_splat/results/FINDINGS.md` §7. `kernel_family_ablation.py` uses the
hardcoded value for that one cell rather than trusting an overfit number,
noted explicitly rather than silently picked.

**With fitted bandwidths for both kernels on all three checkpoints, a
clean, consistent, three-way trade-off emerges — neither kernel wins
outright:**

| checkpoint | metric | RBF | Matern-3/2 | winner |
|---|---|---|---|---|
| lego wide | sparsity r | -0.726 | -0.590 | RBF |
| thin-rod, from-scratch | sparsity r | -0.985 | -0.920 | RBF |
| thin-rod, reference-strategy | sparsity r | -0.950 | -0.814 | RBF |
| lego wide | calibration r | 0.065 | 0.277 | Matern |
| thin-rod, from-scratch | calibration r | **-0.281** (wrong sign) | -0.170 | Matern |
| thin-rod, reference-strategy | calibration r | 0.067 | 0.165 | Matern |
| lego wide | held-out NLL (lower better) | 28,490 | **1,394,541** | RBF, by far |
| thin-rod, from-scratch | held-out NLL | 166 | 4,676 | RBF |
| thin-rod, reference-strategy | held-out NLL | 514 | 3,164 | RBF |

**RBF wins the sparsity-correlation claim and the NLL-calibration claim,
consistently, on every checkpoint** (lego's NLL gap is dramatic — Matern's
small fitted bandwidth produces some very small variances that blow up the
`squared_error/variance` term whenever real error is nonzero there, a
severe absolute-scale miscalibration specific to that small a bandwidth).
**Matern wins the calibration *ranking* correlation, consistently, on
every checkpoint** — including flipping RBF's wrong-signed
`thinrod_fromscratch` result (`-0.281`) to a less-wrong (still negative,
but smaller-magnitude) `-0.170`.

**Reading, stated carefully rather than picking a winner to fit a
narrative**: this project's two headline real-data claims are best served
by *different* kernels. The sparsity-correlation claim (the one most of
this project's results are built on) is more robust under RBF. The
calibration *ranking* signal (§29's more favorable metric, and the one
`pruning_experiment.py`'s success actually depends on) is more robust
under Matern. Absolute-scale calibration (NLL) favors RBF strongly enough
that Matern's small fitted bandwidths should not be used where a literal
per-point confidence value is needed without further work. A paper
claiming one universally-superior kernel family would not be supported by
this data — the honest claim is "kernel choice trades off which specific
property is best-served," which is itself a real, reportable finding
about this project's central mechanism, not a hedge to bury.

## 32. All 8 standard NeRF-Synthetic scenes (ROADMAP.md items 4 and 7): the headline claim replicates everywhere, calibration doesn't, and one new methodological wrinkle

Every real-data result before this section ran on lego alone — the "full
NeRF-Synthetic, all 8 scenes" gap ROADMAP.md item 4 named explicitly, and
the multi-scene statistics item 7 asks for. This closes it: chair, drums,
ficus, hotdog, materials, mic, and ship, the remaining 7 standard scenes,
downloaded, trained, and evaluated with the identical sparsity-correlation
and leave-one-out calibration protocol (`multi_scene_experiment.py`) used
on lego throughout items 4-6, at `sigma=0.05`, `window_radius=0.08` (lego's
own established convention, not the thin-rod family's).

**Two real, practical frictions before any numbers, both worth recording
plainly for anyone reproducing this**: the mirror lego's data came from
(`phuckstnk63/nerf-synthetic`, §20) turns out to contain *only* lego —
not a partial mirror of all 8 scenes, an accidental single-scene one. The
complete 8-scene mirror used here (`pablovela5620/nerf-synthetic-mirror`,
verified file-by-file: 100 train / 100 val / 200 real color test images
per scene, all 8) is a different repo. And: launching all 7 downloads in
parallel against the public HF endpoint immediately hit its anonymous
rate limit (429, "1000 api requests per 5 minute period") — repeatedly,
even on retry, until downloads were serialized one at a time. A real
practical constraint for anyone trying to reproduce this at similar scale
without a paid HF account, not a one-off flake.

**Scope choice, stated up front**: unlike lego's original ~80,000-splat-
cap run, these 7 scenes trained at a deliberately lighter budget
(`n_splats=2000`, `max_splats=15000`, `n_iters=2500`) for feasible
wall-clock across 7 scenes in one sitting. Reconstruction quality was not
the point — both checks measure the relationship between local density
and BQ variance/error, not how good the render looks — but this means
these 7 checkpoints have far fewer splats (2,378-3,869) than lego's
35,819, which turns out to matter (see below).

| scene | n_splats | median local count | sparsity r | calib r | NLL(bq) | NLL(const) |
|---|---|---|---|---|---|---|
| lego | 35,819 | 120.0 | -0.736 | -0.032 | 8,023.56 | 3,697.50 |
| chair | 2,621 | 1.0 | -0.959 | 0.092 | 469.41 | 418.72 |
| drums | 2,669 | 1.0 | -0.938 | 0.146 | 393.97 | 359.87 |
| ficus | 2,953 | 1.0 | -0.950 | 0.007 | 687.92 | 636.15 |
| hotdog | 2,961 | 1.0 | -0.970 | 0.102 | 521.89 | 451.99 |
| materials | 3,869 | 1.0 | -0.916 | 0.109 | 398.00 | 360.43 |
| mic | 2,378 | 1.0 | -0.935 | 0.079 | 442.06 | 402.75 |
| ship | 3,101 | 1.0 | -0.960 | 0.280 | 290.34 | 268.56 |

(all `sparsity_r` values `p<1e-25`; lego re-evaluated fresh with this
section's exact script/RNG rather than quoting §24's or §26's numbers,
which used different RNG states — the point of this table is a genuinely
matched comparison, not a reproduction of an earlier figure.)

**The sparsity-correlation claim replicates strongly on every single one
of the 8 standard NeRF-Synthetic scenes** — `r` between -0.74 and -0.97,
every one significant beyond any reasonable doubt. This is the strongest
multi-scene evidence for this project's central claim gathered so far:
not one scene, not a hand-picked pair, the complete standard benchmark.

**Calibration does not replicate as a positive result — consistent with
§29's finding, now on 7 more scenes.** `calib_r` is weak everywhere
(0.007 to 0.28, one value even slightly negative on lego at this specific
evaluation) and **held-out NLL is worse than a flat constant-variance
baseline on all 8 scenes without exception**. This is not a new negative
finding so much as the same one from §29 holding up under much broader
testing — worth stating precisely: §29 checked 3 checkpoints and found
calibration weak/inconsistent; this checks 8 and finds the same pattern
every time, which makes it a considerably more solid basis for the
"sparsity correlation is real and robust, absolute calibration is not
yet established" claim than 3 checkpoints could.

**A new methodological wrinkle, not previously visible at lego's scale**:
every one of the 7 lighter-budget checkpoints has a *median local count
of exactly 1.0* at `window_radius=0.08` — meaning the typical query
point's window contains at most one other splat, a far coarser
discretization than lego's median of 120. The strong correlations on
these 7 scenes are real (not a bug — spot-checked the underlying counts
directly) but are being driven by a much coarser "isolated vs.
not-isolated" contrast than lego's smoother density gradient, not
necessarily the same thing being measured. This doesn't undermine the
replication — if anything, a strong, consistently-signed correlation
surviving under much coarser binning is a real point in the claim's
favor — but it means these 7 numbers and lego's aren't measuring
identically fine-grained density variation, and a like-for-like
replication (matching lego's splat count/density more closely, at the
cost of the wall-clock this section's lighter budget was chosen to save)
is the natural, explicitly flagged follow-up before citing "8/8 scenes,
`r<-0.7` everywhere" as a headline paper number without this caveat
attached.

## 33. A real view-direction uncertainty *gradient*, not just a binary split

Every prior directional-uncertainty result in this project — the toy
isolation experiment (`bq_splat/results/FINDINGS.md` §9, 2.46x), the real
GS wide/narrow ratio (§17-19, §22, 18.7x-4.54x) — compares exactly two
discrete coverage conditions. Real deployments (a robot scanning past a
shelf, a SLAM system that revisits some regions more than others) don't
produce two buckets, they produce a continuum. This is the first
experiment built to have one, and to check whether BQ directional
variance actually recovers a *designed, continuous* coverage gradient,
not just tell two extremes apart.

**Construction (`scene_spec.gradient_scene`, real Blender rendering, real
gsplat training — not a toy/mock scene).** 5 identical thin-rod clusters
(same rod count/spread as `differentiation_scene`'s zones — spatial
density held exactly equal across zones by construction, the same
confound control `validate_directional_combined.py` and
`differentiation_scene` already use), spaced along a line. Each zone gets
its own turntable-arc camera rig, all centered on the *same* absolute
azimuth (`theta_center_deg=200`) — only the arc's angular half-width
varies, linearly from `8 deg` (zone 0, narrowest) to `180 deg` (zone 4,
equivalent to a full ring) — a real, monotonic angular-coverage gradient
by construction, real cameras, real render, real training (3000
iterations, real gradient-triggered densification, 1500 -> 4169 splats).
A single fixed query direction — the azimuth diametrically opposite the
shared arc center (`20 deg`), computed via a real camera pose's
direction-to-a-point (`directions_from_positions_to_camera`, the same
robust construction `differentiation_experiment.py`'s real-scene builder
uses, not hand-derived spherical trigonometry — see that module's
documented elevation bug from doing it the naive way) — is genuinely
comparable across every zone, since the arc center itself never moves.

**Result: a clean, strictly monotonic gradient, recovered exactly.**

| zone | half-width (deg) | directional variance | spatial-only variance (control) |
|---|---|---|---|
| 0 | 8 | 15.288 | 0.170 |
| 1 | 51 | 15.259 | 0.163 |
| 2 | 94 | 14.602 | 0.262 |
| 3 | 137 | 6.511 | 0.197 |
| 4 | 180 | 1.179 | 0.149 |

Directional variance is strictly monotonically decreasing across every one
of the 5 zones as coverage widens (rank correlation `rho=1.000`), a
`12.97x` range from narrowest to widest — stronger than every prior
binary-split result in this project (2.46x toy, up to 18.7x on the
original real wide/narrow split, but this is a genuinely different,
harder claim: not "high vs. low," a full graded curve matching a designed
5-level gradient exactly). The position-only control stays far flatter
(`1.76x` range, no consistent trend with half-width) — confirming the
effect is directional, not a spatial-density artifact the geometry-
matching construction failed to fully equalize. Plot in
`gs_experiment/results/directional_gradient.png`.

**What this adds beyond the existing wide/narrow results**: a binary
comparison can show "BQ tells covered from uncovered apart" without
showing it tracks *degree* of coverage in any principled way — a
threshold detector could pass that test. A monotonic response across 5
designed intermediate levels is a meaningfully stronger claim, and the
realistic framing (a scene where coverage genuinely varies continuously
across space, the way it would in an actual partial-mapping scenario) is
closer to what ROADMAP.md item 8's realistic NBV framing and the original
SLAM motivation (`bq_splat/results/FINDINGS.md` §9) actually need than a
single wide-vs-narrow pair.

**What this doesn't yet do**: one scene, one seed, one query direction
per zone (always "diametrically opposite," not a sweep of query angles
within each zone — a natural next check, since the *within-zone* angular
profile of directional variance, not just the *across-zone* one, is what
a real NBV candidate-scoring policy would actually need). Also still a
hand-built thin-rod scene, not a real captured/benchmark one — the same
gap flagged throughout this project's real-data sections.

## 34. The same gradient experiment on a real NeRF-Synthetic scene: the effect does not transfer, and a real, diagnosed reason why

§33's designed 5-level coverage gradient (hand-built thin-rod clusters,
purpose-built Blender camera arcs) gave a clean, strictly monotonic
12.97x directional-variance range. The natural next question -- does the
same effect show up on a real benchmark object, not a scene built to make
it easy -- was tested directly rather than assumed to transfer. It does
not, and the reason was chased down rather than left as an unexplained
null result.

**Construction (`real_directional_gradient_experiment.py`).** A real
camera rig can't be redesigned for a real dataset -- lego's 100 real
training poses are fixed. `prepare_nerf_synthetic.select_gradient_subset`
approximates the same idea by subsampling the real pool into 5 conditions
of *equal view count*, drawn from windows of increasing real angular
spread around a shared reference view (holding count fixed the same way
§33 held rod-cluster geometry fixed across zones). Each condition trains
its own real gsplat checkpoint from that specific real 6-view subset of
the same 100 real training photos. Directional and position-only BQ
variance are queried at the same fixed point (world origin, NeRF-
Synthetic's own object-centering convention) and the same fixed, real
query direction (the real camera direction from the full 100-view pool
most dissimilar to the reference view -- the same "opposite side"
construction `differentiation_experiment.py`'s real-scene builder uses)
across all 5 conditions.

**First attempt: a clean-looking but degenerate null (window radius, not
the underlying phenomenon).** Reusing lego's established
`window_radius=0.08` gave an almost perfectly flat directional variance
(1.00x range) -- but a direct check found *zero or one* splat within that
radius of the query point (origin) in every single condition. These
lighter, 6-15-view checkpoints have ~1,500-2,900 splats total (vs. the
full wide checkpoint's 35,819), so `0.08` -- already shown in §28/§30 to
need rescaling to a checkpoint's actual density -- was reaching essentially
empty space, not real local geometry. The exact §28/§30 pitfall,
recognized and checked before trusting the number, not reported as a
finding.

**Second attempt, correctly diagnosed: window radius fixed
(`0.5`, confirmed hundreds of real neighbors reached), and a real
achievable-spread-range concern also fixed and re-checked.** With the
original `n_per_zone=15`, the tightest cluster of 15 real views achievable
anywhere in lego's 100-view pool is already `37 deg` wide (real dataset
density limit, not a construction bug -- checked directly: the 5 closest
real views to any reference already span `13 deg`, but 15 do not fit in
that a cone). Reducing to `n_per_zone=6` recovered a real range comparable
to §33's designed one (`13.3 deg` to `150 deg`, ~11x, vs. §33's `8-180
deg`, ~22.5x) -- ruling out "the real dataset just can't produce as
extreme a manipulation" as the explanation before concluding anything.

**With both confounds checked and ruled out, the result is a genuine,
robust null**: directional variance is `0.885, 0.885, 0.885, 0.884,
0.858` across the 5 conditions (13.3 deg -> 150 deg spread) -- a `1.03x`
range, statistically indistinguishable in magnitude from the position-only
*control*'s own `1.04x` range. Where §33 showed a directional signal
~7x larger than its control's noise floor, this shows no directional
signal standing out above the control at all. Plot in
`gs_experiment/results/real_directional_gradient.png`.

**A real, reasoned candidate explanation, stated as a hypothesis, not
asserted as confirmed**: §33's thin-rod clusters are small, isolated, and
nearly rotationally symmetric -- every camera in a zone's rig has an
unoccluded view of the whole cluster, so camera-*position* spread and
per-splat observed-*direction* spread are the same thing by construction.
A real object like lego has genuine self-occlusion and locally-varying
surface normals: a splat on one face can only ever be legitimately
observed from a limited cone of angles regardless of how wide the overall
camera rig spans, and `visibility_attribution`'s real geometric occlusion
test (not an assignment rule) determines per-splat which of the selected
cameras actually see it. A `150 deg`-wide camera-*position* rig may still
only provide a much narrower *effective* observed-direction range for
most individual splats than the rig's own spread suggests -- decoupling
the manipulated variable (camera position spread) from what the
directional kernel actually measures (per-splat observed-direction
spread) in a way the toy scene's isolated, occlusion-free rods
structurally could not. Not verified further here (would need comparing
each splat's actual attributed-observation-direction spread against its
camera rig's nominal spread, per condition) -- named as the concrete next
diagnostic rather than left as a shrug.

**Recorded plainly**: a clean, strong effect on a scene built to show it
does not automatically transfer to a real object, even after ruling out
two real candidate artifacts first. This is a genuine limit on the §33
result's generality, not a contradiction of it -- and a substantive,
reportable finding in its own right about what real self-occluding
geometry does to the directional-uncertainty mechanism, worth exactly as
much attention in any write-up as §33's positive result.

## 35. The same test on an actual photographed scene (GAVIS/PUP-style real capture): the null replicates, on a stronger test

§34 found the directional-gradient effect doesn't transfer from the
hand-built thin-rod scene to real (but still synthetic-Blender-rendered)
lego geometry. The natural next question — does it fare any differently
on a genuinely *photographed* scene, with real COLMAP-estimated poses,
the actual kind of data GAVIS and PUP 3D-GS report numbers on — was
tested directly rather than left to the lego result to answer by
extrapolation.

**A new capability this required, not previously needed anywhere in this
project**: `colmap_loader.py` reads COLMAP's binary `cameras.bin`/
`images.bin` format (the standard pose output for real-captured NeRF/GS
datasets) and converts to this project's `transforms.json` convention.
Every prior scene had either hand-authored ground-truth poses (Blender)
or a synthetic dataset's own pre-baked poses (NeRF-Synthetic); this is
the first time poses themselves are an SfM *estimate* from real
photographs, not known exactly. Verified independently, not just run:
`colmap_image_to_c2w_opengl` is documented as the exact inverse of
`nerf_transforms.opencv_viewmat_from_c2w`'s operation, and a unit test
checks this by round-tripping a random pose through both functions and
confirming it recovers the original world-to-camera matrix exactly, not
just "runs without crashing." Known, stated limitation: COLMAP's
distortion parameters are read but not applied (this project's rendering
pipeline has no distortion model) — a real, acknowledged source of error
specific to using genuine photographs that no synthetic scene has.

**Scene**: Mip-NeRF360 "bonsai" (`nvs-bench/mipnerf360` on Hugging Face,
COLMAP `PINHOLE` model, 292 real photographs, verified single shared
camera matching this project's one-`camera_angle_x`-per-scene
convention). Same construction as §34: 5 equal-view-count (8 views each)
conditions of increasing real angular spread around a shared reference
view, built directly on COLMAP-derived poses via the same
`select_gradient_subset` used for lego (no change needed — it only reads
frame translations), each trained into its own real checkpoint (3000
iterations, real densification, reasonable convergence: 20-24dB train
PSNR at this light budget).

**The exact same query-point pitfall as §28/§30/§34 showed up again, in a
new form, and was caught the same way — by checking, not assuming.**
World origin is close to the camera-center *centroid* here (confirmed:
`[0.07, 0.07, 0.05]`), which was the working assumption carried over from
NeRF-Synthetic's object-centered convention — but a COLMAP reconstruction
places its origin from SfM geometry, with no guarantee the *photographed
object itself* sits there just because the cameras orbit near it. Checked
directly: the nearest real splat to the origin was ~0.7-0.9 units away
with a handful of real neighbors at best in a reasonable window, in every
one of the 5 checkpoints — an even more clear-cut version of the same
degenerate-query problem, this time from a wrong assumption about the
scene's own coordinate convention rather than a mismatched kernel
parameter. Fixed the same way as before (check, don't assume): computed
each checkpoint's own median high-opacity splat position and used the
mean of those five real, data-derived points (`[0.61, 1.17, 1.51]`) as a
genuinely representative query point, then re-swept window radius until a
real, substantial neighbor pool (hundreds of real splats) was confirmed
reached in every condition (`window_radius=0.8`).

**With the query point and window radius both grounded in real,
confirmed data rather than assumed convention, the result replicates
§34's null, on a harder, more externally valid test:**

| zone | real spread (deg) | n_neighbors | directional variance | spatial-only variance |
|---|---|---|---|---|
| 0 | 23.9 | 400 | 3.79719 | 3.65249 |
| 1 | 35.9 | 400 | 3.79719 | 3.66404 |
| 2 | 61.8 | 400 | 3.79719 | 3.67942 |
| 3 | 103.4 | 266 | 3.79709 | 3.74072 |
| 4 | 172.9 | 268 | 3.77485 | 3.73572 |

Directional variance range: `1.01x`. Position-only control range:
`1.02x` — if anything, the control varies slightly *more* than the
directional signal, the same "no signal standing out above the control's
own noise" pattern §34 found on lego. Plot in `gs_experiment/results/
real_capture_directional_gradient.png`.

**This is now a two-for-two null on real geometry, against a one-for-one
positive on designed geometry — worth stating plainly rather than
averaged into a vague "mixed results".** The hand-built thin-rod scene
(§33) reliably produces the effect (12.97x, clean monotonic gradient).
Both attempts at real or real-ish geometry (§34's lego, this section's
genuinely photographed bonsai) show no directional signal distinguishable
from a position-only control's own noise floor, despite two independent,
scene-specific artifact checks (window radius, query point) being caught
and fixed *before* trusting either null result — these are not two
untroubleshot negative results, they're two results that survived real
troubleshooting and still came back null. The self-occlusion /
locally-varying-surface-normal hypothesis from §34 gains real, if still
unconfirmed, support: it would predict exactly this pattern (isolated,
occlusion-free toy geometry shows the effect; any real object with
self-occlusion, synthetic or photographed, does not), and now has two
independent real-geometry data points consistent with it rather than one.

**What a paper claim should say given this, right now**: "BQ directional
variance recovers a designed view-coverage gradient on isolated,
occlusion-free geometry" is supported. "BQ directional variance tracks
view-coverage gradients on real objects/scenes" is *not* supported by
this project's evidence and should not be asserted without either (a)
confirming and then correcting for the self-occlusion mechanism, or (b) a
different real-scene construction where the manipulated variable (camera
spread) and the measured one (per-splat observed-direction spread) are
less decoupled by real geometry than an arbitrary query point on a
complex object apparently allows.

## 36. Replacing the point-sample summaries with full per-pixel animated sweeps, on all three scenes

§33-35's numbers (5 zone-center point samples per scene) are honest but,
fairly, easy to distrust: five numbers cannot rule out real spatial
structure the sample points happened to miss, in either direction. This
section replaces the point samples with a full per-pixel, per-frame
visualization on all three scenes tested so far, to see the whole picture
rather than infer it from five points.

**Mechanism (`render_directional_uncertainty_sweep.py`).** Same real-
depth-unprojection construction as `render_sweep_gif.py` (gsplat's own
expected-depth output, not an interpolated proxy), extended two ways: the
BQ term queried is *directional* (position+direction), not position-only;
and the query *direction* at every pixel is the real direction from that
pixel's actual unprojected 3D point to the *current frame's* camera
position — not one fixed direction reused for a whole sweep. As the
camera orbits, this is the natural generalization of the aggregate
experiments' single query direction to the full range an NBV/SLAM policy
would actually evaluate. A framing bug was caught and fixed before
producing anything worth keeping: the first render used a fixed square
crop and a far camera, wasting most of the frame on empty space around
the (elongated, small) toy scene — fixed with a wide (640x240) aspect
ratio and a tighter, geometry-matched radius/FOV.

**Toy gradient scene (§33's positive result): confirmed, with real
sub-structure the point samples didn't show.** [`directional_uncertainty_sweep_toy.gif`]
Across the 60-frame orbit, the designed left-to-right (narrow-to-wide)
gradient is visible directly: zone 0 (narrowest, 8 deg) lights up sharply
whenever the current viewing angle falls outside its training arc, while
zone 4 (full-ring) stays dark almost everywhere. It also shows real
structure the 5 point samples entirely missed — e.g. zone 0 has a
consistent *partial* dark patch even at angles mostly outside its
training arc (a real sub-region that happens to be better-constrained
than the rest of that zone), and at the sweep's ~90/270-degree marks the
row is viewed end-on (a real, expected geometric consequence of orbiting
a linear arrangement of zones, not an artifact). The point-sample summary
was a fair compression of a real, rich pattern, not misleading.

**Real lego (§34's null): confirmed, and a real interpretive caveat
surfaced that the aggregate number didn't make visible.**
[`lego_gradient_0_sweep.gif`, `lego_gradient_4_sweep.gif`] Both the
narrowest (13.3 deg) and widest (150 deg) real-view conditions show
directional uncertainty saturated near its maximum across almost the
entire frame, in every frame of both sweeps — visually confirming the
1.01x-range null, not contradicting it. But the reconstructions
themselves are also visibly poor quality in both conditions (heavy
floater/artifact noise) — because each condition trains on only *6 real
views total*, `select_gradient_subset`'s fixed-count design (needed to
hold count constant while varying spread — see §34) means even the
"wide" 150-degree condition is a severe absolute view-count shortage on
real, complex geometry, not just a spread manipulation. Seeing this
required looking at the actual renders — the aggregate PSNR-style numbers
never reported here would have shown "some reconstruction happened" without
revealing how poor. **This is a real confound the point-sample results
didn't surface**: uniformly-high uncertainty in both conditions may partly
reflect "not enough total data to reconstruct at all" rather than being
cleanly diagnostic of coverage *spread* specifically.

**Real bonsai (§35's null): confirmed, but visually a genuinely different
character than lego, still without a clean directional trend.**
[`bonsai_gradient_0_sweep.gif`, `bonsai_gradient_4_sweep.gif`] Unlike
lego's near-total saturation, both bonsai conditions show real, rich
spatial structure — patchy, high-frequency light/dark regions that shift
noticeably frame to frame. This is genuinely more structure than a flat
null would suggest, and worth taking seriously rather than dismissing. But
comparing the narrowest and widest conditions side by side, the pattern
looks comparably noisy/patchy in *both* — no obvious, consistent
difference distinguishing them the way the toy scene's zones were
visually distinct. This is consistent with, not a contradiction of, the
quantitative null (1.01x range, indistinguishable from the position-only
control's own 1.02x) — real per-splat variation exists, but it does not
resolve into a visible camera-coverage-tracking trend.

**What this changes about the real-geometry null going forward**: the
self-occlusion hypothesis from §34 remains the leading, still-unconfirmed
explanation, but this section adds a second, concrete, testable
alternative specific to the real-scene *construction* used so far: both
lego and bonsai's gradient conditions confound "narrow angular spread"
with "few total views," because `select_gradient_subset` was built to
hold view *count* fixed while it varies spread — which, given real
datasets don't offer arbitrarily many views within an arbitrarily tight
cone, meant picking a small fixed count (6-8) that both conditions share.
**A cleaner real-scene test, not yet run**: a condition with enough total
views to reconstruct well (e.g. 30-50), varying only how tightly those
views cluster in angle — isolating the spread variable the way the toy
scene's construction did (matched rod-cluster geometry, only arc width
varied), rather than conflating it with a shared view-count shortage.

All five GIFs saved under `gs_experiment/results/`.

## 37. Correction: §34-36's real-geometry results are not a null, they're inconclusive

Directly challenged, correctly: "the bonsai reconstruction is awful, so
nothing here is meaningful." Right, and it applies to lego too, not just
bonsai — §36 already noted the poor reconstruction quality as a
*confound* on the interpretation, but still described the underlying
result as "the null is confirmed, not contradicted." That's the wrong
framing, and worth being precise about why, not just softening the
language.

A BQ posterior variance is a statement about how well a *given* set of
splats, at their *actual* fitted positions/colors, constrains a query —
it says nothing trustworthy if those splats themselves are a bad fit to
the real scene. §36's own screenshots showed heavy floater/artifact noise
in every real-scene reconstruction (lego's 6-view conditions, bonsai's
8-view conditions) — splats scattered through free space, fitting noise
rather than real geometry. A BQ variance number computed on top of that
isn't measuring "how well does viewing-angle coverage constrain this real
surface point" — the *positions and colors feeding the query are
themselves close to garbage*. Whether the resulting variance number comes
out flat, noisy, high, or low is close to uninformative either way: a
flat, saturated result (lego) doesn't confirm "no directional effect on
real geometry" any more than a noisy, patchy one (bonsai) would have
confirmed the opposite. **Both are consistent with "the input was too
degraded for this measurement to mean anything," which is a different,
weaker claim than either a positive or a negative result.**

**What this changes, precisely**: §34's "the effect does not transfer,"
§35's "the null replicates on a stronger test," and §36's "the null is
confirmed, not contradicted" should all be read as **inconclusive**, not
negative — real attempts, with real diagnostic work that ruled out two
genuine artifacts (window radius, query point) before reaching this
point, but built on reconstructions too degraded to license a conclusion
either way. This is not a small caveat added after the fact — it's a
retraction of the confidence level those sections claimed, prompted by
direct scrutiny of the same visual evidence §36 itself introduced to be
more trustworthy than the point samples, not less.

**What a real test needs, and doesn't have yet**: enough total views for
the checkpoint itself to be a reasonable reconstruction (real PSNR in a
normal range for the scene, not a floater-dominated fit) — likely 30-50+
real views per condition, not 6-8 — while still varying only the angular
*spread* of those views between conditions, the way the toy scene's
construction varied only arc width while holding rod-cluster geometry
identical. Every real-scene attempt so far conflated "few views" with
"narrow spread" because holding view *count* fixed (needed for a clean
spread-only comparison) forced picking a small shared count. The honest
status of the real-geometry question, right now: **untested**, not
"tested and negative."

## Bottom line for real-benchmark validation

Getting onto a real, standardized benchmark surfaced a real bug (RGBA
compositing) and a real dataset-quality issue (an incomplete public
mirror) before anything downstream could be trusted -- consistent with
this project's pattern of real friction at every step of leaving
synthetic/toy scenes, not a smooth validation exercise. The central
"uncertainty nearly for free" claim now has direct, strong, real-data
support that doesn't route through a geometric-fineness proxy: BQ
variance correlates strongly with local splat sparsity (§24, r=-0.74,
p=8e-27) and responds specifically to genuine angular coverage gaps
rather than raw view count (§25, 2.75x from clustering alone, count held
fixed) -- both closed-form, both computed from nothing but each
checkpoint's own splat positions. The directional/observation-count
cross-checkpoint claim from §22 also replicates cleanly and more strongly
than at toy scale (4.54x vs. 2.46x-18.7x). The harder, more specific
same-checkpoint thin-vs-thick claim (§23) remains an open question with
concrete next steps identified, not a failure being hidden -- but it's no
longer the load-bearing claim for what this project's real-data validation
needs to show.
