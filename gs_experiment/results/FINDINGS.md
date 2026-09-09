# gs_experiment findings (summary)

Real Gaussian-Splatting results — real `gsplat` training, real checkpoints,
real cameras. This is the primary results document for the project.

This is a **current-conclusions** summary: every number below was produced
by the rendering-aware BQ engine (`LocalUncertaintyEngine.rendering_aware_variance*`,
`gs_experiment/quadrature.py`) that is now this repo's only production
uncertainty method — an earlier box-quadrature engine (uniform-domain
integration) was retired once this one replaced it everywhere; see git
history if the earlier numbers are ever needed for comparison.

## Session update: four real pipeline bugs found and fixed; two sections below revised

A push to build two paper figures (a cross-scene error-vs-uncertainty
gallery, and a pixel-wise view-coverage sweep) surfaced four real,
independent bugs in the pipeline every number in this document up to this
point was computed with -- found by directly interrogating unexpected
figure output rather than trusting it:

1. **Training background-color mismatch**: the lego gap-coverage
   checkpoints silently trained against a dark background while their
   images composite onto white, producing degenerate reconstructions.
2. **Missing `checkpoint_dir`** in the coverage-sweep figure script:
   camera attribution ran against the held-out eval split (untouched by
   the gap) instead of the actual gap-restricted training pool, making
   the directional signal blind to the coverage manipulation it was
   supposed to measure.
3. **Unfit, wildly inconsistent kernel hyperparameters**: `sigma` (RBF
   bandwidth) ranged from 0.05 to 0.9 and `kappa` (directional
   concentration) was hardcoded at 4.0 across different scripts, never
   checked against the data. Marginal-likelihood fitting
   (`hyperparams.py`, pooled across 9 real checkpoints) gives `sigma
   ~= 0.069`, `kappa ~= 0.745` -- the old `sigma=0.9` was off by enough
   to lose by **-35.8 million** in held-out log marginal likelihood (vs.
   -8607 at the fitted value); the old `kappa=4.0` was ~5x too
   concentrated (202 vs. 1413 held-out log marginal likelihood).
4. **Opacity-blind occlusion attribution**: `occlusion_mask` (which
   decides which training cameras get credit for observing each splat)
   had no opacity awareness -- a splat at opacity 0.001 hard-occluded
   real splats behind it exactly like one at opacity 1.0. GS training
   reliably leaves behind floaters (near-transparent stray splats that
   drift outside the intended training volume, and -- confirmed
   separately -- a *second*, opaque-but-badly-placed floater variant on
   some scenes) as a normal optimization artifact; these were spuriously
   "occluding" real, well-observed splats and starving them of camera
   attribution.

Bugs 1-2 are now fixed outright. Bug 3's fitted values are wired in as the
new defaults. Bug 4 is *partially* fixed: `attribute_observations` now
takes a `min_opacity` filter so low-opacity splats can't act as occluders
(this alone closed an 8x attribution gap between two nominally-identical
checkpoints of the same scene) -- but a related, unaddressed mechanism
remains open, see "Floaters" below.

Sections 2 and 3 below are revised accordingly. Section 1's sparsity
result is less affected (it uses the position-only, occlusion-independent
variance) but was computed at the old `sigma=0.05`, close to the now-fitted
value; not re-run.

### View-dependent coverage: uncertainty captures it cleanly (revises section 3)

With the checkpoint_dir bug fixed and hyperparameters properly fit, the
lego gap experiment (0/15/30/50/75deg gap half-widths, all 5 conditions
confirmed **completely floater-free**) now gives a dramatically cleaner
result than the scalar single-point-query version below: the pixel-wise
uncertainty ratio (`variance / prior_variance`, bounded [0,1]) for the
*same* held-out view, rendered against each condition's own checkpoint,
sweeps from **0.03 (gap 0deg) to 0.97 (gap 75deg)**, strictly increasing,
matching reconstruction quality's own monotonic decline (PSNR 35.1dB ->
16.1dB) — visually, a clean dark (well-informed) object silhouette at
gap 0 growing to a nearly solid bright (uninformed) one at gap 75. This is
now the single cleanest, most convincing result in the project.

### Floaters: a second, real, and distinct driver of uncertainty (new)

Independently of view coverage, the uncertainty ratio also responds
strongly and specifically to GS-training floaters — confirmed by direct
comparison of two checkpoints of the *same scene* (lego) with *identical*
real camera coverage (100 views) and splat count (300k): the checkpoint
with floaters (4% of splats displaced beyond the training bounds, near-
transparent) showed a mean uncertainty ratio 35x higher (0.36 vs. 0.01)
than the floater-free one, for the same nominal coverage.

The mechanism is not the occlusion bug above (already fixed) but a
*second*, still-open one: candidate selection for the local render weight
`a_q` is purely bearing-space (angular) with no real-3D-distance gate, so
a splat far along the same line of sight — different depth, same
bearing — can enter the candidate set with a real (opacity-weighted)
transmittance weight. Because variance is a second moment (`Sigma_q = sum
w_i (x_i-mu)(x_i-mu)^T`), a single such outlier contributes `~w*D^2`
to the covariance but only `~w*D` to the mean (and outliers in different
directions partially cancel in the mean but never in the covariance) — so
`a_q`'s spread inflates dramatically (measured: ~250-300x per axis, one
concrete case) while its center barely moves. That inflated spread then
weakens the kernel's *own* subsequent distance discounting (which runs at
scale `Sigma_q + sigma^2`, and `Sigma_q` now dominates `sigma^2`), so the
kernel ends up discounting distance at the corrupted scale instead of the
fitted one. A census across all 8 checkpoints found *two* floater
varieties, only one of which the current `min_opacity` occlusion fix
touches at all: near-transparent (opacity ~0.02-0.04, lego/hotdog/ship,
4-8% of splats) and, separately, **opaque-but-displaced** (opacity
0.7-0.92, chair/drums/ficus/mic, 1.8-**18%** of splats) — a real,
higher-opacity 3DGS failure mode distinct from the classic transparent
floater, unaffected by any opacity threshold.

Whether this is a bug or a feature depends on what the uncertainty is for
(see "How can uncertainty be used" in `ROADMAP.md`): it is *not* a
statement about camera-view-angle coverage (the mechanism above has
nothing to do with how many cameras observed anything), but it may be a
genuinely useful, independent signal for flagging unresolved training
artifacts — provided that use case is made explicit rather than the two
effects being silently conflated in one number.

### Rendering error: the old "no correlation" finding does not survive a clean re-test

Section 2 below reports "no correlation with squared error" as an
already-settled negative result, reached with a pipeline now known to
have the four bugs above. It does not survive a clean re-test.

Using the 5 floater-free lego gap checkpoints (fixed sigma=0.069,
kappa=0.745, correct attribution), real per-pixel `|error|` against the
uncertainty ratio, pooled over 6 held-out views per condition
(1,551,299 pixels total):

| condition | Pearson r | Spearman rho |
|---|---|---|
| gap 0deg (full coverage) | 0.048 | 0.158 |
| gap 15deg | 0.160 | 0.330 |
| gap 30deg | 0.390 | 0.434 |
| gap 50deg | 0.644 | 0.683 |
| gap 75deg | 0.628 | 0.682 |
| **pooled** | **0.692** | **0.560** |

All correlations significant at p < 1e-300 given the sample sizes
involved. This is a real, substantial, previously-hidden correlation, not
a rounding-error-scale effect — the earlier `|r|<0.13` conclusion was an
artifact of the buggy pipeline, not a property of the underlying
uncertainty construction.

The per-condition pattern is itself informative, not just the pooled
number: correlation is weak (though still real) under full coverage
(r=0.05) and climbs sharply as the gap widens, reaching r~0.63-0.64
*within* a single condition once coverage genuinely dominates. Read
together with the two mechanisms above, this is a coherent story: when
coverage is good, `|error|` is dominated by optimization/representation
limits (systematic edge/silhouette bias — see the epistemic-vs-error
discussion this session had) that the coverage-driven uncertainty has no
way to see, so the two are nearly independent; once coverage becomes the
dominant driver of *both* quantities, they move together strongly. The
"qualified pass, not a finished result" framing at the bottom of this
document is now itself out of date and needs revisiting once this result
is checked on more than one scene.

## 1. Does BQ variance track real sparse/missing coverage? (the headline claim)

Sample many query points across a real trained checkpoint, measure local
splat density (a KD-tree count) and rendering-aware BQ position-only
variance at each, and check whether they correlate — no geometric
classification, no synthetic scene needed.

**Yes, robustly, across all 8 standard NeRF-Synthetic scenes** (chair,
drums, ficus, hotdog, lego, materials, mic, ship — the complete standard
benchmark): Pearson `r` between **-0.296 and -0.560**, every scene
significant (`p<0.001`, lego the weakest at `p=2.3e-4`). Consistent sign
and real significance everywhere is the load-bearing part of this claim —
the exact magnitude is checkpoint-dependent (chair `r=-0.529`, drums
`r=-0.554`, ficus `r=-0.516`, hotdog `r=-0.453`, lego `r=-0.296`, materials
`r=-0.398`, mic `r=-0.405`, ship `r=-0.560`).

## 2. Is the signal calibrated, not just correlated?

**Superseded — see the session-update section at the top of this
document.** The finding below was this project's standing view for a
while and is kept for the record, but a clean re-test (once the pipeline
bugs listed above were fixed) found a real, substantial correlation
(`r=0.69` pooled) where this section reports none. Treat what follows as
a hypothesis that one implementation produced one discouraging number
for, not a settled conclusion — exactly the trap the session-update
section describes.

Leave-one-out cross-validation on real splat colors (hide a real splat
from its own local neighborhood, predict it from real neighbors alone,
compare to the real held-out value), across all 8 scenes.

Direct correlation
between variance and squared error is essentially zero everywhere
(`|r|<0.13` on every scene, mixed sign: chair `+0.007`, drums `-0.000`,
ficus `-0.078`, hotdog `+0.117`, lego `+0.058`, materials `-0.078`, mic
`-0.021`, ship `-0.128`). Held-out Gaussian NLL beats a flat,
constant-variance baseline on only 3 of 8 scenes (chair, hotdog, mic) and
is substantially worse on two (materials `6.11` vs. `1.59`; ship `10.12`
vs. `1.60`). A deeper look at one checkpoint (lego, 300 leave-one-out
points): ranking-based calibration (AUSE) is a real, if modest, positive
signal — BQ ordering (`0.346`) beats random ordering (`0.440`) against the
oracle sparsification curve — so the *relative ranking* carries some real
information even where the *absolute scale* clearly does not.

## 3. The directional / viewing-angle-coverage story

A second, related uncertainty signal: not "is this region under-resolved
by the current splats" but "is *this specific viewing angle* under-covered
by training views." The same product-kernel formalism produces both from
one posterior.

**On real geometry, with a coverage manipulation designed to not confound
with overall reconstruction quality: works cleanly.** Starting from the
full real 100-view lego pool and removing a single deliberate angular gap
around one query direction (leaving every other view untouched, so
density stays high everywhere except inside the gap itself) — held-out
PSNR stays tight (18.5→18.2→18.1dB) across gap half-widths 0→15→30→50°,
only softening at the most extreme 75° gap (16.0dB, expected — that
condition drops almost half the training pool), while PSNR measured
*only on eval views that actually fall inside each gap* is consistently
worse (8-10dB) than the overall number — evidence the manipulation
creates a real, localized coverage problem, not a global one. Against
that design: **directional BQ variance tracks the gap size cleanly** —
strictly monotonic across all 5 conditions (rank correlation `rho=1.000`),
from `0.0000023` at no-gap to `0.0792` at widest-gap (a `35060x` range),
while a position-only control stays far flatter (`2.41x`, no clean trend).

**A real bug was found and fixed getting to that result, worth recording.**
The directional query here has no real camera behind it (it asks "how
uncertain at this point, looking this direction," not "at this specific
training view"), so a synthetic stand-in camera is built from the
checkpoint's own real camera-distance scale
(`real_directional_coverage_experiment.synthetic_camera_for_query`).
Candidate selection for that synthetic query was initially ranking
candidates by bearing distance alone; since the input is a camera-expanded
observation array (one row per (splat, observing-camera) pair), many rows
share one physical splat's exact bearing, and an overflowing candidate cap
kept an arbitrary subset of those rows rather than the ones actually near
the query direction — a real architectural gap (confirmed: `angular_tol`
made no difference across a 10x range, ruling out a tuning fix), not a
hyperparameter issue. Fixed by adding directional-alignment tie-breaking
to `CameraSplatIndex.query`/`GsplatCameraProjection.query_pixel` whenever a
query direction is supplied — see their docstrings.

**Real geometry, genuinely photographed (Mip-NeRF360 "bonsai"): still
open.** The same gap design has been run once, on one condition of five,
with a quality review still in progress — see `ROADMAP.md`. Not yet a
result either way.

## 4. Training directly under the likelihood: one early attempt, not a settled conclusion

A natural direction — training with a Gaussian-NLL loss weighted by the
closed-form BQ variance itself, and/or using BQ variance (instead of the
standard view-space gradient) to trigger densification — was tried once
(`train_minimal_gsplat.py --nll-experiment`), under an implementation
from the same pre-fix era as the calibration finding in section 2, which
turned out not to survive a clean re-test. This result hasn't been
re-run since and should be read the same way: a hypothesis that got one
discouraging number once, not a closed question.

What was observed at the time: variance-driven densification produced
worse reconstruction and fewer splats than standard gradient-based
densification, with a specific candidate mechanism (BQ variance is high
in genuinely empty space too, and that run had no opacity floor guarding
against it, unlike the pruning use in section 6 below, which does) — and
the NLL loss term alone was close to a no-op. Whether either of those
holds up under the current, fixed pipeline (correct attribution, fitted
sigma/kappa) is untested. `ROADMAP.md` proposes a more targeted version of
this idea (regularizing on the render-weight spread specifically, aimed
at floaters) worth testing on its own terms rather than assumed to fail
for the same reason this earlier, broader, differently-implemented
attempt did.

## 5. A real methodological lesson: kernel parameters must match the checkpoint's actual scale

Caught more than once, in different forms: reusing a `window_radius`/
query-point convention that worked for one checkpoint on a
differently-scaled or differently-structured one produced a wrong-signed
or degenerate result until checked directly. Practical guidance: pick
`window_radius` well below the point where a typical query's window
captures a large fraction of the checkpoint's total splats.

## 6. Historical record: the designed hand-built-scene track

Before the real-checkpoint path above existed, an earlier hand-built-scene
track (thin-rod clusters, a next-best-view candidate pool, a
coverage-gradient camera arc) de-risked the core claim and tested two
downstream combinations. That code (`scene_spec.py`, `blender_render.py`,
`visibility_baseline.py`, `designed_scene_experiments.py`) is retired now
that the real-checkpoint path is the current, active one — see git
history to resurrect it. Kept here as historical record, not current best
evidence:

- **Differentiation (go/no-go)**: on two identical thin-rod clusters (one
  40-view ring, one 10-view arc), position-only BQ variance correctly
  flagged the well-observed-but-poorly-resolved cluster, replicated across
  two seeds and two kernel families, and survived three independent
  "maybe this is an artifact" checks. The mechanism itself was never fully
  settled (a real, still-open question at the time).
- **Pruning**: combining BQ position-only variance with the standard
  opacity-based pruning heuristic (floored at a minimum opacity) beat
  opacity-only pruning at a tight splat budget (+2.3dB) and was a strict
  no-op at looser budgets.
- **Next-best-view**: scoring candidate next-views by BQ position+direction
  variance plus a visibility proxy, then actually retraining with the
  top-scored candidate vs. a deliberately poor one, improved held-out PSNR
  ~3x more than the poor choice (+1.89dB vs. +0.65dB) — though this simple
  scene didn't give BQ and the visibility proxy room to disagree
  (correlation 1.000 on this candidate pool).

## Bottom line

The rendering-aware construction (`a_q = T_q sigma G_q` folded into the
kernel, in place of a uniform-box integration domain) is now this
project's only production uncertainty engine, validated directly against
real checkpoints rather than assumed to carry over from the earlier
box-quadrature engine's results. This section used to assert a specific
"qualified pass, not a finished result" verdict here (absolute
calibration open, ranking usefulness modest); that verdict was built on
section 2's pre-fix finding, which the session-update section at the top
of this document overturns. Rather than replace one premature verdict
with another, the honest summary right now is: the headline sparsity
signal (section 1) and the directional/coverage-gap signal (section 3,
now with a real per-pixel result) both hold up; the error-correlation
question has gone from "confirmed negative" to "confirmed positive on one
scene, not yet checked on others" (session-update section); and the
NLL-training question (section 4) has gone from "confirmed negative" to
"untested under the current pipeline." None of these should be read past
what they actually say until re-checked further — see `ROADMAP.md` for
what that further checking looks like.
