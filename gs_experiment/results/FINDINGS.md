# gs_experiment findings (summary)

Real Gaussian-Splatting results — real `gsplat` training, real checkpoints,
real cameras. This is the primary results document for the project.

This is a **current-conclusions** summary: every number below was produced
by the rendering-aware BQ engine (`LocalUncertaintyEngine.rendering_aware_variance*`,
`gs_experiment/quadrature.py`) that is now this repo's only production
uncertainty method — an earlier box-quadrature engine (uniform-domain
integration) was retired once this one replaced it everywhere; see git
history if the earlier numbers are ever needed for comparison.

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

Leave-one-out cross-validation on real splat colors (hide a real splat
from its own local neighborhood, predict it from real neighbors alone,
compare to the real held-out value), across all 8 scenes.

**No, not yet — a real, honestly negative result.** Direct correlation
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

## 4. What didn't work: training directly under the likelihood

A natural next step — training with a Gaussian-NLL loss weighted by the
closed-form BQ variance itself, and/or using BQ variance (instead of the
standard view-space gradient) to trigger densification — was tried
directly rather than assumed to help (`train_minimal_gsplat.py
--nll-experiment`). **It didn't, on either count.** Variance-driven
densification was a real regression (worse reconstruction *and* fewer
splats than standard gradient-based densification) for a specific,
diagnosed reason: BQ variance is high in genuinely empty space too, and
the densification version had no opacity floor to guard against that
(unlike the pruning use in section 6 below, which does). The NLL loss
term alone was close to a no-op. Reported as a genuine negative result,
not softened. (This experiment's engine call was carried over to the
rendering-aware method along with everything else in this project, but
the negative result itself hasn't been independently re-run since —
the diagnosed mechanism, BQ variance being uninformatively high in empty
space, doesn't depend on which quadrature construction computes that
variance, so there's no specific reason to expect it would reverse.)

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
box-quadrature engine's results. What holds up: the headline sparsity
signal, real and significant on every one of the 8 standard benchmark
scenes, and the directional/coverage-gap signal, cleanly monotonic on a
real, carefully-controlled lego experiment. What's a real, open gap:
absolute calibration — held-out likelihood beats a flat baseline on only
3 of 8 scenes, and ranking-based usefulness, while real, is modest. This
is a qualified pass, not a finished result: useful for ranking regions
(pruning, active-view selection), not yet trustworthy as an absolute
confidence value, and honestly reported that way rather than rounded to a
cleaner story.
