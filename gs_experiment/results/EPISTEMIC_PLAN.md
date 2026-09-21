# Pre-registered protocol: epistemic uncertainty in coverage gaps, at scale

Written and committed **before** any run, in the same discipline as the
evaluation protocol of [`FINDINGS.md`](FINDINGS.md) section 12. Everything
below -- metrics, admission rule, arms, budget -- is frozen at this commit.

## The question

Does post-hoc quadrature uncertainty predict error in the regions the
training views failed to determine, (a) better than uncertainty that has to
be trained in, (b) better than free geometry, and (c) at what fraction of
the cost?

Sections 21-23 of FINDINGS answer this at n=3 real scenes, one hold-out
severity, one baseline. That is a promising result, not a headline. This is
the experiment that makes it one, or kills it.

## 0. Pilot gate -- PASSED

The riskiest assumption is not that we beat U-3DGS. It is that we beat the
**free** baseline: angular distance to the nearest training view, a k-d tree
and no model at all. FINDINGS section 21 found isolation alone ranks views at
Spearman 0.767 against our 0.716. If a reviewer can reproduce the headline
without a model, there is no headline.

The resolution is that ranking correlation is the wrong question. Score the
decision instead: the agent may revisit `K` of the `N` unvisited poses and
picks the `K` it distrusts most; its value is the error it thereby catches,
normalised so random = 0 and oracle = 1, averaged over all `K`.

| | farthest-point (free) | U-3DGS | ours |
|---|---|---|---|
| kitchen | 0.388 | 0.599 | **0.711** |
| counter | 0.472 | 0.823 | **0.928** |
| bonsai | 0.478 | 0.558 | **0.570** |
| **mean** | 0.446 | 0.660 | **0.736** |

Best on 3 of 3, and the free baseline is far behind. This also explains the
section 21 caveat rather than explaining it away: isolation ranks *isolation*
well and *chooses* badly, because the unvisited region is often simply easy.

Still n=3. The gate says the experiment is worth its compute, nothing more.

## 1. Design

**Axis 1 -- gap severity.** `train_fraction` in {0.9, 0.8, 0.7, 0.6, 0.5} via
`build_colmap_gap_scene.py --split_mode trajectory`. Results are reported
against the **measured** median angular isolation of each cell, not against
`train_fraction`: the map from fraction to isolation is scene-dependent, and
`room` (FINDINGS section 21) is the proof -- 30% held out, 2.2 degrees of
isolation, no gap at all.

**Axis 2 -- scenes.** The 13 COLMAP scenes already on disk and already run
end-to-end for section 23: Mip-NeRF 360 x9, Tanks & Temples x2, Deep
Blending x2. No new data required for the core sweep.

**Axis 3 -- methods.** All scored on identical checkpoints.

| arm | what it is | cost |
|---|---|---|
| ours | post-hoc, reads the finished checkpoint | 43 s |
| U-3DGS | their released `train_errors.py` | 2 min + a modified training |
| farthest-point | angular isolation, model-free | ~0 |
| deep ensemble | strongest baseline, FINDINGS section 16 | 3x a full training |
| random / oracle | scale anchors, 0 and 1 by construction | -- |

Ensemble runs on a 4-scene x 3-severity subset only; at 3x training per cell
it cannot be afforded everywhere, and that restriction is declared here
rather than discovered later.

## 2. Metrics, frozen

1. **PRIMARY -- selection efficiency**, as defined in the gate above. This is
   a decision metric with a fixed scale, not a correlation, and the scale
   anchors are in the table so a weak result cannot hide. Chosen *because*
   it is the one metric on which the free baseline is a serious opponent.
2. **AUSE-L1 and Pearson-L1** under U-3DGS's unmodified
   `uncertainty_metrics.py`, for continuity with sections 21-23. This is the
   within-view question, which is theirs to win when the scene is saturated.
3. **Cost** -- wall-clock and peak RSS of the uncertainty stage alone,
   excluding the training both arms share. "Nearly free" is a claim with a
   number in it and the number goes in every table.

Secondary, reported but not headline: isolation-vs-uncertainty Spearman
(section 21's metric), kept so the earlier result remains comparable.

## 3. Admission rule, pre-registered

A cell is **admitted** iff its median test-view angular isolation >= 5
degrees. Below that there is no coverage gap to measure and the cell tests
nothing.

Isolation is computed by `build_colmap_gap_scene.py` at construction time,
before any training, so the rule is blind to results. **Every constructed
cell is reported with its isolation, admitted or not**, in an appendix
table. The `room` exclusion in section 21 was correct but currently rests on
our say-so; this makes every such exclusion auditable.

## 4. The crossover claim, and what would falsify it

Sections 21 and 22 are currently two disconnected results: we lose 5/5 on
dense captures, we win 2/3 + 1 tie on trajectory hold-outs. The severity
sweep should show these are **one continuous curve** -- advantage against
measured isolation, crossing zero somewhere in between.

The headline figure is that curve: x = median angular isolation, y = AUSE
advantage (theirs minus ours), zero line marked, the dense-capture and
gapped regimes annotated at either end. One line, no explanation needed.

It is falsifiable in the way that matters: **if the advantage is flat in
isolation, the epistemic-coverage story is wrong** and what we have is a
method that happens to be tuned differently, not one that measures what the
training views failed to determine. That outcome gets reported.

## 5. Budget

Measured from the section 23 sweep: 21 min and 1.1-3.5 GB per cell (train
15-17, their fit 2, ours 0.7, scoring 0.3).

| | cells | GPU-hours |
|---|---|---|
| core sweep, 13 scenes x 5 severities | 65 | 23 |
| ensemble subset, 4 x 3 x 3 members | 36 | 9 |
| seed variance, 4 scenes x 3 seeds | 12 | 8 |
| **total** | | **~40, about 2 days** |

Three training seeds on a 4-scene subset size the run-to-run variance; that
variance becomes the error bar everywhere else, rather than every cell being
seeded, which the budget does not allow.

## 6. Disk is the binding constraint, not compute

14 GB free. 65 cells at 1.1-3.5 GB is 100+ GB if checkpoints are kept, so
the driver **deletes each cell's checkpoint after scoring** and keeps only
the metric JSON, the gap manifest and a fixed 3-view render strip for
figures -- a few MB per cell. Peak is then ~2 concurrent cells, about 7 GB.

This is the generalisation of the mistake already made once: 28 result JSONs
totalling 116 KB were living untracked on a full disk underneath 20 GB of
regenerable checkpoints. Durable and regenerable were on the wrong sides.

Reclaim available if needed: the full-resolution `images/` directories in
`mipnerf360_raw` total ~11 GB and are never read -- every run uses
`images_4`.

## 7. Risks

| risk | status |
|---|---|
| the free geometric baseline wins | **gate passed**, 0.446 vs 0.736 |
| the advantage curve is flat in isolation | open -- this is the real test |
| "you carved your own gaps" | open -- see below |
| n too small to fit a curve | 65 cells against the current 3 |

## 8. The one open decision

Every gap in this plan is carved by us from a photographer's orbit. The
strongest answer to "you built the hole you then found" is a dataset where
the trajectory is genuinely an agent's -- Replica (perfect poses, easy, but
synthetic) or ScanNet++ (real rooms, a published novel-view split, harder to
ingest). Either needs roughly 12 GB and a loader.

It is the single biggest credibility win available and it is not required
for the core sweep, so it is recorded here as a decision rather than an
assumption.
