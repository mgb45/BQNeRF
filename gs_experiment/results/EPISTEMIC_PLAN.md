# Pre-registered protocol: epistemic uncertainty in gaps

Written and committed **before** any run, in the same discipline as the
evaluation protocol of [`FINDINGS.md`](FINDINGS.md) section 12.

**Two experiments, deliberately kept apart.** The first draft of this plan
crossed severity x scene x arm x metric into one factorial. That was the
mistake: every extra axis is another thing a reader must be walked through
before the result lands, and a protocol that needs teaching does not get
believed. Each experiment below has one idea in it.

## 0. Pilot gate -- PASSED

The riskiest assumption is not that we beat U-3DGS. It is that we beat the
**free** baseline: angular distance to the nearest training view, a k-d tree
and no model. FINDINGS section 21 has that baseline *ahead* of us on rank
correlation, 0.767 against our 0.716. If a reviewer reproduces the headline
without a model, there is no headline.

Rank correlation is the wrong question. Score the decision: the agent may
revisit `K` of the `N` unvisited poses and picks the `K` it distrusts most;
its value is the error it thereby catches, on a scale where random = 0 and
oracle = 1, averaged over `K`.

| | farthest-point (free) | U-3DGS | ours |
|---|---|---|---|
| kitchen | 0.388 | 0.599 | **0.711** |
| counter | 0.472 | 0.823 | **0.928** |
| bonsai | 0.478 | 0.558 | **0.570** |
| **mean** | 0.446 | 0.660 | **0.736** |

Best on 3 of 3, and the free baseline is nowhere. This explains the section
21 caveat rather than explaining it away: isolation ranks *isolation* well
and *chooses* badly, because the unvisited region is often simply easy.

CPU-only, off artefacts already on disk. n=3; the gate says the experiment
is worth its compute, nothing more.

## 1. Experiment A -- the 2x2

Two axes, both binary.

* **regime**: dense capture / coverage gap
* **question**: within-view (where in this image is the error?) / per-view
  (should I trust this pose at all?)

| | within-view -- their AUSE-L1 | per-view -- selection efficiency |
|---|---|---|
| dense capture | U-3DGS wins 5/5 (s22) | ? |
| coverage gap | ours, 2 wins + 1 tie (s21) | ours, 3/3 (gate above) |

Three cells already have evidence; one is empty. Fill all four on the same
scenes, same checkpoints, same scorer. The claim then reads straight off a
four-cell table: **within-view and per-view uncertainty are different
quantities, and the gap column is ours.** Nothing to explain beyond the two
words on each axis.

**Scale.** The 13 COLMAP scenes already on disk, one gap construction each
at `train_fraction = 0.7` -- the setting section 21 already used, not a
sweep. About 13 cells, ~5 GPU-hours, no new data.

**Dropped from the first draft, deliberately**: the 5-point severity sweep,
the crossover curve, the seeded subset, the deep-ensemble arm, the admission
appendix. Each was defensible and each cost a paragraph of explanation.

**Kept, because removing them would make the result attackable**: the
free farthest-point baseline; the isolation >= 5 degree admission rule (one
sentence, and `room` genuinely needs it -- 30% held out, 2.2 degrees of
isolation, no gap at all); and the cost column in seconds, since "nearly
free" is a claim with a number in it.

## 2. Experiment B -- the SLAM-like loop

The combined result, and the one that needs no protocol explained at all:
an agent does the task and the picture shows whether it worked.

1. Fit a deliberately **cheap** map on a short prefix of the trajectory --
   few splats, few iterations. This is the whole point of an uncertainty
   that is post-hoc and nearly free: you can afford it at every step of a
   loop, which a method requiring its own training run cannot.
2. **Loop**: score the candidate poses, acquire the best one, refit cheaply,
   repeat.
3. **Evaluate by retraining at full capacity** on the acquired view set and
   scoring on a fixed held-out test set.

Step 3 is what makes it honest. It separates "did the cheap uncertainty
choose good views?" from "was the cheap map any good?" -- the arms differ
only in which views they chose, and are then all given the same generous
budget to make use of them.

**Arms**: ours, farthest-point (free), random, and U-3DGS if its cost in the
loop is tolerable. **Three seeds on random**, non-negotiable: a discarded
run in this project once had a random arm go 11.56 -> 10.84 -> 10.90 dB.

**Candidate pool**: the unvisited photographs of the capture itself, so
ground truth exists at every acquirable pose and no renderer or simulator is
needed. Pool-based acquisition, which is also what makes it reproducible by
anyone holding the same public scene.

**Figure**: PSNR of the full-capacity retrain against number of views
acquired. Three lines, one of them ours and above. No caption required.

## 3. Budget and disk

Measured from the section 23 sweep: 21 min and 1.1-3.5 GB per trained cell.

| | GPU-hours |
|---|---|
| A: 13 gap cells | ~5 |
| B: cheap loop steps + 4 arms x full retrains x acquisition sizes | ~10 |
| **total** | **~15, under a day** |

14 GB free, so the driver **deletes each checkpoint after scoring** and
keeps the metric JSON, the gap manifest and a fixed render strip -- a few MB
per cell. This is the generalisation of a mistake already made here: 28
result JSONs totalling 116 KB were living untracked on a full disk beneath
20 GB of regenerable checkpoints. Reclaim in reserve: the full-resolution
`images/` in `mipnerf360_raw` total ~11 GB and are never read, since every
run trains on `images_4`.

## 4. What would falsify each

* **A**: if we lose the per-view gap cell, or the free baseline closes on
  us at n=13, the epistemic claim does not survive and gets reported as
  such.
* **B**: if the full-capacity retrain shows no gap between our acquisition
  curve and random, the signal is not *useful* whatever it correlates with.
  This is the harsher of the two tests and it is the one a reader will
  believe, because nothing about it depends on a metric we chose.
