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

## 1a. AMENDMENT, after construction and before any gap training

Building the hold-outs revealed that the admission rule rejects four of the
thirteen scenes -- `room`, `truck`, `train`, `drjohnson` -- and that halving
their training set does not rescue them (FINDINGS section 26). They are
multi-loop captures: any contiguous PREFIX has already covered every viewing
direction, so a gap in them is not a moment in time but a DIRECTION.

**The construction rule becomes a function of the capture, applied uniformly
to all thirteen**: a trajectory prefix where that admits, otherwise a 30
degree arc, otherwise excluded. An incomplete loop -- the agent never went
round that side -- is the most ordinary partial capture there is, and it
gives the four rejected scenes 25.5 to 31.8 degrees of median isolation,
the same range as the trajectory cells (kitchen 24.2, stump 22.1). The gap
row then covers every benchmark scene with no exclusions at all.

**This is an amendment to a pre-registration and is flagged as one.**
What protects it: the isolation that drives it is computed before any
training and is blind to every result; 30 degrees is the *smallest* width
tested that admits all four, not a tuned one; and the choice was fixed
before a single gap checkpoint existed. What it costs: the gap row now
contains two constructions, and the tables say per scene which one was used
rather than averaging over the distinction.

At 45 degrees `room` blows up to 137 degrees of isolation -- held-out views
never observed from any direction at all -- which is why the width is not
simply made large enough to be safe. Section 21 rejected the arc
construction on exactly that pathology; at 30 degrees it does not occur.

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

**Candidate pool: the capture's own photographs, one COLMAP solve, fixed
poses.** The alternative -- re-running COLMAP inside the loop as the agent
acquires, at a real-time splat budget -- is more faithful and would damage
the experiment, which is a different objection from being expensive.

Experiment B persuades for exactly one reason: the arms differ *only* in
which views they chose, and are then given identical treatment. COLMAP in
the loop breaks that. Each arm ends with a different SfM solution -- its own
coordinate frame, its own pose errors, its own registration failures -- so
the full-capacity retrains are no longer comparable, the held-out test set
has to be re-registered per arm, and registration failure is itself a
function of which views were chosen. The number at the end would then mix
"which strategy reduces render error" with "which strategy makes COLMAP
happy", and no amount of seeds separates them. We claim the uncertainty
picks useful views; we do not claim to have built a SLAM system, and the
harder experiment tests the claim we are not making.

What is worth taking from the SLAM framing, because it costs nothing:

* **A real-time splat budget in the loop.** The in-loop map is deliberately
  cheap -- capped iterations and splat count -- and only the final
  evaluation is full capacity. This is the cost argument made concrete: a
  method needing its own training run cannot be in this loop at all.
* **A trajectory constraint on acquisition.** At each step the candidate set
  is restricted to poses reachable from the current one, so an arm builds a
  *path* rather than a set. A real agent cannot teleport, and this is where
  unconstrained pool-based acquisition is genuinely unrealistic rather than
  merely idealised. It also makes the task harder and more discriminating,
  since a greedy grab at the single most uncertain pose may be unreachable.
* **Pose noise as an ablation, not SfM.** To test robustness to imperfect
  localisation, perturb the poses at the magnitude COLMAP would leave. That
  isolates the effect under control, where running COLMAP confounds it with
  registration failure.

Fixed poses from a single solve is also the field's convention for
acquisition experiments (ActiveNeRF, FisherRF), so it keeps us comparable.
COLMAP-in-the-loop is recorded as future work, and as the thing that would
be required if the claim ever became "a mapping system" rather than "an
uncertainty that picks views".

**What this does not show**, stated so it is not discovered later: nothing
about localisation, loop closure, or drift. The agent is choosing where to
look, not working out where it is.

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

## 5. Amendment: capacity withheld, and a prediction registered against us

Views withheld is one way to remove information. **Capacity withheld** is the
other, and it is a sharper test because it separates two things the paper has
so far only asserted are different.

Same scenes, same views, same train/test split. The only change is the splat
budget, controlled through `--densify_grad_threshold` (raising it produces
progressively fewer splats). Error rises as capacity falls; the question is
what our uncertainty does.

### The prediction, registered before the run

**Our sigma should FALL as splats are removed, while error RISES.** This is
not a hedge written after seeing a bad result -- it follows directly from the
construction. The per-splat posterior precision is

    P_i = Lambda + D_i / sigma_n^2,   D_i = sum_q beta_{q,i}^2

and `D_i` accumulates the squared compositing weight over every pixel the
splat touches. Fewer splats means each surviving splat covers more pixels,
so `D_i` grows, precision grows, and sigma falls. Meanwhile the
representation can express less, so error grows.

If that is what happens, the two curves move in opposite directions and any
correlation across capacity levels is negative.

### Why registering a prediction against ourselves is the point

The claim this project makes is narrow and specific: the posterior estimates
**what the training views failed to determine**, not what the representation
cannot express. Those are different quantities, and section 5 of FINDINGS
already says so. A capacity sweep is the experiment that makes the
distinction falsifiable rather than rhetorical.

There are three outcomes and all three are informative:

* **sigma falls while error rises** -- the prediction. The method does not
  see representational inadequacy, exactly as claimed, and the claim is now
  demonstrated rather than asserted. This also gives ROADMAP item 5
  (`u_spatial_BQ`, the finite-representation term, currently orphaned) a
  concrete job: it answers "is the node set adequate?", which is precisely
  the question capacity withheld asks and the coefficient posterior cannot.
* **sigma rises with error** -- the prediction is wrong and the separation
  between epistemic and representational uncertainty is not as clean as
  claimed. That would need reporting prominently.
* **sigma is flat** -- the construction is insensitive to capacity in either
  direction, which is weaker than the first outcome but still consistent
  with the claim.

What must NOT happen is reporting only a within-capacity-level correlation
and quietly omitting the across-level one. Both are recorded.

### Design

* scenes: `bonsai` (indoor), `garden` (outdoor), `playroom` (Deep Blending)
* capacity levels: `densify_grad_threshold` in {0.0002 (default), 0.0008,
  0.0032, 0.0128}, giving four budgets per scene
* held-out views and split: **unchanged** from the standard benchmark, so
  the only varying quantity is capacity
* recorded per level: splat count, PSNR, per-view and per-pixel error, our
  sigma, theirs, and AUSE under their scorer
* 12 cells at ~25 min, about 5 GPU-hours

Reported as two separate numbers, not one: the correlation **within** each
capacity level (does sigma find the error at a fixed budget?) and the
relationship **across** levels (does sigma track the error that removing
capacity caused?). Conflating them is how this experiment would be made to
look better than it is.
