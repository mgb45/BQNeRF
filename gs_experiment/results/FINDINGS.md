# gs_experiment findings (summary)

Real Gaussian-Splatting results — real `gsplat` training, real checkpoints,
real cameras — building on the toy-scale validation in
[`bq_splat/results/FINDINGS.md`](../../bq_splat/results/FINDINGS.md).
This is the primary results document for the project.

This file is a short, **current-conclusions** summary, organized by real-
scene results first. The complete chronological account — every bug,
every intermediate number, the full reasoning behind each fix, and the
full multi-step correction history behind a couple of results below — is
preserved in [`ARCHIVE_FULL_LOG.md`](ARCHIVE_FULL_LOG.md); each section
links its matching archive section(s). Where a result was later corrected
or refined, this file states the current, final conclusion directly
rather than requiring a re-read of the whole history to find out what's
currently true.

## 1. Does BQ variance track real sparse/missing coverage? (the headline claim)

The most direct, minimal version of this project's central claim: sample
many query points across a real trained checkpoint, measure local splat
density (a trivial KD-tree count) and BQ position-only variance at each,
and check whether they correlate — no geometric classification, no
synthetic scene needed.

**Yes, strongly and robustly.** On the real NeRF-Synthetic "lego"
checkpoint: Pearson `r=-0.74` (`p=8e-27`), 3.49x higher variance in the
sparsest-20% vs. densest-20% regions. Extended to **all 8 standard
NeRF-Synthetic scenes** (chair, drums, ficus, hotdog, lego, materials,
mic, ship — the complete standard benchmark, not one scene picked for
convenience): `r` between **-0.74 and -0.97 on every single scene**, all
significant beyond any reasonable doubt — the strongest multi-scene
evidence gathered for this claim. It also responds specifically to
genuine *angular coverage gaps*, not just raw view count: five checkpoints
at 100/50/25/12 random-subset views plus 12 angularly-clustered views (all
queried identically) show variance essentially flat across 100→12 random
views, but 2.75x higher for the *same* 12-view count when those views are
angularly clustered instead of random — the signal isn't fooled by frame
count alone, which is exactly the property a real active-view-planning
policy needs. Checked to hold across a full order of magnitude of the
`window_radius` hyperparameter (0.2x-2x an established value, on every
checkpoint tested) — the claim doesn't depend on a lucky parameter pick,
though it does eventually degrade and flip sign at very large windows
(explained below, §6). *(Archive §22, §24, §25, §30, §32.)*

**Validated against a real reference trainer, not just this project's own
code.** Every result above used this project's own from-scratch trainer.
Re-running the same sparsity check on a checkpoint trained by `gsplat`'s
own official reference densification strategy (`gsplat.strategy.
DefaultStrategy` — code this project didn't write) gives `r=-0.92`,
matching sign and strength — direct evidence the finding isn't an
artifact of a simplified training loop. *(Archive §28.)*

## 2. Is the signal calibrated, not just correlated?

A stricter, different question: does a claimed "2x higher variance"
region actually have ~2x the squared prediction error, on genuinely
held-out data? Tested via leave-one-out cross-validation on real splat
colors (hide a real splat from its own local neighborhood, predict it
from real neighbors alone, compare to the real held-out value) across
three real checkpoints.

**A real, nuanced gap — not a clean pass.** Direct correlation between
variance and squared error is weak everywhere (`|r| < 0.21`, wrong-signed
on one checkpoint) — much weaker than the sparsity correlations above.
Ranking-based calibration (AUSE / sparsification curves) is more
encouraging, meaningfully beating random ordering on most checkpoints —
consistent with `pruning_experiment.py`'s already-positive ranking-based
result (§7 below). Held-out Gaussian NLL is **worse than a flat,
constant-variance baseline on every checkpoint tested** — the clearest
negative signal: the *absolute scale* of the variance isn't yet a
trustworthy per-point confidence value, even though its *relative
ranking* is informative. These are different claims: "BQ variance tracks
sparsity" (§1, robust) and "BQ variance is a calibrated absolute error
bound" (not yet supported) should not be conflated in a paper claim.
*(Archive §29.)*

## 3. Does kernel choice (RBF vs. Matérn-3/2) matter?

On real data, the two kernels agree closely on spatial *pattern*
(correlation 0.98-0.995 between their variance grids across every real
checkpoint tested) while differing by ~150x in absolute *scale* — expected,
since Matérn's rougher smoothness assumption shrinks posterior variance
less per observation than RBF's infinitely-smooth prior does. With
properly *fitted* (not arbitrary) bandwidths for each kernel, a real,
consistent trade-off emerges across three checkpoints: **RBF wins the
sparsity-correlation claim and absolute (NLL) calibration on every
checkpoint; Matérn wins ranking-based calibration on every checkpoint.**
No kernel dominates — a paper claiming one universally-superior kernel
family isn't supported by this data. Fitting the bandwidth itself (via
marginal likelihood, the same method `bq_splat/hyperparams.py` validated
at toy scale) matters: on lego it corrected `sigma` from a hardcoded
`0.05` to `0.062` and closed a real, substantial held-out-likelihood gap,
without materially changing the sparsity-correlation number itself
(`r=-0.61` either way) — real evidence the hardcoded bandwidth used
throughout this project was leaving real calibration on the table even
where it wasn't obviously *wrong*. Which kernel needs the larger
bandwidth correction turned out to be scene-specific, not a fixed rule
(opposite directions on lego vs. a thin-rod scene). *(Archive §8, §22,
§26, §31; toy-scale kernel-fitting foundations in `bq_splat/results/
FINDINGS.md`.)*

## 4. The directional / viewing-angle-coverage story

A second, related uncertainty signal: not "is this region under-resolved
by the current splats" but "is *this specific viewing angle* under-covered
by training views." The same product-kernel formalism produces both from
one posterior (see `bq_splat/PROOF_alpha_compositing_equivalence.md` §5
for why this is provably one theorem, not two mechanisms).

**On a scene designed to have a real, continuous coverage gradient: works
cleanly.** Five zones of identical, isolated, occlusion-free geometry,
each observed by its own camera arc, with angular half-width increasing
linearly from 8° to a full 180° ring — a real, monotonic gradient by
construction, real Blender rendering, real `gsplat` training. Directional
BQ variance recovers the gradient exactly: strictly monotonic across all
5 zones (rank correlation `rho=1.000`), a `12.97x` range, while a
geometry-matched position-only control stays far flatter (`1.76x`, no
trend) — confirming the effect is genuinely directional. A full per-pixel
animated sweep (not just 5 point samples) confirms this visually, with
real sub-structure the point samples missed. *(Archive §33, §36.)*

**On real geometry: works cleanly too, once the coverage manipulation
is designed to not confound with overall reconstruction quality.** Two
earlier attempts on real lego (subsampling the real 100-view pool into
equal-count, increasing-spread conditions, `real_directional_gradient_
experiment.py`) and on an actual photographed scene (Mip-NeRF360
"bonsai," 292 real photos, real COLMAP-estimated poses) ran into a real,
carefully-diagnosed confound: holding total view *count* fixed while
widening the angular window they're drawn from also thins local view
*density* everywhere, not just near the region being tested, so held-out
PSNR degraded as spread widened for reasons unrelated to the directional
signal being measured. This wasn't a training-budget artifact either —
retraining the worst condition with 2.5x the iterations barely moved
held-out PSNR (confirmed directly, see the archive), and a real
background-compositing bug in the PSNR check itself was also found and
fixed along the way (caught a good checkpoint measuring 1.8dB instead of
its real 27dB). **The fix was a different experimental design, not more
compute or more debugging of the old one:** `gap_directional_experiment.py`
starts from the full real 100-view lego pool and removes a single
deliberate angular gap around one query direction, leaving every other
view in the pool untouched — so density stays high everywhere except
inside the gap itself, and overall reconstruction quality shouldn't move
much with gap size. Real held-out PSNR confirms the design works as
intended: overall PSNR stays tight (18.5→18.2→18.1dB) across gap
half-widths 0→15→30→50°, only softening at the most extreme 75° gap
(16.0dB, expected — that condition drops almost half the training pool),
while PSNR measured *only on eval views that actually fall inside each
gap* is consistently and dramatically worse (8-10dB) than the overall
number — direct evidence the manipulation creates a real, localized
coverage problem rather than a global one. Against that properly-
controlled design: **directional BQ variance tracks the gap size
cleanly** — strictly monotonic across all 5 conditions (rank correlation
`rho=1.000`), a `6.95x` range from no-gap to widest-gap, while a
position-only control stays far flatter (`1.34x`).

**But looking at the actual renders (not just the PSNR average) surfaced
a real quality problem worth stating plainly, not glossing over: the
widest-gap checkpoint's held-out views are, for the most part, genuinely
unrecognizable** — soft color blobs, not a lego bulldozer — except for
the rare held-out view lucky enough to land near a surviving training
angle (`gs_experiment/results/gap4_heldout_reconstruction.png` vs. the
much healthier `gap0_heldout_reconstruction.png` baseline). The overall
PSNR average (16.0dB) hides a wide per-view spread that only became
visible by rendering and looking, exactly the failure mode item 2 of
`ROADMAP.md` exists to catch — caught here because a direct visual
question ("I have no idea what object is in those images") is a faster
and more honest check than trusting an averaged number, precisely the
shift `ROADMAP.md` commits to.

That raises the obvious follow-up question directly: is the elevated
directional variance on the widest-gap checkpoint tracking the actual
missing direction, or is it just picking up general floater noise from
an overall-broken reconstruction? Checked directly, not assumed: querying
the *same* widest-gap checkpoint at the gap's own direction gives
`14.78`, more than double the `6.98` from querying the opposite
(well-covered) direction on that identical checkpoint — a real,
direction-specific contrast within one checkpoint, not just "this
checkpoint is bad everywhere." The clean baseline checkpoint (no gap at
all) shows no comparable spike at that same direction (`2.13` vs. `8.27`
at the two directions — if anything the untouched, naturally-covered
direction already runs higher, a real asymmetry in NeRF-Synthetic's own
camera distribution worth remembering when picking a "control" direction
in future runs). So the signal does appear to be tracking something real
and localized, on top of — not merely because of — the checkpoint's
general messiness.

**Honest overall status: a real, positive, and now more carefully
checked result, but resting on checkpoints whose absolute reconstruction
quality is genuinely poor by this project's own >20dB bar** (every one of
the 5 gap conditions falls short of it, some — the widest gap especially
— by a lot). The within-checkpoint contrast check is real supporting
evidence, not a full substitute for reconstructions clean enough to trust
outright; a larger real view pool (lego's is capped at 100 total views)
or a genuinely denser real capture would be needed to clear that bar
properly. The same design hasn't been run on the photographed bonsai
scene yet — the natural next step, and one where reconstruction quality
will need checking with the same skepticism from the start, not after a
GIF prompts the question. *(Archive §34, §35, §36, §37, §38 for the full
confound-diagnosis history; the gap-based result above, with this
follow-up, is the current, most-trustworthy picture on real geometry.)*

## 5. What didn't work: training directly under the likelihood

A natural next step — training with a Gaussian-NLL loss weighted by the
closed-form BQ variance itself, and/or using BQ variance (instead of the
standard view-space gradient) to trigger densification — was tried
directly rather than assumed to help. **It didn't, on either count.**
Variance-driven densification is a real regression (worse reconstruction
*and* fewer splats than standard gradient-based densification, not a
favorable trade-off) for a specific, diagnosed reason: BQ variance is
high in genuinely empty space too, and — unlike the pruning experiment
below, which floors the signal by opacity — the densification version had
no such floor. Adding one confirmed the mechanism but didn't fix the
regression (it overcorrected into uncontrolled growth instead), pointing
to a more structural mismatch between BQ variance and a densification
scheme built around gradient-magnitude signals. The NLL loss term alone
was close to a no-op, plausibly because of how sparse a signal this
installment's construction provided, not necessarily because the
underlying idea is wrong. Reported as a genuine negative result, not
softened. *(Archive §27.)*

## 6. A real methodological lesson: kernel parameters must match the checkpoint's actual scale

Caught twice, independently, in different forms: reusing a
`window_radius`/query-point convention that worked for one checkpoint on
a differently-scaled or differently-structured one produced a
wrong-signed or degenerate result until checked directly. A full sweep
(§1 above) shows this precisely: the sparsity correlation is strongly
negative and robust across 0.2x-2x a sensible window size, then degrades
and flips sign past ~4x — a general, predictable pattern across every
checkpoint tested, not a one-off artifact. Practical guidance: pick
`window_radius` well below the point where a typical query's window
captures a large fraction of the checkpoint's total splats. *(Archive
§28, §30, §34, §35.)*

## 7. Downstream combination experiments: pruning and next-best-view

**Pruning**: combining BQ position-only variance with the standard
opacity-based pruning heuristic (floored at a minimum opacity, since BQ
variance is uninformatively high in empty space otherwise) beats
opacity-only pruning at a tight splat budget (+2.3dB) and is a strict
no-op (never worse) at looser budgets where opacity-only already retains
everything with real content. *(Archive §15-16.)*

**Next-best-view**: scoring a discrete pool of candidate next-views by BQ
position+direction variance plus a visibility proxy (both free, no
retraining needed) and actually retraining with the top-scored candidate
vs. a deliberately poor one shows the guided pick improves held-out PSNR
~3x more than the poor choice (+1.89dB vs. +0.65dB). Caveat: this simple
scene didn't give BQ and the visibility proxy room to disagree
(correlation 1.000 on this candidate pool), so it demonstrates "guided
beats poor," not yet the more specific "the two signals combined beat
either alone" — a scene designed so they can diverge is the natural next
step. *(Archive §17-19.)*

## 8. Foundational engineering: the original hand-built differentiation scene

Before any of the real-benchmark work above, the core go/no-go question
(can BQ variance flag a region that's well-*observed* but poorly
*resolved*, something a visibility-only signal structurally can't see)
was tested on a hand-built scene: two identical thin-rod clusters, one
shot from a 40-view ring, one from a 10-view arc. Getting from "the
pipeline runs" to "the numbers are trustworthy" surfaced four real bugs
(an occlusion-attribution default two orders of magnitude too aggressive
for dense real geometry; a query-direction construction that silently
broke when two camera rigs shared an elevation; an uncapped local-
neighbor count that pegged 18 CPU cores for half an hour; a scale-
initialization bug that produced a blank reconstruction behind a
deceptively reasonable PSNR) — each is a real, generally-useful lesson
about training/evaluating Gaussian Splatting checkpoints, kept in the
archive in full. With those fixed and real densification added (view-
space-gradient-triggered clone/split, calibrated against measured
gradient magnitudes rather than a borrowed constant), the core claim was
demonstrated and replicated across two seeds and two kernel families,
then survived three independent "maybe this is an artifact" checks
(clone-position adjacency, camera-count leaking into the signal, a
controlled declustering isolation test that directly refuted the leading
hypothesis for the mechanism) — the mechanism itself remains genuinely
open, a real result whose *why* isn't yet settled. This scene was later
superseded, for headline claims, by the real-benchmark results in §1-4
above — kept here as the foundational validation that de-risked the real-
benchmark work, not as the current best evidence. *(Archive §1-14; the
full "Bottom line for the go/no-go gate" summary is preserved there too.)*

## 9. Getting real benchmark data into the pipeline

Adapting the standard NeRF-Synthetic "lego" benchmark (100 real training
views + an official held-out test split) surfaced a real RGBA-compositing
bug (naively dropping the alpha channel instead of compositing onto a
background silently corrupts ground truth) and an incomplete public
dataset mirror (a `transforms_test.json` listing 200 frames with only 36
shipping real color images) — both caught by checking file existence and
pixel content directly, not assumed correct. Reconstruction quality was
verified on genuine held-out test views (27.17dB wide / 19.80dB narrow —
the expected sparse-view generalization gap) before trusting any
uncertainty number built on top of either checkpoint. *(Archive §20-21.)*

## Bottom line

The central "uncertainty nearly for free" claim — that recognizing
rendering as Bayesian quadrature gives a real, closed-form uncertainty
signal from the same kernel structure already used to represent the
scene — has real, multi-scene, cross-trainer support for the *sparsity/
coverage-tracking* version of the claim (§1), robust across all 8
standard benchmark scenes and independent of this project's own training
code. Calibration in an absolute sense is not yet established (§2) —
ranking-based uses (like pruning) are on firmer ground than literal
confidence values. Kernel choice is a real trade-off, not a solved
question (§3). The directional/viewing-angle-coverage half of the
unification claim works cleanly on designed geometry, and — after
diagnosing and designing out a real reconstruction-quality confound, then
a second round of scrutiny prompted by directly looking at the renders
and asking "what object is this even supposed to be" — now also on real
geometry (§4), with a real, direction-specific within-checkpoint contrast
as supporting evidence, though still resting on checkpoints below this
project's own quality bar. One genuine negative result along the way
(§5) and one real methodological lesson that generalizes across most of
the above (§6). See `ARCHIVE_FULL_LOG.md` for the complete process,
including everything that didn't make this summary.
