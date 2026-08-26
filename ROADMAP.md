# Roadmap

For the theory, see [`README.md`](README.md). For results, see each
package's `results/FINDINGS.md`. This document is the forward plan.

## The shift: optimize for working, general code — not exhaustive proof

The last phase of this project produced a lot of small, statistically-
elaborate results (point-sample correlations, leave-one-out calibration
curves, kernel/window ablations on tiny synthetic windows) that took
several rounds of after-the-fact debugging each time a number looked odd
— wrong-scaled window radius, a query point sitting in empty space, a
reconstruction too poor to trust either way. Each of those was real
signal, eventually, but the debugging loop was slow and the resulting
evidence was still thin: a handful of numbers from an under-resourced
checkpoint.

**Going forward, the priority is a real, general, working system,
reached as fast as possible** — not incremental statistical rigor toward
a paper. Two concrete changes to how this project validates itself:

1. **Train to genuinely good quality, always.** No more "lighter budget"
   shortcuts or 6-view toy conditions that leave every result confounded
   by reconstruction quality. If a checkpoint isn't good, nothing
   measured on it is trustworthy — so make "good" the floor, not a thing
   checked after the fact.
2. **Render it and look, before reaching for statistics.** A full
   per-pixel uncertainty sweep rendered over an orbit, viewed directly (by
   eye, or by asking for feedback), answers "does this signal behave
   sensibly?" faster and more convincingly than a correlation coefficient
   computed on a handful of query points — and it doesn't hide a bad
   reconstruction the way a summary statistic can. Point-sample
   statistical tests aren't gone for good, but they're no longer the
   first or only thing reached for; they're a follow-up once a rendered
   result already looks right.

## Active work

Short list, in order.

### 1. One clean, general tool, not a pile of one-off scripts

`gs_experiment/` is ~40 scripts, many hardcoded to a specific scene/path
from whatever question was being answered that day. Consolidate the
actual reusable path — load any real `gsplat` checkpoint, compute BQ
quadrature and directional variance, render it — into one general
CLI/API (`LocalUncertaintyEngine` + `render_directional_uncertainty_sweep.py`
are already most of the way there). Should work on an arbitrary
checkpoint without per-scene tuning.

### 2. Make "reconstruction is good enough to trust" a gate, not a footnote

`render_reconstruction.py` and the held-out-PSNR check already exist;
wire them into the tool itself as a mandatory first step (refuse or
loudly flag any uncertainty computation on a checkpoint that hasn't
passed it), instead of a manual step that's easy to skip under time
pressure — which is exactly how the bonsai/lego confounds happened.

### 3. Settle the real-geometry directional question by rendering it well

The open question from the old plan (does directional BQ variance track
real viewing-angle coverage gaps on a real, well-reconstructed scene?)
gets one more real attempt — full training budget, competitive quality,
gated by item 2 — and the answer comes from watching
`render_directional_uncertainty_sweep.py`'s output, not from a five-point
correlation on an under-resourced checkpoint. If it looks right, that's
the result. If it doesn't, render more views/angles to see where it
breaks, rather than re-running the same small statistical test with
different parameters.

### 4. One default kernel and bandwidth, not continued ablation

RBF vs. Matérn and window-radius sensitivity are already characterized
(see `FINDINGS.md`) — pick the sensible general default (RBF, fitted
bandwidth) and move on. Revisit only if a real use case actually needs
the other trade-off.

## Parked (not active — real, but not on the path to working general code)

Kept for the record, not deleted, and worth returning to if this becomes
a paper later: formal calibration statistics beyond a basic sanity check,
multi-seed significance runs, matched-baseline reproduction of PUP 3D-GS
and GAVIS's exact protocol, SLAM/incremental-mapping integration,
adversarial robustness/failure-mode stress testing, a formal runtime-cost
benchmark table, the literature-review pass, and a paper write-up plan.
None of these make the code more real, more general, or faster to trust
— they make a specific published comparison claim, which isn't the
current goal.

## What's already solid (see FINDINGS.md for numbers)

- The core math is proven, not just checked: BQ posterior mean = alpha
  compositing exactly, with a provable (not heuristic) error bound on the
  variance (`bq_splat/PROOF_alpha_compositing_equivalence.md`).
- The sparsity-uncertainty correlation replicates strongly across all 8
  standard NeRF-Synthetic scenes and against a real reference `gsplat`
  trainer, not just this project's own training loop.
- Kernel hyperparameters can be fit from data instead of hardcoded, at
  real checkpoint scale.
- Training directly under the BQ likelihood (loss term, densification
  trigger) was tried and didn't help — a real negative result, kept.
- The directional/coverage signal is confirmed on designed geometry
  (`gradient_scene`) and open, not resolved, on real photographed
  geometry — the specific question item 3 above re-attacks.
