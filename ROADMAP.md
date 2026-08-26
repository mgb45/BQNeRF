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

### 1. Get every gap-experiment checkpoint above the quality bar, not just the tool

The gap-based directional result (see "what's already solid" below) is
real but every one of its 5 checkpoints falls short of this project's
own >20dB held-out-PSNR bar — caught by directly looking at the renders,
not by the averaged PSNR number, which hid a wide per-view quality
spread. A within-checkpoint contrast check (querying the same messy
checkpoint at the missing direction vs. a well-covered one) supports the
signal being real, not just noise, but that's supporting evidence, not a
substitute for reconstructions clean enough to trust outright. Lego's
100-view pool caps how far this can go; needs either a smaller max gap
width (trading effect size for quality) or a denser real dataset.

### 2. Extend the gap-based directional design to a genuinely photographed scene

The same design (`gap_directional_experiment.py`: remove a deliberate
angular gap from a dense real view pool, leave everything else
untouched) hasn't been run on the actual photographed Mip-NeRF360
"bonsai" scene yet — and per item 1, reconstruction quality needs
checking with the same skepticism from the start this time, not after a
rendered GIF prompts the question.

### 3. Keep kernel choice pluggable — it's a strength, not a loose end

RBF vs. Matérn already showed a real trade-off (see `FINDINGS.md`): no
universal winner, different kernels better for different things. That's
not a gap to close before shipping — the method being kernel-agnostic
(swap in whatever kernel suits the use case, keep the same closed-form
posterior/variance machinery) is a real advantage over bespoke
single-purpose uncertainty methods. Keep the tool's kernel/bandwidth
choice a clean, exposed parameter, not hardcoded to one default. Further
exhaustive kernel exploration is a "let the field discover more kernels"
problem for after publication, not a blocker now.

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

- **One clean, general tool, not a pile of one-off scripts.**
  `render_directional_uncertainty_sweep.py` loads any real `gsplat`
  checkpoint, computes both spatial (quadrature) and directional
  (epistemic) BQ variance, and renders it — auto-framed from the
  checkpoint's own splat extent, kernel family/bandwidth exposed as
  parameters, and gated on a mandatory held-out-PSNR quality check before
  computing anything (refuses, or `--force` to override). This is now
  the primary way this project validates a new result.
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
- **The directional/coverage signal is confirmed on designed geometry
  (`gradient_scene`), and real but not yet fully clean on real geometry.**
  The first two real-geometry attempts hit a genuine reconstruction-
  quality confound (thinning view density everywhere as "spread"
  widened); redesigning the manipulation as a deliberate angular gap
  carved out of an otherwise-dense real view pool
  (`gap_directional_experiment.py`) removed that confound — directional
  BQ variance tracks gap width cleanly (`rho=1.000`, `6.95x` range vs. a
  `1.34x` position-only control), and a within-checkpoint contrast (same
  messy checkpoint, missing direction vs. a well-covered one, `14.78` vs.
  `6.98`) supports the signal being direction-specific, not just general
  checkpoint noise. But every checkpoint in this result is genuinely
  below this project's own quality bar — the averaged PSNR looked
  acceptable, but the actual held-out renders are mostly unrecognizable
  except near surviving training angles, caught only by looking directly
  at them. See active items 1-2.
