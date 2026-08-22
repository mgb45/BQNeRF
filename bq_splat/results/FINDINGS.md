# Milestone 1 findings: BQ math on toy 1D rays

Ran via `scripts/validate_milestone1.py`. Numbers below are from that
script's default settings (200 trials per node count, domain [0, 10],
RBF sigma=0.35, Matérn-3/2 rho=0.5) and are reproducible with the seeds
hardcoded in the script.

## 1. Accuracy: BQ vs. naive Riemann sum

| nodes | Riemann MAE | BQ-RBF MAE | BQ-Matérn32 MAE |
|-------|-------------|------------|------------------|
| 5     | 1.173       | 1.701      | 1.524            |
| 10    | 0.701       | 1.134      | 0.979            |
| 20    | 0.358       | 0.494      | 0.434            |
| 40    | 0.171       | 0.241      | 0.161            |

BQ loses to the naive piecewise-constant Riemann-sum estimator at every
node count except a near-tie at n=40 for Matérn. This reproduces, at toy
scale and with a completely independent implementation, the same qualitative
result already on record for this project (the original NeRF-BQ PSNR
experiment in `inspect_model.ipynb`/`figs/Ns_*.png`, where BQ also lost to
standard quadrature). Two independent implementations losing the same way is
reasonably strong evidence this isn't a codebase-specific bug — it's a real
property of this formulation, at least with these test signals and a single
fixed kernel bandwidth. This is consistent with ROADMAP.md's framing:
raw accuracy is not the claim to defend.

## 2. Calibration: does BQ variance track BQ's actual error?

Correlation between `|BQ error|` and BQ posterior std, pooled across all
node counts:

- RBF: **0.706**
- Matérn-3/2: **0.722**

Both are reasonably well calibrated and close to each other. This is worth
flagging explicitly because an earlier run of this same experiment, before
a numerical-conditioning fix (see §4), showed RBF at only 0.082 — which
would have read as "Matérn is dramatically better calibrated than RBF," a
tempting and clean-sounding result. It wasn't real. Fixing the actual bug
made the two kernels look similar. This is recorded here as a caution against
over-reading small-scale results before checking for numerical artifacts —
and as the reason the codebase's jitter is now relative rather than fixed.

A real kernel-choice comparison (varying bandwidth/smoothness relative to
the signal's own structure, more kernels, statistical testing across many
more trials) is still open and is exactly the "cleanest, least-contested
novelty claim" ROADMAP.md points to — this run doesn't settle it, it mostly
rules out one artifact that would have produced a false positive.

## 3. Gap experiment: does variance rise in an under-resolved-but-visible region?

A region interior to the domain (not occluded, not near an edge) was given
deliberately sparse local node coverage while containing real high-frequency
signal structure (two overlapping narrow bumps).

- Mean local BQ variance **inside** the gap: 0.546
- Mean local BQ variance **outside** the gap: 0.142
- Ratio: **3.85x**

(An earlier version of this script had a plotting bug — a local window with
fewer than 2 nodes was skipped and recorded as `NaN` instead of computing
the variance `bayesian_quadrature` already supports for those cases. That
silently broke the plotted curve inside the gap, at exactly the points that
should matter most, and also biased the "inside"/"outside" means reported
in an earlier draft of this file — 0.456/0.070/6.5x — since points nearest
the gap's sparsest coverage were the ones being dropped. The numbers above
are from the corrected script.)

See `gap_experiment.png`. The curve is more specific than "rises uniformly
inside the sparse region": variance peaks (~1.2) right at the gap's leading
edge, where sparse coverage first meets the two-bump high-frequency
structure, then stays moderately elevated (~0.4-0.6) through the gap's
interior before dropping off outside it. That's still consistent with the
paper's differentiation claim — variance is responding to under-sampling
*relative to local signal structure*, not to node count in isolation — but
it's a more specific and more interesting shape than a flat elevated
plateau, and is worth designing the eventual real (GS-based) differentiation
experiment around: expect the effect to be strongest at the boundary where
coverage drops off against real structure, not necessarily uniform across
an entire under-covered region.

## 4. Numerical conditioning: a real implementation lesson

Random (irregular) node placement occasionally produces near-duplicate
nodes. At n=40 nodes uniform on [0, 10] with RBF sigma=0.35, worst-case Gram
matrix condition number over 200 draws was measured at **1.5e18** — beyond
double-precision's reliable range — causing one specific n=40 trial's BQ-RBF
error to spike to 1.38 (vs. a smooth 0.57 at n=20) in the first version of
this script, which used a fixed jitter of 1e-8. Switching to jitter scaled
to the Gram matrix's own diagonal (`rel_jitter=1e-4`, i.e.
`jitter = 1e-4 * mean(diag(K))`) brought the worst-case condition number
under 1e5 across the same test and fixed both the accuracy spike and the
RBF calibration-correlation artifact in §2.

This matters beyond the toy script: real Gaussian splats can be near-
collocated in a scene (dense regions after densification), so the eventual
gsplat-integrated BQ computation needs the same relative-jitter treatment,
not a fixed constant — noted in `bq_splat/quadrature.py`'s docstring and in
ROADMAP.md's engineering plan.

## 5. Trainable kernel bandwidth: does fitting it close the accuracy gap?

Section 1's diagnosis was that BQ loses to Riemann because both kernels use
one hardcoded bandwidth (RBF sigma=0.35, Matern rho=0.5) across scenes whose
true bump widths range from 0.05 to 0.6 -- the same fixed-bandwidth choice
`models/nerf.py` makes (`sig=0.25`, never adapted). `bq_splat/hyperparams.py`
adds standard GP kernel-hyperparameter fitting (maximize the log marginal
likelihood via a log-spaced grid search + bounded refinement -- no
torch/autodiff needed for this numpy/scipy stage) and
`scripts/validate_trainable_kernel.py` reruns the milestone-1 accuracy sweep
with the bandwidth fit per-trial instead of fixed.

| nodes | Riemann | BQ-RBF fixed | BQ-RBF fit | BQ-Matern fixed | BQ-Matern fit |
|-------|---------|--------------|------------|-------------------|----------------|
| 5     | 1.173   | 1.701        | 1.065      | 1.524             | 1.155          |
| 10    | 0.701   | 1.134        | 0.778      | 0.979             | 0.749          |
| 20    | 0.358   | 0.494        | 0.431      | 0.434             | **0.344**      |
| 40    | 0.171   | 0.241        | 0.232      | 0.161             | **0.148**      |

Fitting closes most of the gap at every node count, and **fitted Matern
beats Riemann outright at n=20 and n=40** (fitted RBF beats Riemann at n=5
but not at higher n). This is a meaningful confirmation of the section-1
diagnosis: the earlier loss to Riemann was substantially a fixed-bandwidth
mismatch problem, not a fundamental limitation of Bayesian quadrature
itself, at least for this class of signals.

Fitted bandwidths vary a lot by scene (RBF sigma: median 0.69, 10-90th
percentile [0.38, 1.57] -- more than 4x the hardcoded 0.35; Matern rho:
median 2.15, [1.09, 5.67] -- an order of magnitude above the hardcoded 0.5),
which is itself informative: no single fixed bandwidth could have served
this whole scene distribution well, and per-scene fitting is doing real
work, not just adding noise. One methodological note: an initial run capped
the fitting search at rho <= 3.0 and 12% of n=20 trials hit that boundary;
widening the bounds to 8.0 changed the reported MAE numbers by only
~0.001-0.0004 (noise-level) while moving the 90th-percentile fitted rho from
a clipped 3.0 to 5.67 -- so the result isn't an artifact of an overly tight
search range, but it's still worth using generous bounds by default.

Caveat: this fits one bandwidth per full node set (i.e., "per ray"), which
is the natural analogue for a per-pixel BQ computation in the eventual GS
setting, but it does mean the fit sees the same data it's then evaluated on
-- there's no held-out validation here, just marginal-likelihood fit
quality. That's standard practice for this kind of hyperparameter fitting,
but worth keeping in mind before treating "fitted Matern beats Riemann" as
a claim about generalization rather than about in-sample fit quality.

**Update, see section 7:** a proper held-out test (fit on one set of
scenes, evaluate on disjoint scenes never seen during fitting) confirms
this generalizes for Matern -- global-fit Matern nearly matches, and
sometimes beats, the in-sample oracle above. It does *not* hold up the same
way for RBF: the population-optimal RBF bandwidth turned out to be almost
exactly the original hardcoded 0.35, meaning RBF's per-scene gains reported
above were substantially the fit adapting to each scene's specific noisy
sample layout, not evidence of a real population-level bandwidth mismatch.
Read this section's RBF numbers with that correction in mind.

## 6. 2D bridge experiment: does the effect survive moving to an image-plane domain?

Sections 1-5 all work in 1D: nodes along a ray's depth axis, treated as
point evaluations of a signal to be integrated over depth. Real GS splats
don't work that way -- they're scattered over the 2D image plane with
anisotropic footprints, and a pixel's color comes from depth-sorted alpha
blending over whichever splats' footprints overlap it, not a 1D ray
integral. Before spending any GPU time on a real GS scene (ROADMAP.md
milestone 2), it's worth checking the central claim survives that change of
domain.

`bq_splat/kernels.py` adds `ProductKernel`: a D-dimensional kernel built as
the product of D 1D kernels, one per axis. For RBF this is *exact* -- an
isotropic Gaussian factorizes exactly into per-axis 1D Gaussians because
squared Euclidean distance is a sum over axes -- so this required no new
integration code or numerical risk; `v`/`vv` over an axis-aligned box just
become products of the already-tested 1D `v`/`vv` calls.
`bq_splat/quadrature.py` adds `bayesian_quadrature_nd` generalizing the
posterior mean/variance computation to this setting (kept as a separate
function from the tested 1D `bayesian_quadrature` to avoid any regression
risk to already-passing tests). `bq_splat/toy_scene_2d.py` mirrors
`toy_scene.py`: a mixture of 2D Gaussian bumps as the true signal, and a
deliberate circular sparse-coverage gap in splat-center placement.

`scripts/validate_2d_gap_experiment.py` reruns the gap experiment over a
10x10 image-plane domain with 250 scattered splat centers and a circular
gap of thinned coverage containing real signal structure:

- Mean local BQ variance **inside** the gap: 2.096
- Mean local BQ variance **outside** the gap: 0.432
- Ratio: **4.85x**

See `gap_experiment_2d.png`. The effect survives the move to 2D, and the
same qualitative shape from the 1D experiment (FINDINGS.md §3) shows up
again: the variance hotspot concentrates near the gap's *coverage boundary*
rather than filling the circle uniformly, not exactly centered on the
brightest visible bump inside it. That's consistent with the mechanism
being "sparse coverage relative to nearby real structure," which is exactly
what a real GS densification/NBV signal would need to be sensitive to. This
doesn't replace the real GS-based differentiation experiment ROADMAP.md
calls for -- it's still an analytic mixture-of-Gaussians signal and
isotropic splat placement, not actual learned splat covariances or camera
geometry -- but it's meaningfully closer, needed no GPU, and the effect
held up rather than disappearing when the domain got harder.

## 7. Held-out validation: does bandwidth fitting generalize, or was it in-sample overfitting?

Section 5's caveat: the bandwidth was fit and evaluated on the *same*
scene's nodes, so "fitted Matern beats Riemann" couldn't rule out unfair
in-sample advantage. `bq_splat/hyperparams.py` adds
`fit_kernel_param_pooled` (fits one shared bandwidth by maximizing summed
log marginal likelihood across many datasets) and
`scripts/validate_trainable_kernel_heldout.py` fits ONE bandwidth per
kernel on 90 calibration (scene, node-count) draws, then evaluates it on
150 *disjoint* test scenes it never saw -- alongside the old hardcoded
bandwidth and a per-scene in-sample oracle fit, for comparison.

Global fitted RBF sigma: **0.348** (vs. hardcoded 0.35). Global fitted
Matern rho: **1.303** (vs. hardcoded 0.5).

| n  | Riemann | RBF fixed | RBF global-fit | RBF oracle | Matern fixed | Matern global-fit | Matern oracle |
|----|---------|-----------|-----------------|------------|----------------|---------------------|-----------------|
| 10 | 0.745   | 1.080     | 1.086           | 0.746      | 0.929          | **0.685**           | 0.689           |
| 20 | 0.387   | 0.506     | 0.510           | 0.453      | 0.435          | **0.355**           | 0.370           |
| 40 | 0.176   | 0.193     | 0.193           | 0.206      | 0.163          | **0.141**           | 0.140           |

Two genuinely different stories here, and this is worth taking at face
value rather than smoothing over:

- **RBF's story changes under held-out evaluation.** The globally-optimal
  RBF bandwidth (0.348) landed almost exactly on the arbitrary hardcoded
  default (0.35) -- the population doesn't actually want a different RBF
  bandwidth. Global-fit RBF barely moves the needle over fixed RBF at any
  n, and at n=40 the *in-sample oracle* (0.206) is actually worse than
  fixed (0.193). Section 5's per-scene RBF gains were mostly the fit
  adapting to each scene's specific finite, noisy sample layout -- real in
  that specific narrow sense, but not a signal that generalizes, and not
  evidence of a true population-level bandwidth mismatch the way the
  original diagnosis (section 1) suggested for RBF specifically.
- **Matern's story holds up, and gets stronger.** Global-fit Matern beats
  fixed Matern at every n, and nearly matches -- at n=20 and n=40, slightly
  *beats* -- the in-sample oracle, despite never seeing the test scenes.
  This is a real, held-out-validated result: a single Matern rho fit once
  captures essentially all of the earlier apparent benefit.

Read together, this sharpens (rather than just confirms) the kernel-choice
claim: it's not "fitting the bandwidth helps in general," it's specifically
that **Matern's rougher, once-differentiable smoothness assumption is a
better structural match to this class of signals than RBF's
infinitely-smooth one, and that mismatch is fixable with a single fitted
constant, not per-scene adaptation.** That is a cleaner, more defensible
version of the "kernel choice matters" claim this project's original goal
(comparing kernels, stalled after only RBF got wired into `models/nerf.py`)
was reaching for.

This also matters for deployment cost, which is the subject of section 8:
if a bandwidth fit once on a calibration set generalizes this well, a real
GS integration would not need to refit a bandwidth per pixel or per local
neighborhood -- fit once offline, reuse everywhere -- which is a much
cheaper computational story than per-query optimization.

## 8. Computational scaling: what does GS-scale (10^5-10^6 splats) per-pixel BQ actually cost?

ROADMAP.md flags "batched closed-form BQ posterior-variance computation
compatible with GS's typical splat counts" as the main engineering risk,
on the assumption the bottleneck would be the linear solve at scale.
`scripts/benchmark_local_bq_scaling.py` measures this directly on CPU, and
the actual bottleneck turned out to be something else entirely.

**Neighbor lookup.** The gap-experiment scripts so far find local nodes by
brute-force masking over *all* nodes -- O(N) per query, fine at N~250, not
at GS scale. A `scipy.spatial.cKDTree` (already a scipy dependency, no new
library) fixes this completely: at N=1,000,000, brute-force is 25.6ms per
query vs. KD-tree's 0.015ms -- **~1,700x faster**, and flat regardless of N
(tree build is a one-time ~420ms cost at N=1M). This part of the concern is
fully resolved with an off-the-shelf data structure.

**The linear solve is not the bottleneck.** A raw `numpy.linalg.solve` on a
20x20 system takes ~0.009ms -- utterly negligible even at hundreds of local
neighbors. Profiling `bayesian_quadrature_nd` end-to-end found **94% of its
~3.75ms per-query cost was `kernel.vv(bounds)`** -- the double integral used
for the prior variance term -- which `bq_splat/kernels.py` computes via
`scipy.integrate.quad` (a deliberate choice, made in the 1D module to avoid
trusting a hand-derived closed-form double integral after this repo's own
"double quad" bug history; see that module's docstring). That numerical-
integration overhead, not the O(k^3) solve ROADMAP.md worried about, is
what makes a naive per-pixel Python loop infeasible: at ~3.75ms/query,
an 800x800 image is ~2,400-2,960s (40-49 minutes) single-threaded, and this
barely depends on N or on local neighborhood size k.

**The fix is exact, not a tradeoff.** For a stationary kernel (RBF, Matern
-- both are), `vv(a, b)` depends only on the window's *shape and size*, not
its position: confirmed numerically to ~1e-13 relative difference by
evaluating the same-size window at three different centers. Real per-pixel
local windows are always the same size, just recentered pixel to pixel, so
`vv` needs computing **once per unique window size**, not once per query.
Caching it (`bq_with_cached_vv` in the benchmark script) gives a measured
**13.4x average speedup** (3.75ms -> 0.21-1.71ms depending on local
neighborhood size), cutting the same 800x800 extrapolation to
**137-421 seconds (2.3-7 minutes), still single-threaded, still pure
Python/numpy, up to N=1,000,000 splats.** (One caveat: this exact identity
holds for interior query points whose window doesn't get clipped by a
scene/image boundary; edge pixels have a differently-shaped window, but
there are only O(border length) distinct clipped shapes, not O(pixels), so
they're still cheaply cacheable, just not for free via the same single
constant.)

A separate, smaller check: batching many independent local solves into one
vectorized `numpy.linalg.solve` call (stacked (n_systems, k, k) arrays
instead of a Python-level loop) measured a further ~2.5x speedup at k=20 on
top of the above. Combined with the vv-caching fix, straightforward
CPU-side batching plausibly gets a full-image pass to well under a minute
without needing GPU code at all, though this wasn't pushed further since
the real deployment target (gsplat/GPU) will batch across many pixels'
local systems on-device rather than in a numpy loop regardless.

**Bottom line for the eventual gsplat integration:** don't build the
"batched closed-form" work ROADMAP.md anticipated as a hard numerical-
linear-algebra problem -- the k-scale solve was never the issue. Two
concrete, already-validated pieces carry over directly: (1) a KD-tree (or
gsplat's own existing tile/neighbor structures, which likely already solve
this) for O(log N) local neighbor lookup instead of brute force, and (2) a
`vv` cache keyed by window size instead of recomputing it per pixel. Both
are exact optimizations, not approximations, and together take a rough
"per-pixel Python loop" implementation from tens of minutes to a few
minutes per image on CPU alone -- a very different, much less risky
starting point for the real experiment than ROADMAP.md's original framing
assumed.

## 9. Directional kernel: does the same BQ formalism catch visibility uncertainty too?

Prompted by a design conversation about SLAM: as splats accumulate over
time, does this project's BQ machinery give you *both* quadrature
uncertainty and the kind of view-direction epistemic uncertainty methods
like GAVIS target? The honest answer worked out in that conversation: not
automatically. A splat existing near a query point reads as "well covered"
to a position-only kernel regardless of whether it was seen from one
grazing angle or from all around -- there's no notion of viewing direction
in `bayesian_quadrature_nd` at all. But the same kernel-product structure
`ProductKernel` already uses generalizes to fix this, rather than needing a
second, separate mechanism.

**Construction.** `DirectionalKernel` (`kernels.py`) is a von Mises-Fisher
kernel on unit direction vectors: `k(w, w') = exp(kappa * (w.w' - 1))`,
positive-definite for kappa >= 0 (w.w' is itself PD; exp of a
positive-scaled PD kernel is PD by the Schur product theorem), with
self-similarity `k(w,w) = 1` always. `bayesian_quadrature_directional`
(`quadrature.py`) combines it with a position kernel multiplicatively --
but *not* symmetrically with position: a rendered pixel integrates over a
spatial footprint but evaluates one specific outgoing direction, it doesn't
integrate over a range of directions. So position is integrated as before
(`v_pos`, `vv_pos`, unchanged) while direction is evaluated pointwise at
the query direction (`k(w_i, w_query)`), giving:

```
K_ij = k_pos(x_i, x_j) * k_dir(w_i, w_j)
v_i  = v_pos(x_i, bounds) * k_dir(w_i, w_query)
vv   = vv_pos(bounds)                            (k_dir(w_query, w_query) == 1)
```

This is a known pattern in Bayesian quadrature (some dimensions integrated
out, one held at a query point), not an ad hoc mixing. A clean correctness
check confirms it: with `kappa=0`, `k_dir` is identically 1 for every pair
regardless of the actual directions, so the whole computation must collapse
EXACTLY onto `bayesian_quadrature_nd`'s answer -- verified to `1e-10` for
random (nonsense) directions in `tests/test_directional.py`, not just
approximately close.

**Isolation experiment** (`validate_directional_isolation.py`, single fixed
spatial location, no position kernel involved at all): a synthetic
view-dependent signal observed from either a wide spread of angles or a
narrow ±0.35 rad cone. Posterior variance at the observed directions is
~0 in both cases; at the direction directly opposite the observations, wide
coverage gives 0.22 (still finite -- 12 random points over the full circle
leave some gaps by chance) while narrow coverage gives exactly 1.0, the
full prior variance -- a **28,715x** ratio vs. wide coverage's 1,452x. See
`directional_isolation.png`.

**Combined experiment** (`validate_directional_combined.py`) is the one
that actually answers the SLAM question: two zones with *spatial* splat
density held exactly equal by construction (both zones get the identical
set of relative offsets from their own center, translated -- not just an
equal count, since two independently-random equal-count placements turned
out to still differ in how clumped vs. spread the points were, a real
confound caught and fixed during this work, not a hypothetical one), one
zone's splats each seen from a wide spread of directions, the other's each
seen from a narrow cone, querying from a direction outside that cone:

- Position-only BQ variance (`bayesian_quadrature_nd`, blind to direction):
  wide zone 1.890, narrow zone 1.831 -- **ratio 0.97x, i.e. no difference**,
  exactly as expected since spatial density is provably matched.
- Position+direction BQ variance (`bayesian_quadrature_directional`): wide
  zone 2.983, narrow zone 7.325 -- **ratio 2.46x**, purely from the
  directional-coverage difference.

See `directional_combined.png`: panel (a) shows both zones as similarly
dark; panel (b) shows the narrow-cone zone lit up while the wide-angle zone
stays dark. This is a direct demonstration that a position-only signal
(everything built through section 8) genuinely cannot see this failure
mode, and a directional extension of the *same* kernel-product formalism
does -- not a different mechanism bolted on, the same one extended.

**What this doesn't yet do.** This is still a toy validation: single
synthetic view-dependent signal, hand-picked kappa, no real per-splat
multi-view observation data, and no attempt yet to fit kappa the way
sections 5/7 fit spatial bandwidths (a natural next step -- kappa fit
per-region would double as a learned specularity estimate, since a
view-dependent/specular surface needs a short angular correlation length
to be well-constrained while a diffuse one doesn't). It also doesn't
replace a real visibility field like GAVIS's -- it's evidence the same
closed-form BQ machinery *can* represent this failure mode mathematically,
which is a different and smaller claim than "this is a better or cheaper
way to compute it at GS scale," which hasn't been tested.

## Bottom line for the go/no-go gate

Milestone 1 (derivation + small-scale validation) is a qualified pass:
the ported math is correct (kernel/BQ unit tests all pass, RBF matches
`models/nerf.py`'s formula exactly), variance is reasonably calibrated for
both kernels, and the gap experiment supports the core differentiation
claim at toy scale. The accuracy gap vs. naive Riemann summation — which,
per ROADMAP.md, the paper isn't trying to close as its main claim — is now
understood precisely (§5): it was substantially a fixed-kernel-bandwidth
mismatch, and fitting the bandwidth per scene closes most of it, with fitted
Matern beating Riemann at n=20 and n=40. That both strengthens the accuracy
story (it's not a dead end, it's an unfitted hyperparameter) and reinforces
why the paper's real claim should be about calibrated uncertainty and the
differentiation experiment rather than accuracy alone. §6 further shows the
differentiation effect survives moving from a 1D ray-depth domain to a 2D
image-plane domain with GS-realistic scattered node placement, which is the
more relevant geometry for milestone 2.

§7's held-out test refines §5's story rather than just confirming it: the
fitted-bandwidth improvement is real and generalizes for Matern (a single
bandwidth fit once on a calibration set nearly matches, sometimes beats, an
in-sample oracle), but not for RBF (the population-optimal RBF bandwidth
turned out to be almost exactly the original hardcoded 0.35, so RBF's
earlier per-scene gains were mostly overfitting to each scene's specific
sample layout). That sharpens the kernel-choice claim into something more
specific and more defensible: Matern's rougher smoothness assumption is a
better structural match to this signal class than RBF's, fixable with one
constant rather than per-instance adaptation.

§8 addresses the other open engineering question head-on: GS-scale
computational feasibility. The bottleneck ROADMAP.md anticipated (an
expensive linear solve at scale) wasn't the real one -- profiling found 94%
of per-query cost was a numerically-integrated `vv` term, fixed by an exact,
zero-cost caching trick (stationary kernels' `vv` only depends on window
size, not position) plus a KD-tree for neighbor lookup, together taking a
naive ~2,400-3,000s single-threaded per-image estimate down to ~140-420s
with no approximation and no GPU involved. This meaningfully de-risks
milestone 2's engineering scope before any of it is built.

§9 answers a question that came up while thinking through deployment in a
SLAM context, where splats accumulate incrementally: does this machinery
give you visibility/epistemic uncertainty too, or only quadrature
uncertainty? On its own, only quadrature uncertainty -- a position-only
kernel can't distinguish a splat seen from every angle from one seen once,
obliquely. But extending the same `ProductKernel` structure with a
directional (von Mises-Fisher) factor, integrated over position as before
but evaluated pointwise at a query viewing direction, lets the identical
closed-form BQ machinery catch that failure mode too: a controlled toy
experiment with spatial density held *exactly* equal between two zones
(by construction, not by luck -- an earlier version of this check relied on
independently-random equal-count placement and it wasn't actually matched,
a real confound worth having caught) shows position-only variance
correctly reports no difference (0.97x) while position+direction variance
correctly reports 2.46x higher variance in the zone observed from a narrow
cone. This doesn't replace a real visibility field, and doesn't yet fit
`kappa` the way spatial bandwidths get fit in §5/§7, but it's evidence the
unification is mathematically real, not just a nice story.

Proceeding to ROADMAP.md's milestone 2 (the real, GS-based differentiation
experiment) is reasonable, now on a firmer footing on the statistical side
(§7), the computational side (§8), and -- more speculatively, since §9 is
toy-scale and single-signal, not yet combined with the spatial toy work --
the scope of what one formalism might eventually cover, than when this
section was first written. If bandwidth fitting carries into that setting,
use the pooled/global marginal-likelihood approach from §7 rather than
per-pixel fitting or a hardcoded value -- both simpler and, per §7, no
worse.
