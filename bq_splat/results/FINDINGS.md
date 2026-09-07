# bq_splat findings (summary)

Toy-scale (1D/2D synthetic signals, pure numpy/scipy, no GPU) validation
of the Bayesian-quadrature math before it was ported to real Gaussian
Splatting data in `gs_experiment/` — where the same questions get
re-asked and answered on real scenes (see
[`gs_experiment/results/FINDINGS.md`](../../gs_experiment/results/FINDINGS.md),
the primary results document).

This file is a short, current-conclusions summary.

## The core math is correct

RBF and Matérn-3/2 kernel/quadrature formulas are unit-tested against an
exact closed-form RBF formula (from this project's original from-scratch
NeRF prototype) and against numerical integration.

A formal proof that the Bayesian-quadrature posterior mean recovers
standard alpha compositing exactly (under the piecewise-constant model
every NeRF/3DGS renderer already assumes), and that the posterior
variance is a *provable* — not merely empirically-correlated — bound on
its own error, is in
[`PROOF_alpha_compositing_equivalence.md`](../PROOF_alpha_compositing_equivalence.md).
This is also the rigorous grounding for this project's "unification"
claim: quadrature uncertainty and directional/epistemic uncertainty turn
out to be the same worst-case-error theorem applied to different linear
functionals on one product-kernel posterior, not two separate mechanisms.

## Raw accuracy: BQ loses to a naive Riemann sum — with a hardcoded
## bandwidth. Fitting it closes most of the gap

With a fixed kernel bandwidth, BQ's posterior mean loses to plain
piecewise-constant (Riemann-sum) integration at every node count tested —
matching the original NeRF-BQ prototype's own result independently.
Fitting the bandwidth per scene via marginal-likelihood optimization
(`hyperparams.py`) closes most of that gap, and fitted Matérn actually
*beats* Riemann summation at n=20/40 nodes. A held-out check refines this:
the fitted-bandwidth improvement generalizes for Matérn (a bandwidth fit
once nearly matches an in-sample oracle on unseen scenes) but not for
RBF — the population-optimal RBF bandwidth turned out to be almost
exactly the original hardcoded value, so RBF's earlier per-scene gains
were mostly overfitting to each scene's own sample layout, not a real
mismatch worth fixing. Raw accuracy was never this project's claim to
defend (see `ROADMAP.md`) — the point of this line of work is that the
bandwidth question is real and kernel-family-dependent, which carries
directly into `gs_experiment/results/FINDINGS.md`'s real-checkpoint
bandwidth-fitting results.

## Posterior variance is reasonably calibrated, and rises in genuinely
## under-resolved regions

BQ posterior standard deviation correlates with actual error at ~0.7 for
both kernels. A deliberately under-sampled-but-visible region (real
signal structure, sparse local node coverage, not occluded) shows ~3.9x
higher average local variance than well-covered regions, peaking specifically
at the region's leading edge — where sparse coverage first meets real
structure — rather than as a flat elevated plateau. The effect survives
moving from a 1D ray-depth domain to a 2D image-plane domain with
scattered node placement (4.85x ratio), which is the geometry a real GS
scene actually has. This is the toy-scale version of the central claim
`gs_experiment/results/FINDINGS.md` later validates on real checkpoints
(§1 there).

One real numerical-conditioning lesson from this work: irregular node
placement can push the Gram matrix condition number past 1e18 with a
fixed jitter; a jitter scaled to the kernel's own diagonal fixes it and
materially changes downstream numbers (an earlier, uncorrected run showed
a spuriously low RBF calibration correlation purely from this).

## Computational cost at real GS scale is dominated by a term you can
## cache exactly, not the linear solve

The originally-assumed bottleneck at Gaussian-Splatting scale (10^5-10^6
splats) — an expensive linear solve — turned out not to be it: profiling
found 94% of per-query cost was a numerically-integrated `vv` term, fixed
*exactly* (not approximated) by caching it per window size, since it's
provably position-independent for a fixed-size window under a stationary
kernel. That plus a KD-tree for neighbor lookup takes a naive
~2,400-3,000s single-threaded per-800×800-image estimate down to
~140-420s, on CPU alone, up to a million synthetic splats — before any
GPU code was written. Both optimizations carry directly into
`gs_experiment/pixel_uncertainty.py`'s `LocalUncertaintyEngine`.

## The directional extension: the same formalism catches viewing-angle
## coverage too

Motivated by a SLAM design question (do accumulating splats give you
*both* quadrature and visibility/epistemic uncertainty, or only the
former?) — on its own, only the former: a position-only kernel can't tell
"seen from every angle" apart from "seen once, obliquely." Extending the
existing `ProductKernel` with a directional (von Mises-Fisher) factor
fixes this with the same closed-form machinery, not a second mechanism: a
controlled toy experiment holding spatial density *exactly* equal between
two zones (by construction, after an earlier independently-random
equal-count placement turned out not to be truly matched — a real
confound caught and fixed) shows position-only variance correctly reports
no difference (0.97x) while position+direction variance correctly reports
2.46x higher variance in a narrow-cone-observed zone. This toy-scale
result is what `gs_experiment/`'s real-checkpoint directional-gradient
work later builds on and stress-tests on real geometry.

## Rendering-aware BQ fixes a real gap the earlier sections above didn't
## close: the quadrature domain/weight wasn't renderer-aware

`PROOF_alpha_compositing_equivalence.md` section 7 flags this precisely:
everything validated above (and `gs_experiment/pixel_uncertainty.py`'s
`LocalUncertaintyEngine`, which every real-checkpoint result in
`gs_experiment/results/FINDINGS.md` is built on) integrates the base
kernel uniformly over an arbitrary box window — the renderer
(transmittance, opacity, footprint, visibility) never enters the
integration functional itself. `bq_splat/render_weight.py`'s
`GaussianRenderWeight` and `bayesian_quadrature_rendering_aware`
(`bq_splat/quadrature.py`) close this: the kernel becomes query-specific,
`k_q(xi, xi') = a_q(xi) k_base(xi, xi') a_q(xi')`, with `a_q = T_q sigma
G_q` a real per-query rendering weight instead of a uniform indicator over
a box. Closed-form for an isotropic RBF `k_base` and Gaussian `a_q` (via
the standard Gaussian-product identity), cross-checked against numerical
integration to `~1e-6` (1D) / Monte Carlo to `~5%` (2D, where nested
`nquad` becomes impractically slow) in `tests/test_render_weight.py`.

`python -m bq_splat.validate --check rendering-aware` builds a genuine
ray/pixel scene — a real transmittance-weighted `a_q(t) = T(t)sigma(t)`
derived from an explicit density (a narrow bump, a hard surface), splats
scattered across the whole ray depth including behind the surface — and
shows the fix operating end to end: adding one occluded splat with a
deliberately wrong color shifts the old box-style BQ mean by `~1.49`
(`~30%` of the true value `0.048`) but the rendering-aware BQ mean by only
`~0.0001` (`~0.2%`) — a ~13,500x smaller shift — because `a_q` at that
splat's depth is correctly near zero. See
`bq_splat/results/rendering_aware.png`.

**Connecting this back to `gs_experiment`**: a real per-query `a_q` built
from actual splat data, not just the toy ray-depth demonstration above.
Two steps are wired in, in increasing order of realism:
`LocalUncertaintyEngine.rendering_aware_variance` (real per-splat opacity
as `a_q`'s amplitude, a Gaussian footprint tied to the query radius, but
occlusion-blind — a flat neighborhood-mean opacity) and
`rendering_aware_variance_along_ray` (real, depth-ordered alpha-
compositing transmittance weights along the specific ray from a given
camera through the query point —
`gs_experiment.visibility_attribution.ray_transmittance_weights`, the
continuous analogue of `occlusion_mask`'s binary flag, using real
per-splat opacity as the discrete alpha in Theorem A's own formula): a
fully-opaque occluder in front of a target on the same ray now pulls the
posterior mean essentially exactly onto the occluder's color
(`tests/test_gs_pixel_uncertainty.py`'s
`test_rendering_aware_variance_along_ray_weights_toward_the_occluder_not_the_occluded_target`),
while the occlusion-blind version lands far from it.

A third step, `rendering_aware_variance_via_gsplat`
(`gs_experiment/gsplat_rendering_weights.py`), closes what the previous
paragraph flagged as still open: it uses gsplat's own differentiable
EWA-splatting projection (`gsplat.fully_fused_projection`, real GPU
computation, validated end to end on an RTX 3090 in
`tests/test_gsplat_rendering_weights.py` and
`tests/test_gs_pixel_uncertainty_gsplat.py`) to get each splat's *real*
anisotropic 2D footprint (from its actual scale/rotation) and real camera
intrinsics, evaluates the exact per-pixel alpha formula gsplat's own CUDA
rasterizer uses (`alpha_i(pixel) = opacity_i * exp(-sigma_i)`, confirmed
to reproduce `opacity_i` exactly at a splat's own projected center), and
runs the same depth-ordered transmittance recursion on top. A fully
opaque, large-footprint occluder again pulls the posterior mean onto its
own color, now via the real anisotropic footprint rather than an
isotropic bearing threshold. **What's still not claimed**: no
antialiasing/sub-pixel footprint integration, no gradient path (runs
under `torch.no_grad()` — an evaluation-time query, not a training
step), and no claim of pixel-exact equivalence to a full production
rasterizer's tile-based compositing order. Requires the documented
gcc-11/nvcc-12.3 toolchain (`requirements-gsplat.txt`) to JIT-compile
gsplat's CUDA kernels — a real, previously-undocumented-here reproduction
of that exact "unsupported GNU version" gotcha was hit and resolved while
building this.

**A real bug found and fixed while generating demo renders against actual
300k-splat checkpoints** (`chair`/`drums`/`hotdog`_prepared, held-out
`eval` views): `_render_weight_from_local_weights`'s moment-matching
treated every candidate splat as a literal point (spread of weighted
centers only), which is a fine approximation when weight mass spreads
across many comparable candidates, but collapses to a near-zero,
jitter-only covariance whenever the real weights concentrate almost
entirely on one dominant splat — confirmed happening in practice for
`rendering_aware_variance_via_gsplat`'s real, much sharper per-pixel
weights (posterior variance as small as `1e-105` in one real query).
Fixed via standard Gaussian-mixture moment matching: covariance =
spread-of-centers **+** the weighted average of each candidate's own real
3D covariance (`quat_scale_to_covariance`, cross-checked against gsplat's
own `quat_scale_to_covar_preci` in `tests/test_gsplat_rendering_weights.py`).
Also: rendering-aware variance on a real scene spans many orders of
magnitude across one image (the render weight's spatial "volume" varies
hugely per query point) — a real, expected property of the theory, not a
bug, but it means a linear color scale is close to useless for these
fields; a log scale is the right default for any future visualization
tool built on this.

**A second, deeper bug: `a_q`'s total mass wasn't pinned to anything
physically meaningful.** Peak-normalizing `amplitude` (fixing it to, say,
a real opacity) and letting `covariance`'s volume float freely means
`a_q`'s *integrated mass* — `amplitude * (2 pi)^(D/2) * |covariance|^(1/2)`
— shrinks to near-zero for a real, physically tiny splat footprint
regardless of how large the opacity actually is (confirmed: this is
exactly what produced the `1e-95`-to-`1e-105`-scale variances above).
Real alpha-compositing weights are bounded (`sum_i T_i alpha_i <= 1`);
a quantity whose scale depends on an unrelated spatial choice cannot
represent that in any comparable way across query points or scenes.
Fixed with `GaussianRenderWeight.from_total_mass` (`bq_splat/render_weight.py`):
pin `total_mass` (e.g. `sum(weights)`, which *is* `1 - T_final` for real
compositing weights) and solve for whatever `amplitude` that requires,
decoupling shape from scale.

**A third, independent bug, found immediately after fixing the above:**
gathering candidates via a 3D-world-space ball query
(`LocalUncertaintyEngine.local_neighbors`) before real alpha weighting
is itself unsound on a dense real checkpoint. Confirmed directly: a
1.6-unit-radius ball query on the `chair` checkpoint (300k splats) found
144,345 candidates — the `max_neighbors=400` cap then keeps a *uniform
random* sample of those, so the one splat actually relevant to a given
pixel survives with probability roughly 0.1%. Fixed by indexing
candidates by real relevance instead of 3D distance:
`gs_experiment.visibility_attribution.CameraSplatIndex` (bearing-space,
pure numpy) and `gs_experiment.gsplat_rendering_weights.GsplatCameraProjection`
(pixel-space, real gsplat projection) each project the whole scene once
per camera and rank any overflow by real bearing/pixel distance, not
randomly — both correct (relevance is about angular/pixel-footprint
alignment, not raw 3D distance) and far cheaper (one projection per
camera instead of one CUDA launch per query point).

**The kernel was still missing half of the original construction.**
Everything above built `k_q(xi, xi') = a_q(xi) k_pos(xi, xi') a_q(xi')` —
position only. The prompt's own construction was always a *joint*
position+direction kernel, `k_base = k_pos * k_dir`; the directional half
existed in this codebase (`DirectionalKernel`, `bayesian_quadrature_directional`)
but had never been merged with the new renderer-aware `a_q` machinery.
Symptom, found by sweeping a camera through a real, designed 150°
training-coverage gap (`lego_prepared/gap_4`): the position-only
rendering-aware variance was *anti-correlated* with real coverage
(Spearman `rho = -0.52` against this project's older, validated
`directional_variance` tool over the same orbit) — confidently "low" in
regions that were actually poorly covered, because a spatially-consistent
local color field says nothing about whether the *viewing angle* being
rendered was ever observed. Fixed with
`bayesian_quadrature_rendering_aware_directional`
(`bq_splat/quadrature.py`): `K` and the moment vector `z_q` each pick up
a `k_dir(d_i, d_j)` / `k_dir(d_i, d_query)` factor (`z_{q,0}` is
unchanged, since `k_dir(d,d) == 1` always), mirroring exactly how
`bayesian_quadrature_directional` already generalized the old box kernel.
Re-running the same gap_4 orbit with the completed joint kernel
(`rendering_aware_variance_via_gsplat_directional`, built from real
per-observation attribution — one row per (splat, observing-camera) pair,
373,387 rows) gives `rho = 0.97` against the classic tool — the
confidently-wrong region is gone, and the two independently-derived
signals now agree almost exactly.

## Scaling the attribution pipeline to real dense checkpoints found three
## more real bottlenecks, in order, each found by actually running at scale

Everything above was validated on `gap_4` (35k splats). Pushing the same
pipeline (`gs_experiment.visibility_attribution`, specifically
`occlusion_mask`, the function `attribute_observations` /
`load_from_gsplat_checkpoint` spend nearly all their time in) to a denser,
larger real checkpoint (`chair_prepared/wide`: 300k splats, 100 cameras)
surfaced a genuine scaling wall, fixed in three iterations:

1. The original implementation (`cKDTree.query_ball_point` in a nested
   Python `for` loop) profiled at 93% of total attribution time on
   gap_4 (100s of 107s, 113 million `abs()` calls at the Python level),
   and made the same step time out entirely (>280s) on the chair
   checkpoint.
2. Replacing the Python-level inner loop with `cKDTree.query_pairs`
   (fully vectorized numpy over every bearing-close pair at once) fixed
   gap_4 (93s -> 15s) but doesn't fix chair: at chair's splat density,
   one camera alone produces 216 million bearing-close pairs (confirmed
   directly), enough to exhaust available memory materializing that
   pair array — this is what actually crashed the host machine during
   a careless follow-up benchmark, not a hypothetical risk.
3. The fix that stuck: a grid/z-buffer rewrite (`occlusion_mask` in
   `gs_experiment/visibility_attribution.py`) that never enumerates
   pairs or per-point neighbor lists at all — it bins bearings into a
   grid, computes each cell's minimum depth once via a sort +
   `np.minimum.reduceat`, and looks up the minimum depth in a fixed,
   density-independent number of surrounding cells per point
   (`np.searchsorted` into sorted cell keys). O(n log n) time, O(n)
   memory, regardless of point density — the actual property `query_pairs`
   lacked. A short proof (in the function's docstring, checked
   numerically) guarantees this never *misses* a true occlusion; it can
   flag some extra ones near a cell corner, since a square cell block
   is necessarily a superset of the true circular search radius. That
   false-positive rate is real (not negligible at coarse settings) but
   was reduced by subdividing the grid finer than `angular_tol` itself
   (`_CELL_SUBDIVISION`, a tunable search radius in finer cells, still
   O(1) lookups per point) — this can only approach the circle's own
   bounding square as a floor, never eliminate the excess entirely,
   which is an accepted, explicitly-documented trade for a function this
   module already calls a cheap proxy rather than a faithful
   reproduction. Verified with a brute-force circular reference across
   both a random synthetic scene and the project's real occluder-scene
   integration test: zero missed occlusions in every case tried.

Separately, every BQ posterior solve (`bq_splat/quadrature.py`) was doing
two independent `np.linalg.solve` calls against the same kernel matrix
(one for the mean, one for the variance) — consolidated into a single
Cholesky factorization (`scipy.linalg.cho_factor`/`cho_solve`, one
factorization shared across both right-hand sides via
`np.column_stack`), roughly 4x fewer flops than two separate LU solves,
with a `np.linalg.solve` fallback if the matrix isn't numerically SPD.

Net result, measured directly on `chair_prepared/wide` (300k splats, 100
cameras) with the fixed pipeline: full checkpoint load + real geometric
attribution completes in **140s, using under 400MB of resident memory**
— down from a configuration that previously either timed out (>280s) or
crashed outright when the pairwise approach's memory use was allowed to
scale with real point density instead of just point count.

## Bottom line

All of the above is a **qualified pass**: the ported math is correct, the
raw-accuracy gap is understood (a fixable bandwidth-mismatch issue, not a
fundamental limitation) and no longer the claim being defended, posterior
variance is reasonably calibrated and responds to under-resolved regions
in the expected way, the computational cost concern that motivated a
possible GPU rewrite was resolved on CPU alone, and the box-quadrature gap
flagged in `PROOF_alpha_compositing_equivalence.md` section 7 now has a
working, validated fix (rendering-aware BQ) rather than remaining an open
item. The rendering-aware construction itself went through three more
rounds of real bugs found only by generating actual renders against real
checkpoints and taking the results seriously rather than at face value
(units/mass-normalization, candidate-selection relevance, and the missing
directional half of the kernel) — each with its own before/after
verification above, ending in a quantitative match (`rho = 0.97`) against
this project's older, independently-validated directional tool on a real,
designed coverage gap. That's a real, substantive result, not yet
evidence that any of it is a *better or cheaper* way to get these signals
than existing methods at real GS scale — that comparison is what
`gs_experiment/` was built to test.
