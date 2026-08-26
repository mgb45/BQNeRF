# bq_splat findings (summary)

Toy-scale (1D/2D synthetic signals, pure numpy/scipy, no GPU) validation
of the Bayesian-quadrature math before it was ported to real Gaussian
Splatting data in `gs_experiment/` — where the same questions get
re-asked and answered on real scenes (see
[`gs_experiment/results/FINDINGS.md`](../../gs_experiment/results/FINDINGS.md),
the primary results document).

This file is a short, current-conclusions summary. The complete
chronological account — every bug, every intermediate number, the full
reasoning behind each fix — is preserved in
[`ARCHIVE_FULL_LOG.md`](ARCHIVE_FULL_LOG.md); each section below links the
matching archive section(s).

## The core math is correct

RBF and Matérn-3/2 kernel/quadrature formulas are unit-tested against
`models/nerf.py`'s exact closed-form RBF formula (see
[archive/original_nerf_prototype](../../archive/original_nerf_prototype/))
and against numerical integration. *(Archive §1, §4.)*

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
*(Archive §10.)*

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
bandwidth-fitting results. *(Archive §1, §5, §7.)*

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
(§S2 there). *(Archive §2, §3, §6.)*

One real numerical-conditioning lesson from this work: irregular node
placement can push the Gram matrix condition number past 1e18 with a
fixed jitter; a jitter scaled to the kernel's own diagonal fixes it and
materially changes downstream numbers (an earlier, uncorrected run showed
a spuriously low RBF calibration correlation purely from this). *(Archive
§4.)*

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
`gs_experiment/pixel_uncertainty.py`'s `LocalUncertaintyEngine`. *(Archive
§8.)*

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
work (§S5 there) later builds on and stress-tests on real geometry.
*(Archive §9.)*

## Bottom line

All of the above is a **qualified pass**: the ported math is correct, the
raw-accuracy gap is understood (a fixable bandwidth-mismatch issue, not a
fundamental limitation) and no longer the claim being defended, posterior
variance is reasonably calibrated and responds to under-resolved regions
in the expected way, the computational cost concern that motivated a
possible GPU rewrite was resolved on CPU alone, and the directional
extension is mathematically real at toy scale. None of this is evidence
yet that any of it is a *better or cheaper* way to get these signals than
existing methods at real GS scale — that comparison is what
`gs_experiment/` was built to test. See `ARCHIVE_FULL_LOG.md`'s own
"Bottom line" section for the original, more detailed version of this
paragraph.
