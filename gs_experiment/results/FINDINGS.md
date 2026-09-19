# gs_experiment findings

Real Gaussian-Splatting results -- real `gsplat` training, real checkpoints,
real cameras. This is the primary results document for the project.

## 0. CORRECTION (supersedes the previous sections 3-4 of this file)

**Every previously recorded `u_SH` result in this project was measuring the
prior, not the data.** The per-splat SH-coefficient "posterior" was, to 13
significant figures, exactly its prior. The cause was the way the real
alpha-compositing weight `beta_{p,i}` was reconstructed, in the now-retired
`gpu_sh_directional_uncertainty.compute_own_alpha_weight_batched`. Two
independent defects compounded:

1. **Centre-pixel-only.** `beta_{p,i}` was sampled at the single pixel the
   splat's own projected centre landed on. A training observation constrains
   splat i through *every* pixel it touches, so the information it carries
   is `sum_q beta_{q,i}^2` over its whole real footprint. A splat covering
   400 pixels and one covering 1 were given comparable weight.
2. **The bearing ball counted non-occluders as occluders.** Candidates came
   from a 0.05 rad (~2.9 deg) bearing ball -- vastly wider than a pixel --
   capped at the 500 nearest, and every one of those entered the
   depth-ordered transmittance product as though it occluded the query.
   `T` collapsed to ~0 for almost every splat.

Measured directly against gsplat on the 300k-splat lego `wide` checkpoint:
the surrogate's median `beta` over genuinely visible splats was **1.1e-12**
(numerically zero) where the real rasterizer gives those same splats a
footprint-summed `sum_q beta^2` of **6.5e-5** -- a median ratio of
**2.1e18**. Downstream, the accumulated data term came out **~1e-13 times
the prior precision** in every SH band for every splat.

A third, separate defect made the two sides incommensurable even had `beta`
been right: the likelihood's observation noise variance `sigma_n^2` was
missing entirely, so the data term was in squared-colour units while the
prior was in coefficient units. No scalar `lam` can balance those. This is
the real explanation for the hand-tuning recorded in the retired
`render_angle_sweep.py`'s `LAM` comment (`lam=10` "goes blind", `lam=1e-3`
"saturates", `lam=0.1` a "checked middle ground") -- all three were
describing a prior being balanced against nothing.

**Why the existing tests did not catch it.** `tests/gs_experiment/
test_gpu_sh_directional_uncertainty.py` cross-validated the *batched*
surrogate against the *scalar* surrogate, and they agreed exactly. Both
were wrong in the same way. The lesson is recorded here deliberately:
cross-validating two implementations of the same construction establishes
only that the construction was implemented twice. The check that found this
compared against the **real rasterizer**, which is the only external
reference that exists for a quantity defined as "what the renderer does".

Retracted as a result: FINDINGS sections 3-4 below as they concern `u_SH`,
`gs_experiment/results/sparse_gp_uncertainty.png`, `angle_sweep.png`, and
`splat_count_sweep.png`. Whatever structure those figures showed in their
`u_SH` panels is the prior pushed through query-side alpha weights -- an
opacity/coverage map -- not directional coverage. `u_spatial_BQ`
(`gpu_uncertainty.py`) is a separate term and is **not** affected.

## 1. The claim: 3DGS rendering is a quadrature rule

Alpha compositing is already a weighted quadrature sum,

    C(q) = sum_i beta_{q,i} c_i(d_q),    beta_{q,i} = T_i * alpha_i

with nodes = splats and weights = `beta_{q,i}` -- and the rasterizer
computes those weights anyway, as part of rendering. So a posterior over the
splats' appearance parameters pushes forward to a per-pixel predictive
variance at *render cost*. Uncertainty is nearly free, in the literal sense
that it costs extra renders and nothing else.

The cheapest and most legible way to push the posterior through is not a
quadratic form but sampling: draw `theta^(s) ~ N(theta_hat, Sigma_theta)`,
render each draw with the real rasterizer, take the per-pixel spread. No
retraining, no model ensemble, one checkpoint, k renders.

## 2. Why the earlier directional-kernel construction was retired

(Unchanged, and still correct -- this concerns an earlier formulation than
the one section 0 corrects.)

The project's original formulation built a Bayesian-quadrature posterior
directly *over* the rendering integral: a query-specific renderer weight
`a_q` combined with a base kernel `k_base`, solved for BQ-optimal weights
`w* = Kxx^-1 z`, and hoped `w*` would explain real alpha compositing's own
weights `w_alpha`. For a **position-only** kernel and a **mixture** `a_q`,
that hope is a theorem: with point splats, `w* = w_alpha` exactly, for any
bandwidth. But adding a directional kernel factor `k_dir(d, d')` breaks it.
In the moment vector direction enters as `k_dir(d_i, d_query)`; in the Gram
matrix as `k_dir(d_i, d_j)`. These play structurally different roles and do
not cancel -- exact at kappa->0, off by more than 100x at a realistic kappa.
Visually this showed up as chromatic speckle in the BQ-mean colour render.

The fix was recognizing that a splat's stored SH coefficients are not a
point observation of the radiance field -- they are a learned, localized
basis function, and the representation photometric training actually
optimizes. That much survives section 0's correction intact.

## 3. The corrected construction

Per-splat Bayesian linear regression over the real SH basis, with all three
of section 0's defects fixed (`gs_experiment/rasterized_sh_precision.py`):

    D_i = sum_p (sum_q beta_{q,i,p}^2) phi(d_{i,p}) phi(d_{i,p})^T
    P_i = Lambda + D_i / sigma_n^2

- **`sum_q beta_{q,i,p}^2` from the real rasterizer.** gsplat renders
  `I(q) = sum_i beta_{q,i} c_i` for arbitrary per-splat features `c_i`, so
  backpropagating an image `r` gives `dL/dc_i = sum_q beta_{q,i} r_q`
  exactly. With `r` Rademacher, `E[(dL/dc_i)^2] = sum_q beta_{q,i}^2` --
  the wanted quantity, footprint-summed, under the renderer's own weights,
  with no bearing ball, no candidate cap and no surrogate compositing model.
  Independent probes ride as independent *channels* of one render (`beta` is
  geometric, hence channel-independent), so the whole accumulation is one
  forward+backward pass per training camera.
- **`sigma_n^2`** fit as the mean squared residual between the real render
  and the real training images (`estimate_noise_variance`). On lego `wide`:
  `sigma_n = 0.0129`.
- **`Lambda` by empirical Bayes, per SH band and per channel**:
  `lambda_{l,c} = 1/Var_i[theta_{i,c,k} : k in band l]`, read straight off
  the checkpoint's own coefficient population. This replaces the hand-picked
  scalar. A single scalar is badly mis-specified regardless of the other
  bugs: the l=0 (DC colour) and l=3 coefficients of a real checkpoint differ
  in natural scale by orders of magnitude. On lego `wide`, fitted
  `lambda_l0` is 1.55/1.92/3.48 (RGB) against `lambda_l2` ~10.6-11.0.

Measured effect of the correction, lego `wide`, 300k splats, 100 cameras:

| | retired KNN surrogate | rasterizer |
|---|---|---|
| data/prior precision, l=0, median | ~1e-13 | **25.7** |
| fraction of splats data-dominated | 0.000 | **0.851** |
| accumulation wall-clock | 223 s | **5.0 s** |

The corrected accumulation is both right and 45x faster: the surrogate's
cost was a KD-tree candidate search per camera, the rasterizer's is one
render per camera.

## 4. Posterior-ensemble rendering, and what it shows

`gs_experiment/scripts/render_posterior_ensemble.py` draws from the per-splat
posterior and renders each draw through gsplat. Cost on lego `wide`: 8 draws
= 323 ms of renders on a **21.5 ms** render, plus a one-time 306 ms Cholesky
per scene -- so ~15x one render per additional view once factored.

**On the full 100-view checkpoint the posterior is genuinely tight**:
per-pixel std 0.0020 (mean), 0.023 (max); four draws are visually identical
(`gs_experiment/results/posterior_ensemble.png`). This is the correct answer
and a useful sanity result -- a well-trained 300k-splat model fit to 100
views really is confident about appearance -- but it is not a figure.

**Spread appears when the conditioning set is thin.**
`gs_experiment/scripts/render_posterior_view_sweep.py` freezes ONE
checkpoint (geometry, opacities, stored coefficients, query camera and
render all untouched) and varies only how many real training cameras the
posterior is conditioned on. Strictly nested conditioning, no retraining
confound of any kind -- unlike comparing independently trained checkpoints,
where splat count, positions, opacities, learned coefficients and the
query-side weights all move at once:

| training views | l=0 data/prior (median) | per-pixel std on object | max |
|---|---|---|---|
| 100 | 25.7 | 0.0057 | 0.023 |
| 25 | 6.27 | 0.0100 | 0.041 |
| 8 | 1.55 | 0.0373 | 0.231 |
| 3 | 0.344 | 0.0430 | 0.251 |

`gs_experiment/results/posterior_view_sweep.png`: the top row's draws are
pixel-identical with a black std map; the bottom rows visibly disagree.

## 5. Open, and honestly negative so far

- **Correlation with real held-out error is weak where it counts.** On lego
  `wide` view 21, whole-frame Spearman between per-pixel posterior std and
  `|held-out error|` is 0.95 -- but that is almost entirely the
  object/white-background split, since both are ~0 on background. Restricted
  to object pixels it is **0.28** (Pearson 0.23). A whole-frame correlation
  on a NeRF-Synthetic scene mostly reports "found the silhouette" and should
  not be quoted as calibration.
- **The draws show chromatic speckle**, not structured disagreement. That is
  the block-diagonal approximation showing through: `Sigma_theta` treats
  every splat's coefficients as independent, so each splat's colour wobbles
  on its own. Photometric training only ever constrains the *sum* of
  contributions along a ray, so splats in an overlapping stack are jointly
  non-identifiable, and the true posterior has strong cross-splat
  correlations that the block diagonal discards. This is the leading
  suspect for the weak object-pixel correlation above, and the next thing
  to fix.
- **Geometry is not in the posterior.** Positions, scales and opacities are
  held fixed and only appearance is sampled, so an error of geometric origin
  (a floater in the wrong place with a confidently-fit colour) need not
  light up.

## 6. Cross-splat coupling: implemented, validated, and a negative result

`gs_experiment/coupled_sh_posterior.py` samples from the FULL joint
posterior over every splat's coefficients, never forming or inverting it.
The construction is that every operation `A` needs is a render:

- **Matvec.** `g_p^T v` is the rendered image when splat i is given the
  scalar feature `phi(d_{i,c})^T v_i`, and `sum_p g_p s_p` is that render's
  backward pass against `s`. Taking `s` to be the rendered image itself, the
  data term of `A v` is exactly the gradient of `0.5*||render||^2` -- one
  forward+backward per training camera.
- **Sampling.** `b ~ N(0, A)` is drawn in closed form (`Lambda^{1/2} r_0 +
  (1/sigma_n) sum_p g_p eps_p`, the second term another backward pass, this
  time against a white-noise image), then `A x = b` is solved by
  preconditioned CG. Since `Cov(b) = A`, `Cov(x) = A^-1` exactly.

Validated in `tests/gs_experiment/test_coupled_sh_posterior.py` by forming
`A` explicitly on a scene small enough to allow it (8 splats, degree 1, 4
cameras) and checking the sampler's empirical covariance against `A^-1`
directly -- relative Frobenius error under 15% at 1200 samples -- plus that
the operator is symmetric positive definite and that `(A^-1)_ii >=
(A_ii)^-1` holds per splat.

**On the real scene it does not help.** lego `wide`, 300k splats, 25
training views, 400 CG iterations to a 6.2e-3 relative residual:

| | block-diagonal | coupled |
|---|---|---|
| per-pixel std on object | 0.0100 | 0.0110 |
| Spearman vs `|held-out error|`, object pixels | **0.241** | **0.202** |
| wall-clock to sample 8 draws | 0.2 s | 209 s |

Coupling raises the per-pixel std by a median factor of 1.125 (90th
percentile 2.02) and makes the correlation with real held-out error
slightly WORSE. The hypothesis recorded in section 5 -- that block-diagonal
independence was what capped the error correlation -- is **not supported**.
A 1000x cost increase buys a ~12% variance correction and no calibration
gain, so the block-diagonal posterior is the one to use.

One structural detail is worth keeping, because it is the non-identifiability
showing up correctly. Each splat's MARGINAL variance must increase under
coupling (`(A^-1)_ii >= (A_ii)^-1`), but the per-PIXEL variance
`b_q^T Sigma b_q` need not: the 10th percentile of the coupled/block ratio is
0.66, i.e. it often falls. Overlapping splats are *anti*-correlated -- only
their sum along a ray is constrained -- so their errors partially cancel in
the rendered sum. The coupled posterior is saying the individual splats are
less determined than the block diagonal claims while the thing you actually
render is better determined. That is the right answer, and it is why the
block diagonal is not costing calibration here.

So the weak object-pixel correlation (section 5) remains unexplained, and
the leading suspect is now the one listed there third rather than second:
geometry is not in the posterior at all.

## 7. Geometry in the posterior: also a negative result

`gs_experiment/rasterized_opacity_precision.py` puts opacity into the
posterior, in logit space (`o_i = sigmoid(u_i)`, so a Gaussian posterior on
`u` can never leave `(0, 1)`), with the same empirical-Bayes prior rule
(`lambda_u = 1/Var_i[logit(o_i)]`) and the same probe construction:
`dC_c(q)/du_i` is what gsplat's backward returns when opacity is a leaf
requiring grad, so Rademacher probes give `F_i = sum_{q,c}(dC_c/du_i)^2` in
a few backward passes per camera (7.4 s for lego `wide`).

Mean over three real held-out lego views, object pixels only:

| conditioning | Spearman vs error | AUSE |
|---|---|---|
| SH only | **0.265** | **0.358** |
| opacity only | 0.170 | 0.365 |
| SH + opacity | 0.152 | 0.393 |

Adding geometry makes per-pixel calibration WORSE. The likely mechanism is
visible in the fit: 10%+ of splats have posterior std on logit-opacity equal
to the prior (4.73), i.e. they are entirely unconstrained -- because they are
occluded and do not affect the render. Perturbing those by +-4.7 in logit
space is far outside the regime the linearization was derived in, and can
turn a buried splat into a visible occluder, an effect the Fisher
information cannot see. So this is not evidence that geometry is irrelevant;
it is evidence that a Laplace approximation is the wrong tool for a
parameter whose posterior is this wide.

## 8. What the signal is actually for (the positive result)

Sections 6 and 7 both made calibration worse, which points at the premise
rather than the model. A posterior over splat parameters measures
**epistemic** uncertainty -- what the training views failed to determine. On
a 300k-splat checkpoint fit to 100 well-spread views, held-out error is
mostly NOT epistemic: it is misspecification and resolution limits (thin
geometry, edge aliasing), which no posterior over the fitted parameters can
see, because the data really does pin those parameters down. A weak
correlation there is the correct behaviour, not a failure.

`gs_experiment/scripts/render_epistemic_regime.py` tests that directly on
the `gap_*` checkpoints, each trained with a deliberate angular hole in its
training views, each posterior conditioned on its own real training cameras,
every one of the 30 held-out eval views scored:

| checkpoint | gap half-width | training views | per-VIEW Spearman (mean uncertainty vs mean error, 30 views) |
|---|---|---|---|
| gap_0 | 0 deg | 100 | 0.613 |
| gap_2 | 30 deg | 89 | 0.817 |
| gap_4 | 75 deg | 51 | **0.973** |

The uncertainty tracks held-out error better and better as the error becomes
more epistemic. At gap_4, view-level calibration is essentially perfect. And
inside vs. outside the hole on that checkpoint:

| | n | mean error | mean predicted std | per-pixel Spearman | AUSE |
|---|---|---|---|---|---|
| inside gap | 18 | 0.2012 | 0.0415 | 0.314 | 0.278 |
| outside gap | 12 | 0.0141 | 0.0042 | 0.254 | 0.389 |

Both error and predicted uncertainty are ~10x higher inside the hole, and
per-pixel ranking and AUSE also improve there. The error/std ratio is 4.85
inside against 3.35 outside -- the signal is underconfident by a roughly
constant factor, which is a calibration-SCALE issue, not a ranking failure,
and is the thing a fitted scale would fix.

`gs_experiment/results/epistemic_regime.png` shows this as a scatter of
per-view predicted std against per-view error, one panel per gap width: an
uninformative blob at gap_0 becoming a near-perfect line at gap_4.

Two consequences worth stating plainly:

- **Per-pixel calibration is modest everywhere (Spearman 0.25-0.31) and
  should not be claimed.** Pixel-level error is dominated by high-frequency
  misspecification in every regime tested.
- **Per-view calibration is excellent where it matters (0.97).** That is
  exactly the aggregate next-best-view selection consumes, so NBV now rests
  on a signal measured to track real error, not an assumed one.

## 9. Per-pixel calibration: achievable, and achieved (correcting section 8)

Section 8 concluded that per-pixel calibration was "modest (Spearman
0.25-0.31) and should not be claimed". That conclusion was wrong, for two
measurement errors rather than anything about the method.

**Error 1: averaging over RGB before correlating.** The predicted std and
the residual were each averaged over the three colour channels and only then
compared, which throws away the per-channel pairing. Scoring per channel
raises the same number substantially.

**Error 2: scoring against an unreachable ceiling.** Rank correlation
between a predicted `sigma` and a single realization `|eps|` is bounded well
below 1 even for a PERFECTLY calibrated sigma, because the observable is
`|eps_q| = sigma_q |z_q|` with `z_q ~ N(0,1)` independent of everything:
`Var(log|z|) = pi^2/8 ~ 1.23`, so `|z|` alone destroys rank information
unless `log sigma` varies by more than that. Comparing 0.27 against an
implicit 1.0 was meaningless. The same objection applies to AUSE, whose
usual oracle (sort by TRUE error) is likewise unattainable.

`gs_experiment/scripts/analyse_pixel_calibration.py` scores the uncertainty
against its own attainable ceiling -- obtained by simulating
`eps* ~ N(0, sigma_pred^2)` and re-running the identical metric -- and fits
the calibration on HALF the held-out views, scoring on the other half.
30 eval views per checkpoint, object pixel-channels only:

| | lego `wide` (100 views) | lego `gap_4` (75 deg hole) |
|---|---|---|
| spread of `log sigma` | 2.02 | 2.22 |
| Spearman, observed | 0.271 | 0.754 |
| Spearman, attainable ceiling | 0.517 | 0.767 |
| **fraction of attainable** | **52%** | **98%** |
| AUSE / attainable floor | 0.452 / 0.208 | 0.114 / 0.085 |

**In the epistemic regime the per-pixel ranking is 98% of everything a
perfectly calibrated uncertainty could achieve.** It is not modest; it is
very nearly optimal, and the earlier number was an artefact of how it was
scored.

What the raw signal does get wrong is SCALE, which rank metrics cannot see.
Binned calibration (bin by predicted sigma, compare against the RMS residual
actually observed in each bin) shows a strikingly CONSTANT ratio across bins
spanning a decade of sigma -- the shape is right and one number is wrong. A
two-parameter fit `sigma_total^2 = s^2 sigma_pred^2 + sigma_0^2`, fitted on
held-in views and scored on held-out ones, fixes it:

| Gaussian NLL, held-out views | `wide` | `gap_4` |
|---|---|---|
| raw, uncalibrated | 1.1e+18 | 8.7e+19 |
| constant variance (no uncertainty at all) | -1.8051 | -0.3472 |
| **posterior + fitted floor** | **-1.8841** | **-1.0298** |
| gain over constant variance | +0.079 nats | **+0.683 nats** |

`s` is 5.5 (`wide`) and 6.4 (`gap_4`): the raw posterior is underconfident by
a factor of ~6, which is why the uncalibrated NLL is astronomically bad. The
factor being nearly the same on two very different checkpoints is mild
evidence it may transfer, but that is not established.

**A spatially-varying aleatoric floor does not help.** Replacing the
constant `sigma_0^2` with a render-derived regression on `sum_i beta_{q,i}^2`
(the per-pixel weight concentration, from one probe render -- see
`rasterized_sh_precision.pixel_weight_concentration`) and on the render's
own gradient magnitude changes held-out NLL by -0.010 on `wide` and +0.034
on `gap_4`. This is the third richer model tried and the third that does not
pay (after coupling, section 6, and opacity, section 7).

**So: calibrated per-pixel uncertainty is achievable and is achieved.** The
honest qualification is about its VALUE, not its validity: the gain over
simply reporting a constant variance is large when the error is epistemic
(+0.68 nats) and small when the model is fully constrained (+0.08 nats).
That is the correct behaviour of an epistemic posterior, not a deficiency of
it -- on a checkpoint fit to 100 well-spread views there is little epistemic
uncertainty left to report, and the residual error is misspecification which
neither the posterior nor the render-derived features above can predict.

## 10. Does the calibration transfer across scenes? Only when anchored

ROADMAP item 2 asked whether the two calibration constants from section 9
are properties of the construction or of each scene.
`gs_experiment/scripts/analyse_calibration_transfer.py` settles it over all
7 NeRF-Synthetic scenes (`wide` checkpoints, 30 eval views each, fitted on
half and scored on the other half), by leave-one-scene-out: fit on six
scenes, score the seventh.

The two constants behave completely differently:

| scene | fitted `s` | fitted `sigma_0` | scene's own training `sigma_n` | `sigma_0/sigma_n` |
|---|---|---|---|---|
| chair | 3.73 | 0.0110 | 0.0078 | 1.41 |
| drums | 5.78 | 0.0516 | 0.0244 | 2.11 |
| ficus | 4.08 | 0.0330 | 0.0114 | 2.89 |
| hotdog | 5.29 | 0.0053 | 0.0065 | 0.81 |
| lego | 5.58 | 0.0191 | 0.0129 | 1.48 |
| mic | 4.97 | 0.0368 | 0.0129 | 2.86 |
| ship | 5.38 | 0.0102 | 0.0125 | 0.82 |

**`s` is nearly a constant** -- mean 4.97, sd 0.72, max/min 1.55. The
posterior is underconfident by a factor of about 5 regardless of scene,
which supports reading it as a property of the construction (the
block-diagonal approximation, the frozen geometry, and a `sigma_n` fitted on
training rather than held-out residuals all push the same way).

**`sigma_0` is not** -- max/min 9.8, from 0.0053 (hotdog, smooth and
well-resolved) to 0.0516 (drums, fine mesh and specularity). That is exactly
what it should do: `sigma_0` is the scene's own irreducible misspecification
level, and scenes differ in how hard they are.

So transferring both constants fails, and fails in the informative way:

| | mean held-out NLL | vs constant variance | short of a per-scene fit |
|---|---|---|---|
| constant variance | -1.5354 | -- | -- |
| transfer both constants | -1.5010 | **-0.034 (worse)** | +0.179 |
| **anchor `sigma_0` to the scene's own `sigma_n`** | **-1.6314** | **+0.096** | **+0.049** |
| per-scene fit (the target) | -1.6800 | +0.145 | -- |

Anchoring means fitting `sigma_total^2 = s^2 sigma_pred^2 + (c sigma_n)^2`,
where `sigma_n` is the RMS residual on the scene's OWN TRAINING images --
already computed by `estimate_noise_variance`, needing no held-out views at
all -- so the only things transferred are two dimensionless numbers
(`s ~ 5.0`, `c ~ 1.8`). Dividing by `sigma_n` cuts the across-scene spread
of the floor from 9.8x to 3.6x.

That recovers **66% of the available calibration gain with no held-out views
whatsoever**, i.e. calibrated out of the box. It rescues the case that broke
naive transfer outright (drums: -0.121 -> -0.794 against a -0.853 target)
and on two scenes it actually BEATS the per-scene fit (hotdog -2.350 vs
-2.300, lego -1.880 vs -1.863), because pooling across six scenes
regularizes constants that a 15-view per-scene fit overfits.

The remaining outlier is `mic` (-0.925 anchored against -1.228 own fit),
whose `sigma_0/sigma_n` of 2.86 is well above the mean 1.77 -- a largely
specular object, where held-out error is much worse than training residuals
imply and a single scalar `c` cannot know it.

## 11. Novelty check against the concurrent literature (2025-2026)

The method changed twice since the paper's Related Work section was last
written (section 0's correction, then the retirement of the directional
kernel for the posterior-ensemble construction in sections 1-4). A
literature pass was done against the CURRENT construction -- closed-form
per-splat Bayesian linear regression over SH coefficients, precision built
from the real rasterizer's own compositing weights via a Rademacher-probe
backward pass, sampled and pushed through the unmodified rasterizer, mean
pinned exactly to the trained checkpoint -- rather than against what the
paper draft still describes. Findings, closest-prior-art first:

- **The core "post-hoc, no retraining, mean pinned to the real render"
  framing is not unique to this project, but the reason our posterior
  precision is closed-form is not shared by the two closest photometric
  competitors.** Galappaththige et al. (`galappaththige2026predictive`,
  ECCV 2026, arXiv:2603.22786) also freeze the map and add a per-primitive,
  SH-represented channel post-hoc -- but they get it by *training* a
  Bayesian-regularized linear least-squares fit against real photometric
  residuals, i.e. their channel is supervised on the error itself. Han and
  Dumery (`han2025viewdependent`, arXiv:2504.07370) also give each splat an
  SH-valued field, fit with a hand-designed loss that suppresses uncertainty
  along observed directions and inflates it along the antipodal one. Both
  need an optimization loop after freezing. Ours needs none: `P_i = Lambda +
  D_i / sigma_n^2` (section 3) is read off the map directly -- a Fisher/
  coverage statistic of how much and from where each splat was already
  constrained -- with no loss function, no gradient descent, and no target
  to regress against. That the field has now converged on "SH per splat" as
  the representation for uncertainty three times independently (these two
  plus GAVIS/`xue2026`, already cited) is worth noting in itself: it is
  clearly the right basis to reach for, and the open ground is in how the
  coefficients are obtained, not in using SH at all.

- **The Rademacher-probe trick for `sum_q beta_{q,i}^2` is a real technique
  (Hutchinson's trace estimator) applied somewhere new (a stock differentiable
  rasterizer's autograd graph), not an original estimator.** FisherRF
  (`jiang2024`, already cited) computes a comparable per-primitive
  information quantity for active view selection, but with a bespoke CUDA
  kernel for the exact diagonal Hessian/Fisher matrix. The contribution here
  is narrower and should be described that way: the same kind of quantity
  falls out of one generic backward pass through gsplat's *existing*,
  unmodified autodiff graph, with no custom kernel, by exploiting that `beta`
  is channel-independent and pushing independent Rademacher probes through
  as independent render channels. Worth a citation to FisherRF for the
  general idea of "Fisher information from the renderer," with the
  distinction stated precisely rather than implied.

- **Posterior-ensemble rendering through an unchanged rasterizer is used
  concurrently, but always for a different part of the model.** Jia et al.
  (`jia2026rendering`, arXiv:2607.05522) sample from a Normal-Inverse-Wishart
  posterior over Gaussian *geometry* (means/covariances), fit jointly with
  training, and re-render for predictive intervals. Wu et al.
  (`wu2026perturbed`, arXiv:2603.06852, ECCV 2026) render an ensemble of
  perturbed *densities* for sparse-view X-ray CT. Horseshoe Splatting
  (`wu2026horseshoe`, ICLR 2026) puts a sparsity prior on per-splat
  *covariance* and fits it variationally, in-the-loop. None samples the
  appearance/SH coefficients of a frozen, already-trained map the way
  sections 3-4 do, and section 7's own negative result (adding an opacity
  posterior on top of the SH one made calibration WORSE, not better) is
  direct evidence that "which part of the model gets the posterior" is not
  a free choice -- geometry-in-the-loop and appearance-only-post-hoc are
  different constructions with different failure modes, not two
  instantiations of one idea.

- **The single most load-bearing finding for the current write-up --
  that whole-frame correlation is mostly the silhouette, and the real
  signal is in the epistemic regime -- has an independent, cross-modality
  replication.** Zhao et al. (`zhao2026posterior`, arXiv:2607.13682) derive
  a closed-form posterior variance for radiative (X-ray) Gaussian splatting
  from the exact linearity of X-ray attenuation in per-Gaussian density --
  the same meta-move as section 3 (exploit the renderer's linearity in the
  unknowns to get a closed form the unchanged rasterizer verifies exactly,
  rather than solving a system that does not track it), applied to a
  different physical forward model -- and title their paper on the finding
  that this variance "ranks true error on 14 of 15 scenes" overall but
  "collapses" inside the reconstructed object (median Spearman 0.11, 0/15
  scenes passing), matching a deep ensemble baseline exactly. That is our
  own section 5/8/9 arc (whole-frame Spearman 0.95 is background/object
  split; object-restricted correlation is weak except in the epistemic
  regime) reached independently, in a different modality, by a different
  team. This raises the finding from "something our implementation does"
  to "a structural property of renderer-linear closed-form posteriors,"
  which is a stronger and more citable claim than either paper makes alone.

- **Cross-scene calibration transfer (section 10) is being attacked from a
  different, complementary angle concurrently.** Chu et al.
  (`chu2026conformal`, arXiv:2609.10307, posted this month) propose
  View-Structured Conformal Prediction: split the pre-calibration scale into
  a renderer-derived spatial shape and a transferable per-view difficulty
  scalar, then use a held-out per-view quantile for a finite-sample coverage
  guarantee that holds even on an unseen scene. Section 10 instead transfers
  two dimensionless constants of a parametric Gaussian-NLL fit by anchoring
  the aleatoric floor to the scene's own training residual -- no
  distribution-free guarantee, but no held-out views either. These answer
  related but different questions (certified coverage vs. a calibrated
  predictive density) and are not competitors; wrapping the anchored
  posterior here in a conformal layer, scored against `chu2026conformal`'s
  own coverage metric, is a well-defined follow-up rather than something
  this project needs to preempt.

- **What is NOT concurrent, and should be stated as a real gap rather than
  covered by any of the above**: the item-1 roadmap direction (aggregate
  per-pixel uncertainty over candidate views for next-best-view selection)
  already has active competition -- OUGS (arXiv:2511.09397) does Gaussian-
  parameter-covariance-to-Fisher-information-to-uncertainty for object-aware
  active view selection in 3DGS. It is not cited above because it targets a
  downstream task (view selection) rather than the uncertainty construction
  itself, but it should be read before ROADMAP item 1 is attempted, since it
  may already answer the exact question that item poses.

**Net verdict.** The specific closed-form construction in sections 1-4 (SH-
coefficient posterior with a renderer-native, training-free precision, and
posterior-ensemble rendering through an unmodified rasterizer restricted to
appearance) does not have a direct hit in the literature found -- every
close paper differs in at least one of: what part of the model gets the
posterior (appearance vs. geometry vs. density), whether the uncertainty
representation is trained/fit or read off in closed form, or the rendering
modality (photometric SH vs. X-ray density). But every *individual move* --
SH-valued per-splat uncertainty, post-hoc/frozen-map construction,
posterior-ensemble rendering through the real rasterizer, Fisher information
from a renderer, exploiting renderer linearity for an exact closed form, and
even the silhouette-dominated-whole-frame-correlation finding -- has been
reached by at least one concurrent 2025-2026 paper, several within the last
three months. The honest framing for the paper draft is a *combination and
mechanism* novelty claim (this specific closed form, this specific
efficient computation of it, applied to this specific part of the model),
not a category claim ("nobody does closed-form post-hoc SH uncertainty for
3DGS") -- the category is now crowded. The paper's Related Work
(`paper/main.tex`) has been updated with the papers above; its Method,
Abstract and Introduction still describe the RETIRED directional-kernel
sparse-GP construction (last touched at commit `4bea5e7`, before sections
0-10 of this file) and need a rewrite pass to match sections 1-4 before
submission -- tracked as a new, high-priority ROADMAP item.

## 12. Pre-registered evaluation protocol, and the OUGS verdict

### 12.1 Why the protocol is frozen before the baselines exist

`gs_experiment/evaluation.py` is committed BEFORE any comparative baseline is
implemented, deliberately. Section 9 records that this project's headline
per-pixel number moved from "0.27, modest, do not claim it" to "98% of
attainable" purely because of how it was scored. A metric discovered to be
wrong after the fact is recoverable; a metric *chosen* after the fact is not.
With a comparison against the literature about to be run, every scoring
decision has to predate seeing any baseline's numbers.

The frozen decisions are listed in that module's docstring. The load-bearing
ones: per-channel never RGB-averaged; object pixels as the headline with
whole-frame reported only for comparability and never as calibration; rank
metrics quoted as a FRACTION OF ATTAINABLE (and AUSE against its attainable
floor); NLL only after a calibration fitted on disjoint views; the anchored
variance model as default; every method compared against a constant-variance
baseline; cost and retraining-requirement recorded beside every accuracy
number; >=3 seeds for anything stochastic; both regimes and both capacities.

One fairness bug was caught by the protocol's own tests before any baseline
ran: `fit_variance_model` initialised every fit at `s = 1`, which gives a
better-converged optimum to methods whose raw sigma already sits near the
residual scale and penalises baselines reported in arbitrary units (coverage
counts, Fisher traces). The initialisation is now scale-free by moments, and
`test_calibration_fit_is_invariant_to_the_methods_raw_units` holds it across
nine orders of magnitude. Had this been found after running the baselines it
would have been indistinguishable from tuning.

### 12.2 OUGS (`li2026ougs`, arXiv:2511.09397, December 2025)

Read in full. It derives uncertainty from the explicit Gaussian parameters
(position, scale, rotation, opacity, SH) as a DIAGONAL Fisher matrix
accumulated by an exponential moving average of squared gradients DURING
TRAINING, propagates it to pixels through the rendering Jacobian
(`Sigma_C(u) = J_u (diag I + lambda I)^-1 J_u^T`), and multiplies by a SAM-2
semantic mask for object-awareness. Evaluated on Mip-NeRF 360, Light-Field
and Tanks&Temples against ActiveNeRF, FisherRF, Bayes' Rays and GauSS-MI.

Three consequences for this project:

- **It does not answer ROADMAP item 1, and it is not the same construction.**
  OUGS is in-the-loop: its Fisher information only exists because it was
  accumulated across training iterations. Ours is read off a frozen,
  already-trained checkpoint. That distinction survives intact.
- **It contains no calibration numbers at all.** Its Table 1 is entirely
  downstream reconstruction quality (PSNR/SSIM/LPIPS at 5-30 views). There is
  no correlation against held-out error, no NLL, no AUSE, no coverage. An
  entire uncertainty paper with no measurement of whether the uncertainty is
  right. Combined with `zhao2026posterior` -- who does measure it, and reports
  collapse inside the object -- this says the open ground is calibration, not
  view selection, and the comparative effort should lead there.
- **Its motivating figure is section 9's silhouette finding from the other
  direction.** OUGS Figure 1 argues a global uncertainty score "is often
  dominated by complex but irrelevant background clutter" and misleads view
  selection; they fix it with a semantic mask, we fix it by restricting to
  object pixels. Independent convergence on the same hazard, worth citing as
  such.

It also creates one sharp, cheap experiment that neither paper can run alone.
Section 7 found that adding geometry (opacity) to the posterior made
CALIBRATION worse; OUGS includes all geometry parameters and wins at VIEW
SELECTION. Those are not contradictory -- they are different questions -- and
testing "does geometry help next-best-view even though it hurts calibration?"
on a common harness is a genuinely novel comparison.

### 12.3 Consequence for ROADMAP

Next-best-view is DEMOTED from item 1. The field is crowded (ActiveNeRF,
FisherRF, Bayes' Rays, GauSS-MI, OUGS, and `xue2026`), competing head-on
means reimplementing four strong baselines on three datasets this project
does not currently use, and the differentiator -- calibration -- is measurably
unoccupied. The comparative effort leads with calibration against the
post-hoc appearance-uncertainty cluster, on the frozen protocol above.

## 13. Comparative results: epistemic and aleatoric uncertainty are different quantities

First run of the pre-registered protocol (section 12.1) against reimplemented
competitors, on lego, identical checkpoints, identical held-out views, object
pixels. `gs_experiment/baselines.py`, `scripts/run_comparison.py`.

### Saturated regime -- `wide`, 100 training views, 30 held-out views

| method | Spearman | AUSE | per-view Sp | NLL gain vs constant | cost |
|---|---|---|---|---|---|
| ours (SH posterior) | 0.264 | 0.458 | **0.644** | +0.069 | 9.1 s |
| residual-supervised SH | **0.353** | **0.407** | -0.027 | **+0.147** | 103.5 s |
| uniform-coverage SH | 0.085 | 0.617 | 0.537 | +0.004 | 8.5 s |
| weight concentration | 0.010 | 0.708 | 0.564 | -0.027 | 0.3 s |
| render gradient | 0.249 | 0.454 | 0.209 | +0.023 | 0.1 s |

**We lose per-pixel here, and it goes in the paper as a loss.** The
Galappaththige-style baseline is supervised on the very residual the
evaluation scores, and it wins on per-pixel Spearman, AUSE and NLL, at 11x
the cost. More awkward still, a 0.1 s render-gradient edge detector matches
our AUSE (0.454 vs 0.458). On a checkpoint fit to 100 well-spread views,
per-pixel error is largely edge misspecification, and a method that fits
misspecification directly will win at predicting it.

But it has **no per-view signal at all** (-0.027 against our 0.644).

### Epistemic regime -- `gap_4`, 51 training views with a 75 deg hole

| method | Spearman | AUSE | per-view Sp | NLL gain vs constant | cost |
|---|---|---|---|---|---|
| **ours (SH posterior)** | **0.750** | **0.117** | **0.973** | **+0.662** | 7.4 s |
| residual-supervised SH | **-0.452** | 1.367 | **-0.923** | -0.048 | 44.9 s |
| uniform-coverage SH | 0.536 | 0.250 | 0.961 | +0.214 | 7.5 s |
| weight concentration | 0.243 | 0.494 | 0.951 | +0.026 | 0.3 s |
| render gradient | 0.035 | 0.747 | -0.807 | -0.048 | 0.1 s |

The table inverts, and not by a little. Ours reaches 97% of attainable
per-pixel ranking and 0.973 per-view. The supervised baseline goes
**ANTI-correlated at both levels** (-0.452 per-pixel, -0.923 per-view), and
its calibration fit degenerates to the constant-variance solution (`s = 0.00`),
i.e. the protocol independently concludes its sigma carries no usable
information.

The mechanism is not subtle. Fitted on 51 training views that by construction
exclude the 75 deg cone, it learns "error lives at edges, in the regions I
saw", and therefore predicts LOW uncertainty inside the gap, which is exactly
where the error is largest. It is confidently wrong precisely where it
matters. The render-gradient floor fails the same way (-0.807 per-view) for
the same reason.

### What this means

**Per-pixel and per-view uncertainty are measuring different things, and no
method tested does both.** A construction fitted to observed residuals learns
an ALEATORIC map -- where error sits within a view -- and by construction
cannot learn anything about directions the training views never covered. An
epistemic posterior does the reverse. This is not a ranking of methods; it is
a statement about what each quantity is for:

- *"Where in this render should I not trust the pixels?"* on a well-observed
  scene: a supervised aleatoric map wins, at 11x the cost.
- *"Which view is unreliable? Where should I capture next? Has this scene
  been covered?"*: only the epistemic posterior carries signal at all, and
  the supervised map is actively harmful.

Two secondary results worth keeping:

- **The `beta^2` Fisher weighting earns its place.** Uniform-coverage --
  our construction with every observation weighted equally instead of by its
  real compositing weight -- drops from 0.264 to 0.085 in the saturated
  regime and from 0.750 to 0.536 in the epistemic one. Plain angular coverage
  is a decent epistemic signal but a clearly worse one, which answers the
  Han-style mechanism question directly.
- **The render-gradient floor matching our AUSE on `wide` is evidence FOR the
  dissociation, not against the method.** A trivial edge detector matching a
  posterior at per-pixel sparsification says the thing being predicted there
  is edge misspecification. The same detector is anti-correlated the moment
  real epistemic error exists.

One consequence for how any of this can be used: **you cannot tell which
regime you are in from the aleatoric side.** The supervised baseline scores
well on `wide` and catastrophically on `gap_4` while reporting nothing that
distinguishes the two. The epistemic posterior's own magnitude does
distinguish them (FINDINGS section 8: per-view Spearman 0.61 -> 0.97 as the
gap widens, mean std rising ~10x inside the hole).

### Protocol note

A display bug surfaced here and is fixed: "fraction of attainable" is now
reported as `n/a` for an anti-correlated predictor. A negative fraction is a
category error rather than a weak score -- such a method is worse than
uninformative -- and the signed raw Spearman carries the verdict.

## 14. Cross-scene comparison, and a correction to section 13

Section 13 reported the saturated-regime comparison from **lego alone** and
concluded "we lose per-pixel, and it goes in the paper as a loss". Run across
all 7 NeRF-Synthetic scenes plus **bonsai** (a real Mip-NeRF 360 capture,
2.07M splats, 262 training views, 30 held-out views), that conclusion does
not hold: lego is one of the scenes we lose on, not a representative one.
`scripts/summarise_comparison.py`, aggregating only what the frozen protocol
wrote.

| metric (8 scenes) | ours | residual-supervised | ours wins |
|---|---|---|---|
| per-pixel Spearman, mean | **0.313** | 0.267 | 4/8 |
| **per-view Spearman, mean** | **0.757** | **-0.037** | **8/8** |
| AUSE, mean (lower better) | **0.392** | 0.481 | 5/8 |
| NLL gain over constant, mean | **+0.133** | **-0.193** | 7/8 |
| cost, mean | **10.2 s** | 142.9 s | **14x** |

Per-pixel is a genuine split: we lose on chair, drums, hotdog and lego, win
on ficus, mic, ship and bonsai, with a slightly higher mean. The single-scene
claim in section 13 was an over-generalisation from n=1 and is withdrawn.

What survives, and strengthens:

- **Per-view separation is total: 8/8, mean 0.757 against -0.037.** The
  supervised baseline's mean per-view correlation is NEGATIVE across scenes.
  It is not a weaker view-level signal; it is not a view-level signal at all.
- **On average the supervised baseline is worse than reporting a constant**
  (mean NLL gain -0.193 against our +0.133, ours winning 7/8). It collapses
  hardest on `mic` (-0.935) and `ship` (-0.715) -- the specular object and
  the one with thin rigging, i.e. exactly where training-view residuals
  mislead about held-out behaviour. `mic` is also the scene section 10
  flagged as the outlier for calibration transfer (`sigma_0/sigma_n = 2.86`),
  which is the same phenomenon seen from the calibration side.
- **The real capture behaves like the synthetic scenes.** bonsai: ours
  0.266 per-pixel / 0.709 per-view / +0.016 NLL; supervised 0.114 / -0.370 /
  -0.000. Every other number in this project is NeRF-Synthetic, so this is
  the first evidence the construction is not an artefact of synthetic data.
- **The `beta^2` Fisher weighting is decisively justified.** The
  uniform-coverage ablation wins 0 of 8 on every metric, means 0.070
  per-pixel against our 0.313, and goes negative per-pixel on chair, hotdog
  and ship. Plain angular coverage does retain a per-view signal (mean 0.551)
  -- which makes sense, since "seen from few directions" is genuinely
  view-level information -- but loses the per-pixel signal almost entirely.
- **The floors fail as floors should.** Weight concentration has a negative
  mean per-pixel correlation (-0.045). Render gradient is the only
  competitive floor (mean 0.219 per-pixel, 0 outright wins, 1 AUSE win on
  chair) and its per-view mean is 0.228 with two negative scenes -- it finds
  edges, which correlates with error within a view and says nothing about
  which view is bad.

`gs_experiment/results/comparison_summary.png` shows the two panels side by
side: within-view (mixed) against across-view (one method consistently high,
the supervised baseline negative on 5 of 8 scenes).

The section 13 reading is unchanged and now rests on 8 scenes instead of 1:
a construction fitted to observed residuals learns an aleatoric map that is
competitive within a view and carries no information across views; an
epistemic posterior does the reverse. The correction is only to the strength
of the per-pixel claim, which is a tie rather than a loss.

## 15. All-parameter Fisher (FisherRF/OUGS-style): appearance-only is a win, not a simplification

`gs_experiment/rasterized_parameter_fisher.py` builds the nearest competitor
in construction: a diagonal Fisher over EVERY Gaussian parameter -- position,
log-scale, quaternion, logit-opacity and SH -- estimated by the same
Rademacher probes, sampled and rendered through the same rasterizer, scored
by the same frozen protocol on the same checkpoints. The only difference from
ours is which parameters carry the posterior.

Across the 8 saturated scenes:

| metric | ours (SH only) | all-parameter Fisher | ours wins |
|---|---|---|---|
| per-pixel Spearman, mean | **0.313** | 0.090 | 8/8 except chair |
| per-view Spearman, mean | **0.757** | 0.279 | 8/8 |
| AUSE, mean (lower better) | **0.392** | 0.557 | 7/8 |
| NLL gain over constant, mean | **+0.133** | +0.027 | 8/8 |
| cost, mean | **10.2 s** | 16.8 s | -- |

and in the epistemic regime (lego `gap_4`) the gap is far wider: ours
0.750 / 0.973 / +0.662 against 0.130 / 0.253 / -0.048.

The all-parameter posterior goes NEGATIVE per-pixel on drums (-0.153) and mic
(-0.066), and negative per-view on drums (-0.459). Its only win anywhere is
per-pixel on chair (0.302 against our 0.247).

**This corroborates FINDINGS section 7 in a controlled setting.** There,
adding an opacity posterior to ours made calibration worse, and the suspected
mechanism was that a Laplace/linearised treatment is invalid for parameters
whose posterior is wide -- an unconstrained splat gets a prior-sized
perturbation far outside the linear regime, and a buried splat perturbed into
visibility changes the render in a way no Jacobian anticipated. Extending
that from opacity alone to all of geometry reproduces the same failure and
amplifies it. Restricting the posterior to appearance is therefore a
*modelling decision that pays*, not a simplification we should apologise for,
and that is now measured against the competing premise rather than asserted.

**What this does NOT show.** OUGS's own claim is about next-best-view
selection, and its Table 1 reports reconstruction quality, not calibration
(section 12.2). It also adds object-aware semantic masking, which is absent
here. So the finding is precisely "an all-parameter diagonal Fisher posterior
calibrates worse than an SH-only one under identical estimation and scoring",
not "OUGS is wrong at the task it claims". Whether geometry helps VIEW
SELECTION even though it hurts calibration is still open, and is exactly the
experiment ROADMAP item 2 keeps.

Two caveats on the reimplementation, stated so the comparison is not read as
stronger than it is: OUGS accumulates its Fisher during training as an EMA of
squared gradients, which is unavailable post-hoc, so the same diagonal is
estimated directly on the frozen map here (a cleaner estimator of the same
quantity, but not identical); and FisherRF uses a bespoke exact CUDA kernel
where this is an unbiased Monte-Carlo estimate.

## 16. Deep ensemble: the strongest baseline, and its advantage is largely a capacity artefact

The deep ensemble -- N independent trainings, per-pixel spread across members
-- is the reference epistemic baseline and the one `zhao2026posterior`
benchmarks against. It is also the only baseline here that requires
retraining, so its cost belongs in every comparison.

Matching capacity forced a design choice. An ensemble's members are DIFFERENT
checkpoints, so it cannot be dropped into a comparison that runs every
post-hoc method on one shared map: its mean render, and therefore its
residuals, would differ from every other method's, and the protocol requires
an identical target. So `scripts/run_ensemble_comparison.py` trains N members
on identical views differing only in seed, uses member 0 as the shared
reference every post-hoc method is built on and scored against, and takes the
ensemble's uncertainty as the spread across members. Because that slightly
disadvantages the ensemble -- its spread is centred on the ensemble mean, not
member 0 -- a supplementary row scores it against its own mean too. Both were
committed to before either was seen.

### At reduced capacity the ensemble wins clearly (7 scenes)

| metric | ours | deep ensemble (5x) | ensemble wins |
|---|---|---|---|
| per-pixel Spearman, mean | 0.219 | **0.452** | **7/7** |
| per-view Spearman, mean | 0.705 | **0.811** | 4/7 |
| AUSE, mean (lower better) | 0.540 | **0.278** | -- |
| NLL gain over constant, mean | +0.069 | **+0.272** | **7/7** |
| cost, mean | **7.7 s** | 366 s | 47x |

The mechanism is structural rather than a tuning gap. At 17k-100k splats the
dominant error source is REPRESENTATIONAL -- too few splats, and where they
land depends on the seed. An ensemble sees exactly that, because different
seeds place splats differently. Our posterior conditions on fixed geometry
and measures appearance uncertainty only; it cannot see "there are not enough
splats here". That is a real limitation of the construction.

### But the comparison was run at the capacity that most favours it

Our own per-pixel Spearman improves by +0.101 on every one of the 7 scenes
going from the reduced budget to the 300k-splat `wide` recipe. If the
ensemble's edge comes from representational error, it should shrink where the
map can actually represent the scene. Tested directly, 5 members at full
`wide` capacity (30k iterations, 300k splats; 1835 s and 2613 s of training):

| scene | capacity | ours | ensemble | per-pixel gap | NLL gap | cost ratio |
|---|---|---|---|---|---|---|
| lego | reduced | 0.182 | 0.447 | 0.265 | 0.100 | 102x |
| lego | **full** | 0.245 | 0.391 | **0.146** | **0.047** | 240x |
| mic | reduced | 0.203 | 0.511 | 0.307 | 0.238 | 36x |
| mic | **full** | 0.319 | 0.427 | **0.108** | **0.035** | 290x |

The per-pixel gap shrinks by 45% (lego) and 65% (mic); the NLL gap by 53% and
85%. Per-view at full capacity is a wash -- lego 0.610 against 0.710, mic
**0.785 against 0.758**, i.e. ours ahead on mic where at reduced capacity the
ensemble led 0.965 to 0.617.

### The honest reading

A 5-member deep ensemble remains better calibrated per-pixel than our
posterior at both capacities. That is not in dispute and should be stated
plainly. But:

- the margin is strongly capacity-dependent and collapses by roughly half to
  five-sixths at production capacity, which localises what an ensemble adds
  (splat-placement and optimisation variability) and shows that component is
  small once the map is adequate;
- at full capacity the remaining gap is 0.11-0.15 Spearman and 0.035-0.047
  nats, for **240-290x the compute**, and the ensemble needs five trainings
  where ours needs a checkpoint you already have;
- per-view, the aggregate that next-best-view selection actually consumes,
  the two are indistinguishable at full capacity.

The supplementary own-mean row is mixed and is reported as such: on mic it is
much stronger than the shared-reference row (0.502 per-pixel, +0.386 NLL
against 0.427 and +0.100), on lego slightly weaker (0.338 against 0.391).
Scoring an ensemble against its own mean flatters it where members disagree
about the mean itself, which is exactly the representational variability
above.

`gs_experiment/results/ensemble_capacity.png` shows the convergence.

This also settles the capacity question ROADMAP item 1 raised: capacity is a
real factor, not a confound to be averaged over, and a comparison run only at
a reduced budget would have reported a conclusion that does not survive at
the operating point anyone would actually use.

## 17. The epistemic regime across all scenes, at full capacity

Sections 13-16 compared methods where error is largely misspecification. This
is the other regime, built properly: `scripts/build_gap_checkpoints.py`
removes every training view within 75 degrees of one reference direction and
RETRAINS at the project's full `wide` recipe -- retraining because section 8
showed a frozen map has no epistemic error to predict, and at full capacity
because section 16 showed a reduced budget reports conclusions that do not
survive. The cone removes a different fraction per scene (hotdog keeps
15/100 views, ficus 62/100), since it depends on how the capture orbit sits
relative to the reference direction.

Mean over 7 scenes, object pixels, frozen protocol:

| method | per-pixel Sp | per-view Sp | AUSE | NLL gain | wins (of 4 metrics x 7 scenes) |
|---|---|---|---|---|---|
| **ours (SH posterior)** | **0.661** | **0.951** | **0.183** | **+0.456** | **27/28** |
| uniform-coverage SH | 0.388 | 0.896 | 0.356 | +0.135 | 1/28 |
| weight concentration | 0.151 | 0.719 | 0.531 | +0.039 | 0 |
| all-parameter Fisher | -0.007 | -0.057 | 0.672 | +0.001 | 0 |
| render gradient | -0.018 | -0.849 | 0.733 | -0.008 | 0 |
| residual-supervised SH | **-0.373** | **-0.843** | 1.090 | -0.008 | 0 |

Ours wins 7/7 on per-pixel Spearman, AUSE and NLL gain, and 6/7 on per-view
(uniform-coverage edges ship by 0.001). The two methods that fit OBSERVED
structure -- the residual-supervised channel and the render-gradient floor --
are strongly ANTI-correlated at view level (-0.843 and -0.849 mean), and the
residual-supervised one is anti-correlated per-pixel on 6 of 7 scenes. They
are not merely uninformative here; they point confidently at the wrong
places, because what they learned is where error sat in the regions the
training views covered, and the question being asked is about the region
those views excluded.

The all-parameter Fisher baseline collapses to approximately zero on every
metric (mean per-pixel -0.007, per-view -0.057), reinforcing section 15: the
geometry terms do not merely fail to help, they wash out the appearance
signal that does work.

This is the strongest result in the project. It is also the regime the method
is FOR, and section 13-14's saturated-regime numbers should be read as the
honest boundary of the claim rather than the claim itself.

### A caveat on our own metric

On 3 of 7 scenes our fraction-of-attainable exceeds 1.0 (ficus 1.09, hotdog
1.07, ship 1.04). That is not a method scoring better than perfectly; it is a
limitation of how the ceiling is estimated. The ceiling simulates
`eps* ~ N(0, sigma^2)` -- residuals independent across pixels and Gaussian
given sigma. Real residuals are neither: they are spatially correlated and
heavier-tailed, which makes the ranking task genuinely easier than the
simulation assumes. So the ceiling is a MODEL-BASED estimate of attainable
rank correlation, not a hard bound, and a value near or slightly above 1.0
should be read as "this uncertainty ranks about as well as its own spread
allows", not as a paradox. This does not affect any cross-method comparison,
which the protocol already directs to raw Spearman, AUSE and NLL.

## 18. Conformal coverage: everyone covers, the question is how wide

`gs_experiment/conformal.py` wraps every method in split conformal at a 90%
target, calibrated on even-indexed held-out views and scored on odd-indexed
ones. `chu2026conformal` is not a competitor -- it answers a different
question (distribution-free finite-sample coverage rather than a calibrated
density) and composes with any sigma -- so the comparison is not "who covers"
but "how wide an interval each sigma needs to buy the same guarantee".

Mean over 7 scenes, 90% target:

| | | EPISTEMIC (75 deg gap) | | | SATURATED (all views) | |
|---|---|---|---|---|---|---|
| method | coverage | median width | x ours | coverage | median width | x ours |
| **ours (SH posterior)** | 89.4% | **0.176** | 1.0 | 89.3% | **0.069** | 1.0 |
| uniform-coverage SH | 90.8% | 0.278 | 1.6 | 88.9% | 0.079 | 1.1 |
| weight concentration | 91.0% | 0.330 | 1.9 | 87.7% | 0.072 | 1.1 |
| all-parameter Fisher | 91.2% | 0.468 | 2.7 | 86.0% | 0.071 | 1.0 |
| render gradient | 90.9% | 0.725 | 4.1 | 87.3% | 0.126 | 1.8 |
| residual-supervised SH | 91.3% | 1.970 | **11.2** | 84.3% | 0.065 | 1.0 |

**In the epistemic regime, our uncertainty buys the same 90% guarantee with
an interval 11x narrower than the residual-supervised baseline**, 4.1x
narrower than the render-gradient floor and 2.7x narrower than the
all-parameter Fisher posterior. That is the practically useful form of the
whole comparison: if you want an interval you can act on, this is how much
tighter it gets.

**In the saturated regime conformal cannot separate the methods at all** --
every method lands within 1.0-1.1x of ours except the render-gradient floor.
Consistent with sections 13-14: where there is little epistemic error to
find, there is little for an epistemic posterior to contribute, and the
honest statement is that the methods are equivalent there rather than that
ours wins.

### Coverage deviation is itself diagnostic

Split conformal guarantees marginal coverage under exchangeability of
calibration and test points. Held-out VIEWS are not exchangeable at pixel
level -- residuals are spatially correlated and views differ in difficulty --
so coverage can drift from nominal, and how far it drifts turns out to track
whether a method's sigma follows per-view difficulty. In the saturated
regime the residual-supervised baseline under-covers at 84.3% and the
all-parameter Fisher at 86.0%, the two methods with the weakest per-view
correlation (sections 14-15); ours sits at 89.3%, closest to nominal of any
method in either regime. A method whose sigma is blind to which view is hard
will systematically under-cover on the hard ones, and conformal exposes that
without being told about views at all.
