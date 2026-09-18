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
