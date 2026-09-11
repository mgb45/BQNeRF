# gs_experiment findings

Real Gaussian-Splatting results — real `gsplat` training, real checkpoints,
real cameras. This is the primary results document for the project.

This file was recreated when the repo was rewritten around a single,
more coherent theoretical framing (see `README.md`); prior findings
(kernel-family ablation, likelihood-training experiments, directional-
coverage/floater results built on the older point-evaluation kernel)
still exist in git history if ever needed, but are not reproduced here.

## 1. Why the earlier directional-kernel construction had to be retired

The project's original formulation built a Bayesian-quadrature posterior
directly *over* the rendering integral: a query-specific renderer weight
`a_q` combined with a base kernel `k_base`, solved for BQ-optimal weights
`w* = Kxx^-1 z`, and hoped `w*` would explain (or at least resemble) real
alpha compositing's own weights `w_alpha = T_i*alpha_i`.

For a **position-only** kernel and a **mixture** `a_q` (one Gaussian per
real candidate splat, not moment-matched into one blob), this hope is
actually a theorem: with point splats, `w* = w_alpha` *exactly*, for any
kernel bandwidth (proven directly; see git history for the mixture-BQ
work this superseded). But the moment a directional kernel factor
`k_dir(d, d')` is added — needed to say anything about viewing-angle
coverage — that exact recovery breaks. In the moment vector, direction
enters as `k_dir(d_i, d_query)` (candidate vs. one fixed query
direction); in the Gram matrix, direction enters as `k_dir(d_i, d_j)`
(candidate vs. candidate). These play structurally different roles and
do not cancel — confirmed directly: exact at kappa->0, off by more than
100x at a realistic kappa. Visually, this showed up as real chromatic
speckle in the BQ-mean color render wherever direction was involved,
which is what originally prompted this investigation (a BQ mean should
never look worse than the real alpha-compositing reconstruction it's
supposedly a probabilistic refinement of).

The fix is not a better directional kernel. It's recognizing that a
splat's stored SH coefficients are not a point observation of the
radiance field at all — they're a *learned, localized basis function*,
and the representation that photometric training actually optimizes.

## 2. The renderer-consistent sparse-GP decomposition

Treat 3DGS as a sparse interdomain Gaussian process: each splat's SH
coefficients are an inducing variable of an underlying radiance-field GP,
chosen so the GP's posterior mean under the real renderer weights `b_q`
equals `C_alpha(q)` *exactly* — not approximately, not "hopefully," by
construction. Conditioning on inducing variables then gives the standard
sparse-GP predictive variance, which decomposes into two independent
terms:

    mu_q = C_alpha(q)
    u_q  = u_spatial_BQ(q) + b_q^T Sigma_theta b_q

See `README.md` for the full derivation. Two consequences worth stating
plainly: (1) the mean is now categorically incapable of the chromatic
corruption section 1 describes, since it is never solved for; (2)
`Sigma_theta_i` (the SH-coefficient posterior) does not depend on
observed *colors* at all, only on which directions were observed and how
much each observation's own alpha-compositing weight was — a pure
Fisher-information/coverage statistic, which is what the retired
directional-kernel construction was trying (and structurally failing) to
express.

## 3. What was built

- **`u_spatial_BQ(q)`**
  (`gs_experiment/pixel_uncertainty.LocalUncertaintyEngine.
  rendering_aware_alpha_risk_along_ray`'s `alpha_risk`,
  `gs_experiment/gpu_uncertainty.compute_alpha_risk_batched` for the
  whole-image batched version): the real alpha-compositing weights' own
  RKHS worst-case risk, scored under a position-only kernel — already
  established machinery, now batched and cross-validated to 1e-6 against
  the scalar path on real data (`tests/gs_experiment/
  test_gpu_uncertainty_alpha_risk.py`). A real bug was caught and fixed
  here: an earlier version of the batched risk formula omitted the
  Gram matrix's own relative-jitter term, disagreeing with the scalar
  reference by ~1% on real risk values.

- **`beta_{p,i}`, each splat's real per-training-camera alpha-compositing
  weight** (`gs_experiment/gpu_sh_directional_uncertainty.
  compute_own_alpha_weight_batched`): computed by literally rerendering
  every real training camera — querying at each observed splat's own
  projected bearing and taking its own slot in the real depth-ordered
  transmittance weights, batched per camera. Cross-validated against the
  scalar `visibility_attribution.ray_transmittance_weights` +
  `CameraSplatIndex.query` combination directly.

- **`Sigma_theta_i`** (`gs_experiment/sh_directional_uncertainty.py`'s
  Bayesian linear regression over the real SH basis, `gs_experiment/
  gpu_sh_directional_uncertainty.accumulate_sh_precision` for the
  real-checkpoint accumulation): `Sigma_theta_i^-1 = lam*I + sum_p
  beta_{p,i}^2 phi(d_p)phi(d_p)^T`. `sh_basis` was verified to reproduce
  `spherical_harmonics.eval_sh`'s own basis exactly (machine precision)
  at every SH degree 0-3, and the regression math was checked against a
  Sherman-Morrison sanity property (a single observation reduces
  posterior variance only along its own feature direction, leaving the
  orthogonal complement exactly at the prior).

- **`u_SH(q)`** (`gpu_sh_directional_uncertainty.
  compute_sh_directional_uncertainty_batched`): `sum_i beta_{q,i}^2 *
  phi(d_q)^T Sigma_theta_i phi(d_q)` at real query pixels, cross-validated
  against a scalar reference built from `LocalUncertaintyEngine.
  _along_ray_local_data`'s own real candidate/weight gathering.

`lam` (the SH-coefficient prior precision) is a free hyperparameter, not
fit in this pass — see `ROADMAP.md` item 2.

## 4. Visual result

`gs_experiment/scripts/render_sparse_gp_uncertainty.py` renders, for
three real held-out views (lego/chair/ship, `wide` ~300k-splat
checkpoints): ground truth, `C_alpha(q)` (the real renderer output,
unmodified), `u_spatial_BQ(q)`, `u_SH(q)`, and their sum
(`gs_experiment/results/sparse_gp_uncertainty.png`).

Observed on all three scenes:
- `C_alpha(q)` is visually identical to the real reconstruction, as
  guaranteed by construction — no directional color corruption of the
  kind section 1 describes.
- `u_spatial_BQ` is sharp and structure-following: it tracks fine
  geometric detail (edges, thin structures like the ship's rigging).
- `u_SH` is smoother and more spatially coherent, consistent with
  tracking real training-view angular coverage (a property that varies
  more smoothly across a surface) rather than per-splat spatial density.
- The sum combines both signals, visibly dominated by `u_spatial_BQ`'s
  sharper peaks with `u_SH`'s smoother floor visible elsewhere.

Runtime: `u_spatial_BQ` and `u_SH`'s own query-side evaluation are both
sub-second per view; `accumulate_sh_precision` (rerendering every real
training camera once per scene) took 8-11 minutes per 300k-splat scene —
see `ROADMAP.md` item 3.

Not yet done: a quantitative calibration check against real held-out
rendering error (this project's established practice — see
`quadrature.rendering_aware_alternative_weight_risk`'s own docstring for
why scoring the real renderer's own weights under a real posterior is a
well-posed, honest question) — deferred to `ROADMAP.md` item 1,
deliberately, per explicit direction to look at the renders first before
reaching for statistics again.
