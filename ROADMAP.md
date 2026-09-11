# ROADMAP

Forward experiment plan for the renderer-consistent sparse-GP
decomposition (see [`README.md`](README.md) for the theory and
[`gs_experiment/results/FINDINGS.md`](gs_experiment/results/FINDINGS.md)
for what's already been built and shown). Ordered by priority.

## 1. Calibration against real held-out error

`u_spatial_BQ(q) + u_SH(q)` has been shown to produce visually distinct,
sensible-looking spatial patterns (`gs_experiment/results/
sparse_gp_uncertainty.png`), but not yet checked quantitatively against
real held-out rendering error (correlation, a Gaussian-NLL-style proper
scoring rule, AUSE). The project's own established convention
(`rendering_aware_alternative_weight_risk`'s docstring; the retired
kernel-family-ablation work) is to score this honestly and report a
negative result if it comes out that way, rather than assume a
theoretically-motivated construction is automatically well-calibrated.

## 2. Fitting `lam`, the SH-coefficient prior precision

`gs_experiment/sh_directional_uncertainty.py`'s `Sigma_theta_i^-1 = lam*I
+ sum_p beta_{p,i}^2 phi(d_p)phi(d_p)^T` currently takes `lam` as a
hand-picked scalar (see `render_sparse_gp_uncertainty.py`'s own `LAM`
constant). A marginal-likelihood fit (mirroring `hyperparams.
fit_kernel_param_and_noise_pooled_nd`'s pattern, but over per-splat SH
regression instead of the position kernel) would replace that guess with
a real, data-driven value -- and is a prerequisite for priority 1's
calibration check to mean much.

## 3. `accumulate_sh_precision` performance

Rerendering every real training camera currently takes 8-11 minutes per
300k-splat scene (see `render_sparse_gp_uncertainty.py`'s own timing
prints) -- tractable for a one-off figure, not for an interactive or
per-training-step use. The per-camera loop in
`gpu_sh_directional_uncertainty.accumulate_sh_precision` is a natural
target: batching multiple cameras' gathers together (mirroring
`gpu_visibility_attribution.batched_attribute_observations`'s own
per-camera-chunked batching) rather than one Python-level camera at a
time.

## 4. Next-best-view selection

Use `u_q(pixel)`, aggregated per candidate next training view (e.g. mean
or a high percentile over that view's own visible pixels), to pick which
unobserved view to add next, and check whether it reduces held-out error
faster than a round-robin/random view schedule. Depends on priority 1
(a signal not yet shown to correlate with real error is a weak basis for
choosing views).

## 5. Training under the likelihood

Whether `u_q` (or just `u_SH(q)`, the cheaper term once priority 3 lands)
can inform densification or a loss-reweighting term during training
itself, not just post-hoc diagnosis on a finished checkpoint. An earlier,
now-retired version of this idea (gradient-vs-BQ-variance densification
triggers in `train_minimal_gsplat.py`) was tried against the OLD
single-Gaussian point-evaluation kernel and found genuinely negative
(uncontrolled splat growth without a real quality gain) -- worth
retrying against this decomposition specifically once priorities 1-2 give
a calibrated, real-precision signal to train against, not assumed to work
just because the earlier attempt used a different (and since-diagnosed)
kernel construction.
