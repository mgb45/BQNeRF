# gs_experiment findings

Real Gaussian-Splatting results — real `gsplat` training, real checkpoints,
real cameras. This is the primary results document for the project.

This file was recreated after the repo was deliberately cut down to its
three current headline results (see `README.md`'s "What's been tested"
and the figures under `gs_experiment/results/`); the older, larger
findings log that motivated some of those results (kernel/window_radius
sensitivity, opacity-floor fixes, etc.) still exists in git history if
ever needed, but is not reproduced here. Numbering below starts fresh at
1 for this repo's current scope.

## 1. Training directly under the likelihood (ROADMAP.md item 1)

**Question**: every result kept in this repo computes BQ variance *after*
training, off a checkpoint trained by ordinary photometric loss and
gradient-triggered densification. Does the BQ posterior help *during*
training — either as the densification trigger, or as an auxiliary
uncertainty-weighted loss term — rather than only diagnosing a finished
checkpoint?

**Setup**: `gs_experiment/scripts/likelihood_training_experiment.py`
(new), trains five matched variants of `train_minimal_gsplat.train()` on
the same real scene, same seed (0) and same every other hyperparameter,
via `gs_experiment/local_runs/lego_prepared/narrow` (a real 12-view
training pool) and evaluates each on (a) those same 12 training views and
(b) a disjoint 15-view subsample of `lego_prepared/eval` (the official
NeRF-Synthetic lego test split). Recipe: `n_splats=2000` initial,
`max_splats=8000`, `n_iters=3000`, `sh_degree=3`, white background
(matching lego's white-composited images), position-LR decay to `2e-5`,
SSIM-weighted photometric loss, BQ hyperparameters at this project's
already-fitted lego-scale values (`bq_sigma=0.0694`,
`bq_window_radius=0.08` — not `train()`'s own 0.9/1.6 defaults, which its
docstring flags as belonging to a different, thin-rod scene family).
`nll_weight=0.02` where the NLL term is on, `nll_interval=100`. Five
variants:

| variant | densify_criterion | nll_weight | other |
|---|---|---|---|
| baseline | gradient | 0.0 | — |
| bq_densify | bq_variance | 0.0 | — |
| nll_loss | gradient | 0.02 | — |
| bq_densify+nll | bq_variance | 0.02 | — |
| bq_densify_floor | bq_variance | 0.0 | `bq_densify_min_opacity=0.05` |

**A real bug found and fixed along the way**: the first run of this
experiment reported near-nonsense PSNRs (~1.7-1.9dB for every variant)
despite training logs showing real per-iteration training-view PSNR in
the 26-28dB range. Cause: `train_minimal_gsplat.mean_psnr()` called
`render_reconstruction.render_views()` with no `background_color`
argument, silently defaulting to `render_views`'s own dark
(0.05, 0.05, 0.05) default — but this scene trains against a white
background (matching its white-composited images), the same mismatch
`real_directional_coverage_experiment.py`'s own comments already flagged
as "matters most" for lego-family scenes. Every pixel the splats didn't
opaquely cover then read as a large, spurious error against a mismatched
background, tanking PSNR by ~25dB without the reconstruction itself being
wrong. Fixed by adding a `background_color` parameter to `mean_psnr`
(default unchanged, so existing dark-background callers like
`run_nll_experiment` are unaffected) and threading the matching white
background through from `likelihood_training_experiment.py`. Recorded
here per this project's own norm of reporting real pipeline bugs, not
just final numbers — this one would have silently produced a "both
mechanisms look useless" false negative if not caught.

**Results** (matched budget, `max_splats=8000`, `n_iters=3000`, 12 train /
15 held-out views):

| variant | n_splats | train PSNR | held-out PSNR |
|---|---:|---:|---:|
| baseline | 3108 | 26.65dB | 17.48dB |
| bq_densify | 3333 | 27.48dB | 17.78dB |
| nll_loss | 3095 | 26.51dB | 17.27dB |
| bq_densify+nll | 3339 | 27.51dB | 17.77dB |
| bq_densify_floor | 1181 | 24.94dB | 17.64dB |

Deltas vs. baseline:

| variant | train PSNR | held-out PSNR | n_splats |
|---|---:|---:|---:|
| bq_densify | +0.83dB | +0.29dB | +225 |
| nll_loss | -0.14dB | -0.21dB | -13 |
| bq_densify+nll | +0.86dB | +0.28dB | +231 |
| bq_densify_floor | -1.70dB | +0.16dB | -1927 |

**Honest reading**:

- **BQ-variance densification is a real, if modest, win.** Swapping the
  densification trigger from gsplat's view-space gradient to closed-form
  BQ position-only variance gives +0.83dB train / +0.29dB held-out at a
  matched `max_splats` budget, using ~7% more splats to get there. Small,
  but consistent: it shows up almost identically whether the NLL term is
  on or off (bq_densify: +0.83/+0.29 vs. bq_densify+nll: +0.86/+0.28),
  so it isn't an artifact of one run's noise. This is the first result in
  the project where the BQ posterior actively *changes* a training
  outcome rather than just describing a finished one.
- **The NLL auxiliary loss term is a no-op-to-mild-negative on its own.**
  `nll_loss` (gradient densify + NLL on) is *worse* than baseline on both
  train (-0.14dB) and held-out (-0.21dB) PSNR, at a near-identical splat
  count — so the uncertainty-weighted reweighting of the photometric loss
  did not help generalization at this grid resolution/cadence
  (`nll_grid_res=12`, `nll_interval=100`) and this loss weight (0.02).
  Layering it on top of BQ-densify (`bq_densify+nll` vs. `bq_densify`)
  changes nothing measurable (+0.86 vs. +0.83dB train, +0.28 vs. +0.29dB
  held-out — within run-to-run noise) — the win in the combined variant
  is fully attributable to the densification change, not the loss term.
  Given `train()`'s own docstring already notes the NLL term does not
  differentiate through the BQ posterior itself (gradients flow through
  `pred` only, not through `var`), this is a plausible, honest limit of
  this first installment rather than a surprise: it can reweight *which*
  pixels matter but cannot yet teach the optimizer *why* a region is
  uncertain in a way that would change splat geometry beyond what
  densification already does.
- **The opacity floor on BQ-densify (`bq_densify_min_opacity`) is a real
  efficiency lever, not a quality tax.** `bq_densify_floor` ends with
  62% fewer splats than baseline (1181 vs. 3108) yet held-out PSNR is
  *not* worse — it's marginally better than baseline (+0.16dB) despite
  training-view PSNR dropping notably (-1.70dB). Read together, this
  suggests the un-floored `bq_densify` variant's extra splats (and the
  floored variant's large train/held-out PSNR gap) partly reflect
  overfitting to the 12 training views rather than generalizable detail:
  excluding low-opacity, likely-empty-space splats from competing for
  densification budget produces a substantially sparser reconstruction
  that generalizes just as well. This directly answers one of
  ROADMAP.md item 1's open design questions (does the floor trade
  splat-count growth against held-out quality) — here, no: it buys a
  large splat-count reduction at essentially zero held-out cost.
- **Net honest takeaway**: at this scene scale and iteration budget, BQ
  variance is a better densification signal than the raw view-space
  gradient (small but real quality win, or a much bigger splat-budget win
  if the opacity floor is used), but the auxiliary NLL loss term as
  currently implemented is not pulling its weight and needs either a
  finer/more frequent grid, a different weighting, or differentiating
  through the BQ posterior itself (as `train()`'s docstring already flags
  as the real next step) before it can be trusted to help rather than
  mildly hurt.

**Concrete next untested step**: differentiate the NLL term's `var`
through the BQ posterior itself (currently detached — see `train()`'s
`nll_weight` docstring) and re-run this same five-variant comparison; if
that doesn't move `nll_loss`/`bq_densify+nll` past the current
no-op/mild-negative result, the auxiliary-loss mechanism (as opposed to
the densification-trigger mechanism, which already shows a real win)
should be considered a dead end for this project rather than iterated on
further. A second, cheaper next step worth trying first: repeat this
comparison on `lego_prepared/wide` (100-view pool) to check whether the
BQ-densify win and the NLL no-op both hold at a stronger-coverage
baseline, or are partly artifacts of the narrow 12-view pool's own
overfitting risk.

Scripts/data: `gs_experiment/scripts/likelihood_training_experiment.py`;
checkpoints and per-variant eval scene_dirs under
`gs_experiment/local_runs/likelihood_experiment/` (gitignored).

## 2. Alternative kernels (ROADMAP.md item 2)

**Question**: `gs_experiment/kernels.py` had two families (`RBFKernel`,
`MaternKernel`) behind the same `Kernel` interface. Is a third family
actually worth offering, and does kernel choice matter for this method's
two headline properties — does posterior variance track real splat
sparsity, and is it calibrated against real held-out rendering error?

**New kernel added**: `RationalQuadraticKernel` (`gs_experiment/kernels.py`),
`k(r) = (1 + r^2/(2*alpha*l^2))^(-alpha)` with `alpha` fixed at 1.0 (a
comparatively heavy-tailed choice — the kernel decays as `1/r^2`, not
RBF's `exp(-r^2)`) and length-scale `l` as the single free parameter, the
same single-scalar contract `RBFKernel`/`MaternKernel` already satisfy for
`hyperparams.fit_kernel_param[_pooled_nd]`. Chosen over a periodic kernel
because this project's scene content (a mechanical lego model, not
repeating texture) has no periodic structure to exploit, whereas rational
quadratic is literally a continuous scale-mixture of RBF bandwidths —
plausible for a scene whose splat density (and hence natural local
lengthscale) varies a lot between the sparse background and the
densely-packed model. `v`/`vv` follow `MaternKernel`'s exact pattern
(`scipy.integrate.quad` with the `breakpoints` trick). Unit tests in
`tests/gs_experiment/test_kernels.py`/`test_kernels_nd.py` (PD gram-matrix
check, `v`/`vv` vs. numerical integration, self-similarity `k(0)=1`,
convergence to RBF as `alpha -> infinity`, `ProductKernel` separability).

**Setup**: new `gs_experiment/kernel_family_ablation.py`, run against two
real trained checkpoints of the same lego scene at very different splat
densities — `local_runs/lego_prepared/wide` (300k splats, 99,707 above
opacity 0.1) and `.../budget_500` (500 splats, 428 above opacity 0.1).
Each family's bandwidth is fit via `hyperparams.fit_kernel_param_pooled_nd`
against the *same* real local (position, color) windows per checkpoint
(a generalization of `splat_scene.fit_kernel_hyperparams`'s sigma-fitting
window sampling to an arbitrary `kernel_factory` — `fit_kernel_hyperparams`
itself is untouched). Window radius is chosen per checkpoint, not shared:
`wide` uses `fit_kernel_hyperparams`'s own default (0.08), but that radius
finds a median of ~1-2 real neighbors in `budget_500` (checked directly),
so `budget_500` uses 0.25, the smallest radius giving >=6 real neighbors
for >90% of candidate windows in that checkpoint.

Two metrics, computed via a new function generalizing
`LocalUncertaintyEngine.rendering_aware_variance` (which hard-requires an
RBF `pos_kernel`) to any stationary 1D kernel family, by re-deriving the
same Gaussian-convolution identity RBF's closed form uses via a cached 1D
`scipy.integrate.quad` convolution instead (see
`kernel_family_ablation.py`'s module docstring for the full derivation;
the module's existing `mode="numerical"` `nquad` fallback was tested and
confirmed too slow to use at real-checkpoint scale — a single 3D query
did not finish in 120s):

- **Sparsity correlation**: at 150 real splat positions per checkpoint,
  correlate posterior variance (excluding the query splat itself as a
  neighbor) against distance to its 8th-nearest real neighbor (larger =
  sparser). A working signal is a *positive* correlation (sparser -> more
  variance). Also reported: the same correlation against
  `variance / prior_variance` (the amplitude-normalized ratio
  `compute_uncertainty_maps` already uses for the same reason elsewhere in
  this project) — raw variance turned out to be confounded by each
  window's local mean opacity (checked directly on `wide`: knn-distance
  anti-correlates with local opacity at r=-0.49, and opacity correlates
  with raw variance at r=+0.49), so the ratio is reported as a second,
  deconfounded read on the same question.
- **Calibration**: 6 held-out `lego_prepared/eval` views rendered with the
  real trained checkpoint (`gsplat.rasterization`), depth-unprojected to
  real world-space points (up to 80 per view), paired with real per-pixel
  squared rendering error against ground truth. Reports a calibration
  correlation and a Gaussian-NLL-style score,
  `mean(0.5*(error^2/var + log(var)))` (lower is better).

**Results**:

| checkpoint | family | fitted param | sparsity r (raw var) | sparsity r (ratio) | calibration r | calibration rho | mean NLL |
|---|---|---:|---:|---:|---:|---:|---:|
| wide | rbf | sigma=0.0703 | -0.306 | -0.291 | +0.120 | +0.111 | 9.80 |
| wide | matern32 | rho=0.0186 | -0.441 | -0.019 | +0.062 | -0.125 | 1878.0 |
| wide | rational_quadratic | l=0.0067 | -0.458 | **+0.191** | +0.064 | -0.173 | 6062.0 |
| budget_500 | rbf | sigma=0.0988 | +0.766 | +0.811 | -0.055 | -0.205 | **0.080** |
| budget_500 | matern32 | rho=0.0406 | +0.783 | +0.916 | -0.112 | -0.264 | 87.8 |
| budget_500 | rational_quadratic | l=0.0238 | +0.783 | **+0.921** | -0.112 | -0.265 | 85.4 |

(n=150 for sparsity, n=480 for calibration, per row.)

**Honest reading — a genuine trade-off, no single winner**:

- **Sparsity, raw variance**: on `budget_500` (the sparse checkpoint),
  every family shows the expected strong positive correlation
  (r=+0.77 to +0.78) — the signal works as intended when neighbor counts
  are genuinely limited by real splat density. On `wide` (dense), *every*
  family's raw-variance correlation is *negative* — the opposite of the
  naive expectation. Root cause, confirmed directly: at `wide`'s density
  and this checkpoint's `window_radius=0.08`, 148/150 sampled windows hit
  the engine's `max_neighbors=60` cap regardless of true local density
  (raw neighbor counts there are in the hundreds to low thousands), so
  the raw variance is no longer really tracking splat count at all — it's
  dominated by each window's local mean opacity (a real, measured
  confound, not noise: see the ratio metric's derivation above).
- **Sparsity, amplitude-normalized ratio**: once that confound is
  divided out, `RationalQuadraticKernel` is the only family whose
  correlation sign is *correct* on `wide` (+0.191, vs. RBF's -0.291 and
  Matern's -0.019, both still wrong-signed), and it has the *strongest*
  correlation on `budget_500` too (+0.921 vs. RBF's +0.811 and Matern's
  +0.916). This is a real, reasonably clean win for the new kernel family
  on this specific property, in both density regimes.
- **Calibration**: RBF is dramatically better calibrated by the
  Gaussian-NLL score on *both* checkpoints — roughly 200x better than
  Matern/RationalQuadratic on `wide` (9.8 vs. 1878/6062) and roughly
  1000x better on `budget_500` (0.08 vs. 88/85). Cause: Matern/RQ's own
  marginal-likelihood fit lands on a much smaller bandwidth than RBF's
  (e.g. `budget_500`: l=0.024, rho=0.041 vs. sigma=0.099) — a tighter
  bandwidth that fits the local color-smoothness data well but produces
  much smaller (overconfident) posterior variance at real held-out query
  points, which is catastrophic under the `1/var` term in the NLL score
  whenever the actual held-out error is non-trivial. The calibration
  *correlation* itself is weak for every family on both checkpoints
  (|r| <= 0.21) — none of the three families should be read as "well
  calibrated" against real held-out error in the correlation sense; RBF
  is simply far less badly miscalibrated in the NLL sense.
- **Net honest takeaway**: no family dominates every metric/checkpoint.
  RBF should stay the default where a well-behaved, non-catastrophic
  variance estimate matters most (e.g. any NLL-style loss or
  proper-scoring use, like ROADMAP.md item 1's `nll_weight`).
  `RationalQuadraticKernel` is the more sensitive, better-behaved choice
  specifically for the sparsity/coverage-tracking use case this project's
  other kept results (the directional-coverage and floater figures) care
  about, once its raw variance is read as the amplitude-normalized ratio
  rather than a raw magnitude. Matern-3/2 did not win outright on any
  metric/checkpoint here.

**Concrete next untested step**: the `wide`-checkpoint sparsity confound
traces to `max_neighbors=60` saturating almost universally at that
density/radius combination — re-run the sparsity check with
`max_neighbors` scaled to (or uncapped relative to) each checkpoint's own
density, to check whether the raw-variance sparsity signal (not just the
amplitude-normalized ratio) can be recovered directly on a dense
checkpoint, and whether that changes which family wins.

Scripts/data: `gs_experiment/kernel_family_ablation.py`; checkpoints at
`gs_experiment/local_runs/lego_prepared/{wide,budget_500}` and held-out
views at `.../lego_prepared/eval` (all gitignored, pre-existing).

### 2b. Addendum: expanded to all 7 NeRF-Synthetic scenes

The lego-only setup above was generalized to run on all 7 scenes this
project's other kept results use (chair, drums, ficus, hotdog, lego, mic,
ship — `materials` excluded for the same documented reason
`scripts/render_scene_gallery.py` excludes it: even its best held-out view
stays visibly hazy under this project's training recipe, unrelated to
kernel choice). `kernel_family_ablation.py`'s `CHECKPOINTS`/`EVAL_DIR`
globals became per-scene dicts (`CHECKPOINTS[scene]`, `EVAL_DIRS[scene]`);
`main()` now loops over all 7 scenes and saves full results (all metrics,
not just the printed summary) to
`gs_experiment/results/kernel_family_ablation_results.json`, written
incrementally after each scene so a partial run isn't lost. Everything
else (window sampling, per-family bandwidth fitting, the generalized
rendering-aware variance, sparsity/calibration metric code) is reused
unchanged.

**Window radius**: before trusting lego's `wide`=0.08/`budget_500`=0.25
radii on the other 6 scenes, real neighbor counts were checked directly
per scene/checkpoint (same recipe as the original tuning: KD-tree query
around 150 sampled window centers). Every one of the 14 scene/checkpoint
combinations keeps >=6 real neighbors for at least 86% of sampled windows
(worst cases: `ship`/`budget_500` at 86.7%, `hotdog`/`wide` at 87.3%, both
close to lego's own `budget_500` baseline of 90.7%) — no scene needed a
different radius, consistent with NeRF-Synthetic scenes sharing roughly
the same normalized coordinate bounds.

**Results**: full per-scene tables are in `paper/main.tex`'s Appendix
("Alternative kernels" subsection, Tables II-VIII); raw numbers for every
scene/checkpoint/family are in `kernel_family_ablation_results.json`.

**Cross-scene reading — one part of the lego-only finding generalizes
cleanly, the other doesn't**:

- **Calibration (mean NLL)**: RBF is the best-calibrated family in *all*
  14 scene/checkpoint combinations — a fully universal result, not just a
  lego artifact. Typically 2+ orders of magnitude better than
  Matern/RationalQuadratic, up to ~4 orders of magnitude in the worst case
  (`mic`, where every family is in fact badly miscalibrated in absolute
  terms — real held-out error there is large relative to any family's
  fitted variance — but RBF is still ~1000x less catastrophic). Same
  mechanism as the lego-only finding: Matern/RQ's marginal-likelihood fits
  land on tighter bandwidths that explain local color smoothness well but
  produce overconfident (too-small) variance at real held-out points,
  punished heavily by the NLL's `1/var` term. Calibration *correlation*
  stays weak and sign-inconsistent across every family/scene, exactly as
  on lego alone — none of the three families should be read as tracking
  held-out error in the correlation sense.
- **Sparsity, amplitude-normalized ratio, sparse checkpoint**: broadly
  replicates lego. Raw variance has the expected positive sign for every
  family on all 7 scenes (21/21) on `budget_500`, and RationalQuadratic
  wins the ratio metric on 6 of 7 scenes (all but `ship`, where RBF is
  narrowly ahead: +0.927 vs. +0.887).
- **Sparsity, raw variance, dense (`wide`) checkpoint — does NOT
  generalize**: lego's "wrong sign for every family" result was itself
  scene-dependent, not universal. Raw variance keeps the *correct*
  (positive) sign on `chair` and `ship`, is split by family on `ficus`,
  and is wrong-signed for every family only on `drums`, `hotdog`, `lego`,
  and `mic`. Traced directly to real per-scene density: the median real
  neighbor count inside the same fixed-radius window at `wide` density
  ranges from ~21 (`ficus`) to ~9,000 (`mic`) across scenes that share an
  identical training recipe and window radius — it's specifically the
  high-neighbor-count scenes (well past the engine's `max_neighbors=60`
  cap) where local opacity confounds raw variance, not a universal
  property of "dense" checkpoints in general.
- **Sparsity ratio, dense checkpoint**: correspondingly more mixed than
  lego alone suggested — RationalQuadratic wins on 3/7 scenes (`drums`,
  `ficus`, `lego`), Matern on 2/7 (`chair`, `mic`), RBF on 2/7 (`hotdog`,
  `ship`).
- **Net honest takeaway, updated**: the calibration half of the original
  trade-off (RBF safest for NLL-style use) is now confirmed universal
  across all 7 scenes, stronger evidence than lego alone gave. The
  sparsity-tracking half (RationalQuadratic best for coverage-tracking)
  mostly holds, especially on genuinely sparse checkpoints, but is not a
  scene-independent law at high splat density — the underlying raw-
  variance confound this recommendation is designed to correct for is
  itself scene-dependent, and where that confound doesn't arise (`chair`,
  `ship`), RBF's own raw variance already does the sparsity-tracking job
  correctly, with no need for RQ's ratio-based fix.

Scripts/data: same `gs_experiment/kernel_family_ablation.py`, run with no
`--scenes` argument (defaults to all 7); full results in
`gs_experiment/results/kernel_family_ablation_results.json`; checkpoints
at `gs_experiment/local_runs/<scene>_prepared/{wide,budget_500,eval}` for
each of the 7 scenes (all gitignored, pre-existing).
