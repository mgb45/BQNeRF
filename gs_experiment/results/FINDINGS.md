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

**Update, re-run after the `SplatScene.colors` bug fix (section 4)**: this
experiment's own `LEGO_BQ_SIGMA` was itself one of the three stale
hardcoded bandwidth constants section 4 flagged but didn't re-run against
(0.0694 -> 0.13926, the same ~2.0x correction as every other lego-scale
sigma in this project) -- and the checkpoint colors it trained against were
also pre-fix. Both are now corrected and the full five-variant comparison
was re-run from scratch (new output dir
`gs_experiment/local_runs/likelihood_experiment_v2/`, same recipe/seed/
`n_iters=3000` as before, same
`gs_experiment/scripts/likelihood_training_experiment.py`, unmodified
except for its already-corrected `LEGO_BQ_SIGMA`). The old numbers below
are retained for comparison rather than deleted, per this project's own
norm of reporting real before/afters, not just final numbers.

**Results** (matched budget, `max_splats=8000`, `n_iters=3000`, 12 train /
15 held-out views) -- old (pre-fix) vs. new (post-fix):

| variant | n_splats (old) | n_splats (new) | train PSNR (old) | train PSNR (new) | held-out PSNR (old) | held-out PSNR (new) |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 3108 | 3132 | 26.65dB | 26.71dB | 17.48dB | 17.57dB |
| bq_densify | 3333 | 3330 | 27.48dB | 27.56dB | 17.78dB | 17.81dB |
| nll_loss | 3095 | 3106 | 26.51dB | 26.80dB | 17.27dB | 17.43dB |
| bq_densify+nll | 3339 | 3345 | 27.51dB | 27.38dB | 17.77dB | 17.72dB |
| bq_densify_floor | 1181 | 1179 | 24.94dB | 24.76dB | 17.64dB | 17.55dB |

Deltas vs. baseline -- old vs. new:

| variant | train PSNR (old) | train PSNR (new) | held-out PSNR (old) | held-out PSNR (new) | n_splats (old) | n_splats (new) |
|---|---:|---:|---:|---:|---:|---:|
| bq_densify | +0.83dB | +0.85dB | +0.29dB | +0.24dB | +225 | +198 |
| nll_loss | -0.14dB | +0.09dB | -0.21dB | -0.14dB | -13 | -26 |
| bq_densify+nll | +0.86dB | +0.67dB | +0.28dB | +0.15dB | +231 | +213 |
| bq_densify_floor | -1.70dB | -1.95dB | +0.16dB | -0.02dB | -1927 | -1953 |

Every absolute number moves a little (baseline itself shifts by
+0.06/+0.09dB train/held-out despite `densify_criterion="gradient"` not
directly depending on `bq_sigma` -- plausibly ordinary GPU
non-determinism/run-to-run noise at this splat count, not a systematic
effect of the fix), but the *pattern* the original conclusions rest on is
essentially unchanged for two of three variants and only meaningfully
softens for the third:

- **bq_densify holds almost exactly**: +0.83->+0.85dB train, +0.29->+0.24dB
  held-out -- the same small, real, positive win, well within the old
  run's own noise band.
- **nll_loss's held-out delta stays mildly negative** (-0.21->-0.14dB,
  same sign, similar magnitude); its *train* delta flips sign
  (-0.14->+0.09dB) but is small enough in both directions to read as noise
  around zero either way -- the held-out number (the one that actually
  matters for a generalization claim) is the more stable read and doesn't
  change conclusion.
- **bq_densify_floor's held-out delta crosses zero**: +0.16dB (a small
  measured *win*) in the old run vs. -0.02dB (essentially flat, a
  negligible loss) in the new one. This is the one place the fix
  meaningfully changes the story -- see the updated reading below.
- **bq_densify+nll softens slightly** (+0.86->+0.67dB train,
  +0.28->+0.15dB held-out) -- still a real positive delta, but now
  visibly a bit *below* `bq_densify` alone's own new delta
  (+0.85/+0.24dB) rather than statistically indistinguishable from it as
  in the old run. Consistent with `nll_loss`'s own mildly-negative
  held-out effect now showing through slightly when layered on top of
  BQ-densify, rather than vanishing entirely.

**Honest reading** (updated in place; bullets reordered to flag what
changed first):

- **BQ-variance densification is a real, if modest, win -- confirmed by
  the re-run, essentially unchanged.** +0.85dB train / +0.24dB held-out at
  a matched `max_splats` budget (was +0.83/+0.29), using ~6% more splats
  to get there. Still shows up whether the NLL term is on or off
  (bq_densify: +0.85/+0.24 vs. bq_densify+nll: +0.67/+0.15 -- see below for
  why these two are now a little less identical than before), so this
  remains the project's clearest example of the BQ posterior actively
  *changing* a training outcome rather than just describing a finished one.
- **The NLL auxiliary loss term is still a no-op-to-mild-negative on its
  own, though the picture is a bit noisier than first reported.**
  `nll_loss` (gradient densify + NLL on) is worse than baseline on
  held-out PSNR both before and after (-0.21dB old, -0.14dB new -- same
  sign, similar size); its *train* PSNR delta flips sign (-0.14dB old,
  +0.09dB new), small enough in both directions to read as noise rather
  than a real effect either way. Layering NLL on top of BQ-densify
  (`bq_densify+nll` vs. `bq_densify`) no longer looks like a total
  no-op the way it first did: the old run showed +0.86 vs. +0.83dB train,
  +0.28 vs. +0.29dB held-out (indistinguishable); the new run shows +0.67
  vs. +0.85dB train, +0.15 vs. +0.24dB held-out -- `bq_densify+nll` is now
  visibly a bit *below* `bq_densify` alone on both metrics, consistent
  with `nll_loss`'s own mild held-out drag actually showing through when
  combined, rather than being fully absorbed by the densification change
  as originally read. Given `train()`'s own docstring already notes the
  NLL term does not differentiate through the BQ posterior itself
  (gradients flow through `pred` only, not through `var`), this remains a
  plausible, honest limit of this first installment: it can reweight
  *which* pixels matter but not yet *why* a region is uncertain in a way
  that changes splat geometry beyond what densification already does --
  if anything the re-run makes this limitation slightly more visible, not
  less.
- **The opacity floor's "no quality tax" framing holds; its "marginally
  better than baseline" framing does not survive the re-run and should be
  dropped.** `bq_densify_floor` still ends with a large splat-count
  reduction relative to baseline (1179 vs. 3132, 62.4% fewer -- was 62.0%
  fewer, unchanged in substance) and training-view PSNR still drops
  notably (-1.95dB, was -1.70dB). But held-out PSNR, which the old run
  reported as a small *positive* delta (+0.16dB, described as "not worse
  -- it's marginally better than baseline"), is now -0.02dB: essentially
  exactly flat, not a measured win. The honest updated claim is narrower
  than before: excluding low-opacity, likely-empty-space splats from
  competing for densification budget buys a large splat-count reduction
  at *no measurable held-out cost* (still true, and still the
  practically useful part of this result), not at a measurable held-out
  *gain* (no longer supported -- that was likely just one run's noise
  landing on the positive side of essentially zero).
- **Net honest takeaway, re-affirmed with one claim narrowed**: at this
  scene scale and iteration budget, BQ variance is a better densification
  signal than the raw view-space gradient (small but real quality win, or
  a much bigger splat-budget win at no held-out cost if the opacity floor
  is used -- "no cost" rather than the previously reported "small gain"),
  but the auxiliary NLL loss term as currently implemented is not pulling
  its weight -- if anything the re-run's `bq_densify+nll` vs. `bq_densify`
  gap makes its mild drag slightly more visible than the original run
  did -- and needs either a finer/more frequent grid, a different
  weighting, or differentiating through the BQ posterior itself (as
  `train()`'s docstring already flags as the real next step) before it
  can be trusted to help rather than mildly hurt.

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

Scripts/data: `gs_experiment/scripts/likelihood_training_experiment.py`
(now with the corrected `LEGO_BQ_SIGMA=0.13926`); old (pre-fix) checkpoints
and per-variant eval scene_dirs under
`gs_experiment/local_runs/likelihood_experiment/`, new (post-fix) ones
under `gs_experiment/local_runs/likelihood_experiment_v2/` (both
gitignored, both left on disk for comparison).

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

## 3. Calibration-methodology correction: pairing u_BQ with the mean it was actually computed for

**Question**: every calibration number reported so far (section 2/2b
above, `paper/main.tex`'s Appendix Tables II-VIII) computes the BQ
posterior variance `u_BQ` around the BQ posterior mean `C_BQ`, then scores
it against the squared error of a *different* quantity: `C_alpha`, the
real gsplat alpha-compositing renderer's actual output. `u_BQ` was never
computed as a variance around `C_alpha` — pairing `N(C_alpha, u_BQ)` and
implicitly calling that "the BQ posterior" (what every existing table
does) is not a coherent probabilistic statement, regardless of how it
happens to score. This is a real, independently plausible explanation for
why calibration correlation has stayed weak/sign-inconsistent everywhere
in sections 2/2b despite `u_BQ` clearly tracking *something* real
(sparsity, directional coverage).

**Method**: `gs_experiment.quadrature.rendering_aware_alternative_weight_risk`
(the general RKHS worst-case-squared-error quadratic form `e(w)^2 = z0 -
2w@z + w@K@w` for *any* real weight vector, reducing exactly to the BQ
posterior variance at the BQ-optimal `w*=K^-1 z` and never smaller for any
other real weight vector — proven in
`tests/gs_experiment/test_render_weight.py`) and
`pixel_uncertainty.LocalUncertaintyEngine.rendering_aware_alpha_risk_along_ray`
(which applies it to the real `w_i=T_i*alpha_i` alpha-compositing
transmittance weights already computed by
`visibility_attribution.ray_transmittance_weights`) make five
mean/uncertainty pairings directly comparable at the same query points,
under the same kernel:

1. **existing post-hoc** (what sections 2/2b and Tables II-VIII actually
   compute, unlabeled as such): mean=`C_alpha` (real), var=`u_BQ`
2. **coherent BQ renderer**: mean=`C_BQ`, var=`u_BQ` (the pairing `u_BQ`
   was actually derived for)
3. **BQ risk of the alpha renderer**: mean=`C_alpha` (real),
   var=`R_alpha = u_BQ + (C_BQ - C_alpha)^2` (the posterior expected
   squared error of the real renderer's output, treating `C_BQ` as the
   model's best estimate and `u_BQ` as remaining uncertainty around it)
4. **constant baseline** (sanity-check null model): mean=`C_alpha` (real),
   var=one global constant per checkpoint (MLE = mean squared error over
   that checkpoint's own query points)
5. **alpha's own local quadrature risk**: mean=`C_alpha_local` (a
   real alpha-compositing estimate, but from this call's local candidate
   window only — not guaranteed to bit-match the full-scene renderer's
   `C_alpha`), var=`alpha_risk` (that same estimator's own RKHS risk,
   `e_alpha^2`) — "how good is the renderer's own quadrature rule, under
   this same kernel," no borrowed mean at all

Phase A (mandatory, all 7 scenes × 2 checkpoints — `chair`, `drums`,
`ficus`, `hotdog`, `lego`, `mic`, `ship`, each at `wide`/`budget_500`,
reusing `kernel_family_ablation.py`'s `SCENES`/`CHECKPOINTS`/`EVAL_DIRS`/
`fit_all_families` verbatim, RBF kernel only since the along-ray path
requires it): 480 real held-out query points per checkpoint (6 views ×
up to 80 points, same convention as `kernel_family_ablation.calibration_metrics`),
scoring Gaussian NLL, Pearson/Spearman correlation, empirical 1σ/2σ
coverage, sharpness (mean variance), and an AUSE-style risk-coverage area
for all 5 variants. Phase B (image-level, scoped down for tractability to
`lego` + `chair`/`hotdog`/`ficus`, 2 held-out views each, `wide`
checkpoint only): reconstructs a 112×42 grayscale image from `C_BQ` at
every valid pixel and reports PSNR/SSIM against a same-resolution real
alpha-compositing render, plus the fraction of raw (pre-clip) `C_BQ`
values outside `[0,1]` and the fraction of negative BQ posterior weights
(`w*=K^-1 z` entries).

**A real bug found and fixed while building Phase B**: `render_views`
defaults to a dark `(0.05,0.05,0.05)` background, while this dataset's
real ground truth has a white background (confirmed directly: GT corner
pixels are exactly `[1,1,1]`, the renderer's own corner pixels are
exactly `[0.05,0.05,0.05]`). This never touched Phase A or any earlier
section's numbers (all of them only ever sample points inside the
valid/foreground mask), but a naive whole-frame Phase B PSNR/SSIM is
almost entirely a measurement of that orthogonal background-color
default (confirmed: ~1-3dB PSNR for *both* variants before the fix, since
60-90% of a frame at this resolution is background) rather than of
reconstruction quality. Fixed by neutralizing background/invalid pixels
to GT's own value in both compared images before scoring, so neither
reconstruction is scored on the background at all.

**Phase A results — aggregated across all 14 scene/checkpoint combinations**:

| variant | NLL (median) | NLL (mean) | pearson (median) | spearman (median) | cov 1σ (mean) | cov 2σ (mean) | sharpness (median) | AUSE (mean) |
|---|---|---|---|---|---|---|---|---|
| 1. existing post-hoc | 4.498 | 20.885 | -0.017 | -0.171 | 0.778 | 0.861 | 0.836 | 0.210 |
| 2. coherent BQ renderer | 14.369 | 48.276 | +0.034 | -0.161 | 0.313 | 0.519 | 0.836 | 1.467 |
| 3. BQ risk for alpha renderer | 0.702 | 1.070 | -0.036 | -0.144 | 0.888 | 0.949 | 2.788 | 0.212 |
| 4. constant baseline | -0.392 | -0.337 | n/a (constant) | n/a (constant) | 0.729 | 0.928 | 0.168 | 0.179 |
| 5. alpha's own quadrature risk | 1.694 | 2.385 | -0.104 | -0.152 | 0.707 | 0.881 | 7.559 | 0.992 |

(n=480 query points per checkpoint, 14 checkpoints; lower is better for
NLL and AUSE; cov 1σ/2σ closer to the nominal 0.68/0.95 is better, not
simply "higher"; variant 4's correlation is undefined by construction —
its variance is a single constant per checkpoint, reported as n/a, not a
crash or a silently-recorded garbage value.)

**Win counts** (best variant per checkpoint, out of 14): NLL — variant 4
wins all **14/14**; every one of variants 1/2/3/5 is beaten by the
trivial constant-variance null model on every single checkpoint. AUSE —
variant 4 wins 9/14, variant 3 wins 3/14, variant 1 wins 2/14, variants 2
and 5 win 0/14. Head-to-head, variant 3 beats variant 1 on NLL in 9/14
checkpoints and on AUSE in 6/14. Correlation sign is inconsistent for
every non-null variant across checkpoints (variant 1: 7 positive/7
negative; variant 2: 8/6; variant 3: 5/9; variant 5: 2/12 — variant 5 is
mostly *negative*-signed).

**Per-checkpoint NLL, variants 1/2/3/4/5** (representative full spread,
not just an aggregate):

| scene | checkpoint | v1 (existing) | v2 (coherent BQ) | v3 (R_alpha) | v4 (constant) | v5 (alpha risk) |
|---|---|---|---|---|---|---|
| chair | wide | 29.953 | 66.591 | 1.079 | -0.469 | 2.458 |
| chair | budget_500 | 0.271 | 1.833 | 0.448 | -0.417 | 1.201 |
| drums | wide | 21.857 | 54.339 | 2.984 | -0.072 | 2.441 |
| drums | budget_500 | 1.414 | 6.951 | 0.446 | -0.253 | 1.295 |
| ficus | wide | 10.975 | 14.598 | 1.543 | -0.045 | 1.144 |
| ficus | budget_500 | 0.137 | 14.141 | 0.750 | -0.163 | 3.076 |
| hotdog | wide | 163.163 | 304.455 | 1.612 | -0.534 | 4.545 |
| hotdog | budget_500 | -0.089 | 2.244 | 0.197 | -0.459 | 0.876 |
| lego | wide | 38.632 | 128.426 | 2.415 | -0.368 | 5.633 |
| lego | budget_500 | 0.457 | 5.217 | 0.653 | -0.465 | 1.307 |
| mic | wide | 17.681 | 24.363 | 1.791 | 0.057 | 1.649 |
| mic | budget_500 | 0.597 | 6.150 | 0.560 | -0.028 | 1.739 |
| ship | wide | 7.582 | 43.134 | 0.259 | -0.584 | 5.124 |
| ship | budget_500 | -0.232 | 3.421 | 0.240 | -0.920 | 0.902 |

Variant 3's biggest wins over variant 1 are exactly where variant 1 is
most catastrophic — the dense `wide` checkpoints, where a small `u_BQ`
paired with a large real `C_alpha` error blows up the NLL's `1/var` term
(`hotdog`/`wide`: 163.2 → 1.6; `lego`/`wide`: 38.6 → 2.4). On the sparse
`budget_500` checkpoints, where variant 1 is already reasonably well-
behaved, variant 3 is sometimes worse (e.g. `ficus`/`budget_500`: 0.137 →
0.750) — the fix trades away some already-decent small-NLL cases for
robustness against the large ones.

**Phase B — image-level reconstruction quality and failure-mode diagnostics**:

| scene | view | n valid px | PSNR C_BQ | PSNR alpha | SSIM C_BQ | SSIM alpha | frac. C_BQ out-of-range | frac. negative BQ weights |
|---|---|---|---|---|---|---|---|---|
| chair | 0 | 1199 | 9.01 dB | 11.70 dB | 0.487 | 0.583 | 0.439 | 0.484 |
| chair | 5 | 1326 | 8.70 dB | 11.95 dB | 0.426 | 0.600 | 0.391 | 0.480 |
| ficus | 0 | 991 | 9.21 dB | 8.08 dB | 0.473 | 0.369 | 0.639 | 0.510 |
| ficus | 5 | 896 | 8.82 dB | 8.82 dB | 0.432 | 0.354 | 0.622 | 0.515 |
| hotdog | 0 | 1243 | 9.40 dB | 11.80 dB | 0.484 | 0.638 | 0.593 | 0.480 |
| hotdog | 5 | 1545 | 10.49 dB | 13.01 dB | 0.505 | 0.693 | 0.270 | 0.473 |
| lego | 0 | 1261 | 9.46 dB | 11.02 dB | 0.467 | 0.602 | 0.809 | 0.491 |
| lego | 5 | 1819 | 9.75 dB | 11.59 dB | 0.526 | 0.655 | 0.695 | 0.485 |
| **mean** | | | **9.36 dB** | **11.00 dB** | **0.475** | **0.562** | **0.557** | **0.490** |

`C_BQ` renders competitively but consistently a bit behind the real
renderer on 3 of 4 scenes (chair, hotdog, lego: ~1.5-2.7dB PSNR gap, ~0.1
SSIM gap) — not degenerate, but not an improvement either. On `ficus`
(thin/fine structure), `C_BQ` actually *ties or beats* the real renderer
(9.21 vs 8.08dB on view 0, exactly tied at 8.82dB on view 5) — the real
renderer itself does worse there at this resolution, and `C_BQ`'s local
GP smoothing doesn't lose any further ground on fine structure the way it
might have been expected to.

The two failure-mode diagnostics are the most decisive finding here, and
they are **not rare**: averaged over all 8 Phase-B views, 55.7% of raw
(pre-clip) `C_BQ` predictions fall outside `[0,1]`, and 49.0% of BQ-
optimal weight entries (`w*=K^-1 z`) are negative. These sit close to
half of all cases, at every scene checked (39-81% out-of-range, 47-52%
negative-weight) — not a tail-case rounding artifact, and not something
this experiment can responsibly report as "rare/negligible."

**Honest reading, structured around the decisive-experiment framing this
was set up to test**:

- Regardless of which way the numbers came out, it is worth restating
  plainly: pairing `N(C_alpha, u_BQ)` and calling it "the BQ posterior" —
  implicitly what Tables II-VIII currently do — is not a coherent
  probabilistic statement. `u_BQ` was derived as the variance around
  `C_BQ`, never around `C_alpha`.
- `C_BQ` does **not** render competitively enough, and has a substantial,
  non-negligible rate of both out-of-range predictions (~56%) and
  negative BQ weights (~49%), to be positioned as a full probabilistic
  renderer replacing alpha compositing. Variant 2 (the "coherent" `C_BQ`
  +`u_BQ` pairing) is also the worst-performing variant on every metric
  here — dramatically worse NLL than even variant 1, and badly
  overconfident coverage (31% of points fall within 1σ against a nominal
  68%) — a direct consequence of `C_BQ` itself being a poor predictor of
  real held-out color on `wide` checkpoints in particular, not something
  `u_BQ`'s magnitude was ever sized to cover.
- None of variants 1/2/3/5 beat a trivial single-constant null model on
  Gaussian NLL, on *any* of the 14 checkpoints. Read plainly, real
  per-point calibration signal in the strict proper-scoring-rule sense is
  weak-to-absent for all of them at this query-point granularity — this
  methodology fix does not, by itself, resolve the weak/sign-inconsistent
  correlation problem sections 2/2b already found; correlation stays weak
  and sign-inconsistent for every non-null variant here too.
- That said, variant 3 (`R_alpha`, paired with the real `C_alpha`) is
  clearly the most defensible of the three "real" variants: it beats
  variant 1 on NLL in a majority of checkpoints (9/14) and on AUSE in a
  plurality (6/14 vs. variant 1's 2/14), achieves the best empirical
  coverage of any non-null variant (cov 1σ=0.888, cov 2σ=0.949 — errs
  conservative/over-covering, the safer failure direction for a risk
  bound), and specifically fixes variant 1's worst catastrophic-NLL cases
  on dense checkpoints without requiring `C_BQ` to be a good renderer in
  its own right.
- Variant 5 (alpha's own local quadrature risk, no borrowed `C_BQ` mean
  at all) does not improve on variant 3 — worse median NLL (1.694 vs.
  0.702), worse mean AUSE (0.992 vs. 0.212), and a mostly negative
  correlation sign (12/14 checkpoints) — so "the renderer's own
  quadrature risk, standalone" is not a better answer than `R_alpha` here.
- **Net conclusion, following the user's own framing**: this comes out on
  the "keep alpha compositing as the renderer" side. `C_BQ` should not
  replace alpha compositing as the deployed mean (Phase B's quality gap
  plus the ~50% out-of-range/negative-weight rates are concrete, not
  cosmetic, reasons). The right fix for Tables II-VIII is not to switch
  to variant 2, but to replace `u_BQ` alone with `R_alpha` (variant 3) as
  the principled post-hoc reliability measure paired with the real,
  deployed `C_alpha` — coherent by construction (it is the literal RKHS
  risk of the real alpha-compositing estimator under this project's own
  posterior), measurably more robust against catastrophic NLL blowups,
  and better-calibrated in the coverage sense than the current practice,
  even though — like every variant tested — it still does not beat a
  trivial constant-variance baseline outright.

**Concrete next untested step**: variant 3's remaining NLL losses on
`budget_500` checkpoints (where variant 1 was already reasonably behaved)
suggest `R_alpha`'s `(C_BQ-C_alpha)^2` bias term may be adding more
inflation than needed when `C_alpha` is already well-resolved (sparse,
well-supported regions) — worth checking whether a version that only
adds this term when `C_BQ` and `C_alpha` disagree beyond some
uncertainty-aware threshold (rather than unconditionally) recovers more
of variant 1's sparse-checkpoint behavior while keeping variant 3's
dense-checkpoint robustness.

Scripts/data: `gs_experiment/rendering_aware_calibration_experiment.py`
(Phase A/B driver, reuses `kernel_family_ablation.py`'s
`SCENES`/`CHECKPOINTS`/`EVAL_DIRS`/`fit_all_families`/`_render_and_unproject`
verbatim); full per-scene/per-checkpoint/per-variant numeric results
(including every raw per-point record, not just the aggregated metrics
above) in `gs_experiment/results/rendering_aware_calibration_results.json`;
checkpoints/eval views at
`gs_experiment/local_runs/<scene>_prepared/{wide,budget_500,eval}` for
all 7 scenes (all gitignored, pre-existing). Core math (unmodified,
already unit-tested this session): `gs_experiment.quadrature.
rendering_aware_alternative_weight_risk`,
`gs_experiment.pixel_uncertainty.LocalUncertaintyEngine.
rendering_aware_alpha_risk_along_ray`.

## 4. Bug fix and re-validation: `SplatScene.colors` was raw SH coefficients, not real color (affects sections 2/2b/3's fitted bandwidths)

**The bug**: `SplatScene.colors` (`gs_experiment/splat_scene.py`) — the flat,
position-only fallback color used whenever a splat's color is needed
without a specific viewing direction — was set directly to the raw
spherical-harmonics degree-0 coefficient (`sh_coeffs[:,:,0].mean(axis=1)`).
3DGS stores SH coefficients as offsets from mid-gray:
`real_color = SH_C0 * raw + 0.5` (`gs_experiment/spherical_harmonics.py`'s
own `eval_sh` convention). On a real checkpoint (lego/`wide`), the raw
`colors` field spanned `[-2.39, 2.37]` — not a valid color by any
convention — while the corrected field spans `[-0.17, 1.17]`, matching
`eval_sh(sh_coeffs, ..., degree=0)` exactly (now covered by a regression
test, `test_load_from_gsplat_checkpoint_colors_matches_eval_sh_degree_zero`
in `tests/gs_experiment/test_gs_splat_scene.py`). Fixed in both
`load_from_gsplat_checkpoint` and the mock-scene generator.

**Scope**: `colors` feeds `values=` into every `LocalUncertaintyEngine`
construction project-wide, and into
`splat_scene.fit_kernel_hyperparams`'s **sigma** (RBF position-kernel
bandwidth) marginal-likelihood fit, which samples real local
(position, color) windows via `scene.colors`. Posterior *variance* does
not depend on `values`' absolute scale directly, but the *fitted
bandwidth* does. **Kappa (the directional-kernel concentration) is
unaffected** — its fitting path always calls `eval_sh` directly whenever
real SH coefficients are present (true for every real checkpoint), never
the raw `colors` fallback. Confirmed empirically below (kappa ratio
exactly `1.000` on every checkpoint spot-checked) rather than assumed.

Everything below was re-run *after* the fix landed (see git working tree);
old numbers were read and retained before either results JSON was
overwritten, so every delta below is a real before/after, not a guess.

### 4a. `kernel_family_ablation.py` re-run (Tier 1) — section 2/2b numbers

Full re-run, all 7 scenes x 2 checkpoints x 3 families (42 fits + eval
sweeps); `gs_experiment/results/kernel_family_ablation_results.json`
overwritten in place with the corrected numbers.

- **Fitted bandwidths shift up substantially, as expected**: RBF `sigma`
  grows by 1.45x-2.00x across all 14 scene/checkpoint combinations (mean
  1.72x) — e.g. lego/`wide`: 0.0703 -> 0.1359; ship/`budget_500`: 0.1346 ->
  0.1964. Matern `rho` and RationalQuadratic `l` shift by more, up to
  **18.2x** in one case (ficus/`budget_500` matern32: rho 0.0071 -> 0.1299)
  — these two families' marginal-likelihood fits were evidently more
  sensitive to the color-scale distortion than RBF's.
- **Calibration NLL winner (the metric section 2b's headline claim rests
  on)**: RBF was lowest-NLL (best-calibrated) in **14/14** checkpoints
  before the fix — described as "a fully universal result." After the
  fix it is lowest-NLL in **12/14** — it now loses on hotdog/`wide` (RQ's
  10223 beats RBF's 107213 — RBF got an order of magnitude *worse* there)
  and lego/`wide` (Matern32's 476.6 narrowly beats RBF's 504.9). **The
  "fully universal" claim is no longer strictly true** — it's now 12/14
  (86%), with 2 concrete named exceptions — but RBF remains dramatically
  better-calibrated in the large majority of cases, often by 2+ orders of
  magnitude, exactly as before.
- **Calibration correlation winner** (already weak, |r|<=0.22, in both old
  and new numbers — never claimed as a strong signal): changed in 5/14
  checkpoints (chair/`wide`, drums/`budget_500`, ficus/`wide`,
  hotdog/`wide` rbf->matern32/matern32/RQ/matern32; ship/`budget_500`
  rbf->matern32) — this flips among near-noise values, not a reversal of
  a real signal.
- **Sparsity-ratio winner on the sparse (`budget_500`) checkpoint** — the
  metric section 2b credits RationalQuadratic with winning "6 of 7
  scenes": after the fix it wins only **4/7** (ficus, hotdog, lego, ship —
  note `ship` *flipped to RQ* from RBF), RBF now wins **3/7** (chair,
  drums, mic). This meaningfully weakens the "RQ best for
  sparsity-tracking" recommendation from a strong majority to a plurality.
  **Notable**: mic/`budget_500`'s sparsity-ratio correlation flips sign
  for *all three* families (was +0.03 to +0.17 for every family before,
  now -0.01 to -0.10 for every family) — the sparsity-tracking signal on
  that specific checkpoint may no longer be real/positive at all; it
  should not be cited as a sparsity-tracking success case without a fresh
  look.
- **Sparsity-ratio winner on the dense (`wide`) checkpoint**: also
  reshuffled — changed in 4/7 scenes (chair, ficus, hotdog, mic); new
  counts RQ 4/7, RBF 3/7 (previously RQ 3/7, Matern 2/7, RBF 2/7 — Matern
  no longer wins any scene on this metric).
  - **What did *not* change**: the raw-variance sparsity-sign pattern on
    the dense checkpoint (which scenes show the "wrong-signed" raw-variance
    confound that originally motivated the ratio metric) is unchanged —
    same scenes positive (chair, ship) vs. negative (drums, ficus, hotdog,
    lego, mic) before and after, magnitudes shifting only a few
    hundredths. This specific finding is robust to the bug.

**Updated reading**: RBF-safest-for-NLL and RQ-best-for-sparsity-tracking
both still hold as the *general* pattern, but neither is as clean as
previously reported — RBF's calibration dominance is "12/14, usually by a
wide margin" rather than "14/14, universal," and RQ's sparsity-ratio
advantage on sparse checkpoints is "a 4/7 plurality" rather than "a 6/7
majority." Full corrected numbers for every scene/checkpoint/family are in
the regenerated `kernel_family_ablation_results.json`; old numbers are
preserved in git history (this file's prior committed version) for anyone
wanting to diff further.

### 4b. `rendering_aware_calibration_experiment.py` re-run (Tier 1) — section 3 numbers

Full re-run, all 7 scenes Phase A + 4 scenes Phase B;
`gs_experiment/results/rendering_aware_calibration_results.json`
overwritten in place.

**Phase A — the least clean part, and it needs the mechanism explained,
not just the numbers**: `sigma_rbf` shifts by the same 1.45x-2.00x factor
found in 4a (same fitting code/data — e.g. chair/`wide`: 0.0608 -> 0.1181,
identical to 4a's value). But the mean posterior variance actually realized
at real held-out query points ("sharpness", variant 1's `sharpness_mean_var`)
**dropped** by 5x-16x on every `wide` checkpoint despite sigma growing —
e.g. lego/`wide`: 0.320 -> 0.0194 (16.5x smaller); hotdog/`wide`:
0.0296 -> 0.00052 (57x smaller). Meanwhile `C_alpha` (the real
alpha-compositing renderer's output — never touched by this bug) and its
real squared error against ground truth are exactly unchanged. A Gaussian
NLL's `1/var` term is highly sensitive to the denominator shrinking this
much against an unchanged numerator, so **variant 1 (existing post-hoc)'s
NLL got dramatically worse on every `wide` checkpoint** — roughly 10-20x,
not an improvement: chair 30.0->544.8, drums 21.9->403.2, ficus 11.0->298.1,
hotdog 163.2->3332.7, lego 38.6->635.6, mic 17.7->324.7, ship 7.6->165.7.
**This is not "the fix made things worse"** — `C_alpha`'s real error was
always this large; the fix just stopped a too-broad `u_BQ` from masking
it, making the pre-existing miscalibration variant 1 was already flagged
for *more visible*, not less real.

Variant 3 (`R_alpha = u_BQ + (C_BQ-C_alpha)^2`) held its ground or
*improved* on 6 of 7 `wide` checkpoints despite the same sharpness
collapse — chair 1.08->-0.72, drums 2.98->2.72, ficus 1.54->1.40, hotdog
1.61->-0.88, mic 1.79->1.68, ship 0.26->-0.44 (all flat-to-better; lower
is better) — with only lego/`wide` getting modestly worse (2.42->3.13).
Mechanism: `R_alpha`'s bias term `(C_BQ-C_alpha)^2` dominates its total
variance and itself grew as `C_BQ` (now a much better real-color estimate,
see Phase B below) tracks `C_alpha` more informatively, keeping `R_alpha`
well-scaled even as the pure-`u_BQ` term collapsed. **Net: the core
section-3 finding — "R_alpha fixes variant 1's worst dense-checkpoint NLL
blowups" — holds, and is demonstrated even more starkly than before**: the
gap between how bad variant 1 gets and how well variant 3 holds up is now
larger, not smaller. If anything, the bug fix strengthens the case for
switching Tables II-VIII from `u_BQ` to `R_alpha`, it does not weaken it.

Two specific absolute claims from section 3 change:

- "None of variants 1/2/3/5 beat a trivial constant-variance null model
  (variant 4) on Gaussian NLL, on any of the 14 checkpoints" is now
  **false in 2/14 cases**: variant 3 beats variant 4 at chair/`wide`
  (-0.716 vs -0.469) and hotdog/`wide` (-0.880 vs -0.534). Still true in
  the other 12/14.
- AUSE win counts move substantially in variant 3's favor: **3/14 before
  -> 8/14 after** (a majority), while the constant baseline's AUSE win
  count collapses from 9/14 to 1/14. Variant 3 is now the single most
  common AUSE winner outright, not merely "competitive."
- Variant 3's empirical coverage shifts from clearly over-conservative
  (cov 1sigma=0.888, cov 2sigma=0.949 — "errs conservative, the safer
  failure direction") to closer to nominal (cov 1sigma=0.664, cov
  2sigma=0.814) — no longer clearly over-covering; this specific framing
  in the original section-3 text should be softened, though the coverage
  is still reasonable.
- Correlation stays weak and sign-inconsistent for every variant, old and
  new alike (median pearson roughly -0.03 to +0.11 both before and after)
  — this part of the finding is unchanged.

Aggregate NLL (median across all 14 checkpoints), before -> after:
variant 1: 4.498 -> 94.010; variant 2: 14.369 -> 52.126; variant 3:
0.702 -> 0.728 (essentially flat); variant 4 (constant, fit directly to
the real squared error, never touches `colors`): -0.392 -> -0.392
(unchanged, as expected); variant 5: 1.694 -> 1.870.

**Phase B — the out-of-range collapse hypothesis is confirmed, and PSNR
reverses outright**:

- **Out-of-range fraction**: dropped from a mean of **55.7% to 5.3%**
  across the 8 checkpoint/view combinations checked (chair view0:
  43.9%->2.4%; ficus view0: 63.9%->14.7%; hotdog view0: 59.3%->0.4%;
  lego view0: 80.9%->1.7%; full per-view numbers in the regenerated JSON).
  This confirms the hypothesis exactly: a GP-predicted value near 0 in
  raw-SH space maps to real color ~0.5 (mid-gray), not black — the
  original "collapses to near-black, ~56% out-of-range" reading was
  substantially a units-bug artifact, not a real property of the method.
- **Negative-BQ-weight fraction is essentially unchanged**: 49.0% mean
  before -> 46.6% mean after (per-view range 47.3%-51.5% before vs.
  42.9%-49.8% after) — confirms this is a real, `values`-scale-independent
  property of the RKHS optimal-weight geometry (`K`, `z`), exactly as
  predicted, not an artifact of the units bug.
- **PSNR reverses outright**: `C_BQ` beat real alpha compositing in only
  1 of 8 views before (ficus/view5, an essential tie, 8.82 vs 8.82dB); it
  now beats alpha compositing in **8 of 8 views**, by 2.3-6.3dB (e.g.
  lego/view0: 9.46dB->13.40dB vs. alpha's unchanged 11.02dB; hotdog/view0:
  9.40dB->14.39dB vs. alpha's 11.80dB; chair/view0: 9.01dB->15.04dB vs.
  alpha's 11.70dB). Mean PSNR gap (alpha - BQ) flips from **+1.64dB**
  (alpha ahead) to **-3.10dB** (BQ ahead).
- **SSIM is more nuanced — do not oversimplify to "BQ wins now"**:
  `C_BQ`'s own SSIM improved substantially (mean 0.475->0.560), but real
  alpha-compositing SSIM is unchanged (it never depends on this bug) and
  still wins the per-view comparison in **5 of 8 views** (only ficus/view0,
  ficus/view5, and chair/view0 now favor `C_BQ`), even though the *mean*
  gap crosses to roughly zero (+0.087 -> -0.010) purely because those few
  BQ-favoring views (especially ficus) have large gaps. Read this as
  "roughly on par, no longer clearly behind" — not as a clean BQ win.

**Updated Phase B reading**: `C_BQ` is now a genuinely competitive-to-better
renderer on PSNR at these resolutions — the original "consistently a bit
behind, ~1.5-2.7dB gap" description is now backwards; `C_BQ` leads by a
similar-or-larger margin on every view checked — and roughly on par on
SSIM (not clearly ahead, not clearly behind, unlike the original "0.1 SSIM
gap" reading). The out-of-range collapse that anchored the original "does
not render competitively enough" verdict is substantially resolved. The
one Phase-B finding that stands completely unchanged is the negative-BQ-weight
rate (~47-50%), which remains a real, non-negligible structural property of
the method regardless of this bug — variant 2 (the "coherent" `C_BQ`+`u_BQ`
pairing) is still the worst NLL variant of the five (its own sharpness
collapse behaves the same way as variant 1's), so the section-3 recommendation
to pair `C_alpha` with `R_alpha` rather than switching to a `C_BQ`-mean
pairing still stands — but the case *against* `C_BQ` specifically on
rendering quality grounds is now considerably weaker than section 3
originally reported, and should be revisited before repeating the old
"`C_BQ` does not render competitively" framing anywhere else (paper
included, pending the user's own review).

### 4c. Headline-figure risk (Tier 2) — `render_scene_gallery.py`, `render_splat_sweep_gallery.py`, `render_coverage_uncertainty_sweep.py`

**Superseded — see section 6.** This subsection was originally written as
a risk assessment only (figures not regenerated, numbers estimated from an
indirect scaling argument). All three figures have since actually been
regenerated and the real before/after numbers measured directly — see
section 6 below. Left in place, unedited except for this note, as the
honest record of what was estimated beforehand versus what was actually
found: **the estimate's magnitude was in the right ballpark but its
direction was wrong** — it predicted the coverage-sweep's absolute raw
variance would grow ~6-8x; section 6 measures it directly and finds it
instead *shrinks* by a similar ~6.6-9.8x factor. Do not treat anything
below this note as current without cross-checking section 6.

Not re-rendered in this pass (real image generation across many
scenes/budgets/gap-conditions is expensive and out of scope here) — this
is a quantified risk assessment only, so a decision about regenerating the
three headline PNGs can be made separately.

Spot-check: `fit_kernel_hyperparams` called with old-vs-new colors on
lego/`wide` (used by `scene_gallery.png`, `lego_splat_sweep.png`) and
lego/`gap_0`, `gap_2`, `gap_4` (used by `coverage_uncertainty_sweep.png`,
via `real_directional_coverage_experiment.py`):

| checkpoint | sigma (old) | sigma (new) | ratio | kappa (old) | kappa (new) | ratio |
|---|---:|---:|---:|---:|---:|---:|
| lego/wide | 0.07033 | 0.13587 | 1.932 | 0.66110 | 0.66110 | 1.000 |
| lego/gap_0 | 0.07148 | 0.13270 | 1.856 | 0.59333 | 0.59333 | 1.000 |
| lego/gap_2 | 0.07431 | 0.14013 | 1.886 | 0.63531 | 0.63531 | 1.000 |
| lego/gap_4 | 0.08003 | 0.15331 | 1.916 | 0.73662 | 0.73662 | 1.000 |

Kappa is exactly unchanged (ratio 1.000) in all 4 cases, as expected.
Sigma shifts by 1.86x-1.93x, consistent with Tier 1's 1.72x mean.

- `render_scene_gallery.py` and `render_splat_sweep_gallery.py` both call
  `compute_uncertainty_maps` with `sigma=None`/`kappa=None` (fit per
  checkpoint at runtime) — simply re-running either script picks up the
  corrected, ~1.9x larger sigma automatically, no code change needed. But
  the currently-committed `scene_gallery.png`/`scene_gallery_500.png`/
  `lego_splat_sweep.png` were generated with the old, ~1.9x-too-small
  sigma and have **not** been regenerated in this pass.
- `render_coverage_uncertainty_sweep.py` is different: it does **not**
  fit sigma per checkpoint at runtime. It uses a hardcoded constant,
  `LEGO_GAP_SIGMA = 0.0694` (`real_directional_coverage_experiment.py`),
  itself derived from a one-time pooled marginal-likelihood fit across 9
  real checkpoints using the same buggy `scene.colors`-based mechanism —
  so this constant is **also** stale, and re-running the coverage-sweep
  script as-is (without refitting `LEGO_GAP_SIGMA`) would silently keep
  using the old, too-small bandwidth. Based on the per-checkpoint
  spot-check above (1.86x-1.93x), a corrected pooled refit would likely
  land `LEGO_GAP_SIGMA` around **~0.13**, matching the per-checkpoint
  range (0.133-0.153) found directly on the gap checkpoints.
- **Risk to the absolute raw-variance numbers quoted in `paper/main.tex`'s
  Experiments section** ("mean variance $0.82 \to 8.54 \to 22.12 \to 59.45
  \to 78.85$"): confirmed directly that this quoted figure is genuinely
  **raw** variance (`render_coverage_uncertainty_sweep.py` calls
  `compute_uncertainty_maps(..., return_raw_variance=True)` and plots
  `raw_map`'s mean) — exactly the metric `render_reconstruction.py`'s own
  docstring already flags as dominated by `sigma`, citing a previously
  observed 18x sigma difference producing a ~12,000x raw-variance
  difference on the same checkpoint (an empirical ~sigma^3.25 scaling in
  that one documented instance). Applying that same scaling relationship
  to our ~1.9x sigma shift predicts roughly a **6-8x** increase in the
  absolute raw-variance magnitudes if the sweep were regenerated with a
  corrected sigma. This is an order-of-magnitude estimate via that one
  documented precedent, not a re-derivation from first principles for this
  exact geometry — flagged as an estimate, not a proof. **The absolute
  numbers "0.82" through "78.85" as currently quoted in the paper are
  very likely stale by something on the order of one magnitude and should
  not be trusted as-is.**
- The **relative/qualitative** pattern (variance growing monotonically,
  by roughly two orders of magnitude, as the coverage gap widens) is much
  more likely to survive a re-fit: the same corrected sigma would apply
  uniformly across all 5 gap conditions in this fixed-sigma-across-conditions
  design (that is the whole point of using one pooled sigma rather than
  refitting per-condition), so the relative growth factors between
  conditions are governed mostly by how real local color/opacity data
  differs between conditions at a fixed bandwidth, not by the bandwidth's
  absolute value. This is inference from the design, not something
  directly re-measured here — regenerating the actual figure was
  explicitly out of scope for this pass.
- **Recommendation**: treat all three headline PNGs
  (`scene_gallery.png`/`scene_gallery_500.png`, `lego_splat_sweep.png`,
  `coverage_uncertainty_sweep.png`) and the paper's quoted absolute
  variance numbers as stale pending a decision on whether/when to
  regenerate them (flagged as an open `ROADMAP.md` item, not resolved in
  this pass — regenerating them was explicitly out of scope here).

Scripts/data: bug fix in `gs_experiment/splat_scene.py`; regression test
in `tests/gs_experiment/test_gs_splat_scene.py`; re-run results in
`gs_experiment/results/kernel_family_ablation_results.json` and
`gs_experiment/results/rendering_aware_calibration_results.json` (both
overwritten in place — old numbers preserved in git history / this
section's numbers above); Tier 2 spot-check was a throwaway script, not
committed to the repo.

## 5. Further code review after the colors bug: looking for the same class of issue elsewhere

Section 4's bug (a value in one convention — raw SH coefficient — consumed
elsewhere assuming a different convention — real color — with no
transform) passed 159 tests because nothing checked `colors`' *semantic
validity*, only that the pipeline ran. This section is a systematic
follow-up sweep for the same class of issue across `gs_experiment/*.py`
and `gs_experiment/scripts/*.py`: unit/convention mismatches (opacity
logit vs. probability, raw vs. `exp`'d scale, non-unit vs. normalized
quaternion, a second raw-SH color mistake, camera-convention mixing),
stale hardcoded constants, silent fallback defaults, sign/off-by-one
errors in the dense numerical modules, and test-coverage gaps for exactly
this failure mode. Investigation and safe static fixes only, per this
task's scope — no GPU experiment sweep or figure regeneration was run;
`pytest tests/` (CPU-only) and reading already-existing source were the
only checks performed.

### 5a. Fixed: stale figure in `train_minimal_gsplat.py`'s NLL-experiment docstring

The `train()` docstring's `bq_sigma`/`bq_window_radius` paragraph said the
thin-rod/cylinder scene family's defaults (`bq_sigma=0.9`,
`bq_window_radius=1.6`) were deliberately different from "the lego-scale
`sigma=0.05`/`window_radius=0.08` used elsewhere in `gs_experiment/`." No
`sigma=0.05` value exists anywhere in this repo, past or present — it
matches neither the pre-fix lego value (`0.0694`) nor the post-fix one
(`0.13926`, section 4 above). This looks like a simple transcription slip
(plausibly a stale memory of an early, even-rougher estimate), not a
second functional bug — nothing reads this docstring number
programmatically, it only appears as prose. Fixed to name the actual
current constants (`LEGO_GAP_SIGMA = 0.13926`,
`LEGO_GAP_WINDOW_RADIUS = 0.08` in `real_directional_coverage_experiment.py`)
directly, with a note explaining the correction, so it can't drift stale
silently again. No re-run needed to confirm impact — this was prose only.

### 5b. Investigated, confirmed still unreachable: `NLL_EXPERIMENT_COMMON_KWARGS`/`run_nll_experiment`

`run_nll_experiment` (`train_minimal_gsplat.py`, `--nll-experiment` CLI
mode) trains from `<nbv_dir>/baseline` and evaluates against
`<nbv_dir>/baseline_eval`. Checked directly: no `baseline`/`baseline_eval`
directory pair exists anywhere under `gs_experiment/local_runs/` in this
repo (`local_runs/nbv_out/` exists but contains only `nll_experiment/` and
`reference_strategy/`; the only other `baseline` directory in the repo,
`local_runs/likelihood_experiment/baseline/`, belongs to
`likelihood_training_experiment.py`'s separate, already-current experiment,
not this one). This confirms the state an earlier session in this project
already found: this code path is currently unreachable with any data this
repo has, i.e. dead code in practice, not actively wrong but not
worth chasing further per this task's own time-boxing.

Separately, and independently of reachability: `NLL_EXPERIMENT_COMMON_KWARGS`'s
`bq_sigma=0.9`/`bq_window_radius=1.6` is a hand-picked value matched to
this function's own `bounds=((-2.5,2.5)^3)` mock-scene scale (no
surrounding comment claims a marginal-likelihood fit, unlike
`LEGO_GAP_SIGMA`/`LEGO_BQ_SIGMA`'s explicit "marginal-likelihood-fitted"
provenance) — training in this path starts from `init_splats`, not a
`load_from_gsplat_checkpoint`-loaded checkpoint, so even if the data
layout existed, this particular constant was never downstream of the
colors bug in the first place. If a future re-run ever resurrects this
path (i.e. a `<nbv_dir>/baseline` layout gets regenerated), the thing to
check is not "was this stale from the colors bug" (it wasn't) but whether
`0.9`/`1.6` are still a reasonable match for whatever scene scale that
regenerated data uses.

### 5c. Ruled out after investigation (false leads, each checked directly, not assumed)

- **Quaternion normalization applied inconsistently**: `gsplat.rasterization`/
  `gsplat.quat_scale_to_covar_preci` both explicitly document "quats: (No
  need to be normalized)" (confirmed by reading the installed gsplat
  source directly) — every GPU call site in `gsplat_rendering_weights.py`
  and `train_minimal_gsplat.py` passes raw `params["quats"]`/`rotations`
  straight through, correctly. The plain-numpy CPU path
  (`pixel_uncertainty.quat_scale_to_covariance`) and the checkpoint reader
  (`ply_io.read_3dgs_ply`) both normalize explicitly before use. No site
  found that assumes unit-norm without normalizing or without gsplat doing
  it internally.
- **Opacity logit vs. probability**: every consumer of
  `params["opacity_logits"]`/`params["opacities"]` in `train_minimal_gsplat.py`
  (loss, pruning threshold, densification gate, the two training loops'
  differently-named param dicts) applies `torch.sigmoid(...)` before using
  the value as a real probability; `ply_io.py` applies the matching
  `inverse_sigmoid`/`sigmoid` pair symmetrically on write/read. No site
  found consuming a raw logit as if it were already in `(0, 1)`.
- **Raw scale vs. `exp(scale)`**: every read of `params["log_scales"]`/
  `params["scales"]` (both training loops, checkpoint writers) applies
  `torch.exp(...)` before use; `ply_io.py` mirrors this with matching
  `log`/`exp` on write/read. No site found using a raw log-scale as a real
  scale.
- **A second raw-SH-coefficient color mistake**: grepped every
  `sh_coeffs[...]` and `f_dc`/`f_rest` access project-wide
  (`splat_scene.py`, `ply_io.py`) — the only two `colors = ... + 0.5` sites
  are the two section-4 fixed in `splat_scene.py`; every script
  (`render_reconstruction.py` and others) that touches `sh_coeffs` passes
  it straight to `eval_sh`/`gsplat.rasterization`, never computing its own
  independent "color" from the raw DC term.
- **Camera-convention mixing (OpenCV vs. OpenGL, w2c vs. c2w)**:
  `nerf_transforms.opencv_viewmat_from_c2w` and
  `camera.viewmat_from_camera_pose` are already cross-validated against
  each other for the same pose in
  `tests/gs_experiment/test_gsplat_rendering_weights.py` (and elsewhere) —
  read both implementations directly; the OpenGL->OpenCV y/z-flip is
  applied exactly once, consistently, not mixed with an already-flipped
  input anywhere.
- **Sign/off-by-one errors in the dense numerical modules**
  (`quadrature.py`, `render_weight.py`, `visibility_attribution.py`,
  `gsplat_rendering_weights.py`, `hyperparams.py`, `kernels.py`): read all
  six in full. Found no hedging language (`grep`'d for
  TODO/FIXME/"should be"/"probably"/"I think"/"not sure"/"hack"/XXX across
  every source file in scope: zero matches) and every closed-form
  derivation in `quadrature.py`/`render_weight.py` already documents and
  is cross-checked in its matching test file against either a numerical
  (`nquad`/Monte Carlo) fallback or an independent GPU implementation
  (`gsplat_covariances` vs. `quat_scale_to_covariance`). The
  `occlusion_mask` grid-search z-buffer in `visibility_attribution.py` is
  the densest piece of index arithmetic in the codebase and is unusually
  thoroughly self-documented (including a stated, bounded false-positive
  rate and a reference-implementation cross-check in the test suite) —
  read it in full, found nothing suspicious.
- **Silent fallback defaults masking a real failure**: the
  `sigma if sigma is not None else fallback`-shaped pattern in
  `render_reconstruction.py`'s `compute_uncertainty_maps` prints which
  case fired (`" (fitted)"` vs. `" (fallback)"`) every time it runs, so a
  silent-fallback failure mode would still be visible in the script's own
  stdout, not swallowed. No other fallback-default pattern of this shape
  was found in `gs_experiment/*.py`/`gs_experiment/scripts/*.py`.

### 5d. Test coverage gaps closed (the "used but never range-checked" gap the colors bug exploited)

Two new tests, chosen as the highest-value gaps of this specific shape
(a value crossing a documented convention boundary — logit/probability,
log/real, non-unit/unit-quaternion — with every *existing* round-trip test
happening to use values, like identity quaternions and mid-range
opacities/scales, that would pass even if the conversion were silently
skipped):

- `tests/gs_experiment/test_gs_ply_io.py` (new file — `ply_io.py` had no
  dedicated test file before this): three tests round-tripping
  deliberately non-unit-norm quaternions, opacities near both `(0, 1)`
  boundaries, and scales spanning `1e-4` to `10`, asserting the values
  actually land in their documented post-read range/normalization, not
  just "close to some round-tripped number."
- `tests/gs_experiment/test_gs_splat_scene.py::test_load_from_gsplat_checkpoint_fields_satisfy_their_documented_invariants`:
  loads a synthetic checkpoint through the *real* `load_from_gsplat_checkpoint`
  path (not just `ply_io` in isolation) with non-unit quaternions and
  opacities/scales spanning several orders of magnitude, and asserts
  positions are finite, opacities are in `(0, 1)`, scales are positive,
  rotations are unit-norm, and `colors` lands in a sane real-color range —
  the same shape of check that would have caught the section-4 bug
  directly (and does, via the narrower regression test already added in
  section 4).

Both new tests pass, along with the full existing suite:
`pytest tests/` — **163 passed** (159 pre-existing + 4 new: 3 in the new
`test_gs_ply_io.py`, 1 added to `test_gs_splat_scene.py`).

**Candidates considered but not added** (lower value or already covered):
a general opacities/scales/positions range check on every existing
`SplatScene`-constructing test would be broad but mostly redundant with
the one new invariants test above; a numerical cross-check of
`gsplat_alpha_compositing_weights` against `ray_transmittance_weights` for
a shared scene already exists implicitly via each having its own
independent test file and is GPU-gated, not revisited here; deeper
property-based (Hypothesis-style) fuzzing of the kernel math was judged
out of proportion to this pass's time-boxing given no concrete suspicion
was found there.

### What a future re-run should check

Nothing in this section requires a GPU re-run to *confirm impact*, by
design — 5a is a comment-only fix, 5b is a reachability finding (confirmed
directly against the current repo's data, not run), 5c is a set of
ruled-out leads (each confirmed by reading source/tests, not by running
experiments), and 5d is new test coverage (already green under
CPU-only `pytest tests/`). If `run_nll_experiment`'s `<nbv_dir>/baseline`
data layout is ever regenerated in a future session, that would be the one
concrete follow-up worth checking: whether `NLL_EXPERIMENT_COMMON_KWARGS`'s
`bq_sigma=0.9`/`bq_window_radius=1.6` still suit that data's actual spatial
scale (per 5b) — unrelated to re-validating anything from section 4.

## 6. The three headline paper figures, actually regenerated (section 4c's estimate measured directly, and corrected)

Section 4c above flagged the three main-paper figures
(`scene_gallery.png`/Figure 1, `lego_splat_sweep.png`/Figure 2,
`coverage_uncertainty_sweep.png`/Figure 3 in `paper/main.tex`'s
`\S sec:experiments`) as stale but out of scope to regenerate, and
estimated the coverage-sweep figure's absolute raw-variance numbers would
grow roughly 6-8x under a corrected bandwidth. All three have now actually
been regenerated with the corrected sigma/kappa/colors, and the coverage
sweep's real numbers were measured directly rather than estimated.

**Checkpoints were not retrained** — confirmed directly before starting:
all three scripts only read existing `splats.ply` checkpoints under
`gs_experiment/local_runs/*_prepared/` (`render_reconstruction.render_views`/
`compute_uncertainty_maps`, no training call anywhere in any of the three
scripts or their imports). Every checkpoint they need already existed on
disk; none needed retraining.

### 6a. `scene_gallery.png` (Figure 1) — regenerated, self-corrected as expected

`.venv-gsplat/bin/python gs_experiment/scripts/render_scene_gallery.py`,
no code changes needed (`sigma=None`/`kappa=None` fits per checkpoint at
runtime, as section 4c anticipated). Ran clean; new fitted sigmas range
~0.09-0.24 across the 7 scenes x 2 budgets (e.g. lego/500: 0.1658,
lego/wide: 0.1359 — consistent with section 4's ~1.9x correction on the
old 0.070-0.080 range). One checkpoint (mic/wide) fell back to
`FALLBACK_SIGMA` rather than fitting (`sigma=0.1393 (fallback)`) — this
correctly exercises the just-fixed `render_reconstruction.FALLBACK_SIGMA`
constant (0.0694 -> 0.13926, from the earlier constant fix this session),
and the fallback path prints which case fired exactly as section 5c's
audit of this pattern already confirmed it does. Held-out PSNR values are
unchanged from before (e.g. lego/500: 21.41dB, lego/wide: 40.40dB — the
same lego numbers `lego_splat_sweep.png` reports below), as expected: PSNR
never touches `colors`' fitted-bandwidth path. `scene_gallery.png`
overwritten in place; file size 5,824,624 -> 5,818,261 bytes (content
changed — see 6d for git-tracked status).

`scene_gallery_500.png` was **not** regenerated, and on inspection should
not be treated as a live headline figure at all: the current
`render_scene_gallery.py` has no code path that produces a
single-budget-only PNG (its `run()`/`main()` always render both budgets
into one combined figure via `BUDGETS`), and `paper/main.tex` does not
reference `scene_gallery_500.png` anywhere (`grep` confirms only
`figures/scene_gallery.png` is `\includegraphics`'d, and only
`scene_gallery.png` exists under `paper/figures/`). Per `git log`, this
file was last written by an earlier version of the pipeline (commit
`5605af5`, whose own message says the new `render_splat_sweep_gallery.py`
was added "instead of separate `scene_gallery_*.png` files") and has not
been touched by any script since — it is a stale, orphaned, still-tracked
artifact from a superseded pipeline stage, not a currently-produced or
paper-referenced figure. Flagged here rather than silently left alone or
silently deleted; no action taken on it beyond this note, since deleting a
tracked file wasn't asked for.

### 6b. `lego_splat_sweep.png` (Figure 2) — a real, pre-existing bug found and fixed along the way; PSNR confirmed unaffected

First attempt crashed after successfully computing and printing all 6
budgets' rows: `KeyError: 'budget_rows'` in `render_scene_gallery.plot_gallery`.
Cause, unrelated to the colors bug: `plot_gallery` was refactored in an
earlier commit (`58006cd`) to a multi-budget side-by-side layout, where
each row dict must carry a `row["budget_rows"]` list of `(label, row)`
pairs (used to show two budgets side by side per scene in
`scene_gallery.png`). `render_splat_sweep_gallery.run()` was never updated
to match — it still built old-style flat row dicts (one budget per row,
no `budget_rows` key) and passed them straight to the new `plot_gallery`.
This means the currently-committed `lego_splat_sweep.png` (last written at
commit `58006cd`, i.e. after the refactor) could not actually have been
produced by the code as it stood at that commit; it must predate the
refactor's own commit despite `git log` attributing it there (plausibly
carried through unchanged in that commit's diff, or produced by a locally
modified/reverted copy of the script — not investigated further, out of
scope for this task). Fixed by wrapping each single-budget row as its own
one-entry `budget_rows` list
(`gs_experiment/scripts/render_splat_sweep_gallery.py`, `run()`), matching
`render_scene_gallery.py`'s current row-per-budget contract instead of the
side-by-side one it stopped assuming.

Re-run after the fix, `.venv-gsplat/bin/python
gs_experiment/scripts/render_splat_sweep_gallery.py`: succeeded, all 6
budgets (500, 10k, 30k, 100k, wide/300k, 1M). Held-out PSNR:
**21.41, 30.50, 34.85, 38.74, 40.40, 41.01 dB** — matches
`paper/main.tex`'s quoted "$21.4\,$dB at $500$ splats to $30.5$, $34.9$,
$38.7$, $40.4$, and $41.0\,$dB at $1{,}000{,}000$" exactly (to rounding),
confirming directly (not just by inference) that PSNR is unaffected by the
colors bug, as expected — it is pure reconstruction quality, computed by
the real gsplat renderer against ground truth, and never touches
`SplatScene.colors` or any fitted bandwidth. Fitted sigma ranges from
0.1658 (500 splats) down to 0.1359 (wide) and back up to 0.1534
(1M splats) — non-monotonic in splat count, unlike the old pre-fix
sigmas, but this pattern (fit sigma dips at the "wide" density and rises
again at the extremes) is a property of the marginal-likelihood fit
itself, not something this task re-derives further. `lego_splat_sweep.png`
overwritten in place; 3,644,741 -> 3,341,849 bytes.

### 6c. `coverage_uncertainty_sweep.png` (Figure 3) — the one expected to move, and it did, in the opposite direction from the section-4c estimate

Confirmed before running: `render_coverage_uncertainty_sweep.py` imports
`LEGO_GAP_SIGMA`/`LEGO_GAP_KAPPA`/`LEGO_GAP_WINDOW_RADIUS` directly from
`real_directional_coverage_experiment.py` and uses them as `build_columns`'s
and `main`'s own default argument values (no separately hardcoded value in
this script itself) — so the constant fix already applied to
`real_directional_coverage_experiment.py` this session (`LEGO_GAP_SIGMA`
0.0694 -> 0.13926) is picked up by a bare re-run with no further code
change, exactly as the task's plan anticipated.

`.venv-gsplat/bin/python gs_experiment/scripts/render_coverage_uncertainty_sweep.py gs_experiment/local_runs/lego_prepared`:

| gap half-width | PSNR (paper, old) | PSNR (measured, new) | mean var (paper, old) | mean var (measured, new) | median var (new) | ratio new/old (mean) |
|---|---:|---:|---:|---:|---:|---:|
| 0deg  | 35.1dB | 35.12dB | 0.82  | 0.0838  | 0.0757  | 0.102 (**9.79x smaller**) |
| 15deg | 29.2dB | 29.19dB | 8.54  | 1.2536  | 1.2482  | 0.147 (**6.81x smaller**) |
| 30deg | 22.9dB | 22.91dB | 22.12 | 3.3428  | 3.1356  | 0.151 (**6.62x smaller**) |
| 50deg | 15.1dB | 15.13dB | 59.45 | 8.8861  | 8.9996  | 0.149 (**6.69x smaller**) |
| 75deg | 16.1dB | 16.09dB | 78.85 | 11.7689 | 12.0745 | 0.149 (**6.70x smaller**) |

PSNR matches the paper's quoted numbers almost exactly (35.1/29.2/22.9/
15.1/16.1dB vs. measured 35.12/29.19/22.91/15.13/16.09dB) — confirmed
directly, not assumed, that reconstruction-quality PSNR is unaffected by
the colors bug here too, same reasoning as 6b.

**Mean raw variance did not grow 6-8x as section 4c estimated — it
*shrank* by almost exactly the same magnitude, 6.6x-9.8x, at every gap
condition.** This is the opposite direction from the prediction, not just
a different magnitude, and is worth stating plainly rather than folding
into a vague "the estimate was off." Section 4c's reasoning extrapolated
from an unrelated, indirectly-documented empirical precedent (an "18x
sigma difference -> ~12,000x raw-variance difference" observation cited in
`render_reconstruction.py`'s docstring, direction unspecified there) and
implicitly assumed larger sigma means larger raw variance. That assumption
was wrong for this specific kernel. Checked directly in
`gs_experiment/kernels.py`: `RBFKernel.k(x, y)` is a **normalized Gaussian
density**, `exp(-(x-y)^2/(2*sigma^2)) / (sigma*sqrt(2*pi))`, not a
fixed-amplitude covariance — so its own self-covariance
`k(x,x) = 1/(sigma*sqrt(2*pi))` *shrinks* as sigma grows, the opposite of
a standard fixed-amplitude RBF covariance kernel. The position kernel used
here is a 3-way `ProductKernel` of this 1D RBF over x/y/z
(`gs_experiment/pixel_uncertainty.py`'s `make_default_3d_position_kernel`),
so the joint prior self-covariance scales as `1/sigma^3`. The sigma
correction here is `0.13926/0.0694 = 2.0066`, and `2.0066^3 = 8.08` — a
predicted ~8x *decrease* in prior (and hence posterior) variance purely
from this normalization effect, closely matching the measured 6.6x-9.8x
decrease across all 5 gap conditions. This is also consistent with, not
contradicted by, section 4b's own earlier finding (the calibration
experiment's "sharpness"/mean realized posterior variance on `wide`
checkpoints dropped 5x-16x as sigma grew 1.45x-2.00x) — 4b already showed
this same shrinks-with-sigma direction directly; section 4c's estimate for
this specific figure should have used that same measured direction instead
of the unrelated docstring precedent, and didn't.

**What is preserved, confirmed directly rather than just inferred from
design as section 4c did**: the qualitative claim in `paper/main.tex`
("the raw rendering-aware posterior variance grows monotonically with the
gap, by roughly two orders of magnitude") holds and is if anything
slightly *stronger* now: old max/min ratio (gap75/gap0) was
`78.85/0.82 = 96.2x`; new is `11.7689/0.0838 = 140.4x`. The condition-to-
condition growth factors are nearly unchanged in relative terms (old
stepwise ratios 10.41x / 2.59x / 2.69x / 1.33x for the four steps
0->15->30->50->75deg; new stepwise ratios 14.96x / 2.67x / 2.66x / 1.32x)
— three of the four steps match old to within 3%, exactly as section 4c's
design-based reasoning predicted (one shared pooled sigma applied
uniformly across all 5 conditions means the *relative* growth pattern is
governed by real per-condition data differences, not by sigma's absolute
value). Only the first step (gap 0 -> 15deg) shifted more than the others
(10.41x -> 14.96x), consistent with the gap-0 condition's own
disproportionately larger 9.79x shrink (vs. ~6.6-6.8x for the other four)
— not explained further here.

**What this means for `paper/main.tex` (not edited — flagged for the
user's review)**: the quoted sentence "the raw rendering-aware posterior
variance grows monotonically with the gap, by roughly two orders of
magnitude: mean variance $0.82 \to 8.54 \to 22.12 \to 59.45 \to 78.85$"
needs its five absolute numbers replaced with
$0.084 \to 1.254 \to 3.343 \to 8.886 \to 11.769$ (or the corresponding
medians, $0.076 \to 1.248 \to 3.136 \to 9.000 \to 12.075$, if the paper
would rather quote medians) — the "roughly two orders of magnitude"
qualitative framing needs no change and remains accurate (if anything
slightly conservative: it's now closer to 2.15 orders of magnitude,
140x, than the old 96x). `coverage_uncertainty_sweep.png` overwritten in
place; 1,198,388 -> 1,121,711 bytes.

### 6d. Git-tracked status of the four PNGs — a real, direct content change to committed files

`git check-ignore -v` on all four returns nothing for any of them (exit
code 1: none are gitignored). All four are tracked
(`git ls-files gs_experiment/results/*.png` lists all four), and
`git status --short` now shows three of them as modified in the working
tree: `scene_gallery.png`, `lego_splat_sweep.png`,
`coverage_uncertainty_sweep.png` (all regenerated this session — real
binary content changes to already-committed files, not new files).
`scene_gallery_500.png` shows **no** diff — untouched, per 6a's finding
that it is stale/orphaned and not produced by any current script path.
None of these were staged or committed (per this task's own instructions);
they sit as unstaged working-tree changes alongside the already-modified
`FINDINGS.md`/`kernel_family_ablation_results.json`/
`rendering_aware_calibration_results.json` from section 4, pending the
user's own review before any commit.

Scripts/data: `gs_experiment/scripts/render_scene_gallery.py`,
`render_splat_sweep_gallery.py` (bug fix as described in 6b),
`render_coverage_uncertainty_sweep.py`; regenerated PNGs at
`gs_experiment/results/{scene_gallery,lego_splat_sweep,coverage_uncertainty_sweep}.png`
(tracked, now modified in the working tree — see 6d); checkpoints at
`gs_experiment/local_runs/{lego,chair,drums,ficus,hotdog,mic,ship}_prepared/`
(all pre-existing, none retrained).
