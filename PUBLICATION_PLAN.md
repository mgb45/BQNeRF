# Plan: a publication-ready codebase

For a branch (`release`) to be cut **after** the current experiments finish.
Nothing here is started yet. Goal: a reader clones, installs, runs one
command per table or figure in the paper, and gets the numbers in the paper.

## Where the repository actually is

Measured, not estimated:

| | count | LOC |
|---|---|---|
| tracked files | 202 | |
| Python | 68 | 12,083 |
| library modules (`gs_experiment/*.py`) | 24 | 4,678 |
| scripts | 22 | 4,576 |
| tests | 21 | 2,825 |
| tracked results (JSON/PNG/MD) | 111 | |

Three problems, in order of size.

### 1. About a third of the code belongs to a retired construction

FINDINGS section 0 retired the directional-kernel construction and replaced
it with the rasterizer-probe posterior. The old stack was never removed. It
is imported only by itself and its own tests, and no current result depends
on it:

`kernels` · `quadrature` · `pixel_uncertainty` · `render_weight` ·
`hyperparams` · `visibility_attribution` · `gpu_visibility_attribution` ·
`gpu_uncertainty` · `colmap_loader` · `sh_directional_uncertainty`

That is ~2,340 LOC of library plus ~1,480 LOC of tests: **~3,800 of 12,100
lines, none of which produces a number in the paper.** Deleting it is the
single largest clarity gain available and carries almost no risk, because
nothing imports it.

Two of these need a decision rather than a deletion:

* `gpu_uncertainty` holds `u_spatial_BQ`, the finite-representation term
  (ROADMAP item 5), which answers a different question and is currently
  orphaned. This branch forces the choice: fit a signal amplitude and use
  it, or drop it and say so in the paper.
* `sh_directional_uncertainty` is still imported by
  `frozen_map_monotonicity_test`, which is a correctness check worth
  keeping. Port the check onto the current posterior first, then delete.

### 2. Third-party code the results depend on is not in the repository

`third_party/` is in `.gitignore`, so GS-U -- whose `train.py`,
`train_errors.py` and `uncertainty_metrics.py` produced every benchmark
number -- is absent, along with the commit it was taken at. Our own
`third_party_patches/gsu-save_ply-memory.patch` is tracked but has nothing
to apply to.

Fix: a git submodule pinned to the exact commit, plus a script that applies
the patch and verifies its effect. Without this the headline experiment is
not reproducible by anyone.

### 3. Installation is the step most likely to defeat a reader

Every failure we hit is a failure someone else will hit: `ninja` missing
from the service PATH, CUDA 12.3's nvcc rejecting gcc 13, gsplat JIT-
compiling on first use, GS-U's `save_ply` building 11 GB of Python tuples.
This knowledge currently lives in a comment block in
`requirements-gsplat.txt` and in commit messages.

Fix: one `install.sh` that checks the toolchain before it starts (CUDA
version against host gcc version, `ninja` present, driver present) and fails
with the actual remedy rather than a compile error 44 minutes in.

## Proposed structure

```
README.md              install, then one command per table/figure
install.sh             toolchain checks, venv, submodule, patch
Makefile               one target per paper artefact
pyproject.toml         replaces requirements.txt + requirements-gsplat.txt
splatunc/              the library (renamed from gs_experiment/)
  posterior.py           rasterized_sh_precision: the method
  baselines.py           the six reimplemented competitors
  evaluation.py          our pre-registered protocol
  protocol_gsu.py        their protocol, ported and cross-validated
  conformal.py
  io.py                  ply_io + nerf_transforms
  ablations/             coupled.py, opacity.py, fisher.py (negative results)
experiments/
  01_synthetic_gap/      75 deg cone, six baselines
  02_benchmark/          13 scenes through their pipeline
  03_selection/          per-view selection efficiency
  04_trajectory_gap/     hold-out construction and scoring
  figures/               videos and paper figures
results/                 the JSONs behind every published number
tests/                   five files
third_party/             GS-U submodule, pinned, plus the patch
```

The rename `gs_experiment` -> `splatunc` is proposed because the current
name tells a reader nothing. It touches every import, but it is a
mechanical change covered by the remaining tests.

## Entry points

A `Makefile` with **one target per artefact in the paper**, each printing
the table it reproduces and naming the FINDINGS section it corresponds to.
`make table3` should print table 3. That mapping is the single most useful
thing a reproduction codebase can offer, and it is currently absent -- the
per-view numbers in FINDINGS section 25, for instance, were computed by
ad-hoc scoring at the shell and exist in no committed script.

Also needed:

* `make smoke` -- one synthetic scene end to end in a few minutes, so the
  install can be verified before committing GPU-days.
* Documented dataset URLs **and the resolution convention**: outdoor
  `images_4`, indoor `images_2`, Tanks & Temples and Deep Blending the
  shipped `images`. Getting this wrong produces numbers that look
  comparable to the published table and are not.
* Expected runtime and peak RSS per target, from the measured figures
  (~23 min training, 2.7 min their fit, 56 s ours, per scene).

## Tests: five, not twenty-one

Keep only those protecting a claim a reader depends on:

| test | protects |
|---|---|
| `test_protocol_gsu` | the port of their scorer, cross-validated against their source -- the basis of every benchmark number |
| `test_rasterized_sh_precision` | the method itself |
| `test_evaluation` | the pre-registered protocol |
| `test_conformal` | the interval-width claim |
| `test_coupled_sh_posterior` | the ablation's correctness |

~640 LOC against the current 2,825. The other sixteen test the retired
construction and go with it.

## Sequence

1. Cut `release` from `main` once experiments are final.
2. Port the monotonicity check off `sh_directional_uncertainty`; decide
   `gpu_uncertainty`.
3. Delete the retired stack and its tests. Verify the five remaining tests
   and every experiment script still run.
4. Submodule GS-U at its pinned commit; make the patch applicable and
   checked.
5. Restructure and rename; fix imports.
6. Write `install.sh` and the `Makefile`; verify on a clean checkout in a
   fresh venv, which is the only test that matters here.
7. Rewrite `README.md` against the final structure.

## Two things deliberately not done

* **No history rewrite.** The commit history records the corrections --
  the retracted n=1 readings, the coordinate-convention bug, the KNN
  surrogate that was wrong by 2.1e18. Squashing it would hide the part of
  the record that shows the results were checked.
* **No `paper/` in the release tree** unless the venue wants it; it has its
  own build and its own `.gitignore`.
