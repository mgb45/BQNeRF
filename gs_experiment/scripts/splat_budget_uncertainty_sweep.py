"""Splat-budget sweep: how does uncertainty behave as a function of how many
splats a checkpoint is allowed to grow to (`max_splats`), holding everything
else -- scene, view coverage, training recipe -- fixed?

Hypothesis under test, not assumed (this script exists to check it, not to
confirm it): too few splats should leave real, under-resolved geometry that
BQ uncertainty should track (the same mechanism as FINDINGS.md section 1's
sparsity result); too many should let training's own floater failure mode
(FINDINGS.md's floater section) multiply, which BQ uncertainty should also
track, for a different reason. If both hold, uncertainty vs. max_splats
should be U-shaped -- high at both ends of the sweep, low in the middle --
not monotonic in splat count either way. The numbers below are reported
as-is; nothing here should be read as confirming the U-shape until the
figure is actually looked at.

Trains lego's full "wide" 100-view pool at several `max_splats` values,
reusing wide/'s own existing checkpoint for whichever budget already matches
it (avoids retraining the one data point this project already has the most
confidence in). Every condition uses the same recipe already validated for
this scene (`real_directional_coverage_experiment.LEGO_GAP_TRAIN_KWARGS`,
background_color=(1,1,1) -- see that module's docstring for why the
background color matters). All budgets are evaluated against the same
held-out wide_eval/ views.

Every budget is scored TWICE, deliberately, not once: once at the fixed,
pooled-fitted sigma/kappa used throughout this project (LEGO_GAP_SIGMA/KAPPA
below), and once at a sigma refit against that specific checkpoint's own
local neighborhoods (`hyperparams.fit_kernel_param_pooled_nd`, the same
per-checkpoint marginal-likelihood procedure `fit_hyperparameters.py` already
validates on real checkpoints). Neither choice is free of a confound on its
own: a fixed bandwidth tuned at ~300k splats' spacing may simply have too
little support at 10k splats and too much at 3M, regardless of whether the
geometry is genuinely under/over-resolved -- so an apparent effect at the
fixed sigma could be bandwidth mismatch, not real signal. But a fully
per-checkpoint-refit sigma has the opposite problem: it lets the kernel
adapt to exactly the density difference this sweep exists to measure, which
could just as easily erase a real effect as remove an artifact. Reporting
both side by side is the actual check: if the pattern (e.g. a U-shape in
uncertainty vs. max_splats) survives refitting, that's evidence it's real;
if refitting flattens either end, that end's fixed-sigma result was likely
bandwidth mismatch, not signal.

Run, wrapped in its own cgroup with a hard memory ceiling -- NOT just
`.venv-gsplat/bin/python ...` directly. Confirmed the hard way: a --budgets
3000000 run OOM-killed the whole host multiple times on a 30GB machine even
with MAX_SAFE_BUDGET/ROW_BUDGET in place (see the memory-model comment
below -- that guard undercounted real cost). Because the python process was
a child of the terminal's own shell, on a `systemd`-managed desktop that
shell is normally in the *same* cgroup as whatever terminal app spawned it,
so the OOM killer took the whole app down together with this script, not
just this process. `systemd-run --user --scope` starts a *new* cgroup, so a
future miscalibration (this one is measurement-backed but not proof against
every machine/scene) kills only this job:

    systemd-run --user --scope -p MemoryMax=20G --collect \\
        .venv-gsplat/bin/python gs_experiment/scripts/splat_budget_uncertainty_sweep.py \\
        gs_experiment/local_runs/lego_prepared

(pick MemoryMax comfortably under this machine's free RAM at launch time,
e.g. `free -h`'s "available" column minus a few GB of margin.)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from gs_experiment.hyperparams import fit_kernel_param_pooled_nd
from gs_experiment.kernels import ProductKernel, RBFKernel
from gs_experiment.nerf_transforms import load_transforms
from gs_experiment.scripts.real_directional_coverage_experiment import (
    LEGO_GAP_KAPPA,
    LEGO_GAP_SIGMA,
    LEGO_GAP_TRAIN_KWARGS,
    LEGO_GAP_WINDOW_RADIUS,
)
from gs_experiment.scripts.render_reconstruction import RESULTS_DIR, compute_uncertainty_maps, render_views
from gs_experiment.scripts.train_minimal_gsplat import train

DEPTH_RES = 64
N_EVAL_VIEWS = 8
TRAIN_BOUNDS = LEGO_GAP_TRAIN_KWARGS["bounds"]  # ((-2.5, 2.5), (-2.5, 2.5), (-2.5, 2.5))
NEAR_TRANSPARENT_OPACITY = 0.05
OPAQUE_DISPLACED_RANGE = (0.7, 0.92)

BUDGETS = [10_000, 30_000, 100_000, 300_000, 1_000_000]
BASELINE_BUDGET = 300_000  # matches wide/'s existing recipe exactly -- reuse it, don't retrain

# Two independent, additive costs make up compute_uncertainty_maps's host RSS,
# both measured directly (not estimated) against real checkpoints from this
# project's own local_runs/lego_prepared/ -- one at 1,000,000 splats, one at
# 3,000,000 (the exact checkpoint a --budgets 3000000 run produced right
# before OOM-killing the host, repeatedly, on a 30GB machine):
#
#   n_splats=1,000,000, cap=20  -> 19,683,660 rows: attribution baseline
#       2757 MiB, +2467 MiB after row-expansion (125.4 bytes/row marginal).
#   n_splats=3,000,000, cap=20  -> 59,168,609 rows: attribution baseline
#       7007 MiB, +6948 MiB after row-expansion (123.1 bytes/row marginal).
#
# The row-expansion cost is exactly what the old ROW_BUDGET capped; the
# attribution-baseline cost (reading the checkpoint + gpu_visibility_
# attribution's per-camera index lists -- scales with n_splats, not rows)
# was NOT bounded by anything, and at 3M splats alone it's already ~7GB --
# larger than this whole memory budget below. That's the actual bug: the
# old MAX_SAFE_BUDGET (derived only from ROW_BUDGET) let a 3M-splat budget
# through because its *row* count fit, never checking whether its
# unrelated, unbounded *baseline* cost already didn't. Fit as a line through
# the two measured points above (both terms, in bytes):
_ATTRIBUTION_FIXED_OVERHEAD_BYTES = 700 * 1024 * 1024  # process/import/checkpoint-read floor
_ATTRIBUTION_BYTES_PER_SPLAT = 2200  # measured ~2125 B/splat marginal; rounded up for margin
_BYTES_PER_OBSERVATION_ROW = 140  # measured ~123-125 B/row; rounded up for margin

# Target ceiling for ONE compute_uncertainty_maps call. evaluate_budget calls it
# TWICE per budget (fixed sigma, then the per-checkpoint sigma refit) in the same
# process, so this is deliberately well under this host's free RAM (`free -h`,
# ~21GB available while otherwise idle) -- not just under it -- to leave room for
# both calls plus normal desktop load (browser/Slack/etc., ~9-10GB measured here)
# plus matplotlib/render buffers, without relying on swap (already full on this
# machine from unrelated processes). Lower this further on a smaller machine;
# raise it only after remeasuring (see gs_experiment/scripts/measure_engine_memory.py
# pattern in this function's git history) -- don't just guess a bigger number.
PER_CALL_MEMORY_BUDGET_BYTES = 5 * 1024 ** 3
MIN_OBSERVATIONS_PER_SPLAT = 8  # floor -- below this the directional kernel sees too few real
# observations per splat for its per-camera fit to mean much, regardless of memory pressure.


def _attribution_baseline_bytes(n_splats: int) -> int:
    return _ATTRIBUTION_FIXED_OVERHEAD_BYTES + _ATTRIBUTION_BYTES_PER_SPLAT * n_splats


def max_observations_per_splat_for_budget(budget: int, n_training_views: int = 100) -> Optional[int]:
    """None (no cap) whenever `n_training_views` itself already bounds
    per-splat observation count below what PER_CALL_MEMORY_BUDGET_BYTES
    allows -- keeps small budgets byte-for-byte reproducible rather than
    introducing subsampling noise where it isn't needed.

    Raises ValueError if `budget`'s attribution baseline alone (before a
    single observation row is added) already exceeds the memory budget --
    that budget can't be made safe by capping rows at all, regardless of
    how low; refuse rather than silently under-cap it (see the module-level
    comment above for why the old row-only guard missed exactly this case).
    """
    baseline = _attribution_baseline_bytes(budget)
    if baseline >= PER_CALL_MEMORY_BUDGET_BYTES:
        raise ValueError(
            f"budget={budget:,}: attribution alone costs an estimated {baseline / 1024**3:.1f}GB, "
            f"already at or over PER_CALL_MEMORY_BUDGET_BYTES={PER_CALL_MEMORY_BUDGET_BYTES / 1024**3:.1f}GB "
            f"-- no observation cap can make this budget safe. This is the exact shape of budget that "
            f"OOM-killed the host repeatedly before this guard existed; lower --budgets rather than bypass this."
        )
    remaining = PER_CALL_MEMORY_BUDGET_BYTES - baseline
    max_rows = remaining // _BYTES_PER_OBSERVATION_ROW
    cap = max(MIN_OBSERVATIONS_PER_SPLAT, max_rows // budget)
    if cap * budget * _BYTES_PER_OBSERVATION_ROW > remaining and cap == MIN_OBSERVATIONS_PER_SPLAT:
        raise ValueError(
            f"budget={budget:,}: even the MIN_OBSERVATIONS_PER_SPLAT={MIN_OBSERVATIONS_PER_SPLAT} floor "
            f"would exceed the remaining {remaining / 1024**3:.1f}GB after attribution's "
            f"{baseline / 1024**3:.1f}GB baseline -- refuse rather than exceed PER_CALL_MEMORY_BUDGET_BYTES."
        )
    return None if cap >= n_training_views else cap

# Per-checkpoint sigma-refit config -- same procedure/defaults as fit_hyperparameters.py,
# applied per-budget here instead of once against a single checkpoint.
SIGMA_FIT_WINDOW_RADIUS = 0.08
SIGMA_FIT_N_WINDOWS = 25
SIGMA_FIT_MAX_WINDOW_SIZE = 60
SIGMA_FIT_MIN_OPACITY = 0.1
SIGMA_FIT_BOUNDS = (0.005, 1.0)


def _collect_windows(positions, colors, query_points, window_radius, max_window_size, seed):
    from scipy.spatial import cKDTree

    tree = cKDTree(positions)
    rng = np.random.default_rng(seed)
    datasets = []
    for p in query_points:
        idx = np.array(tree.query_ball_point(p, window_radius), dtype=int)
        if len(idx) < 6:
            continue
        if len(idx) > max_window_size:
            idx = rng.choice(idx, size=max_window_size, replace=False)
        datasets.append((positions[idx], colors[idx]))
    return datasets


def fit_sigma_for_checkpoint(checkpoint: dict, seed: int = 0):
    """Marginal-likelihood sigma refit against this specific checkpoint's own
    local neighborhoods -- the direct check for whether the fixed, pooled
    sigma is the wrong measuring stick at this checkpoint's density, which is
    exactly what varies across this sweep. Returns None if there aren't
    enough above-threshold splats to build any real window (can happen at
    the smallest budgets)."""
    keep = checkpoint["opacities"] > SIGMA_FIT_MIN_OPACITY
    positions = checkpoint["positions"][keep]
    if len(positions) < 6:
        return None
    colors = checkpoint["sh_coeffs"][keep, :, 0].mean(axis=1)
    rng = np.random.default_rng(seed)
    query_idx = rng.choice(len(positions), size=min(SIGMA_FIT_N_WINDOWS, len(positions)), replace=False)
    datasets = _collect_windows(positions, colors, positions[query_idx], SIGMA_FIT_WINDOW_RADIUS, SIGMA_FIT_MAX_WINDOW_SIZE, seed)
    if not datasets:
        return None
    fit = fit_kernel_param_pooled_nd(datasets, lambda s: ProductKernel([RBFKernel(sigma=s)] * 3), bounds=SIGMA_FIT_BOUNDS, n_grid=25)
    return float(fit.param)


def floater_fraction(positions: np.ndarray, opacities: np.ndarray, bounds=TRAIN_BOUNDS):
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    outside = np.any((positions < lo) | (positions > hi), axis=1)
    near_transparent = outside & (opacities < NEAR_TRANSPARENT_OPACITY)
    opaque_displaced = outside & (opacities >= OPAQUE_DISPLACED_RANGE[0]) & (opacities <= OPAQUE_DISPLACED_RANGE[1])
    n = len(positions)
    return dict(
        near_transparent_frac=float(near_transparent.sum()) / n,
        opaque_displaced_frac=float(opaque_displaced.sum()) / n,
        any_floater_frac=float((near_transparent | opaque_displaced).sum()) / n,
    )


def train_budget(prepared_dir: str, budget: int) -> str:
    """Returns the directory holding this budget's splats.ply + transforms.json."""
    wide_dir = os.path.join(prepared_dir, "wide")
    if budget == BASELINE_BUDGET:
        return wide_dir

    budget_dir = os.path.join(prepared_dir, f"budget_{budget}")
    ply_path = os.path.join(budget_dir, "splats.ply")
    if not os.path.exists(ply_path):
        os.makedirs(budget_dir, exist_ok=True)
        shutil.copy(os.path.join(wide_dir, "transforms.json"), os.path.join(budget_dir, "transforms.json"))
        kwargs = dict(LEGO_GAP_TRAIN_KWARGS)
        kwargs["max_splats"] = budget
        print(f"[budget={budget}] training...")
        train(wide_dir, ply_path, **kwargs)
    return budget_dir


def evaluate_budget(
    prepared_dir: str,
    budget_dir: str,
    view_indices,
    eval_frames,
    camera_angle_x,
    sigma: float,
    kappa: float,
    window_radius: float,
    depth_res: int,
    max_observations_per_splat: Optional[int] = None,
):
    eval_dir = os.path.join(prepared_dir, "wide_eval")
    results, checkpoint = render_views(eval_dir, view_indices, checkpoint_dir=budget_dir, background_color=(1.0, 1.0, 1.0))
    height, width = results[0][1].shape[:2]

    fitted_sigma = fit_sigma_for_checkpoint(checkpoint)

    def score(sigma_value):
        maps = compute_uncertainty_maps(
            eval_dir, view_indices, eval_frames, camera_angle_x, width, height, checkpoint,
            checkpoint_dir=budget_dir, sigma=sigma_value, kappa=kappa, window_radius=window_radius,
            depth_width=depth_res, depth_height=depth_res, max_observations_per_splat=max_observations_per_splat,
        )
        all_err, all_ratio = [], []
        for (idx, gt, recon), (_, dir_map) in zip(results, maps):
            err = np.abs(gt - recon).mean(axis=-1)
            valid = ~np.isnan(dir_map)
            all_err.append(err[valid])
            all_ratio.append(dir_map[valid])
        all_err = np.concatenate(all_err)
        all_ratio = np.concatenate(all_ratio)
        r, p = pearsonr(all_ratio, all_err)
        psnr = -10.0 * np.log10(max(float(np.mean(all_err ** 2)), 1e-10))
        gallery_idx = len(results) // 2
        return dict(
            psnr=psnr, mean_ratio=float(np.mean(all_ratio)), median_ratio=float(np.median(all_ratio)),
            mean_err=float(np.mean(all_err)), pearson_r=float(r), pearson_p=float(p),
            ratio_map=maps[gallery_idx][1],
        )

    fixed = score(sigma)
    fitted = score(fitted_sigma) if fitted_sigma is not None else None

    floaters = floater_fraction(checkpoint["positions"], checkpoint["opacities"])
    n_splats = len(checkpoint["positions"])

    # keep the middle held-out view's full pixel maps for the gallery row
    gallery_idx = len(results) // 2
    gallery_err = np.abs(results[gallery_idx][1] - results[gallery_idx][2]).mean(axis=-1)
    gallery = dict(
        gt=results[gallery_idx][1], recon=results[gallery_idx][2], err=gallery_err,
        ratio_map_fixed=fixed["ratio_map"], ratio_map_fitted=fitted["ratio_map"] if fitted else None,
    )

    return dict(
        budget=budget_dir, n_splats=n_splats, fitted_sigma=fitted_sigma,
        fixed=fixed, fitted=fitted, gallery=gallery, **floaters,
    )


def run(prepared_dir: str, budgets=BUDGETS, out_path=None, sigma=LEGO_GAP_SIGMA, kappa=LEGO_GAP_KAPPA,
        window_radius=LEGO_GAP_WINDOW_RADIUS, depth_res=DEPTH_RES, n_eval_views=N_EVAL_VIEWS):
    # Fail fast, before spending any GPU time training: raises ValueError (see
    # max_observations_per_splat_for_budget) for any budget the memory model above
    # can't make safe at all, same check the per-budget loop below applies anyway.
    for budget in budgets:
        max_observations_per_splat_for_budget(budget)
    eval_dir = os.path.join(prepared_dir, "wide_eval")
    camera_angle_x, eval_frames = load_transforms(os.path.join(eval_dir, "transforms.json"))
    n_views = len(eval_frames)
    view_indices = sorted(set(np.linspace(0, n_views - 1, min(n_eval_views, n_views)).round().astype(int).tolist()))

    rows = []
    for budget in budgets:
        budget_dir = train_budget(prepared_dir, budget)
        max_obs = max_observations_per_splat_for_budget(budget)
        result = evaluate_budget(
            prepared_dir, budget_dir, view_indices, eval_frames, camera_angle_x, sigma, kappa, window_radius, depth_res,
            max_observations_per_splat=max_obs,
        )
        result["requested_budget"] = budget
        rows.append(result)
        fitted_str = (
            f"fitted sigma={result['fitted_sigma']:.4f} -> mean ratio={result['fitted']['mean_ratio']:.4f}"
            if result["fitted"] is not None else "fitted sigma=n/a (too few above-threshold splats)"
        )
        cap_str = f"max_observations_per_splat={max_obs}" if max_obs is not None else "no observation cap"
        print(
            f"budget={budget:>8} (actual n_splats={result['n_splats']:>7}, {cap_str}): "
            f"held-out PSNR={result['fixed']['psnr']:.2f}dB, "
            f"[fixed sigma={sigma:.4f}] mean uncertainty ratio={result['fixed']['mean_ratio']:.4f}, "
            f"pearson r(ratio,|error|)={result['fixed']['pearson_r']:.3f}  |  [{fitted_str}]  |  "
            f"floaters(near-transparent)={result['near_transparent_frac']:.4%}, "
            f"floaters(opaque-displaced)={result['opaque_displaced_frac']:.4%}"
        )

    out_path = Path(out_path) if out_path else (RESULTS_DIR / "splat_budget_uncertainty_sweep.png")
    plot_sweep(rows, out_path)
    return rows


def plot_sweep(rows, out_path):
    n = len(rows)
    # 5 image rows: gt/recon/|error| (matching render_scene_gallery.py's and
    # render_coverage_uncertainty_sweep.py's layout), then uncertainty ratio at the fixed
    # pooled sigma AND at this checkpoint's own refit sigma, side by side -- so a reader
    # can see, by eye, both whether uncertainty tracks real |error|/floaters at all, and
    # whether that picture actually changes once the kernel is allowed to adapt to this
    # checkpoint's own density (the confound the fixed-sigma choice alone can't rule out).
    fig = plt.figure(figsize=(3 * n, 19.5))
    gs = fig.add_gridspec(6, n, height_ratios=[3, 3, 3, 3, 3, 2])
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(n)] for r in range(5)])
    trend_ax = fig.add_subplot(gs[5, :])

    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad(color=(0.4, 0.4, 0.4))
    err_vmax = float(np.percentile(np.stack([row["gallery"]["err"] for row in rows]), 95))

    for col, row in enumerate(rows):
        g = row["gallery"]
        axes[0, col].imshow(g["gt"])
        axes[0, col].set_title(
            f"max_splats={row['requested_budget']:,}\n(actual n={row['n_splats']:,}), {row['fixed']['psnr']:.1f}dB"
        )
        axes[1, col].imshow(g["recon"])
        im_err = axes[2, col].imshow(g["err"], cmap="inferno", vmin=0, vmax=err_vmax)
        fig.colorbar(im_err, ax=axes[2, col], fraction=0.046, pad=0.04)
        im_fixed = axes[3, col].imshow(g["ratio_map_fixed"], cmap=cmap, vmin=0, vmax=1)
        fig.colorbar(im_fixed, ax=axes[3, col], fraction=0.046, pad=0.04)
        axes[3, col].set_xlabel(f"mean ratio={row['fixed']['mean_ratio']:.3f}", fontsize=9)
        if g["ratio_map_fitted"] is not None:
            im_fitted = axes[4, col].imshow(g["ratio_map_fitted"], cmap=cmap, vmin=0, vmax=1)
            fig.colorbar(im_fitted, ax=axes[4, col], fraction=0.046, pad=0.04)
            axes[4, col].set_xlabel(
                f"sigma={row['fitted_sigma']:.4f}, mean ratio={row['fitted']['mean_ratio']:.3f}", fontsize=9
            )
        else:
            axes[4, col].text(0.5, 0.5, "n/a", ha="center", va="center", transform=axes[4, col].transAxes)
        for ax in axes[:, col]:
            ax.set_xticks([])
            ax.set_yticks([])

    axes[0, 0].set_ylabel("ground truth\n(held-out view)", fontsize=10)
    axes[1, 0].set_ylabel("gsplat\nreconstruction", fontsize=10)
    axes[2, 0].set_ylabel("|error|\n(mean over RGB)", fontsize=10)
    axes[3, 0].set_ylabel(f"uncertainty ratio\n(fixed sigma={LEGO_GAP_SIGMA:.4f})", fontsize=10)
    axes[4, 0].set_ylabel("uncertainty ratio\n(per-checkpoint refit sigma)", fontsize=10)

    budgets = [row["requested_budget"] for row in rows]
    mean_ratios_fixed = [row["fixed"]["mean_ratio"] for row in rows]
    mean_ratios_fitted = [row["fitted"]["mean_ratio"] if row["fitted"] else np.nan for row in rows]
    floater_fracs = [row["any_floater_frac"] for row in rows]
    psnrs = [row["fixed"]["psnr"] for row in rows]

    trend_ax.plot(budgets, mean_ratios_fixed, "o-", color="tab:red", label="mean uncertainty ratio (fixed sigma)")
    trend_ax.plot(budgets, mean_ratios_fitted, "d-", color="tab:orange", label="mean uncertainty ratio (refit sigma)")
    trend_ax.plot(budgets, floater_fracs, "^--", color="tab:purple", label="floater fraction")
    trend_ax.set_xscale("log")
    trend_ax.set_xlabel("max_splats (log scale)")
    trend_ax.set_ylabel("uncertainty ratio / floater fraction (0-1)")
    trend_ax.set_ylim(0, 1)
    trend_ax.grid(alpha=0.3)

    psnr_ax = trend_ax.twinx()
    psnr_ax.plot(budgets, psnrs, "s:", color="tab:blue", label="held-out PSNR")
    psnr_ax.set_ylabel("held-out PSNR (dB)", color="tab:blue")

    lines1, labels1 = trend_ax.get_legend_handles_labels()
    lines2, labels2 = psnr_ax.get_legend_handles_labels()
    trend_ax.legend(lines1 + lines2, labels1 + labels2, loc="upper center", fontsize=9)

    fig.suptitle(
        "Lego, fixed 100-view coverage: uncertainty ratio (fixed vs. per-checkpoint-refit sigma), "
        "floater fraction, and reconstruction quality vs. splat budget\n"
        "(hypothesis under test: U-shaped -- high at both too-few and too-many splats -- see script docstring; "
        "not assumed confirmed until the refit-sigma row/line is checked against the fixed one)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prepared_dir", help="e.g. gs_experiment/local_runs/lego_prepared")
    parser.add_argument("--out", default=None)
    parser.add_argument("--budgets", type=int, nargs="+", default=BUDGETS)
    parser.add_argument("--sigma", type=float, default=LEGO_GAP_SIGMA)
    parser.add_argument("--kappa", type=float, default=LEGO_GAP_KAPPA)
    parser.add_argument("--window-radius", type=float, default=LEGO_GAP_WINDOW_RADIUS)
    parser.add_argument("--depth-res", type=int, default=DEPTH_RES)
    parser.add_argument("--n-eval-views", type=int, default=N_EVAL_VIEWS)
    args = parser.parse_args()

    run(
        args.prepared_dir, budgets=args.budgets, out_path=args.out, sigma=args.sigma, kappa=args.kappa,
        window_radius=args.window_radius, depth_res=args.depth_res, n_eval_views=args.n_eval_views,
    )


if __name__ == "__main__":
    main()
