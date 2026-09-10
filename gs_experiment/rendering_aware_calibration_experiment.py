"""Calibration-methodology follow-up to `kernel_family_ablation.py`: every
calibration number this project has reported so far (that script's own
`calibration_metrics`, `gs_experiment/results/FINDINGS.md` sections 2/2b,
`paper/main.tex`'s Appendix Tables II-VIII) pairs the BQ posterior variance
`u_BQ` -- computed around the BQ posterior mean `C_BQ` -- against the squared
error of a *different* quantity, `C_alpha`: the real gsplat alpha-compositing
renderer's actual output. `u_BQ` was never computed as a variance around
`C_alpha`; pairing them is not obviously a coherent probabilistic statement,
and is a real candidate explanation for why calibration correlation has been
weak/sign-inconsistent everywhere despite `u_BQ` clearly tracking *something*
real (sparsity, directional coverage).

This module tests the coherent fix directly against real checkpoints, using
machinery already built and unit-tested this session -- NOT reimplemented
here:

  - `gs_experiment.quadrature.rendering_aware_alternative_weight_risk`: the
    general RKHS worst-case-squared-error quadratic form `e(w)^2 = z0 -
    2w@z + w@K@w` for ANY real weight vector `w` (not just the BQ-optimal
    `w* = K^-1 z`), which reduces exactly to the BQ posterior variance at
    `w=w*` (proven by `tests/gs_experiment/test_render_weight.py`) and is
    never smaller for any other real weight vector.
  - `gs_experiment.pixel_uncertainty.LocalUncertaintyEngine.
    rendering_aware_alpha_risk_along_ray`: returns
    `RenderingAwareAlphaRisk(mean, variance, alpha_mean, alpha_risk)` --
    `mean`/`variance` are the usual BQ-optimal (C_BQ, u_BQ) pair;
    `alpha_mean`/`alpha_risk` are the real local alpha-compositing
    quadrature rule's own predicted value and RKHS risk, using the real
    `w_i = T_i*alpha_i` transmittance weights from
    `visibility_attribution.ray_transmittance_weights`.

Five variants are compared at every query point (see this module's
docstrings below and `gs_experiment/results/FINDINGS.md`'s new section for
the full table):

  1. existing post-hoc (what's currently in the paper): mean=C_alpha (real),
     var=u_BQ
  2. coherent BQ renderer: mean=C_BQ, var=u_BQ
  3. BQ risk for the alpha renderer: mean=C_alpha (real),
     var=R_alpha = u_BQ + (C_BQ - C_alpha)^2
  4. constant baseline (sanity-check null model): mean=C_alpha (real),
     var=one global fitted constant (MLE = mean squared error over all this
     checkpoint's query points)
  5. real alpha-compositing quadrature's own risk: mean=C_alpha_local
     (windowed, NOT the same as the real full-scene C_alpha -- see below),
     var=alpha_risk

All means/uncertainties here are SCALAR (this project's established
`scene.colors`/`values` convention -- a per-splat scalar, the SH DC term
averaged over channels), so `C_alpha`/ground truth are also reduced to a
scalar (channel-mean of the real RGB pixel value at the same macropixel) for
a like-for-like comparison across all five variants -- see
`_macropixel_gray`. This differs from `kernel_family_ablation.py`'s own
`squared_error` (a channel-wise MSE over full RGB, never reduced to a
scalar first); that's the right metric for a raw-variance-vs-RGB-error
check, but not usable here since C_BQ/C_alpha_local only exist as scalars
by construction.

Phase A (per-point, all 7 scenes x 2 checkpoints, mandatory): reuses
`kernel_family_ablation.py`'s `SCENES`/`CHECKPOINTS`/`EVAL_DIRS`/
`fit_all_families`/`_render_and_unproject` verbatim. Phase B (image-level,
scope reduced for tractability): reconstructs a 112x42 "image" from C_BQ at
every valid pixel of a few held-out views for a handful of scenes, and
reports PSNR/SSIM against the real renderer's own output at the same
resolution, plus the two failure-mode diagnostics the user anticipated:
fraction of C_BQ predictions outside [0,1] before clipping, and fraction of
negative BQ posterior weights (w* = K^-1 z entries).

Needs the gsplat/torch interpreter (`.venv-gsplat/bin/python`).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.stats import pearsonr, spearmanr

from gs_experiment.kernel_family_ablation import CHECKPOINTS, EVAL_DIRS, SCENES, _render_and_unproject, fit_all_families
from gs_experiment.kernels import ProductKernel, RBFKernel
from gs_experiment.pixel_uncertainty import LocalUncertaintyEngine

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_JSON = REPO_ROOT / "gs_experiment" / "results" / "rendering_aware_calibration_results.json"

# Phase B (image-level) scope: full 7x2 would be a lot of per-pixel along-ray
# queries at 112x42 resolution (each roughly as expensive as one BQ solve
# over its own candidate window) -- reduced to lego (mandatory) + 3 other
# scenes spanning different splat/geometry character (chair: simple convex
# shape; hotdog: large smooth surface; ficus: thin/fine structure), 2 views
# each, `wide` checkpoint only (the best-quality reconstruction, the
# relevant regime for "does this render competitively").
PHASE_B_SCENES = ["lego", "chair", "hotdog", "ficus"]
PHASE_B_N_VIEWS = 2


# ---------------------------------------------------------------------------
# Shared low-res macropixel sampling (scalar/grayscale, matching this
# project's scalar-color convention -- see module docstring).
# ---------------------------------------------------------------------------


def _macropixel_gray(img: np.ndarray, depth_width: int, depth_height: int) -> np.ndarray:
    """Channel-mean value at each macropixel, nearest-neighbor sampled from
    a full-res (H, W, 3) image -- the same NN grid `kernel_family_ablation.
    _downsample_squared_error` uses (co-located with the depth/alpha map
    `_render_and_unproject` renders at exactly this resolution), collapsed
    to a scalar via channel mean so it's directly comparable to this
    project's scalar `C_BQ`/`C_alpha_local`."""
    height, width, _ = img.shape
    ys = (np.arange(depth_height) * height / depth_height).astype(int).clip(0, height - 1)
    xs = (np.arange(depth_width) * width / depth_width).astype(int).clip(0, width - 1)
    return img[np.ix_(ys, xs)].mean(axis=-1)


# ---------------------------------------------------------------------------
# Phase A: per-point 5-variant calibration comparison.
# ---------------------------------------------------------------------------


def _build_engine(scene, sigma_rbf: float, seed: int = 0) -> LocalUncertaintyEngine:
    """One engine per checkpoint, reused across every view/query point --
    includes real scales/rotations (unlike kernel_family_ablation.py's own
    engine, which is position-only) so the along-ray render weight's
    moment-matched footprint uses each candidate's real physical covariance
    rather than treating it as a point (see
    pixel_uncertainty._render_weight_from_local_weights's docstring for why
    that matters for a real, sharply-weighted checkpoint) -- matching
    render_reconstruction.compute_uncertainty_maps's own directional_engine
    construction, the project's actual production usage of the along-ray
    path, rather than kernel_family_ablation.py's simplified box-weight
    ablation harness (a different, position-only render weight entirely,
    so exact numeric agreement with that script's own RBF row isn't
    expected here anyway)."""
    bounds = tuple((scene.positions[:, d].min(), scene.positions[:, d].max()) for d in range(3))
    return LocalUncertaintyEngine(
        positions=scene.positions,
        values=scene.colors,
        pos_kernel=ProductKernel([RBFKernel(sigma=sigma_rbf)] * 3),
        scene_bounds=bounds,
        opacities=scene.opacities,
        scales=scene.scales,
        rotations=scene.rotations,
        seed=seed,
    )


def _collect_phase_a_records(
    engine: LocalUncertaintyEngine,
    sigma_rbf: float,
    radius: float,
    checkpoint_dir: Path,
    eval_dir: Path,
    view_indices,
    max_points_per_view: int = 80,
    depth_width: int = 112,
    depth_height: int = 42,
    seed: int = 0,
) -> List[dict]:
    """Per query point: real GT/C_alpha (channel-mean, `_macropixel_gray`)
    plus `rendering_aware_alpha_risk_along_ray`'s full (C_BQ, u_BQ,
    C_alpha_local, alpha_risk) tuple -- exactly the ingredients every one of
    the 5 variants needs. Reuses `_render_and_unproject` for the real
    GT/recon/world_points/valid data (identical to
    kernel_family_ablation.calibration_metrics's own real-held-out-view
    pipeline) and the same up-to-`max_points_per_view`-per-view subsampling
    convention.
    """
    from gs_experiment.nerf_transforms import camera_pose_from_c2w, load_transforms

    rendered = _render_and_unproject(checkpoint_dir, eval_dir, view_indices, depth_width=depth_width, depth_height=depth_height)
    _, frames = load_transforms(str(eval_dir / "transforms.json"))

    rng = np.random.default_rng(seed)
    records = []
    for view_idx, view in zip(view_indices, rendered):
        _, c2w = frames[view_idx]
        camera = camera_pose_from_c2w(c2w)
        camera_index = engine.build_bearing_index(camera)

        gt_gray = _macropixel_gray(view["gt"], depth_width, depth_height)
        recon_gray = _macropixel_gray(view["recon"], depth_width, depth_height)

        ys, xs = np.where(view["valid"])
        if len(ys) == 0:
            continue
        if len(ys) > max_points_per_view:
            keep = rng.choice(len(ys), size=max_points_per_view, replace=False)
            ys, xs = ys[keep], xs[keep]

        for y, x in zip(ys, xs):
            point = view["world_points"][y, x]
            result = engine.rendering_aware_alpha_risk_along_ray(point, camera_index, radius=radius, sigma_rbf=sigma_rbf)
            records.append(
                {
                    "gt": float(gt_gray[y, x]),
                    "c_alpha": float(recon_gray[y, x]),
                    "c_bq": float(result.mean),
                    "u_bq": float(result.variance),
                    "c_alpha_local": float(result.alpha_mean),
                    "alpha_risk": float(result.alpha_risk),
                }
            )
    return records


def _ause(uncertainty: np.ndarray, se: np.ndarray) -> float:
    """AUSE-style risk-coverage area: sort ascending by `uncertainty` (most
    confident first), compute the cumulative mean squared error over the
    k most-confident points for every k=1..n (the sparsification curve),
    and compare against the oracle curve (the same construction, sorted by
    the TRUE `se` instead). Returned as the trapezoidal-integrated area
    between the two curves over fraction-of-data-retained in [0,1] --
    near 0 means the predicted uncertainty ranks points almost as well as
    the true error itself would; lower is better; it is signed (can go
    slightly negative from numerical noise when n is small, but is
    bounded below by 0 in expectation since the oracle curve is itself the
    best-possible monotonic sparsification of `se`)."""
    n = len(se)
    if n < 2:
        return float("nan")
    order_pred = np.argsort(uncertainty)
    order_oracle = np.argsort(se)
    pred_curve = np.cumsum(se[order_pred]) / np.arange(1, n + 1)
    oracle_curve = np.cumsum(se[order_oracle]) / np.arange(1, n + 1)
    fractions = np.arange(1, n + 1) / n
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2.0 renamed trapz -> trapezoid
    return float(trapezoid(pred_curve - oracle_curve, fractions))


def _variant_metrics(se: np.ndarray, var: np.ndarray) -> dict:
    """The 5 metrics the prompt specifies for one (squared_error, variance)
    pair: Gaussian NLL, Pearson/Spearman correlation, 1-sigma/2-sigma
    empirical coverage, sharpness (mean variance), and AUSE."""
    se = np.asarray(se, dtype=float)
    var = np.asarray(var, dtype=float)
    var_floored = np.maximum(var, 1e-12)

    if np.std(var) > 0 and np.std(se) > 0:
        pearson_r, pearson_p = pearsonr(var, se)
        spearman_r, spearman_p = spearmanr(var, se)
    else:
        pearson_r = pearson_p = spearman_r = spearman_p = float("nan")

    nll = 0.5 * (se / var_floored + np.log(var_floored))
    return {
        "n": int(len(se)),
        "mean_nll": float(np.mean(nll)),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "coverage_1sigma": float(np.mean(se <= var_floored)),
        "coverage_2sigma": float(np.mean(se <= 4.0 * var_floored)),
        "sharpness_mean_var": float(np.mean(var)),
        "ause": _ause(var, se),
    }


def compute_five_variants(records: List[dict]) -> dict:
    """The 5-variant table (see module docstring) from a list of per-point
    records (each with gt/c_alpha/c_bq/u_bq/c_alpha_local/alpha_risk)."""
    gt = np.array([r["gt"] for r in records])
    c_alpha = np.array([r["c_alpha"] for r in records])
    c_bq = np.array([r["c_bq"] for r in records])
    u_bq = np.array([r["u_bq"] for r in records])
    c_alpha_local = np.array([r["c_alpha_local"] for r in records])
    alpha_risk = np.array([r["alpha_risk"] for r in records])

    se_alpha = (c_alpha - gt) ** 2
    se_bq = (c_bq - gt) ** 2
    se_alpha_local = (c_alpha_local - gt) ** 2
    r_alpha = u_bq + (c_bq - c_alpha) ** 2

    const_var = float(np.mean(se_alpha))  # MLE of a single global constant variance

    variants = {
        "1_existing_posthoc": _variant_metrics(se_alpha, u_bq),
        "2_coherent_bq_renderer": _variant_metrics(se_bq, u_bq),
        "3_bq_risk_for_alpha_renderer": _variant_metrics(se_alpha, r_alpha),
        "4_constant_baseline": _variant_metrics(se_alpha, np.full_like(se_alpha, const_var)),
        "5_alpha_quadrature_own_risk": _variant_metrics(se_alpha_local, alpha_risk),
    }
    variants["4_constant_baseline"]["fitted_constant_variance"] = const_var
    return variants


def run_checkpoint_phase_a(scene_name: str, checkpoint_name: str, seed: int = 0) -> dict:
    from gs_experiment.nerf_transforms import load_transforms
    from gs_experiment.splat_scene import load_from_gsplat_checkpoint

    spec = CHECKPOINTS[scene_name][checkpoint_name]
    checkpoint_dir, radius = spec["dir"], spec["window_radius"]
    eval_dir = EVAL_DIRS[scene_name]

    print(f"\n=== phase A: {scene_name}/{checkpoint_name} ({checkpoint_dir}) ===")
    scene = load_from_gsplat_checkpoint(str(checkpoint_dir), use_gpu_attribution=True, attribution_min_opacity=0.1)
    print(f"  {len(scene.positions)} splats, {int((scene.opacities > 0.1).sum())} above opacity 0.1")

    fitted = fit_all_families(scene, window_radius=radius, seed=seed)
    sigma_rbf = fitted["rbf"]
    print(f"  fitted rbf sigma = {sigma_rbf:.5f}")

    engine = _build_engine(scene, sigma_rbf, seed=seed)

    _, eval_frames = load_transforms(str(eval_dir / "transforms.json"))
    n_frames = len(eval_frames)
    view_indices = list(range(0, n_frames, max(1, n_frames // 6)))[:6]

    records = _collect_phase_a_records(engine, sigma_rbf, radius, checkpoint_dir, eval_dir, view_indices, seed=seed)
    variants = compute_five_variants(records)

    for name, m in variants.items():
        print(
            f"  [{name:30s}] n={m['n']:4d} nll={m['mean_nll']:9.3f} pearson={m['pearson_r']:+.3f} "
            f"spearman={m['spearman_r']:+.3f} cov1s={m['coverage_1sigma']:.3f} cov2s={m['coverage_2sigma']:.3f} "
            f"sharp={m['sharpness_mean_var']:.4g} ause={m['ause']:+.4g}"
        )

    return {"sigma_rbf": sigma_rbf, "window_radius": radius, "variants": variants, "records": records}


# ---------------------------------------------------------------------------
# Phase B: image-level reconstruction quality + failure-mode diagnostics.
# ---------------------------------------------------------------------------


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


def _valid_bbox(valid: np.ndarray, pad: int = 2):
    """Tight bounding box (with a small pad) around the valid/foreground
    mask, clipped to the array's own extent. Used to scope Phase B's
    PSNR/SSIM to the reconstructed object, not the surrounding background
    -- see run_view_phase_b's docstring for why the whole-frame comparison
    is not a fair/informative one here."""
    ys, xs = np.where(valid)
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, valid.shape[0])
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, valid.shape[1])
    return y0, y1, x0, x1


def _ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    """Wraps train_minimal_gsplat._ssim (expects (H, W, 3) float32 in
    [0, 1]) for a single-channel/grayscale array by replicating it across 3
    channels -- reused, not reimplemented, per the task's explicit
    instruction (no new SSIM implementation, no new dependency)."""
    import torch

    from gs_experiment.scripts.train_minimal_gsplat import _ssim

    a3 = torch.tensor(np.stack([a, a, a], axis=-1), dtype=torch.float32)
    b3 = torch.tensor(np.stack([b, b, b], axis=-1), dtype=torch.float32)
    return float(_ssim(a3, b3).item())


def _bq_mean_and_weights(engine: LocalUncertaintyEngine, camera_index, point, radius, sigma_rbf, angular_tol=0.05, max_candidates=500):
    """One combined candidate-gather + solve for both C_BQ's mean and the
    literal BQ-optimal weight vector w* = K^-1 z -- reuses the exact same
    shared plumbing `rendering_aware_alpha_risk_along_ray` itself uses
    internally (`LocalUncertaintyEngine._along_ray_local_data` +
    `quadrature._rendering_aware_moments`), not a new implementation, and
    mirrors `test_render_weight.py::
    test_alternative_weight_risk_reduces_to_bq_variance_at_bq_weights`'s own
    `w_bq = np.linalg.solve(kxx, z)` construction. Doing both from one
    candidate-gather (rather than calling `rendering_aware_variance_along_ray`
    for the mean and a second, separate call for w*) halves Phase B's
    per-pixel cost -- measured to matter directly: Phase B's per-pixel
    along-ray candidate gathering is the dominant cost at real checkpoint
    density (~150-200s for a single 112x42 view's ~1200 valid pixels on
    the `wide` (300k-splat) checkpoint), so avoiding a second redundant
    pass over the same candidates is not optional at this project's real
    scale. `mean = w* @ local_values` is exactly `_posterior_mean_variance`'s
    own `moment_vector @ solve(kxx, values)` formula, not an approximation
    of it.
    """
    from gs_experiment.quadrature import _rendering_aware_moments

    idx, local_positions, local_values, render_weight, _alpha_weights = engine._along_ray_local_data(
        point, camera_index, radius, None, angular_tol, max_candidates
    )
    _, kxx, z, z0 = _rendering_aware_moments(local_positions, render_weight, sigma_rbf, None, None, "closed_form", 1e-4)
    if kxx is None or z.shape[0] == 0:
        return 0.0, np.zeros(0)
    try:
        w_star = np.linalg.solve(kxx, z)
    except np.linalg.LinAlgError:
        return 0.0, np.zeros(0)
    return float(w_star @ local_values), w_star


def run_view_phase_b(
    engine: LocalUncertaintyEngine,
    sigma_rbf: float,
    radius: float,
    checkpoint_dir: Path,
    eval_dir: Path,
    view_idx: int,
    depth_width: int = 112,
    depth_height: int = 42,
) -> dict:
    """Reconstructs a (depth_height, depth_width) grayscale image from
    C_BQ at every valid pixel and reports PSNR/SSIM against GT, side by
    side with the real renderer's own PSNR/SSIM at the same resolution --
    plus the two failure-mode diagnostics: fraction of raw (pre-clip) C_BQ
    values outside [0, 1], and fraction of negative BQ posterior weights
    across all valid pixels.

    Background/invalid pixels are set to GT's OWN value in both the BQ and
    the real-renderer image before scoring -- neither reconstruction is
    scored on the background at all. This matters because of a real,
    pre-existing background-color mismatch found while building this:
    `render_views` (used by `_render_and_unproject`, unmodified here)
    defaults to a dark (0.05, 0.05, 0.05) background, while this dataset's
    real ground truth has a WHITE background (confirmed directly: GT
    corner pixels are exactly [1,1,1], recon corner pixels are exactly
    [0.05,0.05,0.05]). Every OTHER metric in this project (this module's
    own Phase A, kernel_family_ablation.calibration_metrics) never touches
    this mismatch because they only ever sample points inside the valid/
    foreground mask -- but a naive whole-frame PSNR/SSIM here would be,
    since ~70-90% of a typical NeRF-Synthetic frame at this resolution is
    background, almost entirely a measurement of that orthogonal
    background-color default rather than of reconstruction quality (it
    affects the real alpha-compositing renderer's own whole-frame score
    identically to the BQ reconstruction's, so it isn't a BQ-specific bug,
    but it swamps the actual signal for both -- confirmed directly: even
    restricting to a tight bounding box around the foreground mask left
    PSNR pinned at ~2.5dB for both variants, since the mask itself is an
    irregular silhouette and the bbox still contains substantial
    background). Neutralizing background in both compared images to GT's
    own value removes this confound entirely rather than only partially.
    """
    from gs_experiment.nerf_transforms import camera_pose_from_c2w, load_transforms

    rendered = _render_and_unproject(checkpoint_dir, eval_dir, [view_idx], depth_width=depth_width, depth_height=depth_height)
    view = rendered[0]
    _, frames = load_transforms(str(eval_dir / "transforms.json"))
    _, c2w = frames[view_idx]
    camera = camera_pose_from_c2w(c2w)
    camera_index = engine.build_bearing_index(camera)

    gt_gray = _macropixel_gray(view["gt"], depth_width, depth_height)
    recon_gray_full = _macropixel_gray(view["recon"], depth_width, depth_height)
    recon_gray = gt_gray.copy()  # background neutralized to GT's own value; overwritten at valid pixels below
    bq_gray_raw = gt_gray.copy()

    ys, xs = np.where(view["valid"])
    recon_gray[ys, xs] = recon_gray_full[ys, xs]

    n_negative_weights = 0
    n_total_weights = 0
    n_out_of_range = 0
    for y, x in zip(ys, xs):
        point = view["world_points"][y, x]
        mean, w_star = _bq_mean_and_weights(engine, camera_index, point, radius, sigma_rbf)
        bq_gray_raw[y, x] = mean
        if mean < 0.0 or mean > 1.0:
            n_out_of_range += 1
        if w_star.size > 0:
            n_negative_weights += int(np.sum(w_star < 0.0))
            n_total_weights += int(w_star.size)

    frac_out_of_range = float(n_out_of_range / len(ys)) if len(ys) > 0 else 0.0
    bq_gray = np.clip(bq_gray_raw, 0.0, 1.0)

    y0, y1, x0, x1 = _valid_bbox(view["valid"])
    gt_crop, recon_crop, bq_crop = gt_gray[y0:y1, x0:x1], recon_gray[y0:y1, x0:x1], bq_gray[y0:y1, x0:x1]

    return {
        "view_idx": int(view_idx),
        "n_valid_pixels": int(len(ys)),
        "bbox": [y0, y1, x0, x1],
        "psnr_bq": _psnr(bq_crop, gt_crop),
        "psnr_alpha": _psnr(recon_crop, gt_crop),
        "ssim_bq": _ssim_gray(bq_crop, gt_crop),
        "ssim_alpha": _ssim_gray(recon_crop, gt_crop),
        "frac_c_bq_out_of_range": frac_out_of_range,
        "frac_negative_bq_weights": float(n_negative_weights / n_total_weights) if n_total_weights > 0 else float("nan"),
        "n_weight_entries": n_total_weights,
    }


def run_scene_phase_b(scene_name: str, seed: int = 0, n_views: int = PHASE_B_N_VIEWS) -> dict:
    from gs_experiment.nerf_transforms import load_transforms
    from gs_experiment.splat_scene import load_from_gsplat_checkpoint

    spec = CHECKPOINTS[scene_name]["wide"]
    checkpoint_dir, radius = spec["dir"], spec["window_radius"]
    eval_dir = EVAL_DIRS[scene_name]

    print(f"\n=== phase B: {scene_name}/wide ({checkpoint_dir}) ===")
    scene = load_from_gsplat_checkpoint(str(checkpoint_dir), use_gpu_attribution=True, attribution_min_opacity=0.1)
    fitted = fit_all_families(scene, window_radius=radius, seed=seed)
    sigma_rbf = fitted["rbf"]
    engine = _build_engine(scene, sigma_rbf, seed=seed)

    _, eval_frames = load_transforms(str(eval_dir / "transforms.json"))
    n_frames = len(eval_frames)
    view_indices = list(range(0, n_frames, max(1, n_frames // 6)))[:n_views]

    views = []
    for view_idx in view_indices:
        result = run_view_phase_b(engine, sigma_rbf, radius, checkpoint_dir, eval_dir, view_idx)
        print(
            f"  view {view_idx}: PSNR bq={result['psnr_bq']:.2f}dB alpha={result['psnr_alpha']:.2f}dB | "
            f"SSIM bq={result['ssim_bq']:.4f} alpha={result['ssim_alpha']:.4f} | "
            f"frac_out_of_range={result['frac_c_bq_out_of_range']:.3f} "
            f"frac_neg_weight={result['frac_negative_bq_weights']:.3f} (n_valid={result['n_valid_pixels']})"
        )
        views.append(result)

    return {"sigma_rbf": sigma_rbf, "window_radius": radius, "views": views}


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def main(scenes: Optional[List[str]] = None, run_phase_b: bool = True, save_json: Path = RESULTS_JSON):
    import json

    scenes = scenes or SCENES
    all_results: Dict[str, dict] = {"phase_a": {}, "phase_b": {}}

    for scene_name in scenes:
        print("\n" + "#" * 100)
        print(f"# scene: {scene_name}")
        print("#" * 100)
        scene_results = {}
        for checkpoint_name in CHECKPOINTS[scene_name]:
            scene_results[checkpoint_name] = run_checkpoint_phase_a(scene_name, checkpoint_name)
        all_results["phase_a"][scene_name] = scene_results

        if run_phase_b and scene_name in PHASE_B_SCENES:
            all_results["phase_b"][scene_name] = run_scene_phase_b(scene_name)

        if save_json is not None:
            save_json.parent.mkdir(parents=True, exist_ok=True)
            with open(save_json, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\nSaved results so far to {save_json}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="+", default=None, choices=SCENES, help="Subset of scenes to run (default: all 7).")
    parser.add_argument("--no-phase-b", action="store_true", help="Skip the image-level Phase B reconstruction.")
    args = parser.parse_args()
    main(scenes=args.scenes, run_phase_b=not args.no_phase_b)
