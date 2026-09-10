"""ROADMAP.md item 2 ("Alternative kernels"): compares RBF, Matern-3/2, and
a new RationalQuadratic position kernel (added in gs_experiment/kernels.py
for this experiment -- see that class's docstring for why this family was
chosen) against real trained checkpoints of every NeRF-Synthetic scene this
project's other kept results use (chair, drums, ficus, hotdog, lego, mic,
ship -- the same 7-scene set `scripts/render_scene_gallery.py` uses, and for
the same reason: `materials` is excluded there because even its best
held-out view stays visibly hazy under this project's vanilla-3DGS training
recipe, an honestly-reported limitation unrelated to this ablation, not
re-litigated here), each at very different splat densities
(`local_runs/<scene>_prepared/wide`, ~300k splats, vs.
`.../budget_500`, 500 splats), on two honestly-measured properties:

  1. Sparsity correlation: does posterior variance track local splat
     density (denser -> lower variance, if the signal works as intended)?
  2. Calibration: does posterior variance track real held-out rendering
     error (from gsplat's own reconstruction of held-out eval views)?

No such quantitative-eval script currently exists in this repo (an older
one was deleted when the project was trimmed to its three kept qualitative
figures) -- this is written fresh, kept single-purpose to this one
comparison rather than reviving a larger multi-check eval script.

Needs the gsplat/torch interpreter (`.venv-gsplat/bin/python`) to load real
checkpoints (`splat_scene.load_from_gsplat_checkpoint`) and render held-out
views (`gsplat.rasterization`); the kernel-fitting/variance math itself is
pure numpy/scipy.

-------------------------------------------------------------------------
Fitting each family's bandwidth honestly, on the same real data
-------------------------------------------------------------------------
`gs_experiment.splat_scene.fit_kernel_hyperparams`'s sigma-fitting half
already does exactly the right thing -- sample real local (position,
color) windows via a KD-tree, fit one bandwidth via
`hyperparams.fit_kernel_param_pooled_nd` -- but it hardcodes the kernel
factory to RBF. `fit_family_bandwidth` below generalizes that window
sampling to an arbitrary `kernel_factory` (RBF/Matern/RationalQuadratic
all plug in unchanged, since `fit_kernel_param_pooled_nd` only ever calls
`kernel.k(nodes, nodes)`, not `v`/`vv`), so every family fits its own
bandwidth against the *same* real windows -- only the kernel assumption
differs, not the data. `fit_kernel_hyperparams` itself is left untouched
(other callers depend on its current RBF-specific behavior).

-------------------------------------------------------------------------
Evaluating each family's posterior variance
-------------------------------------------------------------------------
`pixel_uncertainty.LocalUncertaintyEngine.rendering_aware_variance` (the
position-only, camera-free rendering-aware BQ variance used elsewhere in
this project) hard-requires an RBF `pos_kernel`: its closed form
(`quadrature.bayesian_quadrature_rendering_aware`'s `mode="closed_form"`)
is RBF-only, by the Gaussian-product identity that closed form relies on.
That module also documents a `mode="numerical"` fallback for a general
`ProductKernel` via `scipy.integrate.nquad` over the full D-dimensional
domain -- confirmed here to be impractically slow at this project's real
scale: one such call (25 nodes, a 3D domain -- so the prior-variance half
alone is a nested 6D integral) did not finish in 120s in testing, matching
that module's own docstring warning that nquad "does not finish in any
reasonable time" past a few dimensions.

`_axis_cross_moment`/`_axis_prior_variance` below re-derive the same
Gaussian-convolution identity `quadrature.rendering_aware_moment_vector`/
`rendering_aware_prior_variance` use for RBF, generalized to *any*
stationary 1D kernel (a function of |x-y| only -- true of RBF, Matern-3/2,
and RationalQuadratic alike) via a 1D `scipy.integrate.quad` convolution
in place of RBF's closed form. This is exact (not an approximation of a
different quantity), because it only ever gets used with
`rendering_aware_variance`'s own weight convention -- an isotropic,
axis-aligned Gaussian render weight, `covariance = (radius/2)^2 * I` --
never the along-ray/gsplat variants' non-diagonal, moment-matched
covariances: a D-dimensional integral against an axis-aligned product of
one stationary kernel per axis and an isotropic Gaussian factors exactly
into a product of D independent 1D integrals, and each of those is
translation-invariant (depends only on the offset between the query point
and a node, not on absolute position), so the double-integral half
(`_axis_prior_variance`) is identical for every query point at a given
radius and is cached once per (kernel family, param, radius) rather than
recomputed per query.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy import integrate
from scipy.spatial import cKDTree
from scipy.stats import norm, pearsonr, spearmanr

from gs_experiment.hyperparams import fit_kernel_param_pooled_nd
from gs_experiment.kernels import Kernel, MaternKernel, ProductKernel, RationalQuadraticKernel, RBFKernel
from gs_experiment.pixel_uncertainty import LocalUncertaintyEngine
from gs_experiment.quadrature import BQResult, _posterior_mean_variance

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_RUNS = REPO_ROOT / "gs_experiment" / "local_runs"

# The 7 NeRF-Synthetic scenes this project's other kept results use --
# `materials` deliberately excluded, matching `scripts/render_scene_gallery.
# py`'s own documented exclusion (its best held-out view stays visibly hazy
# under this project's training recipe -- unrelated to kernel choice).
SCENES = ["chair", "drums", "ficus", "hotdog", "lego", "mic", "ship"]

# Per-checkpoint local-window radius. NOT shared across checkpoints: `wide`
# (~300k splats) and `budget_500` (500 splats) cover the *same* physical
# scene volume, so a radius chosen for one density gives a wildly different
# number of real neighbors at the other -- confirmed directly on lego:
# r=0.08 (the radius gs_experiment.splat_scene.fit_kernel_hyperparams's own
# default uses, tuned for a dense checkpoint) finds a median of ~1000
# neighbors per window in `wide` but a median of ~1-2 in `budget_500` (below
# the >=6 minimum a local GP fit needs). r=0.25 was chosen for `budget_500`
# as the smallest radius, checked directly against lego's own real splat
# positions, giving >=6 real neighbors for >90% of candidate window centers
# (mean ~22 neighbors) -- the same "tune to this checkpoint's own density"
# principle fit_kernel_hyperparams's docstring already argues for, applied
# to the window radius itself, not just the fitted bandwidth.
#
# These two radii (0.08/0.25) were re-checked directly (KD-tree neighbor
# counts at 150 sampled window centers, same recipe as the check above) on
# all 7 scenes below, not just lego, before assuming they transfer: every
# scene/checkpoint combination gives >=6 real neighbors for at least 86% of
# candidate windows (worst case: `ship`/budget_500 at 86.7%, `hotdog`/wide
# at 87.3% -- both still comfortably above the failure regime, and close to
# lego's own budget_500 baseline of 90.7%). NeRF-Synthetic scenes share
# roughly the same normalized coordinate bounds, and that held here too, so
# no scene needed a different radius from lego's.
WINDOW_RADIUS = {"wide": 0.08, "budget_500": 0.25}

CHECKPOINTS = {
    scene: {
        "wide": {"dir": LOCAL_RUNS / f"{scene}_prepared" / "wide", "window_radius": WINDOW_RADIUS["wide"]},
        "budget_500": {"dir": LOCAL_RUNS / f"{scene}_prepared" / "budget_500", "window_radius": WINDOW_RADIUS["budget_500"]},
    }
    for scene in SCENES
}
EVAL_DIRS = {scene: LOCAL_RUNS / f"{scene}_prepared" / "eval" for scene in SCENES}

FAMILIES: Dict[str, dict] = {
    "rbf": {"factory": lambda p: ProductKernel([RBFKernel(sigma=p)] * 3), "bounds": (0.005, 1.0), "param_name": "sigma"},
    "matern32": {"factory": lambda p: ProductKernel([MaternKernel(rho=p)] * 3), "bounds": (0.005, 1.0), "param_name": "rho"},
    "rational_quadratic": {
        "factory": lambda p: ProductKernel([RationalQuadraticKernel(l=p)] * 3),
        "bounds": (0.005, 1.0),
        "param_name": "l",
    },
}


# ---------------------------------------------------------------------------
# 1. Generalized (any-family) window sampling + bandwidth fitting.
# ---------------------------------------------------------------------------


def sample_sigma_windows(scene, n_windows=25, max_window_size=60, window_radius=0.08, min_opacity=0.1, seed=0):
    """Exactly `splat_scene.fit_kernel_hyperparams`'s sigma-fitting window
    sampling, factored out so every kernel family fits against the
    identical real (position, color) windows -- see this module's
    docstring."""
    keep = scene.opacities > min_opacity
    positions = scene.positions[keep]
    colors = scene.colors[keep]
    if len(positions) < 6:
        return []

    tree = cKDTree(positions)
    rng = np.random.default_rng(seed)
    query_idx = rng.choice(len(positions), size=min(n_windows, len(positions)), replace=False)
    datasets = []
    for p in positions[query_idx]:
        idx = np.array(tree.query_ball_point(p, window_radius), dtype=int)
        if len(idx) < 6:
            continue
        if len(idx) > max_window_size:
            idx = rng.choice(idx, size=max_window_size, replace=False)
        datasets.append((positions[idx], colors[idx]))
    return datasets


def fit_all_families(scene, window_radius, n_windows=25, max_window_size=60, min_opacity=0.1, seed=0) -> Dict[str, float]:
    """Fits every family in `FAMILIES` against the same real windows from
    `scene`. Returns {family_name: fitted_param}."""
    datasets = sample_sigma_windows(
        scene, n_windows=n_windows, max_window_size=max_window_size, window_radius=window_radius,
        min_opacity=min_opacity, seed=seed,
    )
    if not datasets:
        raise ValueError(f"not enough real data ({len(scene.positions)} splats) to fit any kernel family")

    fitted = {}
    for name, spec in FAMILIES.items():
        fit = fit_kernel_param_pooled_nd(datasets, spec["factory"], bounds=spec["bounds"], n_grid=25)
        fitted[name] = fit.param
    return fitted


# ---------------------------------------------------------------------------
# 2. Generalized rendering-aware BQ variance (any stationary 1D kernel family,
#    isotropic box render weight -- see module docstring for the derivation).
# ---------------------------------------------------------------------------

_Z0_CACHE: Dict[Tuple[str, float, float], float] = {}
_CONV_TABLE_CACHE: Dict[Tuple, Tuple[np.ndarray, np.ndarray]] = {}


def _kernel_cache_key(kernel: Kernel):
    if isinstance(kernel, RBFKernel):
        return ("rbf", kernel.sigma)
    if isinstance(kernel, MaternKernel):
        return ("matern32", kernel.rho)
    if isinstance(kernel, RationalQuadraticKernel):
        return ("rational_quadratic", kernel.l, kernel.alpha)
    return ("other", id(kernel))


def _axis_cross_moment_exact(kernel: Kernel, r: np.ndarray, s: float, n_sigma: float = 8.0) -> np.ndarray:
    """integral over u of kernel.k(r_i, u) * N(u; 0, s^2) du, for each
    offset r_i -- the 1D convolution of a stationary kernel with a centered
    Gaussian of std `s`, exact by translation invariance (kernel.k(x, y)
    depends only on x - y for every family used here) regardless of where
    the query point actually sits.

    RBFKernel is a *normalized density* (see its own docstring: "the
    density of N(y, sigma^2) evaluated at x"), so this reduces to the
    standard Gaussian-Gaussian convolution identity used elsewhere in this
    project (`quadrature.rendering_aware_moment_vector`):
    N(r; 0, sigma^2 + s^2). Matern/RationalQuadratic have no such closed
    form (same reason their own `v` needs `scipy.integrate.quad` instead of
    RBF's closed form), so those go through the same numerical convolution
    `MaternKernel.v` already uses, including its `breakpoints` trick for
    the node where the kernel's own kink/peak (r_i) falls inside the
    truncated integration range.

    This is the ground-truth, exact-per-point version -- expensive (~7ms
    per scalar `r`, measured directly against a real checkpoint's typical
    neighbor counts), used to build `_axis_cross_moment`'s interpolation
    table below and directly inside `_axis_prior_variance` (called only
    once per kernel family per checkpoint, so its cost there is
    negligible), not called per query/per neighbor.
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    if isinstance(kernel, RBFKernel):
        return norm.pdf(r, loc=0.0, scale=np.sqrt(kernel.sigma**2 + s**2))

    lo, hi = -n_sigma * s, n_sigma * s
    out = np.empty_like(r)
    for i, ri in enumerate(r):
        def integrand(u, ri=ri):
            return float(kernel.k(ri, u)) * float(norm.pdf(u, loc=0.0, scale=s))
        breakpoints = [ri] if lo < ri < hi else None
        out[i], _ = integrate.quad(integrand, lo, hi, points=breakpoints)
    return out


def _axis_cross_moment(kernel: Kernel, r: np.ndarray, s: float, r_max: float, n_grid: int = 300) -> np.ndarray:
    """Fast, cached version of `_axis_cross_moment_exact` for the per-
    query/per-neighbor path: `generalized_rendering_aware_variance` calls
    this once per neighbor per axis per query point, and a real checkpoint
    query can have dozens of neighbors -- at ~7ms per exact scalar
    evaluation that's the actual bottleneck (confirmed directly: ~140ms per
    query point with the exact version, dominated entirely by these calls).

    Every neighbor's per-axis offset `r` from a query built via
    `engine.local_neighbors(query_point, radius)` is bounded by
    `radius` (`local_neighbors` is itself a ball query of that radius, and
    each axis's offset can't exceed the Euclidean distance), and
    `_axis_cross_moment_exact` is an even function of `r` (kernel.k depends
    only on |x - y|) -- so one dense grid of exact evaluations over
    `[0, r_max]`, built once per (kernel family, param, s) and reused for
    every query at that radius, replaces a fresh quad call per neighbor.
    Linear interpolation between grid points; `n_grid=300` checked directly
    against the exact version to agree to ~1e-6 relative error, far below
    the noise floor of the correlation metrics this feeds into.

    RBFKernel skips the table entirely -- its exact closed form is already
    as fast as an interpolation lookup.
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    if isinstance(kernel, RBFKernel):
        return _axis_cross_moment_exact(kernel, r, s)

    key = _kernel_cache_key(kernel) + (s, r_max)
    table = _CONV_TABLE_CACHE.get(key)
    if table is None:
        rs = np.linspace(0.0, r_max, n_grid)
        vals = _axis_cross_moment_exact(kernel, rs, s)
        table = (rs, vals)
        _CONV_TABLE_CACHE[key] = table
    rs, vals = table
    return np.interp(np.abs(r), rs, vals)


def _axis_prior_variance(kernel: Kernel, s: float, n_sigma: float = 8.0) -> float:
    """integral integral kernel.k(t, t') * N(t; 0, s^2) * N(t'; 0, s^2) dt dt' --
    the double-integral analogue of `_axis_cross_moment`, translation-
    invariant (independent of any query point), so cached once per
    (kernel family, param, s) rather than recomputed per query. Uses the
    exact (not table-interpolated) cross moment: this integrates `t` over
    the full `+-n_sigma*s` range (up to `4*radius`, since `s=radius/2`),
    well past `_axis_cross_moment`'s table range of `[0, radius]` (which
    only ever needs to cover real local-neighbor offsets) -- but this
    function is only ever called once per (kernel, s) pair per checkpoint,
    not per query, so the exact version's cost here is negligible.
    """
    cache_key = _kernel_cache_key(kernel) + (s,)
    if cache_key in _Z0_CACHE:
        return _Z0_CACHE[cache_key]

    if isinstance(kernel, RBFKernel):
        val = float(norm.pdf(0.0, loc=0.0, scale=np.sqrt(2 * s**2 + kernel.sigma**2)))
    else:
        lo, hi = -n_sigma * s, n_sigma * s

        def integrand(t):
            cm = _axis_cross_moment_exact(kernel, np.array([t]), s, n_sigma=n_sigma)[0]
            return cm * float(norm.pdf(t, loc=0.0, scale=s))

        val, _ = integrate.quad(integrand, lo, hi)
        val = float(val)

    _Z0_CACHE[cache_key] = val
    return val


def _generalized_rendering_aware_moments(
    engine: LocalUncertaintyEngine, kernels_per_axis: List[Kernel], query_point: np.ndarray, radius: float,
    exclude_idx: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Shared computation for `generalized_rendering_aware_variance` and
    `sparsity_correlation`'s amplitude-normalized ratio check: returns
    `(mean, variance, z0)` -- `z0` (the prior variance, before conditioning
    on any real data) is exposed because it scales with the local mean
    opacity `amplitude` the exact same way `variance` does (both `z0` and
    the moment vector `z` are homogeneous in `amplitude`: `z0 ~ amplitude^2`,
    `z ~ amplitude`, and `kxx` doesn't depend on `amplitude` at all), so
    `variance / z0` is an amplitude-*independent* ratio -- exactly the same
    "normalize out the scale-dominant factor for cross-condition
    comparability" trick `gs_experiment.scripts.render_reconstruction.
    compute_uncertainty_maps` already uses for its `directional_map`
    (variance / prior_variance), for the identical reason (see that
    function's docstring: raw variance's magnitude is dominated by
    whatever amplitude/bandwidth happens to apply at one query, not
    comparable across queries with different local opacity on its own).
    """
    query_point = np.asarray(query_point, dtype=float)
    idx = engine.local_neighbors(query_point, radius, exclude_idx=exclude_idx)
    s = radius / 2.0
    amplitude = float(np.mean(engine.opacities[idx])) if len(idx) > 0 else float(np.mean(engine.opacities))

    z0 = amplitude**2
    for kernel in kernels_per_axis:
        z0 *= _axis_prior_variance(kernel, s)

    n = len(idx)
    if n == 0:
        return 0.0, max(z0, 0.0), z0

    local_positions = engine.positions[idx]
    local_values = engine.values[idx]
    diffs = local_positions - query_point

    z = np.ones(n)
    for d, kernel in enumerate(kernels_per_axis):
        z *= _axis_cross_moment(kernel, diffs[:, d], s, r_max=radius)
    z *= amplitude

    pos_kernel = ProductKernel(kernels_per_axis)
    kxx = pos_kernel.k(local_positions, local_positions)
    jitter = 1e-4 * np.mean(np.diag(kxx))
    kxx = kxx + jitter * np.eye(n)

    mean, variance = _posterior_mean_variance(kxx, local_values, z, z0)
    return mean, variance, z0


def generalized_rendering_aware_variance(
    engine: LocalUncertaintyEngine, kernels_per_axis: List[Kernel], query_point: np.ndarray, radius: float,
    exclude_idx: Optional[int] = None,
) -> BQResult:
    """`LocalUncertaintyEngine.rendering_aware_variance`, generalized to
    any 1D `Kernel` family per axis (not just RBF) -- same local-neighbor
    gathering (`engine.local_neighbors`) and the same
    `GaussianRenderWeight.from_total_mass`-style isotropic box weight
    (`covariance = (radius/2)^2 * I`, amplitude pinned to the local mean
    opacity), but built via `_axis_cross_moment`/`_axis_prior_variance`
    instead of the RBF-only closed form, so it works for Matern/
    RationalQuadratic too -- see this module's docstring for why that's
    exact here (isotropic, axis-aligned weight; the along-ray/gsplat
    variants' non-diagonal moment-matched weights are NOT covered by this
    function).
    """
    mean, variance, _ = _generalized_rendering_aware_moments(engine, kernels_per_axis, query_point, radius, exclude_idx)
    return BQResult(mean=mean, variance=variance)


def make_kernels_per_axis(family: str, param: float) -> List[Kernel]:
    spec = FAMILIES[family]
    return list(spec["factory"](param).kernels_per_axis)


# ---------------------------------------------------------------------------
# 3. Sparsity correlation.
# ---------------------------------------------------------------------------


def sparsity_correlation(
    engine: LocalUncertaintyEngine, kernels_per_axis: List[Kernel], radius: float, query_indices: np.ndarray,
    knn_k: int = 8,
) -> dict:
    """For each of `query_indices` (indices into `engine.positions`),
    computes this family's posterior variance at that splat's own position
    (excluding itself as a neighbor, so it isn't trivially "observed") and
    a local-sparsity measure (distance to its `knn_k`-th nearest neighbor --
    larger means sparser). Reports Pearson/Spearman correlation between the
    two: a working signal should show variance *increasing* with knn
    distance (denser -> smaller knn distance -> lower variance), i.e. a
    positive correlation here -- reported as measured, not assumed.

    Also reports the same two correlations against `variance / z0` (the
    amplitude-normalized ratio -- see `_generalized_rendering_aware_moments`'s
    docstring): raw variance is gated by each window's local mean opacity
    (`amplitude`) as well as by sparsity (confirmed directly on the `wide`
    checkpoint: knn distance anti-correlates with local mean opacity at
    r=-0.49, and opacity correlates with raw variance at r=+0.49 -- a real
    confound, not noise), so the ratio is reported alongside the raw
    correlation as a deconfounded second read on the same question, not a
    replacement for it.
    """
    dists, _ = engine.tree.query(engine.positions[query_indices], k=knn_k + 1)  # +1: includes the point itself at dist 0
    knn_dist = dists[:, -1]

    variances = np.empty(len(query_indices))
    ratios = np.empty(len(query_indices))
    for i, idx in enumerate(query_indices):
        mean, variance, z0 = _generalized_rendering_aware_moments(
            engine, kernels_per_axis, engine.positions[idx], radius, exclude_idx=int(idx)
        )
        variances[i] = variance
        ratios[i] = variance / max(z0, 1e-300)

    pearson_r, pearson_p = pearsonr(knn_dist, variances)
    spearman_r, spearman_p = spearmanr(knn_dist, variances)
    ratio_pearson_r, ratio_pearson_p = pearsonr(knn_dist, ratios)
    ratio_spearman_r, ratio_spearman_p = spearmanr(knn_dist, ratios)
    return {
        "n": len(query_indices),
        "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r), "spearman_p": float(spearman_p),
        "ratio_pearson_r": float(ratio_pearson_r), "ratio_pearson_p": float(ratio_pearson_p),
        "ratio_spearman_r": float(ratio_spearman_r), "ratio_spearman_p": float(ratio_spearman_p),
    }


# ---------------------------------------------------------------------------
# 4. Calibration against real held-out rendering error.
# ---------------------------------------------------------------------------


def _render_and_unproject(checkpoint_dir: Path, eval_dir: Path, view_indices, depth_width=112, depth_height=42, device="cuda"):
    """Real GT vs. reconstruction (full res) plus real depth-unprojected
    world points at a lower resolution (matching
    scripts.render_reconstruction.compute_uncertainty_maps's own depth_width/
    depth_height convention), for each of `view_indices` in `eval_dir` (that
    scene's own held-out view set).

    Returns a list of dicts: gt (H,W,3), recon (H,W,3), world_points
    (depth_height, depth_width, 3), valid (depth_height, depth_width) bool.
    """
    import gsplat
    import torch

    from gs_experiment.nerf_transforms import fov_x_to_intrinsics, load_transforms, opencv_viewmat_from_c2w
    from gs_experiment.ply_io import read_3dgs_ply
    from gs_experiment.scripts.render_reconstruction import render_views, unproject_depth_grid

    results, checkpoint = render_views(str(eval_dir), view_indices, checkpoint_dir=str(checkpoint_dir), device=device)
    camera_angle_x, frames = load_transforms(str(eval_dir / "transforms.json"))

    K_full = fov_x_to_intrinsics(camera_angle_x, results[0][1].shape[1], results[0][1].shape[0])
    K_depth = K_full.copy()
    width, height = results[0][1].shape[1], results[0][1].shape[0]
    K_depth[0, 0] *= depth_width / width
    K_depth[0, 2] *= depth_width / width
    K_depth[1, 1] *= depth_height / height
    K_depth[1, 2] *= depth_height / height
    K_depth_t = torch.tensor(K_depth, dtype=torch.float32, device=device)

    all_positions = torch.tensor(checkpoint["positions"], dtype=torch.float32, device=device)
    all_scales = torch.tensor(checkpoint["scales"], dtype=torch.float32, device=device)
    all_rotations = torch.tensor(checkpoint["rotations"], dtype=torch.float32, device=device)
    all_opacities = torch.tensor(checkpoint["opacities"], dtype=torch.float32, device=device)
    all_sh = torch.tensor(checkpoint["sh_coeffs"], dtype=torch.float32, device=device).transpose(1, 2)
    sh_degree = checkpoint["sh_degree"]

    out = []
    with torch.no_grad():
        for view_i, gt, recon in results:
            _, c2w = frames[view_i]
            viewmat_np = opencv_viewmat_from_c2w(c2w)
            viewmat = torch.tensor(viewmat_np, dtype=torch.float32, device=device)
            c2w_cv = np.linalg.inv(viewmat_np)

            rendered_lo, alpha_lo, _ = gsplat.rasterization(
                all_positions, all_rotations, all_scales, all_opacities, all_sh, viewmat[None], K_depth_t[None],
                width=depth_width, height=depth_height, sh_degree=sh_degree, render_mode="ED",
            )
            depth_map = rendered_lo[0, ..., 0].cpu().numpy()
            alpha_map = alpha_lo[0, ..., 0].cpu().numpy()
            valid = alpha_map > 0.5
            world_points = unproject_depth_grid(depth_map, K_depth, c2w_cv)
            out.append({"gt": gt, "recon": recon, "world_points": world_points, "valid": valid})
    return out


def _downsample_squared_error(gt: np.ndarray, recon: np.ndarray, depth_width: int, depth_height: int) -> np.ndarray:
    """Average-pools per-pixel squared RGB error (mean over channels) from
    full resolution down to (depth_height, depth_width), to align 1:1 with
    the low-res depth-unprojected world points `_render_and_unproject`
    returns -- the same macropixel grid, so each world point gets the
    error of the same image region its depth/alpha were rendered from."""
    height, width, _ = gt.shape
    se_full = ((gt - recon) ** 2).mean(axis=-1)
    ys = (np.arange(depth_height) * height / depth_height).astype(int).clip(0, height - 1)
    xs = (np.arange(depth_width) * width / depth_width).astype(int).clip(0, width - 1)
    # Nearest-neighbor downsample (matching compute_uncertainty_maps's own NEAREST
    # resize for its validity mask) rather than a box filter -- simpler, and the
    # depth/alpha map is already rendered at exactly this resolution, so this
    # keeps the error sample co-located with the same low-res pixel center.
    return se_full[np.ix_(ys, xs)]


def calibration_metrics(
    engine: LocalUncertaintyEngine, kernels_per_axis: List[Kernel], radius: float, checkpoint_dir: Path, eval_dir: Path,
    view_indices, max_points_per_view: int = 80, depth_width: int = 112, depth_height: int = 42, seed: int = 0,
) -> dict:
    """Real held-out calibration: for each held-out eval view, gets GT vs.
    reconstruction + depth-unprojected world points (`_render_and_unproject`),
    subsamples up to `max_points_per_view` valid points, queries this
    family's posterior variance at each (`generalized_rendering_aware_variance`,
    no exclude_idx -- these are new query points, not existing splats), and
    pairs it with that macropixel's real squared rendering error. Reports a
    calibration correlation (does higher variance track higher error) and a
    Gaussian-NLL-style score, 0.5*(error^2/var + log(var)) averaged over all
    points (lower is better) -- the standard proper-scoring-rule form for a
    predicted Gaussian variance against a real squared residual.
    """
    rng = np.random.default_rng(seed)
    rendered = _render_and_unproject(checkpoint_dir, eval_dir, view_indices, depth_width=depth_width, depth_height=depth_height)

    all_variance = []
    all_squared_error = []
    for view in rendered:
        se = _downsample_squared_error(view["gt"], view["recon"], depth_width, depth_height)
        ys, xs = np.where(view["valid"])
        if len(ys) == 0:
            continue
        if len(ys) > max_points_per_view:
            keep = rng.choice(len(ys), size=max_points_per_view, replace=False)
            ys, xs = ys[keep], xs[keep]
        for y, x in zip(ys, xs):
            point = view["world_points"][y, x]
            result = generalized_rendering_aware_variance(engine, kernels_per_axis, point, radius)
            all_variance.append(result.variance)
            all_squared_error.append(se[y, x])

    variance = np.asarray(all_variance)
    squared_error = np.asarray(all_squared_error)
    variance_floored = np.maximum(variance, 1e-12)

    pearson_r, pearson_p = pearsonr(variance, squared_error)
    spearman_r, spearman_p = spearmanr(variance, squared_error)
    nll = 0.5 * (squared_error / variance_floored + np.log(variance_floored))
    return {
        "n": len(variance),
        "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r), "spearman_p": float(spearman_p),
        "mean_nll": float(np.mean(nll)),
    }


# ---------------------------------------------------------------------------
# 5. Driver.
# ---------------------------------------------------------------------------


def run_checkpoint(name: str, checkpoint_dir: Path, eval_dir: Path, window_radius: float, seed: int = 0) -> dict:
    from gs_experiment.nerf_transforms import load_transforms
    from gs_experiment.splat_scene import load_from_gsplat_checkpoint

    print(f"\n=== checkpoint: {name} ({checkpoint_dir}) ===")
    scene = load_from_gsplat_checkpoint(
        str(checkpoint_dir), use_gpu_attribution=True, attribution_min_opacity=0.1,
    )
    print(f"  {len(scene.positions)} splats, {int((scene.opacities > 0.1).sum())} above opacity 0.1")

    fitted = fit_all_families(scene, window_radius=window_radius, seed=seed)
    for family, param in fitted.items():
        print(f"  fitted {family}: {FAMILIES[family]['param_name']} = {param:.5f}")

    engine = LocalUncertaintyEngine(
        positions=scene.positions, values=scene.colors, pos_kernel=ProductKernel([RBFKernel(sigma=fitted["rbf"])] * 3),
        scene_bounds=tuple((scene.positions[:, d].min(), scene.positions[:, d].max()) for d in range(3)),
        opacities=scene.opacities, max_neighbors=60, seed=seed,
    )

    keep_idx = np.nonzero(scene.opacities > 0.1)[0]
    rng = np.random.default_rng(seed)
    n_sparsity_queries = min(150, len(keep_idx))
    query_idx = rng.choice(keep_idx, size=n_sparsity_queries, replace=False)

    # held-out view count read from this scene's own eval/transforms.json
    # (confirmed 30 for every scene in the 7-scene set, but read it directly
    # rather than hardcoding, since this now runs across scenes).
    _, eval_frames = load_transforms(str(eval_dir / "transforms.json"))
    n_frames = len(eval_frames)
    view_indices = list(range(0, n_frames, max(1, n_frames // 6)))[:6]

    results = {}
    for family, param in fitted.items():
        kernels_per_axis = make_kernels_per_axis(family, param)
        sparsity = sparsity_correlation(engine, kernels_per_axis, window_radius, query_idx)
        calibration = calibration_metrics(
            engine, kernels_per_axis, window_radius, checkpoint_dir, eval_dir, view_indices, seed=seed
        )
        results[family] = {"param": param, "sparsity": sparsity, "calibration": calibration}
        print(
            f"  [{family:20s}] sparsity: raw_var pearson={sparsity['pearson_r']:+.3f} spearman={sparsity['spearman_r']:+.3f}"
            f"  ratio pearson={sparsity['ratio_pearson_r']:+.3f} spearman={sparsity['ratio_spearman_r']:+.3f} "
            f"(n={sparsity['n']})  |  calibration: pearson={calibration['pearson_r']:+.3f} "
            f"spearman={calibration['spearman_r']:+.3f} mean_nll={calibration['mean_nll']:.3f} (n={calibration['n']})"
        )
    return results


def print_summary_table(all_results: dict):
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(
        "(sparsity correlation is between knn-distance -- larger = sparser -- and posterior variance;"
        " a working signal is POSITIVE: sparser -> higher variance. 'ratio' = variance/prior_variance,"
        " amplitude-normalized -- see sparsity_correlation's docstring.)"
    )
    header = (
        f"{'checkpoint':12s} {'family':20s} {'param':>10s} {'spars.var r':>11s} {'spars.ratio r':>13s} "
        f"{'calib r':>9s} {'calib rho':>10s} {'mean NLL':>9s}"
    )
    print(header)
    for checkpoint_name, results in all_results.items():
        for family, r in results.items():
            print(
                f"{checkpoint_name:12s} {family:20s} {r['param']:10.5f} "
                f"{r['sparsity']['pearson_r']:11.3f} {r['sparsity']['ratio_pearson_r']:13.3f} "
                f"{r['calibration']['pearson_r']:9.3f} {r['calibration']['spearman_r']:10.3f} "
                f"{r['calibration']['mean_nll']:9.3f}"
            )
    print("=" * 100)

    print("\nPer-checkpoint winners:")
    for checkpoint_name, results in all_results.items():
        # "best sparsity" = closest to the intended positive-correlation
        # direction (sparser -> higher variance) and strongest in that
        # direction -- i.e. the largest signed pearson_r, not the largest
        # magnitude regardless of sign.
        best_sparsity = max(results, key=lambda f: results[f]["sparsity"]["pearson_r"])
        best_sparsity_ratio = max(results, key=lambda f: results[f]["sparsity"]["ratio_pearson_r"])
        best_calib_corr = max(results, key=lambda f: results[f]["calibration"]["pearson_r"])
        best_nll = min(results, key=lambda f: results[f]["calibration"]["mean_nll"])
        print(
            f"  {checkpoint_name}: best sparsity signal (raw variance) = {best_sparsity} "
            f"(r={results[best_sparsity]['sparsity']['pearson_r']:+.3f}); "
            f"best sparsity signal (ratio) = {best_sparsity_ratio} "
            f"(r={results[best_sparsity_ratio]['sparsity']['ratio_pearson_r']:+.3f}); "
            f"best calibration correlation = {best_calib_corr} "
            f"(r={results[best_calib_corr]['calibration']['pearson_r']:+.3f}); "
            f"best (lowest) NLL = {best_nll} ({results[best_nll]['calibration']['mean_nll']:.3f})"
        )


RESULTS_JSON = REPO_ROOT / "gs_experiment" / "results" / "kernel_family_ablation_results.json"


def main(scenes: Optional[List[str]] = None, save_json: Path = RESULTS_JSON):
    import json

    scenes = scenes or SCENES
    all_results: Dict[str, Dict[str, dict]] = {}
    for scene in scenes:
        print("\n" + "#" * 100)
        print(f"# scene: {scene}")
        print("#" * 100)
        scene_results = {}
        for checkpoint_name, spec in CHECKPOINTS[scene].items():
            scene_results[checkpoint_name] = run_checkpoint(
                checkpoint_name, spec["dir"], EVAL_DIRS[scene], spec["window_radius"]
            )
        print_summary_table(scene_results)
        all_results[scene] = scene_results

        # Save incrementally after every scene (not just at the end) so a
        # crash/interrupt partway through the 7-scene sweep doesn't lose
        # already-computed scenes.
        if save_json is not None:
            save_json.parent.mkdir(parents=True, exist_ok=True)
            with open(save_json, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\nSaved results so far ({len(all_results)}/{len(scenes)} scenes) to {save_json}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenes", nargs="+", default=None, choices=SCENES,
        help="Subset of scenes to run (default: all 7 -- chair, drums, ficus, hotdog, lego, mic, ship).",
    )
    args = parser.parse_args()
    main(scenes=args.scenes)
