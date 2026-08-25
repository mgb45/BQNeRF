"""Minimal from-scratch 3D Gaussian Splatting trainer built directly on
gsplat's CUDA rasterizer. Produces real trained-checkpoint .ply files that
gs_experiment.splat_scene.load_from_gsplat_checkpoint can be validated
against, and that the differentiation experiment (ROADMAP.md milestone 2)
can be run on.

Deliberately not gsplat's own examples/simple_trainer.py, which needs a
heavier example-only dependency set (nerfview, viser, fused-ssim) not
needed here. This is the minimum viable trainer for this project's actual
purpose -- a real checkpoint plus real camera poses to run the BQ/
visibility comparison on -- not a quality- or performance-competitive
reimplementation of 3DGS training.

Densification/pruning (`densify=True`, off by default) is real, not a
placeholder: standard gradient-triggered clone/split plus opacity-based
pruning (`densify_and_prune`), against gsplat's own view-space positional
gradient (`meta["means2d"].grad`) -- the same signal the reference
implementation uses. This matters beyond final quality: ROADMAP.md's core
differentiation claim needs splat *density* to depend on view coverage
(a well-observed region should accumulate more splats than a poorly-
observed one with identical geometry), which a fixed splat count from
random initialization structurally cannot provide -- see
gs_experiment/results/FINDINGS.md sections 6-7 for the empirical trail
that motivated adding this. With `densify=False`, splat count stays fixed
at initialization; a small opacity-sparsity regularizer
(`opacity_reg_weight`) substitutes for pruning in that mode: without it,
splats that never earn photometric gradient (most of a randomly-
initialized population that's larger than the scene needs) sit inert at
their initial opacity forever and contaminate downstream local-
neighborhood statistics -- see the comment at the loss computation for
the empirical motivation. Both mechanisms can run together when
densifying (regularizer discourages inert splats between densify cycles,
pruning removes them outright at each cycle).

Needs torch + gsplat (requirements-gsplat.txt); not imported by anything
that runs on requirements.txt alone.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import torch
from PIL import Image

from gs_experiment.nerf_transforms import fov_x_to_intrinsics, load_transforms, opencv_viewmat_from_c2w
from gs_experiment.pixel_uncertainty import LocalUncertaintyEngine, make_default_3d_position_kernel
from gs_experiment.ply_io import write_3dgs_ply
from gs_experiment.spherical_harmonics import N_COEFFS_FOR_DEGREE


def load_dataset(scene_dir: str, device: str):
    """Returns (images (T,H,W,3) float32 in [0,1] torch tensor on `device`,
    viewmats (T,4,4), Ks (T,3,3), width, height)."""
    camera_angle_x, frames = load_transforms(os.path.join(scene_dir, "transforms.json"))

    images = []
    viewmats = []
    for file_path, c2w in frames:
        img_path = os.path.join(scene_dir, file_path + ".png")
        img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        images.append(img)
        viewmats.append(opencv_viewmat_from_c2w(c2w))

    height, width = images[0].shape[:2]
    K = fov_x_to_intrinsics(camera_angle_x, width, height)

    images_t = torch.tensor(np.stack(images), dtype=torch.float32, device=device)
    viewmats_t = torch.tensor(np.stack(viewmats), dtype=torch.float32, device=device)
    Ks_t = torch.tensor(np.stack([K] * len(frames)), dtype=torch.float32, device=device)
    return images_t, viewmats_t, Ks_t, width, height


def init_splats(n_splats: int, bounds, sh_degree: int, device: str, rng: np.random.Generator, init_scale: Optional[float] = None):
    """`init_scale`, if given, sets every splat's initial (isotropic)
    scale directly -- pass the actual expected feature size (e.g. a thin
    structure's radius), not something derived from `bounds`. Deriving it
    from the scene's bounding-box extent (the original approach here)
    silently assumes objects fill a roughly constant fraction of that
    whole volume; for a scene with two small, widely-separated
    fine-detail clusters inside a much larger empty bounding volume, that
    assumption is badly wrong (measured: it produced an initial scale
    ~25x the actual rod radius, and 6000 iterations of single-random-view
    SGD wasn't enough to shrink it -- splats stayed huge, low-opacity
    blobs that minimized photometric loss by matching background color
    rather than resolving any real geometry, despite reporting a
    deceptively reasonable-looking PSNR since background dominates the
    frame). Defaults to the old bounds-derived heuristic only when
    `init_scale` is omitted, for scenes where that assumption does hold
    (e.g. quick_validation_scene's objects, which fill a large fraction
    of their own bounds).
    """
    (x0, x1), (y0, y1), (z0, z1) = bounds
    positions = np.stack(
        [rng.uniform(x0, x1, n_splats), rng.uniform(y0, y1, n_splats), rng.uniform(z0, z1, n_splats)], axis=1
    )
    if init_scale is None:
        scene_extent = max(x1 - x0, y1 - y0, z1 - z0)
        init_scale = scene_extent / (n_splats ** (1.0 / 3.0) + 1e-6)
    log_scales = np.log(np.full((n_splats, 3), init_scale))
    quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_splats, 1)) + rng.normal(scale=0.01, size=(n_splats, 4))
    opacity_logits = np.full(n_splats, -1.0)  # sigmoid(-1) ~= 0.27, a modest start

    n_coeffs = N_COEFFS_FOR_DEGREE[sh_degree]
    sh = np.zeros((n_splats, 3, n_coeffs))
    sh[:, :, 0] = rng.uniform(-0.5, 0.5, size=(n_splats, 3))  # around the eval_sh "+0.5" baseline

    def as_param(arr):
        return torch.nn.Parameter(torch.tensor(arr, dtype=torch.float32, device=device))

    return dict(
        positions=as_param(positions),
        log_scales=as_param(log_scales),
        quats=as_param(quats),
        opacity_logits=as_param(opacity_logits),
        sh=as_param(sh),
    )


LR_BY_NAME = {"positions": 2e-3, "log_scales": 5e-3, "quats": 1e-3, "opacity_logits": 5e-2, "sh": 2.5e-3}


def make_optimizer(params: dict) -> torch.optim.Optimizer:
    return torch.optim.Adam([{"params": [params[name]], "lr": lr, "name": name} for name, lr in LR_BY_NAME.items()])


def _splat_positions_and_colors(params: dict):
    """Detached numpy snapshot of current splat state, in the (positions,
    scalar-color) form LocalUncertaintyEngine needs -- shared by both
    ROADMAP.md item 3 mechanisms below (the NLL loss term and
    variance-driven densification), since both need to build a fresh
    engine from whatever the splat population looks like *right now*
    (positions and count both change at every densify cycle)."""
    positions_np = params["positions"].detach().cpu().numpy()
    colors_np = params["sh"].detach().cpu().numpy()[:, :, 0].mean(axis=1)
    return positions_np, colors_np


def _build_uncertainty_engine(params: dict, sigma: float, max_neighbors: int) -> LocalUncertaintyEngine:
    positions_np, colors_np = _splat_positions_and_colors(params)
    bounds = tuple((positions_np[:, d].min() - 0.3, positions_np[:, d].max() + 0.3) for d in range(3))
    pos_kernel = make_default_3d_position_kernel(sigma=sigma)
    return LocalUncertaintyEngine(
        positions=positions_np, values=colors_np, pos_kernel=pos_kernel, scene_bounds=bounds, max_neighbors=max_neighbors,
    )


def compute_per_splat_bq_variance(params: dict, sigma: float, window_radius: float, max_neighbors: int, device: str) -> torch.Tensor:
    """BQ position-only variance at every current splat's own position --
    the closed-form, "uncertainty-driven" analogue of the standard 3DGS
    view-space-gradient densification signal (`avg_grad` in `train`):
    where the gradient signal asks "does the optimizer keep wanting to
    move this splat," this asks "is this splat sitting in a
    poorly-covered region of the scene," directly from the same BQ
    machinery this project's uncertainty claims are built on, not a proxy
    for it. Pure numpy/scipy (LocalUncertaintyEngine), so this detaches
    from the training graph entirely -- used only to pick *which* splats
    to split/clone, not backpropagated through.
    """
    positions_np, _ = _splat_positions_and_colors(params)
    engine = _build_uncertainty_engine(params, sigma, max_neighbors)
    variances = np.array([engine.spatial_only_variance(p, window_radius).variance for p in positions_np])
    return torch.tensor(variances, dtype=torch.float32, device=device)


def unproject_depth_grid(depth: np.ndarray, K: np.ndarray, c2w_cv: np.ndarray) -> np.ndarray:
    """Same construction as gs_experiment/render_sweep_gif.py's function of
    the same name (not imported from there to avoid a matplotlib/PIL
    import chain inside the training hot path): depth (H, W) in OpenCV
    camera space -> (H, W, 3) world-space points, the real ray-surface hit
    for every pixel of a low-res depth pass."""
    h, w = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    us, vs = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    x_cam = (us - cx) / fx * depth
    y_cam = (vs - cy) / fy * depth
    cam_points = np.stack([x_cam, y_cam, depth, np.ones_like(depth)], axis=-1)
    world_points = cam_points @ c2w_cv.T
    return world_points[..., :3]


def compute_nll_loss_term(
    params: dict,
    pred: torch.Tensor,
    gt: torch.Tensor,
    viewmat_np: np.ndarray,
    K_full_np: np.ndarray,
    width: int,
    height: int,
    sh_degree: int,
    device: str,
    grid_res: int,
    sigma: float,
    window_radius: float,
    max_neighbors: int,
    alpha_threshold: float,
    variance_floor: float,
) -> Optional[torch.Tensor]:
    """Gaussian-NLL loss at a sparse grid of real ray-surface points,
    weighted by closed-form BQ position-only variance -- ROADMAP.md item
    3's "training under the likelihood," scoped honestly (see the
    docstring on `train`'s `nll_weight` argument for exactly what this
    does and does not do).

    Returns None if the current view has no valid (alpha > threshold)
    points on the query grid (e.g. very early in training, before any
    splat has opacity/coverage at this view) -- callers should skip adding
    the term for that iteration rather than treat this as an error.
    """
    import gsplat

    with torch.no_grad():
        quats = params["quats"]
        scales = torch.exp(params["log_scales"])
        opacities = torch.sigmoid(params["opacity_logits"])
        sh_for_gsplat = params["sh"].transpose(1, 2)

        K_grid_np = K_full_np.copy()
        K_grid_np[0, 0] *= grid_res / width
        K_grid_np[0, 2] *= grid_res / width
        K_grid_np[1, 1] *= grid_res / height
        K_grid_np[1, 2] *= grid_res / height
        K_grid = torch.tensor(K_grid_np, dtype=torch.float32, device=device)
        viewmat_t = torch.tensor(viewmat_np, dtype=torch.float32, device=device)

        rendered_lo, alpha_lo, _ = gsplat.rasterization(
            params["positions"], quats, scales, opacities, sh_for_gsplat,
            viewmat_t[None], K_grid[None], width=grid_res, height=grid_res, sh_degree=sh_degree, render_mode="ED",
        )
        depth_map = rendered_lo[0, ..., 0].cpu().numpy()
        alpha_map = alpha_lo[0, ..., 0].cpu().numpy()
        valid = alpha_map > alpha_threshold
        if not valid.any():
            return None

        c2w_cv = np.linalg.inv(viewmat_np)
        world_points = unproject_depth_grid(depth_map, K_grid_np, c2w_cv)

        engine = _build_uncertainty_engine(params, sigma, max_neighbors)
        ys, xs = np.where(valid)
        variances = np.array([engine.spatial_only_variance(world_points[y, x], window_radius).variance for y, x in zip(ys, xs)])

        rows = np.clip(((ys.astype(np.float64) + 0.5) * height / grid_res).astype(int), 0, height - 1)
        cols = np.clip(((xs.astype(np.float64) + 0.5) * width / grid_res).astype(int), 0, width - 1)

    variance_t = torch.tensor(variances, dtype=torch.float32, device=device).clamp_min(variance_floor)
    pred_at_pts = pred[rows, cols]
    gt_at_pts = gt[rows, cols]
    sq_err = ((pred_at_pts - gt_at_pts) ** 2).mean(dim=-1)
    return 0.5 * (sq_err / variance_t + torch.log(variance_t)).mean()


def densify_and_prune(
    params: dict,
    avg_grad: torch.Tensor,
    grad_threshold: float,
    min_opacity: float,
    split_scale_threshold: float,
    device: str,
    max_splats: int,
) -> dict:
    """Standard 3DGS adaptive density control (Kerbl et al. 2023), against
    view-space positional gradient rather than reimplementing it from
    scratch differently: splats with a large accumulated view-space
    gradient (meaning the optimizer keeps wanting to move them --
    evidence the region around them is under-reconstructed) get either
    SPLIT (if already large -- replaced by 2 smaller children sampled from
    the original's own extent) or CLONED (if small -- duplicated as-is,
    letting gradient descent separate the copies over subsequent
    iterations) depending on current scale, matching the reference
    implementation's own split-vs-clone criterion. Splats below
    `min_opacity` are pruned outright. This is the mechanism ROADMAP.md's
    differentiation claim actually needs and gs_experiment/results/
    FINDINGS.md sections 6-7 found missing: splat *density* becoming
    view-coverage-dependent, not just splat *quality*.

    Clones are offset from their parent by a small (0.5x scale) random
    jitter rather than starting exactly coincident: the reference
    implementation nudges clones along the triggering gradient direction,
    which needs backprojecting the 2D view-space gradient into 3D (not
    done here, simpler to jitter directly in world space); an earlier,
    unoffset version of this function measurably distorted downstream BQ
    variance (many clone pairs still <0.001 apart after thousands of
    iterations at a kernel bandwidth two orders of magnitude larger --
    see gs_experiment/results/FINDINGS.md), which the offset fixes.
    `max_splats` caps
    growth as a hard safety net -- gradient-triggered splitting can
    compound quickly if `grad_threshold` is set too low, and this project
    already has one real incident (FINDINGS.md section 5) from an
    unbounded-growth computation left to run unattended.
    """
    scales = torch.exp(params["log_scales"])
    opacities = torch.sigmoid(params["opacity_logits"])
    max_scale = scales.max(dim=1).values

    should_prune = opacities < min_opacity
    room_left = max(0, max_splats - int((~should_prune).sum().item()))

    should_densify = (avg_grad >= grad_threshold) & ~should_prune
    if room_left <= 0:
        should_densify &= False
    should_split = should_densify & (max_scale > split_scale_threshold)
    should_clone = should_densify & ~should_split

    # if densification would exceed max_splats, keep only the
    # highest-gradient candidates that fit in the remaining budget
    n_requested = int(should_split.sum().item()) + int(should_clone.sum().item())
    if n_requested > room_left:
        candidate_idx = (should_split | should_clone).nonzero(as_tuple=True)[0]
        ranked = candidate_idx[torch.argsort(avg_grad[candidate_idx], descending=True)]
        drop = ranked[room_left:]
        should_split[drop] = False
        should_clone[drop] = False

    keep_mask = ~should_prune & ~should_split

    def gather(name, idx=None):
        t = params[name].detach()
        return t[idx] if idx is not None else t[keep_mask]

    new_positions = [gather("positions")]
    new_log_scales = [gather("log_scales")]
    new_quats = [gather("quats")]
    new_opacity_logits = [gather("opacity_logits")]
    new_sh = [gather("sh")]

    clone_idx = should_clone.nonzero(as_tuple=True)[0]
    if len(clone_idx) > 0:
        # offset the clone, not the parent, by a small fraction of the
        # splat's own scale -- matches the reference implementation's
        # intent (parent and clone should separate, not sit exactly on
        # top of each other receiving near-identical gradients) without
        # needing to backproject the 2D view-space gradient that
        # triggered densification into a 3D direction. Not cosmetic:
        # exact-duplicate positions from an *unoffset* clone measurably
        # distorted downstream BQ variance in this project's first
        # densification run (many clone pairs still <0.001 apart after
        # thousands of iterations, on a kernel bandwidth two orders of
        # magnitude larger) -- caught and fixed here rather than reported
        # as a "well-observed regions have higher uncertainty" finding
        # before ruling out that it was actually a clone-adjacency
        # artifact.
        clone_offset = torch.randn(len(clone_idx), 3, device=device) * (0.5 * scales[clone_idx].detach())
        new_positions.append(gather("positions", clone_idx) + clone_offset)
        new_log_scales.append(gather("log_scales", clone_idx))
        new_quats.append(gather("quats", clone_idx))
        new_opacity_logits.append(gather("opacity_logits", clone_idx))
        new_sh.append(gather("sh", clone_idx))

    split_idx = should_split.nonzero(as_tuple=True)[0]
    if len(split_idx) > 0:
        split_scales = scales[split_idx].detach()
        split_log_scales = params["log_scales"].detach()[split_idx] - float(np.log(1.6))  # shrink children
        for _ in range(2):
            offset = torch.randn(len(split_idx), 3, device=device) * split_scales
            new_positions.append(gather("positions", split_idx) + offset)
            new_log_scales.append(split_log_scales.clone())
            new_quats.append(gather("quats", split_idx))
            new_opacity_logits.append(gather("opacity_logits", split_idx))
            new_sh.append(gather("sh", split_idx))

    def as_param(tensors):
        return torch.nn.Parameter(torch.cat(tensors, dim=0))

    return dict(
        positions=as_param(new_positions),
        log_scales=as_param(new_log_scales),
        quats=as_param(new_quats),
        opacity_logits=as_param(new_opacity_logits),
        sh=as_param(new_sh),
    )


def train(
    scene_dir: str,
    out_path: str,
    n_splats: int = 4000,
    bounds=((-2.5, 2.5), (-2.5, 2.5), (-2.5, 2.5)),
    sh_degree: int = 1,
    n_iters: int = 2500,
    seed: int = 0,
    log_every: int = 250,
    device: str = "cuda",
    background_color=(0.05, 0.05, 0.05),
    opacity_reg_weight: float = 0.01,
    init_scale: Optional[float] = None,
    densify: bool = False,
    densify_interval: int = 300,
    densify_start: int = 300,
    densify_end: Optional[int] = None,
    densify_grad_percentile: float = 80.0,
    min_opacity: float = 0.005,
    split_scale_threshold: Optional[float] = None,
    max_splats: int = 30000,
    densify_criterion: str = "gradient",
    bq_sigma: float = 0.9,
    bq_window_radius: float = 1.6,
    bq_max_neighbors: int = 150,
    nll_weight: float = 0.0,
    nll_interval: int = 100,
    nll_grid_res: int = 12,
    nll_alpha_threshold: float = 0.5,
    nll_variance_floor: float = 1e-8,
):
    """`densify`, off by default for backward compatibility with existing
    callers/results: turns on gradient-based clone/split + opacity
    pruning (`densify_and_prune`) every `densify_interval` iterations
    within `[densify_start, densify_end)`. The gradient-magnitude cutoff
    is picked per-cycle as the `densify_grad_percentile`-th
    percentile of that cycle's own observed gradient distribution, not a
    hardcoded absolute value: gsplat's view-space gradient magnitudes
    turned out to be ~1e-6 to ~1e-5 for this project's scenes (measured
    directly, not assumed) -- nowhere near the original 3DGS paper's
    normalized-image-space threshold of 0.0002, since the two aren't even
    in the same units. A relative (percentile) threshold sidesteps
    needing to re-calibrate an absolute cutoff per scene/loss/image-
    resolution combination. `split_scale_threshold`
    defaults to `init_scale` (or, if that's also unset, the bounds-derived
    heuristic scale) when omitted: a splat larger than its own starting
    scale is a reasonable proxy for "already covering more than one
    feature's worth of space, so cloning it wouldn't help; splitting
    would."

    ROADMAP.md item 3 ("training under the likelihood"), first
    installment, two independent knobs:

    - `densify_criterion`: `"gradient"` (default, unchanged behavior) or
      `"bq_variance"` -- swaps the densification trigger from gsplat's
      view-space positional gradient to real closed-form BQ position-only
      variance (`compute_per_splat_bq_variance`), queried at every
      current splat's own position via `LocalUncertaintyEngine`, every
      `densify_interval` iterations (same cadence, not every iteration --
      a KD-tree + one BQ solve per splat isn't free). Same percentile-
      threshold split/clone/prune logic either way; only what counts as
      "this splat needs more coverage" changes.
    - `nll_weight` (0.0 = off by default): adds an uncertainty-weighted
      Gaussian-NLL auxiliary loss term every `nll_interval` iterations,
      `0.5 * ((pred-gt)^2 / var + log(var))` averaged over a sparse
      `nll_grid_res` x `nll_grid_res` grid of REAL ray-surface points
      (gsplat's own expected-depth output, unprojected -- same
      construction as `render_sweep_gif.py`, not an approximation of
      pixel positions), `var` the real closed-form BQ position-only
      variance at each of those points. Honest scope note: `var` is
      computed via `LocalUncertaintyEngine` (pure numpy/scipy) from a
      detached snapshot of the current splat state and is *not* itself
      differentiated through -- gradients flow through `pred` in the
      `(pred-gt)^2/var` term as usual (so the practical effect is a
      real-uncertainty-weighted reweighting of the photometric loss,
      down-weighting already-well-resolved points relative to
      under-resolved ones), but the `log(var)` calibration term
      contributes zero gradient to the splat parameters since `var` is
      fixed for that step. Differentiating through the BQ posterior
      itself (a KD-tree ball query + linear solve) is a real further
      step, not done here -- see ROADMAP.md item 3 and
      gs_experiment/results/FINDINGS.md for the account of what this
      first installment does and does not establish.
    - `bq_sigma`/`bq_window_radius`/`bq_max_neighbors`: shared by both
      mechanisms above, not independently tunable per-mechanism in this
      first installment -- defaults match the thin-rod/cylinder scene
      family's established convention (`nbv_experiment.py`,
      `differentiation_experiment.py`), not the lego-scale
      `sigma=0.05`/`window_radius=0.08` used elsewhere in
      `gs_experiment/` -- pick values matching the actual scene's spatial
      scale, not these defaults blindly, for a different scene family.
    """
    import gsplat  # deferred: only needed here, keeps ply_io/loader torch-free

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    images, viewmats, Ks, width, height = load_dataset(scene_dir, device)
    n_views = images.shape[0]

    params = init_splats(n_splats, bounds, sh_degree, device, rng, init_scale=init_scale)
    if densify_end is None:
        densify_end = n_iters - max(500, n_iters // 10)
    if split_scale_threshold is None:
        if init_scale is not None:
            split_scale_threshold = init_scale
        else:
            (x0, x1), (y0, y1), (z0, z1) = bounds
            split_scale_threshold = max(x1 - x0, y1 - y0, z1 - z0) / (n_splats ** (1.0 / 3.0) + 1e-6)

    optimizer = make_optimizer(params)
    grad_accum = torch.zeros(n_splats, device=device)
    grad_denom = torch.zeros(n_splats, device=device)
    # gsplat's default `packed=True` rasterization path expects a single
    # unbatched (channels,) background, not a (1, channels) per-camera
    # background -- confirmed against gsplat.cuda._wrapper.rasterize_to_
    # pixels's shape assertion, which only matches the packed-mode shape
    # when there's no explicit camera-batch dimension.
    background = torch.tensor(background_color, dtype=torch.float32, device=device)

    for it in range(n_iters):
        view_idx = int(rng.integers(0, n_views))
        gt = images[view_idx : view_idx + 1].reshape(height, width, 3)

        quats = params["quats"]
        scales = torch.exp(params["log_scales"])
        opacities = torch.sigmoid(params["opacity_logits"])

        # gsplat's SH color convention is (N, K, 3) (coefficient axis
        # before channel axis); this project's convention everywhere else
        # (spherical_harmonics.eval_sh, ply_io) is (N, 3, K), matching the
        # reference 3DGS ply schema -- transpose only at this boundary
        # rather than picking a convention that fights one side or the
        # other.
        sh_for_gsplat = params["sh"].transpose(1, 2)

        rendered, _, meta = gsplat.rasterization(
            params["positions"],
            quats,
            scales,
            opacities,
            sh_for_gsplat,
            viewmats[view_idx : view_idx + 1],
            Ks[view_idx : view_idx + 1],
            width=width,
            height=height,
            sh_degree=sh_degree,
            backgrounds=background,
        )
        pred = rendered[0]
        if densify:
            # view-space (pixel-space) positional gradient of each
            # contributing splat -- the standard 3DGS densification
            # signal (Kerbl et al. 2023): a splat the optimizer keeps
            # wanting to move in screen space is evidence the region
            # around it is under-reconstructed. means2d is an
            # intermediate (non-leaf) tensor inside gsplat's autograd
            # graph, so its .grad only populates if retained before
            # backward().
            meta["means2d"].retain_grad()

        # opacity sparsity pressure: with no densify/prune step, splats
        # that never earn real photometric gradient (e.g. randomly
        # initialized far from anything the cameras actually see) would
        # otherwise just sit at their initial opacity forever, since zero
        # gradient means zero pull toward zero. This pushes every splat's
        # opacity down by default, so only splats that *do* get enough
        # photometric signal to counteract it stay non-negligible --
        # meaningfully reducing (not eliminating) the "junk splat"
        # contamination of local BQ neighborhoods
        # (bq_splat/results/FINDINGS.md-style empirical finding: without
        # this, ~90% of splats in a differentiation-scene run sat at
        # their initial ~0.27 opacity, scattered across the whole
        # bounding volume, dominating local-neighborhood statistics in
        # regions with no real geometry).
        loss = (
            torch.nn.functional.l1_loss(pred, gt)
            + torch.nn.functional.mse_loss(pred, gt)
            + opacity_reg_weight * opacities.mean()
        )

        nll_term = None
        if nll_weight > 0 and it % nll_interval == 0 and it > 0:
            nll_term = compute_nll_loss_term(
                params, pred, gt, viewmats[view_idx].cpu().numpy(), Ks[view_idx].cpu().numpy(), width, height,
                sh_degree, device, nll_grid_res, bq_sigma, bq_window_radius, bq_max_neighbors,
                nll_alpha_threshold, nll_variance_floor,
            )
            if nll_term is not None:
                loss = loss + nll_weight * nll_term

        optimizer.zero_grad()
        loss.backward()

        if densify and meta["means2d"].grad is not None:
            with torch.no_grad():
                grad_norms = meta["means2d"].grad.norm(dim=-1)
                gaussian_ids = meta["gaussian_ids"]
                grad_accum.index_add_(0, gaussian_ids, grad_norms)
                grad_denom.index_add_(0, gaussian_ids, torch.ones_like(grad_norms))

        optimizer.step()

        if it % log_every == 0 or it == n_iters - 1:
            with torch.no_grad():
                psnr = -10.0 * torch.log10(torch.nn.functional.mse_loss(pred, gt).clamp_min(1e-10))
            n_now = params["positions"].shape[0]
            nll_str = f"  nll {nll_term.item():.4f}" if nll_term is not None else ""
            print(f"iter {it:5d}/{n_iters}  loss {loss.item():.4f}{nll_str}  psnr(train view) {psnr.item():.2f}dB  n_splats {n_now}")

        if densify and densify_start <= it < densify_end and (it - densify_start) % densify_interval == 0 and it > 0:
            with torch.no_grad():
                if densify_criterion == "bq_variance":
                    # closed-form BQ position-only variance at each
                    # splat's own position, in place of the view-space
                    # gradient signal -- every splat has a well-defined
                    # variance, so (unlike the gradient path) there's no
                    # "never received a gradient yet" mask to apply.
                    avg_grad = compute_per_splat_bq_variance(params, bq_sigma, bq_window_radius, bq_max_neighbors, device)
                    has_data = torch.ones_like(avg_grad, dtype=torch.bool)
                elif densify_criterion == "gradient":
                    avg_grad = grad_accum / grad_denom.clamp(min=1)
                    has_data = grad_denom > 0
                else:
                    raise ValueError(f"unknown densify_criterion: {densify_criterion!r}")
                grad_threshold = (
                    torch.quantile(avg_grad[has_data], densify_grad_percentile / 100.0).item()
                    if has_data.any()
                    else float("inf")
                )
                n_before = params["positions"].shape[0]
                params = densify_and_prune(
                    params, avg_grad, grad_threshold, min_opacity, split_scale_threshold, device, max_splats,
                )
            n_after = params["positions"].shape[0]
            optimizer = make_optimizer(params)
            grad_accum = torch.zeros(n_after, device=device)
            grad_denom = torch.zeros(n_after, device=device)
            print(f"iter {it:5d}: densify+prune (grad_threshold={grad_threshold:.2e})  n_splats {n_before} -> {n_after}")

    with torch.no_grad():
        positions = params["positions"].detach().cpu().numpy()
        scales = torch.exp(params["log_scales"]).detach().cpu().numpy()
        rotations = params["quats"].detach().cpu().numpy()
        opacities = torch.sigmoid(params["opacity_logits"]).detach().cpu().numpy()
        sh_coeffs = params["sh"].detach().cpu().numpy()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_3dgs_ply(out_path, positions, scales, rotations, opacities, sh_coeffs, sh_degree)
    print(f"wrote {positions.shape[0]} splats to {out_path}")


def train_with_reference_strategy(
    scene_dir: str,
    out_path: str,
    n_splats: int = 1500,
    bounds=((-2.5, 2.5), (-2.5, 2.5), (-2.5, 2.5)),
    sh_degree: int = 1,
    n_iters: int = 3000,
    seed: int = 0,
    log_every: int = 1000,
    device: str = "cuda",
    background_color=(0.05, 0.05, 0.05),
    init_scale: Optional[float] = None,
    opacity_reg_weight: float = 0.003,
    strategy_kwargs: Optional[dict] = None,
):
    """ROADMAP.md item 4: validate this project's BQ findings against
    gsplat's own official reference densification strategy
    (`gsplat.strategy.DefaultStrategy` -- the standard duplicate/split/
    prune/opacity-reset algorithm from the original 3DGS paper, exposed by
    gsplat as a reusable strategy object, not this project's from-scratch
    reimplementation in `densify_and_prune`), on the same scene/loss/init
    `train` uses everywhere else in this project.

    Deliberately *not* a full reproduction of gsplat's official example
    script (`examples/simple_trainer.py`), which also uses SSIM loss and a
    heavier example-only dependency set (fused-ssim, viser, nerfview -- see
    this module's top docstring for why those were avoided from the
    start): this function keeps the loss, initialization, and optimizer
    schedule identical to `train`, swapping *only* the densification
    mechanism for gsplat's real one. That's a deliberate scope choice, not
    an oversight -- it isolates "is this project's from-scratch
    densification the reason for its BQ findings" from every other way a
    full reference reproduction could differ, at the cost of not being a
    complete SSIM-loss/full-pipeline reference reproduction.

    `params` uses gsplat's own naming convention (`"means"`, `"scales"`,
    `"quats"`, `"opacities"`, plus `"sh"` as a project-specific extra key --
    `DefaultStrategy`'s split/duplicate/prune/reset ops handle any key
    generically once `check_sanity` confirms the four required keys are
    present), and one optimizer per parameter (`Dict[str, Optimizer]`), not
    `train`'s single multi-group Adam -- both are exactly what
    `gsplat.strategy.DefaultStrategy` expects, confirmed against its own
    source (`gsplat/strategy/default.py`, `ops.py`) rather than assumed.
    """
    import gsplat
    from gsplat.strategy import DefaultStrategy

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    images, viewmats, Ks, width, height = load_dataset(scene_dir, device)
    n_views = images.shape[0]

    raw = init_splats(n_splats, bounds, sh_degree, device, rng, init_scale=init_scale)
    key_map = {"means": "positions", "scales": "log_scales", "quats": "quats", "opacities": "opacity_logits", "sh": "sh"}
    params = torch.nn.ParameterDict({dst: raw[src] for dst, src in key_map.items()})
    optimizers = {
        dst: torch.optim.Adam([{"params": [params[dst]], "lr": LR_BY_NAME[src], "name": dst}])
        for dst, src in key_map.items()
    }

    strategy = DefaultStrategy(**(strategy_kwargs or {}))
    strategy.check_sanity(params, optimizers)

    c2ws = np.linalg.inv(viewmats.cpu().numpy())
    centers = c2ws[:, :3, 3]
    scene_scale = float(np.median(np.linalg.norm(centers - centers.mean(axis=0), axis=1))) or 1.0
    state = strategy.initialize_state(scene_scale=scene_scale)

    background = torch.tensor(background_color, dtype=torch.float32, device=device)

    for it in range(n_iters):
        view_idx = int(rng.integers(0, n_views))
        gt = images[view_idx : view_idx + 1].reshape(height, width, 3)

        sh_for_gsplat = params["sh"].transpose(1, 2)
        rendered, _, info = gsplat.rasterization(
            params["means"], params["quats"], torch.exp(params["scales"]), torch.sigmoid(params["opacities"]),
            sh_for_gsplat, viewmats[view_idx : view_idx + 1], Ks[view_idx : view_idx + 1],
            width=width, height=height, sh_degree=sh_degree, backgrounds=background,
        )
        pred = rendered[0]

        strategy.step_pre_backward(params, optimizers, state, it, info)

        loss = (
            torch.nn.functional.l1_loss(pred, gt)
            + torch.nn.functional.mse_loss(pred, gt)
            + opacity_reg_weight * torch.sigmoid(params["opacities"]).mean()
        )
        for opt in optimizers.values():
            opt.zero_grad()
        loss.backward()
        for opt in optimizers.values():
            opt.step()

        # gsplat.rasterization defaults to packed=True; step_post_backward
        # defaults to packed=False and reads info's tensors in a shape
        # specific to whichever mode actually produced them (confirmed via
        # gsplat/strategy/default.py's source, not assumed) -- passing the
        # wrong one throws inside _update_state rather than silently
        # misbehaving, which is how this was caught.
        strategy.step_post_backward(params, optimizers, state, it, info, packed=True)

        if it % log_every == 0 or it == n_iters - 1:
            with torch.no_grad():
                psnr = -10.0 * torch.log10(torch.nn.functional.mse_loss(pred, gt).clamp_min(1e-10))
            print(
                f"iter {it:5d}/{n_iters}  loss {loss.item():.4f}  psnr(train view) {psnr.item():.2f}dB  "
                f"n_splats {params['means'].shape[0]}"
            )

    with torch.no_grad():
        positions = params["means"].detach().cpu().numpy()
        scales = torch.exp(params["scales"]).detach().cpu().numpy()
        rotations = params["quats"].detach().cpu().numpy()
        opacities = torch.sigmoid(params["opacities"]).detach().cpu().numpy()
        sh_coeffs = params["sh"].detach().cpu().numpy()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_3dgs_ply(out_path, positions, scales, rotations, opacities, sh_coeffs, sh_degree)
    print(f"wrote {positions.shape[0]} splats to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_dir", help="directory with transforms.json + images/ (gs_experiment.blender_render output)")
    parser.add_argument("out_path", help="output .ply path")
    parser.add_argument("--n-splats", type=int, default=4000)
    parser.add_argument("--sh-degree", type=int, default=1)
    parser.add_argument("--n-iters", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--opacity-reg-weight", type=float, default=0.01)
    parser.add_argument("--init-scale", type=float, default=None, help="see init_splats' docstring")
    parser.add_argument("--densify", action="store_true", help="enable gradient-based clone/split + opacity pruning")
    parser.add_argument("--densify-interval", type=int, default=300)
    parser.add_argument("--densify-start", type=int, default=300)
    parser.add_argument("--densify-end", type=int, default=None)
    parser.add_argument("--densify-grad-percentile", type=float, default=80.0)
    parser.add_argument("--min-opacity", type=float, default=0.005)
    parser.add_argument("--split-scale-threshold", type=float, default=None)
    parser.add_argument("--max-splats", type=int, default=30000)
    parser.add_argument("--densify-criterion", choices=["gradient", "bq_variance"], default="gradient")
    parser.add_argument("--bq-sigma", type=float, default=0.9)
    parser.add_argument("--bq-window-radius", type=float, default=1.6)
    parser.add_argument("--nll-weight", type=float, default=0.0, help="0 = off; see train()'s docstring")
    parser.add_argument("--nll-interval", type=int, default=100)
    parser.add_argument("--nll-grid-res", type=int, default=12)
    args = parser.parse_args()

    train(
        args.scene_dir,
        args.out_path,
        n_splats=args.n_splats,
        sh_degree=args.sh_degree,
        n_iters=args.n_iters,
        seed=args.seed,
        device=args.device,
        opacity_reg_weight=args.opacity_reg_weight,
        init_scale=args.init_scale,
        densify=args.densify,
        densify_interval=args.densify_interval,
        densify_start=args.densify_start,
        densify_end=args.densify_end,
        densify_grad_percentile=args.densify_grad_percentile,
        min_opacity=args.min_opacity,
        split_scale_threshold=args.split_scale_threshold,
        max_splats=args.max_splats,
        densify_criterion=args.densify_criterion,
        bq_sigma=args.bq_sigma,
        bq_window_radius=args.bq_window_radius,
        nll_weight=args.nll_weight,
        nll_interval=args.nll_interval,
        nll_grid_res=args.nll_grid_res,
    )


if __name__ == "__main__":
    main()
