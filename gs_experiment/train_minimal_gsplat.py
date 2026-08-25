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
            print(f"iter {it:5d}/{n_iters}  loss {loss.item():.4f}  psnr(train view) {psnr.item():.2f}dB  n_splats {n_now}")

        if densify and densify_start <= it < densify_end and (it - densify_start) % densify_interval == 0 and it > 0:
            with torch.no_grad():
                avg_grad = grad_accum / grad_denom.clamp(min=1)
                has_data = grad_denom > 0
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
    )


if __name__ == "__main__":
    main()
