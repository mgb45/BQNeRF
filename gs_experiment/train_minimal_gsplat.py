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
reimplementation of 3DGS training. No densification/pruning: the splat
count is fixed at initialization (randomly seeded, generously sized for
the target scene), which keeps the trainer simple at the cost of some
final quality relative to a real adaptive-density trainer -- acceptable
here since the point is to have real optimized splat geometry to compute
uncertainty over, not to hit a PSNR target. A small opacity-sparsity
regularizer (`opacity_reg_weight`) substitutes for real pruning: without
it, splats that never earn photometric gradient (most of a randomly-
initialized population that's larger than the scene needs) sit inert at
their initial opacity forever and contaminate downstream local-
neighborhood statistics -- see the comment at the loss computation for
the empirical motivation.

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
):
    import gsplat  # deferred: only needed here, keeps ply_io/loader torch-free

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    images, viewmats, Ks, width, height = load_dataset(scene_dir, device)
    n_views = images.shape[0]

    params = init_splats(n_splats, bounds, sh_degree, device, rng, init_scale=init_scale)

    param_groups = [
        {"params": [params["positions"]], "lr": 2e-3, "name": "positions"},
        {"params": [params["log_scales"]], "lr": 5e-3, "name": "log_scales"},
        {"params": [params["quats"]], "lr": 1e-3, "name": "quats"},
        {"params": [params["opacity_logits"]], "lr": 5e-2, "name": "opacity_logits"},
        {"params": [params["sh"]], "lr": 2.5e-3, "name": "sh"},
    ]
    optimizer = torch.optim.Adam(param_groups)
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

        rendered, _, _ = gsplat.rasterization(
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
        optimizer.step()

        if it % log_every == 0 or it == n_iters - 1:
            with torch.no_grad():
                psnr = -10.0 * torch.log10(torch.nn.functional.mse_loss(pred, gt).clamp_min(1e-10))
            print(f"iter {it:5d}/{n_iters}  loss {loss.item():.4f}  psnr(train view) {psnr.item():.2f}dB")

    with torch.no_grad():
        positions = params["positions"].detach().cpu().numpy()
        scales = torch.exp(params["log_scales"]).detach().cpu().numpy()
        rotations = params["quats"].detach().cpu().numpy()
        opacities = torch.sigmoid(params["opacity_logits"]).detach().cpu().numpy()
        sh_coeffs = params["sh"].detach().cpu().numpy()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_3dgs_ply(out_path, positions, scales, rotations, opacities, sh_coeffs, sh_degree)
    print(f"wrote {n_splats} splats to {out_path}")


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
    )


if __name__ == "__main__":
    main()
