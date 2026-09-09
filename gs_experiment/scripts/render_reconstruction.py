"""Render ground-truth vs. gsplat-reconstruction comparisons for a
trained checkpoint (gs_experiment.scripts.train_minimal_gsplat's output), plus
error maps and (by default) real per-pixel BQ uncertainty maps -- so a
reader can see, side by side on the exact same held-out/test views used
to judge quality, both where the reconstruction is actually wrong and
where BQ itself expects to be uncertain (ROADMAP.md item 2: "render it
and look, before reaching for statistics"). Uncertainty is computed the
same way as render_directional_uncertainty_sweep.py's per-pixel field
(real depth-unprojection via gsplat's own "ED" output, real closed-form
BQ position-only and position+direction variance at each ray-surface
hit) rather than approximated, so a bad-quality view and a
low-confidence view can be told apart on sight rather than conflated.

Needs torch + gsplat (requirements-gsplat.txt).

Run: .venv-gsplat/bin/python gs_experiment/render_reconstruction.py <scene_dir> [--view-indices 0 20 40] [--out <path>] [--no-uncertainty]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from gs_experiment.nerf_transforms import fov_x_to_intrinsics, load_transforms, opencv_viewmat_from_c2w
from gs_experiment.ply_io import read_3dgs_ply

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def render_views(scene_dir: str, view_indices, checkpoint_dir=None, background_color=(0.05, 0.05, 0.05), device="cuda"):
    """`scene_dir` supplies transforms.json + images (the views to render);
    `checkpoint_dir` (defaults to `scene_dir`) supplies splats.ply. Pass a
    different `checkpoint_dir` to render a checkpoint trained on one split
    (e.g. `wide/`) against another split's views (e.g. a held-out `eval/`),
    for genuine held-out-test-image comparisons rather than reconstruction
    on the views the model was fit to."""
    import gsplat

    checkpoint = read_3dgs_ply(os.path.join(checkpoint_dir or scene_dir, "splats.ply"))
    camera_angle_x, frames = load_transforms(os.path.join(scene_dir, "transforms.json"))

    with Image.open(os.path.join(scene_dir, frames[0][0] + ".png")) as im:
        width, height = im.size
    K = fov_x_to_intrinsics(camera_angle_x, width, height)
    background = torch.tensor(background_color, dtype=torch.float32, device=device)

    positions = torch.tensor(checkpoint["positions"], dtype=torch.float32, device=device)
    scales = torch.tensor(checkpoint["scales"], dtype=torch.float32, device=device)
    rotations = torch.tensor(checkpoint["rotations"], dtype=torch.float32, device=device)
    opacities = torch.tensor(checkpoint["opacities"], dtype=torch.float32, device=device)
    # (N, 3, K) on disk (this project's convention, see ply_io); gsplat wants (N, K, 3)
    sh = torch.tensor(checkpoint["sh_coeffs"], dtype=torch.float32, device=device).transpose(1, 2)
    sh_degree = checkpoint["sh_degree"]

    results = []
    with torch.no_grad():
        for i in view_indices:
            file_path, c2w = frames[i]
            viewmat = torch.tensor(opencv_viewmat_from_c2w(c2w), dtype=torch.float32, device=device)[None]
            Ks = torch.tensor(K, dtype=torch.float32, device=device)[None]

            rendered, _, _ = gsplat.rasterization(
                positions, rotations, scales, opacities, sh, viewmat, Ks,
                width=width, height=height, sh_degree=sh_degree, backgrounds=background,
            )
            recon = rendered[0].clamp(0, 1).cpu().numpy()

            gt_path = os.path.join(scene_dir, file_path + ".png")
            gt = np.asarray(Image.open(gt_path).convert("RGB"), dtype=np.float32) / 255.0

            results.append((i, gt, recon))
    return results, checkpoint


def unproject_depth_grid(depth: np.ndarray, K: np.ndarray, c2w_cv: np.ndarray) -> np.ndarray:
    """Same construction as train_minimal_gsplat.py's/render_directional_uncertainty_sweep.py's
    function of the same name (duplicated rather than imported, matching
    this project's established convention for this exact helper -- see
    train_minimal_gsplat.py's module docstring): depth (H, W) in OpenCV
    camera space -> (H, W, 3) world-space points."""
    h, w = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    us, vs = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    x_cam = (us - cx) / fx * depth
    y_cam = (vs - cy) / fy * depth
    cam_points = np.stack([x_cam, y_cam, depth, np.ones_like(depth)], axis=-1)
    world_points = cam_points @ c2w_cv.T
    return world_points[..., :3]


def compute_uncertainty_maps(
    scene_dir: str,
    view_indices,
    frames,
    camera_angle_x: float,
    width: int,
    height: int,
    checkpoint: dict,
    checkpoint_dir=None,
    device: str = "cuda",
    sigma: float = 0.9,
    # Marginal-likelihood-fitted (hyperparams.py::fit_kernel_param_pooled_nd, applied to
    # DirectionalKernel directly -- it already accepts the same (N,D) array interface),
    # pooled across real per-splat multi-view (direction, color) observations from 7
    # real checkpoints -- not hand-picked. Held-out log marginal likelihood 1413 at this
    # value vs 202 at the old hardcoded 4.0 (kappa=4.0 was ~5x too concentrated: real
    # per-splat color varies much less with viewing angle than that implied, so
    # moderately-off-angle real observations were being treated as almost uncorrelated
    # when the data says they should still count).
    kappa: float = 0.745,
    window_radius: float = 1.6,
    max_neighbors: int = 150,
    alpha_threshold: float = 0.5,
    depth_width: int = 112,
    depth_height: int = 42,
    attribution_angular_tol: float = 0.01,
    max_observations_per_splat: Optional[int] = None,
):
    """Real per-pixel BQ uncertainty at every view in `view_indices`, on
    the exact same checkpoint `render_views` just rendered RGB from --
    the same depth-unprojection + closed-form-variance construction as
    render_directional_uncertainty_sweep.py's orbit GIF (real ray-surface
    hits via gsplat's "ED" render mode, not an approximation), just
    evaluated at real dataset views instead of a synthetic sweep so it
    lines up 1:1 with the GT/recon/error columns already being rendered.

    `scene_dir` supplies the views being queried (`frames`); `checkpoint_dir`
    (defaults to `scene_dir`, matching `render_views`) supplies splats.ply
    *and* the transforms.json used to attribute which training views
    observed each splat -- that attribution must come from the checkpoint's
    own training views, not whatever split is being rendered here, or a
    held-out eval split would silently masquerade as the training-view pool.

    `sigma`/`kappa`/`window_radius`/`max_neighbors` match
    render_directional_uncertainty_sweep.py's own defaults, tuned for
    that script's synthetic-scene convention -- pick values matching the
    actual scene's spatial scale for a different scene family (see
    train_minimal_gsplat.train's docstring for the same caveat).

    Returns a list of (spatial_map, directional_map) aligned with
    `view_indices`, each (height, width) with NaN outside the region
    gsplat itself reports as covered (alpha <= alpha_threshold).
    `directional_map` is `variance / prior_variance` (see
    gpu_uncertainty.compute_directional_variance_batched's docstring), a
    bounded [0,1] ratio, not raw posterior variance -- its absolute scale
    is comparable across scenes/checkpoints/figures, unlike the raw
    variance, whose magnitude is dominated by `sigma` (an 18x sigma
    difference between two scripts once produced a ~12,000x difference in
    raw variance on the *same* checkpoint -- not a real signal). `spatial_map`
    is still raw (position-only) variance -- not used in the current figures.

    `max_observations_per_splat`: passed straight through to
    `load_from_gsplat_checkpoint` -- caps the (splat, observing-camera)
    row count the directional engine below is built from, which is what
    actually determines its host memory footprint at high splat counts
    (see that function's docstring). `None` (the default) keeps every
    observation, i.e. unchanged from before this parameter existed.
    """
    import gsplat

    from gs_experiment.gpu_uncertainty import compute_directional_variance_batched
    from gs_experiment.kernels import DirectionalKernel
    from gs_experiment.nerf_transforms import camera_pose_from_c2w
    from gs_experiment.pixel_uncertainty import LocalUncertaintyEngine, make_default_3d_position_kernel
    from gs_experiment.splat_scene import load_from_gsplat_checkpoint, splat_observations

    scene = load_from_gsplat_checkpoint(
        checkpoint_dir or scene_dir, attribution_angular_tol=attribution_angular_tol, use_gpu_attribution=True,
        # 0.1 matches this project's existing min-opacity convention elsewhere (e.g.
        # fit_hyperparameters.py) -- excludes GS-training floaters from hard-occluding
        # real splats during attribution (see load_from_gsplat_checkpoint's docstring).
        attribution_min_opacity=0.1,
        max_observations_per_splat=max_observations_per_splat,
    )
    obs_positions, obs_directions, obs_values, obs_opacities, obs_scales, obs_rotations = splat_observations(
        scene, include_render_attrs=True
    )
    bounds = tuple((obs_positions[:, d].min() - 1.0, obs_positions[:, d].max() + 1.0) for d in range(3))

    pos_kernel = make_default_3d_position_kernel(sigma=sigma)
    dir_kernel = DirectionalKernel(kappa=kappa)
    # deduplicated splat positions for the position-only engine, camera-
    # expanded observation rows for the directional one -- reusing the
    # expanded rows for the position-only query would silently leak
    # observation-count into a signal meant to be blind to direction (see
    # render_directional_uncertainty_sweep.py's run_view_projection for
    # the same distinction).
    spatial_engine = LocalUncertaintyEngine(
        positions=scene.positions, values=scene.colors, pos_kernel=pos_kernel, scene_bounds=bounds,
        max_neighbors=max_neighbors, opacities=scene.opacities,
    )
    directional_engine = LocalUncertaintyEngine(
        positions=obs_positions, values=obs_values, pos_kernel=pos_kernel, scene_bounds=bounds,
        directions=obs_directions, dir_kernel=dir_kernel, max_neighbors=max_neighbors,
        opacities=obs_opacities, scales=obs_scales, rotations=obs_rotations,
    )

    K_full = fov_x_to_intrinsics(camera_angle_x, width, height)
    K_depth = K_full.copy()
    K_depth[0, 0] *= depth_width / width
    K_depth[0, 2] *= depth_width / width
    K_depth[1, 1] *= depth_height / height
    K_depth[1, 2] *= depth_height / height

    all_positions = torch.tensor(checkpoint["positions"], dtype=torch.float32, device=device)
    all_scales = torch.tensor(checkpoint["scales"], dtype=torch.float32, device=device)
    all_rotations = torch.tensor(checkpoint["rotations"], dtype=torch.float32, device=device)
    all_opacities = torch.tensor(checkpoint["opacities"], dtype=torch.float32, device=device)
    all_sh = torch.tensor(checkpoint["sh_coeffs"], dtype=torch.float32, device=device).transpose(1, 2)
    sh_degree = checkpoint["sh_degree"]
    K_depth_t = torch.tensor(K_depth, dtype=torch.float32, device=device)

    def upsample(field, valid):
        field_img = Image.fromarray(np.nan_to_num(field, nan=0.0).astype(np.float32), mode="F")
        valid_img = Image.fromarray((valid * 255).astype(np.uint8))
        field_up = np.array(field_img.resize((width, height), Image.BILINEAR))
        valid_up = np.array(valid_img.resize((width, height), Image.NEAREST)) > 127
        return np.where(valid_up, field_up, np.nan)

    maps = []
    with torch.no_grad():
        for i in view_indices:
            _, c2w = frames[i]
            viewmat_np = opencv_viewmat_from_c2w(c2w)
            viewmat = torch.tensor(viewmat_np, dtype=torch.float32, device=device)
            c2w_cv = np.linalg.inv(viewmat_np)
            cam_center = c2w[:3, 3]

            rendered_lo, alpha_lo, _ = gsplat.rasterization(
                all_positions, all_rotations, all_scales, all_opacities, all_sh, viewmat[None], K_depth_t[None],
                width=depth_width, height=depth_height, sh_degree=sh_degree, render_mode="ED",
            )
            depth_map = rendered_lo[0, ..., 0].cpu().numpy()
            alpha_map = alpha_lo[0, ..., 0].cpu().numpy()
            valid = alpha_map > alpha_threshold

            world_points = unproject_depth_grid(depth_map, K_depth, c2w_cv)
            camera = camera_pose_from_c2w(c2w)
            camera_index = directional_engine.build_bearing_index(camera)

            spatial_field = np.full((depth_height, depth_width), np.nan)
            dir_field = np.full((depth_height, depth_width), np.nan)
            ys, xs = np.where(valid)
            points = world_points[ys, xs]
            to_camera = cam_center[None, :] - points
            query_directions = to_camera / np.linalg.norm(to_camera, axis=1, keepdims=True)

            for y, x, point in zip(ys, xs, points):
                spatial_field[y, x] = spatial_engine.rendering_aware_variance(point, window_radius).variance
            # Batched over every valid pixel in this view at once (one GPU pass instead of one
            # Python call per pixel) -- see gpu_uncertainty.py's module docstring: this is the
            # same closed-form math rendering_aware_variance_along_ray_directional computes per
            # call, verified to agree with it to ~1e-10 on real data
            # (tests/gs_experiment/test_gpu_uncertainty.py), not an approximation.
            if len(ys) > 0:
                # dir_field is variance/prior_variance (in [0,1]: 0 = fully informed by real
                # data, 1 = no relevant observations at all), not raw posterior variance.
                # The raw variance's absolute scale is dominated by sigma_rbf (differs by
                # orders of magnitude for a small sigma change, on the *same* checkpoint --
                # not a real cross-scene/cross-condition difference), so it isn't comparable
                # across panels/scenes/figures; this ratio is much less sensitive to sigma
                # (numerator and denominator scale together) and is bounded, so a reader can
                # compare it directly across panels without a per-panel autoscaled colorbar.
                variance, prior_variance = compute_directional_variance_batched(
                    directional_engine, camera_index, points, query_directions,
                    angular_tol=0.05, sigma_rbf=sigma, kappa=kappa, max_candidates=500, device=device,
                )
                dir_field[ys, xs] = variance / np.maximum(prior_variance, 1e-300)

            maps.append((upsample(spatial_field, valid), upsample(dir_field, valid)))

    return maps


def plot_comparisons(results, out_path, title, uncertainty_maps=None):
    """`uncertainty_maps`, if given, must be aligned 1:1 with `results`
    (see `compute_uncertainty_maps`) -- adds two more columns (spatial and
    position+direction BQ variance) so a reader can compare reconstruction
    error against BQ's own uncertainty on the same views at a glance."""
    n = len(results)
    n_cols = 5 if uncertainty_maps is not None else 3
    fig, axes = plt.subplots(n, n_cols, figsize=(3 * n_cols, 3 * n))
    if n == 1:
        axes = axes[None, :]

    if uncertainty_maps is not None:
        spatial_vmax = np.nanpercentile(np.stack([m[0] for m in uncertainty_maps]), 95)
        dir_vmax = np.nanpercentile(np.stack([m[1] for m in uncertainty_maps]), 95)

    for row, (i, gt, recon) in enumerate(results):
        err = np.abs(gt - recon).mean(axis=-1)
        axes[row, 0].imshow(gt)
        axes[row, 0].set_title(f"view {i}: ground truth" if row == 0 else "")
        axes[row, 1].imshow(recon)
        axes[row, 1].set_title("gsplat reconstruction" if row == 0 else "")
        im = axes[row, 2].imshow(err, cmap="inferno", vmin=0, vmax=0.3)
        axes[row, 2].set_title("|error| (mean over RGB)" if row == 0 else "")
        fig.colorbar(im, ax=axes[row, 2], fraction=0.046, pad=0.04)

        if uncertainty_maps is not None:
            spatial_map, dir_map = uncertainty_maps[row]
            cmap = plt.get_cmap("inferno").copy()
            cmap.set_bad(color=(0.05, 0.05, 0.05))
            im3 = axes[row, 3].imshow(spatial_map, cmap=cmap, vmin=0, vmax=spatial_vmax)
            axes[row, 3].set_title("spatial (quadrature) BQ variance" if row == 0 else "")
            fig.colorbar(im3, ax=axes[row, 3], fraction=0.046, pad=0.04)

            im4 = axes[row, 4].imshow(dir_map, cmap=cmap, vmin=0, vmax=dir_vmax)
            axes[row, 4].set_title("position+direction BQ variance" if row == 0 else "")
            fig.colorbar(im4, ax=axes[row, 4], fraction=0.046, pad=0.04)

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel(f"view {i}", fontsize=9)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_dir", help="directory with transforms.json + images -- the views to render")
    parser.add_argument(
        "--checkpoint-dir", default=None,
        help="directory with splats.ply (default: scene_dir). Pass a training split's dir here while "
        "scene_dir points at a held-out split (e.g. an eval/ built by prepare_nerf_synthetic.py) to "
        "render genuine held-out-test-image comparisons instead of reconstruction on training views.",
    )
    parser.add_argument("--view-indices", type=int, nargs="+", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--background-color", type=float, nargs=3, default=[0.05, 0.05, 0.05],
        help="must match the background the checkpoint was trained against (e.g. 1 1 1 for NeRF-Synthetic's white)",
    )
    parser.add_argument(
        "--no-uncertainty", action="store_true",
        help="skip the two BQ uncertainty columns (faster; RGB/error only, the old behavior)",
    )
    parser.add_argument("--sigma", type=float, default=0.9, help="position kernel bandwidth; see compute_uncertainty_maps' docstring")
    parser.add_argument("--kappa", type=float, default=4.0, help="directional kernel concentration")
    parser.add_argument("--window-radius", type=float, default=1.6)
    parser.add_argument("--max-neighbors", type=int, default=150)
    parser.add_argument("--depth-width", type=int, default=112)
    parser.add_argument("--depth-height", type=int, default=42)
    args = parser.parse_args()

    camera_angle_x, frames = load_transforms(os.path.join(args.scene_dir, "transforms.json"))
    n_views = len(frames)
    view_indices = args.view_indices or sorted(set([0, n_views // 4, n_views // 2, n_views - 1]))
    view_indices = [i for i in view_indices if 0 <= i < n_views]

    results, checkpoint = render_views(
        args.scene_dir, view_indices, checkpoint_dir=args.checkpoint_dir, background_color=tuple(args.background_color),
    )

    psnrs = []
    for i, gt, recon in results:
        mse = float(np.mean((gt - recon) ** 2))
        psnr = -10.0 * np.log10(max(mse, 1e-10))
        psnrs.append(psnr)
        print(f"view {i}: PSNR {psnr:.2f}dB")
    print(f"mean PSNR over shown views: {np.mean(psnrs):.2f}dB  ({checkpoint['positions'].shape[0]} splats)")

    uncertainty_maps = None
    if not args.no_uncertainty:
        height, width = results[0][1].shape[:2]
        uncertainty_maps = compute_uncertainty_maps(
            args.scene_dir, view_indices, frames, camera_angle_x, width, height, checkpoint,
            checkpoint_dir=args.checkpoint_dir,
            sigma=args.sigma, kappa=args.kappa, window_radius=args.window_radius, max_neighbors=args.max_neighbors,
            depth_width=args.depth_width, depth_height=args.depth_height,
        )

    out_path = args.out or (RESULTS_DIR / f"reconstruction_{Path(args.scene_dir).name}.png")
    plot_comparisons(
        results, out_path, title=f"gsplat reconstruction vs. ground truth ({args.scene_dir})",
        uncertainty_maps=uncertainty_maps,
    )


if __name__ == "__main__":
    main()
