"""A sweeping-camera GIF: RGB reconstruction and a genuinely per-pixel BQ
position-only uncertainty render, side by side, animated around a full
360-degree turntable orbit of a real trained checkpoint.

Earlier version of this script queried BQ variance at ~200 *splat*
positions and interpolated the result over the image -- a compute-budget
shortcut, not a limit of the method. BQ variance can be evaluated at any
3D point a camera ray actually hits, which is what this version does:
gsplat's own expected-depth output ("ED" render mode) gives a real
per-pixel depth, unprojected through the camera intrinsics into a real
world-space surface point for *every* pixel on a query grid -- then each
of those real points gets its own BQ solve, no interpolation from splat
positions involved.

Full display resolution (e.g. 400x400) at every pixel would mean one BQ
solve per pixel per frame -- measured at ~1-5ms/solve depending on
`max_neighbors` (gs_experiment/results/FINDINGS.md-style timing done
before committing to a resolution/frame count, not assumed), so
160,000 pixels/frame x 60 frames is hours, not minutes. The uncertainty
pass therefore runs on a coarser `depth_resolution` grid (still genuinely
per-pixel *at that resolution*, every point a real ray-surface hit, not
an interpolated splat sample) and is upsampled by simple image resize for
display -- the RGB panel still renders at full `resolution`.

Needs torch + gsplat (requirements-gsplat.txt) + Pillow.

Run: .venv-gsplat/bin/python gs_experiment/render_sweep_gif.py <scene_dir> --depth-resolution 80
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from gs_experiment.camera import turntable_ring
from gs_experiment.nerf_transforms import fov_x_to_intrinsics
from gs_experiment.pixel_uncertainty import LocalUncertaintyEngine, make_default_3d_position_kernel
from gs_experiment.ply_io import read_3dgs_ply

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def c2w_from_camera_pose(camera) -> np.ndarray:
    right = np.cross(camera.forward, camera.up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, camera.forward)
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -camera.forward
    c2w[:3, 3] = camera.center
    return c2w


def opencv_viewmat_from_c2w(c2w: np.ndarray) -> np.ndarray:
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    return np.linalg.inv(c2w @ flip)


def unproject_depth_grid(depth: np.ndarray, K: np.ndarray, c2w_cv: np.ndarray) -> np.ndarray:
    """depth: (H, W) per-pixel depth in OpenCV camera space (x right, y
    down, z forward). Returns (H, W, 3) world-space points -- the real
    ray-surface hit for every pixel, via standard pinhole unprojection
    then the camera-to-world transform (inverse of the OpenCV viewmat)."""
    h, w = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    us, vs = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    x_cam = (us - cx) / fx * depth
    y_cam = (vs - cy) / fy * depth
    cam_points = np.stack([x_cam, y_cam, depth, np.ones_like(depth)], axis=-1)  # (H, W, 4)
    world_points = cam_points @ c2w_cv.T
    return world_points[..., :3]


def run(
    scene_dir: str,
    n_frames: int = 60,
    radius: float = 4.0,
    phi_deg: float = 25.0,
    fov_deg: float = 39.6,
    resolution: int = 400,
    depth_resolution: int = 80,
    sigma: float = 0.05,
    window_radius: float = 0.08,
    min_opacity: float = 0.1,
    max_neighbors: int = 150,
    alpha_threshold: float = 0.5,
    background_color=(1.0, 1.0, 1.0),
    device: str = "cuda",
):
    import gsplat

    ck = read_3dgs_ply(f"{scene_dir}/splats.ply")
    keep = ck["opacities"] > min_opacity
    positions_np = ck["positions"][keep]
    colors_np = ck["sh_coeffs"][keep, :, 0].mean(axis=1)

    bounds = tuple((positions_np[:, d].min() - 0.3, positions_np[:, d].max() + 0.3) for d in range(3))
    pos_kernel = make_default_3d_position_kernel(sigma=sigma)
    engine = LocalUncertaintyEngine(
        positions=positions_np, values=colors_np, pos_kernel=pos_kernel, scene_bounds=bounds, max_neighbors=max_neighbors,
    )

    # calibrate actual per-solve cost on this checkpoint before committing
    # to depth_resolution x depth_resolution x n_frames of them
    t0 = time.time()
    for p in positions_np[:50]:
        engine.spatial_only_variance(p, window_radius)
    per_query_s = (time.time() - t0) / 50
    total_queries = depth_resolution * depth_resolution * n_frames
    print(
        f"measured {per_query_s * 1000:.2f} ms/BQ-solve (max_neighbors={max_neighbors}); "
        f"{depth_resolution}x{depth_resolution} x {n_frames} frames = {total_queries} solves "
        f"-> est. {total_queries * per_query_s / 60:.1f} min total"
    )

    all_positions = torch.tensor(ck["positions"], dtype=torch.float32, device=device)
    all_scales = torch.tensor(ck["scales"], dtype=torch.float32, device=device)
    all_rotations = torch.tensor(ck["rotations"], dtype=torch.float32, device=device)
    all_opacities = torch.tensor(ck["opacities"], dtype=torch.float32, device=device)
    all_sh = torch.tensor(ck["sh_coeffs"], dtype=torch.float32, device=device).transpose(1, 2)
    sh_degree = ck["sh_degree"]
    background = torch.tensor(background_color, dtype=torch.float32, device=device)
    K_display = fov_x_to_intrinsics(np.deg2rad(fov_deg), resolution, resolution)
    K_depth = fov_x_to_intrinsics(np.deg2rad(fov_deg), depth_resolution, depth_resolution)

    cameras = turntable_ring(radius=radius, n_views=n_frames, phi_deg=phi_deg)

    frame_paths = []
    frames_dir = RESULTS_DIR / "sweep_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad(color=background_color)

    all_variances_for_scale = []

    with torch.no_grad():
        for i, cam in enumerate(cameras):
            c2w = c2w_from_camera_pose(cam)
            viewmat_np = opencv_viewmat_from_c2w(c2w)
            viewmat = torch.tensor(viewmat_np, dtype=torch.float32, device=device)
            c2w_cv = np.linalg.inv(viewmat_np)

            # display-resolution RGB
            rendered, _, _ = gsplat.rasterization(
                all_positions, all_rotations, all_scales, all_opacities, all_sh, viewmat[None],
                torch.tensor(K_display, dtype=torch.float32, device=device)[None],
                width=resolution, height=resolution, sh_degree=sh_degree, backgrounds=background,
            )
            recon = rendered[0].clamp(0, 1).cpu().numpy()

            # depth-resolution expected-depth pass, for real ray-surface
            # points -- "ED" alone (not "RGB+ED"): backgrounds only
            # applies to color channels in gsplat, and getting its shape
            # right for a combined RGB+depth packed-mode call turned out
            # fiddly (a real, hit-directly issue, not assumed) -- depth-
            # only avoids it entirely, and background pixels are masked
            # out via alpha regardless of what depth value they get.
            rendered_lo, alpha_lo, meta_lo = gsplat.rasterization(
                all_positions, all_rotations, all_scales, all_opacities, all_sh, viewmat[None],
                torch.tensor(K_depth, dtype=torch.float32, device=device)[None],
                width=depth_resolution, height=depth_resolution, sh_degree=sh_degree,
                render_mode="ED",
            )
            depth_map = rendered_lo[0, ..., 0].cpu().numpy()
            alpha_map = alpha_lo[0, ..., 0].cpu().numpy()
            valid = alpha_map > alpha_threshold

            world_points = unproject_depth_grid(depth_map, K_depth, c2w_cv)

            field = np.full((depth_resolution, depth_resolution), np.nan)
            ys, xs = np.where(valid)
            for y, x in zip(ys, xs):
                field[y, x] = engine.spatial_only_variance(world_points[y, x], window_radius).variance
            all_variances_for_scale.append(field[valid])

            field_img = Image.fromarray(np.nan_to_num(field, nan=0.0).astype(np.float32), mode="F")
            valid_img = Image.fromarray((valid * 255).astype(np.uint8))
            field_up = np.array(field_img.resize((resolution, resolution), Image.BILINEAR))
            valid_up = np.array(valid_img.resize((resolution, resolution), Image.NEAREST)) > 127
            field_up = np.where(valid_up, field_up, np.nan)

            if i == 0:
                vmax = np.nanpercentile(field_up, 95)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(recon)
            axes[0].set_title("reconstruction", fontsize=10)
            axes[0].axis("off")

            im = axes[1].imshow(field_up, cmap=cmap, vmin=0, vmax=vmax)
            axes[1].set_title(f"BQ position-only uncertainty\n(per-pixel, {depth_resolution}x{depth_resolution} real ray-hits)", fontsize=9)
            axes[1].axis("off")
            fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

            fig.tight_layout(pad=0.3)
            frame_path = frames_dir / f"frame_{i:03d}.png"
            fig.savefig(frame_path, dpi=120)
            plt.close(fig)
            frame_paths.append(frame_path)
            print(f"rendered frame {i + 1}/{n_frames}")

    images = [Image.open(p).convert("RGB") for p in frame_paths]
    out_path = RESULTS_DIR / "uncertainty_sweep.gif"
    images[0].save(out_path, save_all=True, append_images=images[1:], duration=80, loop=0, optimize=True)
    print(f"\nSaved {out_path} ({len(images)} frames)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene_dir")
    parser.add_argument("--n-frames", type=int, default=60)
    parser.add_argument("--radius", type=float, default=4.0)
    parser.add_argument("--phi-deg", type=float, default=25.0)
    parser.add_argument("--fov-deg", type=float, default=39.6)
    parser.add_argument("--resolution", type=int, default=400)
    parser.add_argument("--depth-resolution", type=int, default=80)
    parser.add_argument("--max-neighbors", type=int, default=150)
    args = parser.parse_args()
    run(
        args.scene_dir, n_frames=args.n_frames, radius=args.radius, phi_deg=args.phi_deg, fov_deg=args.fov_deg,
        resolution=args.resolution, depth_resolution=args.depth_resolution, max_neighbors=args.max_neighbors,
    )


if __name__ == "__main__":
    main()
