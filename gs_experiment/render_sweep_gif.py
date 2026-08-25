"""A sweeping-camera GIF: RGB reconstruction with BQ position-only
uncertainty overlaid, animated around a full 360-degree turntable orbit
of a real trained checkpoint.

Reuses render_uncertainty_views.py's approach (BQ variance computed once
at a fixed set of real 3D query points, then projected into each frame's
camera) but as a smooth animation rather than a couple of static views:
the query points and their BQ variance are computed exactly once (world-
space variance doesn't depend on camera pose), so sweeping the camera
around is cheap -- only the projection and the gsplat render change per
frame, which is exactly the "closed-form, essentially free" property
this project's central claim is about, made visible as a moving picture
rather than a table of numbers.

Needs torch + gsplat (requirements-gsplat.txt) + Pillow (already a
dependency, used here for GIF assembly).

Run: .venv-gsplat/bin/python gs_experiment/render_sweep_gif.py <scene_dir> --radius 4.0 --fov-deg 39.6
"""

from __future__ import annotations

import argparse
import sys
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
from gs_experiment.render_uncertainty_views import project_to_pixels

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


def run(
    scene_dir: str,
    n_frames: int = 60,
    radius: float = 4.0,
    phi_deg: float = 25.0,
    fov_deg: float = 39.6,
    resolution: int = 400,
    n_query_points: int = 220,
    sigma: float = 0.05,
    window_radius: float = 0.08,
    min_opacity: float = 0.1,
    background_color=(1.0, 1.0, 1.0),
    seed: int = 0,
    device: str = "cuda",
):
    import gsplat

    ck = read_3dgs_ply(f"{scene_dir}/splats.ply")
    keep = ck["opacities"] > min_opacity
    positions_np = ck["positions"][keep]
    colors_np = ck["sh_coeffs"][keep, :, 0].mean(axis=1)

    bounds = tuple((positions_np[:, d].min() - 0.3, positions_np[:, d].max() + 0.3) for d in range(3))
    pos_kernel = make_default_3d_position_kernel(sigma=sigma)
    engine = LocalUncertaintyEngine(positions=positions_np, values=colors_np, pos_kernel=pos_kernel, scene_bounds=bounds)

    rng = np.random.default_rng(seed)
    query_idx = rng.choice(len(positions_np), size=min(n_query_points, len(positions_np)), replace=False)
    query_points = positions_np[query_idx]
    print(f"computing BQ variance once for {len(query_points)} fixed query points...")
    bq_var = np.array([engine.spatial_only_variance(p, window_radius).variance for p in query_points])
    vmax = np.percentile(bq_var, 95)
    print(f"BQ variance range: {bq_var.min():.6f} - {bq_var.max():.6f} (color scale capped at 95th pct {vmax:.6f})")

    # full checkpoint (not opacity-filtered) for rendering -- gsplat handles low-opacity splats fine
    all_positions = torch.tensor(ck["positions"], dtype=torch.float32, device=device)
    all_scales = torch.tensor(ck["scales"], dtype=torch.float32, device=device)
    all_rotations = torch.tensor(ck["rotations"], dtype=torch.float32, device=device)
    all_opacities = torch.tensor(ck["opacities"], dtype=torch.float32, device=device)
    all_sh = torch.tensor(ck["sh_coeffs"], dtype=torch.float32, device=device).transpose(1, 2)
    sh_degree = ck["sh_degree"]
    background = torch.tensor(background_color, dtype=torch.float32, device=device)
    K = fov_x_to_intrinsics(np.deg2rad(fov_deg), resolution, resolution)

    cameras = turntable_ring(radius=radius, n_views=n_frames, phi_deg=phi_deg)

    frame_paths = []
    frames_dir = RESULTS_DIR / "sweep_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i, cam in enumerate(cameras):
            c2w = c2w_from_camera_pose(cam)
            viewmat_np = opencv_viewmat_from_c2w(c2w)
            viewmat = torch.tensor(viewmat_np, dtype=torch.float32, device=device)[None]
            Ks = torch.tensor(K, dtype=torch.float32, device=device)[None]

            rendered, _, _ = gsplat.rasterization(
                all_positions, all_rotations, all_scales, all_opacities, all_sh, viewmat, Ks,
                width=resolution, height=resolution, sh_degree=sh_degree, backgrounds=background,
            )
            recon = rendered[0].clamp(0, 1).cpu().numpy()

            pixels, in_front = project_to_pixels(query_points, viewmat_np, K)
            in_view = (
                in_front & (pixels[:, 0] >= 0) & (pixels[:, 0] < resolution) & (pixels[:, 1] >= 0) & (pixels[:, 1] < resolution)
            )

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(recon)
            sc = ax.scatter(
                pixels[in_view, 0], pixels[in_view, 1], c=bq_var[in_view], cmap="inferno",
                vmin=0, vmax=vmax, s=13, alpha=0.8, edgecolors="white", linewidths=0.3,
            )
            ax.set_xlim(0, resolution)
            ax.set_ylim(resolution, 0)
            ax.axis("off")
            ax.set_title("BQ position-only uncertainty (real trained checkpoint)", fontsize=9)
            fig.tight_layout(pad=0.3)
            frame_path = frames_dir / f"frame_{i:03d}.png"
            fig.savefig(frame_path, dpi=120)
            plt.close(fig)
            frame_paths.append(frame_path)
            if i % 10 == 0:
                print(f"rendered frame {i + 1}/{n_frames}")

    images = [Image.open(p).convert("RGB") for p in frame_paths]
    out_path = RESULTS_DIR / "uncertainty_sweep.gif"
    images[0].save(
        out_path, save_all=True, append_images=images[1:], duration=80, loop=0, optimize=True,
    )
    print(f"\nSaved {out_path} ({len(images)} frames)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene_dir")
    parser.add_argument("--n-frames", type=int, default=60)
    parser.add_argument("--radius", type=float, default=4.0)
    parser.add_argument("--phi-deg", type=float, default=25.0)
    parser.add_argument("--fov-deg", type=float, default=39.6)
    parser.add_argument("--resolution", type=int, default=400)
    args = parser.parse_args()
    run(args.scene_dir, n_frames=args.n_frames, radius=args.radius, phi_deg=args.phi_deg, fov_deg=args.fov_deg, resolution=args.resolution)


if __name__ == "__main__":
    main()
