"""The general entry point for looking at BQ uncertainty on a real,
arbitrary `gsplat` checkpoint: a sweeping-camera GIF with RGB, spatial
(quadrature) uncertainty, and directional (epistemic) uncertainty side by
side, at every pixel, as the camera orbits -- not the handful of
point-sample summary numbers the earlier aggregate experiments reported,
which (fairly) obscure whatever spatial structure exists between sample
points. A region genuinely under-covered by training views should
visibly light up as the camera sweeps past viewing angles training never
saw; a well-covered region should stay dark throughout.

Per pixel, per frame, the query *direction* is the real direction from
that pixel's actual unprojected 3D point to the *current* camera position
-- not a single fixed direction reused across a whole sweep. A rendered
pixel always implies a specific viewing direction, and as the camera
moves, that implied direction sweeps through the full range an NBV/SLAM
system would actually query.

Real-depth-unprojection construction throughout (gsplat's own "ED"
expected-depth output, not an interpolated/sparse proxy).

Before rendering anything, this checks the checkpoint's own reconstruction
quality (held-out PSNR against `<scene_dir>/../eval` if that sibling
split exists, else a weaker training-view check) and refuses to proceed
below `--min-psnr` unless `--force` is passed -- a BQ uncertainty number
computed on a checkpoint nobody checked could actually reconstruct the
scene isn't trustworthy in either direction (see ROADMAP.md item 2 and
FINDINGS.md's bonsai/lego confound writeups).

Kernel family (`--kernel-family rbf|matern`) and bandwidth are exposed,
not hardcoded -- kernel flexibility is a strength of the method, not a
gap to close (ROADMAP.md item 4).

Needs torch + gsplat + Pillow.

Run: .venv-gsplat/bin/python gs_experiment/render_directional_uncertainty_sweep.py <scene_dir> --center 36 0 0 --radius 80
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

from bq_splat.kernels import DirectionalKernel
from gs_experiment.camera import translate_cameras, turntable_ring
from gs_experiment.nerf_transforms import fov_x_to_intrinsics, load_transforms
from gs_experiment.pixel_uncertainty import (
    LocalUncertaintyEngine,
    make_default_3d_matern_kernel,
    make_default_3d_position_kernel,
)
from gs_experiment.ply_io import read_3dgs_ply
from gs_experiment.splat_scene import load_from_gsplat_checkpoint, splat_observations

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def check_quality_gate(
    scene_dir: str, eval_dir: str | None, min_psnr: float, force: bool, background_color=(1.0, 1.0, 1.0)
) -> float:
    """Held-out PSNR if a sibling eval split exists (the trustworthy
    check -- see prepare_nerf_synthetic.py's `eval` condition), else a
    training-view PSNR with a loud caveat (can be inflated by
    overfitting/floaters, so it's a fallback, not an equal substitute).
    `background_color` must match what the checkpoint was actually
    trained against (NeRF-Synthetic scenes composite ground-truth photos
    onto white; get this wrong and PSNR is meaningless -- e.g. a real
    28.5dB lego checkpoint measures 1.8dB against the wrong background,
    a bug caught exactly this way rather than trusted as a real result).
    Raises SystemExit below min_psnr unless force=True.
    """
    from gs_experiment.render_reconstruction import render_views

    if eval_dir is None:
        candidate = Path(scene_dir).parent / "eval"
        eval_dir = str(candidate) if candidate.is_dir() else None

    if eval_dir is not None:
        _, eval_frames = load_transforms(str(Path(eval_dir) / "transforms.json"))
        import shutil

        eval_copy_dir = str(Path(scene_dir)) + "_eval_gate"
        Path(eval_copy_dir).mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(eval_dir) / "transforms.json", Path(eval_copy_dir) / "transforms.json")
        images_link = Path(eval_copy_dir) / "test"
        if not images_link.exists():
            images_link.symlink_to((Path(eval_dir) / "test").resolve())
        shutil.copy(Path(scene_dir) / "splats.ply", Path(eval_copy_dir) / "splats.ply")
        results, _ = render_views(eval_copy_dir, list(range(len(eval_frames))), background_color=background_color)
        kind = "held-out"
    else:
        _, frames = load_transforms(str(Path(scene_dir) / "transforms.json"))
        results, _ = render_views(scene_dir, list(range(len(frames))), background_color=background_color)
        kind = "TRAINING-VIEW (no sibling eval/ split found -- weaker check, treat cautiously)"

    psnr = float(np.mean([-10.0 * np.log10(max(float(np.mean((gt - recon) ** 2)), 1e-10)) for _, gt, recon in results]))
    print(f"quality gate: {kind} PSNR = {psnr:.2f}dB (threshold {min_psnr:.1f}dB)")
    if psnr < min_psnr:
        msg = (
            f"reconstruction quality gate FAILED ({psnr:.2f}dB < {min_psnr:.1f}dB) -- "
            "any uncertainty number computed here isn't trustworthy in either direction. "
            "Retrain with more views/iterations, or pass --force to proceed anyway."
        )
        if force:
            print(f"WARNING: {msg} Proceeding anyway (--force).")
        else:
            raise SystemExit(msg)
    return psnr


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
    h, w = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    us, vs = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    x_cam = (us - cx) / fx * depth
    y_cam = (vs - cy) / fy * depth
    cam_points = np.stack([x_cam, y_cam, depth, np.ones_like(depth)], axis=-1)
    world_points = cam_points @ c2w_cv.T
    return world_points[..., :3]


def run(
    scene_dir: str,
    center=None,
    n_frames: int = 60,
    radius: float | None = None,
    phi_deg: float = 35.0,
    fov_deg: float = 90.0,
    width: int = 640,
    height: int = 240,
    depth_width: int = 112,
    depth_height: int = 42,
    kernel_family: str = "rbf",
    bandwidth: float = 0.9,
    kappa: float = 4.0,
    window_radius: float = 1.6,
    min_opacity: float = 0.1,
    max_neighbors: int = 150,
    alpha_threshold: float = 0.5,
    background_color=(0.05, 0.05, 0.05),
    attribution_angular_tol: float = 0.01,
    device: str = "cuda",
    output_name: str = "directional_uncertainty_sweep",
    eval_dir: str | None = None,
    min_psnr: float = 20.0,
    force: bool = False,
    gate_background_color=(1.0, 1.0, 1.0),
):
    import gsplat

    check_quality_gate(scene_dir, eval_dir, min_psnr, force, background_color=gate_background_color)

    ck = read_3dgs_ply(f"{scene_dir}/splats.ply")
    scene = load_from_gsplat_checkpoint(scene_dir, attribution_angular_tol=attribution_angular_tol)
    obs_positions, obs_directions, obs_values = splat_observations(scene)
    bounds = tuple((obs_positions[:, d].min() - 1.0, obs_positions[:, d].max() + 1.0) for d in range(3))

    # Auto-frame from the checkpoint's own splat extent unless the caller
    # overrides -- a fixed camera radius/center is scene-scale-specific
    # (found the hard way: a radius tuned for one scene's world units left
    # a NeRF-Synthetic-scale object, extent ~1-2 units, a tiny speck in an
    # otherwise-empty frame), and this tool needs to work on an arbitrary
    # checkpoint without per-scene tuning.
    scene_center = ck["positions"].mean(axis=0)
    scene_radius = float(np.percentile(np.linalg.norm(ck["positions"] - scene_center, axis=1), 95))
    if center is None:
        center = scene_center
    center = np.asarray(center, dtype=float)
    if radius is None:
        radius = max(2.5 * scene_radius, 1e-3)
        print(f"auto-framing: scene extent ~{scene_radius:.2f} -> camera radius {radius:.2f}, center {center}")

    if kernel_family == "rbf":
        pos_kernel = make_default_3d_position_kernel(sigma=bandwidth)
    elif kernel_family == "matern":
        pos_kernel = make_default_3d_matern_kernel(rho=bandwidth)
    else:
        raise ValueError(f"unknown kernel_family {kernel_family!r}, expected 'rbf' or 'matern'")
    dir_kernel = DirectionalKernel(kappa=kappa)
    engine = LocalUncertaintyEngine(
        positions=obs_positions, values=obs_values, pos_kernel=pos_kernel, scene_bounds=bounds,
        directions=obs_directions, dir_kernel=dir_kernel, max_neighbors=max_neighbors,
    )

    t0 = time.time()
    dummy_dir = np.array([0.0, 0.0, 1.0])
    for p in obs_positions[:30]:
        engine.directional_variance(p, dummy_dir, window_radius)
        engine.spatial_only_variance(p, window_radius)
    per_query_s = (time.time() - t0) / 30
    total_queries = depth_width * depth_height * n_frames
    print(
        f"measured {per_query_s * 1000:.2f} ms/pixel (both BQ solves, {kernel_family} kernel, "
        f"max_neighbors={max_neighbors}); {depth_width}x{depth_height} x {n_frames} frames = {total_queries} pixels "
        f"-> est. {total_queries * per_query_s / 60:.1f} min total"
    )

    all_positions = torch.tensor(ck["positions"], dtype=torch.float32, device=device)
    all_scales = torch.tensor(ck["scales"], dtype=torch.float32, device=device)
    all_rotations = torch.tensor(ck["rotations"], dtype=torch.float32, device=device)
    all_opacities = torch.tensor(ck["opacities"], dtype=torch.float32, device=device)
    all_sh = torch.tensor(ck["sh_coeffs"], dtype=torch.float32, device=device).transpose(1, 2)
    sh_degree = ck["sh_degree"]
    background = torch.tensor(background_color, dtype=torch.float32, device=device)
    K_display = fov_x_to_intrinsics(np.deg2rad(fov_deg), width, height)
    K_depth = fov_x_to_intrinsics(np.deg2rad(fov_deg), depth_width, depth_height)

    cameras = translate_cameras(turntable_ring(radius=radius, n_views=n_frames, phi_deg=phi_deg), center)

    frames_dir = RESULTS_DIR / f"{output_name}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []

    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad(color=background_color)

    with torch.no_grad():
        for i, cam in enumerate(cameras):
            c2w = c2w_from_camera_pose(cam)
            viewmat_np = opencv_viewmat_from_c2w(c2w)
            viewmat = torch.tensor(viewmat_np, dtype=torch.float32, device=device)
            c2w_cv = np.linalg.inv(viewmat_np)

            rendered, _, _ = gsplat.rasterization(
                all_positions, all_rotations, all_scales, all_opacities, all_sh, viewmat[None],
                torch.tensor(K_display, dtype=torch.float32, device=device)[None],
                width=width, height=height, sh_degree=sh_degree, backgrounds=background,
            )
            recon = rendered[0].clamp(0, 1).cpu().numpy()

            rendered_lo, alpha_lo, _ = gsplat.rasterization(
                all_positions, all_rotations, all_scales, all_opacities, all_sh, viewmat[None],
                torch.tensor(K_depth, dtype=torch.float32, device=device)[None],
                width=depth_width, height=depth_height, sh_degree=sh_degree, render_mode="ED",
            )
            depth_map = rendered_lo[0, ..., 0].cpu().numpy()
            alpha_map = alpha_lo[0, ..., 0].cpu().numpy()
            valid = alpha_map > alpha_threshold

            world_points = unproject_depth_grid(depth_map, K_depth, c2w_cv)

            dir_field = np.full((depth_height, depth_width), np.nan)
            spatial_field = np.full((depth_height, depth_width), np.nan)
            ys, xs = np.where(valid)
            cam_center = cam.center
            for y, x in zip(ys, xs):
                point = world_points[y, x]
                to_camera = cam_center - point
                query_direction = to_camera / np.linalg.norm(to_camera)
                dir_field[y, x] = engine.directional_variance(point, query_direction, window_radius).variance
                spatial_field[y, x] = engine.spatial_only_variance(point, window_radius).variance

            def upsample(field):
                field_img = Image.fromarray(np.nan_to_num(field, nan=0.0).astype(np.float32), mode="F")
                valid_img = Image.fromarray((valid * 255).astype(np.uint8))
                field_up = np.array(field_img.resize((width, height), Image.BILINEAR))
                valid_up = np.array(valid_img.resize((width, height), Image.NEAREST)) > 127
                return np.where(valid_up, field_up, np.nan)

            dir_up = upsample(dir_field)
            spatial_up = upsample(spatial_field)

            if i == 0:
                dir_vmax = np.nanpercentile(dir_up, 95)
                spatial_vmax = np.nanpercentile(spatial_up, 95)

            fig, axes = plt.subplots(3, 1, figsize=(9, 9))
            axes[0].imshow(recon)
            axes[0].set_title("reconstruction", fontsize=10)
            axes[0].axis("off")

            im0 = axes[1].imshow(spatial_up, cmap=cmap, vmin=0, vmax=spatial_vmax)
            axes[1].set_title(
                f"spatial (quadrature) BQ uncertainty -- how poorly the finite splat set\n"
                "resolves this region, independent of viewing direction", fontsize=8,
            )
            axes[1].axis("off")
            fig.colorbar(im0, ax=axes[1], fraction=0.046, pad=0.04)

            im1 = axes[2].imshow(dir_up, cmap=cmap, vmin=0, vmax=dir_vmax)
            axes[2].set_title(
                f"directional (epistemic) BQ uncertainty (per-pixel, {depth_width}x{depth_height} real ray-hits)\n"
                "query direction = real direction from each point to THIS frame's camera", fontsize=8,
            )
            axes[2].axis("off")
            fig.colorbar(im1, ax=axes[2], fraction=0.046, pad=0.04)

            fig.tight_layout(pad=0.3)
            frame_path = frames_dir / f"frame_{i:03d}.png"
            fig.savefig(frame_path, dpi=120)
            plt.close(fig)
            frame_paths.append(frame_path)
            print(f"rendered frame {i + 1}/{n_frames}")

    images = [Image.open(p).convert("RGB") for p in frame_paths]
    out_path = RESULTS_DIR / f"{output_name}.gif"
    images[0].save(out_path, save_all=True, append_images=images[1:], duration=100, loop=0, optimize=True)
    print(f"\nSaved {out_path} ({len(images)} frames)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene_dir")
    parser.add_argument("--center", type=float, nargs=3, default=None, help="default: auto, from the checkpoint's own splat extent")
    parser.add_argument("--n-frames", type=int, default=60)
    parser.add_argument("--radius", type=float, default=None, help="default: auto, from the checkpoint's own splat extent")
    parser.add_argument("--phi-deg", type=float, default=35.0)
    parser.add_argument("--fov-deg", type=float, default=90.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--depth-width", type=int, default=112)
    parser.add_argument("--depth-height", type=int, default=42)
    parser.add_argument("--kernel-family", choices=["rbf", "matern"], default="rbf")
    parser.add_argument("--bandwidth", type=float, default=0.9, help="sigma (rbf) or rho (matern)")
    parser.add_argument("--kappa", type=float, default=4.0)
    parser.add_argument("--window-radius", type=float, default=1.6)
    parser.add_argument("--max-neighbors", type=int, default=150)
    parser.add_argument("--output-name", default="directional_uncertainty_sweep")
    parser.add_argument("--eval-dir", default=None, help="held-out split for the quality gate; default: <scene_dir>/../eval if present")
    parser.add_argument("--min-psnr", type=float, default=20.0, help="quality gate threshold")
    parser.add_argument("--force", action="store_true", help="proceed even if the quality gate fails")
    parser.add_argument(
        "--gate-background-color", type=float, nargs=3, default=[1.0, 1.0, 1.0],
        help="must match what the checkpoint was trained against (NeRF-Synthetic: white, the default); wrong value makes the PSNR gate meaningless",
    )
    args = parser.parse_args()
    run(
        args.scene_dir, center=(tuple(args.center) if args.center is not None else None), n_frames=args.n_frames,
        radius=args.radius, phi_deg=args.phi_deg,
        fov_deg=args.fov_deg, width=args.width, height=args.height, depth_width=args.depth_width, depth_height=args.depth_height,
        kernel_family=args.kernel_family, bandwidth=args.bandwidth, kappa=args.kappa, window_radius=args.window_radius,
        max_neighbors=args.max_neighbors, output_name=args.output_name, eval_dir=args.eval_dir, min_psnr=args.min_psnr,
        force=args.force, gate_background_color=tuple(args.gate_background_color),
    )


if __name__ == "__main__":
    main()
