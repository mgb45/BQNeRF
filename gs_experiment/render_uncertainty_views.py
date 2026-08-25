"""Per-camera-view uncertainty renders: the RGB reconstruction next to BQ
uncertainty projected into the *same* image plane, for a real trained
checkpoint.

Unlike differentiation_experiment.py's top-down 2D-slice grid (an
abstract world-space view, not tied to any camera), this script queries
BQ variance directly at real splat positions near a chosen camera and
projects those query points into that camera's own pixel coordinates --
what an eventual pixel-level reprojection (still deferred, see
splat_scene.py's module docstring) will need for a full per-pixel field,
approximated here at the splat level rather than left undone. The
position+direction query direction is the direction each splat is
*actually seen from by this camera* (via camera.
directions_from_positions_to_camera), not the single global
"discriminating direction" the top-down analysis uses -- a more natural,
per-view question ("how uncertain is BQ about what this camera would see
here") than a fixed direction picked for cross-zone comparison.

Needs torch + gsplat (requirements-gsplat.txt).

Run: .venv-gsplat/bin/python gs_experiment/render_uncertainty_views.py <scene_dir> \
    --view-indices 0 45 --zone-centers 0,0,0 18,0,0
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

from bq_splat.kernels import DirectionalKernel
from gs_experiment.camera import directions_from_positions_to_camera
from gs_experiment.nerf_transforms import camera_pose_from_c2w, fov_x_to_intrinsics, load_transforms, opencv_viewmat_from_c2w
from gs_experiment.pixel_uncertainty import LocalUncertaintyEngine, make_default_3d_position_kernel
from gs_experiment.render_reconstruction import render_views
from gs_experiment.splat_scene import load_from_gsplat_checkpoint, splat_observations

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def project_to_pixels(positions: np.ndarray, viewmat: np.ndarray, K: np.ndarray):
    """positions (N,3) world-space -> (pixels (N,2), in_front (N,) bool).
    pixels are NaN where in_front is False (behind the camera)."""
    n = positions.shape[0]
    homog = np.concatenate([positions, np.ones((n, 1))], axis=1)
    cam = (viewmat @ homog.T).T  # (n, 4), OpenCV camera-space
    depth = cam[:, 2]
    in_front = depth > 1e-4
    proj = (K @ cam[:, :3].T).T
    pixels = np.full((n, 2), np.nan)
    pixels[in_front, 0] = proj[in_front, 0] / proj[in_front, 2]
    pixels[in_front, 1] = proj[in_front, 1] / proj[in_front, 2]
    return pixels, in_front


def render_uncertainty_views(
    scene_dir: str,
    view_indices,
    zone_centers,
    zone_radius: float = 1.6,
    sigma: float = 0.9,
    kappa: float = 4.0,
    window_radius: float = 1.6,
    max_points_per_view: int = 500,
    min_opacity_for_display: float = 0.05,
    attribution_angular_tol: float = 0.01,
    seed: int = 0,
    device: str = "cuda",
):
    scene = load_from_gsplat_checkpoint(scene_dir, attribution_angular_tol=attribution_angular_tol)
    positions, directions, values = splat_observations(scene)

    pos_margin = 1.0
    bounds = tuple(
        (positions[:, d].min() - pos_margin, positions[:, d].max() + pos_margin) for d in range(3)
    )
    pos_kernel = make_default_3d_position_kernel(sigma=sigma)
    dir_kernel = DirectionalKernel(kappa=kappa)
    engine = LocalUncertaintyEngine(
        positions=positions, values=values, pos_kernel=pos_kernel, scene_bounds=bounds,
        directions=directions, dir_kernel=dir_kernel,
    )

    rgb_results, checkpoint = render_views(scene_dir, view_indices, device=device)
    camera_angle_x, frames = load_transforms(str(Path(scene_dir) / "transforms.json"))
    height, width = rgb_results[0][1].shape[:2]
    K = fov_x_to_intrinsics(camera_angle_x, width, height)

    rng = np.random.default_rng(seed)
    splat_positions = checkpoint["positions"]
    splat_opacities = checkpoint["opacities"]
    zone_centers = [np.asarray(c, dtype=float) for c in zone_centers]

    fig, axes = plt.subplots(len(view_indices), 3, figsize=(13, 4.2 * len(view_indices)))
    if len(view_indices) == 1:
        axes = axes[None, :]

    for row, (view_idx, (_, gt, recon)) in enumerate(zip(view_indices, rgb_results)):
        c2w = frames[view_idx][1]
        viewmat = opencv_viewmat_from_c2w(c2w)
        cam_pose = camera_pose_from_c2w(c2w)

        center = min(zone_centers, key=lambda c: np.linalg.norm(cam_pose.center - c))
        near = np.linalg.norm(splat_positions - center, axis=1) < zone_radius
        near &= splat_opacities > min_opacity_for_display
        near_idx = np.where(near)[0]
        if len(near_idx) > max_points_per_view:
            near_idx = rng.choice(near_idx, size=max_points_per_view, replace=False)

        query_positions = splat_positions[near_idx]
        query_dirs = directions_from_positions_to_camera(query_positions, cam_pose)

        pos_var = np.array([engine.spatial_only_variance(p, window_radius).variance for p in query_positions])
        dir_var = np.array(
            [engine.directional_variance(p, d, window_radius).variance for p, d in zip(query_positions, query_dirs)]
        )
        pixels, in_front = project_to_pixels(query_positions, viewmat, K)
        in_view = in_front & (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)

        axes[row, 0].imshow(recon)
        axes[row, 0].set_title(f"view {view_idx}: reconstruction" if row == 0 else f"view {view_idx}")
        axes[row, 0].axis("off")

        for ax, var, title in [
            (axes[row, 1], pos_var, "position-only BQ variance"),
            (axes[row, 2], dir_var, "position+direction BQ variance"),
        ]:
            ax.imshow(recon, alpha=0.7)
            sc = ax.scatter(
                pixels[in_view, 0], pixels[in_view, 1], c=var[in_view], cmap="inferno", s=14,
                edgecolors="white", linewidths=0.3,
            )
            ax.set_xlim(0, width)
            ax.set_ylim(height, 0)
            ax.set_title(title if row == 0 else "", fontsize=10)
            ax.axis("off")
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Per-view RGB + splat-projected BQ uncertainty ({scene_dir})")
    fig.tight_layout()
    out = RESULTS_DIR / "uncertainty_views.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene_dir")
    parser.add_argument("--view-indices", type=int, nargs="+", default=[0, 45])
    parser.add_argument(
        "--zone-centers", type=str, nargs="+", default=["0,0,0", "18,0,0"],
        help="comma-separated x,y,z per zone center; each view is matched to its nearest one",
    )
    parser.add_argument("--zone-radius", type=float, default=1.6)
    parser.add_argument("--angular-tol", type=float, default=0.01)
    args = parser.parse_args()

    zone_centers = [tuple(float(v) for v in s.split(",")) for s in args.zone_centers]
    render_uncertainty_views(
        args.scene_dir, args.view_indices, zone_centers, zone_radius=args.zone_radius,
        attribution_angular_tol=args.angular_tol,
    )


if __name__ == "__main__":
    main()
