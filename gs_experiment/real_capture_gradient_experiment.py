"""The directional-uncertainty-gradient experiment
(`directional_gradient_experiment.py`, real-object follow-up:
`real_directional_gradient_experiment.py`) on a genuinely real *captured*
scene -- a photographed object with COLMAP-estimated poses (the kind of
scene GAVIS/PUP 3D-GS actually report numbers on: Mip-NeRF360 /
Tanks & Temples), not a synthetic Blender render. Every prior real-data
result in this project (lego and the rest of the 8 NeRF-Synthetic scenes)
used synthetic, exactly-known camera poses; this is the first scene where
poses are themselves an estimate from real photographs, via
`colmap_loader` -- a new source of error (SfM pose noise, unmodeled lens
distortion, real image noise/exposure variation) no prior experiment in
this project has had to contend with.

Scene: Mip-NeRF360 "bonsai" (`nvs-bench/mipnerf360` on Hugging Face) --
292 real photographs of a bonsai plant, COLMAP PINHOLE-camera poses.
Camera centers' centroid sits close to the world origin (checked, not
assumed), matching the object-centered convention every other scene in
this project already uses.

Same construction as `real_directional_gradient_experiment.py`: 5
equal-view-count conditions of increasing real angular spread around a
shared reference view (`prepare_nerf_synthetic.select_gradient_subset`,
which only needs a (file_path, c2w) frame list -- works unchanged on
COLMAP-derived frames), each trained into its own real checkpoint, BQ
directional/spatial-only variance queried at a fixed point (world origin)
and fixed real query direction across all 5.

Needs torch + gsplat (training) and PIL (image resizing).

Run: .venv-gsplat/bin/python gs_experiment/real_capture_gradient_experiment.py <colmap_scene_dir> <out_dir>
  e.g. gs_experiment/local_runs/mipnerf360_raw/bonsai gs_experiment/local_runs/bonsai_prepared
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from gs_experiment.colmap_loader import load_colmap_scene
from gs_experiment.nerf_transforms import write_transforms_json
from gs_experiment.prepare_nerf_synthetic import select_gradient_subset, write_condition

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_FRACTIONS = [0.06, 0.15, 0.35, 0.6, 1.0]
N_PER_ZONE = 8
REFERENCE_IDX = 0
TARGET_WIDTH = 400

SIGMA = 0.05
WINDOW_RADIUS = 0.8
KAPPA = 4.0
# World origin (COLMAP's own SfM-chosen frame) is NOT reliably where the
# photographed object actually sits -- checked directly, not assumed: the
# nearest real splat to the origin was consistently ~0.8-0.9 units away
# with only a handful of real neighbors within a 0.3 window, in every one
# of the 5 checkpoints. A camera-center centroid near the origin (true
# here) does not imply the object itself is there. QUERY_POINT below is
# the mean, across all 5 checkpoints, of each checkpoint's own median
# high-opacity splat position -- a real, data-derived "where the object
# actually is" estimate, not assumed from dataset convention the way
# NeRF-Synthetic's object-centered origin could be.
QUERY_POINT = np.array([0.612, 1.174, 1.507])

TRAIN_KWARGS = dict(
    n_splats=2000, sh_degree=1, n_iters=3000, seed=0,
    init_scale=0.03, opacity_reg_weight=0.01, densify=True, densify_interval=300, densify_start=300,
    min_opacity=0.005, max_splats=15000, log_every=500,
)


def prepare_images(colmap_scene_dir: str, out_dir: str):
    camera_angle_x, frames = load_colmap_scene(os.path.join(colmap_scene_dir, "sparse", "0"))
    src_images_dir = os.path.join(colmap_scene_dir, "images")

    all_dir = os.path.join(out_dir, "all")
    os.makedirs(all_dir, exist_ok=True)

    kept_frames = []
    for file_stem, c2w in frames:
        src = os.path.join(src_images_dir, file_stem + ".JPG")
        if not os.path.exists(src):
            src = os.path.join(src_images_dir, file_stem + ".jpg")
        if not os.path.exists(src):
            continue
        im = Image.open(src).convert("RGB")
        w, h = im.size
        target_height = round(TARGET_WIDTH * h / w)
        im = im.resize((TARGET_WIDTH, target_height), Image.LANCZOS)
        dst = os.path.join(all_dir, file_stem + ".png")
        im.save(dst)
        kept_frames.append((f"all/{file_stem}", c2w))

    if len(kept_frames) < len(frames):
        print(f"{len(kept_frames)}/{len(frames)} frames had a matching image file (rest skipped)")

    write_transforms_json(
        os.path.join(out_dir, "all_transforms.json"), camera_angle_x,
        [{"file_path": fp, "transform_matrix": c2w} for fp, c2w in kept_frames],
    )
    return camera_angle_x, kept_frames


def build_conditions(out_dir: str, camera_angle_x: float, frames):
    centers = np.array([c2w[:3, 3] for _, c2w in frames])
    dirs = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    print(f"camera center centroid (should be near origin): {centers.mean(axis=0)}")

    ref_dir = dirs[REFERENCE_IDX]
    query_direction = dirs[np.argmin(dirs @ ref_dir)]

    zone_dirs, spreads = [], []
    for i, window_fraction in enumerate(WINDOW_FRACTIONS):
        idx = select_gradient_subset(frames, n_per_zone=N_PER_ZONE, window_fraction=window_fraction, reference_idx=REFERENCE_IDX)
        # write_condition expects frame file_paths relative to out_dir and
        # symlinks split_prefix -- our images already live under
        # "all/", referenced via that same relative path in `frames`, so
        # split_prefix="all" points write_condition's symlink at the
        # directory that's already there.
        zone_dir = write_condition(out_dir, camera_angle_x, frames, idx, f"gradient_{i}", "all")
        zone_dirs.append(zone_dir)
        min_sim = float((dirs[idx] @ ref_dir).min())
        spreads.append(np.degrees(np.arccos(np.clip(min_sim, -1.0, 1.0))))

    return zone_dirs, np.array(spreads), query_direction


def train_zones(zone_dirs, bounds):
    from gs_experiment.train_minimal_gsplat import train

    ply_paths = []
    for zone_dir in zone_dirs:
        ply_path = os.path.join(zone_dir, "splats.ply")
        if not os.path.exists(ply_path):
            print(f"training {zone_dir}...")
            train(zone_dir, ply_path, bounds=bounds, **TRAIN_KWARGS)
        else:
            print(f"reusing existing checkpoint at {ply_path}")
        ply_paths.append(ply_path)
    return ply_paths


def analyze(zone_dirs, spreads_deg, query_direction):
    from bq_splat.kernels import DirectionalKernel
    from gs_experiment.pixel_uncertainty import LocalUncertaintyEngine, make_default_3d_position_kernel
    from gs_experiment.splat_scene import load_from_gsplat_checkpoint, splat_observations

    directional_vars, spatial_vars, n_neighbors_list = [], [], []
    query_point = QUERY_POINT

    for zone_dir in zone_dirs:
        scene = load_from_gsplat_checkpoint(zone_dir, attribution_angular_tol=0.01)
        positions, directions, values = splat_observations(scene)
        bounds3d = tuple((positions[:, d].min() - 1.0, positions[:, d].max() + 1.0) for d in range(3))

        pos_kernel = make_default_3d_position_kernel(sigma=SIGMA)
        dir_kernel = DirectionalKernel(kappa=KAPPA)
        engine = LocalUncertaintyEngine(
            positions=positions, values=values, pos_kernel=pos_kernel, scene_bounds=bounds3d,
            directions=directions, dir_kernel=dir_kernel,
        )
        n_neighbors_list.append(len(engine.local_neighbors(query_point, WINDOW_RADIUS)))
        dir_result = engine.directional_variance(query_point, query_direction, WINDOW_RADIUS)
        spatial_result = engine.spatial_only_variance(query_point, WINDOW_RADIUS)
        directional_vars.append(dir_result.variance)
        spatial_vars.append(spatial_result.variance)

    directional_vars = np.array(directional_vars)
    spatial_vars = np.array(spatial_vars)

    order = np.argsort(spreads_deg)
    print(f"\n{'zone':>4}{'spread (deg)':>14}{'n_neighbors':>13}{'directional var':>18}{'spatial-only var':>18}")
    for i in order:
        print(f"{i:>4}{spreads_deg[i]:>14.1f}{n_neighbors_list[i]:>13}{directional_vars[i]:>18.5f}{spatial_vars[i]:>18.5f}")

    dir_sorted = directional_vars[order]
    is_monotonic = bool(np.all(np.diff(dir_sorted) <= 1e-12))
    rho = float(np.corrcoef(np.argsort(np.argsort(spreads_deg)), np.argsort(np.argsort(-directional_vars)))[0, 1])
    print(f"\ndirectional variance monotonically decreasing as spread widens: {is_monotonic}")
    print(f"rank correlation (spread vs. directional variance): rho={rho:.3f}")
    print(f"directional variance range (narrowest/widest): {dir_sorted[0] / max(dir_sorted[-1], 1e-12):.2f}x")
    print(f"spatial-only variance range (control): {spatial_vars.max() / max(spatial_vars.min(), 1e-12):.2f}x")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(spreads_deg[order], directional_vars[order], "o-", color="tab:red", label="directional variance")
    ax1.set_xlabel("real angular spread of selected views (deg, min similarity to reference)")
    ax1.set_ylabel("position+direction BQ variance", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(spreads_deg[order], spatial_vars[order], "s--", color="tab:blue", label="spatial-only variance (control)")
    ax2.set_ylabel("position-only BQ variance", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    fig.suptitle("Real captured scene (Mip-NeRF360 bonsai): directional BQ variance\nvs. real view-coverage spread")
    fig.tight_layout()
    out_path = RESULTS_DIR / "real_capture_directional_gradient.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_path}")


def run(colmap_scene_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    all_transforms_path = os.path.join(out_dir, "all_transforms.json")
    if os.path.exists(all_transforms_path):
        from gs_experiment.nerf_transforms import load_transforms

        print("reusing already-prepared images")
        camera_angle_x, frames = load_transforms(all_transforms_path)
    else:
        camera_angle_x, frames = prepare_images(colmap_scene_dir, out_dir)

    centers = np.array([c2w[:3, 3] for _, c2w in frames])
    margin = 1.0
    bounds = tuple((float(centers[:, d].min() - margin), float(centers[:, d].max() + margin)) for d in range(3))
    print(f"scene bounds (from camera-center extent + margin): {bounds}")

    zone_dirs, spreads_deg, query_direction = build_conditions(out_dir, camera_angle_x, frames)
    train_zones(zone_dirs, bounds)
    analyze(zone_dirs, spreads_deg, query_direction)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("colmap_scene_dir")
    parser.add_argument("out_dir")
    args = parser.parse_args()
    run(args.colmap_scene_dir, args.out_dir)


if __name__ == "__main__":
    main()
