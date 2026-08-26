"""A realistic experiment with views deliberately chosen to produce a
view-direction *uncertainty gradient*, not the binary wide-vs-narrow split
every prior directional result in this project used (toy scale:
bq_splat/results/FINDINGS.md section 9; real scale: gs_experiment/
results/FINDINGS.md sections 17-19, 22). `scene_spec.gradient_scene`
builds 5 identical thin-rod clusters (spatial density held equal across
zones, isolating the directional effect the same way `differentiation_scene`
and `validate_directional_combined.py` already do) along a line, each
observed by its own turntable-arc camera rig, all centered on the *same*
azimuth but with angular half-width increasing linearly from zone 0
(narrowest, most under-covered) to the last zone (widest, effectively a
full ring) -- a real, monotonic angular-coverage gradient, real Blender
rendering, real gsplat training, not a synthetic/toy signal.

A single fixed query direction -- the azimuth diametrically opposite the
shared arc center -- is genuinely consistent across every zone (since
`theta_center_deg` doesn't vary, unlike each zone's own local convention),
computed the same robust way `differentiation_experiment.py`'s real-scene
builder does (a real camera pose's direction-to-a-point, not hand-derived
spherical trigonometry -- see that module's comment about a real
elevation bug from doing it the naive way).

Pipeline (three separate steps, since Blender's `bpy` can only run inside
a Blender process):
  1. `.venv-gsplat/bin/python -m gs_experiment.directional_gradient_experiment prepare <out_dir>`
     -- builds the scene spec + zone metadata, writes JSON.
  2. `blender --background --python gs_experiment/blender_render.py -- <out_dir>/scene_spec.json <out_dir>`
     -- real rendering (see that module's docstring).
  3. `.venv-gsplat/bin/python -m gs_experiment.directional_gradient_experiment train-and-analyze <out_dir>`
     -- real gsplat training with densification, then queries directional
     and position-only BQ variance at each zone's center, reporting
     whether variance actually rises monotonically with the designed
     coverage gradient.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from gs_experiment.camera import directions_from_positions_to_camera, translate_camera, turntable_camera
from gs_experiment.scene_spec import gradient_scene

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def prepare(out_dir: str, n_zones: int = 5, n_views_per_zone: int = 10, radius: float = 6.5):
    os.makedirs(out_dir, exist_ok=True)
    spec, info = gradient_scene(n_zones=n_zones, n_views_per_zone=n_views_per_zone, radius=radius)

    with open(os.path.join(out_dir, "scene_spec.json"), "w") as fh:
        json.dump(spec.to_json_dict(), fh, indent=2)

    np.savez(
        os.path.join(out_dir, "gradient_info.npz"),
        zone_centers=info["zone_centers"], half_widths_deg=info["half_widths_deg"],
        zone_camera_ranges=np.array(info["zone_camera_ranges"]),
        theta_center_deg=info["theta_center_deg"], query_theta_deg=info["query_theta_deg"], radius=radius,
    )
    print(f"wrote {len(spec.objects)} objects, {len(spec.cameras)} cameras across {n_zones} zones to {out_dir}")
    print(f"half-widths (deg): {info['half_widths_deg']}")
    print(f"shared query azimuth: {info['query_theta_deg']:.1f} deg")
    print(f"\nNext: blender --background --python gs_experiment/blender_render.py -- "
          f"{out_dir}/scene_spec.json {out_dir}")


def train_and_analyze(
    out_dir: str, n_splats: int = 1500, n_iters: int = 3000, seed: int = 0,
    window_radius: float = 1.6, sigma: float = 0.9, kappa: float = 4.0, min_opacity: float = 0.1,
):
    from bq_splat.kernels import DirectionalKernel
    from gs_experiment.pixel_uncertainty import LocalUncertaintyEngine, make_default_3d_position_kernel
    from gs_experiment.splat_scene import load_from_gsplat_checkpoint, splat_observations
    from gs_experiment.train_minimal_gsplat import train

    info = np.load(os.path.join(out_dir, "gradient_info.npz"))
    zone_centers = info["zone_centers"]
    half_widths_deg = info["half_widths_deg"]
    query_theta_deg = float(info["query_theta_deg"])
    radius = float(info["radius"])
    n_zones = len(zone_centers)

    ply_path = os.path.join(out_dir, "splats.ply")
    if not os.path.exists(ply_path):
        print("training...")
        span = float(zone_centers[:, 0].max() - zone_centers[:, 0].min())
        bounds = ((-2.0, span + 2.0), (-2.5, 2.5), (-2.5, 2.5))
        train(
            out_dir, ply_path, n_splats=n_splats, bounds=bounds, sh_degree=1, n_iters=n_iters, seed=seed,
            init_scale=0.1, opacity_reg_weight=0.003, densify=True, densify_interval=300, densify_start=300,
            densify_grad_percentile=80.0, min_opacity=0.005, max_splats=8000, log_every=500,
        )
    else:
        print(f"reusing existing checkpoint at {ply_path}")

    scene = load_from_gsplat_checkpoint(out_dir, attribution_angular_tol=0.01)
    positions, directions, values = splat_observations(scene)
    bounds3d = tuple((positions[:, d].min() - 1.0, positions[:, d].max() + 1.0) for d in range(3))

    pos_kernel = make_default_3d_position_kernel(sigma=sigma)
    dir_kernel = DirectionalKernel(kappa=kappa)
    engine = LocalUncertaintyEngine(
        positions=positions, values=values, pos_kernel=pos_kernel, scene_bounds=bounds3d,
        directions=directions, dir_kernel=dir_kernel,
    )

    directional_vars, spatial_vars = [], []
    for center in zone_centers:
        query_cam = translate_camera(turntable_camera(radius, 35.0, query_theta_deg), center)
        query_direction = directions_from_positions_to_camera(center.reshape(1, -1), query_cam)[0]

        dir_result = engine.directional_variance(center, query_direction, window_radius)
        spatial_result = engine.spatial_only_variance(center, window_radius)
        directional_vars.append(dir_result.variance)
        spatial_vars.append(spatial_result.variance)

    directional_vars = np.array(directional_vars)
    spatial_vars = np.array(spatial_vars)

    print(f"\n{'zone':>4}{'half-width (deg)':>18}{'directional var':>18}{'spatial-only var':>18}")
    for i in range(n_zones):
        print(f"{i:>4}{half_widths_deg[i]:>18.1f}{directional_vars[i]:>18.5f}{spatial_vars[i]:>18.5f}")

    is_monotonic = bool(np.all(np.diff(directional_vars) <= 1e-12))
    spearman_rho = float(np.corrcoef(np.argsort(np.argsort(-half_widths_deg)), np.argsort(np.argsort(directional_vars)))[0, 1])
    spatial_range_ratio = float(spatial_vars.max() / max(spatial_vars.min(), 1e-12))

    print(f"\ndirectional variance strictly monotonically decreasing with half-width: {is_monotonic}")
    print(f"rank correlation (narrowing half-width vs. rising directional variance): rho={spearman_rho:.3f}")
    print(f"directional variance range (narrowest/widest zone): {directional_vars[0] / max(directional_vars[-1], 1e-12):.2f}x")
    print(f"spatial-only variance max/min across zones (should be small -- geometry is matched): {spatial_range_ratio:.2f}x")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(half_widths_deg, directional_vars, "o-", color="tab:red", label="directional variance")
    ax1.set_xlabel("zone's camera-arc half-width (deg) -- designed coverage gradient")
    ax1.set_ylabel("position+direction BQ variance", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.plot(half_widths_deg, spatial_vars, "s--", color="tab:blue", label="spatial-only variance (control)")
    ax2.set_ylabel("position-only BQ variance", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    fig.suptitle("Directional BQ variance vs. a designed view-coverage gradient\n(5 zones, identical geometry, shared query direction)")
    fig.tight_layout()
    out_path = RESULTS_DIR / "directional_gradient.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prepare = sub.add_parser("prepare")
    p_prepare.add_argument("out_dir")
    p_prepare.add_argument("--n-zones", type=int, default=5)
    p_prepare.add_argument("--n-views-per-zone", type=int, default=10)

    p_train = sub.add_parser("train-and-analyze")
    p_train.add_argument("out_dir")
    p_train.add_argument("--n-iters", type=int, default=3000)
    p_train.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    if args.cmd == "prepare":
        prepare(args.out_dir, n_zones=args.n_zones, n_views_per_zone=args.n_views_per_zone)
    elif args.cmd == "train-and-analyze":
        train_and_analyze(args.out_dir, n_iters=args.n_iters, seed=args.seed)


if __name__ == "__main__":
    main()
