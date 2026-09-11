"""Visual comparison figure for the renderer-consistent sparse-GP
decomposition ("practical first implementation" -- see
gs_experiment/sh_directional_uncertainty.py and
gs_experiment/gpu_sh_directional_uncertainty.py's own module docstrings):

    mu_q = C_alpha(q)                                    (real renderer mean, untouched)
    u_q  = u_spatial_BQ(q) + u_SH(q)
         = u_spatial_BQ(q) + sum_i beta_{q,i}^2 * phi(d_q)^T Sigma_theta_i phi(d_q)

For a few real held-out views on real `wide` checkpoints, shows ground
truth, the real alpha-compositing reconstruction (mu_q, unmodified), and
the two uncertainty terms plus their sum:

  1. C_alpha        -- the real renderer's own output (`render_views`).
  2. u_spatial_BQ    -- the finite-spatial-representation term: the real
                        alpha weights' own RKHS risk under a position-only
                        kernel (`gpu_uncertainty.compute_alpha_risk_batched`).
  3. u_SH            -- the new directional term: each candidate splat's
                        own SH-coefficient posterior uncertainty at the
                        query viewing direction, propagated through the
                        real alpha weights (`gpu_sh_directional_uncertainty.
                        accumulate_sh_precision` + `compute_sh_directional_
                        uncertainty_batched`).
  4. u_spatial_BQ + u_SH -- their sum, the full posterior variance under
                        this decomposition.

Reuses this project's already-tested machinery throughout
(`render_reconstruction.render_views`/`_render_and_unproject`,
`splat_scene.load_from_gsplat_checkpoint`/`fit_kernel_hyperparams`) --
nothing here is a new rendering implementation, only a new per-pixel
comparison layout.

`lam` (the SH-coefficient prior precision) is a free hyperparameter, not
fit here -- see sh_directional_uncertainty.py's own docstring for why a
single scalar is the deliberately simple "practical first implementation"
choice, not a new hyperparameter search.

Needs torch + gsplat (requirements-gsplat.txt).

Run: .venv-gsplat/bin/python gs_experiment/scripts/render_sparse_gp_uncertainty.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from PIL import Image

from gs_experiment.gpu_sh_directional_uncertainty import accumulate_sh_precision, compute_sh_directional_uncertainty_batched
from gs_experiment.gpu_uncertainty import compute_alpha_risk_batched
from gs_experiment.kernels import ProductKernel, RBFKernel
from gs_experiment.nerf_transforms import camera_pose_from_c2w, load_transforms
from gs_experiment.pixel_uncertainty import LocalUncertaintyEngine
from gs_experiment.scripts.render_reconstruction import CHECKPOINTS, EVAL_DIRS, RESULTS_DIR, _render_and_unproject, render_views
from gs_experiment.sh_directional_uncertainty import invert_precision
from gs_experiment.splat_scene import fit_kernel_hyperparams, load_from_gsplat_checkpoint

SCENE_VIEWS = {"lego": 21, "chair": 29, "ship": 21}
BACKGROUND_COLOR = (1.0, 1.0, 1.0)
DEPTH_RES = 64
ANGULAR_TOL = 0.05
LAM = 10.0  # SH-coefficient prior precision -- free hyperparameter, see module docstring


def _upsample(field: np.ndarray, valid: np.ndarray, width: int, height: int) -> np.ndarray:
    field_img = Image.fromarray(np.nan_to_num(field, nan=0.0).astype(np.float32), mode="F")
    valid_img = Image.fromarray((valid * 255).astype(np.uint8))
    field_up = np.array(field_img.resize((width, height), Image.BILINEAR))
    valid_up = np.array(valid_img.resize((width, height), Image.NEAREST)) > 127
    return np.where(valid_up, field_up, np.nan)


def _build_position_only_engine(scene_obj, sigma_rbf: float) -> LocalUncertaintyEngine:
    bounds = tuple((scene_obj.positions[:, d].min(), scene_obj.positions[:, d].max()) for d in range(3))
    return LocalUncertaintyEngine(
        positions=scene_obj.positions,
        values=scene_obj.colors,
        pos_kernel=ProductKernel([RBFKernel(sigma=sigma_rbf)] * 3),
        scene_bounds=bounds,
        opacities=scene_obj.opacities,
        scales=scene_obj.scales,
        rotations=scene_obj.rotations,
    )


def build_row(scene: str, view_idx: int) -> dict:
    spec = CHECKPOINTS[scene]["wide"]
    checkpoint_dir, radius = spec["dir"], spec["window_radius"]
    eval_dir = EVAL_DIRS[scene]

    results, checkpoint = render_views(
        str(eval_dir), [view_idx], checkpoint_dir=str(checkpoint_dir), background_color=BACKGROUND_COLOR
    )
    _, gt, recon = results[0]  # recon == C_alpha == mu_q, real renderer output, untouched
    height, width = gt.shape[:2]

    geo = _render_and_unproject(checkpoint_dir, eval_dir, [view_idx], depth_width=DEPTH_RES, depth_height=DEPTH_RES)[0]
    world_points, valid = geo["world_points"], geo["valid"]
    ys, xs = np.where(valid)
    points = world_points[ys, xs]
    print(f"{scene}: view {view_idx}, {len(ys)} valid low-res query points")

    scene_obj = load_from_gsplat_checkpoint(str(checkpoint_dir), use_gpu_attribution=True, attribution_min_opacity=0.1)
    sigma_rbf = fit_kernel_hyperparams(scene_obj, window_radius=radius, seed=0)
    print(f"  sigma_rbf={sigma_rbf:.5f} sh_degree={scene_obj.sh_degree} n_splats={scene_obj.positions.shape[0]}")

    engine = _build_position_only_engine(scene_obj, sigma_rbf)
    _, eval_frames = load_transforms(str(eval_dir / "transforms.json"))
    _, c2w = eval_frames[view_idx]
    camera = camera_pose_from_c2w(c2w)
    camera_index = engine.build_bearing_index(camera)

    # u_spatial_BQ(q): position-only, real alpha weights' own RKHS risk.
    t0 = time.time()
    _alpha_mean, u_spatial = compute_alpha_risk_batched(
        engine, camera_index, points, angular_tol=ANGULAR_TOL, sigma_rbf=sigma_rbf, max_candidates=500, device="cuda",
    )
    print(f"  u_spatial_BQ done in {time.time() - t0:.1f}s")

    # Sigma_theta_i: accumulated by rerendering every real training camera.
    t0 = time.time()
    precision = accumulate_sh_precision(
        scene_obj, degree=scene_obj.sh_degree, lam=LAM, angular_tol=ANGULAR_TOL, max_candidates=500, device="cuda",
    )
    sigma_theta = invert_precision(precision)
    print(f"  accumulate_sh_precision done in {time.time() - t0:.1f}s ({len(scene_obj.cameras)} training cameras)")

    # u_SH(q): query-side evaluation at the held-out view's own real pixels.
    t0 = time.time()
    to_camera = camera.center[None, :] - points
    query_directions = to_camera / np.linalg.norm(to_camera, axis=1, keepdims=True)
    u_sh = compute_sh_directional_uncertainty_batched(
        engine, camera_index, points, query_directions, sigma_theta, scene_obj.sh_degree,
        angular_tol=ANGULAR_TOL, max_candidates=100, device="cuda",
    )
    print(f"  u_SH done in {time.time() - t0:.1f}s")

    u_spatial_lowres = np.full((DEPTH_RES, DEPTH_RES), np.nan)
    u_sh_lowres = np.full((DEPTH_RES, DEPTH_RES), np.nan)
    u_spatial_lowres[ys, xs] = u_spatial
    u_sh_lowres[ys, xs] = u_sh

    u_spatial_map = _upsample(u_spatial_lowres, valid, width, height)
    u_sh_map = _upsample(u_sh_lowres, valid, width, height)
    u_total_map = u_spatial_map + u_sh_map

    return dict(
        scene=scene, view_idx=view_idx, gt=gt, recon=recon,
        u_spatial=u_spatial_map, u_sh=u_sh_map, u_total=u_total_map,
    )


def plot_comparison(rows: list[dict], out_path: Path):
    n = len(rows)
    n_cols = 5
    col_titles = ["ground truth", "C_alpha (mu_q)", "u_spatial_BQ", "u_SH", "u_spatial_BQ + u_SH"]
    fig, axes = plt.subplots(n, n_cols, figsize=(3.4 * n_cols, 3.1 * n))
    if n == 1:
        axes = axes[None, :]

    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad(color=(0.4, 0.4, 0.4))

    for row_idx, row in enumerate(rows):
        uncertainty_maps = [row["u_spatial"], row["u_sh"], row["u_total"]]
        pooled = np.concatenate([m[np.isfinite(m)] for m in uncertainty_maps])
        pooled = pooled[pooled > 0]
        if pooled.size > 0:
            row_vmin = float(pooled.min())
            row_vmax = max(float(pooled.max()), row_vmin * 1.01)
        else:
            row_vmin, row_vmax = 1e-6, 1.0
        print(f"{row['scene']}: shared uncertainty range = [{row_vmin:.4g}, {row_vmax:.4g}]")

        axes[row_idx, 0].imshow(row["gt"])
        axes[row_idx, 1].imshow(np.clip(row["recon"], 0.0, 1.0))
        for col_idx, key in enumerate(["u_spatial", "u_sh", "u_total"]):
            im = axes[row_idx, 2 + col_idx].imshow(row[key], cmap=cmap, norm=LogNorm(vmin=row_vmin, vmax=row_vmax))
            fig.colorbar(im, ax=axes[row_idx, 2 + col_idx], fraction=0.046, pad=0.04)

        if row_idx == 0:
            for k in range(n_cols):
                axes[row_idx, k].set_title(col_titles[k], fontsize=9)
        for k in range(n_cols):
            axes[row_idx, k].set_xticks([])
            axes[row_idx, k].set_yticks([])
        axes[row_idx, 0].set_ylabel(f"{row['scene']}\n(view {row['view_idx']})", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def run(scene_views=SCENE_VIEWS, out_path=None):
    rows = [build_row(scene, view_idx) for scene, view_idx in scene_views.items()]
    out_path = Path(out_path) if out_path else (RESULTS_DIR / "sparse_gp_uncertainty.png")
    plot_comparison(rows, out_path)
    return out_path


if __name__ == "__main__":
    run()
