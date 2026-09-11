"""Shared rendering/uncertainty library: renders ground-truth vs. gsplat
reconstruction for a trained checkpoint (gs_experiment.scripts.train_minimal_gsplat's
output) and computes real per-pixel BQ uncertainty on the same views (real
depth-unprojection via gsplat's own "ED" output, real closed-form BQ position-only
and position+direction variance at each ray-surface hit, not approximated) --
so a bad-quality view and a low-confidence view can be told apart on sight
rather than conflated (ROADMAP.md item 2: "render it and look, before reaching
for statistics").

No CLI of its own -- render_views/compute_uncertainty_maps are imported by the
actual figure-generation scripts (render_scene_gallery.py,
render_coverage_uncertainty_sweep.py, render_splat_sweep_gallery.py).

Needs torch + gsplat (requirements-gsplat.txt).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from PIL import Image

from gs_experiment.nerf_transforms import fov_x_to_intrinsics, load_transforms, opencv_viewmat_from_c2w
from gs_experiment.ply_io import read_3dgs_ply

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Used by compute_uncertainty_maps only when per-scene fitting (splat_scene.
# fit_kernel_hyperparams) can't produce a value (too little real data in this
# checkpoint) -- not used when fitting succeeds, which is now the default.
# These are the project's own marginal-likelihood pooled fits (see that
# function's docstring), not arbitrary guesses. FALLBACK_SIGMA matches
# real_directional_coverage_experiment.py's LEGO_GAP_SIGMA -- see that
# constant's own comment for the 2025-09 refit story (a real bug,
# SplatScene.colors being raw SH coefficients rather than real color,
# gs_experiment/results/FINDINGS.md section 4). FALLBACK_KAPPA is
# unaffected by that bug (its fitting path never used the raw fallback).
FALLBACK_SIGMA = 0.13926
FALLBACK_KAPPA = 0.745

# Shared log-scale color range for raw (unnormalized) posterior variance, used by
# every figure that plots it (render_scene_gallery.py, render_coverage_uncertainty_sweep.py,
# render_splat_sweep_gallery.py) so panels are visually comparable across scenes,
# splat budgets, and figures -- a per-panel autoscale would hide exactly the
# cross-condition magnitude differences those figures exist to show. Fixed, not
# computed per-run: covers the empirical range observed across every real
# checkpoint rendered so far in this project, from ~1e-3 at the best-supported
# points (300k-splat scene checkpoints) to ~134 at the least-supported (the
# widest, 51-training-view coverage-gap condition), with headroom on each end
# rather than clipping to exactly what's been seen so far.
RAW_VARIANCE_VMIN = 1e-3
RAW_VARIANCE_VMAX = 2e2


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
    """Same construction as train_minimal_gsplat.py's function of the same
    name (duplicated rather than imported, so that training doesn't depend on
    this rendering/plotting script -- see train_minimal_gsplat.py's own
    docstring on that function): depth (H, W) in OpenCV camera space ->
    (H, W, 3) world-space points."""
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
    # None (the default) fits sigma/kappa against THIS checkpoint's own real data
    # (splat_scene.fit_kernel_hyperparams) rather than reusing a bandwidth pooled
    # once across a different calibration set -- a bandwidth tuned at one splat
    # density/coverage has no reason to be right at another (confirmed directly:
    # a 300k-splat-tuned sigma is a real mismatch at 500 splats, not just a
    # theoretical concern). Pass an explicit value to opt back out of fitting,
    # e.g. for exact reproducibility across a sweep or a fixed-vs-fitted
    # comparison (an earlier, retired sweep script did its own explicit sigma
    # refit and deliberately reported both side by side rather than using
    # this default; see git history).
    sigma: Optional[float] = None,
    kappa: Optional[float] = None,
    window_radius: float = 1.6,
    max_neighbors: int = 150,
    alpha_threshold: float = 0.5,
    depth_width: int = 112,
    depth_height: int = 42,
    attribution_angular_tol: float = 0.01,
    max_observations_per_splat: Optional[int] = None,
    return_raw_variance: bool = False,
    noise_variance: Optional[float] = None,
    fit_noise_variance: bool = False,
    return_bq_mean: bool = False,
):
    """Real per-pixel BQ uncertainty at every view in `view_indices`, on
    the exact same checkpoint `render_views` just rendered RGB from -- real
    depth-unprojection (gsplat's own "ED" render mode, not an approximation)
    plus a closed-form BQ variance at every resulting ray-surface hit,
    evaluated at real dataset views so it lines up 1:1 with the GT/recon/error
    columns already being rendered.

    `scene_dir` supplies the views being queried (`frames`); `checkpoint_dir`
    (defaults to `scene_dir`, matching `render_views`) supplies splats.ply
    *and* the transforms.json used to attribute which training views
    observed each splat -- that attribution must come from the checkpoint's
    own training views, not whatever split is being rendered here, or a
    held-out eval split would silently masquerade as the training-view pool.

    `sigma`/`kappa`: `None` (the default) fits both against this checkpoint's
    own data (see the parameter comment above and splat_scene.fit_kernel_hyperparams);
    pass explicit values to disable fitting. `window_radius`/`max_neighbors`
    default to a synthetic-scene scale -- pick values matching the actual
    scene's spatial scale for a different scene family (see
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

    `return_raw_variance`: if True, each tuple gains a third element,
    `raw_directional_variance` -- the same numerator `directional_map` divides
    by `prior_variance` to get its bounded ratio, kept unnormalized this time.
    Not comparable across scenes/checkpoints/figures (see above -- it's
    dominated by whatever `sigma`/`kappa` this call fit or was given), so a
    caller plotting it needs its own per-panel scale, not the ratio's shared
    fixed [0,1] one. Default False, i.e. every existing 2-tuple-unpacking
    caller is unaffected.

    `max_observations_per_splat`: passed straight through to
    `load_from_gsplat_checkpoint` -- caps the (splat, observing-camera)
    row count the directional engine below is built from, which is what
    actually determines its host memory footprint at high splat counts
    (see that function's docstring). `None` (the default) keeps every
    observation, i.e. unchanged from before this parameter existed.

    `noise_variance`/`fit_noise_variance`: real homoscedastic observation-
    noise support (see `gs_experiment.quadrature._rendering_aware_moments`'s
    docstring for the model/motivation, and
    `splat_scene.fit_kernel_hyperparams_with_noise` for the fitting
    procedure) -- default `noise_variance=None`, `fit_noise_variance=False`
    reproduces the exact noiseless behavior every existing caller already
    gets. Pass `fit_noise_variance=True` (with `sigma=None`, the default)
    to jointly fit sigma and a real noise variance against this
    checkpoint's own data instead of `fit_kernel_hyperparams`'s noiseless
    fit -- found, on every real checkpoint checked so far, to fit
    dramatically better (a real, large marginal-likelihood improvement,
    not a close call) and to visibly reduce the negative-BQ-weight/
    color-speckle artifacts the noiseless fit produces. Pass an explicit
    `noise_variance` to opt out of fitting it (same "explicit value skips
    fitting" convention `sigma`/`kappa` already use).

    `return_bq_mean` (default False, unchanged behavior): if True, each
    returned tuple gains one more element (appended after `raw_field`
    when `return_raw_variance` is also True, otherwise appended right
    after `dir_field`) -- `C_BQ`, the BQ posterior mean, as a real
    (height, width, 3) RGB array at the same valid pixels as every other
    field. This project's core GP machinery is single-channel by design
    elsewhere (`scene.colors`/`obs_values`, the mean of the SH DC term
    across RGB -- see `splat_scene.SplatScene.colors`), but a genuine
    3-channel `C_BQ` is available for near-zero extra cost via
    `splat_observations`'s `return_rgb=True` (the real per-channel SH
    values `obs_values` is itself collapsed from) combined with
    `compute_directional_variance_batched`'s own `values_rgb` parameter,
    which solves all 3 channels through the SAME Cholesky factorization
    the variance computation already performs (candidate gathering and
    `Kxx` don't depend on color at all, only which right-hand-side column
    does). This is `u_BQ`'s own mean, i.e. the *coherent* pairing --
    unlike `C_alpha` (`render_views`'s real alpha-compositing output),
    which `u_BQ` was never derived as a variance around (see
    `gs_experiment/results/FINDINGS.md` section 3).
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
        # splat_scene.fit_kernel_hyperparams) -- excludes GS-training floaters from
        # hard-occluding real splats during attribution (see
        # load_from_gsplat_checkpoint's docstring).
        attribution_min_opacity=0.1,
        max_observations_per_splat=max_observations_per_splat,
    )

    if noise_variance is None and fit_noise_variance and sigma is None:
        from gs_experiment.splat_scene import fit_kernel_hyperparams_with_noise

        fitted_sigma, fitted_noise_variance, fitted_kappa = fit_kernel_hyperparams_with_noise(scene)
        used_sigma = fitted_sigma if fitted_sigma is not None else FALLBACK_SIGMA
        used_noise_variance = fitted_noise_variance if fitted_noise_variance is not None else 0.0
        used_kappa = kappa if kappa is not None else (fitted_kappa if fitted_kappa is not None else FALLBACK_KAPPA)
        print(
            f"compute_uncertainty_maps: sigma={used_sigma:.4f}"
            f"{' (fitted w/ noise)' if fitted_sigma is not None else ' (fallback)'}, "
            f"noise_variance={used_noise_variance:.5f}"
            f"{' (fitted)' if fitted_noise_variance is not None else ' (fallback: 0)'}, "
            f"kappa={used_kappa:.4f}"
            f"{' (fitted)' if kappa is None and fitted_kappa is not None else ' (fallback)' if kappa is None else ''}"
        )
        sigma, kappa, noise_variance = used_sigma, used_kappa, used_noise_variance
    elif sigma is None or kappa is None:
        from gs_experiment.splat_scene import fit_kernel_hyperparams

        fitted_sigma, fitted_kappa = fit_kernel_hyperparams(scene)
        used_sigma = sigma if sigma is not None else (fitted_sigma if fitted_sigma is not None else FALLBACK_SIGMA)
        used_kappa = kappa if kappa is not None else (fitted_kappa if fitted_kappa is not None else FALLBACK_KAPPA)
        print(
            f"compute_uncertainty_maps: sigma={used_sigma:.4f}"
            f"{' (fitted)' if sigma is None and fitted_sigma is not None else ' (fallback)' if sigma is None else ''}, "
            f"kappa={used_kappa:.4f}"
            f"{' (fitted)' if kappa is None and fitted_kappa is not None else ' (fallback)' if kappa is None else ''}"
        )
        sigma, kappa = used_sigma, used_kappa

    noise_variance = noise_variance if noise_variance is not None else 0.0

    obs_positions, obs_directions, obs_values, obs_opacities, obs_scales, obs_rotations, obs_values_rgb = (
        splat_observations(scene, include_render_attrs=True, return_rgb=True)
    )
    bounds = tuple((obs_positions[:, d].min() - 1.0, obs_positions[:, d].max() + 1.0) for d in range(3))

    pos_kernel = make_default_3d_position_kernel(sigma=sigma)
    dir_kernel = DirectionalKernel(kappa=kappa)
    # deduplicated splat positions for the position-only engine, camera-
    # expanded observation rows for the directional one -- reusing the
    # expanded rows for the position-only query would silently leak
    # observation-count into a signal meant to be blind to direction.
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

    def upsample_rgb(field_rgb, valid):
        # `upsample`'s PIL "F" mode is single-channel -- upsample each
        # channel independently (same bilinear/nearest resize, same valid
        # mask) and stack, rather than a separate RGB-aware implementation.
        return np.stack([upsample(field_rgb[..., c], valid) for c in range(field_rgb.shape[-1])], axis=-1)

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
            raw_field = np.full((depth_height, depth_width), np.nan)
            bq_mean_field = np.full((depth_height, depth_width, 3), np.nan)
            ys, xs = np.where(valid)
            points = world_points[ys, xs]
            to_camera = cam_center[None, :] - points
            query_directions = to_camera / np.linalg.norm(to_camera, axis=1, keepdims=True)

            for y, x, point in zip(ys, xs, points):
                spatial_field[y, x] = spatial_engine.rendering_aware_variance(
                    point, window_radius, noise_variance=noise_variance
                ).variance
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
                batched_result = compute_directional_variance_batched(
                    directional_engine, camera_index, points, query_directions,
                    angular_tol=0.05, sigma_rbf=sigma, kappa=kappa, max_candidates=500, device=device,
                    noise_variance=noise_variance,
                    values_rgb=(obs_values_rgb if return_bq_mean else None),
                )
                if return_bq_mean:
                    variance, prior_variance, bq_mean_rgb = batched_result
                    bq_mean_field[ys, xs] = bq_mean_rgb
                else:
                    variance, prior_variance = batched_result
                dir_field[ys, xs] = variance / np.maximum(prior_variance, 1e-300)
                raw_field[ys, xs] = variance

            row = (upsample(spatial_field, valid), upsample(dir_field, valid))
            if return_raw_variance:
                row = row + (upsample(raw_field, valid),)
            if return_bq_mean:
                row = row + (upsample_rgb(bq_mean_field, valid),)
            maps.append(row)

    return maps

