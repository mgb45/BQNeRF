"""Batched-GPU equivalent of `visibility_attribution.attribute_observations`
(and the `occlusion_mask` it calls once per camera), for computing every
training camera's visibility attribution in one batched pass instead of a
100-iteration Python loop.

Why this exists: profiling `splat_scene.load_from_gsplat_checkpoint` on a
real checkpoint (after gpu_uncertainty.py's fix to the per-pixel BQ solve)
showed `attribute_observations` at ~81s of a ~98s total, almost entirely
inside `occlusion_mask`'s per-camera spatial-hash z-buffer -- specifically
~289 `np.searchsorted` calls per camera (one per (2*_CELL_SUBDIVISION+1)^2
neighbor-cell offset), each a CPU-serial lookup against that camera's own
sorted unique-cell-key array.

The key reformulation: `occlusion_mask`'s neighbor-cell lookup ("what's the
minimum depth among my cell and its (2S+1)x(2S+1)-cell neighborhood, treating
an empty cell as +inf") is *exactly* a min-pool over a 2D grid -- a single,
hardware-accelerated operation (`F.max_pool2d` on negated depth), not
something that needs an explicit per-offset loop at all. Because
`attribute_observations` is always called with one shared `fov_deg`/
`angular_tol` across every camera (see splat_scene.load_from_gsplat_checkpoint's
one call site), every camera's bearing grid has the *same* fixed extent and
cell size, so this also batches trivially across all cameras as one extra
leading dimension on the pooling call -- something the original per-camera
Python loop had no way to exploit.

This is a *reformulation* of occlusion_mask's own dense-grid semantics
(itself already an established grid-cell method, see its docstring), not a
different algorithm: for cell size `c = angular_tol/subdivision`, an
S-cell-radius neighborhood is still a superset of the true circular
`angular_tol` radius (same proof occlusion_mask's docstring gives), so this
carries the exact same false-positive-only (never false-negative)
guarantee, and should agree with attribute_observations to floating-point
precision, not just approximately -- verified directly in
tests/gs_experiment/test_gpu_visibility_attribution.py.

Needs torch (a GPU is not strictly required but is the point).
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from gs_experiment.visibility_attribution import _CELL_SUBDIVISION, in_frustum, project_to_camera_local


def batched_attribute_observations(
    positions: np.ndarray,
    cameras: list,
    fov_deg: float = 60.0,
    angular_tol: float = 0.05,
    depth_margin: float = 0.05,
    device: str = "cuda",
    subdivision: int = _CELL_SUBDIVISION,
) -> List[np.ndarray]:
    """Batched equivalent of
    `visibility_attribution.attribute_observations(positions, cameras,
    fov_deg=fov_deg, angular_tol=angular_tol, depth_margin=depth_margin)`
    -- same return contract: a list of length `len(cameras)`, each entry
    the splat indices that camera plausibly observes (in frustum, not
    occluded).
    """
    n_cameras = len(cameras)
    n = positions.shape[0]
    dtype = torch.float64

    # --- 1. project every position into every camera's bearing/depth once
    # (cheap and already vectorized per camera -- profiling showed this was
    # never the bottleneck; occlusion_mask's neighbor search was). ---
    all_bx = np.empty((n_cameras, n))
    all_by = np.empty((n_cameras, n))
    all_depth = np.empty((n_cameras, n))
    all_visible = np.empty((n_cameras, n), dtype=bool)
    for c, camera in enumerate(cameras):
        bx, by, depth = project_to_camera_local(positions, camera)
        all_bx[c], all_by[c], all_depth[c] = bx, by, depth
        half_tan = np.tan(np.deg2rad(fov_deg) / 2.0)
        in_front = (depth > 1e-3) & np.isfinite(depth)
        within_fov = (np.abs(bx) < half_tan) & (np.abs(by) < half_tan)
        all_visible[c] = in_front & within_fov & ~np.isnan(bx)

    bx_t = torch.nan_to_num(torch.tensor(all_bx, dtype=dtype, device=device), nan=0.0)
    by_t = torch.nan_to_num(torch.tensor(all_by, dtype=dtype, device=device), nan=0.0)
    depth_t = torch.tensor(all_depth, dtype=dtype, device=device)
    visible_t = torch.tensor(all_visible, dtype=torch.bool, device=device)

    # --- 2. bin into a dense grid covering the whole (shared, fov_deg-bounded)
    # bearing range -- every camera uses the *same* grid shape since fov_deg/
    # angular_tol are shared, which is exactly what makes batching possible. ---
    half_tan = float(np.tan(np.deg2rad(fov_deg) / 2.0))
    cell = max(float(angular_tol) / subdivision, 1e-12)
    half_extent_cells = int(np.ceil(half_tan / cell)) + 1  # +1 cell of margin past the frustum edge
    grid_size = 2 * half_extent_cells + 1
    center = half_extent_cells

    cx = torch.floor(bx_t / cell).to(torch.int64) + center
    cy = torch.floor(by_t / cell).to(torch.int64) + center
    in_bounds = (cx >= 0) & (cx < grid_size) & (cy >= 0) & (cy < grid_size)
    use = visible_t & in_bounds
    flat_cell = cy * grid_size + cx

    # --- 3. per-cell minimum depth, all cameras at once: scatter-reduce
    # (amin) instead of occlusion_mask's sort + reduceat, since we now have
    # a dense index space to scatter into rather than a sparse sorted-key one. ---
    flat_cell_safe = torch.where(use, flat_cell, torch.zeros_like(flat_cell))
    depth_for_scatter = torch.where(use, depth_t, torch.full_like(depth_t, float("inf")))
    grid = torch.full((n_cameras, grid_size * grid_size), float("inf"), dtype=dtype, device=device)
    grid.scatter_reduce_(1, flat_cell_safe, depth_for_scatter, reduce="amin", include_self=True)

    # --- 4. the actual reformulation: occlusion_mask's explicit
    # (2*subdivision+1)^2-offset Python loop over searchsorted lookups is
    # exactly a min-pool over that same window (max_pool2d on negated depth;
    # out-of-grid padding acts as -inf pre-negation, i.e. +inf post-negation
    # -- "no data here", never a spurious occluder, matching the original's
    # "cell not found -> infinity" treatment). One call, every camera at once. ---
    grid2d = grid.view(n_cameras, 1, grid_size, grid_size)
    kernel = 2 * subdivision + 1
    pooled = -F.max_pool2d(-grid2d, kernel_size=kernel, stride=1, padding=subdivision)
    pooled_flat = pooled.view(n_cameras, -1)

    flat_cell_gather = flat_cell.clamp(0, grid_size * grid_size - 1)
    best_neighbor_depth = torch.gather(pooled_flat, 1, flat_cell_gather)

    occluded = best_neighbor_depth < depth_t - depth_margin * depth_t.abs()
    occluded_np = occluded.cpu().numpy()

    return [np.where(all_visible[c] & ~occluded_np[c])[0] for c in range(n_cameras)]
