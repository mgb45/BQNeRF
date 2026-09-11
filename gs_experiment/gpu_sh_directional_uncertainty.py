"""Real per-(splat, training-camera) alpha-compositing weight
beta_{p,i} = T_{p,i} * alpha_{p,i}, and the accumulated per-splat SH-
coefficient precision `sh_directional_uncertainty.py`'s Bayesian linear
regression needs -- both computed by rerendering the fixed training
cameras (batched over GPU), not by retraining anything.

`beta_{p,i}` is splat i's own real depth-ordered alpha-compositing weight
at the pixel its own projected center lands on, in training camera p --
exactly `visibility_attribution.ray_transmittance_weights(...)`'s value at
that splat's own index within its own bearing neighborhood. This is a
genuinely different quantity from `observed_camera_idx`'s binary
visible/occluded flag (`gpu_visibility_attribution.
batched_attribute_observations`): a splat can be geometrically visible
(unoccluded enough to be attributed at all) yet contribute almost nothing
to its own pixel if it sits behind a nearly-opaque closer splat on the
same ray, or contribute a large weight if it's the dominant, frontmost
splat there -- exactly the real per-observation "how much did training's
photometric loss actually constrain this splat's SH coefficients from
this direction" signal `Sigma_theta_i` needs.

Needs torch (a GPU is not strictly required but is the point).
"""

from __future__ import annotations

import numpy as np
import torch

from gs_experiment.camera import directions_from_positions_to_camera
from gs_experiment.sh_directional_uncertainty import sh_basis
from gs_experiment.spherical_harmonics import N_COEFFS_FOR_DEGREE
from gs_experiment.visibility_attribution import CameraSplatIndex, project_to_camera_local


def _gather_own_alpha_weight_batched(
    ref_bearing, query_global_idx, angular_tol, k_cap, device, dtype,
    index_bearings, index_depths, index_opacities, index_global_idx,
):
    """One chunk of `compute_own_alpha_weight_batched`: real candidate
    search + depth-ordered transmittance weights (identical construction to
    `gpu_uncertainty._gather_position_only_candidates_and_weights`), then
    picks out each query row's OWN candidate slot (the one whose gathered
    global index matches `query_global_idx[row]`) instead of returning
    every candidate's weight -- a query splat is always within its own
    bearing ball at distance exactly 0 (the global minimum), so it is
    always kept regardless of `k_cap` capping; `own_mask` therefore always
    has exactly one nonzero entry per valid row."""
    dist = torch.cdist(ref_bearing, index_bearings)
    in_radius = dist < angular_tol
    masked_dist = torch.where(in_radius, dist, torch.full_like(dist, float("inf")))
    top_values, top_idx = torch.topk(masked_dist, k=k_cap, dim=1, largest=False)
    valid = torch.isfinite(top_values)

    depths = index_depths[top_idx]
    opacities = index_opacities[top_idx]
    global_idx = index_global_idx[top_idx]

    depths_for_sort = torch.where(valid, depths, torch.full_like(depths, float("inf")))
    order = torch.argsort(depths_for_sort, dim=1, stable=True)
    opacities = torch.gather(opacities, 1, order)
    global_idx = torch.gather(global_idx, 1, order)
    valid = torch.gather(valid, 1, order)

    alpha = torch.where(valid, opacities.clamp(0.0, 1.0), torch.zeros_like(opacities))
    one_minus_alpha = 1.0 - alpha
    inclusive = torch.cumprod(one_minus_alpha, dim=1)
    transmittance = torch.cat([torch.ones_like(inclusive[:, :1]), inclusive[:, :-1]], dim=1)
    weights = transmittance * alpha
    weights = torch.where(valid, weights, torch.zeros_like(weights))

    own_mask = (global_idx == query_global_idx.unsqueeze(1)).to(dtype)
    return (weights * own_mask).sum(dim=1)


def compute_own_alpha_weight_batched(
    positions: np.ndarray,
    opacities: np.ndarray,
    camera,
    query_splat_idx: np.ndarray,
    angular_tol: float,
    max_candidates: int = 500,
    device: str = "cuda",
    pixel_chunk_bytes: float = 1.5e9,
) -> np.ndarray:
    """beta_i = T_i*alpha_i for each splat index in `query_splat_idx`, at
    its own real projected bearing in `camera`, rendered against every
    other splat in `positions` -- exactly
    `visibility_attribution.ray_transmittance_weights(positions[idx],
    opacities[idx], camera, own_bearing, angular_tol)[own_slot]` for each
    query row, batched (cross-validated directly against that scalar call
    in tests/gs_experiment/test_gpu_sh_directional_uncertainty.py).

    A query splat behind the camera or with undefined bearing (shouldn't
    normally happen -- `observed_camera_idx` is built from a frustum+depth
    check -- but not assumed) gets beta=0 rather than a bogus value.
    """
    dtype = torch.float64
    positions = np.asarray(positions, dtype=float)
    query_splat_idx = np.asarray(query_splat_idx, dtype=np.int64)
    p_total = query_splat_idx.shape[0]
    if p_total == 0:
        return np.zeros(0, dtype=np.float64)

    camera_index = CameraSplatIndex.build(positions, camera)
    k_cap = min(max_candidates, camera_index.indices.shape[0])
    if k_cap == 0:
        return np.zeros(p_total, dtype=np.float64)

    query_positions = positions[query_splat_idx]
    ref_bx, ref_by, ref_depth = project_to_camera_local(query_positions, camera)
    valid_query = np.isfinite(ref_bx) & (ref_depth > 0)
    ref_bearing = torch.tensor(np.nan_to_num(np.stack([ref_bx, ref_by], axis=1)), dtype=dtype, device=device)
    query_global_idx = torch.tensor(query_splat_idx, dtype=torch.int64, device=device)

    index_bearings = torch.tensor(camera_index.bearings, dtype=dtype, device=device)
    index_depths = torch.tensor(camera_index.depths, dtype=dtype, device=device)
    index_opacities = torch.tensor(np.asarray(opacities, dtype=float)[camera_index.indices], dtype=dtype, device=device)
    index_global_idx = torch.tensor(camera_index.indices, dtype=torch.int64, device=device)

    m = camera_index.indices.shape[0]
    bytes_per_row = m * 8 * 2  # dist + masked_dist, this path's only (P, M) matrices
    chunk_size = max(1, min(p_total, int(pixel_chunk_bytes / max(bytes_per_row, 1))))

    beta = np.empty(p_total, dtype=np.float64)
    for start in range(0, p_total, chunk_size):
        end = min(start + chunk_size, p_total)
        beta[start:end] = (
            _gather_own_alpha_weight_batched(
                ref_bearing[start:end], query_global_idx[start:end], angular_tol, k_cap, device, dtype,
                index_bearings, index_depths, index_opacities, index_global_idx,
            )
            .cpu()
            .numpy()
        )
    beta[~valid_query] = 0.0
    return beta


def accumulate_sh_precision(
    scene,
    degree: int,
    lam: float,
    angular_tol: float = 0.05,
    max_candidates: int = 500,
    device: str = "cuda",
) -> np.ndarray:
    """Sigma_theta_i^-1 = lam*I + sum_p beta_{p,i}^2 * phi(d_p) phi(d_p)^T
    for every splat i in `scene`, accumulated by rerendering every real
    training camera in `scene.cameras` (via `compute_own_alpha_weight_
    batched` above) -- see `sh_directional_uncertainty.py`'s module
    docstring for the full formula/motivation. Returns
    `(n_splats, n_coeffs, n_coeffs)` precision matrices (not yet inverted
    -- pass through `sh_directional_uncertainty.invert_precision`).

    Loops once per real training camera (typically tens to a few hundred),
    each iteration doing one batched GPU pass over that camera's own real
    observed splats (`scene.observed_camera_idx`, inverted to per-camera
    lists below) -- not a per-splat or per-(splat,camera) Python loop.
    """
    n_splats = scene.positions.shape[0]
    n_coeffs = N_COEFFS_FOR_DEGREE[degree]
    precision = np.tile(lam * np.eye(n_coeffs), (n_splats, 1, 1))

    per_camera: list[list[int]] = [[] for _ in scene.cameras]
    for s, cams in enumerate(scene.observed_camera_idx):
        for c in cams:
            per_camera[c].append(s)

    for c, splat_ids in enumerate(per_camera):
        if len(splat_ids) == 0:
            continue
        splat_ids = np.asarray(splat_ids, dtype=np.int64)
        camera = scene.cameras[c]
        beta = compute_own_alpha_weight_batched(
            scene.positions, scene.opacities, camera, splat_ids, angular_tol, max_candidates, device=device,
        )
        nonzero = beta > 0
        if not np.any(nonzero):
            continue
        splat_ids_nz = splat_ids[nonzero]
        beta_nz = beta[nonzero]
        directions = directions_from_positions_to_camera(scene.positions[splat_ids_nz], camera)
        phi = sh_basis(directions, degree)
        outer = (beta_nz**2)[:, None, None] * (phi[:, :, None] * phi[:, None, :])
        np.add.at(precision, splat_ids_nz, outer)
    return precision


# ---------------------------------------------------------------------------
# u_SH(q) = sum_i beta_{q,i}^2 * phi(d_q)^T Sigma_theta_i phi(d_q): the query-
# side evaluation, at real held-out pixels, of the directional term whose
# per-splat Sigma_theta this module's own accumulate_sh_precision (+
# sh_directional_uncertainty.invert_precision) already builds. Batched the
# same way gpu_uncertainty.compute_alpha_risk_batched batches u_spatial_BQ.
# ---------------------------------------------------------------------------


def _sh_basis_torch(directions: torch.Tensor, degree: int) -> torch.Tensor:
    """torch equivalent of `sh_directional_uncertainty.sh_basis`, for
    directions that are already GPU tensors (query directions, one per
    pixel) -- same real SH basis, same coefficient ordering."""
    from gs_experiment.spherical_harmonics import SH_C0, SH_C1, SH_C2, SH_C3

    x, y, z = directions[..., 0], directions[..., 1], directions[..., 2]
    cols = [torch.full_like(x, SH_C0)]
    if degree > 0:
        cols += [-SH_C1 * y, SH_C1 * z, -SH_C1 * x]
        if degree > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            cols += [
                SH_C2[0] * xy,
                SH_C2[1] * yz,
                SH_C2[2] * (2.0 * zz - xx - yy),
                SH_C2[3] * xz,
                SH_C2[4] * (xx - yy),
            ]
            if degree > 2:
                cols += [
                    SH_C3[0] * y * (3 * xx - yy),
                    SH_C3[1] * xy * z,
                    SH_C3[2] * y * (4 * zz - xx - yy),
                    SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy),
                    SH_C3[4] * x * (4 * zz - xx - yy),
                    SH_C3[5] * z * (xx - yy),
                    SH_C3[6] * x * (xx - 3 * yy),
                ]
    return torch.stack(cols, dim=-1)


def compute_sh_directional_uncertainty_batched(
    engine,
    camera_index: CameraSplatIndex,
    query_points: np.ndarray,
    query_directions: np.ndarray,
    sigma_theta: np.ndarray,
    degree: int,
    angular_tol: float,
    max_candidates: int = 100,
    device: str = "cuda",
    pixel_chunk_bytes: float = 1.0e9,
) -> np.ndarray:
    """u_SH(q) for each real query pixel: gathers the same real, depth-
    ordered candidate splats and alpha weights `gpu_uncertainty.
    compute_alpha_risk_batched` does (position-only ball query, ascending-
    bearing-distance overflow ranking), but additionally keeps each
    candidate's own global splat index so `sigma_theta` (accumulated by
    `accumulate_sh_precision` + `sh_directional_uncertainty.
    invert_precision`, `(n_splats, n_coeffs, n_coeffs)`) can be gathered
    per candidate, then evaluates `phi(d_q)^T Sigma_theta_i phi(d_q)` per
    candidate and sums `beta_{q,i}^2 * s_i^2(d_q)` over the ray.

    `query_directions`: one real viewing direction per query pixel (e.g.
    `camera.center - point`, normalized -- this project's established
    convention), shared across all of that pixel's own candidates (the
    query direction is a property of the ray, not of any one candidate).
    """
    dtype = torch.float64
    query_points = np.atleast_2d(np.asarray(query_points, dtype=float))
    query_directions = np.atleast_2d(np.asarray(query_directions, dtype=float))
    p_total = query_points.shape[0]
    k_cap = min(max_candidates, camera_index.indices.shape[0])
    if k_cap == 0 or p_total == 0:
        return np.zeros(p_total, dtype=np.float64)

    m = camera_index.indices.shape[0]
    n_coeffs = sigma_theta.shape[-1]
    index_bearings = torch.tensor(camera_index.bearings, dtype=dtype, device=device)
    index_depths = torch.tensor(camera_index.depths, dtype=dtype, device=device)
    index_opacities = torch.tensor(engine.opacities[camera_index.indices], dtype=dtype, device=device)
    index_global_idx = torch.tensor(camera_index.indices, dtype=torch.int64, device=device)
    sigma_theta_t = torch.tensor(sigma_theta, dtype=dtype, device=device)

    bytes_per_row = m * 8 * 2 + k_cap * n_coeffs * n_coeffs * 8
    chunk_size = max(1, min(p_total, int(pixel_chunk_bytes / max(bytes_per_row, 1))))

    out = np.empty(p_total, dtype=np.float64)
    for start in range(0, p_total, chunk_size):
        end = min(start + chunk_size, p_total)
        ref_bx, ref_by, _ = project_to_camera_local(query_points[start:end], camera_index.camera)
        ref_bearing = torch.tensor(np.stack([ref_bx, ref_by], axis=1), dtype=dtype, device=device)

        dist = torch.cdist(ref_bearing, index_bearings)
        in_radius = dist < angular_tol
        masked_dist = torch.where(in_radius, dist, torch.full_like(dist, float("inf")))
        top_values, top_idx = torch.topk(masked_dist, k=k_cap, dim=1, largest=False)
        valid = torch.isfinite(top_values)

        depths = index_depths[top_idx]
        opacities = index_opacities[top_idx]
        global_idx = index_global_idx[top_idx]

        depths_for_sort = torch.where(valid, depths, torch.full_like(depths, float("inf")))
        order = torch.argsort(depths_for_sort, dim=1, stable=True)
        opacities = torch.gather(opacities, 1, order)
        global_idx = torch.gather(global_idx, 1, order)
        valid = torch.gather(valid, 1, order)

        alpha = torch.where(valid, opacities.clamp(0.0, 1.0), torch.zeros_like(opacities))
        one_minus_alpha = 1.0 - alpha
        inclusive = torch.cumprod(one_minus_alpha, dim=1)
        transmittance = torch.cat([torch.ones_like(inclusive[:, :1]), inclusive[:, :-1]], dim=1)
        weights = transmittance * alpha
        weights = torch.where(valid, weights, torch.zeros_like(weights))

        global_idx_safe = torch.where(valid, global_idx, torch.zeros_like(global_idx))
        sigma_gathered = sigma_theta_t[global_idx_safe]  # (chunk, k_cap, n_coeffs, n_coeffs)

        directions_t = torch.tensor(query_directions[start:end], dtype=dtype, device=device)
        phi = _sh_basis_torch(directions_t, degree)  # (chunk, n_coeffs)
        phi_exp = phi.unsqueeze(1).expand(-1, k_cap, -1)
        s2 = torch.einsum("pki,pkij,pkj->pk", phi_exp, sigma_gathered, phi_exp)
        s2 = torch.where(valid, s2, torch.zeros_like(s2))

        u_sh = (weights**2 * s2).sum(dim=1)
        out[start:end] = u_sh.cpu().numpy()
    return out
