"""Batched-GPU computation of u_spatial_BQ(q): the position-only (direction-
blind) "finite spatial representation" term of the renderer-consistent
sparse-GP decomposition of 3D Gaussian Splatting (see
gs_experiment/sh_directional_uncertainty.py's module docstring for the full
picture):

    mu_q = C_alpha(q)                              (real renderer mean, untouched)
    u_q  = u_spatial_BQ(q) + u_SH(q)

u_spatial_BQ(q) is exactly `pixel_uncertainty.LocalUncertaintyEngine.
rendering_aware_alpha_risk_along_ray(...).alpha_risk`: the RKHS worst-case
risk `e(w_alpha)^2 = z0 - 2*w_alpha@z + w_alpha@Kxx@w_alpha` of the REAL
alpha-compositing weights `w_alpha = T_i*alpha_i`, scored (not solved for)
under a position-only, single-Gaussian-a_q kernel. `compute_alpha_risk_
batched` batches that scalar method the same way profiling once showed a
per-pixel Python loop splits its cost between candidate lookup (a KD-tree
ball query) and the BQ linear algebra -- both are naturally batchable
across pixels and independent of each other, so there is no correctness
reason to keep either in a Python loop.

Needs torch (a GPU is not strictly required but is the point). Uses
float64 throughout to match the scalar path's numpy/scipy float64
precision. Cross-validated directly against the scalar path in
tests/gs_experiment/test_gpu_uncertainty_alpha_risk.py (exact per-pixel
agreement on real data, not just a smoke test).
"""

from __future__ import annotations

import numpy as np
import torch

from gs_experiment.visibility_attribution import CameraSplatIndex, project_to_camera_local


def _zero_candidate_variance(covariance: torch.Tensor, sigma_rbf: float, d: int) -> np.ndarray:
    a = 1e-12
    cov_sum2 = 2 * covariance + (sigma_rbf**2) * torch.eye(d, dtype=covariance.dtype, device=covariance.device)
    logdet = torch.linalg.slogdet(cov_sum2)[1]
    z0 = (a**2) / ((2 * np.pi) ** (d / 2.0) * torch.exp(0.5 * logdet))
    return z0.clamp_min(0.0).cpu().numpy()


def _gather_position_only_candidates_and_weights(
    camera_index, query_points, angular_tol, k_cap, device, dtype,
    index_bearings, index_depths, index_positions, index_values, index_opacities, index_covariances,
):
    """Real candidate search (exact ball-query semantics via a dense
    bearing-distance matrix -- equivalent to `CameraSplatIndex.query` at the
    same angular_tol, not an approximation, just not tree-accelerated) plus
    real depth-ordered alpha-compositing transmittance weights. Ranks
    overflow past `k_cap` by ASCENDING BEARING DISTANCE (nearest first),
    matching `CameraSplatIndex.query`'s own default `query_direction=None`
    branch -- this kernel is deliberately direction-blind (directional
    uncertainty is a separate term, see module docstring), so no
    `directions` array is gathered or needed at all."""
    ref_bx, ref_by, _ = project_to_camera_local(query_points, camera_index.camera)
    ref_bearing = torch.tensor(np.stack([ref_bx, ref_by], axis=1), dtype=dtype, device=device)  # (P, 2)

    dist = torch.cdist(ref_bearing, index_bearings)  # (P, M)
    in_radius = dist < angular_tol
    masked_dist = torch.where(in_radius, dist, torch.full_like(dist, float("inf")))
    top_values, top_idx = torch.topk(masked_dist, k=k_cap, dim=1, largest=False)  # (P, k_cap), nearest first
    valid = torch.isfinite(top_values)

    positions = index_positions[top_idx]
    depths = index_depths[top_idx]
    opacities = index_opacities[top_idx]
    values = index_values[top_idx]
    covariances = index_covariances[top_idx]

    depths_for_sort = torch.where(valid, depths, torch.full_like(depths, float("inf")))
    order = torch.argsort(depths_for_sort, dim=1, stable=True)
    positions = torch.gather(positions, 1, order.unsqueeze(-1).expand(-1, -1, 3))
    opacities = torch.gather(opacities, 1, order)
    values = torch.gather(values, 1, order)
    covariances = torch.gather(covariances, 1, order.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 3))
    valid = torch.gather(valid, 1, order)

    alpha = torch.where(valid, opacities.clamp(0.0, 1.0), torch.zeros_like(opacities))
    one_minus_alpha = 1.0 - alpha
    inclusive = torch.cumprod(one_minus_alpha, dim=1)
    transmittance = torch.cat([torch.ones_like(inclusive[:, :1]), inclusive[:, :-1]], dim=1)
    weights = transmittance * alpha
    weights = torch.where(valid, weights, torch.zeros_like(weights))
    return positions, values, covariances, weights, valid


def _alpha_risk_chunk(positions, values, covariances, weights, valid, query_points_t, sigma_rbf, rel_jitter, device, dtype):
    """Position-only, single-Gaussian-a_q RKHS risk, scoring the REAL fixed
    weights `weights` (e(w_alpha)^2) rather than solving for the BQ-optimal
    w* -- no Cholesky solve needed at all: padding slots have weight exactly
    0 (see the gather fn), so `w @ Kxx @ w` and `w @ z` already equal the
    real-candidate-only quadratic form regardless of what Kxx/z hold at
    padding entries.

    `jitter_scalar`: the scalar reference (`quadrature._rendering_aware_
    moments`) adds `rel_jitter * mean(diag(kxx))` to Kxx's diagonal before
    scoring any weight vector against it -- every diagonal entry of an RBF
    Gram matrix is `k_pos(x_i, x_i) = 1/(sigma_rbf*sqrt(2*pi))^3` (the
    self-distance is always 0, so this doesn't depend on position or on how
    many real candidates a given query has), so this constant is exactly
    that mean regardless of candidate count -- confirmed the hard way: an
    earlier version of this function omitted this jitter term entirely and
    disagreed with the scalar path by ~1% on real risk values (a real,
    caught-by-cross-validation bug, not a rounding difference). Since
    jitter only touches the diagonal, `w @ (Kxx + jitter*I) @ w = w @ Kxx @ w
    + jitter * sum(w_i^2)` -- added as a separate term below rather than
    folded into `k_pos_mat` itself."""
    eye3 = torch.eye(3, dtype=dtype, device=device)
    total_weight = weights.sum(dim=1)
    has_weight = total_weight > 0
    safe_total = total_weight.clamp_min(1e-300)
    normalized = weights / safe_total.unsqueeze(1)

    mean = (normalized.unsqueeze(-1) * positions).sum(dim=1)
    diff = positions - mean.unsqueeze(1)
    spread = torch.einsum("pk,pki,pkj->pij", normalized, diff, diff)
    within = torch.einsum("pk,pkij->pij", normalized, covariances)
    covariance = spread + within + 1e-6 * eye3

    # Fallback when every candidate has weight 0 (nothing on-ray, or truly
    # zero candidates): eye3 covariance, total_mass=1e-12 -- the exact
    # covariance scale doesn't matter once total_mass is negligible, so this
    # function stays radius-free.
    mean = torch.where(has_weight.view(-1, 1), mean, query_points_t)
    covariance = torch.where(has_weight.view(-1, 1, 1), covariance, eye3.unsqueeze(0))
    total_mass = torch.where(has_weight, total_weight, torch.full_like(total_weight, 1e-12))

    d = 3
    cov0 = 2 * covariance + (sigma_rbf**2) * eye3
    logdet0 = torch.linalg.slogdet(cov0)[1]
    z0 = (total_mass**2) / ((2 * np.pi) ** (d / 2.0) * torch.exp(0.5 * logdet0))

    cov_pos = covariance + (sigma_rbf**2) * eye3
    inv_pos = torch.linalg.inv(cov_pos)
    logdet_pos = torch.linalg.slogdet(cov_pos)[1]
    norm_const = (2 * np.pi) ** (d / 2.0) * torch.exp(0.5 * logdet_pos)

    diff_q = mean.unsqueeze(1) - positions
    quad = torch.einsum("pki,pij,pkj->pk", diff_q, inv_pos, diff_q)
    gauss = torch.exp(-0.5 * quad) / norm_const.unsqueeze(1)
    z = total_mass.unsqueeze(1) * gauss
    z = torch.where(valid, z, torch.zeros_like(z))

    pos_dist2 = torch.cdist(positions, positions) ** 2
    k_pos_mat = torch.exp(-pos_dist2 / (2 * sigma_rbf**2)) / (sigma_rbf * np.sqrt(2 * np.pi)) ** 3
    jitter_scalar = rel_jitter / (sigma_rbf * np.sqrt(2 * np.pi)) ** 3

    w = weights
    risk = (
        z0 - 2.0 * (w * z).sum(dim=1) + torch.einsum("pk,pkj,pj->p", w, k_pos_mat, w) + jitter_scalar * (w**2).sum(dim=1)
    )
    alpha_mean = (w * values).sum(dim=1)
    return alpha_mean.cpu().numpy(), risk.clamp_min(0.0).cpu().numpy()


def compute_alpha_risk_batched(
    engine,
    camera_index: CameraSplatIndex,
    query_points: np.ndarray,
    angular_tol: float,
    sigma_rbf: float,
    max_candidates: int = 500,
    rel_jitter: float = 1e-4,
    device: str = "cuda",
    pixel_chunk_bytes: float = 1.5e9,
) -> np.ndarray:
    """Batched equivalent of calling
    `engine.rendering_aware_alpha_risk_along_ray(query_points[p],
    camera_index, radius, angular_tol=angular_tol,
    max_candidates=max_candidates, sigma_rbf=sigma_rbf)` once per row p and
    collecting `.alpha_mean`/`.alpha_risk` -- this is u_spatial_BQ(q) (see
    this module's own docstring). `radius` doesn't appear: it's only used
    by the scalar path for a fallback covariance on a near-zero-weight
    query, and that covariance's exact scale doesn't matter once its
    total_mass is negligible.

    Returns `(alpha_mean, alpha_risk)`, each (P,) float64, same order as
    `query_points`.
    """
    dtype = torch.float64
    query_points = np.atleast_2d(np.asarray(query_points, dtype=float))
    p_total = query_points.shape[0]
    d = query_points.shape[1]
    k_cap = min(max_candidates, camera_index.indices.shape[0])
    if k_cap == 0:
        cov = torch.zeros((p_total, d, d), dtype=dtype, device=device)
        cov[:, torch.arange(d), torch.arange(d)] = 1e-12
        z0 = _zero_candidate_variance(cov, sigma_rbf, d)
        return np.zeros(p_total, dtype=np.float64), z0

    m = camera_index.indices.shape[0]
    index_bearings = torch.tensor(camera_index.bearings, dtype=dtype, device=device)
    index_depths = torch.tensor(camera_index.depths, dtype=dtype, device=device)
    index_positions = torch.tensor(engine.positions[camera_index.indices], dtype=dtype, device=device)
    index_values = torch.tensor(engine.values[camera_index.indices], dtype=dtype, device=device)
    index_opacities = torch.tensor(engine.opacities[camera_index.indices], dtype=dtype, device=device)
    if engine.scales is not None and engine.rotations is not None:
        index_covariances = torch.tensor(engine.covariances()[camera_index.indices], dtype=dtype, device=device)
    else:
        index_covariances = torch.zeros(camera_index.indices.shape[0], 3, 3, dtype=dtype, device=device)

    bytes_per_pixel_row = m * 8 * 2  # dist + masked_dist, this path's only (P, M) matrices
    chunk_size = max(1, min(p_total, int(pixel_chunk_bytes / max(bytes_per_pixel_row, 1))))

    alpha_means = np.empty(p_total, dtype=np.float64)
    alpha_risks = np.empty(p_total, dtype=np.float64)
    for start in range(0, p_total, chunk_size):
        end = min(start + chunk_size, p_total)
        positions, values, covariances, weights, valid = _gather_position_only_candidates_and_weights(
            camera_index, query_points[start:end], angular_tol, k_cap, device, dtype,
            index_bearings, index_depths, index_positions, index_values, index_opacities, index_covariances,
        )
        query_points_t = torch.tensor(query_points[start:end], dtype=dtype, device=device)
        alpha_means[start:end], alpha_risks[start:end] = _alpha_risk_chunk(
            positions, values, covariances, weights, valid, query_points_t, sigma_rbf, rel_jitter, device, dtype,
        )
    return alpha_means, alpha_risks
