"""Batched-GPU implementation of `pixel_uncertainty.LocalUncertaintyEngine
.rendering_aware_variance_along_ray_directional`, for computing an entire
view's worth of per-pixel directional BQ variance in one batched torch pass
instead of one Python call per pixel.

Why this exists: profiling `render_reconstruction.compute_uncertainty_maps`
on a real checkpoint showed its per-pixel loop (~1600 valid pixels/view)
split roughly 21% candidate lookup (`CameraSplatIndex.query`, a scipy KD-tree
ball query called once per pixel) + 15% the actual BQ linear algebra (kernel
Gram matrix + Cholesky solve, also once per pixel) -- both are naturally
batchable across pixels (fixed-shape per-pixel work, independent across
pixels) and neither depends on the other pixel-by-pixel, so there is no
correctness reason to keep them in a Python loop. This module batches both;
`compute_directional_variance_batched` is the entry point.

Every formula here is the *same* closed-form math
`quadrature.bayesian_quadrature_rendering_aware_directional` /
`pixel_uncertainty._render_weight_from_local_weights` /
`visibility_attribution.ray_transmittance_weights` already implement --
this is a batched re-expression, not a different or approximate model.
Correctness is argued in-line at each step (see comments) and verified
directly against the scalar path in
tests/gs_experiment/test_gpu_uncertainty.py (exact per-pixel agreement on
real data, not just a smoke test).

Padding: each pixel's real candidate count varies, but a batched tensor
needs a fixed size. Padding candidate slots are made *exactly* inert by
construction, not merely small: their Gram-matrix row/column is replaced
with an identity, making `K` block-diagonal (real block ⊕ identity padding
block); since the block-diagonal inverse decouples the two blocks, this is
mathematically identical to solving the real-candidate-only system, not an
approximation (see `_padded_gram_matrix`'s docstring for the short proof).

Needs torch (a GPU is not strictly required but is the point). Uses
float64 throughout to match the scalar path's numpy/scipy float64
precision -- these variances are already tiny (~1e-5-1e-4), and this
module trades some GPU throughput (float64 matmul/Cholesky is slower than
float32 on most consumer GPUs) for staying numerically comparable to the
already-validated scalar implementation, rather than introducing a new
source of disagreement on top of the batching rewrite itself.
"""

from __future__ import annotations

import numpy as np
import torch

from gs_experiment.camera import CameraPose
from gs_experiment.visibility_attribution import CameraSplatIndex, project_to_camera_local


def _padded_gram_matrix(k_pos: torch.Tensor, k_dir: torch.Tensor, valid: torch.Tensor, rel_jitter: float) -> torch.Tensor:
    """K = k_pos * k_dir, with every row/column touching a padding index
    (valid == False) replaced by an identity row/column.

    Proof this doesn't change the real-block answer: for block-diagonal
    M = [[A, 0], [0, B]], M^-1 = [[A^-1, 0], [0, B^-1]] -- solving M x = b
    with b = [b_real, b_pad] gives x = [A^-1 b_real, B^-1 b_pad], so the
    real block's solve is exactly A^-1 b_real, unaffected by B or b_pad.
    Here A is the true candidate-candidate Gram matrix (+ jitter) and B is
    the identity, so this recovers exactly the scalar path's `kxx`, and the
    padding block's arbitrary-but-well-defined solve never leaks into it.
    """
    k = k_pos * k_dir
    valid_f = valid.to(k.dtype)
    k = k * valid_f.unsqueeze(-1) * valid_f.unsqueeze(-2)  # zero any row/col touching padding (incl. padding diagonal)
    n_real = valid_f.sum(dim=-1).clamp_min(1.0)
    diag_sum = torch.diagonal(k, dim1=-2, dim2=-1).sum(dim=-1)
    jitter = (rel_jitter * diag_sum / n_real).view(-1, 1, 1)
    eye = torch.eye(k.shape[-1], dtype=k.dtype, device=k.device).unsqueeze(0)
    pad_diag = torch.diag_embed(1.0 - valid_f)  # 1 on the diagonal exactly at padding slots
    return k + jitter * eye + pad_diag


def compute_directional_variance_batched(
    engine,
    camera_index: CameraSplatIndex,
    query_points: np.ndarray,
    query_directions: np.ndarray,
    angular_tol: float,
    sigma_rbf: float,
    kappa: float,
    max_candidates: int = 500,
    rel_jitter: float = 1e-4,
    device: str = "cuda",
    pixel_chunk_bytes: float = 1.5e9,
) -> np.ndarray:
    """Batched equivalent of calling
    `engine.rendering_aware_variance_along_ray_directional(query_points[p],
    query_directions[p], camera_index, radius, angular_tol=angular_tol,
    max_candidates=max_candidates, sigma_rbf=sigma_rbf)` once per row p and
    collecting `.variance` -- `radius` itself doesn't appear because the
    scalar path only ever uses it for a fallback covariance on a
    zero-candidate query, reproduced below via the same `total_weight <= 0`
    branch. `engine.dir_kernel.kappa` is passed explicitly as `kappa`
    (rather than reading it off `engine`) so this function has no other
    dependency on `engine` beyond the arrays it already holds.

    `camera_index`'s candidate pool M is the *whole* camera-expanded
    observation array filtered only by depth>0/in-frustum -- on a real
    360-degree-orbit dataset this is close to the entire scene (confirmed:
    ~1.15M rows on a real lego checkpoint, not the few-hundred-per-window
    scale the scalar path's KD-tree ball query only ever materializes
    implicitly). A dense (P, M) distance/alignment matrix at that M is a
    multi-GB allocation per matrix even for a modest pixel batch -- so
    pixels are processed in chunks sized to keep each chunk's dense
    matrices under `pixel_chunk_bytes` (float64: 8 bytes/element, and this
    function needs a small constant number of (P_chunk, M)-shaped
    matrices alive at once), not batched all at once. The per-camera index
    tensors (bearings/directions/depths/positions/values/opacities/
    covariances -- all O(M), independent of which pixels are being
    queried) are moved to `device` once, outside the chunk loop, so a
    multi-chunk view doesn't repeatedly re-transfer the same ~1.15M-row
    arrays.

    Returns `(variance, prior_variance)`, each (P,) float64, same order as
    `query_points`. `prior_variance` (z_0: the variance this same query
    would have with zero relevant observations -- the posterior's own
    ceiling, since posterior variance = prior variance - a nonnegative
    reduction term) is returned alongside the posterior `variance` so a
    caller can express uncertainty as a *ratio* `variance / prior_variance`
    in [0, 1] -- a scale-robust, cross-scene-comparable quantity, unlike
    the raw posterior variance whose absolute magnitude is dominated by
    `sigma_rbf` (see the conversation this was added for: two figures
    using different sigma values produced posterior variances differing by
    four orders of magnitude on the *same* checkpoint, purely from that
    choice -- the ratio to each query's own prior is far less sensitive to
    it, since both numerator and denominator scale together).
    """
    dtype = torch.float64
    query_points = np.atleast_2d(np.asarray(query_points, dtype=float))
    query_directions = np.atleast_2d(np.asarray(query_directions, dtype=float))
    p_total = query_points.shape[0]
    d = query_points.shape[1]
    k_cap = min(max_candidates, camera_index.indices.shape[0])
    if k_cap == 0:
        # No candidates observed this camera at all -- every query collapses
        # to the same zero-candidate fallback the scalar path returns (a_q's
        # own self-variance, z = 0 since there are no nodes to condition on).
        # Posterior == prior here (no data to reduce it), so both outputs match.
        cov = torch.full((p_total, d, d), 0.0, dtype=dtype, device=device)
        cov[:, torch.arange(d), torch.arange(d)] = 1e-12
        z0 = _zero_candidate_variance(cov, sigma_rbf, d)
        return z0, z0.copy()

    m = camera_index.indices.shape[0]
    index_bearings = torch.tensor(camera_index.bearings, dtype=dtype, device=device)  # (M, 2)
    index_directions = torch.tensor(camera_index.directions, dtype=dtype, device=device)  # (M, 3)
    index_depths = torch.tensor(camera_index.depths, dtype=dtype, device=device)  # (M,)
    index_positions = torch.tensor(engine.positions[camera_index.indices], dtype=dtype, device=device)  # (M, 3)
    index_values = torch.tensor(engine.values[camera_index.indices], dtype=dtype, device=device)  # (M,)
    index_opacities = torch.tensor(engine.opacities[camera_index.indices], dtype=dtype, device=device)  # (M,)
    # Matches _along_ray_local_data's own fallback: candidate covariances are optional
    # (only real if the engine was built with scales/rotations); zero, not an error, when
    # unset -- moment-matching then falls back to the spread-of-centers term alone, same
    # as the scalar path's `within=0` when `local_covariances is None`.
    if engine.scales is not None and engine.rotations is not None:
        index_covariances = torch.tensor(engine.covariances()[camera_index.indices], dtype=dtype, device=device)
    else:
        index_covariances = torch.zeros(camera_index.indices.shape[0], 3, 3, dtype=dtype, device=device)

    # 4 concurrent (P_chunk, M) float64 matrices (dist, in_radius-as-float
    # internally, alignment, masked_alignment) is a conservative constant
    # to size chunks against -- better to under-use the budget than OOM.
    bytes_per_pixel_row = m * 8 * 4
    chunk_size = max(1, min(p_total, int(pixel_chunk_bytes / max(bytes_per_pixel_row, 1))))

    variances = np.empty(p_total, dtype=np.float64)
    prior_variances = np.empty(p_total, dtype=np.float64)
    for start in range(0, p_total, chunk_size):
        end = min(start + chunk_size, p_total)
        variances[start:end], prior_variances[start:end] = _compute_chunk(
            engine, camera_index, query_points[start:end], query_directions[start:end],
            angular_tol, sigma_rbf, kappa, k_cap, rel_jitter, device, dtype,
            index_bearings, index_directions, index_depths, index_positions, index_values,
            index_opacities, index_covariances,
        )
    return variances, prior_variances


def _gather_candidates_and_weights(
    camera_index, query_points, query_directions, angular_tol, k_cap, device, dtype,
    index_bearings, index_directions, index_depths, index_positions, index_values,
    index_opacities, index_covariances,
):
    """Shared by both quadrature variants below (single-Gaussian and mixture
    a_q): candidate search, depth-ordering, and real transmittance weights --
    identical for both, since they only differ in how the resulting
    (position, covariance, weight) triples per candidate are turned into a_q.
    """
    # --- 1. reference bearings for every query point in this chunk (one
    # batched call, replacing what the scalar path did once per pixel) ---
    ref_bx, ref_by, _ = project_to_camera_local(query_points, camera_index.camera)
    ref_bearing = torch.tensor(np.stack([ref_bx, ref_by], axis=1), dtype=dtype, device=device)  # (P, 2)
    query_dir_t = torch.tensor(query_directions, dtype=dtype, device=device)  # (P, 3)

    # --- 2. candidate search: exact ball-query semantics via a dense
    # distance matrix (equivalent to the scalar path's cKDTree.query_ball_point
    # at the same angular_tol -- not an approximation, just not tree-accelerated),
    # then rank overflow by directional alignment (matching
    # CameraSplatIndex.query's query_direction branch, which is what every
    # caller of this directional path uses) and keep the top k_cap. ---
    dist = torch.cdist(ref_bearing, index_bearings)  # (P, M)
    in_radius = dist < angular_tol
    alignment = query_dir_t @ index_directions.T  # (P, M)
    masked_alignment = torch.where(in_radius, alignment, torch.full_like(alignment, float("-inf")))
    top_values, top_idx = torch.topk(masked_alignment, k=k_cap, dim=1)  # (P, k_cap) each
    valid = torch.isfinite(top_values)  # False exactly where a pixel had fewer than k_cap real candidates

    # Gather every per-candidate quantity into this pixel's own padded slots.
    positions = index_positions[top_idx]  # (P, k_cap, 3)
    directions = index_directions[top_idx]
    depths = index_depths[top_idx]
    opacities = index_opacities[top_idx]
    values = index_values[top_idx]
    covariances = index_covariances[top_idx]

    # --- 3. real, depth-ordered alpha-compositing transmittance weights,
    # per pixel (ray_transmittance_weights, batched): sort each pixel's
    # candidates by depth, then w_i = T_i * alpha_i with
    # T_i = prod_{j closer} (1 - alpha_j), via an exclusive cumprod. ---
    depths_for_sort = torch.where(valid, depths, torch.full_like(depths, float("inf")))
    order = torch.argsort(depths_for_sort, dim=1)
    positions = torch.gather(positions, 1, order.unsqueeze(-1).expand(-1, -1, 3))
    directions = torch.gather(directions, 1, order.unsqueeze(-1).expand(-1, -1, 3))
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
    return positions, directions, values, covariances, weights, valid, query_dir_t


def _compute_chunk(
    engine, camera_index, query_points, query_directions, angular_tol, sigma_rbf, kappa, k_cap, rel_jitter,
    device, dtype, index_bearings, index_directions, index_depths, index_positions, index_values,
    index_opacities, index_covariances,
):
    positions, directions, values, covariances, weights, valid, query_dir_t = _gather_candidates_and_weights(
        camera_index, query_points, query_directions, angular_tol, k_cap, device, dtype,
        index_bearings, index_directions, index_depths, index_positions, index_values,
        index_opacities, index_covariances,
    )

    # --- 4. render weight a_q: moment-match the weighted candidates into a
    # single Gaussian (mean, covariance), with A = total_mass exactly (see
    # module docstring: from_total_mass makes amplitude*(2pi)^(d/2)|cov|^0.5
    # equal total_mass by construction, so it never needs to be computed). ---
    total_weight = weights.sum(dim=1)  # (P,)
    has_weight = total_weight > 0
    safe_total = total_weight.clamp_min(1e-300)
    normalized = weights / safe_total.unsqueeze(1)

    mean = (normalized.unsqueeze(-1) * positions).sum(dim=1)  # (P, 3)
    diff = positions - mean.unsqueeze(1)
    spread = torch.einsum("pk,pki,pkj->pij", normalized, diff, diff)
    within = torch.einsum("pk,pkij->pij", normalized, covariances)
    eye3 = torch.eye(3, dtype=dtype, device=device)
    covariance = spread + within + 1e-6 * eye3

    fallback_covariance = eye3  # matches the scalar path's (radius/2)^2*I only up to scale;
    # z/z0 use A=1e-12 in the fallback case regardless (see below), so this
    # covariance's exact scale doesn't affect the returned variance -- it
    # only would if A were non-negligible, which it isn't in this branch.
    mean = torch.where(has_weight.view(-1, 1), mean, torch.tensor(query_points, dtype=dtype, device=device))
    covariance = torch.where(has_weight.view(-1, 1, 1), covariance, fallback_covariance.unsqueeze(0))
    total_mass = torch.where(has_weight, total_weight, torch.full_like(total_weight, 1e-12))

    return _solve_batched(
        total_mass, mean, covariance, positions, directions, values, valid, query_dir_t, sigma_rbf, kappa, rel_jitter,
    )


def _zero_candidate_variance(covariance: torch.Tensor, sigma_rbf: float, d: int) -> np.ndarray:
    a = 1e-12
    cov_sum2 = 2 * covariance + (sigma_rbf**2) * torch.eye(d, dtype=covariance.dtype, device=covariance.device)
    logdet = torch.linalg.slogdet(cov_sum2)[1]
    z0 = (a**2) / ((2 * np.pi) ** (d / 2.0) * torch.exp(0.5 * logdet))
    return z0.clamp_min(0.0).cpu().numpy()


def compute_directional_variance_batched_mixture(
    engine,
    camera_index: CameraSplatIndex,
    query_points: np.ndarray,
    query_directions: np.ndarray,
    angular_tol: float,
    sigma_rbf: float,
    kappa: float,
    max_candidates: int = 100,
    rel_jitter: float = 1e-4,
    device: str = "cuda",
    pixel_chunk_bytes: float = 1.0e9,
) -> np.ndarray:
    """Exploratory alternative to compute_directional_variance_batched: models
    a_q as a MIXTURE of K Gaussians -- one per real local candidate, using
    each candidate's own real weight/position/covariance directly -- instead
    of moment-matching all K candidates into a single Gaussian. Written to
    test whether that moment-matching collapse (a candidate's own real
    per-splat compositing structure discarded in favor of one blob per
    query) materially changes the resulting uncertainty map, e.g. its
    spatial "blobbiness".

    Still exactly closed-form against a Gaussian k_pos, by the same
    Gaussian-product identity used throughout this module, applied once per
    (component, target) pair instead of once per query:

        a_q(x) = sum_k w_k * N(x; x_k, Sigma_k)
        z_i    = [sum_k w_k * N(x_k; x_i, Sigma_k + sigma_rbf^2 I)] * k_dir(d_i, d_query)
        z_0    = sum_k sum_l w_k*w_l * N(x_k; x_l, Sigma_k+Sigma_l+sigma_rbf^2 I)

    (w_k = A_k directly, not w_k/normalizer: N(.;x_k,Sigma_k) is already a
    normalized density integrating to 1, so its mass under weight A_k is
    just A_k -- matches the codebase's existing "A = total_mass" convention,
    see compute_directional_variance_batched's docstring. Verified against
    brute-force nquad integration on a 1D toy mixture during development,
    not checked into the test suite as an ongoing regression test since this
    is an exploratory alternative, not the validated production path.)

    O(K^2) per pixel (a pairwise K x K Gaussian evaluation), vs. the
    production path's O(K) -- max_candidates defaults far lower accordingly.
    Otherwise shares candidate search/depth-ordering/transmittance-weight
    code exactly with compute_directional_variance_batched (see
    _gather_candidates_and_weights) and K (the Gram matrix) is identical
    between the two paths (it doesn't depend on a_q's shape at all).

    Returns (P,) variances, float64, same order as `query_points`.
    """
    dtype = torch.float64
    query_points = np.atleast_2d(np.asarray(query_points, dtype=float))
    query_directions = np.atleast_2d(np.asarray(query_directions, dtype=float))
    p_total = query_points.shape[0]
    d = query_points.shape[1]
    k_cap = min(max_candidates, camera_index.indices.shape[0])
    if k_cap == 0:
        cov = torch.full((p_total, d, d), 0.0, dtype=dtype, device=device)
        cov[:, torch.arange(d), torch.arange(d)] = 1e-12
        return _zero_candidate_variance(cov, sigma_rbf, d)

    m = camera_index.indices.shape[0]
    index_bearings = torch.tensor(camera_index.bearings, dtype=dtype, device=device)
    index_directions = torch.tensor(camera_index.directions, dtype=dtype, device=device)
    index_depths = torch.tensor(camera_index.depths, dtype=dtype, device=device)
    index_positions = torch.tensor(engine.positions[camera_index.indices], dtype=dtype, device=device)
    index_values = torch.tensor(engine.values[camera_index.indices], dtype=dtype, device=device)
    index_opacities = torch.tensor(engine.opacities[camera_index.indices], dtype=dtype, device=device)
    if engine.scales is not None and engine.rotations is not None:
        index_covariances = torch.tensor(engine.covariances()[camera_index.indices], dtype=dtype, device=device)
    else:
        index_covariances = torch.zeros(camera_index.indices.shape[0], 3, 3, dtype=dtype, device=device)

    # (P_chunk, k_cap, k_cap) pairwise 3x3-covariance/quadratic-form tensors dominate
    # memory here (O(K^2) vs. the production path's O(K)), so chunks are sized much
    # smaller accordingly.
    bytes_per_pixel_row = k_cap * k_cap * 9 * 8 * 4
    chunk_size = max(1, min(p_total, int(pixel_chunk_bytes / max(bytes_per_pixel_row, 1))))

    variances = np.empty(p_total, dtype=np.float64)
    for start in range(0, p_total, chunk_size):
        end = min(start + chunk_size, p_total)
        positions, directions, values, covariances, weights, valid, query_dir_t = _gather_candidates_and_weights(
            camera_index, query_points[start:end], query_directions[start:end], angular_tol, k_cap, device, dtype,
            index_bearings, index_directions, index_depths, index_positions, index_values,
            index_opacities, index_covariances,
        )
        variances[start:end] = _solve_batched_mixture(
            positions, directions, values, covariances, weights, valid, query_dir_t, sigma_rbf, kappa, rel_jitter,
        )
    return variances


def _pairwise_gaussian_density(mu_a, mu_b, cov_pair):
    """N(mu_a[p,i]; mu_b[p,k], cov_pair[p,i,k]) for every (i,k) pair at once.
    mu_a: (P,I,3), mu_b: (P,K,3), cov_pair: (P,I,K,3,3). Returns (P,I,K)."""
    diff = mu_a.unsqueeze(2) - mu_b.unsqueeze(1)  # (P, I, K, 3)
    inv = torch.linalg.inv(cov_pair)  # (P, I, K, 3, 3)
    logdet = torch.linalg.slogdet(cov_pair)[1]  # (P, I, K)
    quad = torch.einsum("pikd,pikde,pike->pik", diff, inv, diff)
    norm_const = (2 * np.pi) ** 1.5 * torch.exp(0.5 * logdet)
    return torch.exp(-0.5 * quad) / norm_const


def _solve_batched_mixture(positions, directions, values, covariances, weights, valid, query_dir_t, sigma_rbf, kappa, rel_jitter):
    dtype = covariances.dtype
    device = covariances.device
    eye3 = torch.eye(3, dtype=dtype, device=device)
    p, k = positions.shape[0], positions.shape[1]

    # z_0 = sum_k sum_l w_k*w_l*N(x_k;x_l,Sigma_k+Sigma_l+sigma^2 I). Padding
    # candidates have weight 0 (see _gather_candidates_and_weights), so they
    # contribute exactly 0 to every sum below regardless of their (dummy,
    # gathered-but-unused) position/covariance values -- no identity-block
    # trick needed here, unlike the Gram matrix's Cholesky solve.
    cov0_pair = covariances.unsqueeze(2) + covariances.unsqueeze(1) + (sigma_rbf**2) * eye3  # (P,K,K,3,3)
    gauss0 = _pairwise_gaussian_density(positions, positions, cov0_pair)  # (P,K,K)
    z0 = torch.einsum("pk,pl,pkl->p", weights, weights, gauss0)

    # z_i = [sum_k w_k*N(x_k;x_i,Sigma_k+sigma^2 I)] * k_dir(d_i,d_query).
    # Sigma_k+sigma^2*I depends only on the *component* k, broadcast across
    # every target i.
    cov_k = covariances + (sigma_rbf**2) * eye3  # (P,K,3,3)
    cov_pair = cov_k.unsqueeze(1).expand(-1, k, -1, -1, -1)  # (P,I=K,K,3,3): component cov, same for every target i
    gauss = _pairwise_gaussian_density(positions, positions, cov_pair)  # (P,I,K): N(x_k; x_i, Sigma_k+sigma^2 I)
    z_raw = torch.einsum("pk,pik->pi", weights, gauss)
    dir_align = torch.einsum("pki,pi->pk", directions, query_dir_t)
    k_dir_query = torch.exp(kappa * (dir_align - 1.0))
    z = z_raw * k_dir_query
    z = torch.where(valid, z, torch.zeros_like(z))

    pos_dist2 = torch.cdist(positions, positions) ** 2
    k_pos_mat = torch.exp(-pos_dist2 / (2 * sigma_rbf**2)) / (sigma_rbf * np.sqrt(2 * np.pi)) ** 3
    dot_mat = torch.einsum("pki,pji->pkj", directions, directions)
    k_dir_mat = torch.exp(kappa * (dot_mat - 1.0))
    kxx = _padded_gram_matrix(k_pos_mat, k_dir_mat, valid, rel_jitter)

    values_masked = torch.where(valid, values, torch.zeros_like(values))
    rhs = torch.stack([values_masked, z], dim=-1)
    chol = torch.linalg.cholesky(kxx)
    solved = torch.cholesky_solve(rhs, chol)
    solved_z = solved[..., 1]

    variance = z0 - (z * solved_z).sum(dim=1)
    return variance.clamp_min(0.0).cpu().numpy()


def _solve_batched(total_mass, mean, covariance, positions, directions, values, valid, query_dir_t, sigma_rbf, kappa, rel_jitter):
    dtype = covariance.dtype
    device = covariance.device
    d = 3
    eye3 = torch.eye(d, dtype=dtype, device=device)

    # z0 = A^2 * N(mu_q; mu_q, 2*Sigma_q + sigma_rbf^2 I) -- a self-density,
    # independent of any candidate (see quadrature.rendering_aware_prior_variance).
    cov0 = 2 * covariance + (sigma_rbf**2) * eye3
    logdet0 = torch.linalg.slogdet(cov0)[1]
    z0 = (total_mass**2) / ((2 * np.pi) ** (d / 2.0) * torch.exp(0.5 * logdet0))

    # z_i = A * N(mu_q; x_i, Sigma_q + sigma_rbf^2 I) * k_dir(d_i, d_query)
    # (see quadrature.rendering_aware_moment_vector / bayesian_quadrature_
    # rendering_aware_directional). Sigma_q+sigma_rbf^2*I depends only on
    # the pixel (not the candidate), so its inverse/logdet are computed once
    # per pixel and reused across all k_cap candidates.
    cov_pos = covariance + (sigma_rbf**2) * eye3
    inv_pos = torch.linalg.inv(cov_pos)
    logdet_pos = torch.linalg.slogdet(cov_pos)[1]
    norm_const = (2 * np.pi) ** (d / 2.0) * torch.exp(0.5 * logdet_pos)

    diff = mean.unsqueeze(1) - positions  # (P, K, 3)
    quad = torch.einsum("pki,pij,pkj->pk", diff, inv_pos, diff)
    gauss = torch.exp(-0.5 * quad) / norm_const.unsqueeze(1)
    dir_align = torch.einsum("pki,pi->pk", directions, query_dir_t)
    k_dir_query = torch.exp(kappa * (dir_align - 1.0))
    z = total_mass.unsqueeze(1) * gauss * k_dir_query
    z = torch.where(valid, z, torch.zeros_like(z))

    # K_ij = k_pos(x_i,x_j) * k_dir(d_i,d_j) (nodes are noiseless
    # observations of c(x_i), never weighted by a_q -- see quadrature.py's
    # module docstring on the double-quad bug this avoids).
    pos_dist2 = torch.cdist(positions, positions) ** 2
    k_pos_mat = torch.exp(-pos_dist2 / (2 * sigma_rbf**2)) / (sigma_rbf * np.sqrt(2 * np.pi)) ** 3
    dot_mat = torch.einsum("pki,pji->pkj", directions, directions)
    k_dir_mat = torch.exp(kappa * (dot_mat - 1.0))
    kxx = _padded_gram_matrix(k_pos_mat, k_dir_mat, valid, rel_jitter)

    values_masked = torch.where(valid, values, torch.zeros_like(values))
    rhs = torch.stack([values_masked, z], dim=-1)  # (P, K, 2)
    chol = torch.linalg.cholesky(kxx)
    solved = torch.cholesky_solve(rhs, chol)  # (P, K, 2)
    solved_z = solved[..., 1]

    variance = z0 - (z * solved_z).sum(dim=1)
    return variance.clamp_min(0.0).cpu().numpy(), z0.clamp_min(0.0).cpu().numpy()
