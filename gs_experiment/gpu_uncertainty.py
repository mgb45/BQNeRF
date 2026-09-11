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

from typing import Optional

import numpy as np
import torch

from gs_experiment.camera import CameraPose
from gs_experiment.visibility_attribution import CameraSplatIndex, project_to_camera_local


def _padded_gram_matrix(
    k_pos: torch.Tensor, k_dir: torch.Tensor, valid: torch.Tensor, rel_jitter: float, noise_variance: float = 0.0
) -> torch.Tensor:
    """K = k_pos * k_dir, with every row/column touching a padding index
    (valid == False) replaced by an identity row/column.

    Proof this doesn't change the real-block answer: for block-diagonal
    M = [[A, 0], [0, B]], M^-1 = [[A^-1, 0], [0, B^-1]] -- solving M x = b
    with b = [b_real, b_pad] gives x = [A^-1 b_real, B^-1 b_pad], so the
    real block's solve is exactly A^-1 b_real, unaffected by B or b_pad.
    Here A is the true candidate-candidate Gram matrix (+ jitter +
    noise_variance) and B is the identity, so this recovers exactly the
    scalar path's `kxx`, and the padding block's arbitrary-but-well-defined
    solve never leaks into it.

    `noise_variance` (default 0.0, unchanged behavior): added to the whole
    diagonal alongside `jitter`, same as `jitter` itself -- see
    `gs_experiment.quadrature._rendering_aware_moments`'s docstring for the
    real homoscedastic observation-noise model this implements. Safe to add
    to the padding-block diagonal too (unlike `jitter`, which is already
    added there): the block-diagonal argument above only needs `B` positive
    definite, not exactly the identity, and `pad_diag`'s own `1.0` already
    dominates any reasonable `noise_variance`.
    """
    k = k_pos * k_dir
    valid_f = valid.to(k.dtype)
    k = k * valid_f.unsqueeze(-1) * valid_f.unsqueeze(-2)  # zero any row/col touching padding (incl. padding diagonal)
    n_real = valid_f.sum(dim=-1).clamp_min(1.0)
    diag_sum = torch.diagonal(k, dim1=-2, dim2=-1).sum(dim=-1)
    jitter = (rel_jitter * diag_sum / n_real).view(-1, 1, 1)
    eye = torch.eye(k.shape[-1], dtype=k.dtype, device=k.device).unsqueeze(0)
    pad_diag = torch.diag_embed(1.0 - valid_f)  # 1 on the diagonal exactly at padding slots
    return k + (jitter + noise_variance) * eye + pad_diag


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
    noise_variance: float = 0.0,
    return_mean: bool = False,
    values_rgb: Optional[np.ndarray] = None,
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

    `noise_variance` (default 0.0, unchanged behavior): a real
    homoscedastic observation-noise variance added to the Gram matrix's
    diagonal -- see `gs_experiment.quadrature._rendering_aware_moments`'s
    docstring for the model and motivation (real splat positions routinely
    include near-duplicate points, which makes the noiseless posterior
    ill-conditioned and prone to extreme, oscillatory weights). Threaded
    through to `_padded_gram_matrix` alongside `rel_jitter`.

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

    `return_mean` (default False, unchanged behavior): also return `C_BQ`
    -- the BQ posterior mean at each query point -- as a third array.
    Computed for near-zero extra cost: `_solve_batched`'s existing
    Cholesky solve already factors `Kxx` and solves against `[values, z]`
    jointly (one solve, two right-hand sides), so `C_BQ = z @ Kxx^-1 @
    values` reuses the `Kxx^-1 @ values` column that solve already
    produces (previously computed and discarded) rather than requiring a
    second factorization. On a zero-candidate query, `C_BQ` falls back to
    `0.0` (the GP prior mean, matching `_bq_mean_and_weights`'s own
    scalar-path fallback in `rendering_aware_calibration_experiment.py`).

    `values_rgb` (default None): an optional `(engine.values.shape[0], 3)`
    array -- real per-observation RGB values aligned with `engine.values`'s
    own indexing (see `splat_scene.splat_observations`'s `return_rgb`
    parameter for how to build one), for a genuine 3-channel BQ posterior
    mean instead of the grayscale `C_BQ` above. `engine.values`/`C_BQ`
    stay single-channel by design elsewhere in this project (this
    method's scalar-valued GP machinery), so this is opt-in and additive,
    not a replacement: candidate gathering and `Kxx` itself don't depend
    on color at all, only which right-hand-side column does, so this reuses
    the SAME Cholesky factorization as `variance`/`C_BQ` (2 more solved
    columns, R/G/B, not a second per-channel factorization). When given,
    the returned tuple gains one final `(P, 3)` array, `mean_rgb`, appended
    after whatever `return_mean` already appends -- i.e. `(variance,
    prior_variance, mean_rgb)` if `return_mean=False`, or `(variance,
    prior_variance, mean, mean_rgb)` if `return_mean=True`. Zero-candidate
    queries fall back to `[0.0, 0.0, 0.0]`, the same GP-prior-mean
    convention as the grayscale fallback.
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
        result = (z0, z0.copy())
        if return_mean:
            result = result + (np.zeros(p_total, dtype=np.float64),)
        if values_rgb is not None:
            result = result + (np.zeros((p_total, 3), dtype=np.float64),)
        return result

    m = camera_index.indices.shape[0]
    index_bearings = torch.tensor(camera_index.bearings, dtype=dtype, device=device)  # (M, 2)
    index_directions = torch.tensor(camera_index.directions, dtype=dtype, device=device)  # (M, 3)
    index_depths = torch.tensor(camera_index.depths, dtype=dtype, device=device)  # (M,)
    index_positions = torch.tensor(engine.positions[camera_index.indices], dtype=dtype, device=device)  # (M, 3)
    index_values = torch.tensor(engine.values[camera_index.indices], dtype=dtype, device=device)  # (M,)
    index_values_rgb = (
        torch.tensor(values_rgb[camera_index.indices], dtype=dtype, device=device) if values_rgb is not None else None
    )  # (M, 3)
    # Each candidate's row index into engine.positions/engine.values -- i.e.
    # camera_index.indices itself, moved to `device` once -- used only to
    # canonicalize candidate order before the depth-sort (see
    # _gather_candidates_and_weights's own `index_global_idx` docstring for why:
    # matches the scalar path's tie-break so real depth ties, routine in this
    # camera-expanded observation index, composite identically either way).
    index_global_idx = torch.tensor(camera_index.indices, dtype=torch.int64, device=device)  # (M,)
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
    means = np.empty(p_total, dtype=np.float64) if return_mean else None
    means_rgb = np.empty((p_total, 3), dtype=np.float64) if values_rgb is not None else None
    for start in range(0, p_total, chunk_size):
        end = min(start + chunk_size, p_total)
        chunk_result = _compute_chunk(
            engine, camera_index, query_points[start:end], query_directions[start:end],
            angular_tol, sigma_rbf, kappa, k_cap, rel_jitter, device, dtype,
            index_bearings, index_directions, index_depths, index_positions, index_values,
            index_opacities, index_covariances, noise_variance, return_mean, index_values_rgb,
            index_global_idx,
        )
        idx = 0
        variances[start:end] = chunk_result[idx]
        idx += 1
        prior_variances[start:end] = chunk_result[idx]
        idx += 1
        if return_mean:
            means[start:end] = chunk_result[idx]
            idx += 1
        if values_rgb is not None:
            means_rgb[start:end] = chunk_result[idx]
            idx += 1

    result = (variances, prior_variances)
    if return_mean:
        result = result + (means,)
    if values_rgb is not None:
        result = result + (means_rgb,)
    return result


def _gather_candidates_and_weights(
    camera_index, query_points, query_directions, angular_tol, k_cap, device, dtype,
    index_bearings, index_directions, index_depths, index_positions, index_values,
    index_opacities, index_covariances, index_values_rgb=None, index_global_idx=None,
):
    """Shared by both quadrature variants below (single-Gaussian and mixture
    a_q): candidate search, depth-ordering, and real transmittance weights --
    identical for both, since they only differ in how the resulting
    (position, covariance, weight) triples per candidate are turned into a_q.

    `index_values_rgb` (default None, unchanged behavior otherwise): an
    optional (M, 3) per-candidate RGB value array (see
    `compute_directional_variance_batched`'s own `values_rgb` parameter),
    gathered/depth-reordered by the exact same `top_idx`/`order` as
    `values` so it stays aligned with every other per-candidate quantity.
    Always returns 8 values now (the last is `None` when not requested)
    rather than a variable-length tuple, so every caller's unpacking stays
    fixed-arity regardless of whether RGB was asked for.

    `index_global_idx` (default None): an (M,) tensor of each candidate's
    row index into the original `engine.positions`/`engine.values` arrays
    (i.e. `camera_index.indices` itself, on `device`). When given, gathered
    candidates are first canonicalized to ascending-global-index order,
    THEN depth-sorted with a stable sort -- see the depth-sort comment
    below for why real candidate depths routinely tie exactly (many rows
    here are the same physical splat observed by different training
    cameras) and why this two-step order matters: without a shared
    canonical baseline, `pixel_uncertainty.LocalUncertaintyEngine.
    _along_ray_local_data`'s scalar candidate order and this batched
    path's `torch.topk` order can differ even for the identical candidate
    SET, and neither path's depth-sort is stable against ITS OWN arrival
    order -- so tied-depth candidates could get alpha-composited in a
    different sequence (hence a different transmittance-weighted color)
    between the two paths (confirmed directly: identical 500-candidate
    sets, bit-identical alignment scores, yet BQ color means differing by
    up to 0.11 in [0,1] color space, traced to exactly this). `None`
    (unchanged behavior) skips canonicalization -- every pre-existing
    caller of this function that doesn't pass it is unaffected in the
    sense that its own depth-sort still runs, just without the
    cross-implementation tie-break guarantee.
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
    values_rgb = index_values_rgb[top_idx] if index_values_rgb is not None else None  # (P, k_cap, 3)
    global_idx = index_global_idx[top_idx] if index_global_idx is not None else None  # (P, k_cap)

    if global_idx is not None:
        # Canonicalize to ascending-global-index order first -- tie-free (top_idx
        # never repeats a candidate within one row), so this is a safe, fully
        # deterministic baseline shared with the scalar path's own canonicalization
        # (pixel_uncertainty.LocalUncertaintyEngine._along_ray_local_data). See this
        # function's own `index_global_idx` docstring for why.
        canon_order = torch.argsort(global_idx, dim=1)
        positions = torch.gather(positions, 1, canon_order.unsqueeze(-1).expand(-1, -1, 3))
        directions = torch.gather(directions, 1, canon_order.unsqueeze(-1).expand(-1, -1, 3))
        depths = torch.gather(depths, 1, canon_order)
        opacities = torch.gather(opacities, 1, canon_order)
        values = torch.gather(values, 1, canon_order)
        covariances = torch.gather(covariances, 1, canon_order.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 3))
        if values_rgb is not None:
            values_rgb = torch.gather(values_rgb, 1, canon_order.unsqueeze(-1).expand(-1, -1, 3))
        valid = torch.gather(valid, 1, canon_order)

    # --- 3. real, depth-ordered alpha-compositing transmittance weights,
    # per pixel (ray_transmittance_weights, batched): sort each pixel's
    # candidates by depth, then w_i = T_i * alpha_i with
    # T_i = prod_{j closer} (1 - alpha_j), via an exclusive cumprod. Real
    # candidate depths routinely tie exactly (many rows here are the same
    # physical splat observed by different training cameras -- same position,
    # same depth from any query point), so this sort is `stable=True`,
    # preserving the canonical (global-index) order above for ties rather
    # than an implementation-defined order that need not match the scalar
    # path's own tie-break. ---
    depths_for_sort = torch.where(valid, depths, torch.full_like(depths, float("inf")))
    order = torch.argsort(depths_for_sort, dim=1, stable=True)
    positions = torch.gather(positions, 1, order.unsqueeze(-1).expand(-1, -1, 3))
    directions = torch.gather(directions, 1, order.unsqueeze(-1).expand(-1, -1, 3))
    opacities = torch.gather(opacities, 1, order)
    values = torch.gather(values, 1, order)
    covariances = torch.gather(covariances, 1, order.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 3))
    if values_rgb is not None:
        values_rgb = torch.gather(values_rgb, 1, order.unsqueeze(-1).expand(-1, -1, 3))
    valid = torch.gather(valid, 1, order)

    alpha = torch.where(valid, opacities.clamp(0.0, 1.0), torch.zeros_like(opacities))
    one_minus_alpha = 1.0 - alpha
    inclusive = torch.cumprod(one_minus_alpha, dim=1)
    transmittance = torch.cat([torch.ones_like(inclusive[:, :1]), inclusive[:, :-1]], dim=1)
    weights = transmittance * alpha
    weights = torch.where(valid, weights, torch.zeros_like(weights))
    return positions, directions, values, covariances, weights, valid, query_dir_t, values_rgb


def _compute_chunk(
    engine, camera_index, query_points, query_directions, angular_tol, sigma_rbf, kappa, k_cap, rel_jitter,
    device, dtype, index_bearings, index_directions, index_depths, index_positions, index_values,
    index_opacities, index_covariances, noise_variance=0.0, return_mean=False, index_values_rgb=None,
    index_global_idx=None,
):
    positions, directions, values, covariances, weights, valid, query_dir_t, values_rgb = _gather_candidates_and_weights(
        camera_index, query_points, query_directions, angular_tol, k_cap, device, dtype,
        index_bearings, index_directions, index_depths, index_positions, index_values,
        index_opacities, index_covariances, index_values_rgb, index_global_idx,
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
        noise_variance, return_mean, values_rgb,
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
    noise_variance: float = 0.0,
    return_mean: bool = False,
    values_rgb: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Alternative to compute_directional_variance_batched: models a_q as a
    MIXTURE of K Gaussians -- one per real local candidate, using each
    candidate's own real weight/position/covariance directly -- instead of
    moment-matching all K candidates into a single Gaussian. Written to test
    whether that moment-matching collapse (a candidate's own real per-splat
    compositing structure discarded in favor of one blob per query)
    materially changes the resulting uncertainty map/mean.

    This is the mathematically motivated fix for the mean specifically (see
    the conversation this was built for): with a_q collapsed into one
    Gaussian, the BQ-optimal weight vector w* = Kxx^-1 z has no structural
    relationship to the real alpha-compositing weights w_i = T_i*alpha_i
    for ANY kernel choice. With a_q kept as the exact mixture below, w* = w_i
    exactly whenever Kxx is built the same (unsmeared) way z is -- the
    current code here does exactly that (Kxx below is the plain,
    un-smeared k_pos*k_dir Gram matrix; only z uses each candidate's own
    Sigma_k) -- so w* is now a *controlled* approximation to w_alpha (exact
    in the point-splat limit Sigma_k->0, and close whenever real footprints
    are small relative to sigma_rbf^2), not an unrelated quantity the way
    the single-Gaussian moment-matched a_q's w* was.

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
    brute-force nquad integration on a 1D toy mixture during development;
    now also cross-validated against a scalar reference implementation in
    tests/gs_experiment/test_gpu_uncertainty_mixture.py, matching this
    module's established practice for every other batched path.)

    O(K^2) per pixel (a pairwise K x K Gaussian evaluation), vs. the
    production path's O(K) -- max_candidates defaults far lower accordingly.
    Otherwise shares candidate search/depth-ordering/transmittance-weight
    code exactly with compute_directional_variance_batched (see
    _gather_candidates_and_weights) and K (the Gram matrix) is identical
    between the two paths (it doesn't depend on a_q's shape at all).

    `noise_variance`/`return_mean`/`values_rgb`: see
    compute_directional_variance_batched's own docstrings for these three
    parameters -- identical contract here (`noise_variance` added to Kxx's
    diagonal alongside `rel_jitter`, same as the production path; return
    tuple grows by `mean` then `mean_rgb`, in that order, only for the ones
    actually requested; zero-candidate fallback is 0.0/[0,0,0]).

    Returns `(variance, prior_variance[, mean][, mean_rgb])`, matching
    compute_directional_variance_batched's own return-tuple convention
    (previously just `variance` alone -- now returns `prior_variance` too,
    for the same reason: cheap, already computed as z0, and needed by any
    caller wanting the normalized variance/prior_variance ratio).
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
        z0 = _zero_candidate_variance(cov, sigma_rbf, d)
        result = (z0, z0.copy())
        if return_mean:
            result = result + (np.zeros(p_total, dtype=np.float64),)
        if values_rgb is not None:
            result = result + (np.zeros((p_total, 3), dtype=np.float64),)
        return result

    m = camera_index.indices.shape[0]
    index_bearings = torch.tensor(camera_index.bearings, dtype=dtype, device=device)
    index_directions = torch.tensor(camera_index.directions, dtype=dtype, device=device)
    index_depths = torch.tensor(camera_index.depths, dtype=dtype, device=device)
    index_positions = torch.tensor(engine.positions[camera_index.indices], dtype=dtype, device=device)
    index_values = torch.tensor(engine.values[camera_index.indices], dtype=dtype, device=device)
    index_values_rgb = (
        torch.tensor(values_rgb[camera_index.indices], dtype=dtype, device=device) if values_rgb is not None else None
    )
    index_global_idx = torch.tensor(camera_index.indices, dtype=torch.int64, device=device)
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
    prior_variances = np.empty(p_total, dtype=np.float64)
    means = np.empty(p_total, dtype=np.float64) if return_mean else None
    means_rgb = np.empty((p_total, 3), dtype=np.float64) if values_rgb is not None else None
    for start in range(0, p_total, chunk_size):
        end = min(start + chunk_size, p_total)
        positions, directions, values, covariances, weights, valid, query_dir_t, values_rgb_gathered = (
            _gather_candidates_and_weights(
                camera_index, query_points[start:end], query_directions[start:end], angular_tol, k_cap, device, dtype,
                index_bearings, index_directions, index_depths, index_positions, index_values,
                index_opacities, index_covariances, index_values_rgb, index_global_idx,
            )
        )
        chunk_result = _solve_batched_mixture(
            positions, directions, values, covariances, weights, valid, query_dir_t, sigma_rbf, kappa, rel_jitter,
            noise_variance=noise_variance, return_mean=return_mean, values_rgb=values_rgb_gathered,
        )
        idx = 0
        variances[start:end] = chunk_result[idx]
        idx += 1
        prior_variances[start:end] = chunk_result[idx]
        idx += 1
        if return_mean:
            means[start:end] = chunk_result[idx]
            idx += 1
        if values_rgb is not None:
            means_rgb[start:end] = chunk_result[idx]
            idx += 1

    result = (variances, prior_variances)
    if return_mean:
        result = result + (means,)
    if values_rgb is not None:
        result = result + (means_rgb,)
    return result


def _pairwise_gaussian_density(mu_a, mu_b, cov_pair):
    """N(mu_a[p,i]; mu_b[p,k], cov_pair[p,i,k]) for every (i,k) pair at once.
    mu_a: (P,I,3), mu_b: (P,K,3), cov_pair: (P,I,K,3,3). Returns (P,I,K)."""
    diff = mu_a.unsqueeze(2) - mu_b.unsqueeze(1)  # (P, I, K, 3)
    inv = torch.linalg.inv(cov_pair)  # (P, I, K, 3, 3)
    logdet = torch.linalg.slogdet(cov_pair)[1]  # (P, I, K)
    quad = torch.einsum("pikd,pikde,pike->pik", diff, inv, diff)
    norm_const = (2 * np.pi) ** 1.5 * torch.exp(0.5 * logdet)
    return torch.exp(-0.5 * quad) / norm_const


def _solve_batched_mixture(
    positions, directions, values, covariances, weights, valid, query_dir_t, sigma_rbf, kappa, rel_jitter,
    noise_variance=0.0, return_mean=False, values_rgb=None,
):
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
    kxx = _padded_gram_matrix(k_pos_mat, k_dir_mat, valid, rel_jitter, noise_variance)

    values_masked = torch.where(valid, values, torch.zeros_like(values))
    rhs_cols = [values_masked, z]
    if values_rgb is not None:
        # Same additive-RHS-column trick as _solve_batched: one shared
        # Cholesky factorization, 3 more solved columns (R, G, B), not a
        # second factorization -- see that function's own values_rgb comment.
        values_rgb_masked = torch.where(valid.unsqueeze(-1), values_rgb, torch.zeros_like(values_rgb))
        rhs_cols.extend([values_rgb_masked[..., c] for c in range(values_rgb_masked.shape[-1])])
    rhs = torch.stack(rhs_cols, dim=-1)
    chol = torch.linalg.cholesky(kxx)
    solved = torch.cholesky_solve(rhs, chol)
    solved_values = solved[..., 0]
    solved_z = solved[..., 1]

    variance = z0 - (z * solved_z).sum(dim=1)
    variance_np = variance.clamp_min(0.0).cpu().numpy()
    z0_np = z0.clamp_min(0.0).cpu().numpy()

    result = (variance_np, z0_np)
    if return_mean:
        mean_bq = (z * solved_values).sum(dim=1)
        result = result + (mean_bq.cpu().numpy(),)
    if values_rgb is not None:
        solved_rgb = solved[..., 2:]
        mean_rgb = torch.einsum("pk,pkc->pc", z, solved_rgb)
        result = result + (mean_rgb.cpu().numpy(),)
    return result


def _solve_batched(
    total_mass, mean, covariance, positions, directions, values, valid, query_dir_t, sigma_rbf, kappa, rel_jitter,
    noise_variance=0.0, return_mean=False, values_rgb=None,
):
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
    kxx = _padded_gram_matrix(k_pos_mat, k_dir_mat, valid, rel_jitter, noise_variance)

    values_masked = torch.where(valid, values, torch.zeros_like(values))
    rhs_cols = [values_masked, z]
    if values_rgb is not None:
        # 3 more RHS columns (R, G, B), masked the same way as the scalar
        # `values` column -- solved by the SAME Cholesky factorization
        # below (one factorization, 2-5 right-hand sides), not a second,
        # per-channel solve. See compute_directional_variance_batched's
        # `values_rgb` docstring for why: candidate gathering and `kxx`
        # itself don't depend on color at all, only which RHS column does.
        values_rgb_masked = torch.where(valid.unsqueeze(-1), values_rgb, torch.zeros_like(values_rgb))
        rhs_cols.extend([values_rgb_masked[..., c] for c in range(values_rgb_masked.shape[-1])])
    rhs = torch.stack(rhs_cols, dim=-1)  # (P, K, 2) or (P, K, 5) with values_rgb
    chol = torch.linalg.cholesky(kxx)
    solved = torch.cholesky_solve(rhs, chol)  # (P, K, 2 or 5)
    solved_values = solved[..., 0]
    solved_z = solved[..., 1]

    variance = z0 - (z * solved_z).sum(dim=1)
    variance_np = variance.clamp_min(0.0).cpu().numpy()
    z0_np = z0.clamp_min(0.0).cpu().numpy()

    result = (variance_np, z0_np)
    if return_mean:
        # C_BQ = moment_vector @ solve(kxx, values) -- the exact same formula
        # quadrature._posterior_mean_variance uses (mean = z @ Kxx^-1 @ values),
        # here for free: Kxx^-1 @ values_masked is solved[..., 0], the OTHER
        # column of the same Cholesky solve variance already needed for
        # Kxx^-1 @ z (solved_z), so this costs one extra (P, K) reduction, not a
        # second factorization/solve.
        mean_bq = (z * solved_values).sum(dim=1)
        result = result + (mean_bq.cpu().numpy(),)
    if values_rgb is not None:
        # Same formula, one column per channel: mean_rgb[:, c] = z @ Kxx^-1 @
        # values_rgb[..., c] = z @ solved[..., 2+c].
        solved_rgb = solved[..., 2:]  # (P, K, 3)
        mean_rgb = torch.einsum("pk,pkc->pc", z, solved_rgb)
        result = result + (mean_rgb.cpu().numpy(),)
    return result


def compute_directional_alpha_risk_batched_mixture(
    engine,
    camera_index: CameraSplatIndex,
    query_points: np.ndarray,
    query_directions: np.ndarray,
    angular_tol: float,
    sigma_rbf: float,
    kappa: float,
    max_candidates: int = 100,
    rel_jitter: float = 1e-4,
    noise_variance: float = 0.0,
    device: str = "cuda",
    pixel_chunk_bytes: float = 1.0e9,
) -> np.ndarray:
    """Batched-GPU equivalent of `quadrature.
    rendering_aware_alternative_weight_risk_mixture_directional`, evaluated
    at `w := weights` (the real alpha-compositing transmittance weights
    `_gather_candidates_and_weights` already computes) -- the production,
    real-checkpoint-scale counterpart of that scalar reference (which is
    an O(K^2) Python double loop, fine for cross-validation on toy data,
    much too slow at real candidate counts).

    No Cholesky factorization/solve at all (unlike
    `compute_directional_variance_batched_mixture`, which solves for the
    BQ-optimal weights): `w` is already given, so this is just three
    quadratic-form evaluations (z0, w@z, w@Kxx@w) per pixel -- cheaper than
    the solve-based path, not just a variant of it.

    Returns `(mean, risk)`, each `(P,)` float64: `mean = weights @ values`
    (the real, local alpha-compositing color estimate), `risk = max(z0 - 2
    w@z + w@Kxx@w, 0)`. Zero-candidate queries fall back to `(0.0, z0)`
    from `_zero_candidate_variance` (matching every other batched path's
    convention here: nothing to average, and the risk of the trivial
    zero-weight estimator is exactly the prior self-variance).
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
        z0 = _zero_candidate_variance(cov, sigma_rbf, d)
        return np.zeros(p_total, dtype=np.float64), z0

    index_bearings = torch.tensor(camera_index.bearings, dtype=dtype, device=device)
    index_directions = torch.tensor(camera_index.directions, dtype=dtype, device=device)
    index_depths = torch.tensor(camera_index.depths, dtype=dtype, device=device)
    index_positions = torch.tensor(engine.positions[camera_index.indices], dtype=dtype, device=device)
    index_values = torch.tensor(engine.values[camera_index.indices], dtype=dtype, device=device)
    index_global_idx = torch.tensor(camera_index.indices, dtype=torch.int64, device=device)
    index_opacities = torch.tensor(engine.opacities[camera_index.indices], dtype=dtype, device=device)
    if engine.scales is not None and engine.rotations is not None:
        index_covariances = torch.tensor(engine.covariances()[camera_index.indices], dtype=dtype, device=device)
    else:
        index_covariances = torch.zeros(camera_index.indices.shape[0], 3, 3, dtype=dtype, device=device)

    bytes_per_pixel_row = k_cap * k_cap * 9 * 8 * 4
    chunk_size = max(1, min(p_total, int(pixel_chunk_bytes / max(bytes_per_pixel_row, 1))))

    means = np.empty(p_total, dtype=np.float64)
    risks = np.empty(p_total, dtype=np.float64)
    for start in range(0, p_total, chunk_size):
        end = min(start + chunk_size, p_total)
        positions, directions, values, covariances, weights, valid, query_dir_t, _ = _gather_candidates_and_weights(
            camera_index, query_points[start:end], query_directions[start:end], angular_tol, k_cap, device, dtype,
            index_bearings, index_directions, index_depths, index_positions, index_values,
            index_opacities, index_covariances, None, index_global_idx,
        )
        means[start:end], risks[start:end] = _alpha_risk_batched_mixture(
            positions, directions, values, covariances, weights, valid, query_dir_t, sigma_rbf, kappa, rel_jitter,
            noise_variance,
        )
    return means, risks


def _alpha_risk_batched_mixture(
    positions, directions, values, covariances, weights, valid, query_dir_t, sigma_rbf, kappa, rel_jitter,
    noise_variance=0.0,
):
    """Shared z0/z/Kxx construction identical to `_solve_batched_mixture`
    (same formulas, same variable names -- see that function's own
    comments for the derivation), but evaluates the fixed-weight risk
    `e(weights)^2 = z0 - 2 weights@z + weights@Kxx@weights` instead of
    solving `Kxx^-1 z` for the BQ-optimal weights. `mean = weights@values`
    directly -- no kernel/Kxx involved in the mean at all, matching
    `rendering_aware_alternative_weight_risk_mixture_directional`'s own
    `mean = w @ values`.
    """
    dtype = covariances.dtype
    device = covariances.device
    eye3 = torch.eye(3, dtype=dtype, device=device)
    p, k = positions.shape[0], positions.shape[1]

    cov0_pair = covariances.unsqueeze(2) + covariances.unsqueeze(1) + (sigma_rbf**2) * eye3
    gauss0 = _pairwise_gaussian_density(positions, positions, cov0_pair)
    z0 = torch.einsum("pk,pl,pkl->p", weights, weights, gauss0)

    cov_k = covariances + (sigma_rbf**2) * eye3
    cov_pair = cov_k.unsqueeze(1).expand(-1, k, -1, -1, -1)
    gauss = _pairwise_gaussian_density(positions, positions, cov_pair)
    z_raw = torch.einsum("pk,pik->pi", weights, gauss)
    dir_align = torch.einsum("pki,pi->pk", directions, query_dir_t)
    k_dir_query = torch.exp(kappa * (dir_align - 1.0))
    z = z_raw * k_dir_query
    z = torch.where(valid, z, torch.zeros_like(z))

    pos_dist2 = torch.cdist(positions, positions) ** 2
    k_pos_mat = torch.exp(-pos_dist2 / (2 * sigma_rbf**2)) / (sigma_rbf * np.sqrt(2 * np.pi)) ** 3
    dot_mat = torch.einsum("pki,pji->pkj", directions, directions)
    k_dir_mat = torch.exp(kappa * (dot_mat - 1.0))
    kxx = _padded_gram_matrix(k_pos_mat, k_dir_mat, valid, rel_jitter, noise_variance)

    values_masked = torch.where(valid, values, torch.zeros_like(values))
    mean = (weights * values_masked).sum(dim=1)
    wKw = torch.einsum("pi,pij,pj->p", weights, kxx, weights)
    wz = (weights * z).sum(dim=1)
    risk = z0 - 2.0 * wz + wKw
    return mean.cpu().numpy(), risk.clamp_min(0.0).cpu().numpy()
