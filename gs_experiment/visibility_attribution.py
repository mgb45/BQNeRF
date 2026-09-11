"""Frustum + occlusion visibility attribution: which cameras plausibly
observed each splat, for real data where (unlike the mock scene, which
assigns this by fiat for controlled experiments) there's no ground-truth
record of which training views constrained which splat.

Originally a cheap proxy for the real work this docstring used to say was
"deferred until there's an actual renderer to hook into": rendering every
training view and recording each splat's alpha-weighted contribution.
That renderer hook now exists (gs_experiment/gsplat_rendering_weights.py,
via gsplat's own differentiable projection) for a single query ray/pixel;
`attribute_observations` below still uses the cheaper geometric filters,
since attributing *every* splat to *every* camera via the real renderer
for a whole scene is a much larger cost than this module's original
per-query use case needs:

  1. Frustum test: is the splat within the camera's field of view and in
     front of it.
  2. Occlusion test: a soft z-buffer -- project all splats into each
     camera's local angular coordinates (bearing, not full pixel
     projection) and flag a splat as occluded if another splat sits at
     a similar bearing but meaningfully closer to the camera.
  3. ray_transmittance_weights: a continuous, depth-ordered analogue of
     (2) for one specific ray -- real per-splat opacity turned into a
     genuine alpha-compositing transmittance weight, still via bearing
     proximity rather than the real anisotropic 2D footprint (contrast
     with gsplat_rendering_weights.gsplat_alpha_compositing_weights).

All of the above are pure numpy/scipy, no torch/gsplat dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from gs_experiment.camera import CameraPose, camera_local_frame

# occlusion_mask's grid search radius, in cells of size angular_tol/_CELL_SUBDIVISION
# on each axis -- see its docstring for why a finer grid (higher value here)
# reduces false positives at a constant-factor time cost, never a false negative.
# Diminishing returns past ~8: the grid's per-cell-aggregate design can only
# ever approach the axis-aligned bounding *square* of the true circular
# radius (never the circle itself, whatever S is -- see the docstring), so
# this is chosen empirically as a point past which raising it further buys
# little (measured on a real occluder scene: false-positive count 13/100 at
# S=3, 9/100 at S=10, 8/100 at S=60 -- most of the gain is already captured).
_CELL_SUBDIVISION = 8


def project_to_camera_local(positions: np.ndarray, camera: CameraPose):
    """Project `positions` (N, 3) into camera-local angular bearing
    (bearing_x, bearing_y) and depth along the camera's forward axis.
    Bearing here is the tangent-plane projection (position component along
    right/up, divided by forward-depth) -- a pinhole-like angular
    coordinate, not a full intrinsics-based pixel projection (no focal
    length/principal point needed for a bearing-based occlusion test).
    Returns (bearing_x, bearing_y, depth), each shape (N,).
    """
    positions = np.asarray(positions, dtype=float)
    right, up, forward = camera_local_frame(camera)
    rel = positions - camera.center[None, :]
    depth = rel @ forward
    safe_depth = np.where(np.abs(depth) < 1e-9, np.nan, depth)
    bearing_x = (rel @ right) / safe_depth
    bearing_y = (rel @ up) / safe_depth
    return bearing_x, bearing_y, depth


def in_frustum(positions: np.ndarray, camera: CameraPose, fov_deg: float = 60.0, near: float = 1e-3, far: float = np.inf) -> np.ndarray:
    """Boolean mask: is each position in front of `camera`, within `near`/
    `far`, and within a square field of view of half-angle `fov_deg`/2 in
    both bearing axes."""
    bearing_x, bearing_y, depth = project_to_camera_local(positions, camera)
    half_tan = np.tan(np.deg2rad(fov_deg) / 2.0)
    in_front = (depth > near) & (depth < far)
    within_fov = (np.abs(bearing_x) < half_tan) & (np.abs(bearing_y) < half_tan)
    return in_front & within_fov & ~np.isnan(bearing_x)


def occlusion_mask(
    positions: np.ndarray, camera: CameraPose, angular_tol: float, depth_margin: float = 0.05,
    occluder_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Boolean mask: is each position occluded by another position that
    projects to a similar bearing (within `angular_tol`, in the tangent-
    plane bearing units from project_to_camera_local) but is closer to the
    camera by more than `depth_margin` * that closer point's own depth
    (a relative, scale-aware margin rather than an absolute one, since
    "close" means different absolute distances near vs. far from the
    camera).

    `occluder_mask`, if given (boolean, parallel to `positions`): only
    points flagged `True` can ever occlude something -- every point's own
    occlusion status is still computed (this isn't a pre-filter on which
    points get an answer), only the "is something closer in front of me"
    z-buffer is built from occluder-eligible points alone. Exists because
    this test has no opacity awareness otherwise: a splat at opacity 0.001
    blocks exactly as completely as one at opacity 1.0, which is wrong --
    a near-transparent splat shouldn't hard-occlude anything, real
    geometry or not. This matters most for GS training's floaters (stray,
    near-zero-opacity splats that drift outside the intended training
    volume -- a normal optimization artifact): confirmed directly on a
    real 300k-splat checkpoint that ~4% of splats were such floaters, and
    that they alone were responsible for an 8x collapse in how many real
    training cameras got attributed to genuine splats (mean observations
    per splat 6.0 vs. 50.2 on an otherwise-identical, floater-free
    checkpoint of the same scene) -- pass `opacities >= min_opacity` here
    (see `attribute_observations`) to fix it at the source rather than
    filtering its downstream symptoms.

    Implemented as a real spatial-hash z-buffer -- O(n log n) time, O(n)
    memory, no pairwise enumeration at all. This replaced two earlier
    attempts, in order, each a real bottleneck found by profiling/running
    against actual checkpoints rather than assumed away:

    1. The original per-point `query_ball_point` + nested Python `for`
       loop spent 93% of total checkpoint-attribution time in this one
       function (confirmed via profiling on a real 35k-splat checkpoint:
       100s of 107s, 113 million `abs()` calls), and made the same
       attribution step time out entirely (>120s) on a real 300k-splat
       checkpoint.
    2. A `cKDTree.query_pairs`-based rewrite fixed that (93s -> 15s on the
       35k-splat checkpoint) by replacing the Python-level inner loop
       with vectorized numpy over *all* nearby pairs at once -- but on a
       denser real 300k-splat checkpoint viewed close-up, bearing density
       is high enough (confirmed: ~2,600 average bearing-neighbors per
       splat at a real `angular_tol`) that materializing every pair
       explicitly is itself the bottleneck: 216 million pairs from a
       single camera, enough to exhaust available memory and crash the
       host machine when scaled to a realistic number of cameras.

    This version never enumerates pairs or per-point neighbor lists at
    all: bearings are binned into a grid with cell size `angular_tol`,
    each cell's minimun depth is computed once via a single sort + one
    `np.minimum.reduceat` (O(n)), and each point looks up the minimum
    depth across its own cell and the 8 adjacent cells (9 O(1) hash
    lookups per point, via `np.searchsorted` into the sorted, unique cell
    keys -- still O(n log n) total, not O(n * neighbors)). A point at
    true Euclidean bearing-distance <= `angular_tol` from another is
    *always* within 1 grid cell of it in each axis (a short proof: for
    `c = angular_tol`, `x, x'` with `|x - x'| <= c`, `floor(x'/c) - floor(x/c)`
    is always 0 or 1 when `x' >= x`), so the 3x3 block never *misses* a
    true neighbor -- but it does search a 3x3 square of side `2*angular_tol`
    against a true circular radius of `angular_tol`, a ~2.86x (9/pi) area
    ratio, so it can also flag some extra points just past the true
    circular radius, near a cell corner, as "occluded" when they aren't.

    That excess-area ratio shrinks, at the cost of more (still O(1) per
    point) cell lookups, by using a finer grid than `angular_tol` itself:
    for cell size `angular_tol / S`, the same floor-division argument
    generalizes to "at most `S` cells apart in each axis" (checked
    numerically for S up to 4, not just S=1), so an `S`-cell-radius block
    (`(2S+1)^2` lookups instead of 9) is still a *superset* of the true
    circular neighborhood -- still zero missed occlusions -- while the
    search area shrinks from `9*angular_tol^2` towards the circle's own
    bounding-square area `4*angular_tol^2` as `S` grows (the best a
    square-cell grid can do). `_CELL_SUBDIVISION` below is that `S`;
    raising it trades a constant-factor slowdown (still O(n log n)
    overall, not O(n * density)) for fewer false positives. Measured
    directly against a brute-force circular-radius reference (see
    `_occlusion_mask_reference` in the test suite) with the current
    setting: zero missed occlusions in every case tried, and a
    false-positive rate on the order of several percent to ~20% in the
    densest configurations tried -- real, and stated here plainly rather
    than downplayed, but always conservative in the same direction (never
    a missed real occlusion, only ever an extra one), which is the trade
    that matters for a function this module's own docstring already
    calls a cheap proxy, not a faithful reproduction.
    """
    bearing_x, bearing_y, depth = project_to_camera_local(positions, camera)
    n = positions.shape[0]
    occluded = np.zeros(n, dtype=bool)

    valid = ~np.isnan(bearing_x)
    if valid.sum() < 1:
        return occluded

    if occluder_mask is None:
        occ_valid = valid
    else:
        occ_valid = valid & np.asarray(occluder_mask, dtype=bool)
    if occ_valid.sum() < 1:
        # nothing is eligible to occlude anything -- every valid point is unoccluded.
        return occluded

    subdivision = _CELL_SUBDIVISION
    cell = max(float(angular_tol) / subdivision, 1e-12)
    offset = np.int64(1 << 24)  # generous vs. any plausible bearing/cell-size range

    # --- build the z-buffer grid from occluder-eligible points only. When
    # occluder_mask is None (occ_valid == valid), this is exactly the
    # original single-population behavior -- a point's own cell always
    # includes its own depth, so self-comparison still never triggers a
    # false "occluded" flag for it (see the comment at the query step
    # below, which only needs occ_valid's superset property, not this
    # code path specifically). ---
    occ_idx = np.where(occ_valid)[0]
    obx, oby, od = bearing_x[occ_idx], bearing_y[occ_idx], depth[occ_idx]
    ocx = np.floor(obx / cell).astype(np.int64)
    ocy = np.floor(oby / cell).astype(np.int64)
    occ_key = ocx * (offset * 2) + (ocy + offset)

    order = np.argsort(occ_key, kind="stable")
    sorted_keys = occ_key[order]
    sorted_depth = od[order]
    unique_keys, start_idx = np.unique(sorted_keys, return_index=True)
    cell_min_depth = np.minimum.reduceat(sorted_depth, start_idx)

    # --- query every valid point (not just occluder-eligible ones) against
    # that grid: a non-occluding (e.g. low-opacity) point can still itself
    # be flagged occluded by something real in front of it. ---
    valid_idx = np.where(valid)[0]
    bx, by, d = bearing_x[valid_idx], bearing_y[valid_idx], depth[valid_idx]
    m = bx.shape[0]
    cx = np.floor(bx / cell).astype(np.int64)
    cy = np.floor(by / cell).astype(np.int64)

    best_neighbor_depth = np.full(m, np.inf)
    for dx in range(-subdivision, subdivision + 1):
        for dy in range(-subdivision, subdivision + 1):
            shifted_key = (cx + dx) * (offset * 2) + ((cy + dy) + offset)
            pos = np.searchsorted(unique_keys, shifted_key)
            pos = np.clip(pos, 0, unique_keys.shape[0] - 1)
            found = unique_keys[pos] == shifted_key
            depths_here = np.where(found, cell_min_depth[pos], np.inf)
            best_neighbor_depth = np.minimum(best_neighbor_depth, depths_here)

    # A point that is itself occluder-eligible always finds its own depth
    # via its own cell above (it's part of the grid), so self-comparison
    # never triggers a false "occluded" flag for it: d[i] < d[i] -
    # margin*|d[i]| is false for any positive margin. A non-occluder point
    # has no such self-entry, which is correct -- its own (untrusted, e.g.
    # near-transparent) depth was never meant to protect it from a real
    # occluder in front of it.
    occluded_local = best_neighbor_depth < d - depth_margin * np.abs(d)
    occluded[valid_idx] = occluded_local

    return occluded


def ray_transmittance_weights(
    positions: np.ndarray, opacities: np.ndarray, camera: CameraPose, reference_bearing, angular_tol: float = 0.05
) -> np.ndarray:
    """Real, depth-ordered alpha-compositing transmittance weight
    `w_i = T_i * alpha_i` for each of `positions`, along the single ray at
    `reference_bearing` (a (bearing_x, bearing_y) pair, in `camera`'s local
    angular coordinates -- see project_to_camera_local) -- the continuous,
    occlusion-aware analogue of `occlusion_mask`'s binary yes/no flag.
    `alpha_i` is real per-splat opacity used directly as the discrete alpha
    in the standard alpha-compositing formula
    (`PROOF_alpha_compositing_equivalence.md` Theorem A: `T_i = prod_{j
    closer, same ray} (1 - alpha_j)`), not a synthetic Gaussian bump --
    this is what `gs_experiment/pixel_uncertainty.py`'s
    `rendering_aware_variance` (opacity-averaged, occlusion-blind) is
    missing, and what its own docstring names as the next step.

    Splats more than `angular_tol` from `reference_bearing`, behind the
    camera, or with undefined bearing (see project_to_camera_local) get
    weight 0 regardless of depth or opacity -- they aren't plausibly on
    this ray at all. Splats behind a closer, sufficiently opaque splat on
    the same ray get a small weight via the accumulated transmittance
    product, exactly as real alpha compositing would down-weight them --
    not a hard cutoff the way occlusion_mask's boolean flag is.
    """
    positions = np.asarray(positions, dtype=float)
    opacities = np.asarray(opacities, dtype=float)
    n = positions.shape[0]
    weights = np.zeros(n)
    if n == 0:
        return weights

    bearing_x, bearing_y, depth = project_to_camera_local(positions, camera)
    ref_x, ref_y = reference_bearing
    angular_dist = np.sqrt((bearing_x - ref_x) ** 2 + (bearing_y - ref_y) ** 2)
    on_ray = (angular_dist < angular_tol) & ~np.isnan(bearing_x) & (depth > 0)
    on_ray_idx = np.where(on_ray)[0]
    if on_ray_idx.size == 0:
        return weights

    # kind='stable' so tied depths (routine here -- see this project's caller,
    # pixel_uncertainty.LocalUncertaintyEngine._along_ray_local_data, which now
    # canonicalizes `idx` to ascending global-index order before calling this
    # function specifically so a stable sort resolves ties the same way every
    # time, matching gpu_uncertainty.py's batched equivalent's own canonicalize-
    # then-stable-sort) preserve that canonical order instead of an
    # implementation-defined (and here, non-matching-across-implementations)
    # arbitrary order.
    depth_order = on_ray_idx[np.argsort(depth[on_ray_idx], kind="stable")]
    transmittance = 1.0
    for i in depth_order:
        alpha = float(np.clip(opacities[i], 0.0, 1.0))
        weights[i] = transmittance * alpha
        transmittance *= 1.0 - alpha
    return weights


@dataclass
class CameraSplatIndex:
    """A real per-camera bearing/depth index over an entire scene's
    splats, built once and reused for many per-pixel
    `LocalUncertaintyEngine.rendering_aware_variance_along_ray` queries
    against the same camera.

    Replaces gathering candidates via a 3D-world-space ball query
    (`LocalUncertaintyEngine.local_neighbors`) before handing them to
    `ray_transmittance_weights`: on a real, densely-packed checkpoint, a
    generously-sized 3D radius can return a large fraction of the *entire*
    scene (confirmed directly: ~half of 300k splats within a 1.6-unit
    radius on a real NeRF-Synthetic checkpoint), and `local_neighbors`
    then keeps only `max_neighbors` of those *uniformly at random* --
    almost never the splat that's actually relevant to a given pixel,
    since relevance to one ray is about angular/depth alignment, not raw
    3D Euclidean distance. Indexing by real bearing instead (this class)
    and ranking any overflow by bearing distance (not randomly) fixes
    both the correctness gap and the near-total candidate waste.

    `directions`, if built with it, enables a second overflow-ranking mode
    in `query` (`query_direction=...`): pure bearing-distance ranking is
    the right choice for a position-only query, but for a *directional*
    query it under-serves the actual k_pos*k_dir kernel being computed --
    `positions` here is typically the camera-*expanded* observation array
    (one row per (splat, observing-camera) pair), so a single physical
    splat contributes many rows that all share the exact same bearing
    (bearing depends only on position, never on which camera observed it).
    Those rows are then exact ties under bearing-distance ranking, and
    which of them survive an overflowing `max_candidates` cut becomes an
    arbitrary artifact of `query_ball_point`'s internal ordering -- in
    particular, *not* preferentially the rows whose direction actually
    matches the query direction, even though that is exactly what the
    kernel's own k_dir factor most rewards. Confirmed directly (real
    lego-gap-checkpoint checkpoints, 5 conditions): the resulting
    directional-variance-vs-gap-width trend lost its monotonicity (from a
    clean rho=1.000 to rho=0.600) purely from this tie-breaking artifact
    -- reproduced identically across a 10x range of angular_tol, ruling
    out that parameter as the cause. Passing `query_direction` breaks
    overflow ties by directional alignment (`direction . query_direction`,
    descending) among candidates already inside the bearing gate, so the
    scarce max_candidates budget preserves the rows the kernel actually
    weights most, instead of an arbitrary bearing-tie subset.
    """

    indices: np.ndarray  # into the original positions/opacities arrays -- in front of the camera only
    bearings: np.ndarray  # (M, 2)
    depths: np.ndarray  # (M,)
    camera: CameraPose
    directions: Optional[np.ndarray] = None  # (M, 3), parallel to indices/bearings/depths -- optional
    _tree: Optional[cKDTree] = field(default=None, repr=False)

    def __post_init__(self):
        self._tree = cKDTree(self.bearings) if len(self.indices) > 0 else None

    @classmethod
    def build(cls, positions: np.ndarray, camera: CameraPose, directions: Optional[np.ndarray] = None) -> "CameraSplatIndex":
        bearing_x, bearing_y, depth = project_to_camera_local(positions, camera)
        valid = (depth > 0) & ~np.isnan(bearing_x)
        idx = np.where(valid)[0]
        return cls(
            indices=idx, bearings=np.stack([bearing_x[idx], bearing_y[idx]], axis=1), depths=depth[idx], camera=camera,
            directions=directions[idx] if directions is not None else None,
        )

    def query(
        self, reference_bearing, angular_tol: float, max_candidates: Optional[int] = None, query_direction=None,
    ) -> np.ndarray:
        """Indices (into the original positions/opacities arrays) of
        splats within `angular_tol` of `reference_bearing`. If more than
        `max_candidates` qualify: ranked by bearing distance (nearest
        first) by default -- a real relevance ranking, not a random,
        potentially-irrelevant subsample -- or, if `query_direction` is
        given (and this index was built with `directions`), ranked by
        directional alignment with `query_direction` instead (see this
        class's docstring for why the directional case needs a different
        tie-break than pure bearing distance)."""
        if self._tree is None:
            return np.empty(0, dtype=int)
        local = np.array(self._tree.query_ball_point(reference_bearing, angular_tol), dtype=int)
        if local.size == 0:
            return np.empty(0, dtype=int)
        if max_candidates is not None and local.size > max_candidates:
            if query_direction is not None and self.directions is not None:
                alignment = self.directions[local] @ np.asarray(query_direction)
                local = local[np.argsort(-alignment)[:max_candidates]]
            else:
                dists = np.linalg.norm(self.bearings[local] - np.asarray(reference_bearing), axis=1)
                local = local[np.argsort(dists)[:max_candidates]]
        return self.indices[local]


def attribute_observations(
    positions: np.ndarray, cameras: list, fov_deg: float = 60.0, angular_tol: float = 0.05, depth_margin: float = 0.05,
    opacities: Optional[np.ndarray] = None, min_opacity: float = 0.0,
):
    """For each camera, which splat indices does it plausibly observe
    (in frustum and not occluded). Returns a list of length len(cameras),
    each entry an array of splat indices -- the camera-indexed view of the
    same information SplatScene.observed_camera_idx stores splat-indexed;
    invert this (see gs_experiment.splat_scene) to populate that field for
    real data.

    `opacities`/`min_opacity`: passed straight through to `occlusion_mask`'s
    `occluder_mask` (`opacities >= min_opacity`) -- splats below the
    threshold still get their own occlusion status computed and can still
    be attributed to a camera if nothing real occludes them, they just
    can't hard-occlude anything else themselves. `min_opacity=0.0` (the
    default) keeps every splat occluder-eligible, i.e. unchanged from
    before this parameter existed. See `occlusion_mask`'s docstring for
    why this matters (GS training floaters).
    """
    occluder_mask_full = None if opacities is None else (np.asarray(opacities, dtype=float) >= min_opacity)
    per_camera = []
    for camera in cameras:
        visible = in_frustum(positions, camera, fov_deg=fov_deg)
        if visible.any():
            occluded = np.zeros(positions.shape[0], dtype=bool)
            local_occluder_mask = None if occluder_mask_full is None else occluder_mask_full[visible]
            occluded[visible] = occlusion_mask(positions[visible], camera, angular_tol, depth_margin, occluder_mask=local_occluder_mask)
            visible_idx = np.where(visible)[0]
            per_camera.append(visible_idx[~occluded[visible_idx]])
        else:
            per_camera.append(np.array([], dtype=int))
    return per_camera


def invert_to_observed_camera_idx(per_camera_visible: list, n_splats: int) -> list:
    """Invert attribute_observations's camera-indexed output into the
    splat-indexed observed_camera_idx list SplatScene expects."""
    observed = [[] for _ in range(n_splats)]
    for cam_idx, splat_indices in enumerate(per_camera_visible):
        for s in splat_indices:
            observed[s].append(cam_idx)
    return [np.array(cams, dtype=int) for cams in observed]


def subsample_observed_camera_idx(observed_camera_idx: list, max_per_splat: int, seed: int = 0) -> list:
    """Cap each splat's observation count at `max_per_splat` (uniform
    random subsample without replacement, seeded for reproducibility) --
    splats already at or under the cap are returned unchanged.

    Exists because `splat_scene.splat_observations` expands
    `observed_camera_idx` into one row per (splat, observing-camera) pair,
    and that row count is genuinely what the downstream directional
    `LocalUncertaintyEngine` needs fully resident for its neighbor index
    (confirmed on a real 3M-splat/100-view checkpoint: ~167M rows uncapped,
    ~59M rows even at cap=20, measured at ~14GB resident for that one call
    -- OOM-killed the host repeatedly at that budget; see
    `splat_scene.py`'s `PER_CALL_MEMORY_BUDGET_BYTES` comment). Unlike
    `gpu_visibility_attribution`'s per-camera *construction* chunking,
    capping observation count per splat actually bounds the resulting
    steady-state row count (<= `n_splats * max_per_splat`, independent of
    real camera-coverage density) rather than just smoothing a transient
    peak -- the same idea `max_neighbors`/`max_candidates` already apply
    at BQ query time, just applied earlier, at data-construction time,
    where it's what actually determines resident memory. This is a real
    (if usually small) approximation, not free: a splat observed by more
    cameras than the cap has some of its real observations dropped rather
    than all of them contributing to the directional kernel.
    """
    if max_per_splat is None:
        return observed_camera_idx
    rng = np.random.default_rng(seed)
    result = []
    for cams in observed_camera_idx:
        if len(cams) > max_per_splat:
            cams = rng.choice(cams, size=max_per_splat, replace=False)
        result.append(cams)
    return result
