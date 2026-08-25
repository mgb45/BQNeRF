"""Controlled test for gs_experiment/results/FINDINGS.md section 12's open
question: is the demonstrated go/no-go result (position-only BQ variance
ranking the wide zone as more uncertain than the narrow zone) driven by
view-count itself, or by the splat-clustering/redundancy confound flagged
there as not yet isolated (more views -> more consistent gradient -> more
densification cycles -> more spatially redundant splats, since
clone/split placement puts children near their parent)?

Post-hoc on the already-trained checkpoint, no retraining needed. Three
conditions, same query points/kernel/window throughout, only the wide
zone's splat population changes:

  1. original wide zone (baseline, reproduces FINDINGS.md section 11)
  2. wide zone randomly subsampled to the narrow zone's splat COUNT
     (holds count fixed, doesn't touch spacing/redundancy pattern)
  3. wide zone greedily declustered (Poisson-disk-style minimum-distance
     rejection) to match the narrow zone's median nearest-neighbor
     SPACING (changes both count and redundancy pattern together, since
     they're coupled by construction here)

If (2) still shows the wide-higher-variance effect but (3) removes or
reverses it, that's evidence the effect is about spatial redundancy
specifically, not raw count -- supporting FINDINGS.md section 12's leading
hypothesis. If both (2) and (3) still show the effect, count/redundancy
isn't the driver and something else (e.g. genuine coverage-extent
differences) needs to be considered instead.

Run: .venv-gsplat/bin/python gs_experiment/validate_declustering_isolation.py <scene_dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from gs_experiment.pixel_uncertainty import LocalUncertaintyEngine, make_default_3d_position_kernel
from gs_experiment.splat_scene import load_from_gsplat_checkpoint


def greedy_decluster(positions: np.ndarray, min_dist: float, order: np.ndarray) -> np.ndarray:
    """Greedily keep points at least `min_dist` apart, processing in
    `order`. Poisson-disk-style rejection subsampling: not the only way
    to decluster, but simple and doesn't need a target count picked in
    advance -- `min_dist` directly controls the resulting spacing, which
    is the statistic being matched here. O(n * n_kept), fine at this
    scale (thousands of candidates, expected n_kept in the low
    thousands)."""
    n = positions.shape[0]
    keep = np.zeros(n, dtype=bool)
    kept = np.empty((n, 3))
    n_kept = 0
    for i in order:
        p = positions[i]
        if n_kept > 0:
            d = np.linalg.norm(kept[:n_kept] - p, axis=1).min()
            if d < min_dist:
                continue
        keep[i] = True
        kept[n_kept] = p
        n_kept += 1
    return keep


def median_nn_distance(positions: np.ndarray) -> float:
    from scipy.spatial import cKDTree

    tree = cKDTree(positions)
    dists, _ = tree.query(positions, k=2)
    return float(np.median(dists[:, 1]))


def run(scene_dir: str, zone_radius: float = 1.6, sigma: float = 0.9, window_radius: float = 1.6, seed: int = 0):
    scene = load_from_gsplat_checkpoint(scene_dir, attribution_angular_tol=0.01)
    positions, colors = scene.positions, scene.colors

    wide_center = np.array([0.0, 0.0, 0.0])
    narrow_center = np.array([18.0, 0.0, 0.0])
    d_wide = np.linalg.norm(positions - wide_center, axis=1)
    d_narrow = np.linalg.norm(positions - narrow_center, axis=1)
    wide_mask = d_wide < zone_radius
    narrow_mask = d_narrow < zone_radius
    other_mask = ~wide_mask & ~narrow_mask

    wide_pos, wide_col = positions[wide_mask], colors[wide_mask]
    narrow_pos, narrow_col = positions[narrow_mask], colors[narrow_mask]
    other_pos, other_col = positions[other_mask], colors[other_mask]

    narrow_spacing = median_nn_distance(narrow_pos)
    wide_spacing = median_nn_distance(wide_pos)
    print(f"baseline: wide n={len(wide_pos)} median_NN={wide_spacing:.5f}  narrow n={len(narrow_pos)} median_NN={narrow_spacing:.5f}")

    rng = np.random.default_rng(seed)
    bounds = tuple((positions[:, d].min() - 1.0, positions[:, d].max() + 1.0) for d in range(3))
    pos_kernel = make_default_3d_position_kernel(sigma=sigma)

    def zone_ratio(wide_pos_variant, wide_col_variant, label):
        all_pos = np.concatenate([wide_pos_variant, narrow_pos, other_pos], axis=0)
        all_col = np.concatenate([wide_col_variant, narrow_col, other_col], axis=0)
        engine = LocalUncertaintyEngine(positions=all_pos, values=all_col, pos_kernel=pos_kernel, scene_bounds=bounds, seed=seed)
        wide_var = engine.spatial_only_variance(wide_center, window_radius).variance
        narrow_var = engine.spatial_only_variance(narrow_center, window_radius).variance
        ratio = narrow_var / wide_var
        print(f"{label}: n_wide={len(wide_pos_variant)}  wide_var={wide_var:.4f}  narrow_var={narrow_var:.4f}  ratio={ratio:.3f}x")
        return ratio

    print()
    r1 = zone_ratio(wide_pos, wide_col, "(1) original wide")

    # (2) random subsample of wide to match narrow's count -- holds count
    # fixed, leaves spacing/redundancy pattern (relative to the original
    # population's own structure) otherwise unperturbed
    n_target = len(narrow_pos)
    sub_idx = rng.choice(len(wide_pos), size=min(n_target, len(wide_pos)), replace=False)
    r2 = zone_ratio(wide_pos[sub_idx], wide_col[sub_idx], "(2) wide random-subsampled to narrow's count")

    # (3) greedy decluster wide to match narrow's median spacing
    order = rng.permutation(len(wide_pos))
    keep = greedy_decluster(wide_pos, min_dist=narrow_spacing, order=order)
    declustered_spacing = median_nn_distance(wide_pos[keep]) if keep.sum() > 1 else float("nan")
    r3 = zone_ratio(
        wide_pos[keep], wide_col[keep],
        f"(3) wide declustered to spacing~{declustered_spacing:.5f} (target {narrow_spacing:.5f})",
    )

    print()
    print(f"Summary: ratio(narrow/wide) original={r1:.3f}x  count-matched={r2:.3f}x  spacing-matched={r3:.3f}x")
    print("If (3) moves toward 1x while (2) doesn't: redundancy/spacing is the driver, not raw count.")
    print("If both (2) and (3) stay well below 1x: count/redundancy isn't the (whole) story.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene_dir")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.scene_dir, seed=args.seed)


if __name__ == "__main__":
    main()
