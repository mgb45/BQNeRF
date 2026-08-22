"""Spherical-harmonic view-dependent color, matching the convention real
3D Gaussian Splatting checkpoints use (the widely-reused `eval_sh` utility
originally from Plenoxels/instant-ngp, carried into the 3DGS/gsplat
codebases essentially verbatim). Needed because a real splat's color is
`eval_sh(sh_coeffs, direction)`, not a single flat RGB value -- the mock
scene's flat `colors` field was a scoping simplification, not a model of
what real checkpoints store.

Degree 0 (1 coeff/channel) is view-independent; degrees 1-3 add
view-dependent terms (4, 9, 16 coeffs/channel respectively). Real 3DGS
typically trains up to degree 3.
"""

from __future__ import annotations

import numpy as np

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = np.array(
    [
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396,
    ]
)
SH_C3 = np.array(
    [
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435,
    ]
)

N_COEFFS_FOR_DEGREE = {0: 1, 1: 4, 2: 9, 3: 16}


def eval_sh(sh_coeffs: np.ndarray, directions: np.ndarray, degree: int) -> np.ndarray:
    """sh_coeffs: (..., n_channels, n_coeffs) with n_coeffs >= the number
    required for `degree` (extra coefficients, if any, are ignored -- lets
    a caller pass full degree-3 coefficients but evaluate at a lower
    degree). directions: (..., 3) unit vectors, broadcastable against
    sh_coeffs' leading shape. Returns (..., n_channels).

    Matches the reference `eval_sh` exactly, including the "+ 0.5" DC
    offset convention (3DGS stores SH coefficients as offsets from a
    mid-gray baseline, not as the raw color).
    """
    if degree not in N_COEFFS_FOR_DEGREE:
        raise ValueError(f"degree must be 0-3, got {degree}")
    n_needed = N_COEFFS_FOR_DEGREE[degree]
    if sh_coeffs.shape[-1] < n_needed:
        raise ValueError(f"degree {degree} needs {n_needed} coefficients, got {sh_coeffs.shape[-1]}")

    directions = np.asarray(directions, dtype=float)
    sh = np.asarray(sh_coeffs, dtype=float)

    result = SH_C0 * sh[..., 0]

    if degree > 0:
        # trailing None axis so x/y/z broadcast against sh's channel axis
        # (sh[..., k] keeps a channel dim that directions[..., 0] doesn't
        # have) -- this also lets a single set of directions broadcast
        # against many splats' coefficients, or vice versa, not just the
        # batch-aligned case where each direction already has its own
        # matching splat.
        x, y, z = directions[..., 0, None], directions[..., 1, None], directions[..., 2, None]
        result = result - SH_C1 * y * sh[..., 1] + SH_C1 * z * sh[..., 2] - SH_C1 * x * sh[..., 3]

        if degree > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + SH_C2[0] * xy * sh[..., 4]
                + SH_C2[1] * yz * sh[..., 5]
                + SH_C2[2] * (2.0 * zz - xx - yy) * sh[..., 6]
                + SH_C2[3] * xz * sh[..., 7]
                + SH_C2[4] * (xx - yy) * sh[..., 8]
            )

            if degree > 2:
                result = (
                    result
                    + SH_C3[0] * y * (3 * xx - yy) * sh[..., 9]
                    + SH_C3[1] * xy * z * sh[..., 10]
                    + SH_C3[2] * y * (4 * zz - xx - yy) * sh[..., 11]
                    + SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12]
                    + SH_C3[4] * x * (4 * zz - xx - yy) * sh[..., 13]
                    + SH_C3[5] * z * (xx - yy) * sh[..., 14]
                    + SH_C3[6] * x * (xx - 3 * yy) * sh[..., 15]
                )

    return result + 0.5


def random_sh_coeffs(rng: np.random.Generator, n_splats: int, n_channels: int = 3, degree: int = 3, scale: float = 0.3) -> np.ndarray:
    """Synthetic SH coefficients for testing -- degree-0 term biased
    positive (a plausible-looking base color), higher-degree terms
    zero-mean (view-dependent variation around that base)."""
    n_coeffs = N_COEFFS_FOR_DEGREE[degree]
    coeffs = rng.normal(scale=scale, size=(n_splats, n_channels, n_coeffs))
    coeffs[..., 0] = rng.uniform(0.0, 1.0, size=(n_splats, n_channels))  # plausible base color
    return coeffs
