"""Per-splat SH-coefficient posterior covariance -- the directional term of
the renderer-consistent sparse-GP decomposition of 3D Gaussian Splatting
(see gs_experiment/results/FINDINGS.md section 8/9 and ROADMAP.md): treat
each splat's stored SH coefficients as an interdomain inducing variable of
an underlying radiance-field GP, chosen so the GP's posterior mean under
the real alpha-compositing weights is EXACTLY `C_alpha(q)` (the renderer's
own output, untouched) -- so the entire predictive variance is a genuinely
post-hoc quantity, never coupled to the mean at all:

    mu_q = C_alpha(q)
    u_q  = u_spatial_BQ(q) + sum_i beta_{q,i}^2 * s_i^2(d_q)

`u_spatial_BQ(q)` (the finite-spatial-representation term) is already
implemented: `pixel_uncertainty.LocalUncertaintyEngine.
rendering_aware_alpha_risk_along_ray(...).alpha_risk` (scalar) /
`gpu_uncertainty.compute_alpha_risk_batched` (batched) -- the real alpha
weights' own RKHS risk under a position-only kernel, no direction involved.

This module is the second term: `s_i^2(d_q) = phi(d_q)^T Sigma_theta_i
phi(d_q)`, splat i's own directional-appearance uncertainty at query
direction d_q, where `phi` is the SAME real SH basis 3DGS's renderer
already evaluates against (`sh_basis` below -- not a new color model, just
that basis pulled out as a design matrix) and `Sigma_theta_i` is a
per-splat Bayesian linear regression posterior covariance:

    Sigma_theta_i^-1 = lam * I + sum_{p in D_i} beta_{p,i}^2 * phi(d_p) phi(d_p)^T

`D_i`: splat i's own real observed training directions (one per observing
camera in `scene.observed_camera_idx[i]`). `beta_{p,i} = T_{p,i}
* alpha_{p,i}`: splat i's own REAL alpha-compositing weight in training
observation p -- i.e. how much splat i actually contributed to whatever
pixel its own projected center landed on, in that specific training
camera (`gpu_sh_directional_uncertainty.accumulate_sh_precision`
computes this by rerendering the fixed training cameras; no retraining
needed). `lam`: a scalar prior precision (regularization) -- a free
hyperparameter, not fit here (see that module's own docstring for why a
single scalar is the right starting choice, not a new hyperparameter
search).

Crucially, Sigma_theta_i does NOT depend on the observed *colors* at all,
only on which directions were observed and how much each observation's
own alpha-compositing weight was -- a splat observed with strong, angularly-
diverse real contributions has small Sigma_theta_i everywhere; one barely
contributing to any pixel, or only ever from a narrow cone of directions,
stays uncertain away from that cone. This never touches the mean, so it
cannot corrupt it the way a joint position+direction kernel solve would.
"""

from __future__ import annotations

import numpy as np

from gs_experiment.spherical_harmonics import N_COEFFS_FOR_DEGREE, SH_C0, SH_C1, SH_C2, SH_C3


def sh_basis(directions: np.ndarray, degree: int) -> np.ndarray:
    """phi(d): the real SH basis `spherical_harmonics.eval_sh` implicitly
    evaluates sh_coeffs against, i.e. `eval_sh(coeffs, d, degree)[...,c] ==
    coeffs[...,c,:] @ sh_basis(d, degree) + 0.5` for every channel c (cross-
    validated exactly in tests/gs_experiment/test_sh_directional_uncertainty.py).

    directions: (..., 3) unit vectors. Returns (..., n_coeffs) with
    n_coeffs = N_COEFFS_FOR_DEGREE[degree].
    """
    if degree not in N_COEFFS_FOR_DEGREE:
        raise ValueError(f"degree must be 0-3, got {degree}")
    directions = np.asarray(directions, dtype=float)
    x, y, z = directions[..., 0], directions[..., 1], directions[..., 2]

    cols = [np.full_like(x, SH_C0)]
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
    return np.stack(cols, axis=-1)


def per_splat_coefficient_precision(
    design_matrices: list[np.ndarray], obs_weights: list[np.ndarray], lam: float, n_coeffs: int,
) -> np.ndarray:
    """Sigma_theta_i^-1 = lam*I + sum_p beta_{p,i}^2 * phi(d_p) phi(d_p)^T,
    for each splat i, computed directly (ragged per-splat design matrices,
    an O(N) Python loop) -- the scalar reference
    `gpu_sh_directional_uncertainty.accumulate_sh_precision` (the real,
    checkpoint-scale accumulation, by rerendering training cameras) is
    cross-validated against; not meant for real checkpoint scale.

    `design_matrices[i]`: (m_i, n_coeffs) real `sh_basis` rows at splat i's
    own observed training directions (m_i == 0 is valid -- falls back to
    the prior `lam*I` exactly, the correct no-data precision).
    `obs_weights[i]`: (m_i,) real per-observation beta_{p,i} = T*alpha
    (splat i's OWN real alpha-compositing weight in that training
    observation, not a generic weight -- see this module's own docstring).
    """
    out = np.empty((len(design_matrices), n_coeffs, n_coeffs))
    prior = lam * np.eye(n_coeffs)
    for i, phi in enumerate(design_matrices):
        phi = np.asarray(phi, dtype=float)
        if phi.shape[0] == 0:
            out[i] = prior
            continue
        beta = np.asarray(obs_weights[i], dtype=float)
        out[i] = prior + (phi * (beta**2)[:, None]).T @ phi
    return out


def invert_precision(precision: np.ndarray) -> np.ndarray:
    """Sigma_theta_i = (Sigma_theta_i^-1)^-1, batched over the leading
    (n_splats,) axis -- kept as a separate step from
    `per_splat_coefficient_precision` since the GPU accumulation path
    (`gpu_sh_directional_uncertainty.accumulate_sh_precision`) builds the
    same precision tensor a different way but shares this inversion."""
    return np.linalg.inv(precision)


def directional_variance(sigma_theta: np.ndarray, query_directions: np.ndarray, degree: int) -> np.ndarray:
    """s_i^2(d_q) = phi(d_q)^T Sigma_theta_i phi(d_q), batched over any
    shared leading shape between `sigma_theta` (..., n_coeffs, n_coeffs)
    and `query_directions` (..., 3)."""
    phi = sh_basis(query_directions, degree)
    return np.einsum("...i,...ij,...j->...", phi, sigma_theta, phi)


def propagate_pixel_directional_uncertainty(alpha_weights: np.ndarray, s2: np.ndarray) -> float:
    """u_SH(q) = sum_i beta_{q,i}^2 * s_i^2(d_q) -- independent per-splat
    coefficient posteriors, propagated through the REAL (fixed, not solved
    for) alpha-compositing weights at the query pixel."""
    return float(np.sum(np.asarray(alpha_weights) ** 2 * np.asarray(s2)))
