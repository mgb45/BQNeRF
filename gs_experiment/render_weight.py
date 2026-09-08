"""Renderer-aware weight function a_q(xi) = T_q(xi) sigma(xi) G_q(xi) for a
Bayesian-quadrature query q (a pixel or ray), per the rendering-aware BQ
construction: place a GP prior on radiance c(xi), and note the rendered
integrand f_q(xi) = a_q(xi) c(xi) is then also a GP, with kernel

    k_q(xi, xi') = a_q(xi) k_base(xi, xi') a_q(xi').

This module defines a_q itself, kept separate from k_base (gs_experiment/kernels.py)
since a_q is renderer/query-specific (depends on transmittance, opacity,
footprint, visibility for one particular pixel/ray) while k_base is a
property of the underlying radiance field's assumed smoothness, shared
across all queries.

GaussianRenderWeight models a_q as an *unnormalized* Gaussian bump --
`amplitude` is literally the peak value at `center` (a_q(center) ==
amplitude), not a normalized probability density -- so a real per-splat
opacity (already in [0, 1]) can be passed straight in as `amplitude`
without a separate normalization step. `covariance` is the bump's spread:
the combined transmittance-decay / opacity / footprint envelope width for
this query, in whatever D-dimensional space the base kernel is defined
over (a 1D ray-depth toy domain, 3D world position, or position+direction
jointly).

A real trap with the peak-amplitude convention, found the hard way while
generating real-checkpoint demo renders (see
gs_experiment/results/FINDINGS.md): a_q's *total integrated mass*,
`integral a_q(xi) dxi = amplitude * (2 pi)^(D/2) * |covariance|^(1/2)`,
scales with the *volume* of `covariance` -- so fixing `amplitude` (e.g. to
a real opacity) and letting `covariance` be a real splat's actual (tiny)
3D footprint makes the total mass vanish (down to `1e-100`-scale in
practice), even though the opacity itself is not small at all. Real alpha
compositing weights are dimensionless and bounded (`sum_i T_i alpha_i <=
1`); a mass that shrinks arbitrarily with an unrelated spatial-scale
choice cannot represent that quantity in any comparable way across query
points, footprint scales, or scenes. `from_total_mass` is the fix: build
a GaussianRenderWeight with a *given* shape but a `total_mass` pinned to
a real, bounded quantity, solving for whatever peak `amplitude` that
requires -- use it whenever `amplitude` is meant to represent "how much
real rendering weight is here," not literally the pointwise peak value
of some other, unrelated quantity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GaussianRenderWeight:
    amplitude: float
    center: np.ndarray
    covariance: np.ndarray

    def __post_init__(self):
        self.amplitude = float(self.amplitude)
        self.center = np.atleast_1d(np.asarray(self.center, dtype=float))
        d = self.center.shape[0]
        cov = np.asarray(self.covariance, dtype=float)
        if cov.ndim == 0:
            cov = float(cov) * np.eye(d)
        elif cov.ndim == 1:
            cov = np.diag(cov)
        if cov.shape != (d, d):
            raise ValueError(f"covariance must broadcast to ({d}, {d}), got shape {cov.shape}")
        self.covariance = cov

    @property
    def dim(self) -> int:
        return self.center.shape[0]

    @property
    def total_mass(self) -> float:
        """integral a_q(xi) dxi = amplitude * (2 pi)^(D/2) * |covariance|^(1/2)
        -- computed in log-space (amplitude and the log-determinant term
        separately) since |covariance| can be extremely small for a real,
        tiny splat footprint and would otherwise underflow before the
        multiplication."""
        d = self.dim
        _, logdet = np.linalg.slogdet(self.covariance)
        log_norm = 0.5 * (d * np.log(2 * np.pi) + logdet)
        return float(self.amplitude * np.exp(log_norm))

    @classmethod
    def from_total_mass(cls, total_mass: float, center: np.ndarray, covariance: np.ndarray) -> "GaussianRenderWeight":
        """Build a GaussianRenderWeight with a given spatial shape
        (`center`, `covariance`) but with `amplitude` solved so that
        `total_mass` (see this class's and this module's docstrings for
        why peak-normalizing `amplitude` directly is the wrong default
        when it's meant to represent a real, bounded rendering-weight
        budget) comes out exactly right, however small or large
        `covariance`'s volume happens to be."""
        center_arr = np.atleast_1d(np.asarray(center, dtype=float))
        d = center_arr.shape[0]
        cov = np.asarray(covariance, dtype=float)
        if cov.ndim == 0:
            cov_matrix = float(cov) * np.eye(d)
        elif cov.ndim == 1:
            cov_matrix = np.diag(cov)
        else:
            cov_matrix = cov
        _, logdet = np.linalg.slogdet(cov_matrix)
        log_norm = 0.5 * (d * np.log(2 * np.pi) + logdet)
        amplitude = float(total_mass) * np.exp(-log_norm)
        return cls(amplitude=amplitude, center=center_arr, covariance=covariance)

    def __call__(self, xi: np.ndarray) -> np.ndarray:
        """Pointwise evaluation of a_q at one or more points. `xi`: (D,) or
        (N, D). Returns shape (N,). Used by the numerical-integration
        fallback and by tests cross-checking the closed form -- not on the
        closed-form fast path, which never evaluates a_q pointwise."""
        xi = np.atleast_2d(np.asarray(xi, dtype=float))
        diff = xi - self.center[None, :]
        inv_cov = np.linalg.inv(self.covariance)
        quad = np.einsum("ni,ij,nj->n", diff, inv_cov, diff)
        return self.amplitude * np.exp(-0.5 * quad)
