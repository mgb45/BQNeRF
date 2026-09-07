"""Renderer-aware weight function a_q(xi) = T_q(xi) sigma(xi) G_q(xi) for a
Bayesian-quadrature query q (a pixel or ray), per the rendering-aware BQ
construction: place a GP prior on radiance c(xi), and note the rendered
integrand f_q(xi) = a_q(xi) c(xi) is then also a GP, with kernel

    k_q(xi, xi') = a_q(xi) k_base(xi, xi') a_q(xi').

This module defines a_q itself, kept separate from k_base (bq_splat/kernels.py)
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
