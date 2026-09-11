"""Covariance kernels for Bayesian quadrature over the position domain.

Each kernel provides:
  - k(x, y):    pairwise covariance
  - v(x, a, b): kernel mean embedding, i.e. integral of k(x, t) dt over [a, b]
  - vv(a, b):   double integral of k(x, y) dx dy over [a, b] x [a, b]

`v` is given in closed form (RBF). `vv` is always obtained by integrating
`v` numerically over [a, b] -- this avoids trusting a hand-derived
double-integral antiderivative (this repo's git history already records a
"double quad" bug from that kind of derivation).
"""

from __future__ import annotations

import numpy as np
from scipy import integrate
from scipy.stats import norm


class Kernel:
    name = "kernel"

    def k(self, x, y):
        raise NotImplementedError

    def v(self, x, a, b):
        raise NotImplementedError

    def vv(self, a, b):
        raise NotImplementedError


class RBFKernel(Kernel):
    """Normalized Gaussian kernel: k(x, y) is the density of N(y, sigma^2)
    evaluated at x."""

    name = "rbf"

    def __init__(self, sigma: float):
        self.sigma = float(sigma)

    def k(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        sig = self.sigma
        return np.exp(-((x - y) ** 2) / (2 * sig**2)) / (sig * np.sqrt(2 * np.pi))

    def v(self, x, a, b):
        x = np.asarray(x, dtype=float)
        sig = self.sigma
        return norm.cdf((b - x) / sig) - norm.cdf((a - x) / sig)

    def vv(self, a, b):
        val, _ = integrate.quad(lambda y: float(self.v(y, a, b)), a, b)
        return val


class ProductKernel:
    """A D-dimensional kernel built as the product of D 1D kernels, one per
    axis: k(x, y) = prod_d k_d(x_d, y_d).

    For RBF this is *exact*: an isotropic Gaussian in D dimensions with
    squared-Euclidean distance factorizes exactly into a product of 1D
    Gaussians along each axis (a standard identity, since
    ||x-y||^2 = sum_d (x_d-y_d)^2 and exp of a sum is a product of exps),
    so `ProductKernel([RBFKernel(sigma)]*D)` is exactly the isotropic D-D
    RBF kernel, not an approximation of it.

    Building it this way means `v`/`vv` over an axis-aligned box domain also
    factorize into products of the already-implemented/tested 1D `v`/`vv`
    calls -- no new integration code, and no new numerical risk.
    """

    name = "product"

    def __init__(self, kernels_per_axis):
        self.kernels_per_axis = list(kernels_per_axis)

    @property
    def d(self):
        return len(self.kernels_per_axis)

    def k(self, X, Y):
        """X: (N, D) or (D,); Y: (M, D) or (D,). Returns (N, M)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        out = np.ones((X.shape[0], Y.shape[0]))
        for dim, kernel in enumerate(self.kernels_per_axis):
            out = out * kernel.k(X[:, dim].reshape(-1, 1), Y[:, dim].reshape(1, -1))
        return out

    def v(self, X, bounds):
        """X: (N, D) or (D,). bounds: sequence of D (a_d, b_d) pairs.
        Returns shape (N,)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = np.ones(X.shape[0])
        for dim, kernel in enumerate(self.kernels_per_axis):
            a_d, b_d = bounds[dim]
            out = out * np.atleast_1d(kernel.v(X[:, dim], a_d, b_d))
        return out

    def vv(self, bounds):
        val = 1.0
        for dim, kernel in enumerate(self.kernels_per_axis):
            a_d, b_d = bounds[dim]
            val = val * kernel.vv(a_d, b_d)
        return val
