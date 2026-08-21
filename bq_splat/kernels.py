"""Covariance kernels for 1D Bayesian quadrature over a ray/pixel integral.

Each kernel provides:
  - k(x, y):    pairwise covariance
  - v(x, a, b): kernel mean embedding, i.e. integral of k(x, t) dt over [a, b]
  - vv(a, b):   double integral of k(x, y) dx dy over [a, b] x [a, b]

`v` is given in closed form where cheap (RBF) or by 1D numerical
integration otherwise (Matern). `vv` is always obtained by integrating `v`
numerically over [a, b] — this avoids trusting a hand-derived double-integral
antiderivative (the original models/nerf.py RBF formulas take that riskier
route, and this repo's git history already records a "double quad" bug from
that kind of derivation). A closed-form, batched vv is future engineering
work once this milestone's validation passes (see ROADMAP.md); at toy scale,
one 1D quadrature call per kernel evaluation is not a bottleneck.
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
    evaluated at x. Matches `NeRF.rbf` in models/nerf.py."""

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


class MaternKernel(Kernel):
    """Matern-3/2 kernel: k(r) = (1 + sqrt(3)|r|/rho) exp(-sqrt(3)|r|/rho).

    Once-differentiable sample paths, vs. RBF's infinitely-smooth ones —
    this is the kernel the original tutorial notebook derived but never
    wired into the live NeRF model.
    """

    name = "matern32"

    def __init__(self, rho: float):
        self.rho = float(rho)

    def k(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        r = np.abs(x - y)
        c = np.sqrt(3.0) * r / self.rho
        return (1.0 + c) * np.exp(-c)

    def v(self, x, a, b):
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.empty_like(x_arr)
        for i, xi in enumerate(x_arr):
            breakpoints = [xi] if a < xi < b else None
            out[i], _ = integrate.quad(lambda t, xi=xi: float(self.k(xi, t)), a, b, points=breakpoints)
        return out if out.shape[0] > 1 else out[0]

    def vv(self, a, b):
        val, _ = integrate.quad(lambda y: float(self.v(y, a, b)), a, b)
        return val
