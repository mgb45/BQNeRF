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


class ProductKernel:
    """A D-dimensional kernel built as the product of D 1D kernels, one per
    axis: k(x, y) = prod_d k_d(x_d, y_d).

    For RBF this is *exact*: an isotropic Gaussian in D dimensions with
    squared-Euclidean distance factorizes exactly into a product of 1D
    Gaussians along each axis (a standard identity, since
    ||x-y||^2 = sum_d (x_d-y_d)^2 and exp of a sum is a product of exps),
    so `ProductKernel([RBFKernel(sigma)]*D)` is exactly the isotropic D-D
    RBF kernel, not an approximation of it. For Matern this is a legitimate
    but different (axis-aligned, "tensor-product"/ARD-style) kernel, not
    identical to the radially-isotropic Matern -- a standard, positive-
    definite construction, just worth naming precisely.

    Building it this way means `v`/`vv` over an axis-aligned box domain also
    factorize into products of the already-implemented/tested 1D `v`/`vv`
    calls -- no new integration code, and no new numerical risk, for either
    kernel family.
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


class DirectionalKernel:
    """Von Mises-Fisher-style kernel on directions (unit vectors on
    S^(d-1)): k(w, w') = exp(kappa * (w . w' - 1)).

    Positive-definite for kappa >= 0: w.w' is itself a (linear, hence PD)
    kernel, exp() of a PD kernel scaled by a positive constant is PD (each
    term of its power series is a nonnegative combination of PD kernels, by
    the Schur product theorem), and this is that PD kernel times the
    positive constant exp(-kappa). `kappa` plays the role RBF's 1/sigma^2
    plays for spatial separation, but for angular separation: large kappa
    means only very similar directions are considered correlated (a highly
    view-dependent/specular surface needs many close viewing angles to be
    well-constrained); small kappa means most directions are considered
    similar (near-Lambertian, one observation generalizes across angles).

    Self-similarity k(w, w) = exp(kappa * (1 - 1)) = 1 always — this is
    what makes the mixed integrate-position/evaluate-direction Bayesian
    quadrature in bayesian_quadrature_directional work out cleanly (see
    that function's docstring): the *prior* variance term doesn't depend on
    which direction is queried, only the *posterior reduction* does, via
    the k(w_i, w_query) terms in the mean-embedding-like vector.

    Unlike Kernel (RBFKernel, MaternKernel), this has no v/vv — it's never
    integrated over, only evaluated pointwise at a query direction, since a
    rendered image evaluates one specific outgoing direction per pixel, not
    an integral over a range of directions.
    """

    name = "vonmises"

    def __init__(self, kappa: float):
        self.kappa = float(kappa)

    def k(self, w, w_prime):
        w = np.atleast_2d(np.asarray(w, dtype=float))
        w_prime = np.atleast_2d(np.asarray(w_prime, dtype=float))
        dot = w @ w_prime.T
        return np.exp(self.kappa * (dot - 1.0))


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
