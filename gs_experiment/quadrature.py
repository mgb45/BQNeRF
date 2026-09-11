"""Bayesian quadrature: posterior mean/variance of an integral given noiseless
(or homoscedastic-noise) point evaluations of the integrand, under a GP prior
with a chosen kernel.

Formulation note (important, and the reason this module exists separately
from a direct port of models/nerf.py): the observations passed in here must
be point evaluations g(t_i) of the *integrand* of the rendering integral
C = integral of g(t) dt, not pre-integrated per-bin contributions like
weight_i * color_i (which already bake in an implicit bin width). Treating
an already-integrated quantity as a further point evaluation to integrate
again silently double-counts the bin width — this repo's own git history
has a "fixed bug in bq quadrature, was doing double quad" commit, which is
exactly this trap.

This module supplies u_spatial_BQ(q)'s scalar reference
(`rendering_aware_alternative_weight_risk`, used by
`pixel_uncertainty.LocalUncertaintyEngine.rendering_aware_alpha_risk_along_
ray`) in the renderer-consistent sparse-GP decomposition -- see
gs_experiment/sh_directional_uncertainty.py's module docstring for the full
picture. It is deliberately position-only: directional uncertainty is a
separate term, computed by sh_directional_uncertainty.py /
gpu_sh_directional_uncertainty.py instead of a joint position+direction
kernel here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.stats import multivariate_normal

from gs_experiment.kernels import ProductKernel, RBFKernel


@dataclass
class BQResult:
    mean: float
    variance: float


def _posterior_mean_variance(kxx: np.ndarray, values: np.ndarray, moment_vector: np.ndarray, prior_variance: float):
    """Shared BQ posterior-mean/variance solve: `mean = moment_vector @
    solve(kxx, values)`, `variance = prior_variance - moment_vector @
    solve(kxx, moment_vector)`. Every BQ function in this module needs
    both, against the *same* `kxx` -- solving with one Cholesky
    factorization of `kxx` (symmetric positive-(semi)definite by
    construction: a kernel Gram matrix plus jitter) for both right-hand
    sides at once, instead of two independent `np.linalg.solve` calls
    (each of which would redundantly re-factor `kxx` from scratch via a
    general, non-symmetric LU), is both the numerically appropriate
    choice for an SPD matrix and roughly 4x fewer flops (one O(n^3/3)
    Cholesky factorization instead of two O(2n^3/3) LU factorizations).
    A real, measured bottleneck at real-checkpoint scale, not a
    theoretical one: this is what
    `gs_experiment.pixel_uncertainty.LocalUncertaintyEngine`'s per-pixel
    rendering-aware queries spend most of their time in once candidate
    counts reach the low hundreds.

    Falls back to `np.linalg.solve` (the original approach) if Cholesky
    fails -- e.g. jitter insufficient for a particular Gram matrix to be
    numerically SPD -- trading speed for robustness rather than raising.
    """
    rhs = np.column_stack([values, moment_vector])
    try:
        factor = cho_factor(kxx, lower=True)
        solved = cho_solve(factor, rhs)
    except LinAlgError:
        solved = np.linalg.solve(kxx, rhs)
    mean = float(moment_vector @ solved[:, 0])
    variance = float(prior_variance - moment_vector @ solved[:, 1])
    return mean, max(variance, 0.0)


# ---------------------------------------------------------------------------
# Rendering-aware Bayesian quadrature: k_q(xi, xi') = a_q(xi) k_base(xi, xi')
# a_q(xi'), where a_q = T_q * sigma * G_q is a renderer/query-specific weight
# (gs_experiment/render_weight.py) and k_base is a prior over the *radiance field*
# c(xi), not over the rendering integrand a_q * c directly.
#
# Two quantities matter: z_{q,i} = integral a_q(xi) k_base(xi, x_i) dxi (the
# renderer-aware moment vector -- how much this query's rendering integral
# depends on node i) and z_{q,0} = integral integral a_q(xi) a_q(xi') k_base(xi, xi')
# dxi dxi'. K itself is untouched: K_ij = k_base(x_i, x_j), since nodes are
# noiseless (or homoscedastic-noise) observations of c(x_i), not of a_q(x_i)*c(x_i).
#
# Closed form for an isotropic RBF k_base and a GaussianRenderWeight a_q
# (both Gaussian in xi) via the standard Gaussian-product/convolution
# identity, applied once for z and twice for z0 -- see each function's
# docstring for the derivation.
# ---------------------------------------------------------------------------


def rendering_aware_moment_vector(nodes, render_weight, sigma_rbf: float) -> np.ndarray:
    """Closed-form z_{q,i} = integral a_q(xi) k_base(xi, x_i) dxi, for an
    isotropic RBF k_base(xi, xi') = N(xi; xi', sigma_rbf^2 I) (the already-
    validated normalized-Gaussian-density convention RBFKernel/ProductKernel
    use elsewhere in this package) and a_q a GaussianRenderWeight (an
    *unnormalized* Gaussian bump, a_q(xi) = amplitude * exp(-0.5 (xi-mu_q)^T
    Sigma_q^-1 (xi-mu_q))).

    Derivation: write a_q(xi) = A * N(xi; mu_q, Sigma_q), i.e. fold the
    unnormalized bump into a properly normalized Gaussian density times a
    constant A = amplitude * (2 pi)^(D/2) |Sigma_q|^(1/2) (the normalizer
    that N(xi; mu_q, Sigma_q) itself divides out). Then, by the standard
    identity integral N(xi; a, A) N(xi; b, B) dxi = N(a; b, A+B):

        z_{q,i} = A * integral N(xi; mu_q, Sigma_q) N(xi; x_i, sigma_rbf^2 I) dxi
                = A * N(mu_q; x_i, Sigma_q + sigma_rbf^2 I).

    Exact, not a numerical approximation.
    """
    nodes = np.atleast_2d(np.asarray(nodes, dtype=float))
    d = render_weight.dim
    cov_sum = render_weight.covariance + (sigma_rbf**2) * np.eye(d)
    _, logdet = np.linalg.slogdet(render_weight.covariance)
    log_A = np.log(render_weight.amplitude) + 0.5 * (d * np.log(2 * np.pi) + logdet)
    rv = multivariate_normal(mean=render_weight.center, cov=cov_sum)
    return np.exp(log_A) * np.atleast_1d(rv.pdf(nodes))


def rendering_aware_prior_variance(render_weight, sigma_rbf: float) -> float:
    """Closed-form z_{q,0} = integral integral a_q(xi) a_q(xi') k_base(xi, xi')
    dxi dxi', for the same isotropic-RBF/GaussianRenderWeight pairing as
    rendering_aware_moment_vector.

    Derivation: applying the same Gaussian-product identity twice (first
    integrating out xi against a_q(xi) and k_base(xi, xi'), then integrating
    the result -- itself proportional to a normalized Gaussian in xi' with
    mean mu_q, covariance Sigma_q + sigma_rbf^2 I -- against the remaining
    a_q(xi') factor) collapses the double integral to

        z_{q,0} = A^2 * N(mu_q; mu_q, 2 Sigma_q + sigma_rbf^2 I),

    with A as in rendering_aware_moment_vector. Exact, not a numerical
    approximation.
    """
    d = render_weight.dim
    cov_sum = 2 * render_weight.covariance + (sigma_rbf**2) * np.eye(d)
    _, logdet = np.linalg.slogdet(render_weight.covariance)
    log_A = np.log(render_weight.amplitude) + 0.5 * (d * np.log(2 * np.pi) + logdet)
    rv = multivariate_normal(mean=render_weight.center, cov=cov_sum)
    return float(np.exp(2 * log_A) * rv.pdf(render_weight.center))


def _rendering_aware_moments(nodes, render_weight, sigma_rbf, rel_jitter, noise_variance=0.0):
    """Shared plumbing for `bayesian_quadrature_rendering_aware` and
    `rendering_aware_alternative_weight_risk`: builds K (from the isotropic
    RBF base kernel alone -- nodes are noiseless/homoscedastic-noise
    observations of the radiance field c, not of a_q*c) and the
    renderer-aware z/z0 moments. Returns `(kxx_or_None, z, z0)`; `kxx` is
    `None` when there are zero nodes (nothing to solve against -- the
    posterior collapses to the prior, variance = z0).

    `noise_variance` (default 0.0, i.e. unchanged behavior): a real,
    homoscedastic observation-noise variance added to K's diagonal on top
    of (not instead of) `rel_jitter`'s tiny numerical-conditioning term --
    the standard GP-regression noisy-observation model, `y_i = f(x_i) +
    eps_i`, `eps_i ~ N(0, noise_variance)` iid, rather than treating every
    splat color as an exact, noiseless constraint. `rel_jitter` alone
    (~1e-4 relative to the diagonal) exists purely to keep K numerically
    SPD; it is not a real noise model and was never intended to absorb the
    role `noise_variance` plays here. Motivation: real splat positions
    routinely include near-duplicate points (observed directly on a real
    checkpoint -- pairwise distances as small as 1e-4 units under a
    ~0.1-unit kernel bandwidth), which makes the noiseless K badly
    ill-conditioned and forces the posterior mean to exactly interpolate
    near-duplicate, possibly-conflicting color observations -- a classic
    Runge's-phenomenon-style oscillation, empirically visible as
    per-channel color speckle in `C_BQ` renders even where the
    reconstructed structure is otherwise sharp. A positive `noise_variance`
    relaxes the exact-interpolation constraint, trading a small amount of
    fit-to-data for a much better-conditioned, less-oscillatory posterior.
    """
    nodes = np.atleast_2d(np.asarray(nodes, dtype=float))
    n = nodes.shape[0]
    d = render_weight.dim

    base = ProductKernel([RBFKernel(sigma_rbf)] * d)
    z0 = rendering_aware_prior_variance(render_weight, sigma_rbf)
    z = rendering_aware_moment_vector(nodes, render_weight, sigma_rbf) if n else np.zeros(0)

    if n == 0:
        return None, z, float(z0)

    kxx = base.k(nodes, nodes)
    jitter = rel_jitter * np.mean(np.diag(kxx))
    kxx = kxx + (jitter + noise_variance) * np.eye(n)
    return kxx, z, float(z0)


def bayesian_quadrature_rendering_aware(
    nodes,
    values,
    render_weight,
    sigma_rbf: float,
    rel_jitter: float = 1e-4,
    noise_variance: float = 0.0,
) -> BQResult:
    """The rendering-aware BQ posterior mean/variance itself supplies both
    the rendering estimate and its uncertainty --

        C_q | c ~ N(z_q^T K^-1 c, z_{q,0} - z_q^T K^-1 z_q)

    -- with `values` the observed splat colors c(x_i) (not pre-weighted
    rendering-integrand values -- see this module's earlier double-quad-bug
    warning, which applies here too: `values[i]` must be a color, not
    a_q(x_i)*color_i).

    `noise_variance` (default 0.0): see `_rendering_aware_moments`'s
    docstring -- a real homoscedastic observation-noise term, K -> K +
    noise_variance*I, in place of treating every observed color as exact.

    This mean need not reproduce standard alpha compositing -- see
    `rendering_aware_alternative_weight_risk` for scoring the real
    alpha-compositing weights under this same posterior instead.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    kxx, z, z0 = _rendering_aware_moments(nodes, render_weight, sigma_rbf, rel_jitter, noise_variance)
    if kxx is None:
        return BQResult(mean=0.0, variance=max(z0, 0.0))

    mean, variance = _posterior_mean_variance(kxx, values, z, z0)
    return BQResult(mean=mean, variance=variance)


def rendering_aware_alternative_weight_risk(
    nodes,
    values,
    render_weight,
    weights,
    sigma_rbf: float,
    rel_jitter: float = 1e-4,
    noise_variance: float = 0.0,
) -> tuple[float, float]:
    """Evaluates *any* real, literal weight vector `weights` (aligned 1:1
    with `nodes`) as an estimator of the same rendering functional
    `L_q[f] = integral a_q(x) f(x) p(x) dx` that
    `bayesian_quadrature_rendering_aware` computes the BQ-*optimal* estimator
    for -- most usefully, the real alpha-compositing weights
    `w_i = T_i * alpha_i` from
    `gs_experiment.visibility_attribution.ray_transmittance_weights`, so the
    actual renderer's own quadrature rule can be scored under the same
    kernel/RKHS this project's posterior variance is built from, instead of
    silently pairing that variance with a mean it was never computed for.
    This is `u_spatial_BQ(q)` in the renderer-consistent sparse-GP
    decomposition (see gs_experiment/sh_directional_uncertainty.py).

    This is the general RKHS worst-case-squared-error quadratic form for
    *any* linear estimator `Q_w[f] = sum_i w_i f(x_i)`:

        e(w)^2 = z_0 - 2 w^T z + w^T K w

    (the classical Bayes-Hermite/kernel-quadrature result). It is uniquely
    minimized at the BQ-optimal weights `w* = K^-1 z`
    (`bayesian_quadrature_rendering_aware`'s own weights), where it reduces
    exactly to that function's reported `variance`. Any other real weight
    vector -- in particular, one that actually gets *used* to render (alpha
    compositing) rather than chosen to minimize this quantity -- gives a
    valid but generically *larger* worst-case error: this is the
    quantitative sense in which "how good is the alpha-compositing rule
    itself, under this same posterior" is a well-posed, directly comparable
    question, not a category error.

    Returns `(mean, risk)`: `mean = weights @ values` (the estimator's own
    predicted value, e.g. a real, locally-windowed alpha-compositing color
    estimate -- generally *not* bit-identical to a full scene renderer's
    actual per-pixel output, since it only sees this call's local `nodes`,
    not every splat along the real ray/footprint; report both if that gap
    matters), `risk = max(e(w)^2, 0)` (clamped against small negative
    values from floating-point cancellation, same convention as this
    module's other variance-like returns).
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    kxx, z, z0 = _rendering_aware_moments(nodes, render_weight, sigma_rbf, rel_jitter, noise_variance)

    mean = float(weights @ values) if weights.shape[0] == values.shape[0] and values.shape[0] > 0 else 0.0
    if kxx is None:
        return mean, max(z0, 0.0)

    risk = float(z0 - 2.0 * (weights @ z) + weights @ kxx @ weights)
    return mean, max(risk, 0.0)
