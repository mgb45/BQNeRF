"""Bayesian quadrature: posterior mean/variance of an integral given noiseless
point evaluations of the integrand, under a GP prior with a chosen kernel.

Formulation note (important, and the reason this module exists separately
from a direct port of models/nerf.py): the observations passed in here must
be point evaluations g(t_i) of the *integrand* of the rendering integral
C = integral of g(t) dt, not pre-integrated per-bin contributions like
weight_i * color_i (which already bake in an implicit bin width). Treating
an already-integrated quantity as a further point evaluation to integrate
again silently double-counts the bin width — this repo's own git history
has a "fixed bug in bq quadrature, was doing double quad" commit, which is
exactly this trap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import integrate
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.stats import multivariate_normal

from gs_experiment.kernels import DirectionalKernel, ProductKernel, RBFKernel


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
# c(xi), not over the rendering integrand a_q * c directly (the distinction
# that fixes what the uniform-box quadrature domain this module used to use
# got wrong; retired, see git history).
#
# Two quantities matter: z_{q,i} = integral a_q(xi) k_base(xi, x_i) dxi (the
# renderer-aware moment vector -- how much this query's rendering integral
# depends on node i, in place of v's uniform-box "how much kernel mass is in
# this box") and z_{q,0} = integral integral a_q(xi) a_q(xi') k_base(xi, xi')
# dxi dxi' (in place of vv). K itself is untouched: K_ij = k_base(x_i, x_j),
# since nodes are noiseless observations of c(x_i), not of a_q(x_i)*c(x_i).
#
# Closed form for an isotropic RBF k_base and a GaussianRenderWeight a_q
# (both Gaussian in xi) via the standard Gaussian-product/convolution
# identity, applied once for z and twice for z0 -- see each function's
# docstring for the derivation. A numerical (scipy.integrate.nquad) fallback
# is kept alongside for a general ProductKernel k_base (e.g. Matern) or a
# non-Gaussian a_q, and as the ground-truth cross-check for the closed form
# (tests/gs_experiment/test_render_weight.py) -- the same "closed form where cheap,
# numerically cross-checked" discipline gs_experiment/kernels.py already follows.
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

    Exact, not a numerical approximation -- cross-checked against
    numerical_rendering_moment_vector in tests/gs_experiment/test_render_weight.py.
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
    approximation -- cross-checked against numerical_rendering_prior_variance
    in tests/gs_experiment/test_render_weight.py.
    """
    d = render_weight.dim
    cov_sum = 2 * render_weight.covariance + (sigma_rbf**2) * np.eye(d)
    _, logdet = np.linalg.slogdet(render_weight.covariance)
    log_A = np.log(render_weight.amplitude) + 0.5 * (d * np.log(2 * np.pi) + logdet)
    rv = multivariate_normal(mean=render_weight.center, cov=cov_sum)
    return float(np.exp(2 * log_A) * rv.pdf(render_weight.center))


def numerical_rendering_moment_vector(nodes, render_weight, base_kernel, domain) -> np.ndarray:
    """Numerical fallback for rendering_aware_moment_vector: z_{q,i} =
    integral a_q(xi) k_base(xi, x_i) dxi via scipy.integrate.nquad, for an
    arbitrary `base_kernel` (a ProductKernel -- k(X, Y) taking (N, D)/(M, D)
    arrays, e.g. Matern rather than RBF) and an arbitrary pointwise-callable
    `render_weight` (not necessarily Gaussian). `domain`: a list of D (lo,
    hi) bounds -- for a GaussianRenderWeight this should extend several
    standard deviations past `center` in each axis, since a_q's true support
    is all of R^D.

    Used both as a real fallback for non-RBF/non-Gaussian cases and as the
    ground-truth cross-check for the closed form (tests/gs_experiment/test_render_weight.py).
    """
    nodes = np.atleast_2d(np.asarray(nodes, dtype=float))
    out = np.empty(nodes.shape[0])
    for i, x in enumerate(nodes):
        x_row = x.reshape(1, -1)

        def integrand(*xi):
            xi_row = np.array(xi).reshape(1, -1)
            return float(render_weight(xi_row)[0] * base_kernel.k(xi_row, x_row)[0, 0])

        val, _ = integrate.nquad(integrand, domain)
        out[i] = val
    return out


def numerical_rendering_prior_variance(render_weight, base_kernel, domain) -> float:
    """Numerical fallback for rendering_aware_prior_variance: z_{q,0} =
    integral integral a_q(xi) a_q(xi') k_base(xi, xi') dxi dxi', via nested
    scipy.integrate.nquad over `domain` x `domain`. See
    numerical_rendering_moment_vector for the `base_kernel`/`domain`
    contract; this is its double-integral analogue, and the ground-truth
    cross-check for rendering_aware_prior_variance in tests/gs_experiment/test_render_weight.py
    for D=1 (a tractable 2D nquad integral).

    Scales badly past D=1: this integrates over `2*D` dimensions total (D
    for xi, D for xi'), and nquad's nested adaptive quadrature does not
    finish in any reasonable time once that reaches 4D (confirmed: D=2 does
    not complete in 100s even over a tight, few-sigma domain) -- an
    importance-sampled Monte Carlo estimate (sample xi, xi' directly from
    a_q's own Gaussian shape, average k_base(xi, xi')) is the practical
    fallback for D>=2, used in tests/gs_experiment/test_render_weight.py's D=2 cross-check
    instead of this function.
    """
    d = render_weight.dim

    def integrand(*args):
        xi_row = np.array(args[:d]).reshape(1, -1)
        xip_row = np.array(args[d:]).reshape(1, -1)
        return float(render_weight(xi_row)[0] * render_weight(xip_row)[0] * base_kernel.k(xi_row, xip_row)[0, 0])

    val, _ = integrate.nquad(integrand, list(domain) + list(domain))
    return val


def _rendering_aware_moments(nodes, render_weight, sigma_rbf, base_kernel, domain, mode, rel_jitter, noise_variance=0.0):
    """Shared plumbing for bayesian_quadrature_rendering_aware and
    renderer_centered_residual_variance: builds K (from k_base alone --
    nodes are noiseless observations of the radiance field c, not of
    a_q*c) and the renderer-aware z/z0 moments, via either the closed-form
    Gaussian path (`mode="closed_form"`) or the numerical fallback
    (`mode="numerical"`). Returns (base_kernel_used, kxx_or_None, z, z0);
    kxx is None when there are zero nodes (nothing to solve against -- the
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
    fit-to-data for a much better-conditioned, less-oscillatory posterior
    -- see `hyperparams.fit_kernel_param_and_noise` for fitting it (jointly
    with sigma) via marginal likelihood rather than picking it by hand."""
    nodes = np.atleast_2d(np.asarray(nodes, dtype=float))
    n = nodes.shape[0]
    d = render_weight.dim

    if mode == "closed_form":
        if sigma_rbf is None:
            raise ValueError("mode='closed_form' requires sigma_rbf (isotropic RBF base-kernel bandwidth)")
        base = ProductKernel([RBFKernel(sigma_rbf)] * d)
        z0 = rendering_aware_prior_variance(render_weight, sigma_rbf)
        z = rendering_aware_moment_vector(nodes, render_weight, sigma_rbf) if n else np.zeros(0)
    elif mode == "numerical":
        if base_kernel is None or domain is None:
            raise ValueError("mode='numerical' requires base_kernel and domain")
        base = base_kernel
        z0 = numerical_rendering_prior_variance(render_weight, base_kernel, domain)
        z = numerical_rendering_moment_vector(nodes, render_weight, base_kernel, domain) if n else np.zeros(0)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    if n == 0:
        return base, None, z, float(z0)

    kxx = base.k(nodes, nodes)
    jitter = rel_jitter * np.mean(np.diag(kxx))
    kxx = kxx + (jitter + noise_variance) * np.eye(n)
    return base, kxx, z, float(z0)


def bayesian_quadrature_rendering_aware(
    nodes,
    values,
    render_weight,
    sigma_rbf: float | None = None,
    base_kernel: ProductKernel | None = None,
    domain=None,
    rel_jitter: float = 1e-4,
    mode: str = "closed_form",
    noise_variance: float = 0.0,
) -> BQResult:
    """Formulation 1 ("renderer-aware Bayesian quadrature"): the rendering-
    aware BQ posterior mean/variance itself supplies both the rendering
    estimate and its uncertainty --

        C_q | c ~ N(z_q^T K^-1 c, z_{q,0} - z_q^T K^-1 z_q)

    -- with `values` the observed splat colors c(x_i) (not pre-weighted
    rendering-integrand values -- see this module's earlier double-quad-bug
    warning, which applies here too: `values[i]` must be a color, not
    a_q(x_i)*color_i).

    `mode="closed_form"` (default) requires `sigma_rbf`, the isotropic RBF
    base-kernel bandwidth (see rendering_aware_moment_vector); `mode=
    "numerical"` requires `base_kernel` (a ProductKernel, e.g. Matern) and
    `domain` (integration bounds, see numerical_rendering_moment_vector).

    `noise_variance` (default 0.0): see `_rendering_aware_moments`'s
    docstring -- a real homoscedastic observation-noise term, K -> K +
    noise_variance*I, in place of treating every observed color as exact.

    This mean need not reproduce standard alpha compositing -- see
    renderer_centered_residual_variance for the alternative formulation that
    keeps alpha compositing as the mean and uses only this same variance.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    _, kxx, z, z0 = _rendering_aware_moments(
        nodes, render_weight, sigma_rbf, base_kernel, domain, mode, rel_jitter, noise_variance
    )
    if kxx is None:
        return BQResult(mean=0.0, variance=max(z0, 0.0))

    mean, variance = _posterior_mean_variance(kxx, values, z, z0)
    return BQResult(mean=mean, variance=variance)


def renderer_centered_residual_variance(
    nodes,
    render_weight,
    sigma_rbf: float | None = None,
    base_kernel: ProductKernel | None = None,
    domain=None,
    rel_jitter: float = 1e-4,
    mode: str = "closed_form",
    noise_variance: float = 0.0,
) -> float:
    """Formulation 2 ("renderer-centred probabilistic quadrature"): keeps
    ordinary alpha compositing as the predictive mean, and uses only this
    module's rendering-aware variance to model the unresolved integration
    error around it --

        C_q = C_hat_q^3DGS + eps_q,     eps_q ~ N(0, z_{q,0} - z_q^T K^-1 z_q)

    Returns just that variance (the caller supplies and owns the mean).
    Shares its z/z0/K machinery exactly with bayesian_quadrature_rendering_aware
    (same _rendering_aware_moments call) -- the two formulations differ only
    in which mean the variance is paired with, per the prompt's own "there
    are two possible formulations" framing; neither is picked as uniquely
    correct here. `noise_variance`: see `_rendering_aware_moments`'s docstring.
    """
    _, kxx, z, z0 = _rendering_aware_moments(
        nodes, render_weight, sigma_rbf, base_kernel, domain, mode, rel_jitter, noise_variance
    )
    if kxx is None:
        return max(z0, 0.0)

    try:
        solved_z = cho_solve(cho_factor(kxx, lower=True), z)
    except LinAlgError:
        solved_z = np.linalg.solve(kxx, z)
    variance = float(z0 - z @ solved_z)
    return max(variance, 0.0)


def rendering_aware_alternative_weight_risk(
    nodes,
    values,
    render_weight,
    weights,
    sigma_rbf: float | None = None,
    base_kernel: ProductKernel | None = None,
    domain=None,
    rel_jitter: float = 1e-4,
    mode: str = "closed_form",
    noise_variance: float = 0.0,
) -> tuple[float, float]:
    """Formulation 3 ("BQ risk of an arbitrary quadrature rule"): evaluates
    *any* real, literal weight vector `weights` (aligned 1:1 with `nodes`)
    as an estimator of the same rendering functional
    `L_q[f] = integral a_q(x) f(x) p(x) dx` that
    `bayesian_quadrature_rendering_aware` computes the BQ-*optimal* estimator
    for -- most usefully, the real alpha-compositing weights
    `w_i = T_i * alpha_i` from
    `gs_experiment.visibility_attribution.ray_transmittance_weights`, so the
    actual renderer's own quadrature rule can be scored under the same
    kernel/RKHS this project's posterior variance is built from, instead of
    silently pairing that variance with a mean it was never computed for
    (`C_alpha` from a totally different renderer) -- see
    ROADMAP.md/FINDINGS.md's calibration-methodology follow-up for why this
    matters and how it's used.

    This is the general RKHS worst-case-squared-error quadratic form for
    *any* linear estimator `Q_w[f] = sum_i w_i f(x_i)`:

        e(w)^2 = z_0 - 2 w^T z + w^T K w

    (the classical Bayes-Hermite/kernel-quadrature result -- see e.g. the
    derivation this project's retired `bq_splat/PROOF_alpha_compositing_
    equivalence.md` gives in ray-depth-domain notation, Theorem B, connected
    here to this module's actual `z`/`kxx`/`z0` code paths). It is uniquely
    minimized at the BQ-optimal weights `w* = K^-1 z`
    (`bayesian_quadrature_rendering_aware`'s own weights), where it reduces
    exactly to that function's reported `variance` -- confirmed directly by
    `tests/gs_experiment/test_quadrature.py`'s
    `test_alternative_weight_risk_reduces_to_bq_variance_at_bq_weights` test,
    not just asserted. Any other real weight vector -- in particular, one
    that actually gets *used* to render (alpha compositing) rather than
    chosen to minimize this quantity -- gives a valid but generically
    *larger* worst-case error: this is the quantitative sense in which "how
    good is the alpha-compositing rule itself, under this same posterior" is
    a well-posed, directly comparable question, not a category error.

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
    _, kxx, z, z0 = _rendering_aware_moments(
        nodes, render_weight, sigma_rbf, base_kernel, domain, mode, rel_jitter, noise_variance
    )

    mean = float(weights @ values) if weights.shape[0] == values.shape[0] and values.shape[0] > 0 else 0.0
    if kxx is None:
        return mean, max(z0, 0.0)

    risk = float(z0 - 2.0 * (weights @ z) + weights @ kxx @ weights)
    return mean, max(risk, 0.0)


def bayesian_quadrature_rendering_aware_directional(
    positions,
    directions,
    values,
    render_weight,
    dir_kernel: DirectionalKernel,
    query_direction,
    sigma_rbf: float | None = None,
    base_kernel: ProductKernel | None = None,
    domain=None,
    rel_jitter: float = 1e-4,
    mode: str = "closed_form",
    noise_variance: float = 0.0,
) -> BQResult:
    """The full rendering-aware construction the original prompt specified
    and this module only partially implemented until now: a JOINT
    position+direction base kernel, `k_base(xi, xi') = k_pos(x, x') *
    k_dir(d, d')`, so that `k_q(xi, xi') = a_q(x) k_base(xi, xi') a_q(x')`
    carries both the renderer-specific spatial envelope
    (`bayesian_quadrature_rendering_aware`'s `render_weight`) *and* a
    directional/epistemic term (`dir_kernel`) at once, instead of the two
    living as separate, unconnected code paths. `a_q` itself is still
    position-only (T_q sigma G_q are about *where* along a ray/footprint
    mass concentrates, not which direction a splat happens to have been
    observed from) -- the directional dependence enters purely through
    `k_dir`, exactly as the prompt's `k_dir(d,d') = exp[kappa(d^T d' - 1)]`
    factor does.

    Position is integrated over via `render_weight`'s spatial envelope,
    direction is evaluated at one `query_direction`, not integrated -- a
    rendered pixel looks in one specific outgoing direction. Because
    `dir_kernel.k(d, d) == 1` always (DirectionalKernel's docstring),
    `z_{q,0}` is *exactly* the same spatial-only prior variance
    `rendering_aware_prior_variance`/`bayesian_quadrature_rendering_aware`
    already compute -- only `K` and the moment vector `z_q` pick up a
    directional factor:

        K_ij     = k_pos(x_i, x_j) * k_dir(d_i, d_j)
        z_{q,i}  = [integral a_q(xi) k_pos(xi, x_i) dxi] * k_dir(d_i, d_query)
        z_{q,0}  = integral integral a_q(xi) a_q(xi') k_pos(xi, xi') dxi dxi'

    `positions`/`directions`/`values` are parallel arrays, one row per
    (splat, observing-direction) pair -- a splat needs to be observed from
    *multiple* directions during training for this term to carry any
    signal at all; one row per splat with a single direction each cannot.

    `noise_variance` (default 0.0, unchanged behavior): a real
    homoscedastic observation-noise variance added to K's diagonal -- see
    `_rendering_aware_moments`'s docstring for the model and motivation.
    """
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    directions = np.asarray(directions, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)
    n = positions.shape[0]

    if mode == "closed_form":
        if sigma_rbf is None:
            raise ValueError("mode='closed_form' requires sigma_rbf (isotropic RBF base-kernel bandwidth)")
        d = render_weight.dim
        pos_kernel = ProductKernel([RBFKernel(sigma_rbf)] * d)
        z0 = rendering_aware_prior_variance(render_weight, sigma_rbf)
        z_pos = rendering_aware_moment_vector(positions, render_weight, sigma_rbf) if n else np.zeros(0)
    elif mode == "numerical":
        if base_kernel is None or domain is None:
            raise ValueError("mode='numerical' requires base_kernel and domain")
        pos_kernel = base_kernel
        z0 = numerical_rendering_prior_variance(render_weight, base_kernel, domain)
        z_pos = numerical_rendering_moment_vector(positions, render_weight, base_kernel, domain) if n else np.zeros(0)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    if n == 0:
        return BQResult(mean=0.0, variance=max(float(z0), 0.0))

    kxx = pos_kernel.k(positions, positions) * dir_kernel.k(directions, directions)
    jitter = rel_jitter * np.mean(np.diag(kxx))
    kxx = kxx + (jitter + noise_variance) * np.eye(n)

    v_dir = dir_kernel.k(directions, query_direction).reshape(-1)
    z = z_pos * v_dir

    mean, variance = _posterior_mean_variance(kxx, values, z, float(z0))
    return BQResult(mean=mean, variance=variance)


def bayesian_quadrature_rendering_aware_mixture_directional(
    positions,
    directions,
    values,
    weights,
    dir_kernel: DirectionalKernel,
    query_direction,
    sigma_rbf: float,
    local_covariances=None,
    rel_jitter: float = 1e-4,
    noise_variance: float = 0.0,
) -> BQResult:
    """Scalar (numpy) reference for `gpu_uncertainty.
    compute_directional_variance_batched_mixture`/`_solve_batched_mixture` --
    the mixture-a_q alternative to `bayesian_quadrature_rendering_aware_directional`.

    Motivation (see the conversation this was built for): moment-matching
    a_q into one Gaussian (`bayesian_quadrature_rendering_aware_directional`'s
    `render_weight`) discards which real candidate is which, so the
    resulting BQ weight vector w* = Kxx^-1 z has no structural relationship
    to the real alpha-compositing weights `weights` (w_i = T_i*alpha_i) for
    ANY kernel choice. Keeping a_q as the exact mixture of each candidate's
    own real (weight, position, covariance) instead:

        a_q(x)   = sum_k w_k * N(x; x_k, Sigma_k)
        K_ij     = k_pos(x_i, x_j) * k_dir(d_i, d_j)          -- unsmeared, same K as the single-Gaussian path
        z_i      = [sum_k w_k * N(x_k; x_i, Sigma_k + sigma_rbf^2 I)] * k_dir(d_i, d_query)
        z_0      = sum_k sum_l w_k*w_l * N(x_k; x_l, Sigma_k+Sigma_l+sigma_rbf^2 I)

    makes w* = weights EXACTLY whenever K is built the same (unsmeared) way
    z is -- i.e. in the point-splat limit `local_covariances -> 0`. With
    real (non-degenerate) `local_covariances`, w* is a *controlled*
    approximation to `weights` (the mismatch is O(Sigma_k) relative to the
    fully-consistent-but-invalid asymmetric kernel that would give exact
    equality -- see the conversation's derivation), not an unrelated
    quantity the way the single-Gaussian moment-matched a_q's w* is.

    `local_covariances` (N, 3, 3), optional: each candidate's own real 3D
    covariance (from scale/rotation). `None` (default) treats every
    candidate as a point (Sigma_k=0 for all k) -- the degenerate case that
    gives EXACT `w* = weights` for any sigma_rbf (see docstring above),
    included as a real, checkable special case, not just a fallback.

    Mirrors `gpu_uncertainty._solve_batched_mixture`'s math exactly (same
    variable names/formula structure) so a discrepancy between this and the
    batched path points at an actual implementation bug, not a difference
    in what's being computed -- see
    tests/gs_experiment/test_gpu_uncertainty_mixture.py for the
    cross-validation this enables.
    """
    n = np.atleast_2d(np.asarray(positions, dtype=float)).shape[0]
    if n == 0:
        return BQResult(mean=0.0, variance=0.0)

    z0, z, kxx = _mixture_directional_moments(
        positions, directions, weights, dir_kernel, query_direction, sigma_rbf, local_covariances, rel_jitter,
        noise_variance,
    )
    values = np.asarray(values, dtype=float).reshape(-1)
    mean, variance = _posterior_mean_variance(kxx, values, z, float(z0))
    return BQResult(mean=mean, variance=variance)


def _mixture_directional_moments(
    positions, directions, weights, dir_kernel: DirectionalKernel, query_direction, sigma_rbf: float,
    local_covariances=None, rel_jitter: float = 1e-4, noise_variance: float = 0.0,
):
    """Shared z0/z/kxx construction for
    `bayesian_quadrature_rendering_aware_mixture_directional` (solves for
    the BQ-optimal weights) and
    `rendering_aware_alternative_weight_risk_mixture_directional` (evaluates
    the risk of a FIXED weight vector, no solve at all) -- see the former's
    docstring for the formulas. Not part of the public API of this module
    (leading underscore) since callers always want one of those two
    finished results, never these raw pieces on their own.
    """
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    directions = np.asarray(directions, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    query_direction = np.asarray(query_direction, dtype=float)
    n, d = positions.shape

    if local_covariances is None:
        local_covariances = np.zeros((n, d, d))
    local_covariances = np.asarray(local_covariances, dtype=float)

    eye_d = np.eye(d)

    # z_0 = sum_k sum_l w_k*w_l*N(x_k;x_l,Sigma_k+Sigma_l+sigma_rbf^2 I).
    z0 = 0.0
    for k in range(n):
        for l in range(n):
            cov_kl = local_covariances[k] + local_covariances[l] + (sigma_rbf**2) * eye_d
            z0 += weights[k] * weights[l] * float(multivariate_normal(mean=positions[l], cov=cov_kl).pdf(positions[k]))

    # z_i = [sum_k w_k*N(x_k;x_i,Sigma_k+sigma_rbf^2 I)] * k_dir(d_i,d_query).
    z_pos = np.zeros(n)
    for i in range(n):
        total = 0.0
        for k in range(n):
            cov_k = local_covariances[k] + (sigma_rbf**2) * eye_d
            total += weights[k] * float(multivariate_normal(mean=positions[k], cov=cov_k).pdf(positions[i]))
        z_pos[i] = total
    v_dir = dir_kernel.k(directions, query_direction).reshape(-1)
    z = z_pos * v_dir

    pos_kernel = ProductKernel([RBFKernel(sigma_rbf)] * d)
    kxx = pos_kernel.k(positions, positions) * dir_kernel.k(directions, directions)
    jitter = rel_jitter * np.mean(np.diag(kxx))
    kxx = kxx + (jitter + noise_variance) * np.eye(n)
    return float(z0), z, kxx


def rendering_aware_alternative_weight_risk_mixture_directional(
    positions,
    directions,
    values,
    weights,
    dir_kernel: DirectionalKernel,
    query_direction,
    sigma_rbf: float,
    w=None,
    local_covariances=None,
    rel_jitter: float = 1e-4,
    noise_variance: float = 0.0,
) -> tuple[float, float]:
    """The mixture-a_q, joint-position+direction-kernel analogue of
    `rendering_aware_alternative_weight_risk` -- evaluates the RKHS
    worst-case-squared-error `e(w)^2 = z0 - 2 w^T z + w^T K w` at a real
    weight vector `w`, NOT the BQ-optimal weights -- no Cholesky
    factorization/solve at all, since `w` is already given, just three
    quadratic-form evaluations.

    `weights` defines the mixture a_q (`a_q(x) = sum_k weights_k *
    N(x;x_k,Sigma_k)`, the target measure -- always the real
    alpha-compositing weights `T_i*alpha_i` in this project) and therefore
    z0/z/Kxx; `w` (default `None`, meaning `w := weights`) is the SEPARATE
    vector actually being scored by e(w)^2. These are two different roles
    played by one array in the common case (score alpha compositing's own
    weights, which are also what a_q is built from -- the only way this
    function is actually invoked in this project) but must stay
    independent parameters for the general RKHS theory to hold at all --
    conflating them silently breaks the "reduces to BQ variance at
    w*=Kxx^-1 z" identity, since changing `w` would also change a_q (hence
    z0/z/Kxx) instead of scoring a different vector against the SAME fixed
    target measure. See
    tests/gs_experiment/test_quadrature_mixture_risk.py's
    test_risk_reduces_to_bq_variance_and_mean_at_bq_optimal_weights, which
    exercises `w != weights` specifically to check this.

    Why this exists (see the conversation this was built for): the plain
    directional BQ construction's weights (`bayesian_quadrature_rendering_
    aware_mixture_directional`'s w* = Kxx^-1 z) cannot equal `weights`
    once the directional kernel genuinely varies (z's k_dir(d_i,d_query)
    and Kxx's k_dir(d_i,d_j) play structurally different roles -- proven
    directly in tests/gs_experiment/test_gpu_uncertainty_mixture.py's
    test_directional_kernel_breaks_exact_recovery_at_realistic_kappa). But
    real alpha-compositing weights don't need to be *solved for* -- they're
    already known. This function keeps them fixed and asks the coherent
    question instead: "how good is the real renderer's own quadrature rule,
    under a kernel that DOES genuinely account for real directional
    coverage" -- i.e. does the actual real-per-splat mixture-and-direction
    kernel geometry (not the single-Gaussian moment-matched a_q the
    original `rendering_aware_alternative_weight_risk`/variant-5 comparison
    used) make alpha compositing's own risk a better-behaved, more
    informative calibration signal than the original single-Gaussian
    version found. `variant 5` (section 3/FINDINGS.md) underperformed
    `R_alpha` using the single-Gaussian a_q for its own z0/K/z geometry --
    this function retests that specific comparison with the corrected
    (mixture) geometry, not a new idea, a fairer version of an old one.

    Returns `(mean, risk)`: `mean = w @ values` (the estimator's own
    predicted value under `w` -- identical to `rendering_aware_
    alternative_weight_risk`'s own `mean`, unaffected by which kernel
    geometry scores the risk), `risk = max(e(w)^2, 0)`.
    """
    positions_arr = np.atleast_2d(np.asarray(positions, dtype=float))
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    w = weights if w is None else np.asarray(w, dtype=float).reshape(-1)
    n = positions_arr.shape[0]

    mean = float(w @ values) if w.shape[0] == values.shape[0] and n > 0 else 0.0
    if n == 0:
        return mean, 0.0

    z0, z, kxx = _mixture_directional_moments(
        positions, directions, weights, dir_kernel, query_direction, sigma_rbf, local_covariances, rel_jitter,
        noise_variance,
    )
    risk = float(z0 - 2.0 * (w @ z) + w @ kxx @ w)
    return mean, max(risk, 0.0)


def _mixture_moments(positions, weights, sigma_rbf: float, local_covariances=None, rel_jitter: float = 1e-4, noise_variance: float = 0.0):
    """Position-only (no direction at all) analogue of
    `_mixture_directional_moments` -- the special case that achieves EXACT
    `w*=weights` recovery (up to `rel_jitter`'s own tiny regularization),
    per this module's mixture-directional docstring's own derivation:
    without a directional kernel factor at all, z_i = (K_pos @ weights)_i
    identically (both z and Kxx use the plain, symmetric k_pos(x_i,x_j)
    structure, nothing asymmetric enters), so w* = K_pos^-1 z = weights
    exactly in the point-splat limit, and closely for real (small) splat
    footprints. See
    tests/gs_experiment/test_quadrature_mixture_risk.py's
    test_position_only_recovers_alpha_compositing_weights_exactly for the
    direct numeric confirmation this claim gets (unlike the directional
    case, which does NOT recover exactly -- see
    tests/gs_experiment/test_gpu_uncertainty_mixture.py's own negative
    result for that case).
    """
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    weights = np.asarray(weights, dtype=float).reshape(-1)
    n, d = positions.shape

    if local_covariances is None:
        local_covariances = np.zeros((n, d, d))
    local_covariances = np.asarray(local_covariances, dtype=float)

    eye_d = np.eye(d)

    z0 = 0.0
    for k in range(n):
        for l in range(n):
            cov_kl = local_covariances[k] + local_covariances[l] + (sigma_rbf**2) * eye_d
            z0 += weights[k] * weights[l] * float(multivariate_normal(mean=positions[l], cov=cov_kl).pdf(positions[k]))

    z = np.zeros(n)
    for i in range(n):
        total = 0.0
        for k in range(n):
            cov_k = local_covariances[k] + (sigma_rbf**2) * eye_d
            total += weights[k] * float(multivariate_normal(mean=positions[k], cov=cov_k).pdf(positions[i]))
        z[i] = total

    pos_kernel = ProductKernel([RBFKernel(sigma_rbf)] * d)
    kxx = pos_kernel.k(positions, positions)
    jitter = rel_jitter * np.mean(np.diag(kxx))
    kxx = kxx + (jitter + noise_variance) * np.eye(n)
    return float(z0), z, kxx


def bayesian_quadrature_rendering_aware_mixture(
    positions, values, weights, sigma_rbf: float, local_covariances=None, rel_jitter: float = 1e-4,
    noise_variance: float = 0.0,
) -> BQResult:
    """Position-only mixture BQ (no directional kernel at all) -- see
    `_mixture_moments`'s docstring for why this is the variant that
    achieves EXACT `w*=weights` recovery, unlike
    `bayesian_quadrature_rendering_aware_mixture_directional`."""
    n = np.atleast_2d(np.asarray(positions, dtype=float)).shape[0]
    if n == 0:
        return BQResult(mean=0.0, variance=0.0)
    z0, z, kxx = _mixture_moments(positions, weights, sigma_rbf, local_covariances, rel_jitter, noise_variance)
    values = np.asarray(values, dtype=float).reshape(-1)
    mean, variance = _posterior_mean_variance(kxx, values, z, z0)
    return BQResult(mean=mean, variance=variance)


def rendering_aware_alternative_weight_risk_mixture(
    positions, values, weights, sigma_rbf: float, w=None, local_covariances=None, rel_jitter: float = 1e-4,
    noise_variance: float = 0.0,
) -> tuple[float, float]:
    """Position-only mixture analogue of
    `rendering_aware_alternative_weight_risk_mixture_directional` -- see
    that function's docstring for the `weights` (defines a_q) vs `w`
    (scored vector, defaults to `weights`) distinction, identical here.
    The point of this specific (position-only) variant: at `w=weights`
    (the real alpha-compositing weights, the default and only way this
    project actually calls it), `risk` is the EXACT BQ variance (not an
    approximation needing R_alpha's bias-correction term), because
    `_mixture_moments` gives exact `w*=weights` recovery -- unlike the
    directional variant, whose risk is a genuinely different, larger
    quantity than its own BQ-optimal variance once kappa>0."""
    positions_arr = np.atleast_2d(np.asarray(positions, dtype=float))
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    w = weights if w is None else np.asarray(w, dtype=float).reshape(-1)
    n = positions_arr.shape[0]

    mean = float(w @ values) if w.shape[0] == values.shape[0] and n > 0 else 0.0
    if n == 0:
        return mean, 0.0

    z0, z, kxx = _mixture_moments(positions, weights, sigma_rbf, local_covariances, rel_jitter, noise_variance)
    risk = float(z0 - 2.0 * (w @ z) + w @ kxx @ w)
    return mean, max(risk, 0.0)
