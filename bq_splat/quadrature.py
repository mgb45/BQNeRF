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
exactly this trap. See bq_splat/reference.py and bq_splat/toy_scene.py for
how nodes/values are generated to avoid it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import integrate
from scipy.stats import multivariate_normal

from bq_splat.kernels import DirectionalKernel, Kernel, ProductKernel, RBFKernel


@dataclass
class BQResult:
    mean: float
    variance: float


def bayesian_quadrature_directional(
    positions,
    directions,
    values,
    pos_kernel,
    dir_kernel: DirectionalKernel,
    pos_bounds,
    query_direction,
    rel_jitter: float = 1e-4,
    precomputed_pos_vv: float | None = None,
) -> BQResult:
    """BQ over a joint (position, direction) domain, where position is
    integrated over (as in bayesian_quadrature_nd -- a genuine quadrature
    integral, e.g. a pixel footprint) but direction is evaluated at one
    query direction, not integrated -- a rendered pixel looks in one
    specific outgoing direction, it doesn't average over a range of them.
    See DirectionalKernel's docstring for why this asymmetry is the correct
    generalization, not an approximation of a "properly" symmetric one.

    `positions`/`directions`/`values` are parallel arrays: `values[i]` is an
    observation at `positions[i]` from `directions[i]` (e.g. one training
    view's contribution to one splat). `pos_kernel` must use the
    ProductKernel-style interface -- `v(x, bounds)`/`vv(bounds)` with
    `bounds` a list of (a, b) pairs, even in 1D (wrap a plain Kernel like
    `ProductKernel([RBFKernel(sigma)])`) -- rather than plain Kernel's
    `v(x, a, b)`/`vv(a, b)`, so callers don't need to special-case
    dimensionality. `dir_kernel` is a DirectionalKernel (no v/vv, evaluated
    pointwise).

    K_ij = pos_kernel.k(x_i, x_j) * dir_kernel.k(w_i, w_j)
    v_i   = pos_kernel.v(x_i, pos_bounds) * dir_kernel.k(w_i, w_query)
    vv    = pos_kernel.vv(pos_bounds) * dir_kernel.k(w_query, w_query)
          = pos_kernel.vv(pos_bounds)               (self-similarity is 1)

    `precomputed_pos_vv`: pass `pos_kernel.vv(pos_bounds)` in directly if
    the caller already has it cached -- exact, not approximate, for a
    stationary kernel evaluated on a fixed-size, translated window (see
    bq_splat/results/FINDINGS.md section 8; gs_experiment/pixel_uncertainty.py
    is the first real caller of this).
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim == 1:
        positions = positions.reshape(-1, 1)
    directions = np.asarray(directions, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)
    n = positions.shape[0]

    vv = float(precomputed_pos_vv) if precomputed_pos_vv is not None else float(pos_kernel.vv(pos_bounds))
    if n == 0:
        return BQResult(mean=0.0, variance=vv)

    kxx = pos_kernel.k(positions, positions) * dir_kernel.k(directions, directions)
    jitter = rel_jitter * np.mean(np.diag(kxx))
    kxx = kxx + jitter * np.eye(n)

    v_pos = np.asarray(pos_kernel.v(positions, pos_bounds)).reshape(-1)
    v_dir = dir_kernel.k(directions, query_direction).reshape(-1)
    v = v_pos * v_dir

    solved = np.linalg.solve(kxx, values)
    mean = float(v @ solved)

    solved_v = np.linalg.solve(kxx, v)
    variance = float(vv - v @ solved_v)

    return BQResult(mean=mean, variance=max(variance, 0.0))


def directional_posterior_variance(
    directions, values, dir_kernel: DirectionalKernel, query_direction, rel_jitter: float = 1e-4
) -> BQResult:
    """Pure-directional special case of bayesian_quadrature_directional,
    for a single fixed spatial location (position integration dropped
    entirely rather than degenerated into it) -- standard GP regression
    posterior mean/variance at one query direction, given observations from
    other directions. Used to validate DirectionalKernel's behavior in
    isolation, without conflating it with the position-integration
    machinery bayesian_quadrature_directional also does.
    """
    directions = np.asarray(directions, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)
    n = directions.shape[0]

    prior_variance = 1.0  # dir_kernel.k(q, q) == 1 always
    if n == 0:
        return BQResult(mean=0.0, variance=prior_variance)

    kxx = dir_kernel.k(directions, directions)
    jitter = rel_jitter * np.mean(np.diag(kxx))
    kxx = kxx + jitter * np.eye(n)

    k_query = dir_kernel.k(directions, query_direction).reshape(-1)

    solved = np.linalg.solve(kxx, values)
    mean = float(k_query @ solved)

    solved_k = np.linalg.solve(kxx, k_query)
    variance = float(prior_variance - k_query @ solved_k)

    return BQResult(mean=mean, variance=max(variance, 0.0))


def bayesian_quadrature_nd(
    nodes, values, kernel: ProductKernel, bounds, rel_jitter: float = 1e-4, precomputed_vv: float | None = None
) -> BQResult:
    """Same as `bayesian_quadrature`, generalized to a D-dimensional domain
    via a `ProductKernel` and a per-axis `bounds` list of (a_d, b_d) pairs.
    Kept as a separate function (rather than folding the 1D case into this
    one) so the already-tested 1D `bayesian_quadrature` path is untouched.

    `precomputed_vv`: pass `kernel.vv(bounds)` in directly if the caller
    already has it cached -- exact, not approximate, for a stationary
    kernel evaluated on a fixed-size, translated window (see
    bq_splat/results/FINDINGS.md section 8).
    """
    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim == 1:
        nodes = nodes.reshape(-1, 1)
    values = np.asarray(values, dtype=float).reshape(-1)
    n = nodes.shape[0]
    vv = float(precomputed_vv) if precomputed_vv is not None else float(kernel.vv(bounds))
    if n == 0:
        return BQResult(mean=0.0, variance=vv)

    kxx = kernel.k(nodes, nodes)
    jitter = rel_jitter * np.mean(np.diag(kxx))
    kxx = kxx + jitter * np.eye(n)
    v = kernel.v(nodes, bounds).reshape(-1)

    solved = np.linalg.solve(kxx, values)
    mean = float(v @ solved)

    solved_v = np.linalg.solve(kxx, v)
    variance = float(vv - v @ solved_v)

    return BQResult(mean=mean, variance=max(variance, 0.0))


def bayesian_quadrature(nodes, values, kernel: Kernel, a: float, b: float, rel_jitter: float = 1e-4) -> BQResult:
    """Posterior mean/variance of integral_a^b g(t) dt given g(nodes) = values.

    `rel_jitter` scales the Gram matrix diagonal (jitter = rel_jitter *
    mean(diag(K))) rather than adding a fixed absolute constant. Node
    placements here can be irregular enough to produce near-duplicate nodes
    (unlike the original repo's regularly-spaced ray samples), which drives
    the Gram matrix condition number past 1e18 with a fixed-scale jitter of
    1e-8 -- confirmed empirically for n=40 random uniform nodes at sigma=0.35.
    Splats can be similarly near-collocated in a real GS scene, so this is a
    real stability requirement, not just a toy-script wrinkle.
    """
    nodes = np.asarray(nodes, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float).reshape(-1)
    n = nodes.shape[0]
    if n == 0:
        return BQResult(mean=0.0, variance=float(kernel.vv(a, b)))

    kxx = kernel.k(nodes.reshape(-1, 1), nodes.reshape(1, -1))
    jitter = rel_jitter * np.mean(np.diag(kxx))
    kxx = kxx + jitter * np.eye(n)
    v = kernel.v(nodes, a, b).reshape(-1)

    solved = np.linalg.solve(kxx, values)
    mean = float(v @ solved)

    solved_v = np.linalg.solve(kxx, v)
    variance = float(kernel.vv(a, b) - v @ solved_v)

    return BQResult(mean=mean, variance=max(variance, 0.0))


# ---------------------------------------------------------------------------
# Rendering-aware Bayesian quadrature: k_q(xi, xi') = a_q(xi) k_base(xi, xi')
# a_q(xi'), where a_q = T_q * sigma * G_q is a renderer/query-specific weight
# (bq_splat/render_weight.py) and k_base is a prior over the *radiance field*
# c(xi), not over the rendering integrand a_q * c directly (the distinction
# that fixes what bayesian_quadrature_nd's uniform-box v/vv gets wrong -- see
# bq_splat/PROOF_alpha_compositing_equivalence.md section 7 and
# bq_splat/README.md).
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
# (tests/test_render_weight.py) -- the same "closed form where cheap,
# numerically cross-checked" discipline bq_splat/kernels.py already follows.
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
    numerical_rendering_moment_vector in tests/test_render_weight.py.
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
    in tests/test_render_weight.py.
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
    ground-truth cross-check for the closed form (tests/test_render_weight.py).
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
    cross-check for rendering_aware_prior_variance in tests/test_render_weight.py
    for D=1 (a tractable 2D nquad integral).

    Scales badly past D=1: this integrates over `2*D` dimensions total (D
    for xi, D for xi'), and nquad's nested adaptive quadrature does not
    finish in any reasonable time once that reaches 4D (confirmed: D=2 does
    not complete in 100s even over a tight, few-sigma domain) -- an
    importance-sampled Monte Carlo estimate (sample xi, xi' directly from
    a_q's own Gaussian shape, average k_base(xi, xi')) is the practical
    fallback for D>=2, used in tests/test_render_weight.py's D=2 cross-check
    instead of this function.
    """
    d = render_weight.dim

    def integrand(*args):
        xi_row = np.array(args[:d]).reshape(1, -1)
        xip_row = np.array(args[d:]).reshape(1, -1)
        return float(render_weight(xi_row)[0] * render_weight(xip_row)[0] * base_kernel.k(xi_row, xip_row)[0, 0])

    val, _ = integrate.nquad(integrand, list(domain) + list(domain))
    return val


def _rendering_aware_moments(nodes, render_weight, sigma_rbf, base_kernel, domain, mode, rel_jitter):
    """Shared plumbing for bayesian_quadrature_rendering_aware and
    renderer_centered_residual_variance: builds K (from k_base alone --
    nodes are noiseless observations of the radiance field c, not of
    a_q*c) and the renderer-aware z/z0 moments, via either the closed-form
    Gaussian path (`mode="closed_form"`) or the numerical fallback
    (`mode="numerical"`). Returns (base_kernel_used, kxx_or_None, z, z0);
    kxx is None when there are zero nodes (nothing to solve against -- the
    posterior collapses to the prior, variance = z0)."""
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
    kxx = kxx + jitter * np.eye(n)
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

    This mean need not reproduce standard alpha compositing -- see
    renderer_centered_residual_variance for the alternative formulation that
    keeps alpha compositing as the mean and uses only this same variance.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    _, kxx, z, z0 = _rendering_aware_moments(nodes, render_weight, sigma_rbf, base_kernel, domain, mode, rel_jitter)
    if kxx is None:
        return BQResult(mean=0.0, variance=max(z0, 0.0))

    solved = np.linalg.solve(kxx, values)
    mean = float(z @ solved)

    solved_z = np.linalg.solve(kxx, z)
    variance = float(z0 - z @ solved_z)

    return BQResult(mean=mean, variance=max(variance, 0.0))


def renderer_centered_residual_variance(
    nodes,
    render_weight,
    sigma_rbf: float | None = None,
    base_kernel: ProductKernel | None = None,
    domain=None,
    rel_jitter: float = 1e-4,
    mode: str = "closed_form",
) -> float:
    """Formulation 2 ("renderer-centred probabilistic quadrature"): keeps
    ordinary alpha compositing (e.g. bq_splat.reference.riemann_estimate) as
    the predictive mean, and uses only this module's rendering-aware
    variance to model the unresolved integration error around it --

        C_q = C_hat_q^3DGS + eps_q,     eps_q ~ N(0, z_{q,0} - z_q^T K^-1 z_q)

    Returns just that variance (the caller supplies and owns the mean).
    Shares its z/z0/K machinery exactly with bayesian_quadrature_rendering_aware
    (same _rendering_aware_moments call) -- the two formulations differ only
    in which mean the variance is paired with, per the prompt's own "there
    are two possible formulations" framing; neither is picked as uniquely
    correct here.
    """
    _, kxx, z, z0 = _rendering_aware_moments(nodes, render_weight, sigma_rbf, base_kernel, domain, mode, rel_jitter)
    if kxx is None:
        return max(z0, 0.0)

    solved_z = np.linalg.solve(kxx, z)
    variance = float(z0 - z @ solved_z)
    return max(variance, 0.0)


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
) -> BQResult:
    """The full rendering-aware construction the original prompt specified
    and this module only partially implemented until now: a JOINT
    position+direction base kernel, `k_base(xi, xi') = k_pos(x, x') *
    k_dir(d, d')`, so that `k_q(xi, xi') = a_q(x) k_base(xi, xi') a_q(x')`
    carries both the renderer-specific spatial envelope
    (`bayesian_quadrature_rendering_aware`'s `render_weight`) *and* the
    directional/epistemic term (`bayesian_quadrature_directional`'s
    `dir_kernel`) at once, instead of the two living as separate,
    unconnected code paths. `a_q` itself is still position-only (T_q sigma
    G_q are about *where* along a ray/footprint mass concentrates, not
    which direction a splat happens to have been observed from) -- the
    directional dependence enters purely through `k_dir`, exactly as the
    prompt's `k_dir(d,d') = exp[kappa(d^T d' - 1)]` factor does.

    Mirrors bayesian_quadrature_directional's own generalization of
    bayesian_quadrature_nd (see that function's docstring): position is
    integrated over via `render_weight`'s spatial envelope, direction is
    evaluated at one `query_direction`, not integrated -- a rendered pixel
    looks in one specific outgoing direction. Because
    `dir_kernel.k(d, d) == 1` always (DirectionalKernel's docstring),
    `z_{q,0}` is *exactly* the same spatial-only prior variance
    `rendering_aware_prior_variance`/`bayesian_quadrature_rendering_aware`
    already compute -- only `K` and the moment vector `z_q` pick up a
    directional factor:

        K_ij     = k_pos(x_i, x_j) * k_dir(d_i, d_j)
        z_{q,i}  = [integral a_q(xi) k_pos(xi, x_i) dxi] * k_dir(d_i, d_query)
        z_{q,0}  = integral integral a_q(xi) a_q(xi') k_pos(xi, xi') dxi dxi'

    `positions`/`directions`/`values` are parallel arrays, one row per
    (splat, observing-direction) pair -- the same input shape
    bayesian_quadrature_directional expects (see that function's
    docstring for why: a splat needs to be observed from *multiple*
    directions during training for this term to carry any signal at all;
    one row per splat with a single direction each cannot).
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
    kxx = kxx + jitter * np.eye(n)

    v_dir = dir_kernel.k(directions, query_direction).reshape(-1)
    z = z_pos * v_dir

    solved = np.linalg.solve(kxx, values)
    mean = float(z @ solved)

    solved_z = np.linalg.solve(kxx, z)
    variance = float(z0 - z @ solved_z)

    return BQResult(mean=mean, variance=max(variance, 0.0))
