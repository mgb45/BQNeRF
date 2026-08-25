# Formal statement and proof: BQ posterior mean recovers alpha compositing, with a provable error bound

ROADMAP.md item 1. This makes precise, and proves, the claim that has so
far only been checked empirically (`bq_splat/results/FINDINGS.md` §1-2):
that Bayesian-quadrature (BQ) posterior mean, applied to the volume
rendering integral, recovers standard alpha compositing, and that the BQ
posterior *variance* is not merely a heuristic that happens to correlate
with error, but a provable, non-heuristic bound on it.

The two theorems below are proven separately and then combined:

- **Theorem A** shows discrete alpha compositing is not an approximation of
  the continuous rendering integral at all — under the standard piecewise-
  constant density/color model every NeRF- and 3DGS-style renderer already
  assumes, it is the *exact* value of that integral. This defines precisely
  what "the right answer" is, before asking whether BQ finds it.
- **Theorem B** is the classical result underlying all of Bayesian
  quadrature (O'Hagan 1991, "Bayes-Hermite quadrature"; see Briol, Oates,
  Girolami, Osborne & Sejdinovic, 2019, "Probabilistic Integration: A Role
  in Statistical Computation?", *Statistical Science*, for a modern
  treatment) — not new to this project — restated here in this project's
  exact notation and connected directly to the `v`/`vv`/`K` quantities
  `bq_splat/quadrature.py` actually computes.

Combining them gives the Corollary this item asks for: BQ's posterior mean
equals the alpha-compositing value up to an error that BQ's own reported
variance provably bounds.

## 1. Setup and notation

Along a ray/pixel, parametrize distance by `t in [a, b]` (`near`/`far`).
Let `sigma(t) >= 0` be density, `c(t)` be emitted color, and

```
T(t) = exp(-integral_a^t sigma(s) ds)          (transmittance)
g(t) = T(t) sigma(t) c(t)                       (the rendering integrand)
C    = integral_a^b g(t) dt                     (the rendered value)
```

`C = integral g(t) dt` is a genuine integral of a well-defined function —
this is the quantity BQ is asked to estimate from point evaluations
`g(t_1), ..., g(t_n)`. (This is also precisely why `bq_splat/quadrature.py`
insists observations be point evaluations of `g`, not pre-integrated
`weight_i * color_i` terms that already bake in a bin width — see that
module's docstring and the "double quad" bug it references.)

## 2. Theorem A: alpha compositing is the *exact* value of `C`, not an approximation of it

**Claim.** Partition `[a, b]` into bins `[t_i, t_{i+1})`, widths
`delta_i = t_{i+1} - t_i`. If `sigma` and `c` are each constant within every
bin (`sigma(t) = sigma_i`, `c(t) = c_i` for `t` in bin `i` — the standard
NeRF/3DGS discretization assumption), then

```
C = sum_i w_i c_i,      w_i = alpha_i * T_i,      alpha_i = 1 - exp(-sigma_i delta_i)
```

where `T_i = prod_{j<i} (1 - alpha_j)` — exactly the standard alpha
compositing formula (and exactly what `cumprod_exclusive` computes in
`models/nerf.py` and the `else` branch of `raw2outputs`, `models/nerf.py:325-332`).

**Proof.** `T(t_1) = 1` by definition. Within bin `i`, for
`t in [t_i, t_{i+1})`, constant `sigma_i` gives
`T(t) = T(t_i) exp(-sigma_i (t - t_i))`. So bin `i`'s contribution to `C` is

```
integral_{t_i}^{t_{i+1}} T(t) sigma_i c_i dt
  = c_i T(t_i) integral_0^{delta_i} sigma_i exp(-sigma_i s) ds        (s = t - t_i)
  = c_i T(t_i) [1 - exp(-sigma_i delta_i)]
  = c_i T(t_i) alpha_i
```

and `T(t_{i+1}) = T(t_i) exp(-sigma_i delta_i) = T(t_i)(1 - alpha_i)`, so by
induction `T(t_i) = prod_{j<i}(1 - alpha_j) = T_i`. Summing bin
contributions gives `C = sum_i alpha_i T_i c_i = sum_i w_i c_i`. QED.

No approximation was made beyond the piecewise-constant assumption on
`sigma` and `c` — which is the assumption every discrete NeRF/3DGS
renderer already makes to go from a continuous scene to a finite set of
samples/splats in the first place. **Alpha compositing is the exact
integral of the discretized model, not a numerical approximation of it.**
This is a standard result (Max, 1995, "Optical Models for Direct Volume
Rendering"; the derivation used by the original NeRF paper), stated here
precisely because everything downstream depends on knowing exactly what
"the right answer" `C` is.

Verified numerically to floating-point precision (`2.2e-16` max absolute
error over 20 random piecewise-constant scenes) in
`scripts/validate_alpha_compositing_equivalence.py`, `check_theorem_a` —
see §8 below for the run.

## 3. Theorem B: BQ mean is RKHS-optimal, with a provable error bound

This section restates the standard theory of kernel/Bayesian quadrature —
not a new result — connected explicitly to this project's code.

Let `H_k` be the reproducing kernel Hilbert space (RKHS) of a kernel `k` on
`[a, b]`: for every `h in H_k` and `x in [a,b]`, `k(., x) in H_k` and
`<h, k(., x)>_{H_k} = h(x)` (the reproducing property). Given nodes
`x_1, ..., x_n`, consider *any* linear estimator of the integral
`I[g] = integral_a^b g(t) dt` of the form `Q_w[g] = sum_i w_i g(x_i)`, and
define its worst-case error over the RKHS unit ball:

```
e(w) = sup_{g in H_k, ||g||_{H_k} <= 1} |I[g] - Q_w[g]|
```

**Claim.** The functional `g -> I[g] - Q_w[g]` is a bounded linear
functional on `H_k`, so by the Riesz representation theorem there is a
unique `r_w in H_k` with `I[g] - Q_w[g] = <g, r_w>_{H_k}` for all
`g in H_k`; by Cauchy-Schwarz, `e(w) = ||r_w||_{H_k}`, with equality at
`g = r_w / ||r_w||_{H_k}`. The weights minimizing `e(w)` are exactly

```
w* = K^{-1} v,     K_ij = k(x_i, x_j),     v_i = integral_a^b k(t, x_i) dt
```

— exactly the weights implicit in `bayesian_quadrature`'s
`mean = v @ solve(K, values)` (`bq_splat/quadrature.py`) — and the
minimized worst-case error is

```
e(w*)^2 = integral integral k(s,t) ds dt  -  v^T K^{-1} v  =  vv - v @ solve(K, v)
```

which is exactly `bayesian_quadrature`'s reported `variance`. This is the
classical Bayes-Hermite quadrature result (O'Hagan, 1991; see Briol et al.,
2019, for the general modern treatment) — the derivation is standard, what
matters here is that it maps *exactly* onto this project's `v`, `vv`, `K`,
and `variance` quantities with no reinterpretation needed.

**Corollary (the bound).** For any `g in H_k`,

```
| I[g] - BQ_mean[g] |  <=  ||g||_{H_k} * sqrt(BQ_variance)
```

i.e. BQ's own reported posterior variance is a provable — not merely
empirically-correlated — upper bound on its own mean's error, up to a
fixed multiplicative constant reflecting how "rough" the true integrand
`g` is relative to the kernel's smoothness assumption.

Verified numerically in `scripts/validate_alpha_compositing_equivalence.py`,
`check_theorem_b`: the bound was never violated across 40 random test
functions per kernel (RBF and Matern-3/2), and — more importantly, since
"never violated" alone doesn't rule out the bound being vacuously loose —
the ratio `error / bound` was driven to `~0.999` for a test function
constructed to approximate the error representer `r_{w*}` itself, showing
the bound is not just valid but *achievable*: it is the tightest possible
bound of this form, not an arbitrarily conservative one. See §8.

## 4. Corollary: BQ recovers the alpha-compositing value with a provable error bound

Combining Theorem A and Theorem B: under the piecewise-constant
density/color model, if the rendering integrand `g = T sigma c` (as
realized on `[a,b]` by the scene's actual, possibly-irregular sigma/c) has
finite `H_k`-norm, then

```
| alpha_compositing_value - BQ_mean |  <=  ||g||_{H_k} * sqrt(BQ_variance)
```

This is the precise, defensible version of "BQ recovers alpha
compositing": not literal equality at finite sample size (that would need
either infinitely many nodes or a degenerate/interpolating kernel), but
**exact recovery in the noiseless-observation limit, and a provable,
non-heuristic error bound at any finite sample size** — the bound's own
right-hand side is computed from quantities BQ already reports.

## 5. The same theorem is what produces the directional/epistemic term too

This is the theoretical anchor for the project's "unification" thesis
(ROADMAP.md's central claim), not a separate argument bolted on.

Theorem B is a statement about *any* bounded linear functional `L` on a
kernel's RKHS, not specifically about integration:

```
e(w)^2 = L_s L_t [k(s,t)]  -  v_L^T K^{-1} v_L,      v_L,i = L_t[k(t, x_i)]
```

`bq_splat`'s `ProductKernel` builds an RKHS that is the tensor product of a
spatial RKHS `H_{k_pos}` and a directional RKHS `H_{k_dir}` (a standard
fact about product kernels). Three different choices of `L` on this one
product RKHS, applied to the *same* observed data, give the three
quantities this codebase computes:

| `L` | quantity | code |
|---|---|---|
| integrate over position, direction irrelevant | quadrature/discretization uncertainty | `bayesian_quadrature_nd` |
| evaluate at one position, one query direction (no integration) | directional/epistemic uncertainty | `directional_posterior_variance` |
| integrate over position **at** a fixed query direction | the joint quantity actually rendered per pixel | `bayesian_quadrature_directional` |

All three variances are the same `e(w)^2 = L L[K] - v_L^T K^{-1} v_L`
worst-case-error formula from Theorem B, just with `L` chosen differently —
confirmed directly in code: `bayesian_quadrature_directional`'s `vv` is
`pos_kernel.vv(bounds) * dir_kernel.k(w_query, w_query)`, which is exactly
`L_s L_t[k]` for the joint functional, and reduces to the pure spatial
`vv` because `dir_kernel.k(w, w) == 1` always (`kernels.py`,
`DirectionalKernel` docstring). **Quadrature uncertainty and directional
uncertainty are not two mechanisms — they are the same worst-case-error
theorem, applied to the same posterior, with two different (or combined)
choices of what's being asked about it.** This is the rigorous form of the
claim in `bq_splat/results/FINDINGS.md` §9 that the directional extension
is "the same formalism... not a different mechanism bolted on."

## 6. When the bound is loose: discontinuities, and what the numbers actually show

`g = T sigma c` is continuous in `T` (an integral, hence continuous) but,
under the piecewise-constant-color model, `c(t)` can jump arbitrarily
between adjacent bins (nothing forces adjacent splats/samples to share a
color) — so `g` itself can have genuine jump discontinuities at bin
boundaries. A function with a literal jump is not in the classical RBF
RKHS (RBF's RKHS is Gaussian-tailed in the Fourier domain, incompatible
with a jump's `1/omega` tail) — **and, worth being precise about, it is
also not in Matern-3/2's RKHS either** (a Sobolev space that, in 1D, still
requires continuity by the Sobolev embedding theorem). So the bound in §4
is, strictly, vacuous near a true discontinuity for *both* kernel families
in this project — this is not a clean RBF-specific failure mode, and the
proof should not claim it is.

What's left is an empirical question: at a fixed, finite bandwidth, and
node spacing that doesn't exactly resolve the jump, how does each kernel's
*practical* error and reported variance behave near a discontinuity?
`check_discontinuity` in `scripts/validate_alpha_compositing_equivalence.py`
checks this directly on a single-jump step scene, RBF (`sigma=0.6`) vs.
Matern-3/2 (`rho=0.6`), at matched nominal bandwidth parameter, node counts
10/20/40/80. **The result is the opposite of the naive hypothesis**: RBF
had *lower* error and *lower* reported variance than Matern at every node
count tested (e.g. at `n=40`: RBF error `0.0041`, variance `0.00012`;
Matern error `0.0411`, variance `0.0379` — full table in §8). This directly
contradicts an initial hypothesis that RBF's infinite smoothness should
make it the *worse*-behaved kernel near a jump.

Recorded honestly rather than reframed to fit the hypothesis: the most
likely explanation is that `sigma=0.6` and `rho=0.6` are not a matched
comparison of *effective correlation length* — RBF's and Matern-3/2's
kernels decay to half their peak value at different multiples of their
respective parameters, so this result may be an artifact of comparing
unequal effective lengthscales rather than a real smoothness effect. This
doesn't invalidate Theorem A/B (which hold regardless of kernel choice or
node placement, and were separately verified above) — it means the more
specific claim "RBF is provably worse-behaved near discontinuities than
Matern" is **not yet established**, and needs a proper lengthscale-matched
comparison (in the spirit of the "kernel choice" ablations already planned
in `bq_splat/results/FINDINGS.md` §5-7 and ROADMAP.md item 6) before being
used in any paper claim. Flagged as an open follow-up, not resolved here.

## 7. Scope note: what this proves vs. what `gs_experiment` currently computes

Theorem A is fundamentally a statement about the **ray-depth domain**: it
relies on `T(t)`'s sequential, path-ordered structure along a single ray,
which doesn't generalize verbatim to "integrate over an arbitrary
multi-dimensional domain." Theorem B, by contrast, is a generic fact about
*any* kernel-quadrature estimator regardless of domain dimensionality, and
applies as-is to `bayesian_quadrature_nd` and `bayesian_quadrature_directional`
too.

`gs_experiment/pixel_uncertainty.py`'s `LocalUncertaintyEngine`, which
every real-data result in `gs_experiment/results/FINDINGS.md` is built on,
computes BQ variance over a **3D spatial window around a query point**,
using observed splat colors as node values — not literally `bayesian_quadrature`
applied along the ray-depth domain with `g = T sigma c` as derived above.
It's a reasonable and closely related construction (Theorem B's bound
applies to it unchanged, since it's still a kernel-quadrature estimator on
some RKHS), but connecting it formally back to Theorem A's exact-recovery
statement — i.e. proving the *specific* quantity `LocalUncertaintyEngine`
reports is provably close to the *specific* per-pixel alpha-compositing
value a renderer would produce, not just "some quadrature-type variance
computed nearby" — is **not done in this document** and is flagged as a
concrete next step, not assumed.

## 8. Numerical verification: how to reproduce, and the results referenced above

```
python3 scripts/validate_alpha_compositing_equivalence.py
```

```
=== 1. Theorem A: exact reduction to alpha compositing ===
max |alpha_compositing - true_integral| over 20 random piecewise-constant scenes (6 bins each): 2.220e-16

=== 2. Theorem B: RKHS worst-case-error bound ===
[RBF (sigma=0.6)]
  BQ variance at these nodes: 0.50759
  bound violated: 0/40 random test functions
  error/bound ratio over random test functions: min=0.0031 mean=0.1310 max=0.4807
  error/bound ratio for the representer-fitting test function, as the fitting grid densifies (10/30/80 points): ['0.4391', '0.9990', '0.9994']

[Matern-3/2 (rho=0.6)]
  BQ variance at these nodes: 2.54531
  bound violated: 0/40 random test functions
  error/bound ratio over random test functions: min=0.0009 mean=0.1726 max=0.5107
  error/bound ratio for the representer-fitting test function, as the fitting grid densifies (10/30/80 points): ['0.7475', '0.9732', '0.9979']

=== 3. Discontinuity sensitivity: RBF vs. Matern near a genuine color jump ===
true integral of the step scene: 6.00000
   n   rbf_error     rbf_var   matern_error   matern_var
  10     1.56220     2.43246        2.05121      2.24858
  20     0.03791     0.02125        0.41147      0.43361
  40     0.00414     0.00012        0.04111      0.03786
  80     0.00006     0.00000        0.00379      0.00257
```

## 9. What this settles, and what's still open

**Settled, rigorously:**
- Alpha compositing is the exact rendering integral under the standard
  piecewise-constant model, not an approximation of it (Theorem A).
- BQ posterior mean is the provably worst-case-optimal linear estimator of
  that same integral, and its posterior variance is a provable — not
  heuristic — upper bound on its own error, tight in the achievable sense
  (Theorem B, verified quantitatively, not just checked for violations).
- Quadrature uncertainty and directional/epistemic uncertainty are
  provably the same underlying worst-case-error theorem applied to
  different linear functionals on one product-kernel RKHS posterior — the
  precise form of the "unification" claim (§5).

**Still open:**
- Whether RBF or Matern degrades more gracefully near real discontinuities
  is genuinely unresolved by this check — the naive hypothesis was wrong,
  and a lengthscale-matched follow-up is needed before claiming either
  direction (§6).
- The formal connection between this document's ray-depth-domain theorems
  and `gs_experiment`'s actual production code path (`LocalUncertaintyEngine`'s
  spatial-window formulation) is not yet made (§7).
