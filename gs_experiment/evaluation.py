"""FROZEN evaluation protocol for uncertainty methods. Pre-registered.

This module is committed BEFORE any baseline is implemented, deliberately.
The project has already been burned once by scoring decisions made after
seeing results: FINDINGS section 9 records that the headline per-pixel number
moved from "0.27, modest, do not claim it" to "98% of attainable" purely
because of how it was scored (RGB-averaging, and an unreachable ceiling of
1.0). A metric discovered to be wrong after the fact is recoverable; a metric
*chosen* after the fact is not. So the scoring is fixed here first, and every
method -- ours and every baseline -- is scored by these functions and no
others.

THE FROZEN DECISIONS
--------------------
1. **Per channel, never RGB-averaged.** Predicted sigma and residual are
   paired per (pixel, colour channel). Averaging either over channels before
   comparing discards the pairing and depresses every rank metric.

2. **Object pixels are the headline; whole-frame is reported alongside and is
   NOT a calibration number.** On NeRF-Synthetic a whole-frame correlation is
   ~0.95 for any method that knows where the object is, because both error
   and uncertainty vanish on the white background. `zhao2026posterior` and
   OUGS both report the same hazard from opposite directions. Whole-frame is
   reported only for comparability with published numbers, always labelled.

3. **Rank metrics are reported as a fraction of what is ATTAINABLE.** With a
   perfectly calibrated sigma the observable is still one draw,
   `|eps| = sigma|z|`, `z ~ N(0,1)`; `Var(log|z|) = pi^2/8 ~ 1.23` destroys
   rank information on its own. The ceiling is estimated by simulating
   `eps* ~ N(0, sigma^2)` and re-running the identical metric. AUSE likewise
   gets its attainable FLOOR, since its usual oracle (sort by true error) is
   equally unreachable.

4. **NLL is only quoted after a calibration fitted on DISJOINT views.** Views
   with even index fit, odd index score. Raw uncalibrated NLL is reported too,
   because it is honest about how far off the raw scale is.

5. **The default variance model is the anchored one**, `sigma_total^2 =
   s^2 sigma_pred^2 + (c sigma_n)^2`, with `sigma_n` the scene's own TRAINING
   residual. FINDINGS section 10: transferring a raw `sigma_0` across scenes
   is worse than useless, anchoring it recovers 66% of the gain with no
   held-out views. A method that needs held-out views to calibrate must say so.

6. **Every method is compared against a constant-variance baseline.** "Beats
   a constant" is the minimum bar for claiming an uncertainty is informative,
   and it is not implied by a good correlation.

7. **Cost is part of the result.** Wall-clock seconds and whether the method
   requires retraining are recorded next to every accuracy number.

8. **Anything stochastic gets >= 3 seeds and an error bar.** A discarded NBV
   run had a random arm go 11.56 -> 10.84 -> 10.90 dB: single curves are noise.

9. **Both regimes and both capacities.** Saturated (full training views) and
   epistemic (deliberate angular gap / sparse views); config-A and `wide`
   capacity. FINDINGS section 8 showed methods are indistinguishable in the
   saturated regime, so a single-regime table would be uninformative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.optimize import minimize

OBJECT_MASK_THRESHOLD = 0.99   # a pixel is "object" if any GT channel is below this
N_CEILING_SIMS = 16
N_CALIBRATION_BINS = 15


def object_mask(gt: np.ndarray) -> np.ndarray:
    """(H, W) bool. Frozen definition: a pixel is object if its ground-truth
    colour departs from the white background in any channel. Uses GT, never
    the render, so it is identical for every method compared."""
    return gt.min(axis=2) < OBJECT_MASK_THRESHOLD


@dataclass
class Prediction:
    """One method's output on one checkpoint, pooled over held-out views.

    `sigma`, `residual`: (n,) paired per (pixel, channel) -- see frozen
    decision 1. `view_id`: (n,) which held-out view each sample came from,
    required for the disjoint-view calibration split of decision 4.
    `sigma_n`: the scene's own training residual, for the anchored model of
    decision 5. `seconds` / `requires_retraining`: decision 7.
    """

    sigma: np.ndarray
    residual: np.ndarray
    view_id: np.ndarray
    sigma_n: float
    seconds: float = float("nan")
    requires_retraining: bool = False
    name: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.sigma)
        if not (len(self.residual) == len(self.view_id) == n):
            raise ValueError("sigma, residual and view_id must be the same length")
        if n == 0:
            raise ValueError("empty prediction")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation via rank-transform + Pearson. Ties are broken by
    argsort order; with float sigma and residuals exact ties are negligible."""
    if len(a) < 2:
        return float("nan")
    return float(np.corrcoef(a.argsort().argsort(), b.argsort().argsort())[0, 1])


def ause(uncertainty: np.ndarray, error: np.ndarray, n_steps: int = 100) -> float:
    """Area under the sparsification-error curve: remove the most-uncertain
    pixels progressively, track the mean error of what remains, against the
    oracle ordering by true error. Lower is better. Normalised by the oracle's
    own starting mean error so it is comparable across scenes."""
    n = len(error)
    fracs = np.linspace(0.0, 0.95, n_steps)
    by_unc, by_err = np.argsort(-uncertainty), np.argsort(-error)
    curve_u = np.array([error[by_unc[int(f * n):]].mean() for f in fracs])
    curve_o = np.array([error[by_err[int(f * n):]].mean() for f in fracs])
    denom = curve_o[0] if curve_o[0] > 0 else 1.0
    return float(np.trapezoid((curve_u - curve_o) / denom, fracs))


def attainable(sigma: np.ndarray, metric: Callable[[np.ndarray, np.ndarray], float],
               n_sims: int = N_CEILING_SIMS, seed: int = 0) -> tuple[float, float]:
    """The value `metric` would take if `sigma` were EXACTLY right -- frozen
    decision 3. Simulates `eps* ~ N(0, sigma^2)` and re-runs the same metric.
    Returns (mean, sd) over `n_sims` simulations."""
    rng = np.random.default_rng(seed)
    safe = np.maximum(sigma, 1e-12)
    vals = [metric(sigma, np.abs(rng.normal(0.0, safe))) for _ in range(n_sims)]
    return float(np.mean(vals)), float(np.std(vals))


def binned_calibration(sigma: np.ndarray, residual: np.ndarray,
                       n_bins: int = N_CALIBRATION_BINS) -> tuple[np.ndarray, np.ndarray]:
    """Equal-count bins by predicted sigma; returns (mean predicted sigma,
    observed RMS residual) per bin. Averaging within a bin removes the |z|
    nuisance that caps rank metrics, so this -- not a rank metric -- is what
    shows whether the MAGNITUDES are right."""
    order = np.argsort(sigma)
    bins = np.array_split(order, n_bins)
    return (np.array([sigma[b].mean() for b in bins]),
            np.array([np.sqrt((residual[b] ** 2).mean()) for b in bins]))


def gaussian_nll(variance: np.ndarray, residual: np.ndarray) -> float:
    v = np.maximum(variance, 1e-30)
    return float(np.mean(0.5 * np.log(2 * np.pi * v) + residual ** 2 / (2 * v)))


def fit_variance_model(sigma: np.ndarray, residual: np.ndarray,
                       sigma_n: Optional[float] = None):
    """Fit `sigma_total^2 = s^2 sigma^2 + floor^2` by Gaussian NLL, with all
    coefficients positive by construction.

    `sigma_n` given -> the ANCHORED model of frozen decision 5: the floor is
    `c * sigma_n`, so only the two dimensionless numbers (s, c) are fitted and
    they are the ones that transfer across scenes. `sigma_n=None` -> a free
    constant floor, which does NOT transfer and is reported only as the
    per-scene upper reference.

    Returns (variance_fn, params) where `params` is a dict of s and c/floor.
    """
    anchored = sigma_n is not None
    scale = float(sigma_n) if anchored else 1.0

    def variance(p, sig):
        return np.exp(p[0]) * sig ** 2 + (np.exp(p[1]) * scale) ** 2

    # Scale-free initialisation, by moments. If var = s^2 sigma^2 + floor^2
    # then mean(resid^2) = s^2 mean(sigma^2) + floor^2, so splitting the
    # observed second moment evenly between the two terms gives a starting
    # point that does not depend on the method's raw units. This matters for
    # FAIRNESS, not just convergence: starting every fit at s=1 would give a
    # better-converged optimum to methods whose raw sigma happens to be near
    # the residual scale, and penalise baselines reported in arbitrary units
    # (coverage counts, Fisher traces) that are equally informative.
    m2 = max(float(np.mean(residual ** 2)), 1e-24)
    s_guess = np.sqrt(0.5 * m2 / max(float(np.mean(sigma ** 2)), 1e-24))
    floor_guess = np.sqrt(0.5 * m2) / max(scale, 1e-12)
    x0 = [2.0 * np.log(max(s_guess, 1e-12)), np.log(max(floor_guess, 1e-12))]
    r = minimize(lambda p: gaussian_nll(variance(p, sigma), residual), x0=x0,
                 method="Nelder-Mead",
                 options={"maxiter": 20000, "maxfev": 20000, "fatol": 1e-12, "xatol": 1e-10})
    params = {"s": float(np.sqrt(np.exp(r.x[0])))}
    params["c" if anchored else "floor"] = float(np.exp(r.x[1]) * (1.0 if anchored else scale))
    return (lambda sig: variance(r.x, sig)), params


def evaluate(pred: Prediction, seed: int = 0) -> dict:
    """Score one Prediction by the whole frozen protocol. Calibration is
    fitted on even-indexed held-out views and every number below is reported
    on the odd-indexed ones (frozen decision 4)."""
    views = np.unique(pred.view_id)
    fit_views = set(views[0::2].tolist())
    is_fit = np.isin(pred.view_id, list(fit_views))
    if is_fit.all() or (~is_fit).all():          # single view: score on itself, flagged
        is_fit = np.zeros_like(is_fit, dtype=bool)
        single_view = True
    else:
        single_view = False

    s_fit, r_fit = pred.sigma[is_fit], pred.residual[is_fit]
    s_te, r_te = pred.sigma[~is_fit], pred.residual[~is_fit]
    if single_view:
        s_fit, r_fit = pred.sigma, pred.residual
    err_te = np.abs(r_te)

    sp = spearman(s_te, err_te)
    sp_ceil, sp_sd = attainable(s_te, spearman, seed=seed)
    au = ause(s_te, err_te)
    au_floor, au_sd = attainable(s_te, ause, seed=seed)

    f_anch, p_anch = fit_variance_model(s_fit, r_fit, pred.sigma_n)
    f_free, p_free = fit_variance_model(s_fit, r_fit, None)
    nll_const = gaussian_nll(np.full_like(r_te, np.var(r_fit)), r_te)

    pred_bin, obs_bin = binned_calibration(s_te, r_te)
    return {
        "name": pred.name,
        "n_samples": int(len(s_te)),
        "single_view_fallback": single_view,
        "spearman": sp,
        "spearman_ceiling": sp_ceil,
        "spearman_ceiling_sd": sp_sd,
        "spearman_frac_attainable": sp / sp_ceil if sp_ceil else float("nan"),
        "ause": au,
        "ause_floor": au_floor,
        "ause_floor_sd": au_sd,
        "nll_raw": gaussian_nll(s_te ** 2, r_te),
        "nll_constant": nll_const,
        "nll_anchored": gaussian_nll(f_anch(s_te), r_te),
        "nll_free_floor": gaussian_nll(f_free(s_te), r_te),
        "gain_over_constant": nll_const - gaussian_nll(f_anch(s_te), r_te),
        "calib_s": p_anch["s"],
        "calib_c": p_anch["c"],
        "binned_predicted": pred_bin.tolist(),
        "binned_observed": obs_bin.tolist(),
        "seconds": pred.seconds,
        "requires_retraining": pred.requires_retraining,
        "log_sigma_spread": float(np.std(np.log(np.maximum(s_te, 1e-12)))),
    }


def per_view_spearman(pred: Prediction) -> float:
    """Rank correlation between each view's MEAN predicted sigma and its mean
    absolute residual -- the aggregate next-best-view selection consumes
    (FINDINGS section 8)."""
    views = np.unique(pred.view_id)
    m_sig = np.array([pred.sigma[pred.view_id == v].mean() for v in views])
    m_err = np.array([np.abs(pred.residual[pred.view_id == v]).mean() for v in views])
    return spearman(m_sig, m_err)


def compare(preds: list[Prediction], seed: int = 0) -> list[dict]:
    """Score several methods on identical data. Every method MUST be built
    from the same checkpoints and the same held-out views, or this is
    comparing training noise (frozen decision: identical checkpoints)."""
    rows = [evaluate(p, seed=seed) for p in preds]
    for row, p in zip(rows, preds):
        row["per_view_spearman"] = per_view_spearman(p)
    return rows


def format_table(rows: list[dict]) -> str:
    head = (f"{'method':<24}{'sp':>7}{'ceil':>7}{'%att':>7}{'AUSE':>8}{'floor':>8}"
            f"{'pvSp':>7}{'NLL':>9}{'vsConst':>9}{'s':>7}{'c':>7}{'secs':>8}{'retrain':>9}")
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['name']:<24}{r['spearman']:>7.3f}{r['spearman_ceiling']:>7.3f}"
            f"{100 * r['spearman_frac_attainable']:>6.0f}%{r['ause']:>8.3f}{r['ause_floor']:>8.3f}"
            f"{r['per_view_spearman']:>7.3f}{r['nll_anchored']:>9.3f}"
            f"{r['gain_over_constant']:>+9.3f}{r['calib_s']:>7.2f}{r['calib_c']:>7.2f}"
            f"{r['seconds']:>8.1f}{'yes' if r['requires_retraining'] else 'no':>9}")
    return "\n".join(lines)
