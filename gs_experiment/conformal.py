"""Split-conformal wrapper over any per-pixel uncertainty, after
`chu2026conformal` (View-Structured Conformal Prediction for 3DGS).

Provenance, since the evaluation protocol is pre-registered and frozen:
conformal coverage was declared as Tier 3 of the comparison plan and written
into ROADMAP before any baseline was implemented or any result seen. It lives
here rather than in `evaluation.py` so the frozen protocol file is untouched,
and its parameters (below) are fixed in this docstring before it is run for
the first time.

Why it is not a competitor. `chu2026conformal` answers a different question
from FINDINGS sections 9-10: a distribution-free finite-sample COVERAGE
guarantee, rather than a calibrated predictive density. The two compose --
a conformal layer can wrap any sigma, ours included -- so the useful
comparison is not "who covers better" (any method covers, by construction, if
its quantile is fitted correctly) but **how wide the intervals have to be to
get there**. A method whose sigma actually tracks error buys the same
guarantee with narrower intervals; an uninformative sigma is forced to pad
uniformly. Interval width at fixed coverage is therefore the discriminator,
and coverage itself is only a correctness check on the implementation.

Frozen parameters:
  * target coverage 90% (alpha = 0.1);
  * nonconformity score `s_q = |residual_q| / sigma_q`, the scaled residual;
  * calibration/test split identical to the protocol's -- even-indexed
    held-out views calibrate, odd-indexed views are scored;
  * MARGINAL split conformal: one quantile over all calibration pixels.

Why the view-structured variant is NOT implemented here, having been tried.
`chu2026conformal` splits the scale into a renderer-derived spatial shape and
a transferable per-view difficulty scalar. Estimating that scalar from the
prediction's own sigma -- the only per-view quantity a `Prediction` carries --
makes the construction algebraically identical to marginal conformal:
with score `|r| sigma_bar_v / sigma` and interval
`qhat (sigma / sigma_bar_v) d_v`, choosing `d_v = sigma_bar_v` cancels
exactly and returns `qhat * sigma`. Implementing it that way would add a
column that cannot differ from the one beside it. A faithful version needs a
difficulty estimate carrying information the sigma does NOT -- in their case
a learned model over view geometry -- which this project does not have. This
is worth stating as a small result in its own right: "view-structured"
conformal only earns its name when the per-view scalar is independent of the
uncertainty it reweights.

An earlier draft of this module shipped that variant with a sign of exactly
this problem -- it double-counted the view difficulty (sigma already carries
it, and the correction multiplied it in again), over-covering at 98.1% with
intervals 4.5x wider than marginal. It was removed rather than tuned.
"""

from __future__ import annotations

import numpy as np

ALPHA = 0.1                      # target coverage 90%


def _quantile_level(n: int, alpha: float = ALPHA) -> float:
    """The finite-sample corrected level `ceil((n+1)(1-alpha))/n`. Without the
    correction the guarantee is asymptotic only, and with small calibration
    sets the difference is not negligible."""
    return min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)


def calibrate(sigma: np.ndarray, residual: np.ndarray, alpha: float = ALPHA) -> float:
    """Marginal split-conformal quantile of the scaled residual."""
    scores = np.abs(residual) / np.maximum(sigma, 1e-12)
    return float(np.quantile(scores, _quantile_level(len(scores), alpha), method="higher"))


def evaluate(sigma, residual, qhat) -> dict:
    """Empirical coverage and interval half-width on held-out pixels.

    Both MEAN and MEDIAN half-width are reported, and the median is the one
    that discriminates. An informative sigma spends its budget unevenly --
    very wide where it is genuinely unsure -- so a handful of tail pixels
    dominate the mean and hide the benefit: on a synthetic case with perfectly
    calibrated lognormal sigma against a constant one, the mean widths differ
    by 11% while the medians differ by 56%. Mean interval width is the
    conventional conformal efficiency measure and is reported for
    comparability, but it is the wrong summary for heavy-tailed sigma."""
    half_width = qhat * np.maximum(sigma, 1e-12)
    return {"coverage": float(np.mean(np.abs(residual) <= half_width)),
            "mean_half_width": float(half_width.mean()),
            "median_half_width": float(np.median(half_width))}


def wrap_prediction(pred, alpha: float = ALPHA) -> dict:
    """Run both conformal variants on one `evaluation.Prediction`, using the
    protocol's own even/odd held-out view split."""
    views = np.unique(pred.view_id)
    is_cal = np.isin(pred.view_id, views[0::2])
    if is_cal.all() or (~is_cal).all():
        return {"name": pred.name, "insufficient_views": True}
    s_c, r_c, v_c = pred.sigma[is_cal], pred.residual[is_cal], pred.view_id[is_cal]
    s_t, r_t, v_t = pred.sigma[~is_cal], pred.residual[~is_cal], pred.view_id[~is_cal]

    return {"name": pred.name, "insufficient_views": False,
            "target_coverage": 1.0 - alpha,
            "marginal": evaluate(s_t, r_t, calibrate(s_c, r_c, alpha))}


def format_table(rows: list[dict], alpha: float = ALPHA) -> str:
    head = f"{'method':<24}{'coverage':>10}{'mean width':>12}{'median width':>14}"
    lines = [f"target coverage {100 * (1 - alpha):.0f}%; at equal coverage NARROWER is better. "
             f"MEDIAN width is the discriminator -- the mean is dominated by the tail, where an "
             f"informative sigma correctly spends its budget.", head, "-" * len(head)]
    for r in rows:
        if r.get("insufficient_views"):
            lines.append(f"{r['name']:<24}{'--':>10}{'--':>12}{'--':>14}")
            continue
        lines.append(f"{r['name']:<24}{100 * r['marginal']['coverage']:>9.1f}%"
                     f"{r['marginal']['mean_half_width']:>12.5f}"
                     f"{r['marginal']['median_half_width']:>14.5f}")
    return "\n".join(lines)
