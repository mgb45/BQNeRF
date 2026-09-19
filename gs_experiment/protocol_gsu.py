"""The evaluation protocol of `galappaththige2026predictive` (U-3DGS), ported
so our method can be scored on the numbers its authors actually report.

Motivation. `evaluation.py` is our own pre-registered protocol, and it differs
from the field's convention in five specific ways -- it masks to object
pixels, pairs sigma and residual per COLOUR CHANNEL, uses Spearman, pools
pixels across views, and reports rank metrics against an attainable ceiling.
Every one of those is defensible and several were forced by things we
measured (FINDINGS sections 9, 14). But a protocol we invented cannot be
compared against anyone's published table, and "our numbers are good under
our metric" is not a claim a reviewer should accept.

So this module reproduces U-3DGS's `uncertainty_metrics.py` exactly, from
their released code (github.com/Chumsy0725/GS-U), and
`tests/gs_experiment/test_protocol_gsu.py` cross-validates this port against
their actual function source rather than against our reading of it.

Their protocol, as implemented:

* **whole frame**, no object mask;
* error is `|gt - pred|` averaged over RGB to a per-pixel scalar (a DSSIM
  variant is also reported); uncertainty is likewise a per-pixel scalar;
* **Pearson**, not Spearman;
* **AUSE** over `linspace(0, 0.999, 100)`, normalised by the full-set mean
  error, dropping pixels where error or uncertainty is exactly zero;
* metrics computed PER VIEW and then averaged, not pooled over all pixels.

The per-view averaging and the RGB collapse are the two that most change our
numbers, and neither is wrong -- their uncertainty is a single scalar per
pixel, so RGB-averaging the error is the only consistent choice available to
them. It is simply a different question from the one our protocol asks.
"""

from __future__ import annotations

import numpy as np


def gsu_ause(error: np.ndarray, uncertainty: np.ndarray) -> float:
    """Faithful port of U-3DGS's `ause_torch`. Lower is better.

    Both inputs are per-pixel scalars for ONE view. Pixels where either is
    exactly zero are dropped, as in the original -- on a white-background
    synthetic scene that silently removes much of the background, which is
    worth knowing when reading the number.
    """
    err = np.asarray(error, dtype=np.float64).reshape(-1)
    unc = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    mask = (err != 0.0) & (unc != 0.0)
    if not np.any(mask):
        return 0.0
    err, unc = err[mask], unc[mask]

    # float32 deliberately: the original builds this with torch.linspace,
    # which defaults to float32, and the integer truncation of (1-r)*n below
    # lands on different indices in float64. That mattered -- it moved AUSE by
    # ~6e-6, and was only caught by cross-validating against their code.
    #
    # One difference remains and is left in place on purpose. torch.linspace
    # and np.linspace do not agree bit-for-bit in float32: they differ at 24 of
    # these 100 grid points, by at most 6e-8. Everything downstream is then
    # bit-identical (the sparsification curves match to 0.0), and the residual
    # effect on AUSE is ~1e-8 -- far below any reported precision. Replicating
    # torch's internal linspace rounding would make this port fragile to a
    # torch upgrade, which is the opposite of what a protocol port is for, so
    # the cross-validation test holds 1e-6 rather than bit-equality.
    ratio_removed = np.linspace(0, 0.999, 100, dtype=np.float32)
    n = len(err)
    ratio_idx = ((1.0 - ratio_removed) * n).astype(np.float32).astype(int)[:-1]
    # The original can produce index 0 here and then reads [-1], silently
    # wrapping to the largest slice. That only happens for tiny images; we
    # clamp instead, which is the one place this port deviates.
    ratio_idx = np.maximum(ratio_idx, 1)

    err_sorted = np.sort(err)
    err_slices = np.cumsum(err_sorted)[ratio_idx - 1] / ratio_idx

    order_by_unc = np.argsort(unc, kind="stable")
    err_by_unc = err[order_by_unc]
    err_by_var_slices = np.cumsum(err_by_unc)[ratio_idx - 1] / ratio_idx

    start = err_slices[0]
    if start == 0:
        return 0.0
    return float(np.trapezoid((err_by_var_slices - err_slices) / start,
                              ratio_removed[:len(err_slices)]))


def gsu_pearson(error: np.ndarray, uncertainty: np.ndarray) -> float:
    """Pearson correlation over the whole frame, NaN-filtered as they do."""
    err = np.asarray(error, dtype=np.float64).reshape(-1)
    unc = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    keep = err == err
    err, unc = err[keep], unc[keep]
    if len(err) < 2 or np.std(err) == 0 or np.std(unc) == 0:
        return float("nan")
    return float(np.corrcoef(unc, err)[0, 1])


def rgb_l1_error(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Their error map: absolute difference averaged over colour channels."""
    return np.abs(np.asarray(gt) - np.asarray(pred)).mean(axis=-1)


def score_views(per_view: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """Score a sequence of (error_map, uncertainty_map) pairs the way they do:
    per view, then averaged -- NOT pooled over all pixels. Pooling and
    averaging are genuinely different estimators once views differ in
    difficulty, and theirs is the one their table reports."""
    aus = [gsu_ause(e, u) for e, u in per_view]
    pea = [gsu_pearson(e, u) for e, u in per_view]
    pea_ok = [p for p in pea if p == p]
    return {"AUSE": float(np.mean(aus)), "AUSE_std": float(np.std(aus)),
            "pearson": float(np.mean(pea_ok)) if pea_ok else float("nan"),
            "pearson_std": float(np.std(pea_ok)) if pea_ok else float("nan"),
            "n_views": len(per_view)}
