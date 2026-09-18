"""Tests for the frozen evaluation protocol.

These matter more than usual: every comparative claim the project makes will
be produced by `gs_experiment/evaluation.py`, so a bug there silently
corrupts every result at once. The tests below therefore check the protocol
against constructed cases where the right answer is known analytically -- a
perfectly calibrated predictor, an uninformative one, and one whose ranking
is perfect but whose scale is deliberately wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from gs_experiment.evaluation import (
    format_table,
    Prediction, ause, attainable, binned_calibration, compare, evaluate,
    fit_variance_model, gaussian_nll, object_mask, per_view_spearman, spearman,
)

N_VIEWS = 10
N_PER_VIEW = 3000


def _make(sigma_true, scale=1.0, seed=0, name="m"):
    """A predictor reporting `scale * sigma_true`, with residuals actually
    drawn from N(0, sigma_true^2). scale=1 is perfectly calibrated."""
    rng = np.random.default_rng(seed)
    view_id = np.repeat(np.arange(N_VIEWS), N_PER_VIEW)
    resid = rng.normal(0.0, sigma_true)
    return Prediction(sigma=scale * sigma_true, residual=resid, view_id=view_id,
                      sigma_n=0.01, name=name)


def _sigma_field(seed=0):
    rng = np.random.default_rng(seed)
    return np.exp(rng.normal(np.log(0.02), 1.0, N_VIEWS * N_PER_VIEW))


def test_perfectly_calibrated_predictor_reaches_its_ceiling():
    pred = _make(_sigma_field(), scale=1.0)
    r = evaluate(pred)
    assert 0.9 < r["spearman_frac_attainable"] < 1.1, r["spearman_frac_attainable"]
    # And the ceiling itself must be well below 1: that is the whole point.
    assert r["spearman_ceiling"] < 0.8, r["spearman_ceiling"]


def test_uninformative_predictor_scores_near_zero_and_gains_nothing():
    sigma_true = _sigma_field()
    rng = np.random.default_rng(1)
    flat = Prediction(sigma=np.full_like(sigma_true, sigma_true.mean()),
                      residual=rng.normal(0.0, sigma_true),
                      view_id=np.repeat(np.arange(N_VIEWS), N_PER_VIEW),
                      sigma_n=0.01, name="flat")
    r = evaluate(flat)
    assert abs(r["spearman"]) < 0.05, r["spearman"]
    # A constant sigma cannot beat a constant-variance baseline by anything.
    assert abs(r["gain_over_constant"]) < 0.02, r["gain_over_constant"]


def test_ranking_can_be_perfect_while_the_scale_is_wrong():
    """The case FINDINGS section 9 turned on: mis-scaled by 6x, ranking
    untouched, raw NLL catastrophic, calibration fixes it."""
    sigma_true = _sigma_field()
    pred = _make(sigma_true, scale=6.0, name="misscaled")
    perfect = _make(sigma_true, scale=1.0, name="perfect")
    r_bad, r_ok = evaluate(pred), evaluate(perfect)
    assert abs(r_bad["spearman"] - r_ok["spearman"]) < 0.02      # ranking identical
    assert r_bad["nll_raw"] > r_ok["nll_raw"] + 0.5              # raw NLL much worse
    assert abs(r_bad["nll_anchored"] - r_ok["nll_anchored"]) < 0.05   # calibration repairs it
    assert 5.0 < r_bad["calib_s"] * 6.0 < 7.0 or 0.1 < r_bad["calib_s"] < 0.25


def test_attainable_ceiling_is_below_one_and_rises_with_sigma_spread():
    """Rank metrics are capped by |z| noise; the cap loosens as log-sigma
    spreads out relative to Var(log|z|) = pi^2/8."""
    narrow = np.exp(np.random.default_rng(0).normal(np.log(0.02), 0.2, 20000))
    wide = np.exp(np.random.default_rng(0).normal(np.log(0.02), 2.5, 20000))
    c_narrow, _ = attainable(narrow, spearman)
    c_wide, _ = attainable(wide, spearman)
    assert c_narrow < c_wide < 1.0
    assert c_narrow < 0.4, c_narrow


def test_fit_variance_model_recovers_known_parameters():
    rng = np.random.default_rng(0)
    sigma = np.exp(rng.normal(np.log(0.01), 1.0, 200000))
    s_true, c_true, sigma_n = 3.0, 2.0, 0.01
    var = s_true ** 2 * sigma ** 2 + (c_true * sigma_n) ** 2
    resid = rng.normal(0.0, np.sqrt(var))
    _, p = fit_variance_model(sigma, resid, sigma_n)
    assert abs(p["s"] - s_true) < 0.2, p
    assert abs(p["c"] - c_true) < 0.4, p


def test_calibration_is_fitted_on_disjoint_views():
    """Frozen decision 4. If the split leaked, a predictor with view-specific
    miscalibration would be scored as if it were fine."""
    rng = np.random.default_rng(0)
    view_id = np.repeat(np.arange(N_VIEWS), N_PER_VIEW)
    sigma = np.full(N_VIEWS * N_PER_VIEW, 0.02)
    # Odd views (the SCORED half) have 10x the residual of even views.
    resid = rng.normal(0.0, np.where(view_id % 2 == 1, 0.2, 0.02))
    r = evaluate(Prediction(sigma, resid, view_id, sigma_n=0.01, name="leaky"))
    assert not r["single_view_fallback"]
    # Calibration fitted on the quiet half must UNDERSTATE the noisy half,
    # so NLL cannot reach what an in-sample fit would give.
    _, p_all = fit_variance_model(sigma, resid, 0.01)
    assert r["nll_anchored"] > gaussian_nll(p_all["s"] ** 2 * sigma ** 2
                                            + (p_all["c"] * 0.01) ** 2, resid)


def test_per_view_spearman_picks_up_view_level_structure():
    rng = np.random.default_rng(0)
    view_id = np.repeat(np.arange(N_VIEWS), N_PER_VIEW)
    per_view_scale = np.linspace(0.005, 0.1, N_VIEWS)[view_id]
    pred = Prediction(sigma=per_view_scale, residual=rng.normal(0.0, per_view_scale),
                      view_id=view_id, sigma_n=0.01, name="pv")
    assert per_view_spearman(pred) > 0.9


def test_ause_is_zero_for_the_oracle_and_positive_otherwise():
    rng = np.random.default_rng(0)
    err = np.abs(rng.normal(0, 1, 5000))
    assert abs(ause(err, err)) < 1e-9                       # ranking BY the error itself
    assert ause(rng.permutation(err), err) > 0.05           # a random ranking is worse


def test_object_mask_is_gt_based_and_excludes_white_background():
    gt = np.ones((8, 8, 3), dtype=np.float32)
    gt[2:5, 2:5] = 0.3
    m = object_mask(gt)
    assert m.sum() == 9 and m[3, 3] and not m[0, 0]


def test_compare_scores_methods_on_identical_data():
    sigma_true = _sigma_field()
    rows = compare([_make(sigma_true, 1.0, name="good"),
                    _make(sigma_true, 1.0, seed=5, name="good2")])
    assert len(rows) == 2 and rows[0]["n_samples"] == rows[1]["n_samples"]
    assert all("per_view_spearman" in r for r in rows)


def test_prediction_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        Prediction(np.zeros(3), np.zeros(4), np.zeros(3), sigma_n=0.01)
    with pytest.raises(ValueError):
        Prediction(np.zeros(0), np.zeros(0), np.zeros(0), sigma_n=0.01)


def test_calibration_fit_is_invariant_to_the_methods_raw_units():
    """Fairness: two methods whose sigma differ only by a constant factor are
    the SAME uncertainty and must score identically after calibration. A fit
    initialised at s=1 fails this for methods reported in arbitrary units
    (coverage counts, Fisher traces), silently favouring ones already near
    the residual scale."""
    sigma_true = _sigma_field()
    base = evaluate(_make(sigma_true, scale=1.0, name="unit"))
    for factor in (1e-4, 0.1, 37.0, 1e5):
        r = evaluate(_make(sigma_true, scale=factor, name=f"x{factor}"))
        assert abs(r["nll_anchored"] - base["nll_anchored"]) < 0.02, (factor, r["nll_anchored"])
        assert abs(r["spearman"] - base["spearman"]) < 1e-9
        assert abs(r["calib_s"] * factor - base["calib_s"]) < 0.15 * base["calib_s"]


def test_anticorrelated_predictor_has_no_fraction_of_attainable():
    """A method that points at the WRONG pixels is not "weakly informative":
    a negative fraction-of-attainable is a category error, so it is reported
    as n/a and the signed raw spearman carries the verdict. This case is real
    -- a residual-supervised baseline fitted on training views that exclude a
    coverage gap scores -0.45 inside that gap."""
    sigma_true = _sigma_field()
    rng = np.random.default_rng(3)
    view_id = np.repeat(np.arange(N_VIEWS), N_PER_VIEW)
    flipped = Prediction(sigma=1.0 / sigma_true, residual=rng.normal(0.0, sigma_true),
                         view_id=view_id, sigma_n=0.01, name="anti")
    r = evaluate(flipped)
    assert r["spearman"] < -0.2
    assert not np.isfinite(r["spearman_frac_attainable"])
    assert "n/a" in format_table([{**r, "per_view_spearman": per_view_spearman(flipped)}])
