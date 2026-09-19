"""Tests for the split-conformal wrapper.

Conformal coverage is a guarantee, so the tests check the guarantee holds and
that the DISCRIMINATOR behaves: at equal coverage, a sigma that tracks error
must buy narrower intervals than one that does not.
"""

from __future__ import annotations

import numpy as np

from gs_experiment.conformal import ALPHA, calibrate, evaluate, wrap_prediction
from gs_experiment.evaluation import Prediction

N_VIEWS, N_PER_VIEW = 10, 4000


def _pred(sigma, resid, name="m"):
    return Prediction(sigma=sigma, residual=resid,
                      view_id=np.repeat(np.arange(N_VIEWS), N_PER_VIEW),
                      sigma_n=0.01, name=name)


def test_coverage_hits_the_target_for_a_calibrated_predictor():
    rng = np.random.default_rng(0)
    sigma = np.exp(rng.normal(np.log(0.02), 1.0, N_VIEWS * N_PER_VIEW))
    r = wrap_prediction(_pred(sigma, rng.normal(0.0, sigma)))
    assert abs(r["marginal"]["coverage"] - (1 - ALPHA)) < 0.02


def test_coverage_holds_even_for_a_badly_scaled_sigma():
    """Conformal is scale-free by construction: a sigma 100x too small still
    covers, because the quantile absorbs the scale. This is why coverage
    alone cannot rank methods."""
    rng = np.random.default_rng(1)
    sigma = np.exp(rng.normal(np.log(0.02), 1.0, N_VIEWS * N_PER_VIEW))
    r = wrap_prediction(_pred(sigma / 100.0, rng.normal(0.0, sigma)))
    assert abs(r["marginal"]["coverage"] - (1 - ALPHA)) < 0.02


def test_an_informative_sigma_buys_narrower_intervals_at_equal_coverage():
    """The actual discriminator."""
    rng = np.random.default_rng(2)
    sigma = np.exp(rng.normal(np.log(0.02), 1.2, N_VIEWS * N_PER_VIEW))
    resid = rng.normal(0.0, sigma)
    good = wrap_prediction(_pred(sigma, resid, "good"))
    flat = wrap_prediction(_pred(np.full_like(sigma, sigma.mean()), resid, "flat"))
    assert abs(good["marginal"]["coverage"] - flat["marginal"]["coverage"]) < 0.03
    # MEDIAN width is the discriminator. The mean is dominated by the tail,
    # where an informative sigma correctly spends its budget, so it barely
    # separates the two (0.89 here) while the median separates them clearly.
    assert good["marginal"]["median_half_width"] < 0.6 * flat["marginal"]["median_half_width"]
    assert good["marginal"]["mean_half_width"] < flat["marginal"]["mean_half_width"]


def test_marginal_conformal_handles_view_varying_difficulty():
    """When sigma already tracks per-view difficulty, marginal conformal
    covers correctly without any view-structured correction -- which is the
    algebraic reason that variant is not implemented (see module docstring)."""
    rng = np.random.default_rng(3)
    view_id = np.repeat(np.arange(N_VIEWS), N_PER_VIEW)
    scale = np.linspace(0.005, 0.08, N_VIEWS)[view_id]
    sigma = scale * np.exp(rng.normal(0, 0.3, len(view_id)))
    resid = rng.normal(0.0, scale)
    r = wrap_prediction(_pred(sigma, resid, "pv"))
    assert abs(r["marginal"]["coverage"] - (1 - ALPHA)) < 0.04


def test_finite_sample_quantile_is_conservative():
    rng = np.random.default_rng(4)
    s = np.abs(rng.normal(0, 1, 50))
    q = calibrate(np.ones(50), s)
    assert (np.abs(s) <= q).mean() >= 1 - ALPHA
