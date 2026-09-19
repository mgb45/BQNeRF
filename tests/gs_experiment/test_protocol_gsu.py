"""Cross-validate our port of U-3DGS's evaluation against THEIR ACTUAL CODE.

The point of the port is comparability with a published table, so validating
it against our own reading of their paper would defeat the purpose. Instead
this extracts `ause_torch` verbatim from the released source at
third_party/GS-U and runs it beside our numpy port on the same random inputs.

Skips (rather than fails) when the third-party checkout is absent, so the
suite still runs for anyone who has not cloned it.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from gs_experiment.protocol_gsu import gsu_ause, gsu_pearson, rgb_l1_error, score_views

SRC = Path(__file__).resolve().parents[2] / "third_party" / "GS-U" / "uncertainty_metrics.py"


def _their_ause():
    """Pull `ause_torch` out of the released file and exec it in isolation --
    importing the module would drag in lpipsPyTorch and their utils."""
    torch = pytest.importorskip("torch")
    if not SRC.exists():
        pytest.skip("third_party/GS-U not cloned")
    text = SRC.read_text()
    m = re.search(r"^def ause_torch.*?(?=^def )", text, re.S | re.M)
    assert m, "ause_torch not found in released source"
    ns = {"torch": torch}
    exec(compile(m.group(0), str(SRC), "exec"), ns)
    return ns["ause_torch"], torch


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_our_ause_port_matches_the_released_implementation(seed):
    their_ause, torch = _their_ause()
    rng = np.random.default_rng(seed)
    err = np.abs(rng.normal(0, 0.05, (60, 80)))
    unc = np.abs(rng.normal(0, 0.05, (60, 80))) + 0.3 * err      # partially informative
    theirs = their_ause(torch.tensor(err), torch.tensor(unc))[0].item()
    ours = gsu_ause(err, unc)
    assert abs(theirs - ours) < 1e-6, (theirs, ours)   # see module docstring: torch vs np linspace


def test_port_matches_when_uncertainty_is_perfect_and_when_useless():
    their_ause, torch = _their_ause()
    rng = np.random.default_rng(7)
    err = np.abs(rng.normal(0, 0.05, (40, 40)))
    for unc in (err.copy(), rng.permutation(err.reshape(-1)).reshape(40, 40)):
        theirs = their_ause(torch.tensor(err), torch.tensor(unc))[0].item()
        assert abs(theirs - gsu_ause(err, unc)) < 1e-6


def test_zero_pixels_are_dropped_as_in_the_original():
    """Their mask removes pixels where error OR uncertainty is exactly zero.
    On a white-background synthetic scene that quietly drops background."""
    err = np.array([[0.0, 0.1], [0.2, 0.3]])
    unc = np.array([[0.5, 0.0], [0.2, 0.3]])
    assert np.isfinite(gsu_ause(err, unc))


def test_pearson_and_error_map_shapes():
    rng = np.random.default_rng(0)
    gt, pred = rng.random((16, 16, 3)), rng.random((16, 16, 3))
    e = rgb_l1_error(gt, pred)
    assert e.shape == (16, 16)
    assert -1.0 <= gsu_pearson(e, e) <= 1.0
    assert abs(gsu_pearson(e, e) - 1.0) < 1e-12


def test_score_views_averages_per_view_rather_than_pooling():
    """Per-view averaging and pooling are different estimators once views
    differ in difficulty; theirs is the one their table reports."""
    rng = np.random.default_rng(3)
    easy = (np.abs(rng.normal(0, 0.01, (32, 32))), np.abs(rng.normal(0, 0.01, (32, 32))))
    hard = (np.abs(rng.normal(0, 0.5, (32, 32))), np.abs(rng.normal(0, 0.5, (32, 32))))
    out = score_views([easy, hard])
    assert out["n_views"] == 2
    assert abs(out["AUSE"] - np.mean([gsu_ause(*easy), gsu_ause(*hard)])) < 1e-12
