"""Regression coverage for gs_experiment.ply_io's read/write round trip,
in the same spirit as the colors bug documented in
gs_experiment/results/FINDINGS.md section 4: that bug passed 159 tests
because nothing checked the *semantic validity* of a value crossing a
convention boundary (raw SH coefficient vs. real color), only that the
code ran and returned *a* number. ply_io.py has three such boundaries of
its own (opacity: probability vs. pre-sigmoid logit; scale: real vs. log;
rotation: not-necessarily-unit-norm vs. normalized) -- write_3dgs_ply/
read_3dgs_ply's own docstrings document all three conversions, but until
now nothing exercised them with values that would actually expose a
skipped or doubled transform (identity quaternions round-trip correctly
whether or not normalization is applied at all; opacities/scales well
inside (0, 1)/away from 0 mask a missing clip too).
"""

import numpy as np

from gs_experiment.ply_io import read_3dgs_ply, write_3dgs_ply


def test_round_trip_normalizes_non_unit_quaternions(tmp_path):
    """write_3dgs_ply's docstring says rotations "need not be unit-norm"
    and read_3dgs_ply's says rotations come back "[normalized]" -- but
    every existing round-trip test (test_gs_splat_scene.py) only ever
    writes identity quaternions [1, 0, 0, 0], which are already unit-norm
    and round-trip correctly whether or not normalize-on-read is actually
    wired up. Use real, deliberately non-unit-norm quaternions (as a real
    training loop's raw, unnormalized `quats` parameter would produce) so
    a regression that dropped the `rotations / norm(...)` line in
    read_3dgs_ply would actually fail this test."""
    rng = np.random.default_rng(0)
    n = 16
    positions = rng.uniform(-1.0, 1.0, size=(n, 3))
    scales = rng.uniform(0.01, 0.1, size=(n, 3))
    # deliberately not unit-norm -- norms range roughly 0.3 to 3
    rotations = rng.normal(scale=1.0, size=(n, 4)) + np.array([1.0, 0.0, 0.0, 0.0])
    opacities = rng.uniform(0.1, 0.9, size=n)
    sh_coeffs = rng.normal(scale=0.3, size=(n, 3, 1))

    path = str(tmp_path / "splats.ply")
    write_3dgs_ply(path, positions, scales, rotations, opacities, sh_coeffs, sh_degree=0)
    result = read_3dgs_ply(path)

    norms = np.linalg.norm(result["rotations"], axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    expected_normalized = rotations / np.linalg.norm(rotations, axis=1, keepdims=True)
    np.testing.assert_allclose(result["rotations"], expected_normalized, atol=1e-4)


def test_round_trip_opacities_stay_in_unit_interval_at_the_extremes(tmp_path):
    """opacity is stored as a pre-sigmoid logit (write applies
    inverse_sigmoid, read applies sigmoid) -- values near the (0, 1)
    boundary are exactly where a missing/doubled transform would first
    show up as an out-of-range or badly-off value (e.g. writing the raw
    probability as if it were already a logit would send a near-1 opacity
    through a second sigmoid and collapse it towards ~0.73 instead of
    staying near 1)."""
    n = 5
    positions = np.zeros((n, 3))
    scales = np.full((n, 3), 0.05)
    rotations = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    opacities = np.array([0.001, 0.01, 0.5, 0.99, 0.999])
    sh_coeffs = np.zeros((n, 3, 1))

    path = str(tmp_path / "splats.ply")
    write_3dgs_ply(path, positions, scales, rotations, opacities, sh_coeffs, sh_degree=0)
    result = read_3dgs_ply(path)

    assert np.all(result["opacities"] > 0.0) and np.all(result["opacities"] < 1.0)
    np.testing.assert_allclose(result["opacities"], opacities, atol=1e-3)


def test_round_trip_scales_stay_positive_across_orders_of_magnitude(tmp_path):
    """scale is stored as log(scale) -- a real checkpoint's splats span
    several orders of magnitude (tiny detail splats vs. large background
    ones), so check the round trip holds at both ends, not just a single
    "reasonable" middle value."""
    n = 4
    positions = np.zeros((n, 3))
    scales = np.array([[1e-4, 1e-4, 1e-4], [1e-2, 1e-2, 1e-2], [1.0, 1.0, 1.0], [10.0, 10.0, 10.0]])
    rotations = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    opacities = np.full(n, 0.5)
    sh_coeffs = np.zeros((n, 3, 1))

    path = str(tmp_path / "splats.ply")
    write_3dgs_ply(path, positions, scales, rotations, opacities, sh_coeffs, sh_degree=0)
    result = read_3dgs_ply(path)

    assert np.all(result["scales"] > 0.0)
    np.testing.assert_allclose(result["scales"], scales, rtol=1e-4)
