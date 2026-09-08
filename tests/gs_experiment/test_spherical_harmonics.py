import numpy as np

from gs_experiment.spherical_harmonics import SH_C0, SH_C1, SH_C2, eval_sh, random_sh_coeffs


def test_sh_c0_matches_closed_form_normalization():
    """Y_0^0 = 1/sqrt(4*pi) is a standard, derivable normalization
    constant (so that its squared integral over the sphere is 1) -- check
    the hardcoded SH_C0 against it exactly, rather than just trusting the
    literal constant."""
    assert abs(SH_C0 - 1.0 / np.sqrt(4 * np.pi)) < 1e-15


def test_sh_c1_matches_closed_form_normalization():
    """Y_1^0 = sqrt(3/(4*pi)) * z."""
    assert abs(SH_C1 - np.sqrt(3.0 / (4 * np.pi))) < 1e-15


def test_sh_c2_matches_closed_form_normalizations():
    # |Y_2^{-2}| coefficient = (1/2)*sqrt(15/pi); |Y_2^0| coefficient = (1/4)*sqrt(5/pi);
    # |Y_2^2| coefficient = (1/4)*sqrt(15/pi).
    assert abs(abs(SH_C2[0]) - 0.5 * np.sqrt(15.0 / np.pi)) < 1e-15
    assert abs(abs(SH_C2[2]) - 0.25 * np.sqrt(5.0 / np.pi)) < 1e-15
    assert abs(abs(SH_C2[4]) - 0.25 * np.sqrt(15.0 / np.pi)) < 1e-15


def test_degree_0_is_view_independent():
    rng = np.random.default_rng(0)
    coeffs = random_sh_coeffs(rng, n_splats=5, degree=3)
    dirs_a = rng.normal(size=(5, 3))
    dirs_a /= np.linalg.norm(dirs_a, axis=1, keepdims=True)
    dirs_b = rng.normal(size=(5, 3))
    dirs_b /= np.linalg.norm(dirs_b, axis=1, keepdims=True)

    color_a = eval_sh(coeffs, dirs_a, degree=0)
    color_b = eval_sh(coeffs, dirs_b, degree=0)
    np.testing.assert_allclose(color_a, color_b)

    expected = SH_C0 * coeffs[..., 0] + 0.5
    np.testing.assert_allclose(color_a, expected)


def test_degree_0_and_degree_3_agree_when_higher_terms_are_zero():
    rng = np.random.default_rng(1)
    coeffs = random_sh_coeffs(rng, n_splats=5, degree=3)
    coeffs[..., 1:] = 0.0  # zero out every view-dependent term
    directions = rng.normal(size=(5, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    color_deg0 = eval_sh(coeffs, directions, degree=0)
    color_deg3 = eval_sh(coeffs, directions, degree=3)
    np.testing.assert_allclose(color_deg0, color_deg3)


def test_eval_sh_matches_numerical_directional_average_for_degree_0():
    """The mean of eval_sh over many random directions should converge to
    the (view-independent) degree-0 prediction, since higher-degree terms
    are zero-mean basis functions over the sphere."""
    rng = np.random.default_rng(2)
    coeffs = random_sh_coeffs(rng, n_splats=1, degree=2)[0]  # (n_channels, n_coeffs)

    n_samples = 200_000
    directions = rng.normal(size=(n_samples, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    colors = eval_sh(coeffs[None, :, :], directions, degree=2)

    expected_mean = SH_C0 * coeffs[:, 0] + 0.5
    empirical_mean = colors.mean(axis=0)
    np.testing.assert_allclose(empirical_mean, expected_mean, atol=0.01)


def test_higher_degree_adds_genuine_view_dependence():
    rng = np.random.default_rng(3)
    coeffs = random_sh_coeffs(rng, n_splats=3, degree=3, scale=1.0)
    dirs = rng.normal(size=(3, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    color_deg0 = eval_sh(coeffs, dirs, degree=0)
    color_deg3 = eval_sh(coeffs, dirs, degree=3)
    assert not np.allclose(color_deg0, color_deg3)


def test_raises_on_invalid_degree_or_insufficient_coefficients():
    rng = np.random.default_rng(4)
    coeffs = random_sh_coeffs(rng, n_splats=2, degree=0)
    d = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    try:
        eval_sh(coeffs, d, degree=1)
        assert False, "expected ValueError for insufficient coefficients"
    except ValueError:
        pass
    try:
        eval_sh(coeffs, d, degree=4)
        assert False, "expected ValueError for invalid degree"
    except ValueError:
        pass
