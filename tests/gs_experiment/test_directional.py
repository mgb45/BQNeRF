import numpy as np

from gs_experiment.kernels import DirectionalKernel


def angle_to_unit_vector(theta):
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    return np.stack([np.cos(theta), np.sin(theta)], axis=1)


def test_directional_kernel_self_similarity_is_one():
    kernel = DirectionalKernel(kappa=3.0)
    for theta in [0.0, 1.0, -2.5, np.pi]:
        w = angle_to_unit_vector(theta)
        assert abs(kernel.k(w, w)[0, 0] - 1.0) < 1e-12


def test_directional_kernel_gram_is_positive_semidefinite():
    rng = np.random.default_rng(0)
    kernel = DirectionalKernel(kappa=2.0)
    thetas = rng.uniform(0, 2 * np.pi, size=15)
    w = angle_to_unit_vector(thetas)
    K = kernel.k(w, w)
    eigvals = np.linalg.eigvalsh(K)
    assert eigvals.min() > -1e-8


def test_directional_kernel_decreases_with_angular_separation():
    kernel = DirectionalKernel(kappa=2.0)
    w0 = angle_to_unit_vector(0.0)
    seps = [0.0, 0.3, 1.0, 2.0, np.pi]
    values = [float(kernel.k(w0, angle_to_unit_vector(s))[0, 0]) for s in seps]
    assert all(v1 >= v2 - 1e-12 for v1, v2 in zip(values, values[1:]))


