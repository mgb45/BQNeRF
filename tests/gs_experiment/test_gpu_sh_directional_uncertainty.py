"""GPU-only tests for gpu_sh_directional_uncertainty.py -- the real
per-(splat, training-camera) alpha-compositing weight beta_{p,i}, and the
accumulated SH-coefficient precision it feeds. Skipped automatically (not
failed) when torch/CUDA aren't available, matching every other GPU test
file's own convention.

Core claim tested: numerical agreement with the already-validated scalar
building blocks (`visibility_attribution.CameraSplatIndex.query` +
`ray_transmittance_weights`), not just "runs and returns something
plausible".
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("gpu_sh_directional_uncertainty needs a real GPU", allow_module_level=True)

from gs_experiment.camera import CameraPose, directions_from_positions_to_camera  # noqa: E402
from gs_experiment.gpu_sh_directional_uncertainty import (  # noqa: E402
    accumulate_sh_precision,
    compute_own_alpha_weight_batched,
    compute_sh_directional_uncertainty_batched,
)
from gs_experiment.pixel_uncertainty import LocalUncertaintyEngine, make_default_3d_position_kernel  # noqa: E402
from gs_experiment.sh_directional_uncertainty import (  # noqa: E402
    directional_variance,
    invert_precision,
    per_splat_coefficient_precision,
    sh_basis,
)
from gs_experiment.splat_scene import SplatScene  # noqa: E402
from gs_experiment.visibility_attribution import CameraSplatIndex, ray_transmittance_weights  # noqa: E402

ANGULAR_TOL = 0.2
SIGMA = 1.0


def _toy_scene_and_cameras(n_splats=30, n_cameras=4, seed=0):
    rng = np.random.default_rng(seed)
    positions = np.stack(
        [rng.uniform(4.0, 6.0, n_splats), rng.uniform(-1.5, 1.5, n_splats), rng.uniform(-1.5, 1.5, n_splats)], axis=1
    )
    opacities = rng.uniform(0.1, 0.9, n_splats)
    cameras = []
    for i in range(n_cameras):
        angle = 2 * np.pi * i / n_cameras
        center = np.array([-10.0 * np.cos(angle), -10.0 * np.sin(angle), 0.5 * i])
        forward = -center / np.linalg.norm(center)
        up = np.array([0.0, 0.0, 1.0])
        cameras.append(CameraPose(center=center, forward=forward, up=up))
    return positions, opacities, cameras


def scalar_own_alpha_weight(positions, opacities, camera, splat_idx, angular_tol=ANGULAR_TOL, max_candidates=500):
    """Independent scalar reference: build a position-only CameraSplatIndex
    over the whole scene, query at the splat's own bearing, compute real
    transmittance weights over that candidate set, and pick out the
    query splat's own slot."""
    from gs_experiment.visibility_attribution import project_to_camera_local

    camera_index = CameraSplatIndex.build(positions, camera)
    out = np.zeros(len(splat_idx))
    for row, s in enumerate(splat_idx):
        bx, by, depth = project_to_camera_local(positions[s : s + 1], camera)
        if not np.isfinite(bx[0]) or depth[0] <= 0:
            continue
        reference_bearing = (float(bx[0]), float(by[0]))
        idx = camera_index.query(reference_bearing, angular_tol, max_candidates=max_candidates)
        weights = ray_transmittance_weights(positions[idx], opacities[idx], camera, reference_bearing, angular_tol)
        own = np.where(idx == s)[0]
        if own.size > 0:
            out[row] = weights[own[0]]
    return out


def test_batched_own_alpha_weight_matches_scalar_reference():
    positions, opacities, cameras = _toy_scene_and_cameras()
    camera = cameras[0]
    splat_idx = np.arange(0, 30, 2)  # every other splat, as if observed by this camera

    scalar = scalar_own_alpha_weight(positions, opacities, camera, splat_idx)
    batched = compute_own_alpha_weight_batched(
        positions, opacities, camera, splat_idx, angular_tol=ANGULAR_TOL, max_candidates=500, device="cuda",
    )
    assert np.allclose(batched, scalar, rtol=1e-6, atol=1e-8)


def test_batched_own_alpha_weight_matches_scalar_with_candidate_cap():
    positions, opacities, cameras = _toy_scene_and_cameras(n_splats=50, seed=3)
    camera = cameras[1]
    splat_idx = np.arange(50)

    scalar = scalar_own_alpha_weight(positions, opacities, camera, splat_idx, max_candidates=6)
    batched = compute_own_alpha_weight_batched(
        positions, opacities, camera, splat_idx, angular_tol=ANGULAR_TOL, max_candidates=6, device="cuda",
    )
    assert np.allclose(batched, scalar, rtol=1e-6, atol=1e-8)


def test_batched_own_alpha_weight_empty_camera_index():
    positions, opacities, cameras = _toy_scene_and_cameras()
    far_camera = CameraPose(center=np.array([1000.0, 1000.0, 1000.0]), forward=np.array([0.0, 0.0, 1.0]), up=np.array([0.0, 1.0, 0.0]))
    out = compute_own_alpha_weight_batched(
        positions, opacities, far_camera, np.array([0, 1, 2]), angular_tol=ANGULAR_TOL, max_candidates=500, device="cuda",
    )
    assert np.allclose(out, 0.0)


def test_accumulate_sh_precision_matches_manual_scalar_accumulation():
    """Independent, fully-manual scalar accumulation: for every (splat,
    camera) pair in a small toy scene's own observed_camera_idx, compute
    beta via ray_transmittance_weights directly and accumulate
    lam*I + sum beta^2 phi phi^T by hand -- cross-validates
    accumulate_sh_precision's own batched-per-camera accumulation."""
    positions, opacities, cameras = _toy_scene_and_cameras(n_splats=20, n_cameras=3, seed=7)
    n_splats = positions.shape[0]
    degree = 2
    lam = 0.7

    rng = np.random.default_rng(9)
    observed_camera_idx = [np.array(sorted(rng.choice(3, size=rng.integers(1, 4), replace=False))) for _ in range(n_splats)]

    scene = SplatScene(
        positions=positions,
        colors=rng.uniform(0.0, 1.0, n_splats),
        opacities=opacities,
        scales=np.tile([0.05, 0.05, 0.05], (n_splats, 1)),
        rotations=np.tile([1.0, 0.0, 0.0, 0.0], (n_splats, 1)),
        observed_camera_idx=observed_camera_idx,
        cameras=cameras,
    )

    batched_precision = accumulate_sh_precision(scene, degree=degree, lam=lam, angular_tol=ANGULAR_TOL, device="cuda")

    from gs_experiment.spherical_harmonics import N_COEFFS_FOR_DEGREE

    n_coeffs = N_COEFFS_FOR_DEGREE[degree]
    design_matrices = [[] for _ in range(n_splats)]
    obs_weights = [[] for _ in range(n_splats)]
    for s in range(n_splats):
        for c in observed_camera_idx[s]:
            camera = cameras[c]
            beta = scalar_own_alpha_weight(positions, opacities, camera, [s])[0]
            if beta <= 0:
                continue
            direction = directions_from_positions_to_camera(positions[s : s + 1], camera)[0]
            phi = sh_basis(direction, degree)
            design_matrices[s].append(phi)
            obs_weights[s].append(beta)

    design_matrices = [np.array(d) if len(d) > 0 else np.zeros((0, n_coeffs)) for d in design_matrices]
    obs_weights = [np.array(w) for w in obs_weights]
    expected_precision = per_splat_coefficient_precision(design_matrices, obs_weights, lam=lam, n_coeffs=n_coeffs)

    assert np.allclose(batched_precision, expected_precision, rtol=1e-6, atol=1e-8)


def test_query_side_u_sh_matches_scalar_reference():
    """u_SH(q) = sum_i beta_{q,i}^2 * s_i^2(d_q), cross-validated against a
    scalar reference built directly from `LocalUncertaintyEngine.
    _along_ray_local_data` (real idx/weights) + `directional_variance`."""
    rng = np.random.default_rng(5)
    n_splats = 25
    degree = 2
    n_coeffs = 9
    positions = np.stack(
        [rng.uniform(4.0, 6.0, n_splats), rng.uniform(-1.5, 1.5, n_splats), rng.uniform(-1.5, 1.5, n_splats)], axis=1
    )
    values = rng.uniform(0.0, 10.0, n_splats)
    opacities = rng.uniform(0.1, 0.9, n_splats)
    camera = CameraPose(center=np.array([-10.0, 0.0, 0.0]), forward=np.array([1.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))
    pos_kernel = make_default_3d_position_kernel(sigma=SIGMA)
    engine = LocalUncertaintyEngine(
        positions=positions, values=values, pos_kernel=pos_kernel,
        scene_bounds=((-1.0, 10.0), (-5.0, 5.0), (-5.0, 5.0)), opacities=opacities,
    )
    camera_index = engine.build_bearing_index(camera)

    sigma_theta = rng.normal(size=(n_splats, n_coeffs, n_coeffs))
    sigma_theta = np.einsum("nij,nkj->nik", sigma_theta, sigma_theta)  # PSD

    n_queries = 10
    points = np.stack(
        [rng.uniform(4.5, 5.5, n_queries), rng.uniform(-1.0, 1.0, n_queries), rng.uniform(-1.0, 1.0, n_queries)], axis=1
    )
    directions = rng.normal(size=(n_queries, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    batched = compute_sh_directional_uncertainty_batched(
        engine, camera_index, points, directions, sigma_theta, degree, angular_tol=ANGULAR_TOL,
        max_candidates=100, device="cuda",
    )

    expected = np.zeros(n_queries)
    for i in range(n_queries):
        idx, _local_positions, _local_values, _render_weight, weights = engine._along_ray_local_data(
            points[i], camera_index, radius=3.0, exclude_idx=None, angular_tol=ANGULAR_TOL, max_candidates=100,
        )
        if len(idx) == 0:
            continue
        s2 = directional_variance(sigma_theta[idx], np.tile(directions[i], (len(idx), 1)), degree)
        expected[i] = float(np.sum(weights**2 * s2))

    assert np.allclose(batched, expected, rtol=1e-6, atol=1e-10)
