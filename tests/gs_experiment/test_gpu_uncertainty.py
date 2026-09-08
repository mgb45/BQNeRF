"""GPU-only tests for gpu_uncertainty.compute_directional_variance_batched --
the batched-GPU equivalent of `LocalUncertaintyEngine.
rendering_aware_variance_along_ray_directional`. Skipped automatically (not
failed) when torch/CUDA aren't available, matching
test_gs_pixel_uncertainty_gsplat.py's convention -- see
../requirements-gsplat.txt.

The core claim being tested is numerical agreement with the already-
validated scalar per-pixel path, not just "runs and returns something
plausible": every test here calls both and compares.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("gpu_uncertainty needs a real GPU", allow_module_level=True)

from gs_experiment.camera import CameraPose  # noqa: E402
from gs_experiment.gpu_uncertainty import compute_directional_variance_batched  # noqa: E402
from gs_experiment.kernels import DirectionalKernel  # noqa: E402
from gs_experiment.pixel_uncertainty import LocalUncertaintyEngine, make_default_3d_position_kernel  # noqa: E402

SIGMA = 1.0
KAPPA = 20.0
RADIUS = 3.0
ANGULAR_TOL = 0.2


def build_directional_along_ray_engine(n_splats: int = 24, seed: int = 0):
    """A denser real-ish scene (unlike the two-splat toy fixtures elsewhere
    in this package): enough splats, opacities, and observing directions
    for the batched candidate search/capping/depth-ordering machinery to
    actually be exercised, not trivially satisfied by n<=2 candidates."""
    rng = np.random.default_rng(seed)
    positions = np.stack(
        [rng.uniform(4.0, 6.0, n_splats), rng.uniform(-1.5, 1.5, n_splats), rng.uniform(-1.5, 1.5, n_splats)], axis=1
    )
    values = rng.uniform(0.0, 10.0, n_splats)
    opacities = rng.uniform(0.1, 0.9, n_splats)
    directions = rng.normal(size=(n_splats, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    dir_kernel = DirectionalKernel(kappa=KAPPA)
    camera = CameraPose(center=np.array([-10.0, 0.0, 0.0]), forward=np.array([1.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))
    pos_kernel = make_default_3d_position_kernel(sigma=SIGMA)
    engine = LocalUncertaintyEngine(
        positions=positions, values=values, pos_kernel=pos_kernel,
        scene_bounds=((-1.0, 10.0), (-5.0, 5.0), (-5.0, 5.0)),
        opacities=opacities, directions=directions, dir_kernel=dir_kernel,
    )
    return engine, camera


def scalar_variances(engine, camera_index, points, directions, max_candidates=500):
    return np.array(
        [
            engine.rendering_aware_variance_along_ray_directional(
                p, d, camera_index, RADIUS, angular_tol=ANGULAR_TOL, max_candidates=max_candidates,
            ).variance
            for p, d in zip(points, directions)
        ]
    )


def test_batched_matches_scalar_on_many_random_queries():
    engine, camera = build_directional_along_ray_engine()
    camera_index = engine.build_bearing_index(camera)

    rng = np.random.default_rng(1)
    n_queries = 30
    points = np.stack(
        [rng.uniform(4.5, 5.5, n_queries), rng.uniform(-1.0, 1.0, n_queries), rng.uniform(-1.0, 1.0, n_queries)], axis=1
    )
    directions = rng.normal(size=(n_queries, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    scalar = scalar_variances(engine, camera_index, points, directions)
    batched, prior = compute_directional_variance_batched(
        engine, camera_index, points, directions, angular_tol=ANGULAR_TOL, sigma_rbf=SIGMA, kappa=KAPPA,
        max_candidates=500, device="cuda",
    )

    assert np.allclose(batched, scalar, rtol=1e-6, atol=1e-8)
    assert np.all(batched <= prior + 1e-8)  # posterior variance can never exceed the prior


def test_batched_matches_scalar_with_candidate_cap_forcing_ranking():
    """max_candidates smaller than the real candidate count forces the
    overflow-ranking branch (rank by directional alignment, keep the top
    max_candidates) on both paths -- the case most likely to disagree if
    the batched top-k selection didn't exactly mirror
    CameraSplatIndex.query's own ranking."""
    engine, camera = build_directional_along_ray_engine(n_splats=40)
    camera_index = engine.build_bearing_index(camera)

    rng = np.random.default_rng(2)
    points = np.tile(np.array([5.0, 0.0, 0.0]), (10, 1)) + rng.normal(scale=0.05, size=(10, 3))
    directions = rng.normal(size=(10, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    max_candidates = 10  # well below n_splats, forces capping
    scalar = scalar_variances(engine, camera_index, points, directions, max_candidates=max_candidates)
    batched, _ = compute_directional_variance_batched(
        engine, camera_index, points, directions, angular_tol=1.0, sigma_rbf=SIGMA, kappa=KAPPA,
        max_candidates=max_candidates, device="cuda",
    )

    assert np.allclose(batched, scalar, rtol=1e-6, atol=1e-8)


def test_batched_matches_scalar_with_pixel_chunking():
    """A tiny pixel_chunk_bytes forces the multi-chunk code path (more
    than one chunk for a modest number of queries) -- results must be
    identical to the single-chunk case, since chunking is purely an
    implementation detail for memory, not a change in what's computed."""
    engine, camera = build_directional_along_ray_engine()
    camera_index = engine.build_bearing_index(camera)

    rng = np.random.default_rng(3)
    n_queries = 20
    points = np.stack(
        [rng.uniform(4.5, 5.5, n_queries), rng.uniform(-1.0, 1.0, n_queries), rng.uniform(-1.0, 1.0, n_queries)], axis=1
    )
    directions = rng.normal(size=(n_queries, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    unchunked, unchunked_prior = compute_directional_variance_batched(
        engine, camera_index, points, directions, angular_tol=ANGULAR_TOL, sigma_rbf=SIGMA, kappa=KAPPA,
        max_candidates=500, device="cuda",
    )
    chunked, chunked_prior = compute_directional_variance_batched(
        engine, camera_index, points, directions, angular_tol=ANGULAR_TOL, sigma_rbf=SIGMA, kappa=KAPPA,
        max_candidates=500, device="cuda", pixel_chunk_bytes=1.0,  # forces chunk_size == 1
    )

    assert np.allclose(chunked, unchunked, rtol=1e-6, atol=1e-8)
    assert np.allclose(chunked_prior, unchunked_prior, rtol=1e-6, atol=1e-8)


def test_batched_matches_scalar_without_scales_and_rotations():
    """The engine fixture here has no scales/rotations set (opacities
    only) -- both paths must fall back to the spread-of-centers-only
    moment match (local_covariances=None) rather than erroring."""
    engine, camera = build_directional_along_ray_engine()
    assert engine.scales is None and engine.rotations is None
    camera_index = engine.build_bearing_index(camera)

    point = np.array([[5.0, 0.0, 0.0]])
    direction = np.array([[1.0, 0.0, 0.0]])

    scalar = scalar_variances(engine, camera_index, point, direction)
    batched, _ = compute_directional_variance_batched(
        engine, camera_index, point, direction, angular_tol=ANGULAR_TOL, sigma_rbf=SIGMA, kappa=KAPPA,
        max_candidates=500, device="cuda",
    )
    assert np.allclose(batched, scalar, rtol=1e-6, atol=1e-8)


def test_batched_is_nonnegative_and_finite():
    engine, camera = build_directional_along_ray_engine()
    camera_index = engine.build_bearing_index(camera)
    points = np.array([[5.0, 0.0, 0.0], [5.2, 0.3, -0.2]])
    directions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    result, prior = compute_directional_variance_batched(
        engine, camera_index, points, directions, angular_tol=ANGULAR_TOL, sigma_rbf=SIGMA, kappa=KAPPA,
        max_candidates=500, device="cuda",
    )
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0.0)
    assert np.all(np.isfinite(prior))
    assert np.all(prior >= 0.0)
    assert np.all(result <= prior + 1e-8)


def test_batched_handles_empty_camera_index():
    """A camera with zero in-frustum candidates (query aimed nowhere near
    any observing camera) -- the scalar path's zero-candidate fallback,
    reproduced by the k_cap==0 branch."""
    positions = np.array([[5.0, 0.0, 0.0]])
    values = np.array([1.0])
    opacities = np.array([0.9])
    directions = np.array([[1.0, 0.0, 0.0]])
    dir_kernel = DirectionalKernel(kappa=KAPPA)
    pos_kernel = make_default_3d_position_kernel(sigma=SIGMA)
    engine = LocalUncertaintyEngine(
        positions=positions, values=values, pos_kernel=pos_kernel, scene_bounds=((-1.0, 10.0), (-5.0, 5.0), (-5.0, 5.0)),
        opacities=opacities, directions=directions, dir_kernel=dir_kernel,
    )
    # Camera pointed so nothing in `positions` is in its frustum/front.
    camera = CameraPose(center=np.array([50.0, 50.0, 50.0]), forward=np.array([0.0, 0.0, 1.0]), up=np.array([0.0, 1.0, 0.0]))
    camera_index = engine.build_bearing_index(camera)
    assert camera_index.indices.shape[0] == 0

    result, prior = compute_directional_variance_batched(
        engine, camera_index, np.array([[5.0, 0.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]),
        angular_tol=ANGULAR_TOL, sigma_rbf=SIGMA, kappa=KAPPA, max_candidates=500, device="cuda",
    )
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0.0)
    # No candidates at all -- posterior collapses exactly to the prior (nothing to condition on).
    assert np.allclose(result, prior, rtol=1e-6, atol=1e-8)
