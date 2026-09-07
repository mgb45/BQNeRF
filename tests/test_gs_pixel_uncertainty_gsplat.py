"""GPU-only tests for LocalUncertaintyEngine.rendering_aware_variance_via_gsplat.
Skipped automatically (not failed) when torch/gsplat or a CUDA device
aren't available -- see ../requirements-gsplat.txt -- so `pytest tests/`
stays green in a GPU-free environment.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gsplat")

if not torch.cuda.is_available():
    pytest.skip("gsplat's CUDA kernels need a real GPU", allow_module_level=True)

from bq_splat.kernels import DirectionalKernel  # noqa: E402
from gs_experiment.camera import CameraPose  # noqa: E402
from gs_experiment.pixel_uncertainty import LocalUncertaintyEngine, make_default_3d_position_kernel  # noqa: E402


def build_occluder_engine(occluder_opacity=1.0, occluder_scale=0.5, target_scale=0.1):
    """Same minimal occluder/target ray geometry as
    test_gs_pixel_uncertainty.py::build_occluder_engine, now carrying real
    scales/rotations so rendering_aware_variance_via_gsplat's actual
    anisotropic-footprint path can run."""
    bounds = ((-1.0, 10.0), (-5.0, 5.0), (-5.0, 5.0))
    positions = np.array([[2.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    occluder_color, target_color = 0.0, 10.0
    values = np.array([occluder_color, target_color])
    opacities = np.array([occluder_opacity, 0.9])
    scales = np.array([[occluder_scale] * 3, [target_scale] * 3])
    rotations = np.tile([1.0, 0.0, 0.0, 0.0], (2, 1))
    camera = CameraPose(center=np.array([-10.0, 0.0, 0.0]), forward=np.array([1.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))
    K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    pos_kernel = make_default_3d_position_kernel(sigma=1.0)
    engine = LocalUncertaintyEngine(
        positions=positions, values=values, pos_kernel=pos_kernel, scene_bounds=bounds,
        opacities=opacities, scales=scales, rotations=rotations,
    )
    return engine, camera, K, occluder_color, target_color


def test_rendering_aware_variance_via_gsplat_is_finite_and_nonnegative():
    engine, camera, K, _, _ = build_occluder_engine()
    projection = engine.build_gsplat_projection(camera, K, width=100, height=100)
    result = engine.rendering_aware_variance_via_gsplat(np.array([3.5, 0.0, 0.0]), projection, radius=3.0)
    assert np.isfinite(result.mean)
    assert result.variance >= 0.0


def test_rendering_aware_variance_via_gsplat_weights_toward_the_occluder():
    """The real-footprint analogue of
    test_gs_pixel_uncertainty.py::test_rendering_aware_variance_along_ray_weights_toward_the_occluder_not_the_occluded_target:
    a fully opaque, large-footprint occluder in front of a target on the
    same ray should pull the posterior mean much closer to the occluder's
    own color than to the target's, using gsplat's own real anisotropic
    footprint and depth ordering rather than the isotropic-bearing-
    threshold proxy. Not pinned to ~0 exactly: the render weight now
    carries the occluder's *real* physical covariance (not a degenerate
    point -- see pixel_uncertainty._render_weight_from_local_weights), so
    a small, physically real bleed-through from the target's color is
    expected, not a bug."""
    engine, camera, K, occluder_color, target_color = build_occluder_engine(occluder_opacity=1.0)
    query_point = np.array([3.5, 0.0, 0.0])

    projection = engine.build_gsplat_projection(camera, K, width=100, height=100)
    via_gsplat = engine.rendering_aware_variance_via_gsplat(query_point, projection, radius=3.0)
    occlusion_blind = engine.rendering_aware_variance(query_point, radius=3.0)

    assert abs(via_gsplat.mean - occluder_color) < abs(via_gsplat.mean - target_color)
    assert abs(via_gsplat.mean - occluder_color) < 1.0
    assert abs(occlusion_blind.mean - occluder_color) > 1.0


def test_build_gsplat_projection_requires_scales_and_rotations():
    bounds = ((-1.0, 10.0), (-5.0, 5.0), (-5.0, 5.0))
    positions = np.array([[2.0, 0.0, 0.0]])
    values = np.array([1.0])
    opacities = np.array([0.9])
    pos_kernel = make_default_3d_position_kernel(sigma=1.0)
    engine = LocalUncertaintyEngine(
        positions=positions, values=values, pos_kernel=pos_kernel, scene_bounds=bounds, opacities=opacities,
    )
    camera = CameraPose(center=np.array([-10.0, 0.0, 0.0]), forward=np.array([1.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))
    K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError):
        engine.build_gsplat_projection(camera, K, width=100, height=100)


def test_rendering_aware_variance_via_gsplat_falls_back_gracefully_when_nothing_is_visible():
    engine, camera, K, _, _ = build_occluder_engine()
    projection = engine.build_gsplat_projection(camera, K, width=100, height=100)
    off_screen_query = np.array([3.5, 40.0, 0.0])  # nowhere near this camera's image
    result = engine.rendering_aware_variance_via_gsplat(off_screen_query, projection, radius=1.0)
    assert np.isfinite(result.mean)
    assert abs(result.mean) < 1e-3
    assert result.variance < 1e-3


def build_directional_gsplat_engine():
    """Two distinct, close-together splats on the same ray/pixel, each
    observed from a different direction with a very different color --
    the real-gsplat analogue of
    test_gs_pixel_uncertainty.py::build_directional_along_ray_engine."""
    bounds = ((-1.0, 10.0), (-5.0, 5.0), (-5.0, 5.0))
    positions = np.array([[4.9, 0.0, 0.0], [5.1, 0.0, 0.0]])
    values = np.array([0.0, 10.0])
    opacities = np.array([0.5, 0.5])
    scales = np.array([[0.3, 0.3, 0.3], [0.3, 0.3, 0.3]])
    rotations = np.tile([1.0, 0.0, 0.0, 0.0], (2, 1))
    dir_a = np.array([1.0, 0.0, 0.0])
    dir_b = np.array([0.0, 1.0, 0.0])
    directions = np.array([dir_a, dir_b])
    dir_kernel = DirectionalKernel(kappa=20.0)
    camera = CameraPose(center=np.array([-10.0, 0.0, 0.0]), forward=np.array([1.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))
    K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    pos_kernel = make_default_3d_position_kernel(sigma=1.0)
    engine = LocalUncertaintyEngine(
        positions=positions, values=values, pos_kernel=pos_kernel, scene_bounds=bounds,
        opacities=opacities, scales=scales, rotations=rotations, directions=directions, dir_kernel=dir_kernel,
    )
    return engine, camera, K, dir_a, dir_b


def test_rendering_aware_variance_via_gsplat_directional_is_finite_and_nonnegative():
    engine, camera, K, dir_a, _ = build_directional_gsplat_engine()
    projection = engine.build_gsplat_projection(camera, K, width=100, height=100)
    result = engine.rendering_aware_variance_via_gsplat_directional(
        np.array([5.0, 0.0, 0.0]), dir_a, projection, radius=3.0
    )
    assert np.isfinite(result.mean)
    assert result.variance >= 0.0


def test_rendering_aware_variance_via_gsplat_directional_pulls_mean_toward_matching_direction():
    engine, camera, K, dir_a, dir_b = build_directional_gsplat_engine()
    projection = engine.build_gsplat_projection(camera, K, width=100, height=100)
    query_point = np.array([5.0, 0.0, 0.0])

    toward_a = engine.rendering_aware_variance_via_gsplat_directional(query_point, dir_a, projection, radius=3.0)
    toward_b = engine.rendering_aware_variance_via_gsplat_directional(query_point, dir_b, projection, radius=3.0)

    assert toward_a.mean < toward_b.mean  # splat colored 0.0 is observed from dir_a, 10.0 from dir_b


def test_rendering_aware_variance_via_gsplat_directional_requires_directions_and_dir_kernel():
    engine, camera, K, _, _ = build_occluder_engine()
    projection = engine.build_gsplat_projection(camera, K, width=100, height=100)
    with pytest.raises(ValueError):
        engine.rendering_aware_variance_via_gsplat_directional(
            np.array([3.5, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), projection, radius=3.0
        )
