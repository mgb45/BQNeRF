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
    result = engine.rendering_aware_variance_via_gsplat(
        np.array([3.5, 0.0, 0.0]), camera, K, width=100, height=100, radius=3.0
    )
    assert np.isfinite(result.mean)
    assert result.variance >= 0.0


def test_rendering_aware_variance_via_gsplat_weights_toward_the_occluder():
    """The real-footprint analogue of
    test_gs_pixel_uncertainty.py::test_rendering_aware_variance_along_ray_weights_toward_the_occluder_not_the_occluded_target:
    a fully opaque, large-footprint occluder in front of a target on the
    same ray should pull the posterior mean onto (or very near) the
    occluder's own color, using gsplat's own real anisotropic footprint
    and depth ordering rather than the isotropic-bearing-threshold proxy."""
    engine, camera, K, occluder_color, target_color = build_occluder_engine(occluder_opacity=1.0)
    query_point = np.array([3.5, 0.0, 0.0])

    via_gsplat = engine.rendering_aware_variance_via_gsplat(query_point, camera, K, width=100, height=100, radius=3.0)
    occlusion_blind = engine.rendering_aware_variance(query_point, radius=3.0)

    assert abs(via_gsplat.mean - occluder_color) < 1e-2
    assert abs(occlusion_blind.mean - occluder_color) > 1.0


def test_rendering_aware_variance_via_gsplat_requires_scales_and_rotations():
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
        engine.rendering_aware_variance_via_gsplat(np.array([3.0, 0.0, 0.0]), camera, K, width=100, height=100, radius=3.0)


def test_rendering_aware_variance_via_gsplat_falls_back_gracefully_when_nothing_is_visible():
    engine, camera, K, _, _ = build_occluder_engine()
    off_screen_query = np.array([3.5, 40.0, 0.0])  # nowhere near this camera's image
    result = engine.rendering_aware_variance_via_gsplat(
        off_screen_query, camera, K, width=100, height=100, radius=1.0
    )
    assert np.isfinite(result.mean)
    assert abs(result.mean) < 1e-3
    assert result.variance < 1e-3
