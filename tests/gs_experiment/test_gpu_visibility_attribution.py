"""GPU-only tests for gpu_visibility_attribution.batched_attribute_observations
-- the batched-GPU equivalent of visibility_attribution.attribute_observations
(and the occlusion_mask it calls once per camera). Skipped automatically
(not failed) when torch/CUDA aren't available, matching
test_gs_pixel_uncertainty_gsplat.py's convention.

The core claim being tested is *exact* agreement with the already-validated
scalar attribute_observations, not just "runs and returns something
plausible" -- the dense-grid/max-pool reformulation is argued in
gpu_visibility_attribution.py's module docstring to be mathematically
identical to occlusion_mask's own neighbor-cell lookup, not an
approximation, so exact per-camera index-set agreement is the right bar
(and is what's checked here), unlike occlusion_mask's own test against an
independent brute-force circular reference (which tolerates a bounded
false-positive rate by design -- see test_gs_visibility_attribution.py).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("gpu_visibility_attribution needs a real GPU", allow_module_level=True)

from gs_experiment.camera import CameraPose  # noqa: E402
from gs_experiment.gpu_visibility_attribution import batched_attribute_observations  # noqa: E402
from gs_experiment.visibility_attribution import attribute_observations  # noqa: E402


def make_camera(center=(0.0, 0.0, 0.0), forward=(1.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)):
    return CameraPose(center=np.array(center, dtype=float), forward=np.array(forward, dtype=float), up=np.array(up, dtype=float))


def assert_exact_match(scalar_result, batched_result, n_cameras):
    for c in range(n_cameras):
        np.testing.assert_array_equal(np.sort(scalar_result[c]), np.sort(batched_result[c]))


def test_matches_scalar_on_the_hand_placed_occluder_case():
    """The exact three-point occluder/target/off-bearing scene
    test_gs_visibility_attribution.py::test_occlusion_mask_flags_splat_behind_a_closer_occluder
    uses, run through the full attribute_observations contract."""
    camera = make_camera()
    # y=2.0, not the original test's y=3.0: bearing_y=3/5=0.6 exceeds tan(30deg)=0.577,
    # i.e. outside a 60deg frustum -- fine for occlusion_mask's own direct test (no
    # frustum step there), not for this one's full attribute_observations contract.
    positions = np.array([[2.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 2.0, 0.0]])
    scalar = attribute_observations(positions, [camera], fov_deg=60.0, angular_tol=0.1, depth_margin=0.05)
    batched = batched_attribute_observations(positions, [camera], fov_deg=60.0, angular_tol=0.1, depth_margin=0.05, device="cuda")
    assert_exact_match(scalar, batched, 1)
    # occluder (0) and the off-bearing point (2) survive; the occluded target (1) doesn't.
    assert set(batched[0].tolist()) == {0, 2}


def test_matches_scalar_on_random_multi_camera_scenes():
    rng = np.random.default_rng(0)
    cameras = [
        make_camera(center=(0.0, y, 0.0), forward=(1.0, 0.0, 0.0), up=(0.0, 0.0, 1.0))
        for y in np.linspace(-2.0, 2.0, 6)
    ]
    for angular_tol in (0.02, 0.1, 0.5):
        positions = np.stack(
            [rng.uniform(3.0, 8.0, 200), rng.uniform(-3.0, 3.0, 200), rng.uniform(-3.0, 3.0, 200)], axis=1
        )
        scalar = attribute_observations(positions, cameras, fov_deg=60.0, angular_tol=angular_tol, depth_margin=0.05)
        batched = batched_attribute_observations(
            positions, cameras, fov_deg=60.0, angular_tol=angular_tol, depth_margin=0.05, device="cuda"
        )
        assert_exact_match(scalar, batched, len(cameras))


def test_matches_scalar_with_a_camera_that_sees_nothing():
    """A camera pointed entirely away from every splat -- attribute_observations's
    own empty-frustum branch, reproduced by the fully-visible=False path."""
    positions = np.array([[5.0, 0.0, 0.0], [6.0, 1.0, 0.0]])
    empty_camera = make_camera(center=(100.0, 100.0, 100.0), forward=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0))
    normal_camera = make_camera()
    cameras = [empty_camera, normal_camera]

    scalar = attribute_observations(positions, cameras, fov_deg=60.0, angular_tol=0.1, depth_margin=0.05)
    batched = batched_attribute_observations(positions, cameras, fov_deg=60.0, angular_tol=0.1, depth_margin=0.05, device="cuda")
    assert_exact_match(scalar, batched, 2)
    assert scalar[0].size == 0


def test_matches_scalar_with_dense_clustered_points_forcing_many_shared_cells():
    """Many points crammed close together in bearing (small angular_tol
    relative to point spread) -- exercises heavy occlusion and many points
    sharing/competing for the same grid cells, the case most likely to
    expose an off-by-one in the grid-index/pooling-window construction."""
    rng = np.random.default_rng(1)
    camera = make_camera()
    positions = np.stack(
        [rng.uniform(4.0, 4.5, 150), rng.uniform(-0.2, 0.2, 150), rng.uniform(-0.2, 0.2, 150)], axis=1
    )
    scalar = attribute_observations(positions, [camera], fov_deg=60.0, angular_tol=0.05, depth_margin=0.05)
    batched = batched_attribute_observations(positions, [camera], fov_deg=60.0, angular_tol=0.05, depth_margin=0.05, device="cuda")
    assert_exact_match(scalar, batched, 1)
