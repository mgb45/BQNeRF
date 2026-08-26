import numpy as np

from gs_experiment.prepare_nerf_synthetic import select_gap_subset, select_gradient_subset


def _fake_frames_on_ring(n, radius=5.0):
    """Frames whose c2w translation lies on a ring in the xy-plane at
    evenly spaced angles -- a real dataset's frames carry a full 4x4
    matrix, but select_gradient_subset only reads the translation column,
    so a bare (4,4) array with just that column set is enough here."""
    frames = []
    for i, theta in enumerate(np.linspace(0, 2 * np.pi, n, endpoint=False)):
        c2w = np.eye(4)
        c2w[:3, 3] = [radius * np.cos(theta), radius * np.sin(theta), 0.0]
        frames.append((f"r_{i}", c2w))
    return frames


def test_select_gradient_subset_holds_count_fixed_across_window_fractions():
    frames = _fake_frames_on_ring(100)
    for window_fraction in (0.1, 0.3, 0.6, 1.0):
        idx = select_gradient_subset(frames, n_per_zone=15, window_fraction=window_fraction)
        assert len(idx) == 15


def test_select_gradient_subset_narrow_window_stays_within_wide_windows_similarity_range():
    """A small window_fraction should only draw from views close to the
    reference; a window_fraction=1.0 draws from the whole ring -- checked
    via actual angular similarity to the reference, not just index math."""
    frames = _fake_frames_on_ring(100)
    centers = np.array([c2w[:3, 3] for _, c2w in frames])
    dirs = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    ref = dirs[0]

    idx_narrow = select_gradient_subset(frames, n_per_zone=10, window_fraction=0.1)
    idx_wide = select_gradient_subset(frames, n_per_zone=10, window_fraction=1.0)

    min_similarity_narrow = (dirs[idx_narrow] @ ref).min()
    min_similarity_wide = (dirs[idx_wide] @ ref).min()
    assert min_similarity_narrow > min_similarity_wide


def test_select_gap_subset_zero_width_keeps_everything():
    frames = _fake_frames_on_ring(100)
    idx = select_gap_subset(frames, gap_half_width_deg=0.0)
    assert len(idx) == 100


def test_select_gap_subset_removes_only_views_within_the_gap():
    frames = _fake_frames_on_ring(100)
    centers = np.array([c2w[:3, 3] for _, c2w in frames])
    dirs = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    ref = dirs[0]

    idx = select_gap_subset(frames, gap_half_width_deg=30.0, reference_idx=0)
    kept_angular_dist = np.degrees(np.arccos(np.clip(dirs[idx] @ ref, -1.0, 1.0)))
    assert np.all(kept_angular_dist > 30.0)
    # nothing outside the gap should have been dropped
    all_angular_dist = np.degrees(np.arccos(np.clip(dirs @ ref, -1.0, 1.0)))
    expected_kept = np.where(all_angular_dist > 30.0)[0]
    assert set(idx.tolist()) == set(expected_kept.tolist())


def test_select_gap_subset_wider_gap_removes_more_views():
    frames = _fake_frames_on_ring(100)
    idx_narrow_gap = select_gap_subset(frames, gap_half_width_deg=10.0)
    idx_wide_gap = select_gap_subset(frames, gap_half_width_deg=60.0)
    assert len(idx_wide_gap) < len(idx_narrow_gap) < 100


def test_select_gradient_subset_window_fraction_one_spans_full_index_range():
    frames = _fake_frames_on_ring(100)
    idx = select_gradient_subset(frames, n_per_zone=10, window_fraction=1.0, reference_idx=0)
    # with window_fraction=1.0 the window is the entire similarity-sorted
    # pool; evenly subsampling 10 points from it should include both the
    # most- and least-similar view (first and last of the sorted order).
    centers = np.array([c2w[:3, 3] for _, c2w in frames])
    dirs = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    ref = dirs[0]
    similarity = dirs @ ref
    order = np.argsort(-similarity)
    assert order[0] in idx
    assert order[-1] in idx
