# Тесты препроцессинга: пустые лучи, эхо, плоскость пола, выравнивание.

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from core.preprocess import (
    GroundPlane,
    GroundTracker,
    _collapse_by_sort,
    _collapse_by_stride,
    _regular_pair_stride,
    PreprocessError,
    _up_from_chord,
    check_proper_rotation,
    collapse_by_direction,
    collapse_dual_returns,
    echoes_per_shot,
    estimate_ground_plane,
    level_to_ground,
    preprocess_frame,
    rail_head_heights,
    ranges_m,
    rotation_between,
    shot_key,
    to_rep103,
    valid_mask,
)
from tests.synthetic import (circular_section_rep103, source_frame, spurious_low_plane_rep103,
                             station_rep103, to_source_frame, track_rep103, tunnel_rep103)


def test_valid_mask_drops_zero_points(cfg):
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, -5.0, 0.5]])
    assert valid_mask(xyz, cfg.preprocess.min_range_m).tolist() == [False, True, False, True]


def test_valid_mask_keeps_small_but_real_measurement(cfg):
    xyz = np.array([[1e-3, 0.0, 0.0]])
    assert valid_mask(xyz, cfg.preprocess.min_range_m).all()


def test_valid_mask_rejects_wrong_shape(cfg):
    with pytest.raises(PreprocessError):
        valid_mask(np.zeros((4, 2)), cfg.preprocess.min_range_m)


def _two_shot_frame():
    ranges = np.array([12.0, 10.0, 33.0, 30.0])
    ring = np.array([3, 3, 7, 7], dtype=np.uint16)
    timestamp = np.array([1.0, 1.0, 2.0, 2.0])
    return ranges, ring, timestamp


def test_shot_key_groups_by_ring_and_timestamp():
    _, ring, timestamp = _two_shot_frame()
    key = shot_key(ring, timestamp)
    assert key[0] == key[1] and key[2] == key[3] and key[0] != key[2]


def test_echoes_per_shot_counts_pairs():
    _, ring, timestamp = _two_shot_frame()
    assert echoes_per_shot(ring, timestamp).tolist() == [2, 2]


def test_collapse_keeps_nearest_echo():
    ranges, ring, timestamp = _two_shot_frame()
    primary, secondary = collapse_dual_returns(ranges, ring, timestamp, "nearest")
    assert sorted(ranges[primary].tolist()) == [10.0, 30.0]
    assert sorted(ranges[secondary].tolist()) == [12.0, 33.0]


def test_collapse_can_keep_farthest_echo():
    ranges, ring, timestamp = _two_shot_frame()
    primary, _ = collapse_dual_returns(ranges, ring, timestamp, "farthest")
    assert sorted(ranges[primary].tolist()) == [12.0, 33.0]


def test_collapse_is_invariant_to_input_order():
    ranges, ring, timestamp = _two_shot_frame()
    permutation = np.array([2, 0, 3, 1])
    primary_a, _ = collapse_dual_returns(ranges, ring, timestamp, "nearest")
    primary_b, _ = collapse_dual_returns(
        ranges[permutation], ring[permutation], timestamp[permutation], "nearest"
    )
    assert sorted(ranges[primary_a].tolist()) == sorted(ranges[permutation][primary_b].tolist())


def test_collapse_marks_missing_partner_with_minus_one():
    ranges = np.array([10.0, 30.0, 33.0])
    ring = np.array([3, 7, 7], dtype=np.uint16)
    timestamp = np.array([1.0, 2.0, 2.0])
    primary, partner = collapse_dual_returns(ranges, ring, timestamp, "nearest")
    assert primary.size == partner.size == 2
    pairs = dict(zip(ranges[primary].tolist(), partner.tolist()))
    assert pairs[10.0] == -1
    assert ranges[pairs[30.0]] == 33.0


def test_partner_array_is_aligned_with_primary():
    frame = source_frame(n_rings=8, n_times=40, layout="blocks", empty_fraction=0.0)
    ranges = ranges_m(frame["xyz"])
    primary, partner = collapse_dual_returns(ranges, frame["ring"], frame["timestamp"], "nearest")
    key = shot_key(frame["ring"], frame["timestamp"])
    np.testing.assert_array_equal(key[primary], key[partner])
    assert np.all(ranges[partner] >= ranges[primary])


def test_collapse_handles_empty_input():
    empty = np.zeros(0)
    primary, secondary = collapse_dual_returns(
        empty, empty.astype(np.uint16), empty, "nearest"
    )
    assert primary.size == 0 and secondary.size == 0


def test_collapse_rejects_unknown_policy():
    ranges, ring, timestamp = _two_shot_frame()
    with pytest.raises(PreprocessError):
        collapse_dual_returns(ranges, ring, timestamp, "median")


def test_rotation_maps_forward_minus_y_to_plus_x(cfg):
    forward_in_data = np.array([[0.0, -100.0, 0.0]])
    rotated = to_rep103(forward_in_data, cfg.preprocess.axes.rotation, cfg.preprocess.numerics)
    np.testing.assert_allclose(rotated, [[100.0, 0.0, 0.0]], atol=1e-9)


def test_rotation_keeps_vertical_axis(cfg):
    up = np.array([[0.0, 0.0, 1.0]])
    rotated = to_rep103(up, cfg.preprocess.axes.rotation, cfg.preprocess.numerics)
    np.testing.assert_allclose(rotated, [[0.0, 0.0, 1.0]], atol=1e-9)


def test_config_rotation_is_proper(cfg):
    matrix = check_proper_rotation(cfg.preprocess.axes.rotation, cfg.preprocess.numerics)
    assert np.linalg.det(matrix) == pytest.approx(1.0)


def test_rotation_rejects_reflection(cfg):
    reflection = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.linalg.det(reflection) == pytest.approx(-1.0)
    with pytest.raises(PreprocessError, match="собственным вращением"):
        to_rep103(np.zeros((1, 3)), reflection, cfg.preprocess.numerics)


def test_rotation_rejects_non_orthonormal_matrix(cfg):
    scaled = np.diag([1.0, 1.0, 2.0])
    with pytest.raises(PreprocessError, match="ортонормирован"):
        to_rep103(np.zeros((1, 3)), scaled, cfg.preprocess.numerics)


def test_rotation_preserves_handedness(cfg):
    basis = np.eye(3)
    rotated = to_rep103(basis, cfg.preprocess.axes.rotation, cfg.preprocess.numerics)
    np.testing.assert_allclose(np.cross(rotated[0], rotated[1]), rotated[2], atol=1e-9)


def test_flip_to_z_down_is_a_rotation_not_a_sign_change(cfg):
    flip_x = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    composed = flip_x @ cfg.preprocess.axes.rotation
    assert np.linalg.det(composed) == pytest.approx(1.0)
    check_proper_rotation(composed, cfg.preprocess.numerics)

    naive = np.diag([1.0, 1.0, -1.0]) @ cfg.preprocess.axes.rotation
    assert np.linalg.det(naive) == pytest.approx(-1.0)


def test_rotation_between_handles_antiparallel(cfg):
    down, up = np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 1.0])
    matrix = rotation_between(down, up, cfg.preprocess.numerics)
    np.testing.assert_allclose(matrix @ down, up, atol=1e-9)
    assert np.linalg.det(matrix) == pytest.approx(1.0)


def test_rotation_between_identity_for_same_vector(cfg):
    up = np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(rotation_between(up, up, cfg.preprocess.numerics), np.eye(3), atol=1e-12)


def _rotation_about_y(angle_deg: float) -> np.ndarray:
    angle = np.radians(angle_deg)
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])


def test_ground_plane_recovers_horizontal_floor(cfg):
    plane = estimate_ground_plane(tunnel_rep103(), cfg.preprocess.ground)
    np.testing.assert_allclose(plane.normal, [0.0, 0.0, 1.0], atol=1e-6)
    assert plane.offset == pytest.approx(0.0, abs=1e-6)


def test_ground_plane_recovers_tilted_floor(cfg):
    tilt = _rotation_about_y(8.0)
    scene = tunnel_rep103() @ tilt.T
    plane = estimate_ground_plane(scene, cfg.preprocess.ground)
    np.testing.assert_allclose(plane.normal, tilt @ np.array([0.0, 0.0, 1.0]), atol=1e-5)


def test_ground_plane_ignores_walls_even_when_they_dominate(cfg):
    scene = tunnel_rep103(n_z=120)
    plane = estimate_ground_plane(scene, cfg.preprocess.ground)
    np.testing.assert_allclose(plane.normal, [0.0, 0.0, 1.0], atol=2e-3)
    assert plane.offset == pytest.approx(0.0, abs=cfg.preprocess.ground.distance_threshold_m)


def test_ground_plane_up_is_measured_not_assumed(cfg):
    scene = tunnel_rep103()
    plane_up = estimate_ground_plane(scene, cfg.preprocess.ground)
    plane_down = estimate_ground_plane(scene * np.array([1.0, 1.0, -1.0]), cfg.preprocess.ground)
    assert plane_up.normal[2] > 0 and plane_down.normal[2] < 0
    assert plane_up.up_source == plane_down.up_source == "mass_ratio"


def test_ground_plane_reports_mass_ratio(cfg):
    plane = estimate_ground_plane(tunnel_rep103(), cfg.preprocess.ground)
    assert plane.up_mass_ratio >= cfg.preprocess.ground.up_mass_ratio_min


def test_ground_plane_raises_on_sparse_zone(cfg):
    with pytest.raises(PreprocessError, match="ближней зоне"):
        estimate_ground_plane(tunnel_rep103(n_x=4, n_y=4, n_z=2, n_theta=4), cfg.preprocess.ground)


def test_chord_criterion_points_up_from_flat_side(cfg):
    section = circular_section_rep103()
    sign, margin = _up_from_chord(
        section, np.array([0.0, 0.0, 1.0]), 2.0, cfg.preprocess.ground
    )
    assert sign == 1 and margin > cfg.preprocess.ground.up_margin_m


def test_chord_criterion_flips_with_the_scene(cfg):
    section = circular_section_rep103() * np.array([1.0, 1.0, -1.0])
    sign, _ = _up_from_chord(section, np.array([0.0, 0.0, 1.0]), -2.0, cfg.preprocess.ground)
    assert sign == -1


def test_level_to_ground_puts_tilted_floor_at_zero(cfg):
    tilt = _rotation_about_y(8.0)
    scene = tunnel_rep103() @ tilt.T
    plane = estimate_ground_plane(scene, cfg.preprocess.ground)
    leveled, rotation = level_to_ground(scene, plane, cfg.preprocess.numerics)

    on_floor = np.abs(plane.signed_distance(scene)) <= cfg.preprocess.ground.distance_threshold_m
    assert np.abs(leveled[on_floor][:, 2]).max() < cfg.preprocess.ground.distance_threshold_m
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_level_keeps_structure_above_the_floor(cfg):
    scene = tunnel_rep103()
    plane = estimate_ground_plane(scene, cfg.preprocess.ground)
    leveled, _ = level_to_ground(scene, plane, cfg.preprocess.numerics)
    assert leveled[:, 2].min() > -cfg.preprocess.ground.distance_threshold_m
    assert leveled[:, 2].max() > 1.0


def test_preprocess_frame_counts_match_the_synthetic_scene(cfg):
    frame = source_frame()
    result = preprocess_frame(
        frame["xyz"], frame["intensity"], frame["ring"], frame["timestamp"], cfg.preprocess
    )
    stats = result.stats
    assert stats["n_input"] == 2 * frame["n_shots"]
    assert stats["n_valid"] == 2 * frame["n_live_shots"]
    assert stats["n_primary"] == frame["n_live_shots"]

    assert stats["n_secondary_divergent"] == frame["n_live_shots"]
    assert stats["empty_fraction"] == pytest.approx(frame["n_empty_shots"] / frame["n_shots"])


def test_preprocess_frame_keeps_the_near_echo(cfg):
    frame = source_frame()
    result = preprocess_frame(
        frame["xyz"], frame["intensity"], frame["ring"], frame["timestamp"], cfg.preprocess
    )

    assert np.all(result.intensity == 120.0)
    assert np.all(result.secondary_intensity == 40.0)
    assert ranges_m(result.secondary_xyz).mean() > ranges_m(result.xyz).mean()


def test_preprocess_frame_time_is_relative(cfg):
    frame = source_frame()
    result = preprocess_frame(
        frame["xyz"], frame["intensity"], frame["ring"], frame["timestamp"], cfg.preprocess
    )

    assert result.t_rel.min() == pytest.approx(0.0, abs=1e-9)
    assert result.t_rel.max() < 1.0


def test_preprocess_frame_puts_floor_at_zero_and_forward_along_x(cfg):
    frame = source_frame()
    result = preprocess_frame(
        frame["xyz"], frame["intensity"], frame["ring"], frame["timestamp"], cfg.preprocess
    )
    assert result.plane.normal[2] > 0
    assert result.xyz[:, 0].max() > 30.0
    assert np.percentile(result.xyz[:, 2], 1) > -cfg.preprocess.ground.distance_threshold_m


def test_preprocess_frame_is_bitwise_deterministic(cfg):
    frame = source_frame()
    args = (frame["xyz"], frame["intensity"], frame["ring"], frame["timestamp"], cfg.preprocess)
    first, second = preprocess_frame(*args), preprocess_frame(*args)
    assert first.xyz.tobytes() == second.xyz.tobytes()
    assert first.t_rel.tobytes() == second.t_rel.tobytes()
    assert first.plane.normal.tobytes() == second.plane.normal.tobytes()


def test_preprocess_frame_rejects_mismatched_lengths(cfg):
    frame = source_frame()
    with pytest.raises(PreprocessError, match="одной длины"):
        preprocess_frame(
            frame["xyz"], frame["intensity"][:-1], frame["ring"], frame["timestamp"], cfg.preprocess
        )


def test_preprocess_frame_accepts_precomputed_plane(cfg):
    frame = source_frame()
    baseline = preprocess_frame(
        frame["xyz"], frame["intensity"], frame["ring"], frame["timestamp"], cfg.preprocess
    )
    reused = preprocess_frame(
        frame["xyz"],
        frame["intensity"],
        frame["ring"],
        frame["timestamp"],
        cfg.preprocess,
        plane=baseline.plane,
    )
    assert reused.xyz.tobytes() == baseline.xyz.tobytes()


def test_source_frame_really_looks_forward_along_minus_y():
    forward = to_source_frame(np.array([[50.0, 0.0, 0.0]]))
    np.testing.assert_allclose(forward, [[0.0, -50.0, 0.0]], atol=1e-9)


def test_stride_detected_for_real_like_layout():
    frame = source_frame(n_rings=8, n_times=40, layout="blocks")
    key = shot_key(frame["ring"], frame["timestamp"])
    assert _regular_pair_stride(key) == 8


def test_stride_detected_for_interleaved_layout():
    frame = source_frame(n_rings=8, n_times=40, layout="interleaved")
    key = shot_key(frame["ring"], frame["timestamp"])
    assert _regular_pair_stride(key) == 1


def test_stride_is_none_when_layout_is_irregular():
    frame = source_frame(n_rings=8, n_times=40, layout="shuffled")
    key = shot_key(frame["ring"], frame["timestamp"])
    assert _regular_pair_stride(key) is None


def test_duplicate_second_echo_is_not_stored(cfg):
    frame = source_frame(layout="blocks", secondary_scale=1.0)
    result = preprocess_frame(
        frame["xyz"], frame["intensity"], frame["ring"], frame["timestamp"], cfg.preprocess
    )
    assert result.stats["n_secondary_divergent"] == 0
    assert result.secondary_xyz.shape[0] == 0


def test_both_collapse_paths_select_the_same_echoes():
    frame = source_frame(n_rings=8, n_times=40, layout="blocks")
    ranges, ring, timestamp = ranges_m(frame["xyz"]), frame["ring"], frame["timestamp"]
    key = shot_key(ring, timestamp)

    fast_primary, fast_secondary = _collapse_by_stride(ranges, _regular_pair_stride(key), True)
    slow_primary, slow_secondary = _collapse_by_sort(ranges, key, True)

    np.testing.assert_array_equal(np.sort(fast_primary), np.sort(slow_primary))
    np.testing.assert_array_equal(np.sort(fast_secondary), np.sort(slow_secondary))


def test_collapse_picks_near_echo_regardless_of_slot_order():
    frame = source_frame(n_rings=8, n_times=40, layout="blocks", empty_fraction=0.0)
    primary, _ = collapse_dual_returns(
        ranges_m(frame["xyz"]), frame["ring"], frame["timestamp"], "nearest"
    )
    assert np.all(frame["intensity"][primary] == 120.0)


def test_shuffled_layout_gives_the_same_result_as_regular(cfg):
    regular = preprocess_frame(
        *(source_frame(layout="blocks")[k] for k in ("xyz", "intensity", "ring", "timestamp")),
        cfg.preprocess,
    )
    shuffled = preprocess_frame(
        *(source_frame(layout="shuffled")[k] for k in ("xyz", "intensity", "ring", "timestamp")),
        cfg.preprocess,
    )
    assert regular.stats["n_primary"] == shuffled.stats["n_primary"]
    np.testing.assert_allclose(
        np.sort(ranges_m(regular.xyz)), np.sort(ranges_m(shuffled.xyz)), atol=1e-5
    )


def test_collapsing_before_filtering_matches_filtering_first():
    frame = source_frame(n_rings=8, n_times=200, layout="blocks")
    xyz, ring, timestamp = frame["xyz"], frame["ring"], frame["timestamp"]
    keep = valid_mask(xyz, 1e-6)

    before, _ = collapse_dual_returns(ranges_m(xyz), ring, timestamp, "nearest", valid=keep)
    before = before[keep[before]]

    after, _ = collapse_dual_returns(
        ranges_m(xyz[keep]), ring[keep], timestamp[keep], "nearest"
    )
    np.testing.assert_allclose(
        np.sort(ranges_m(xyz[before])), np.sort(ranges_m(xyz[keep][after])), atol=1e-6
    )


def test_empty_echo_never_wins_over_a_real_one():
    xyz = np.array([[0.0, 0.0, 0.0], [0.0, -25.0, 0.0]], dtype=np.float32)
    ring = np.array([0, 0], dtype=np.uint16)
    timestamp = np.array([1.0, 1.0])
    keep = valid_mask(xyz, 1e-6)
    primary, _ = collapse_dual_returns(ranges_m(xyz), ring, timestamp, "nearest", valid=keep)
    assert primary.tolist() == [1]


def test_tracker_runs_full_ransac_on_the_first_frame(cfg):
    tracker = GroundTracker(cfg.preprocess.ground)
    plane = tracker.update(tunnel_rep103())
    assert plane.source == "ransac" and tracker.n_full == 1 and tracker.n_refit == 0


def test_tracker_refines_instead_of_replanning(cfg):
    tracker = GroundTracker(cfg.preprocess.ground)
    scene = tunnel_rep103()
    tracker.update(scene)
    for _ in range(5):
        plane = tracker.update(scene)
    assert plane.source == "refit" and tracker.n_full == 1 and tracker.n_refit == 5


def test_tracker_refinement_matches_full_ransac(cfg):
    tracker = GroundTracker(cfg.preprocess.ground)
    full = tracker.update(tunnel_rep103())
    refined = tracker.update(tunnel_rep103())
    np.testing.assert_allclose(refined.normal, full.normal, atol=1e-6)
    assert refined.offset == pytest.approx(full.offset, abs=1e-6)


def test_tracker_keeps_up_direction_through_refinement(cfg):
    flipped = tunnel_rep103() * np.array([1.0, 1.0, -1.0])
    tracker = GroundTracker(cfg.preprocess.ground)
    full = tracker.update(flipped)
    refined = tracker.update(flipped)
    assert full.normal[2] < 0 and refined.normal[2] < 0


def test_tracker_replans_when_the_floor_leaves_the_corridor(cfg):
    tracker = GroundTracker(cfg.preprocess.ground)
    tracker.update(tunnel_rep103())
    moved = tunnel_rep103() + np.array([0.0, 0.0, 2.0])
    plane = tracker.update(moved)
    assert plane.source == "ransac" and tracker.n_forced == 1
    assert plane.offset == pytest.approx(-2.0, abs=0.05)


def test_tracker_does_not_stick_to_a_stale_plane(cfg):
    tracker = GroundTracker(cfg.preprocess.ground)
    tracker.update(tunnel_rep103())
    tilt = _rotation_about_y(15.0)
    plane = tracker.update(tunnel_rep103() @ tilt.T)
    assert plane.source == "ransac" and tracker.n_forced == 1
    np.testing.assert_allclose(plane.normal, tilt @ np.array([0.0, 0.0, 1.0]), atol=1e-3)


def test_tracker_replans_on_schedule(cfg):
    import dataclasses

    ground = dataclasses.replace(cfg.preprocess.ground, replan_every_n_frames=2)
    tracker = GroundTracker(ground)
    scene = tunnel_rep103()
    sources = [tracker.update(scene).source for _ in range(5)]
    assert sources == ["ransac", "refit", "refit", "ransac", "refit"]


def test_tracker_reset_forgets_previous_record(cfg):
    tracker = GroundTracker(cfg.preprocess.ground)
    tracker.update(tunnel_rep103())
    tracker.reset()
    assert tracker.plane is None
    assert tracker.update(tunnel_rep103()).source == "ransac"


def test_refit_removes_dependence_on_point_order(cfg):
    scene = tunnel_rep103()
    shuffled = scene[np.random.default_rng(7).permutation(scene.shape[0])]
    a = estimate_ground_plane(scene, cfg.preprocess.ground)
    b = estimate_ground_plane(shuffled, cfg.preprocess.ground)
    np.testing.assert_allclose(a.normal, b.normal, atol=1e-6)


def test_tracker_refines_on_a_noisy_floor(cfg):
    scene = tunnel_rep103(noise_m=0.02)
    tracker = GroundTracker(cfg.preprocess.ground)
    tracker.update(scene)
    sources = [tracker.update(tunnel_rep103(noise_m=0.02, seed=s)).source for s in (1, 2, 3)]
    assert sources == ["refit", "refit", "refit"], f"скатился в полный пересчёт: {sources}"
    assert tracker.n_forced == 0


def test_platform_does_not_win_over_the_track_bed(cfg):
    plane = estimate_ground_plane(station_rep103(), cfg.preprocess.ground)
    assert plane.offset == pytest.approx(0.0, abs=0.05), "выбрана платформа, а не путевое основание"
    assert plane.n_candidates >= 2, "платформа обязана попасть в кандидаты и быть отвергнутой"


def test_sparse_trough_below_does_not_drag_the_plane_down(cfg):
    plane = estimate_ground_plane(spurious_low_plane_rep103(), cfg.preprocess.ground)
    assert plane.offset == pytest.approx(0.0, abs=0.05)


def test_history_wins_when_the_platform_dominates(cfg):
    scene = station_rep103(density_ratio=3.0, platform_width=3.0)
    up = np.array([0.0, 0.0, 1.0])

    blind = estimate_ground_plane(scene, cfg.preprocess.ground)
    assert blind.offset == pytest.approx(-1.0, abs=0.05), "ожидался срыв на платформу"

    guided = estimate_ground_plane(
        scene, cfg.preprocess.ground, expected_offset=0.0, expected_normal=up
    )
    assert guided.offset == pytest.approx(0.0, abs=0.05)
    assert guided.n_rejected >= 1


def test_history_seeds_a_candidate_it_can_prefer(cfg):
    scene = station_rep103(density_ratio=2.0, platform_width=3.0)
    up = np.array([0.0, 0.0, 1.0])
    blind = estimate_ground_plane(scene, cfg.preprocess.ground)
    guided = estimate_ground_plane(
        scene, cfg.preprocess.ground, expected_offset=0.0, expected_normal=up
    )
    assert blind.n_candidates < guided.n_candidates


def test_history_yields_when_support_is_clearly_against(cfg):
    moved = tunnel_rep103(noise_m=0.01) + np.array([0.0, 0.0, 3.0])
    plane = estimate_ground_plane(moved, cfg.preprocess.ground, expected_offset=0.0)
    assert plane.offset == pytest.approx(-3.0, abs=0.05)


def test_tracker_tracks_running_median_not_a_constant(cfg):
    tracker = GroundTracker(cfg.preprocess.ground)
    assert tracker.expected_offset is None
    tracker.update(tunnel_rep103(noise_m=0.01) + np.array([0.0, 0.0, 5.0]))
    assert tracker.expected_offset == pytest.approx(-5.0, abs=0.05)
    tracker.reset()
    assert tracker.expected_offset is None


def test_preprocess_frame_accepts_a_tracker(cfg):
    tracker = GroundTracker(cfg.preprocess.ground)
    frame = source_frame()
    args = (frame["xyz"], frame["intensity"], frame["ring"], frame["timestamp"], cfg.preprocess)
    first = preprocess_frame(*args, tracker=tracker)
    second = preprocess_frame(*args, tracker=tracker)
    assert first.plane.source == "ransac" and second.plane.source == "refit"
    np.testing.assert_allclose(second.plane.normal, first.plane.normal, atol=1e-6)
    assert np.percentile(second.xyz[:, 2], 1) > -cfg.preprocess.ground.distance_threshold_m


def test_plane_and_tracker_are_mutually_exclusive(cfg):
    frame = source_frame()
    with pytest.raises(PreprocessError, match="не оба сразу"):
        preprocess_frame(
            frame["xyz"], frame["intensity"], frame["ring"], frame["timestamp"],
            cfg.preprocess, plane=GroundTracker(cfg.preprocess.ground).update(
                preprocess_frame(*(frame[k] for k in ("xyz", "intensity", "ring", "timestamp")),
                                 cfg.preprocess).xyz),
            tracker=GroundTracker(cfg.preprocess.ground),
        )



def _one_shot_per_direction(scene: np.ndarray) -> np.ndarray:
    unit = np.round(scene / np.linalg.norm(scene, axis=1)[:, None], 4)
    _, first = np.unique(unit, axis=0, return_index=True)
    return scene[np.sort(first)]


def _distinct_frame(**kwargs) -> dict:
    return source_frame(_one_shot_per_direction(tunnel_rep103()), secondary_scale=2.0, **kwargs)


def _shot_pairs(frame: dict, keep: str = "nearest") -> tuple[np.ndarray, np.ndarray]:
    valid = valid_mask(frame["xyz"], 1e-6)
    primary, partner = collapse_dual_returns(ranges_m(frame["xyz"]), frame["ring"],
                                             frame["timestamp"], keep, valid=valid)
    alive = valid[primary]
    return primary[alive], partner[alive]


@pytest.mark.parametrize("layout", ["blocks", "interleaved"])
@pytest.mark.parametrize("keep", ["nearest", "farthest"])
def test_direction_pairs_are_the_shot_pairs_in_the_same_order(layout, keep):
    frame = _distinct_frame(layout=layout)
    valid = valid_mask(frame["xyz"], 1e-6)
    primary, partner = collapse_by_direction(frame["xyz"], ranges_m(frame["xyz"]), keep, 5, valid, 1e-6)
    expected_primary, expected_partner = _shot_pairs(frame, keep)
    np.testing.assert_array_equal(primary, expected_primary)
    np.testing.assert_array_equal(partner, expected_partner)


def test_direction_pairing_skips_empty_points():
    xyz = np.array([[0.0, 0.0, 0.0], [0.0, -25.0, 0.0], [0.0, -50.0, 0.0]], dtype=np.float32)
    valid = valid_mask(xyz, 1e-6)
    primary, partner = collapse_by_direction(xyz, ranges_m(xyz), "nearest", 5, valid, 1e-6)
    assert primary.tolist() == [1] and partner.tolist() == [2]


def _with_points_off_the_rays(frame: dict) -> tuple[np.ndarray, np.ndarray, int]:
    face = np.stack(np.meshgrid(np.linspace(-0.5, 0.5, 7), [-40.0],
                                np.linspace(-1.0, 0.6, 9)), axis=-1).reshape(-1, 3)
    xyz = np.concatenate([frame["xyz"], face.astype(np.float32)])
    intensity = np.concatenate([frame["intensity"], np.full(face.shape[0], 7.0, np.float32)])
    return xyz, intensity, face.shape[0]


def test_points_off_the_rays_are_kept_and_counted_as_unpaired(cfg):
    frame = _distinct_frame()
    xyz, intensity, n_face = _with_points_off_the_rays(frame)
    result = preprocess_frame(xyz, intensity, None, None, cfg.preprocess)
    assert result.stats["dual_return_pairing"] == "direction"
    assert result.stats["n_unpaired"] == n_face
    assert result.stats["n_primary"] == frame["n_live_shots"] + n_face
    assert np.count_nonzero(result.intensity == 7.0) == n_face


def test_preprocess_without_ring_and_time_matches_the_full_fields(cfg):
    frame = _distinct_frame()
    full = preprocess_frame(frame["xyz"], frame["intensity"], frame["ring"], frame["timestamp"],
                            cfg.preprocess)
    bare = preprocess_frame(frame["xyz"], frame["intensity"], None, None, cfg.preprocess)
    assert full.xyz.tobytes() == bare.xyz.tobytes()
    assert full.secondary_xyz.tobytes() == bare.secondary_xyz.tobytes()
    assert full.plane.normal.tobytes() == bare.plane.normal.tobytes()
    assert full.stats["n_unpaired"] == bare.stats["n_unpaired"] == 0
    assert bare.ring is None and bare.t_rel is None and bare.secondary_ring is None


def test_ring_without_timestamp_is_rejected(cfg):
    frame = source_frame()
    with pytest.raises(PreprocessError, match="вместе"):
        preprocess_frame(frame["xyz"], frame["intensity"], frame["ring"], None, cfg.preprocess)


def test_a_pair_straddling_a_cell_border_is_still_paired():
    edge = 0.123455
    points = []
    for x, r in ((edge - 1e-9, 20.0), (edge + 1e-9, 23.0)):
        z = 0.1
        points.append(r * np.array([x, -np.sqrt(1.0 - x * x - z * z), z]))
    xyz = np.array(points)
    assert np.rint(xyz[0, 0] / 20.0 * 1e5) != np.rint(xyz[1, 0] / 23.0 * 1e5)
    primary, partner = collapse_by_direction(xyz, ranges_m(xyz), "nearest", 5, np.ones(2, bool), 1e-6)
    assert primary.tolist() == [0] and partner.tolist() == [1]


def _towards(x: float, y: float) -> np.ndarray:
    return np.array([x, y, -np.sqrt(1.0 - x * x - y * y)])


def test_a_pair_straddling_both_grids_is_paired_by_tolerance():
    x, y, step = -0.42458, -0.903015, 1e-8
    xyz = np.array([4.152 * _towards(x + step, y - step), 5.576 * _towards(x - step, y + step)])
    scaled = xyz / np.linalg.norm(xyz, axis=1)[:, None] * 1e5
    assert np.any(np.rint(scaled[0]) != np.rint(scaled[1]))
    assert np.any(np.floor(scaled[0]) != np.floor(scaled[1]))
    primary, partner = collapse_by_direction(xyz, ranges_m(xyz), "nearest", 5, np.ones(2, bool), 1e-6)
    assert primary.tolist() == [0] and partner.tolist() == [1]


def test_three_points_within_tolerance_stay_unpaired():
    x, y, step = -0.3, -0.9, 3e-5
    three = np.array([10.0 * _towards(x + i * step, y) for i in range(3)])
    primary, partner = collapse_by_direction(three, ranges_m(three), "nearest", 5,
                                             np.ones(3, bool), 1e-4)
    assert primary.tolist() == [0, 1, 2] and partner.tolist() == [-1, -1, -1]
    two = three[:2]
    primary, partner = collapse_by_direction(two, ranges_m(two), "nearest", 5,
                                             np.ones(2, bool), 1e-4)
    assert primary.tolist() == [0] and partner.tolist() == [1]


LIDAR_HEIGHT_M = 1.31


def _guard_on(cfg):
    return dataclasses.replace(
        cfg.preprocess.plane_guard, refit_limits_enabled=True, plausibility_enabled=True,
        hold_enabled=True, rail_check_enabled=True)


def _guarded_tracker(cfg):
    return GroundTracker(cfg.preprocess.ground, _guard_on(cfg), cfg.axis)


def _track_frame():
    return track_rep103() - np.array([0.0, 0.0, LIDAR_HEIGHT_M])


def _low_wide_object_frame(height_m=0.2, x=(5.0, 40.0), half_width=3.0):
    scene = track_rep103()
    covered = (scene[:, 0] >= x[0]) & (scene[:, 0] <= x[1]) & (np.abs(scene[:, 1]) <= half_width) \
        & (scene[:, 2] <= height_m)
    gx, gy = np.meshgrid(np.arange(x[0], x[1], 0.1), np.arange(-half_width, half_width, 0.1), indexing="ij")
    top = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, height_m)], axis=1)
    return np.concatenate([scene[~covered], top]) - np.array([0.0, 0.0, LIDAR_HEIGHT_M])


def _settle(tracker, cfg):
    for _ in range(cfg.preprocess.ground.history_length):
        plane = tracker.update(_track_frame())
    return plane


def test_plane_on_a_low_wide_object_is_not_accepted(cfg):
    tracker = _guarded_tracker(cfg)
    floor = _settle(tracker, cfg)
    expected = tracker.expected_offset
    held = tracker.update(_low_wide_object_frame())
    assert tracker.last_guard["accepted"] is False and tracker.last_guard["held"] is True
    assert "offset_step" in tracker.last_guard["reasons"]
    assert held.offset == pytest.approx(floor.offset, abs=1e-6)
    assert tracker.expected_offset == pytest.approx(expected, abs=1e-9)


def test_without_the_guard_the_low_wide_object_becomes_the_floor(cfg):
    tracker = GroundTracker(cfg.preprocess.ground)
    floor = _settle(tracker, cfg)
    moved = tracker.update(_low_wide_object_frame())
    assert floor.offset - moved.offset == pytest.approx(0.2, abs=0.02)


def test_hold_expires_after_the_axis_hold_path(cfg):
    tracker = _guarded_tracker(cfg)
    _settle(tracker, cfg)
    step = 0.4 * cfg.axis.max_hold_path_m
    tracker.update(_low_wide_object_frame())
    tracker.note_advance(step)
    tracker.update(_low_wide_object_frame())
    tracker.note_advance(step)
    tracker.update(_low_wide_object_frame())
    tracker.note_advance(step)
    with pytest.raises(PreprocessError, match="удержание исчерпано"):
        tracker.update(_low_wide_object_frame())
    assert tracker.last_guard["hold_expired"] is True


def test_accepted_floor_after_a_hold_resets_it(cfg):
    tracker = _guarded_tracker(cfg)
    floor = _settle(tracker, cfg)
    tracker.update(_low_wide_object_frame())
    tracker.note_advance(5.0)
    back = tracker.update(_track_frame())
    assert tracker.last_guard["accepted"] is True
    assert back.offset == pytest.approx(floor.offset, abs=1e-3)


def test_without_hold_a_rejected_plane_makes_the_frame_unusable(cfg):
    guard = dataclasses.replace(_guard_on(cfg), hold_enabled=False)
    tracker = GroundTracker(cfg.preprocess.ground, guard, cfg.axis)
    _settle(tracker, cfg)
    with pytest.raises(PreprocessError, match="отвергнута"):
        tracker.update(_low_wide_object_frame())


def _steep_band(tilt_deg=30.0):
    gx, gy = np.meshgrid(np.linspace(5.0, 40.0, 350), np.linspace(-0.4, 0.4, 40), indexing="ij")
    z = -LIDAR_HEIGHT_M + np.tan(np.radians(tilt_deg)) * gy
    return np.stack([gx.ravel(), gy.ravel(), z.ravel()], axis=1)


def test_without_refit_limits_a_steep_refinement_is_taken(cfg):
    tracker = GroundTracker(cfg.preprocess.ground)
    tracker.update(_track_frame())
    plane = tracker.update(_steep_band())
    tilt = np.degrees(np.arccos(abs(plane.normal[2])))
    assert plane.source == "refit" and tilt == pytest.approx(30.0, abs=0.5)


def test_steep_refinement_is_not_taken_with_refit_limits(cfg):
    guard = dataclasses.replace(cfg.preprocess.plane_guard, refit_limits_enabled=True)
    tracker = GroundTracker(cfg.preprocess.ground, guard, cfg.axis)
    tracker.update(_track_frame())
    before = tracker.n_forced
    with pytest.raises(PreprocessError):
        tracker.update(_steep_band())
    assert tracker.n_forced == before + 1


def _exhaust_hold_standing(tracker, cfg):
    for _ in range(cfg.axis.max_staleness_frames + 1):
        held = tracker.update(_low_wide_object_frame())
        assert tracker.last_guard["held"] is True
        tracker.note_advance(None)
    return held


def test_standing_train_does_not_take_a_wide_object_for_the_floor(cfg):
    tracker = _guarded_tracker(cfg)
    floor = _settle(tracker, cfg)
    _exhaust_hold_standing(tracker, cfg)
    for _ in range(3):
        with pytest.raises(PreprocessError, match="отвергнута"):
            tracker.update(_low_wide_object_frame())
        assert tracker.last_guard["accepted"] is False
    assert tracker.plane.offset == pytest.approx(floor.offset, abs=1e-6)


def test_after_the_hold_the_floor_is_reacquired_by_the_rails(cfg):
    tracker = _guarded_tracker(cfg)
    floor = _settle(tracker, cfg)
    _exhaust_hold_standing(tracker, cfg)
    with pytest.raises(PreprocessError):
        tracker.update(_low_wide_object_frame())
    back = tracker.update(_track_frame())
    assert tracker.last_guard["accepted"] is True
    assert tracker.last_guard["rail_heads_m"] is not None
    assert back.offset == pytest.approx(floor.offset, abs=1e-3)


def _guard_without_rail_check(cfg, **window):
    guard = dataclasses.replace(_guard_on(cfg), rail_check_enabled=False, **window)
    return GroundTracker(cfg.preprocess.ground, guard, cfg.axis)


def test_reacquire_by_the_rails_works_without_the_rail_check_flag(cfg):
    tracker = _guard_without_rail_check(cfg)
    floor = _settle(tracker, cfg)
    _exhaust_hold_standing(tracker, cfg)
    with pytest.raises(PreprocessError, match="удержание исчерпано"):
        tracker.update(_low_wide_object_frame())
    with pytest.raises(PreprocessError, match="reacquire_no_rails"):
        tracker.update(_low_wide_object_frame())
    back = tracker.update(_track_frame())
    assert tracker.last_guard["accepted"] is True
    assert tracker.last_guard["rail_heads_m"] is not None
    assert back.offset == pytest.approx(floor.offset, abs=1e-3)


def test_reacquire_takes_the_head_window_from_the_rail_check_keys(cfg):
    tracker = _guard_without_rail_check(cfg, rail_head_max_m=0.20)
    _settle(tracker, cfg)
    _exhaust_hold_standing(tracker, cfg)
    with pytest.raises(PreprocessError):
        tracker.update(_low_wide_object_frame())
    with pytest.raises(PreprocessError, match="rail_heads"):
        tracker.update(_track_frame())
    assert tracker.last_guard["accepted"] is False


def test_rail_heads_are_found_at_the_rail_head_height(cfg):
    frame = _track_frame()
    heads = rail_head_heights(frame, np.array([0.0, 0.0, 1.0]), LIDAR_HEIGHT_M, cfg.axis,
                              cfg.preprocess.plane_guard.rail_x_min_m,
                              cfg.preprocess.plane_guard.rail_x_max_m,
                              cfg.preprocess.plane_guard.rail_head_percentile)
    assert heads is not None
    assert heads[0] == pytest.approx(0.23, abs=0.01) and heads[1] == pytest.approx(0.23, abs=0.01)
    assert heads[2] == pytest.approx(1.52, abs=0.05)
