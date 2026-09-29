# Тесты сборки ядра: контракт выхода, debug, преобразование систем координат.

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy.spatial import cKDTree

from core.config import Config
from core.pipeline import ObstacleDetector
from core.preprocess import FrameTransform, preprocess_frame
from tests.synthetic import box_rep103, track_rep103


LIDAR_HEIGHT_M = 1.31


def _frames(cfg: Config, n: int, with_body: bool, body: tuple | None = None) -> list[tuple]:
    rotation = cfg.preprocess.axes.rotation
    out = []
    for i in range(n):
        track = track_rep103(yaw_deg=-1.6)
        cloud = track
        if with_body:
            centre = (20.0, float(np.interp(20.0, [0.0, 60.0], [0.0, -1.68])), 1.0)
            cloud = np.concatenate([track, box_rep103(centre)])
        if body is not None:
            cloud = np.concatenate([cloud, box_rep103(*body)])
        cloud = cloud - np.array([0.0, 0.0, LIDAR_HEIGHT_M])
        raw = cloud @ rotation


        index = np.arange(raw.shape[0])
        intensity = np.zeros(raw.shape[0], dtype=np.float32)
        ring = (index % 128).astype(np.uint16)
        t_rel = (index // 128).astype(np.float64) * 1.0e-5
        out.append((raw.astype(np.float32), intensity, ring, t_rel, int(1e8) * (i + 1)))
    return out


def _run(detector: ObstacleDetector, frames: list[tuple]) -> list[dict]:
    rows = []
    for cloud, intensity, ring, t_rel, stamp in frames:
        row = detector.process(cloud, intensity, ring, t_rel, stamp).to_dict()
        row["debug"].pop("timings_ms", None)
        rows.append(row)
    return rows


def test_two_detectors_agree_bit_for_bit(cfg: Config) -> None:
    frames = _frames(cfg, 6, with_body=True)
    assert _run(ObstacleDetector(cfg), frames) == _run(ObstacleDetector(cfg), frames)


def test_reset_returns_the_detector_to_a_clean_state(cfg: Config) -> None:
    frames = _frames(cfg, 6, with_body=True)
    detector = ObstacleDetector(cfg)
    first = _run(detector, frames)
    detector.reset()
    assert _run(detector, frames) == first


def test_confirmation_delays_the_alarm_by_k_minus_one(cfg: Config) -> None:
    frames = _frames(cfg, 5, with_body=True)
    raw = [r["obstacle_found"] for r in _run(ObstacleDetector(cfg, confirm=False), frames)]
    confirmed = [r["obstacle_found"] for r in _run(ObstacleDetector(cfg, confirm=True), frames)]
    assert raw[0] and all(raw)
    assert confirmed[:cfg.confirm.required_frames - 1] == [False] * (
        cfg.confirm.required_frames - 1)
    assert confirmed[cfg.confirm.required_frames - 1]


def test_verdict_sees_candidates_before_the_filter(cfg: Config) -> None:
    frames = _frames(cfg, 2, with_body=True)
    rows = _run(ObstacleDetector(cfg, confirm=True), frames)
    assert rows[0]["obstacle_found"] is False
    assert rows[0]["debug"]["candidates_raw"]["detections"]


def test_debug_carries_everything_the_node_needs(cfg: Config) -> None:
    frames = _frames(cfg, 3, with_body=True)
    debug = _run(ObstacleDetector(cfg), frames)[-1]["debug"]
    for key in ("axis", "corridor", "platform", "confirm", "verdict",
                "advance_m", "candidates_raw", "frame_transform", "checked", "frame_gap"):
        assert key in debug, key


def _broken(kind: str, cfg: Config) -> tuple:
    cloud, intensity, ring, t_rel, stamp = _frames(cfg, 1, with_body=True)[0]
    if kind == "empty":
        zero = np.zeros((0,), dtype=np.float32)
        return (np.zeros((0, 3), np.float32), zero, zero.astype(np.uint16),
                zero.astype(np.float64), stamp)
    if kind == "no_returns":
        return (np.zeros_like(cloud), intensity, ring, t_rel, stamp)
    if kind == "all_nan":
        return (np.full_like(cloud, np.nan), intensity, ring, t_rel, stamp)
    if kind == "short_ring":
        return (cloud, intensity, ring[:-5], t_rel, stamp)
    if kind == "wrong_shape":
        return (cloud[:, :2], intensity, ring, t_rel, stamp)
    raise AssertionError(kind)


def test_an_unusable_frame_is_reported_as_not_checked(cfg: Config) -> None:
    for kind in ("empty", "no_returns", "all_nan", "short_ring", "wrong_shape"):
        result = ObstacleDetector(cfg).process(*_broken(kind, cfg))
        assert result.obstacle_found is False, kind
        assert result.detections == [], kind
        assert result.debug["checked"] is False, kind
        assert result.debug["reason"] == "frame_unusable", kind
        assert result.debug["detail"], kind


def test_a_good_frame_is_marked_checked(cfg: Config) -> None:
    frames = _frames(cfg, 1, with_body=True)
    assert ObstacleDetector(cfg).process(*frames[0]).debug["checked"] is True


def test_an_unusable_frame_does_not_disturb_the_next_one(cfg: Config) -> None:
    frames = _frames(cfg, 4, with_body=True)
    clean = _run(ObstacleDetector(cfg), frames)

    detector = ObstacleDetector(cfg)
    rows = []
    for index, frame in enumerate(frames):
        if index == 2:
            detector.process(*_broken("no_returns", cfg))
        row = detector.process(*frame).to_dict()
        row["debug"].pop("timings_ms", None)
        rows.append(row)

    for row in rows:
        row["debug"].pop("frame_gap", None)
    for row in clean:
        row["debug"].pop("frame_gap", None)
    assert rows == clean


def test_a_skipped_frame_is_added_to_the_next_frame_gap(cfg: Config) -> None:
    frames = _frames(cfg, 3, with_body=True)
    detector = ObstacleDetector(cfg)
    detector.process(*frames[0])
    detector.process(*_broken("no_returns", cfg))
    detector.process(*_broken("empty", cfg))
    assert detector.process(*frames[1]).debug["frame_gap"] == 3
    assert detector.process(*frames[2]).debug["frame_gap"] == 1


def test_frame_transform_round_trip_is_exact(cfg: Config) -> None:
    frame = _frames(cfg, 1, with_body=True)[0]
    outcome = ObstacleDetector(cfg).process(*frame)
    info = outcome.debug["frame_transform"]
    transform = FrameTransform(np.asarray(info["rotation"]), np.asarray(info["translation_m"]))
    raw = frame[0].astype(np.float64)
    assert np.abs(transform.inverse(transform.apply(raw)) - raw).max() <= 1e-9
    rotation = np.asarray(info["rotation"])
    assert np.abs(rotation @ rotation.T - np.eye(3)).max() <= 1e-9
    assert abs(np.linalg.det(rotation) - 1.0) <= 1e-9


def test_frame_transform_maps_the_raw_cloud_onto_the_detector_cloud(cfg: Config) -> None:
    cloud, intensity, ring, t_rel, _ = _frames(cfg, 1, with_body=True)[0]
    prepared = preprocess_frame(cloud, intensity, ring, t_rel, cfg.preprocess)
    mapped = prepared.transform.apply(cloud[np.linalg.norm(cloud, axis=1) > 1e-6])
    gap, _ = cKDTree(prepared.xyz.astype(np.float64)).query(mapped)

    assert gap.max() <= 1e-4
    assert mapped.shape[0] == prepared.xyz.shape[0]


@pytest.mark.parametrize("walls", [False, True])
def test_axis_is_exported_as_a_polyline_over_the_base(cfg: Config, walls: bool) -> None:
    cfg = dataclasses.replace(cfg, axis=dataclasses.replace(
        cfg.axis, walls=dataclasses.replace(cfg.axis.walls, enabled=walls)))
    detector = ObstacleDetector(cfg)
    for frame in _frames(cfg, 3, with_body=False):
        outcome = detector.process(*frame)
    axis = outcome.debug["axis"]
    line = axis["polyline"]
    points = np.asarray(line["points_xy_m"])
    tracked = detector.axis_tracker.last
    assert line["step_m"] == cfg.axis.polyline_step_m
    assert points[0, 0] == round(tracked.x_start_m, 4)
    assert np.allclose(np.diff(points[:-1, 0]), cfg.axis.polyline_step_m, atol=1e-3)
    assert axis["reason"] is None and axis["state"] == "measured"
    if not walls:
        assert points[-1, 0] == round(tracked.x_end_m, 4)
        assert np.allclose(points[:, 1], tracked.y_at(points[:, 0]), atol=1e-4)
        assert set(line["source"]) <= {"rails", "bed"}
        return
    assert axis["walls"]["used"]
    assert points[-1, 0] == pytest.approx(outcome.debug["corridor"]["x_m"][1], abs=0.01)
    near = points[:, 0] <= tracked.x_end_m
    assert np.allclose(points[near, 1], tracked.y_at(points[near, 0]), atol=1e-4)
    assert set(line["source"]) <= {"rails", "bed", "walls"}
    on_walls = np.array([s == "walls" for s in line["source"]])
    assert not on_walls.any() or axis["walls"]["used"]
    if on_walls.any():
        assert on_walls[np.argmax(on_walls):].all()


def test_missing_axis_carries_its_reason(cfg: Config) -> None:
    cloud, intensity, ring, t_rel, stamp = _frames(cfg, 1, with_body=False)[0]


    height = (cloud @ cfg.preprocess.axes.rotation.T)[:, 2] + LIDAR_HEIGHT_M
    keep = (height < cfg.axis.rails_z_min_m) | (height > cfg.axis.rails_z_max_m)
    outcome = ObstacleDetector(cfg).process(
        cloud[keep], intensity[keep], ring[keep], t_rel[keep], stamp)
    axis = outcome.debug["axis"]
    assert axis["state"] == "lost"
    assert axis["reason"] in {"tracer_failed", "anchor_uninitialised"}
    assert outcome.debug["checked"] is True



def _edge_body(cfg: Config, inside_m: float) -> tuple:
    x, z, width = 20.0, 1.0, 0.10
    y_axis = float(np.interp(x, [0.0, 60.0], [0.0, -1.68]))
    edge = float(cfg.gauge.half_width_at(np.array([z - cfg.gauge.rail_head_offset_m]))[0])
    return (x, y_axis + edge - inside_m + 0.5 * width, z), (0.5, width, 0.2)


DEPTH_THRESHOLD_M = 0.045


def _with_depth_threshold(cfg: Config) -> Config:
    return dataclasses.replace(
        cfg, gauge=dataclasses.replace(cfg.gauge, train_enabled=False),
        detector=dataclasses.replace(cfg.detector, min_depth_m=DEPTH_THRESHOLD_M))


def test_a_shallow_candidate_stays_in_debug_but_never_alarms(cfg: Config) -> None:
    cfg = _with_depth_threshold(cfg)
    frames = _frames(cfg, 6, with_body=False, body=_edge_body(cfg, 0.02))
    for confirm in (True, False):
        rows = _run(ObstacleDetector(cfg, confirm=confirm), frames)
        assert not any(r["obstacle_found"] for r in rows)
        raw = rows[-1]["debug"]["candidates_raw"]
        assert raw["detections"], "тело у кромки должно остаться кандидатом"
        assert 0.0 < max(raw["depth_m"]) < cfg.detector.min_depth_m
        assert raw["min_depth_m"] == cfg.detector.min_depth_m


def test_a_candidate_deeper_than_the_threshold_still_alarms(cfg: Config) -> None:
    cfg = _with_depth_threshold(cfg)
    frames = _frames(cfg, 6, with_body=False, body=_edge_body(cfg, 0.12))
    rows = _run(ObstacleDetector(cfg, confirm=True), frames)
    assert rows[-1]["obstacle_found"]
    assert rows[-1]["debug"]["depth_m"][0] >= cfg.detector.min_depth_m


def test_the_default_config_uses_the_train_gauge_rule(cfg: Config) -> None:
    assert cfg.gauge.train_enabled is True
    assert cfg.detector.min_depth_m == 0.0
    assert cfg.detector.min_points_at_reference == 30.0
    assert cfg.detector.floor_above_rail_head_m == 0.02
    assert cfg.gauge.rail_mask_follow_heads is False


def test_a_shallow_candidate_alarms_with_the_threshold_off(cfg: Config) -> None:
    off = dataclasses.replace(cfg, detector=dataclasses.replace(cfg.detector, min_depth_m=None))
    frames = _frames(off, 6, with_body=False, body=_edge_body(off, 0.02))
    rows = _run(ObstacleDetector(off, confirm=True), frames)
    assert rows[-1]["obstacle_found"]
    assert rows[-1]["debug"]["candidates_raw"]["min_depth_m"] is None


def test_frames_without_ring_and_time_are_checked_and_give_the_same_result(cfg: Config) -> None:
    frames = []
    for cloud, intensity, _, _, stamp in _frames(cfg, 6, with_body=True):
        unit = np.round(cloud / np.linalg.norm(cloud, axis=1)[:, None], 4)
        first = np.sort(np.unique(unit, axis=0, return_index=True)[1])
        index = np.arange(first.size)
        ring = (index // 128).astype(np.uint16)
        t_rel = (index % 128).astype(np.float64) * 1.0e-5
        frames.append((cloud[first], intensity[first], ring, t_rel, stamp))
    bare = [(cloud, intensity, None, None, stamp) for cloud, intensity, _, _, stamp in frames]
    full_rows, bare_rows = _run(ObstacleDetector(cfg), frames), _run(ObstacleDetector(cfg), bare)
    assert all(r["debug"]["checked"] for r in bare_rows)
    assert bare_rows[-1]["obstacle_found"]
    full_pairing = [r["debug"].pop("dual_return") for r in full_rows]
    bare_pairing = [r["debug"].pop("dual_return") for r in bare_rows]
    assert {d["pairing"] for d in full_pairing} == {"shot"}
    assert {d["pairing"] for d in bare_pairing} == {"direction"}
    assert [d["n_unpaired"] for d in full_pairing] == [d["n_unpaired"] for d in bare_pairing]
    assert full_rows == bare_rows


def _with_latch(cfg: Config, on: bool) -> Config:
    confirm = dataclasses.replace(
        cfg.confirm, verdict=dataclasses.replace(cfg.confirm.verdict, latch_through_unchecked=on))
    return dataclasses.replace(cfg, confirm=confirm)


def test_a_stop_is_latched_through_an_unchecked_frame(cfg: Config) -> None:
    detector = ObstacleDetector(_with_latch(cfg, True))
    detector.verdict._stopped = True
    empty = np.zeros((10, 3), dtype=np.float32)
    row = detector.process(empty, np.zeros(10, np.float32), None, None, 1).to_dict()
    assert row["debug"]["checked"] is False
    assert row["debug"]["verdict"]["stop"] is True
    assert row["debug"]["verdict"]["reason"] == "latched_through_unchecked"


def test_without_the_latch_an_unchecked_frame_has_no_verdict(cfg: Config) -> None:
    detector = ObstacleDetector(_with_latch(cfg, False))
    detector.verdict._stopped = True
    empty = np.zeros((10, 3), dtype=np.float32)
    row = detector.process(empty, np.zeros(10, np.float32), None, None, 1).to_dict()
    assert "verdict" not in row["debug"]


def _guarded(cfg: Config) -> Config:
    guard = dataclasses.replace(cfg.preprocess.plane_guard, refit_limits_enabled=True,
                                plausibility_enabled=True, hold_enabled=True, rail_check_enabled=True)
    return _with_latch(dataclasses.replace(
        cfg, preprocess=dataclasses.replace(cfg.preprocess, plane_guard=guard)), True)


def test_an_exhausted_hold_gives_an_unchecked_frame_with_the_stop_latched(cfg: Config) -> None:
    guarded = _guarded(cfg)
    detector = ObstacleDetector(guarded)
    for cloud, intensity, ring, t_rel, stamp in _frames(guarded, guarded.preprocess.ground.history_length, with_body=False):
        detector.process(cloud, intensity, ring, t_rel, stamp)
    ground = detector._ground
    ground._holding, ground._hold_unknown_frames = True, guarded.axis.max_staleness_frames + 1
    detector.verdict._stopped = True
    raised = track_rep103(yaw_deg=-1.6) + np.array([0.0, 0.0, 0.5 - LIDAR_HEIGHT_M])
    raw = (raised @ guarded.preprocess.axes.rotation).astype(np.float32)
    n = raw.shape[0]
    row = detector.process(raw, np.zeros(n, np.float32), None, None, int(1e10)).to_dict()
    assert row["debug"]["checked"] is False
    assert "плоскость основания отвергнута" in row["debug"]["detail"]
    assert row["debug"]["verdict"]["stop"] is True
    assert row["debug"]["plane_guard"]["hold_expired"] is True
