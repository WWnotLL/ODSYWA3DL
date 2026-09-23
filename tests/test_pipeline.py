# Тесты сборки ядра: контракт выхода, debug, преобразование систем координат.

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from core.config import Config
from core.pipeline import ObstacleDetector
from core.preprocess import FrameTransform, preprocess_frame
from tests.synthetic import box_rep103, track_rep103


LIDAR_HEIGHT_M = 1.31


def _frames(cfg: Config, n: int, with_body: bool) -> list[tuple]:
    rotation = cfg.preprocess.axes.rotation
    out = []
    for i in range(n):
        track = track_rep103(yaw_deg=-1.6)
        cloud = track
        if with_body:
            centre = (20.0, float(np.interp(20.0, [0.0, 60.0], [0.0, -1.68])), 1.0)
            cloud = np.concatenate([track, box_rep103(centre)])
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


def test_axis_is_exported_as_a_polyline_over_the_base(cfg: Config) -> None:
    detector = ObstacleDetector(cfg)
    for frame in _frames(cfg, 3, with_body=False):
        outcome = detector.process(*frame)
    axis = outcome.debug["axis"]
    line = axis["polyline"]
    points = np.asarray(line["points_xy_m"])
    assert line["step_m"] == cfg.axis.polyline_step_m
    assert points[0, 0] == round(detector.axis_tracker.last.x_start_m, 4)
    assert points[-1, 0] == round(detector.axis_tracker.last.x_end_m, 4)
    assert np.allclose(np.diff(points[:-1, 0]), cfg.axis.polyline_step_m, atol=1e-3)
    assert np.allclose(points[:, 1], detector.axis_tracker.last.y_at(points[:, 0]), atol=1e-4)
    assert set(line["source"]) <= {"rails", "bed"}
    assert axis["reason"] is None and axis["state"] == "measured"


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

