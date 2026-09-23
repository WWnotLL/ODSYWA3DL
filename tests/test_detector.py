# Тесты слоя A: габарит, вырезы, кластеризация, дистанция.

from __future__ import annotations

import numpy as np
import pytest

from core.axis import AxisEstimate, estimate_axis
from core.config import Config
from core.detector import (PER_CLUSTER_DEBUG_KEYS, _connected_components,
                           detect, gauge_bounds)
from tests.synthetic import box_rep103, track_rep103


def _scene(cfg: Config, *objects: np.ndarray, yaw: float = -1.6):
    track = track_rep103(yaw_deg=yaw)
    axis = estimate_axis(track, cfg.axis)
    cloud = np.concatenate([track, *objects]) if objects else track
    return cloud, axis


def _on_axis(axis: AxisEstimate, x: float, z: float, offset: float = 0.0):
    return (x, float(axis.y_at(x)) + offset, z)


def test_connected_components_splits_two_blobs() -> None:
    left = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.int64)
    right = left + np.array([50, 0, 0])
    labels = _connected_components(np.concatenate([left, right]))
    assert len(set(labels[:3])) == 1 and len(set(labels[3:])) == 1
    assert labels[0] != labels[3]


def test_connected_components_joins_a_diagonal_chain() -> None:
    chain = np.arange(20, dtype=np.int64)
    cells = np.stack([chain, chain, chain], axis=1)
    assert len(set(_connected_components(cells))) == 1


def test_connected_components_handles_empty() -> None:
    assert _connected_components(np.empty((0, 3), dtype=np.int64)).size == 0


def test_gauge_is_measured_from_the_rail_head(cfg: Config) -> None:
    floor, ceiling = gauge_bounds(cfg.gauge, cfg.detector)
    assert floor == pytest.approx(cfg.gauge.rail_head_offset_m + cfg.gauge.z_min_m
                                  + cfg.detector.z_ground_margin_m)
    assert ceiling == pytest.approx(cfg.gauge.rail_head_offset_m + cfg.gauge.z_max_m)
    assert floor > cfg.gauge.rail_head_offset_m


def test_rails_themselves_are_never_candidates(cfg: Config) -> None:
    from tests.synthetic import track_rep103

    floor, _ = gauge_bounds(cfg.gauge, cfg.detector)
    cloud = track_rep103(yaw_deg=-1.6)
    rail_band = cloud[np.abs(cloud[:, 2] - cfg.gauge.rail_head_offset_m) < 0.01]
    assert rail_band.shape[0] > 0
    assert rail_band[:, 2].max() < floor


def test_empty_track_gives_no_detections(cfg: Config) -> None:
    cloud, axis = _scene(cfg)
    result = detect(cloud, axis, cfg.gauge, cfg.detector)
    assert not result.obstacle_found
    assert result.nearest_distance_m is None


def test_object_on_the_axis_is_found(cfg: Config) -> None:
    cloud, axis = _scene(cfg)
    box = box_rep103(_on_axis(axis, 20.0, 1.0))
    result = detect(np.concatenate([cloud, box]), axis, cfg.gauge, cfg.detector)
    assert result.obstacle_found
    assert result.nearest_distance_m == pytest.approx(20.0, abs=1.0)
    assert result.detections[0].source == "gauge"


def test_object_on_the_neighbouring_track_is_rejected(cfg: Config) -> None:
    cloud, axis = _scene(cfg)
    box = box_rep103(_on_axis(axis, 20.0, 1.0, offset=2.2))
    result = detect(np.concatenate([cloud, box]), axis, cfg.gauge, cfg.detector)
    assert not result.obstacle_found


def test_judging_by_raw_y_would_get_it_wrong(cfg: Config) -> None:
    cloud, axis = _scene(cfg, yaw=-2.5)
    centre = _on_axis(axis, 50.0, 1.0)
    assert abs(centre[1]) > cfg.gauge.half_width_m
    result = detect(np.concatenate([cloud, box_rep103(centre)]), axis, cfg.gauge, cfg.detector)
    assert result.obstacle_found


def test_low_object_below_the_gauge_floor_is_ignored(cfg: Config) -> None:
    cloud, axis = _scene(cfg)
    flat = box_rep103(_on_axis(axis, 20.0, 0.20), size=(0.6, 0.6, 0.05))
    result = detect(np.concatenate([cloud, flat]), axis, cfg.gauge, cfg.detector)
    assert not result.obstacle_found


def test_corridor_is_truncated_to_the_traced_base(cfg: Config) -> None:
    track = track_rep103(yaw_deg=-1.6, x_max=30.0)
    axis = estimate_axis(track, cfg.axis)
    assert axis.x_end_m < 32.0
    beyond = box_rep103((50.0, float(axis.y_at(50.0)), 1.0))
    result = detect(np.concatenate([track, beyond]), axis, cfg.gauge, cfg.detector)
    assert not result.obstacle_found
    assert result.debug["corridor"]["x_m"][1] == pytest.approx(axis.x_end_m, abs=0.01)
    assert result.debug["corridor"]["truncated_to_axis_base"]


def test_no_axis_is_not_the_same_as_no_obstacle(cfg: Config) -> None:
    cloud, _ = _scene(cfg)
    result = detect(cloud, None, cfg.gauge, cfg.detector)
    assert not result.obstacle_found
    assert result.debug["reason"] == "axis_unavailable"
    assert result.debug["axis"] is None


def test_confidence_falls_for_a_stale_axis(cfg: Config) -> None:
    cloud, axis = _scene(cfg)
    box = box_rep103(_on_axis(axis, 30.0, 1.0, offset=1.2))
    scene = np.concatenate([cloud, box])

    fresh = detect(scene, axis, cfg.gauge, cfg.detector)
    stale = detect(scene, AxisEstimate(**{
        **axis.__dict__, "source": "stale", "staleness_frames": 3,
        "sigma_y_1m": 40.0 * axis.sigma_y_1m,
    }), cfg.gauge, cfg.detector)

    assert fresh.obstacle_found and stale.obstacle_found
    assert stale.detections[0].score < fresh.detections[0].score
    assert stale.detections[0].distance_m == pytest.approx(fresh.detections[0].distance_m)


def test_an_object_on_the_axis_keeps_confidence_when_stale(cfg: Config) -> None:
    cloud, axis = _scene(cfg)
    stale_axis = AxisEstimate(**{
        **axis.__dict__, "source": "stale", "staleness_frames": 2,
        "sigma_y_1m": 20.0 * axis.sigma_y_1m,
    })
    centred = detect(np.concatenate([cloud, box_rep103(_on_axis(axis, 25.0, 1.0))]),
                     stale_axis, cfg.gauge, cfg.detector)
    edging = detect(np.concatenate([cloud, box_rep103(_on_axis(axis, 25.0, 1.0, offset=1.35))]),
                    stale_axis, cfg.gauge, cfg.detector)
    assert centred.detections[0].score > edging.detections[0].score


def test_far_object_is_not_split_into_pieces(cfg: Config) -> None:
    cloud, axis = _scene(cfg)
    box = box_rep103(_on_axis(axis, 50.0, 1.0), n=6)
    result = detect(np.concatenate([cloud, box]), axis, cfg.gauge, cfg.detector)
    assert result.obstacle_found
    assert len(result.detections) == 1


def test_point_threshold_relaxes_with_range(cfg: Config) -> None:
    from core.detector import _min_points

    assert _min_points(10.0, cfg.detector) > _min_points(50.0, cfg.detector)
    assert _min_points(200.0, cfg.detector) >= cfg.detector.min_points_floor


def test_result_serialises(cfg: Config) -> None:
    from core.types import FrameResult

    cloud, axis = _scene(cfg)
    result = detect(np.concatenate([cloud, box_rep103(_on_axis(axis, 20.0, 1.0))]),
                    axis, cfg.gauge, cfg.detector)
    assert FrameResult.from_json(result.to_json()).nearest_distance_m == pytest.approx(
        result.nearest_distance_m
    )


def test_narrow_tunnel_wall_enters_the_corridor(cfg: Config) -> None:
    narrow = track_rep103(yaw_deg=-1.6, wall_offset=1.56)
    axis = estimate_axis(narrow, cfg.axis)
    result = detect(narrow, axis, cfg.gauge, cfg.detector)
    assert result.obstacle_found, "стена обязана попадать в коридор — это измеренный факт"
    wall = max(result.detections, key=lambda d: d.n_points)
    length = wall.bbox_max_xyz[0] - wall.bbox_min_xyz[0]
    height = wall.bbox_max_xyz[2] - wall.bbox_min_xyz[2]
    assert length / height > 6.0, "ложняк протяжённый вдоль пути — именно этим он и отличим"


def test_model_error_enters_the_confidence(cfg: Config) -> None:
    cloud, axis = _scene(cfg)


    edge_box = box_rep103(_on_axis(axis, 25.0, 1.0, offset=1.45), size=(0.4, 0.06, 1.2))

    exact = AxisEstimate(**{**axis.__dict__, "residual_max_m": 0.0})
    sloppy = AxisEstimate(**{**axis.__dict__, "residual_max_m": cfg.axis.clearance_m})

    a = detect(np.concatenate([cloud, edge_box]), exact, cfg.gauge, cfg.detector)
    b = detect(np.concatenate([cloud, edge_box]), sloppy, cfg.gauge, cfg.detector)
    assert a.obstacle_found and b.obstacle_found
    assert b.detections[0].score < a.detections[0].score


def test_model_error_does_not_punish_an_object_on_the_axis(cfg: Config) -> None:
    cloud, axis = _scene(cfg)
    sloppy = AxisEstimate(**{**axis.__dict__, "residual_max_m": cfg.axis.clearance_m})
    result = detect(np.concatenate([cloud, box_rep103(_on_axis(axis, 25.0, 1.0))]),
                    sloppy, cfg.gauge, cfg.detector)
    assert result.detections[0].score == pytest.approx(1.0)


def test_gauge_margin_is_reported_for_diagnosis(cfg: Config) -> None:
    cloud, axis = _scene(cfg)
    result = detect(np.concatenate([cloud, box_rep103(_on_axis(axis, 20.0, 1.0))]),
                    axis, cfg.gauge, cfg.detector)
    assert len(result.debug["gauge_margin_m"]) == len(result.detections)
    assert result.debug["gauge_margin_m"][0] > 0.0


def _wall_along_axis(axis, railhead: float, length_m: float, bottom: float,
                     top: float, lateral: float = 1.42) -> np.ndarray:
    x = np.linspace(18.0 - 0.5 * length_m, 18.0 + 0.5 * length_m, 40)
    y = np.linspace(-0.20, 0.0, 3)
    z = np.linspace(bottom, top, 12)
    gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
    return np.stack([gx.ravel(),
                     axis.y_at(gx.ravel()) - lateral + gy.ravel(),
                     gz.ravel()], axis=1)


def _platform_wall(cfg: Config, axis, length_m: float = 14.0) -> np.ndarray:
    railhead = cfg.gauge.rail_head_offset_m
    x = np.linspace(18.0 - 0.5 * length_m, 18.0 + 0.5 * length_m, 40)


    y = np.linspace(-0.20, 0.0, 3)
    z = np.linspace(railhead + 0.30, railhead + 1.05, 12)
    gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
    return np.stack([gx.ravel(),
                     axis.y_at(gx.ravel()) - 1.42 + gy.ravel(),
                     gz.ravel()], axis=1)


def test_platform_edge_is_cut_when_the_wall_is_continuous(cfg: Config) -> None:
    cloud, axis = _scene(cfg)
    railhead = cfg.gauge.rail_head_offset_m
    scene = np.concatenate([cloud, _platform_wall(cfg, axis)])
    result = detect(scene, axis, cfg.gauge, cfg.detector)
    assert result.debug["platform"]["cut"] == [False, True]
    assert not result.obstacle_found


    above = box_rep103(_on_axis(axis, 18.0, railhead + 1.55, offset=-1.44), size=(0.3, 0.3, 0.3))
    assert detect(np.concatenate([scene, above]), axis,
                  cfg.gauge, cfg.detector).obstacle_found


def test_a_lone_object_in_the_platform_band_is_not_cut(cfg: Config) -> None:
    cloud, axis = _scene(cfg)
    railhead = cfg.gauge.rail_head_offset_m
    lone = box_rep103(_on_axis(axis, 20.0, railhead + 0.60, offset=-1.42), size=(0.3, 0.2, 0.6))
    result = detect(np.concatenate([cloud, lone]), axis, cfg.gauge, cfg.detector)
    assert result.debug["platform"]["cut"] == [False, False]
    assert result.obstacle_found


def test_platform_evidence_needs_continuity_along_the_track(cfg: Config) -> None:
    cloud, axis = _scene(cfg)
    short = detect(np.concatenate([cloud, _platform_wall(cfg, axis, length_m=4.0)]),
                   axis, cfg.gauge, cfg.detector)
    assert short.debug["platform"]["cut"] == [False, False]


def test_platform_cut_does_not_reach_the_track(cfg: Config) -> None:
    cloud, axis = _scene(cfg)
    railhead = cfg.gauge.rail_head_offset_m
    on_track = box_rep103(_on_axis(axis, 20.0, railhead + 0.60), size=(0.3, 0.3, 0.3))
    assert detect(np.concatenate([cloud, on_track]), axis,
                  cfg.gauge, cfg.detector).obstacle_found
    assert not cfg.gauge.in_platform_zone(np.array([0.0]), np.array([0.6]))[0]
    assert cfg.gauge.in_platform_zone(np.array([-1.45]), np.array([1.05]))[0]


def test_rail_mask_cuts_the_heads_but_not_the_space_between_them(cfg: Config) -> None:
    gauge = cfg.gauge
    низко = np.array([0.03, 0.03, 0.03])
    assert gauge.in_rail_mask(np.array([0.76]), низко[:1])[0]
    assert gauge.in_rail_mask(np.array([-0.76]), низко[:1])[0]

    assert not gauge.in_rail_mask(np.array([0.0]), низко[:1])[0]
    assert not gauge.in_rail_mask(np.array([0.40]), низко[:1])[0]
    assert not gauge.in_rail_mask(np.array([1.20]), низко[:1])[0]

    assert not gauge.in_rail_mask(np.array([0.76]), np.array([0.30]))[0]


def test_lower_floor_sees_a_low_object_between_the_rails(cfg: Config) -> None:
    cloud, axis = _scene(cfg)
    railhead = cfg.gauge.rail_head_offset_m
    low = box_rep103(_on_axis(axis, 15.0, railhead + 0.13), size=(0.3, 0.3, 0.25))
    result = detect(np.concatenate([cloud, low]), axis, cfg.gauge, cfg.detector)
    assert result.obstacle_found


def test_every_per_cluster_debug_list_is_registered(cfg: Config) -> None:
    cloud, axis = _scene(cfg)
    scene = np.concatenate([
        cloud,
        box_rep103(_on_axis(axis, 15.0, 1.0)),
        box_rep103(_on_axis(axis, 22.0, 1.0)),
    ])
    result = detect(scene, axis, cfg.gauge, cfg.detector)
    assert len(result.detections) >= 2
    per_cluster = {
        key for key, value in result.debug.items()
        if isinstance(value, list) and len(value) == len(result.detections)
    }
    assert per_cluster == set(PER_CLUSTER_DEBUG_KEYS)


def test_a_long_low_object_is_not_taken_for_a_platform(cfg: Config) -> None:
    cloud, axis = _scene(cfg)
    railhead = cfg.gauge.rail_head_offset_m
    long_low = _wall_along_axis(axis, railhead, length_m=12.0,
                               bottom=railhead, top=railhead + 0.55)
    result = detect(np.concatenate([cloud, long_low]), axis, cfg.gauge, cfg.detector)
    assert result.debug["platform"]["cut"] == [False, False]


    assert result.debug["n_clusters"] >= 1


def test_a_rail_lying_on_the_bed_is_below_the_evidence_band(cfg: Config) -> None:
    cloud, axis = _scene(cfg)
    railhead = cfg.gauge.rail_head_offset_m
    rail = _wall_along_axis(axis, railhead, length_m=12.0,
                            bottom=0.0, top=0.18)
    result = detect(np.concatenate([cloud, rail]), axis, cfg.gauge, cfg.detector)
    assert result.debug["platform"]["cut"] == [False, False]

