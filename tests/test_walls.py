# Тесты дальнего участка оси по стенам тоннеля: две стены, одна стена, смена сечения, выключенный флаг.

from __future__ import annotations

import numpy as np
import pytest

from core.axis import AxisEstimate
from core.config import Config, ConfigError
from core.walls import extend_by_walls

RAILHEAD = 0.23


def _wall(y_of_x, x_from=5.0, x_to=62.0, z_from=0.8, z_to=2.7):
    x, z = np.meshgrid(np.arange(x_from, x_to, 0.1), np.arange(z_from, z_to, 0.1), indexing="ij")
    x, z = x.ravel(), z.ravel()
    return np.column_stack([x, y_of_x(x), z])


def _measured_axis(cfg: Config) -> AxisEstimate:
    rails = np.column_stack([np.arange(6.25, 25.0, 2.5), np.zeros(8)])
    return AxisEstimate(
        slope=0.0, y0_m=0.0, x_start_m=6.25, x_end_m=23.75, x_traced_m=23.75, residual_max_m=0.0,
        cross_check_m=0.0, meets_clearance=True, n_slices=8, method="rails", source="measured",
        staleness_frames=0, sigma_y_1m=0.001, clearance_m=cfg.axis.clearance_m,
        polyline_step_m=cfg.axis.polyline_step_m, rail_centres=rails)


def _curve(x):
    return np.where(x > 25.0, (x - 25.0) ** 2 / 600.0, 0.0)


def test_two_walls_extend_the_axis_along_a_curve(cfg: Config) -> None:
    scene = np.vstack([_wall(lambda x: 2.0 + _curve(x)), _wall(lambda x: -2.0 + _curve(x))])
    axis = extend_by_walls(_measured_axis(cfg), scene, cfg.axis.walls, RAILHEAD)
    assert axis.walls_info["used"]
    assert axis.x_end_m >= 50.0
    for x in (35.0, 45.0, 50.0):
        assert float(axis.y_at(x)) == pytest.approx(float(_curve(np.array(x))), abs=0.10)
    sources = axis.to_debug()["polyline"]["source"]
    assert "walls" in sources and sources[0] == "rails"


def test_one_wall_does_not_extend(cfg: Config) -> None:
    platform = _wall(lambda x: np.full_like(x, -1.47), z_from=RAILHEAD + 0.5, z_to=RAILHEAD + 1.3)
    axis = extend_by_walls(_measured_axis(cfg), platform, cfg.axis.walls, RAILHEAD)
    assert not axis.walls_info["used"]
    assert axis.walls_info["stop"] == "one_wall"
    assert axis.x_end_m == pytest.approx(23.75)


def test_width_change_stops_the_extension(cfg: Config) -> None:
    left = _wall(lambda x: np.where(x < 35.0, 2.03, 2.23))
    right = _wall(lambda x: np.where(x < 35.0, -2.03, -2.23))
    axis = extend_by_walls(_measured_axis(cfg), np.vstack([left, right]), cfg.axis.walls, RAILHEAD)
    assert axis.walls_info["stop"] == "width"
    assert axis.walls_info["reach_two_m"] < 36.0
    assert axis.x_end_m < 36.0


def test_held_axis_is_not_extended(cfg: Config) -> None:
    scene = np.vstack([_wall(lambda x: np.full_like(x, 2.0)), _wall(lambda x: np.full_like(x, -2.0))])
    held = AxisEstimate(**{**_measured_axis(cfg).__dict__, "source": "stale"})
    axis = extend_by_walls(held, scene, cfg.axis.walls, RAILHEAD)
    assert not axis.walls_info["used"]
    assert axis.x_end_m == pytest.approx(23.75)


def test_disabled_walls_leave_the_debug_as_before(raw_config: dict) -> None:
    from core.pipeline import ObstacleDetector
    from tests.synthetic import source_frame

    raw = {**raw_config, "axis": {**raw_config["axis"], "walls": {**raw_config["axis"]["walls"], "enabled": False}}}
    frame = source_frame()
    result = ObstacleDetector(Config.from_dict(raw)).process(
        frame["xyz"], frame["intensity"], frame["ring"], frame["timestamp"], 0)
    axis = result.debug.get("axis") or {}
    assert "walls" not in axis
    assert "walls" not in ((axis.get("polyline") or {}).get("source") or [])


def test_walls_section_is_required(raw_config: dict) -> None:
    axis = {k: v for k, v in raw_config["axis"].items() if k != "walls"}
    with pytest.raises(ConfigError):
        Config.from_dict({**raw_config, "axis": axis})


def test_walls_need_the_rail_head_height(raw_config: dict) -> None:
    gauge = {**raw_config["gauge"], "rail_head_offset_m": None, "train": {**raw_config["gauge"]["train"], "enabled": False}}
    detector = {**raw_config["detector"], "floor_above_rail_head_m": None}
    walls_off = {**raw_config["axis"], "walls": {**raw_config["axis"]["walls"], "enabled": False}}
    Config.from_dict({**raw_config, "gauge": gauge, "detector": detector, "axis": walls_off})
    with pytest.raises(ConfigError, match="axis.walls"):
        Config.from_dict({**raw_config, "gauge": gauge, "detector": detector})
