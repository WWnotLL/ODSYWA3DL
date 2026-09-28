# Тесты загрузки конфига: обязательные ключи, запреты, проверки значений.

from __future__ import annotations

import copy

import numpy as np
import pytest

from core.config import Config, ConfigError


def test_default_config_loads(cfg):
    assert cfg.data.records and cfg.bag.message_type.endswith("PointCloud2")
    assert all(t.startswith("/") for t in cfg.bag.preferred_topics)


def test_rotation_in_default_config_is_proper(cfg):
    assert np.linalg.det(cfg.preprocess.axes.rotation) == pytest.approx(1.0)


def test_clean_records_exclude_the_obstacle_one(cfg):
    assert cfg.data.obstacle_record not in cfg.data.clean_records
    assert len(cfg.data.clean_records) == len(cfg.data.records) - 1


def test_missing_key_is_an_error(raw_config):
    broken = copy.deepcopy(raw_config)
    del broken["preprocess"]["ground"]["seed"]
    with pytest.raises(ConfigError, match="seed"):
        Config.from_dict(broken)


def test_unknown_key_is_an_error(raw_config):
    broken = copy.deepcopy(raw_config)
    broken["preprocess"]["ground"]["max_tilt_degrees"] = 25.0
    with pytest.raises(ConfigError, match="неизвестные ключи"):
        Config.from_dict(broken)


def test_wrong_type_is_an_error(raw_config):
    broken = copy.deepcopy(raw_config)
    broken["preprocess"]["ground"]["ransac_iterations"] = "много"
    with pytest.raises(ConfigError, match="ожидалось целое"):
        Config.from_dict(broken)


def test_unknown_dual_return_policy_is_an_error(raw_config):
    broken = copy.deepcopy(raw_config)
    broken["preprocess"]["dual_return"]["keep"] = "both"
    with pytest.raises(ConfigError, match="вне списка"):
        Config.from_dict(broken)


def test_rail_head_offset_is_measured(cfg):
    assert cfg.gauge.rail_head_offset_m == pytest.approx(0.230, abs=0.03)


def test_track_axis_is_not_a_constant(cfg):
    assert cfg.gauge.y_center_offset_m is None
    assert cfg.gauge.axis_slope is None


def test_detector_must_not_use_intensity(cfg):
    assert cfg.detector.use_intensity is False


def test_enabling_intensity_is_rejected(raw_config):
    import copy
    broken = copy.deepcopy(raw_config)
    broken["detector"]["use_intensity"] = True
    with pytest.raises(ConfigError, match="интенсивность"):
        Config.from_dict(broken)


def test_intensity_flag_must_be_boolean(raw_config):
    import copy
    broken = copy.deepcopy(raw_config)
    broken["detector"]["use_intensity"] = "false"
    with pytest.raises(ConfigError, match="true или false"):
        Config.from_dict(broken)


def test_bad_rotation_shape_is_an_error(raw_config):
    broken = copy.deepcopy(raw_config)
    broken["preprocess"]["axes"]["rotation"] = [[1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(ConfigError, match="матрица 3x3"):
        Config.from_dict(broken)


def test_obstacle_record_must_be_listed(raw_config):
    broken = copy.deepcopy(raw_config)
    broken["data"]["obstacle_record"] = "нет_такой_записи"
    with pytest.raises(ConfigError, match="obstacle_record"):
        Config.from_dict(broken)


def test_zone_bounds_must_be_ordered(raw_config):
    broken = copy.deepcopy(raw_config)
    broken["preprocess"]["ground"]["zone_min_m"] = 99.0
    with pytest.raises(ConfigError, match="zone_min_m"):
        Config.from_dict(broken)


def test_gauge_floor_comes_only_from_the_profile(cfg: Config, raw_config: dict) -> None:
    assert "z_min_m" not in raw_config["gauge"], "пол снова задан вторым местом"
    assert cfg.gauge.z_min_m == pytest.approx(float(cfg.gauge.profile_height_m.min()))
    assert cfg.gauge.z_min_m == pytest.approx(0.025), "нижняя вершина по рис. 7"



def test_depth_threshold_can_be_switched_off(raw_config):
    off = copy.deepcopy(raw_config)
    off["gauge"]["train"]["enabled"] = False
    off["detector"]["min_depth_m"] = None
    assert Config.from_dict(off).detector.min_depth_m is None
    off["detector"]["min_depth_m"] = -0.01
    with pytest.raises(ConfigError, match="min_depth_m"):
        Config.from_dict(off)


def test_direction_tolerance_must_be_positive(raw_config):
    broken = copy.deepcopy(raw_config)
    broken["preprocess"]["dual_return"]["direction_tolerance"] = 0.0
    with pytest.raises(ConfigError, match="direction_tolerance"):
        Config.from_dict(broken)


def test_plane_guard_is_on_by_default_and_validated(cfg, raw_config):
    guard = cfg.preprocess.plane_guard
    assert (guard.refit_limits_enabled and guard.plausibility_enabled
            and guard.hold_enabled and guard.rail_check_enabled)
    assert cfg.confirm.verdict.latch_through_unchecked is True
    broken = copy.deepcopy(raw_config)
    broken["preprocess"]["plane_guard"]["rail_check"]["head_min_m"] = 0.5
    with pytest.raises(ConfigError, match="head_min_m"):
        Config.from_dict(broken)


def test_plausibility_without_hold_and_latch_is_rejected(raw_config):
    broken = copy.deepcopy(raw_config)
    broken["preprocess"]["plane_guard"]["hold"]["enabled"] = False
    broken["confirm"]["verdict"]["latch_through_unchecked"] = False
    with pytest.raises(ConfigError, match="plausibility"):
        Config.from_dict(broken)
    broken["preprocess"]["plane_guard"]["hold"]["enabled"] = True
    with pytest.raises(ConfigError, match="plausibility"):
        Config.from_dict(broken)
    broken["confirm"]["verdict"]["latch_through_unchecked"] = True
    assert Config.from_dict(broken).preprocess.plane_guard.plausibility_enabled


def test_gauge_heights_can_be_counted_from_the_base(cfg, raw_config):
    assert cfg.gauge.height_reference == "rail_head"
    from_base = copy.deepcopy(raw_config)
    from_base["gauge"]["height_reference"] = "base"
    gauge = Config.from_dict(from_base).gauge
    lift = cfg.gauge.rail_head_offset_m
    assert gauge.profile_height_m == pytest.approx(cfg.gauge.profile_height_m - lift)
    assert gauge.z_max_m == pytest.approx(cfg.gauge.z_max_m - lift)
    broken = copy.deepcopy(raw_config)
    broken["gauge"]["height_reference"] = "sleeper"
    with pytest.raises(ConfigError, match="height_reference"):
        Config.from_dict(broken)


def test_floor_above_rail_head_needs_the_rail_head(raw_config):
    broken = copy.deepcopy(raw_config)
    broken["gauge"]["train"]["enabled"] = False
    broken["detector"]["floor_above_rail_head_m"] = 0.02
    broken["gauge"]["rail_head_offset_m"] = None
    with pytest.raises(ConfigError, match="floor_above_rail_head_m"):
        Config.from_dict(broken)


def test_train_gauge_needs_a_depth_threshold(cfg, raw_config):
    assert cfg.gauge.train_enabled is True and cfg.gauge.rail_mask_follow_heads is False
    broken = copy.deepcopy(raw_config)
    broken["gauge"]["train"]["enabled"] = True
    broken["detector"]["min_depth_m"] = None
    with pytest.raises(ConfigError, match="gauge.train"):
        Config.from_dict(broken)
    broken["detector"]["min_depth_m"] = 0.0
    assert Config.from_dict(broken).gauge.train_enabled
