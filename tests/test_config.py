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

