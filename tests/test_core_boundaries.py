# Тесты границ ядра: никаких импортов ROS и визуализации, запрет интенсивности.

from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest

from core.config import Config, ConfigError

CORE = Path(__file__).resolve().parents[1] / "core"


_STATIC_SUPPRESSION = {"split_static", "is_static", "drop_static", "static_only",
                       "suppress_static", "stationary_filter"}


def _core_modules() -> list[Path]:
    return sorted(p for p in CORE.glob("*.py") if p.name != "__init__.py")


def test_core_does_not_import_ros() -> None:
    for path in _core_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                root = name.split(".")[0]
                assert root not in {"rclpy", "rospy", "sensor_msgs", "std_msgs", "tf2_ros"}, (
                    f"{path.name}: ядро не должно зависеть от ROS, найдено {name!r}"
                )


def test_core_does_not_import_heavy_optional_packages() -> None:
    for path in _core_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module] if isinstance(node, ast.ImportFrom) and node.module
                     else [])
            for name in names:
                assert name.split(".")[0] not in {"sklearn", "open3d", "matplotlib", "rosbags"}, (
                    f"{path.name}: {name!r} не входит в зависимости ядра"
                )


def test_intensity_stays_out_of_the_detector(raw_config: dict) -> None:
    raw = copy.deepcopy(raw_config)
    raw["detector"]["use_intensity"] = True
    with pytest.raises(ConfigError):
        Config.from_dict(raw)


def test_config_still_forbids_intensity(cfg: Config) -> None:
    assert cfg.detector.use_intensity is False


def test_core_never_suppresses_static_objects() -> None:
    for path in _core_modules():
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                assert node.name not in _STATIC_SUPPRESSION, (
                    f"{path.name}: {node.name!r} — отсев по неподвижности в ядре недопустим"
                )
            if isinstance(node, ast.ImportFrom) and node.module and "oracle" in node.module:
                raise AssertionError(f"{path.name}: ядро не импортирует оракула разметки")
            if isinstance(node, ast.Attribute) and node.attr in _STATIC_SUPPRESSION:
                raise AssertionError(f"{path.name}: обращение к {node.attr!r} в ядре недопустимо")


def test_oracle_is_allowed_to_do_it() -> None:
    oracle = (CORE.parent / "tools" / "oracle.py").read_text(encoding="utf-8")
    assert "split_static" in oracle

