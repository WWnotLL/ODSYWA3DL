# Тесты перевода выхода ядра в систему лидара для 3D-просмотра.

from __future__ import annotations

import numpy as np

from core.config import Config
from core.pipeline import ObstacleDetector
from core.preprocess import FrameTransform
from tests.test_pipeline import _frames
from tools.view_geometry import frame_lines

COLORS = {"candidate": (1, 1, 0), "alarm": (1, 0, 0), "axis": (0, 0, 1), "corridor": (0, 1, 0)}


def _last_result(cfg: Config) -> tuple[dict, np.ndarray]:
    detector = ObstacleDetector(cfg, confirm=False)
    for frame in _frames(cfg, 3, with_body=True):
        result = detector.process(*frame)
    return result.to_dict(), frame[0]


def test_boxes_go_back_to_exactly_the_core_boxes(cfg: Config) -> None:
    result, _ = _last_result(cfg)
    info = result["debug"]["frame_transform"]
    transform = FrameTransform(np.asarray(info["rotation"]), np.asarray(info["translation_m"]))
    boxes = [g for g in frame_lines(result, cfg.gauge, COLORS) if g.name == "тревоги"]
    assert boxes and result["detections"]
    back = transform.apply(boxes[0].points[:8])
    det = result["detections"][0]
    assert np.allclose(back.min(axis=0), det["bbox_min_xyz"], atol=1e-9)
    assert np.allclose(back.max(axis=0), det["bbox_max_xyz"], atol=1e-9)


def test_box_in_lidar_frame_holds_its_raw_points(cfg: Config) -> None:
    result, raw = _last_result(cfg)
    info = result["debug"]["frame_transform"]
    transform = FrameTransform(np.asarray(info["rotation"]), np.asarray(info["translation_m"]))
    det = result["detections"][0]
    track = transform.apply(raw)
    inside = np.all((track >= np.asarray(det["bbox_min_xyz"]) - 1e-6)
                    & (track <= np.asarray(det["bbox_max_xyz"]) + 1e-6), axis=1)
    assert inside.sum() >= det["n_points"]


def test_every_overlay_is_drawn_for_a_checked_frame(cfg: Config) -> None:
    result, _ = _last_result(cfg)
    names = {g.name for g in frame_lines(result, cfg.gauge, COLORS)}
    assert {"кандидаты", "тревоги", "ось", "коридор", "Ом"} <= names


def test_unchecked_frame_gives_no_overlay(cfg: Config) -> None:
    unchecked = {"debug": {"checked": False}, "detections": []}
    assert frame_lines(unchecked, cfg.gauge, COLORS) == []

