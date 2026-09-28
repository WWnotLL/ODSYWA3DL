# Тесты разбивки кандидатов, снятых отказом от продления оси.

from __future__ import annotations

import numpy as np

from tools.extension_split import follow, split


def _det(x0: float, x1: float, y: float = -1.45) -> dict:
    return {"distance_m": x0, "centroid_xyz": [(x0 + x1) / 2, y, 1.0],
            "bbox_min_xyz": [x0, y - 0.1, 0.8], "bbox_max_xyz": [x1, y + 0.1, 1.3],
            "n_points": 20, "score": 1.0, "source": "gauge"}


def _row(frame: int, dets: list[dict], end: float, advance: float | None = 1.7) -> dict:
    return {"frame": frame, "stamp_ns": frame * 100_000_000, "obstacle_found": False,
            "debug": {"corridor": {"x_m": [5.0, end]}, "advance_m": advance,
                      "axis": {"state": "measured",
                               "polyline": {"points_xy_m": [[5.0, 0.0], [40.0, 0.0]],
                                            "source": ["rails", "bed"]}},
                      "candidates_raw": {"detections": dets,
                                         "lateral_offset_m": [-1.45] * len(dets)}}}


def test_split_classifies_by_the_end_of_the_base_without_extension() -> None:
    with_rows = [_row(0, [_det(30.0, 31.0), _det(23.0, 25.0), _det(15.0, 16.0)], end=40.0)]
    without_rows = [_row(0, [], end=24.0)]
    groups = sorted(r["group"] for r in split(with_rows, without_rows, tolerance=0.3))
    assert groups == ["а", "б", "граница"]


def test_follow_checks_the_place_once_it_is_inside_the_rails_base() -> None:
    rows = [_row(g, [], end=24.0) for g in range(1, 12)]
    rows[5]["debug"]["candidates_raw"] = {"detections": [_det(21.0, 22.5)],
                                          "lateral_offset_m": [-1.45]}
    verdict = follow(rows, 0, np.array([30.0, -1.55, 0.8]), np.array([31.0, -1.35, 1.3]),
                     min_range=5.0, tolerance=1.0, max_frames=20, inside_m=1.0,
                     check_frames=3)
    assert verdict.startswith("ЕСТЬ: 1/3")


def test_follow_stops_on_unknown_advance() -> None:
    rows = [_row(1, [], end=24.0), _row(2, [], end=24.0, advance=None)]
    verdict = follow(rows, 0, np.array([30.0, -1.55, 0.8]), np.array([31.0, -1.35, 1.3]),
                     min_range=5.0, tolerance=1.0, max_frames=20, inside_m=1.0,
                     check_frames=3)
    assert verdict == "путь неизвестен в кадре 2"

