# Тесты контракта выхода и его сериализации.

from __future__ import annotations

import numpy as np
import pytest

from core.types import Detection, FrameResult, to_jsonable


def _detection(**overrides) -> Detection:
    base = dict(
        distance_m=42.125,
        centroid_xyz=(42.0, 0.25, 1.5),
        bbox_min_xyz=(41.0, -0.5, 0.2),
        bbox_max_xyz=(43.0, 1.0, 2.8),
        n_points=137,
        score=0.83,
        source="gauge",
    )
    base.update(overrides)
    return Detection(**base)


def test_detection_is_frozen():
    detection = _detection()
    with pytest.raises(Exception):
        detection.distance_m = 1.0


def test_detection_normalises_sequences_to_tuples():
    detection = _detection(centroid_xyz=[1, 2, 3])
    assert detection.centroid_xyz == (1.0, 2.0, 3.0)


def test_detection_rejects_unknown_source():
    with pytest.raises(ValueError, match="вне контракта"):
        _detection(source="magic")


def test_detection_rejects_wrong_dimension():
    with pytest.raises(ValueError, match="три координаты"):
        _detection(centroid_xyz=(1.0, 2.0))


def test_frame_result_round_trip_through_json():
    result = FrameResult(
        obstacle_found=True,
        nearest_distance_m=42.125,
        detections=[_detection(), _detection(source="background", score=0.5)],
        debug={"stage_ms": {"preprocess": 12.5}, "plane_up_source": "mass_ratio"},
    )
    assert FrameResult.from_json(result.to_json()) == result


def test_frame_result_round_trip_without_detections():
    result = FrameResult(obstacle_found=False, nearest_distance_m=None, detections=[], debug={})
    restored = FrameResult.from_json(result.to_json())
    assert restored == result and restored.nearest_distance_m is None


def test_frame_result_preserves_float_precision():
    value = 123.45678901234567
    result = FrameResult(True, value, [], {})
    assert FrameResult.from_json(result.to_json()).nearest_distance_m == value


def test_debug_accepts_numpy_values():
    debug = {"normal": np.array([0.0, 0.0, 1.0]), "count": np.int64(7), "ratio": np.float64(3.5)}
    payload = to_jsonable(debug)
    assert payload == {"normal": [0.0, 0.0, 1.0], "count": 7, "ratio": 3.5}
    FrameResult(False, None, [], debug).to_json()

