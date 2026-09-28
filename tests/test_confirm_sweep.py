# Тесты переигрывания подтверждения по выходу ядра.

from __future__ import annotations

from tools.confirm_sweep import replay, replay_positive


def _row(frame: int, x: float | None, *, body_x: float | None = None, run: int = 1,
         checked: bool = True, advance: float | None = 1.7) -> dict:
    detections = [] if x is None else [{
        "distance_m": x, "centroid_xyz": [x, 0.0, 0.5],
        "bbox_min_xyz": [x - 0.1, -0.1, 0.4], "bbox_max_xyz": [x + 0.1, 0.1, 0.6],
        "n_points": 100, "score": 1.0, "source": "gauge",
    }]
    box = None if body_x is None else [[body_x - 0.25, -0.25, 0.23], [body_x + 0.25, 0.25, 1.93]]
    return {"frame": frame, "stamp_ns": frame * 100_000_000, "run": run,
            "body_centre_x_m": body_x, "body_box": box, "checked": checked,
            "advance_m": advance,
            "candidates_raw": {"detections": detections,
                               "lateral_offset_m": [0.0] * len(detections)}}


def test_positive_replay_counts_frames_runs_and_first_distance(cfg) -> None:
    xs = [20.0, 18.3, None, 14.9, 13.2]
    rows = [_row(i, x, body_x=20.0 - 1.7 * i) for i, x in enumerate(xs)]
    out = replay_positive(rows, cfg, required=3, window=5, tolerance=0.6)

    assert out == {"frames": 5, "hit_frames": 2, "runs": 1, "runs_found": 1,
                   "first_m": 20.0 - 1.7 * 3}


def test_unchecked_frames_are_not_fed_to_the_confirmer(cfg) -> None:
    rows = [_row(0, 20.0), _row(1, None, checked=False), _row(2, 18.3), _row(3, 16.6)]

    out = replay(rows, cfg, {r["frame"]: r["advance_m"] for r in rows}, 3, 5)
    assert out["raw_frames"] == 3

