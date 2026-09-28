# Эпизод и его путь: одно определение для таблицы эпизодов и скриптов замера.

from __future__ import annotations

from tools.episode_table import episode_path, episodes, filled_advances, group_by_gap


def _row(frame: int, advance: float | None, alarm: bool, stamp_s: float | None = None) -> dict:
    stamp = frame * 0.1 if stamp_s is None else stamp_s
    return {"frame": frame, "stamp_ns": int(round(stamp * 1e9)), "obstacle_found": alarm,
            "nearest_distance_m": 10.0 if alarm else None, "debug": {"advance_m": advance}}


def test_path_counts_movement_after_the_first_frame_of_the_episode():
    rows = [_row(0, 1.0, False), _row(1, 1.0, True), _row(2, 1.0, True), _row(3, None, True), _row(4, 1.0, False)]
    ep = episodes(rows, fill=True)
    assert [e["frames"] for e in ep] == [(1, 3)]
    assert ep[0]["path_m"] == 1.0
    assert ep[0]["unknown_advance"] == 1
    assert ep[0]["path_filled_m"] == 2.0


def test_gap_is_measured_in_time_not_frames():
    rows = [_row(0, 1.0, True, 0.0), _row(1, 1.0, True, 1.0), _row(2, 1.0, True, 2.2)]
    assert [len(g) for g in group_by_gap([r for r in rows if r["obstacle_found"]])] == [2, 1]


def test_custom_flag_and_filled_advance_use_nearby_median():
    rows = [_row(k, None if k == 5 else 0.5 + 0.1 * (k % 2), k in (5, 6)) for k in range(12)]
    filled = filled_advances(rows)
    assert filled[5] == 0.5
    ep = episodes(rows, lambda f: f["frame"] >= 10)
    assert [e["frames"] for e in ep] == [(10, 11)]
    assert episode_path(rows, 4, 6, filled)["path_filled_m"] == 1.0
