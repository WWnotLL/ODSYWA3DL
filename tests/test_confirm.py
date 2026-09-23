# Тесты подтверждения во времени и правила В₂.

from __future__ import annotations

from dataclasses import replace

import pytest

from core.confirm import LATERAL_KEY, AxisLossVerdict, Confirmer
from core.config import Config, ConfigError
from core.types import Detection, FrameResult


def _frame(*clusters: tuple[float, float, float], state: str = "measured") -> FrameResult:
    detections = [
        Detection(
            distance_m=x,
            centroid_xyz=(x, y, z),
            bbox_min_xyz=(x - 0.1, y - 0.1, z - 0.1),
            bbox_max_xyz=(x + 0.1, y + 0.1, z + 0.1),
            n_points=100,
            score=1.0,
            source="gauge",
        )
        for x, y, z in clusters
    ]
    debug = {
        "layer": "A",
        "axis": {"state": state},
        LATERAL_KEY: [y for _, y, _ in clusters],
    }
    return FrameResult(bool(detections), min((d.distance_m for d in detections), default=None),
                       detections, debug)


def _confirmer(cfg: Config, **overrides) -> Confirmer:
    return Confirmer(replace(cfg.confirm, **overrides))


def test_single_frame_flash_is_not_confirmed(cfg: Config) -> None:
    confirmer = _confirmer(cfg, required_frames=3, window_frames=5)
    assert not confirmer.update(_frame((20.0, 0.0, 0.5))).obstacle_found
    assert not confirmer.update(_frame()).obstacle_found


def test_cluster_is_confirmed_on_the_k_th_frame(cfg: Config) -> None:
    confirmer = _confirmer(cfg, required_frames=3, window_frames=5)
    found = [confirmer.update(_frame((20.0, 0.0, 0.5))).obstacle_found for _ in range(3)]
    assert found == [False, False, True]
    assert confirmer.latencies == [2]


def test_confirmation_latches_through_a_missed_frame(cfg: Config) -> None:
    confirmer = _confirmer(cfg, required_frames=3, window_frames=5, max_missed_frames=2)
    for _ in range(3):
        confirmer.update(_frame((20.0, 0.0, 0.5)))
    confirmer.update(_frame())
    assert confirmer.update(_frame((20.0, 0.0, 0.5))).obstacle_found


def test_track_is_forgotten_after_a_long_gap(cfg: Config) -> None:
    confirmer = _confirmer(cfg, required_frames=2, window_frames=5, max_missed_frames=2)
    confirmer.update(_frame((20.0, 0.0, 0.5)))
    for _ in range(3):
        confirmer.update(_frame())

    assert not confirmer.update(_frame((20.0, 0.0, 0.5))).obstacle_found


def test_approaching_body_stays_one_track_when_advance_is_known(cfg: Config) -> None:
    confirmer = _confirmer(cfg, required_frames=3, window_frames=5)
    found = [
        confirmer.update(_frame((20.0 - 1.7 * i, 0.0, 0.5)), advance_m=1.7).obstacle_found
        for i in range(3)
    ]
    assert found[-1]


def test_approaching_body_survives_a_missed_frame_at_speed(cfg: Config) -> None:
    confirmer = _confirmer(cfg, required_frames=3, window_frames=5, max_missed_frames=2)
    xs = [20.0, 18.3, None, 14.9, 13.2]
    found = [confirmer.update(_frame() if x is None else _frame((x, 0.0, 0.5)),
                              advance_m=1.7).obstacle_found for x in xs]
    assert found == [False, False, False, True, True]
    assert confirmer.n_confirmed_tracks == 1


def test_unknown_advance_widens_the_gate_for_every_missed_frame(cfg: Config) -> None:
    confirmer = _confirmer(cfg, required_frames=2, window_frames=5, max_missed_frames=2)
    confirmer.update(_frame((20.0, 0.0, 0.5)), advance_m=1.7)
    confirmer.update(_frame(), advance_m=None)

    assert confirmer.update(_frame((16.6, 0.0, 0.5)), advance_m=None).obstacle_found


def test_body_bound_to_the_train_is_not_confirmed_at_speed(cfg: Config) -> None:
    confirmer = _confirmer(cfg, required_frames=3, window_frames=5)
    found = [confirmer.update(_frame((7.4, -1.8, 0.5)), advance_m=1.7).obstacle_found
             for _ in range(5)]
    assert not any(found)


def test_unknown_advance_widens_only_the_along_gate(cfg: Config) -> None:
    confirmer = _confirmer(cfg, required_frames=2, window_frames=5)
    confirmer.update(_frame((20.0, 0.0, 0.5)), advance_m=None)

    assert confirmer.update(_frame((18.3, 0.0, 0.5)), advance_m=None).obstacle_found

    other = _confirmer(cfg, required_frames=2, window_frames=5)
    other.update(_frame((20.0, 0.0, 0.5)), advance_m=None)

    assert not other.update(_frame((20.0, 4.2, 0.5)), advance_m=None).obstacle_found


def test_two_clusters_do_not_merge_into_one_track(cfg: Config) -> None:
    confirmer = _confirmer(cfg, required_frames=2, window_frames=5)
    confirmer.update(_frame((20.0, -0.9, 0.5), (20.0, 0.9, 0.5)))
    out = confirmer.update(_frame((20.0, -0.9, 0.5), (20.0, 0.9, 0.5)))
    assert len(out.detections) == 2
    assert out.debug["confirm"]["n_tracks"] == 2


def test_candidates_survive_in_debug(cfg: Config) -> None:
    confirmer = _confirmer(cfg, required_frames=3, window_frames=5)
    out = confirmer.update(_frame((20.0, 0.0, 0.5)))
    assert out.detections == []
    assert out.debug["confirm"]["n_candidates"] == 1
    assert out.debug["confirm"]["candidates"][0]["confirmed"] is False


def test_missing_lateral_offset_is_an_error_not_silence(cfg: Config) -> None:
    result = _frame((20.0, 0.0, 0.5))
    del result.debug[LATERAL_KEY]
    with pytest.raises(ValueError, match="lateral_offset_m"):
        _confirmer(cfg).update(result)


def test_reset_clears_the_history(cfg: Config) -> None:
    confirmer = _confirmer(cfg, required_frames=2, window_frames=5)
    confirmer.update(_frame((20.0, 0.0, 0.5)))
    confirmer.reset()
    assert not confirmer.update(_frame((20.0, 0.0, 0.5))).obstacle_found


def test_v2_stops_when_the_axis_fails_after_clusters(cfg: Config) -> None:
    verdict = AxisLossVerdict(cfg.confirm.verdict)
    verdict.update(_frame((20.0, 0.0, 0.5), state="measured"), had_candidate=True)
    out = verdict.update(_frame(state="lost"), had_candidate=False)
    assert out["stop"] is True and out["reason"] == "axis_lost_after_clusters"
    assert verdict.n_triggers == 1


def test_v2_is_silent_when_the_corridor_was_empty(cfg: Config) -> None:
    verdict = AxisLossVerdict(cfg.confirm.verdict)
    verdict.update(_frame(state="measured"), had_candidate=False)
    out = verdict.update(_frame(state="lost"), had_candidate=False)
    assert out["stop"] is False and out["reason"] == "axis_lost_without_clusters"
    assert verdict.n_triggers == 0


def test_v2_does_not_fire_on_a_cold_start(cfg: Config) -> None:
    verdict = AxisLossVerdict(cfg.confirm.verdict)
    out = verdict.update(_frame(state="lost"), had_candidate=True)
    assert out["stop"] is False and out["reason"] == "cold_start"
    assert verdict.n_triggers == 0


def test_v2_counts_an_unconfirmed_cluster(cfg: Config) -> None:
    confirmer = _confirmer(cfg, required_frames=3, window_frames=5)
    verdict = AxisLossVerdict(cfg.confirm.verdict)
    frame = _frame((20.0, 0.0, 0.5), state="measured")
    raw = bool(frame.detections)
    out = confirmer.update(frame)
    assert out.obstacle_found is False
    verdict.update(out, had_candidate=raw)
    assert verdict.update(_frame(state="lost"), had_candidate=False)["stop"] is True


def test_v2_holds_the_stop_until_the_axis_returns(cfg: Config) -> None:
    verdict = AxisLossVerdict(cfg.confirm.verdict)
    verdict.update(_frame((20.0, 0.0, 0.5), state="measured"), had_candidate=True)
    verdict.update(_frame(state="lost"), had_candidate=False)
    assert verdict.update(_frame(state="lost"), had_candidate=False)["stop"] is True
    assert verdict.n_triggers == 1
    assert verdict.update(_frame(state="measured"), had_candidate=False)["stop"] is False


def test_required_above_window_is_a_load_error(raw_config: dict) -> None:
    broken = {**raw_config, "confirm": {**raw_config["confirm"],
                                        "required_frames": 9, "window_frames": 5}}
    with pytest.raises(ConfigError, match="required_frames"):
        Config.from_dict(broken)


def test_per_cluster_debug_shrinks_with_the_detections(cfg: Config) -> None:
    confirmer = _confirmer(cfg, required_frames=2, window_frames=5)
    near, far = (12.0, 0.0, 0.5), (22.0, 0.4, 0.6)
    for _ in range(2):
        confirmer.update(_frame(near, far))
    confirmer.update(_frame(near, far))

    out = confirmer.update(_frame(near))
    assert len(out.detections) == 1
    assert len(out.debug[LATERAL_KEY]) == 1
    assert out.debug["confirm"]["candidate_lateral_offset_m"] == [0.0]


def test_a_confirmed_row_can_be_replayed(cfg: Config) -> None:
    first = Confirmer(cfg.confirm)
    second = Confirmer(cfg.confirm)
    for _ in range(4):
        second.update(first.update(_frame((20.0, 0.0, 0.5), (14.0, 0.7, 0.6))))

