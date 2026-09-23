# Тесты оценки оси пути: трассеры, продление, якорь, удержание.

from __future__ import annotations

import numpy as np
import pytest

from core.axis import AxisEstimate, AxisTracker, estimate_axis
from core.config import AxisMethodError, Config, ConfigError
from tests.synthetic import track_rep103


def test_bed_method_recovers_yaw(cfg: Config) -> None:
    bed_only = type(cfg.axis)(**{**cfg.axis.__dict__, "methods": ("bed",)})
    for yaw in (-2.5, -1.6, 0.0, 1.3):
        axis = estimate_axis(track_rep103(yaw_deg=yaw, y0=0.1), bed_only)
        assert axis is not None
        assert axis.method == "bed"
        assert axis.yaw_deg == pytest.approx(yaw, abs=0.05)
        assert axis.y0_m == pytest.approx(0.1, abs=0.02)


def test_rails_lead_and_bed_extends_the_base(cfg: Config) -> None:
    from core.axis import _extend_base


    axis = estimate_axis(track_rep103(yaw_deg=-1.6, y0=0.1), cfg.axis)
    assert axis is not None
    assert axis.method == "rails"
    assert axis.yaw_deg == pytest.approx(-1.6, abs=0.05)

    kept = np.column_stack([np.arange(5.0, 25.0, 2.5), np.zeros(8)])
    far_x = np.arange(5.0, 52.5, 2.5)
    rival = np.column_stack([far_x, np.full(far_x.size, 0.05)])
    extension = _extend_base(kept, 0.0, 0.0, {"bed": rival}, cfg.axis)
    assert extension is not None
    name, r_slope, r_y0, end, gap = extension
    assert name == "bed"
    assert end == pytest.approx(50.0)
    assert gap == pytest.approx(0.05, abs=0.01)


def test_extension_is_refused_when_the_second_tracer_disagrees(cfg: Config) -> None:
    from core.axis import _extend_base

    kept = np.column_stack([np.arange(5.0, 25.0, 2.5), np.zeros(8)])

    far_x = np.arange(5.0, 52.5, 2.5)
    rival = np.column_stack([far_x, 0.0085 * (far_x - 5.0)])
    assert _extend_base(kept, 0.0, 0.0, {"bed": rival}, cfg.axis) is None


def test_bed_method_refuses_double_track(cfg: Config) -> None:
    cloud = track_rep103(yaw_deg=-1.6, second_track_at=4.2)
    bed_only = type(cfg.axis)(**{**cfg.axis.__dict__, "methods": ("bed",)})
    assert estimate_axis(cloud, bed_only) is None


def test_rails_method_takes_over_on_double_track(cfg: Config) -> None:
    axis = estimate_axis(track_rep103(yaw_deg=-1.6, second_track_at=4.2), cfg.axis)
    assert axis is not None
    assert axis.method == "rails"
    assert axis.yaw_deg == pytest.approx(-1.6, abs=0.05)


def test_unknown_method_rejected(cfg: Config) -> None:
    broken = type(cfg.axis)(**{**cfg.axis.__dict__, "methods": ("lidar_magic",)})
    with pytest.raises(AxisMethodError):
        estimate_axis(track_rep103(), broken)


def test_unknown_method_rejected_by_config(raw_config: dict) -> None:
    import copy

    raw = copy.deepcopy(raw_config)
    raw["axis"]["methods"] = ["bed", "telepathy"]
    with pytest.raises(AxisMethodError):
        Config.from_dict(raw)


def test_width_bounds_must_be_ordered(raw_config: dict) -> None:
    import copy

    raw = copy.deepcopy(raw_config)
    raw["axis"]["bed"]["width_min_m"] = 5.0
    with pytest.raises(ConfigError):
        Config.from_dict(raw)


def test_offset_is_measured_from_the_axis_not_from_zero(cfg: Config) -> None:
    axis = estimate_axis(track_rep103(yaw_deg=-1.6), cfg.axis)
    point = np.array([[55.0, -1.5, 1.0]])
    assert abs(axis.offset(point)[0]) < 0.2
    assert abs(point[0, 1]) > 1.4


def test_sigma_grows_with_range_and_staleness(cfg: Config) -> None:
    fresh = estimate_axis(track_rep103(), cfg.axis)
    assert fresh.sigma_y(50.0) > fresh.sigma_y(20.0)
    stale = AxisEstimate(**{**fresh.__dict__, "source": "stale",
                            "staleness_frames": 3, "sigma_y_1m": 3 * fresh.sigma_y_1m})
    assert stale.sigma_y(50.0) > fresh.sigma_y(50.0)


class _Feeder:
    def __init__(self, clouds: list[np.ndarray | None]) -> None:
        self.clouds = clouds

    def run(self, tracker: AxisTracker) -> list[AxisEstimate | None]:
        empty = np.zeros((0, 3))
        return [tracker.update(c if c is not None else empty) for c in self.clouds]


def test_tracker_bridges_a_gap(cfg: Config) -> None:
    good = track_rep103(yaw_deg=-1.0)
    out = _Feeder([good, None, None, good]).run(AxisTracker(cfg.axis))
    assert out[0].source == "measured"
    assert [o.source for o in out[1:3]] == ["stale", "stale"]
    assert out[1].staleness_frames == 1 and out[2].staleness_frames == 2
    assert out[3].source == "measured"

    assert out[2].sigma_y_1m > out[0].sigma_y_1m
    assert out[2].slope == out[0].slope


def test_tracker_gives_up_after_the_limit(cfg: Config) -> None:
    good = track_rep103(yaw_deg=-1.0)
    gaps = [None] * (cfg.axis.max_staleness_frames + 1)
    out = _Feeder([good, *gaps]).run(AxisTracker(cfg.axis))
    assert out[-1] is None, "после исчерпания лимита ось обязана стать None, а не остаться старой"


def test_tracker_rejects_an_outlier_jump(cfg: Config) -> None:
    tracker = AxisTracker(cfg.axis)
    tracker.update(track_rep103(yaw_deg=-1.0))
    out = tracker.update(track_rep103(yaw_deg=-1.0 + 10.0 * cfg.axis.max_slope_step_deg))
    assert out.source == "stale"
    assert tracker.n_rejected == 1
    assert out.yaw_deg == pytest.approx(-1.0, abs=0.05)


def test_small_change_is_not_rejected(cfg: Config) -> None:
    tracker = AxisTracker(cfg.axis)
    tracker.update(track_rep103(yaw_deg=-1.0))
    out = tracker.update(track_rep103(yaw_deg=-1.0 + 0.5 * cfg.axis.max_slope_step_deg))
    assert out.source == "measured"
    assert tracker.n_rejected == 0


def test_frame_gap_scales_the_threshold(cfg: Config) -> None:
    step = 3.0 * cfg.axis.max_slope_step_deg
    tight = AxisTracker(cfg.axis)
    tight.update(track_rep103(yaw_deg=-1.0))
    assert tight.update(track_rep103(yaw_deg=-1.0 + step)).source == "stale"

    wide = AxisTracker(cfg.axis)
    wide.update(track_rep103(yaw_deg=-1.0))
    assert wide.update(track_rep103(yaw_deg=-1.0 + step), frame_gap=5).source == "measured"


def test_frame_gap_must_be_positive(cfg: Config) -> None:
    tracker = AxisTracker(cfg.axis)
    with pytest.raises(ValueError):
        tracker.update(track_rep103(), frame_gap=0)


def test_traced_base_is_reported(cfg: Config) -> None:
    short = estimate_axis(track_rep103(x_max=30.0), cfg.axis)
    long = estimate_axis(track_rep103(x_max=60.0), cfg.axis)
    assert short.x_end_m < long.x_end_m
    assert short.x_end_m <= 30.0 and long.x_end_m <= 60.0


def test_debug_block_matches_the_agreed_schema(cfg: Config) -> None:
    axis = estimate_axis(track_rep103(), cfg.axis)
    block = axis.to_debug()
    assert set(block) >= {"source", "staleness_frames", "slope_deg", "y_center_m", "sigma_y_1m"}
    assert block["source"] in {"measured", "stale"}


def test_base_is_trimmed_where_a_straight_line_stops_fitting(cfg: Config) -> None:
    straight = estimate_axis(track_rep103(yaw_deg=-1.0), cfg.axis)
    curved = estimate_axis(track_rep103(yaw_deg=-1.0, radius_m=400.0), cfg.axis)
    assert straight is not None and curved is not None
    assert curved.x_end_m < straight.x_end_m
    assert curved.residual_max_m <= cfg.axis.clearance_m

    assert curved.x_traced_m == pytest.approx(straight.x_traced_m, abs=0.5)


def test_a_gentle_curve_does_not_cost_base(cfg: Config) -> None:
    straight = estimate_axis(track_rep103(yaw_deg=-1.0), cfg.axis)
    gentle = estimate_axis(track_rep103(yaw_deg=-1.0, radius_m=2000.0), cfg.axis)
    assert gentle.x_end_m == pytest.approx(straight.x_end_m, abs=0.5)


def test_residual_stays_within_the_clearance(cfg: Config) -> None:
    for radius in (300.0, 500.0, 1000.0, None):
        axis = estimate_axis(track_rep103(yaw_deg=-1.0, radius_m=radius), cfg.axis)
        assert axis is not None
        assert axis.residual_max_m <= cfg.axis.clearance_m, f"R={radius}"


def test_staleness_shortens_range_not_width(cfg: Config) -> None:
    tracker = AxisTracker(cfg.axis)
    fresh = tracker.update(track_rep103(yaw_deg=-1.0))
    stale = tracker.update(np.zeros((0, 3)))
    assert stale.source == "stale"
    assert stale.x_end_m < fresh.x_end_m
    assert stale.slope == fresh.slope and stale.y0_m == fresh.y0_m
    assert stale.x_end_m * stale.sigma_y_1m <= cfg.axis.clearance_m + 1e-9


def test_debug_separates_traced_from_trusted(cfg: Config) -> None:
    block = estimate_axis(track_rep103(yaw_deg=-1.0, radius_m=400.0), cfg.axis).to_debug()
    assert block["x_trusted_m"] <= block["x_traced_m"][1]
    assert "residual_max_m" in block


def test_cross_check_catches_a_shifted_axis(cfg: Config) -> None:
    axis = estimate_axis(track_rep103(yaw_deg=-1.0), cfg.axis)
    assert axis is not None
    assert axis.cross_check_m is not None

    assert axis.cross_check_m < cfg.axis.clearance_m


def test_sigma_edge_includes_all_three_terms(cfg: Config) -> None:
    prior = cfg.axis.cross_check_prior_deg
    base = estimate_axis(track_rep103(yaw_deg=-1.0), cfg.axis)
    plain = AxisEstimate(**{**base.__dict__, "residual_max_m": 0.0, "cross_check_m": 0.0})
    with_model = AxisEstimate(**{**plain.__dict__, "residual_max_m": 0.10})
    with_shift = AxisEstimate(**{**with_model.__dict__, "cross_check_m": 0.10})
    assert (plain.sigma_edge(20.0, prior) < with_model.sigma_edge(20.0, prior)
            < with_shift.sigma_edge(20.0, prior))

    assert plain.sigma_edge(50.0, prior) > plain.sigma_edge(20.0, prior)


def test_missing_cross_check_is_not_treated_as_a_passed_one(cfg: Config) -> None:
    prior = cfg.axis.cross_check_prior_deg
    base = estimate_axis(track_rep103(yaw_deg=-1.0), cfg.axis)
    unchecked = AxisEstimate(**{**base.__dict__, "cross_check_m": None})
    agreed = AxisEstimate(**{**base.__dict__, "cross_check_m": 0.0})
    typical = AxisEstimate(**{
        **base.__dict__, "cross_check_m": float(np.tan(np.radians(prior)) * 20.0)})

    assert unchecked.sigma_edge(20.0, prior) > agreed.sigma_edge(20.0, prior)
    assert unchecked.sigma_edge(20.0, prior) == pytest.approx(
        typical.sigma_edge(20.0, prior), rel=1e-6
    )


def test_clearance_flag_marks_frames_where_the_axis_fails_the_norm(cfg: Config) -> None:
    ok = estimate_axis(track_rep103(yaw_deg=-1.0), cfg.axis)
    assert ok.meets_clearance
    assert ok.residual_max_m <= cfg.axis.clearance_m

    hard = estimate_axis(track_rep103(yaw_deg=-1.0, radius_m=60.0), cfg.axis)
    assert hard is not None
    assert not hard.meets_clearance
    assert hard.n_slices == cfg.axis.min_slices


def test_cross_check_is_zero_when_there_is_nothing_to_compare(cfg: Config) -> None:
    single = type(cfg.axis)(**{**cfg.axis.__dict__, "methods": ("bed",)})
    axis = estimate_axis(track_rep103(yaw_deg=-1.0), single)
    assert axis.cross_check_m is None
    assert "cross_check_m" in axis.to_debug()


def test_cross_check_survives_when_rival_starts_before_the_chosen_base(cfg: Config) -> None:
    from core.axis import _cross_check

    kept = np.column_stack([np.arange(8.0, 28.0, 2.5), np.zeros(8)])

    rival = np.column_stack([np.arange(5.5, 20.5, 2.5), np.full(6, 0.4)])
    gap = _cross_check(kept, slope=0.0, y0=0.0, others=[rival], cfg=cfg.axis)
    assert gap is not None, "сверка исчезла, хотя отрезки перекрываются"
    assert gap == pytest.approx(0.4, abs=0.02)


def test_cross_check_is_none_when_ranges_barely_touch(cfg: Config) -> None:
    from core.axis import _cross_check

    kept = np.column_stack([np.arange(20.0, 40.0, 2.5), np.zeros(8)])
    rival = np.column_stack([np.arange(5.0, 21.0, 2.5), np.full(7, 0.4)])
    assert _cross_check(kept, slope=0.0, y0=0.0, others=[rival], cfg=cfg.axis) is None


def _straight(cfg: Config, y0: float = 0.0, yaw: float = 0.0):
    return track_rep103(yaw_deg=yaw, y0=y0)


def test_anchor_vetoes_an_axis_that_jumped_to_the_neighbouring_track(cfg: Config) -> None:
    tracker = AxisTracker(cfg.axis)
    for _ in range(5):
        assert tracker.update(_straight(cfg, y0=0.05)) is not None
    assert tracker.n_measured == 5


    held = tracker.update(_straight(cfg, y0=2.1))
    assert tracker.n_rejected_gate == 1
    assert held is not None and held.source == "stale"
    assert held.y0_m == pytest.approx(0.05, abs=0.03), "принята уехавшая ось"


def test_anchor_refuses_to_initialise_on_a_displaced_axis(cfg: Config) -> None:
    tracker = AxisTracker(cfg.axis)
    assert tracker.update(_straight(cfg, y0=1.5)) is None
    assert tracker.anchor.ready is False


    assert tracker.n_rejected_gate == 0
    assert tracker.n_uninitialised == 1
    assert tracker.causes == {"anchor_uninitialised": 1}

    assert tracker.update(_straight(cfg, y0=0.05)) is not None
    assert tracker.anchor.ready is True


def test_hold_expires_by_path_not_by_frames(cfg: Config) -> None:
    limit = cfg.axis.max_hold_path_m
    assert limit == pytest.approx(8.24, abs=0.1)

    standing = AxisTracker(cfg.axis)
    standing.update(_straight(cfg, y0=0.05))
    empty = np.zeros((0, 3))
    for _ in range(50):
        assert standing.update(empty, advance_m=0.0) is not None, "удержание истекло у стоящего"

    moving = AxisTracker(cfg.axis)
    moving.update(_straight(cfg, y0=0.05))
    step = 1.7
    alive = [moving.update(empty, advance_m=step) is not None for _ in range(10)]
    assert sum(alive) == int(limit // step), "лимит пути отработал не по метрам"


def test_held_axis_is_shifted_forward_by_odometry(cfg: Config) -> None:
    tracker = AxisTracker(cfg.axis)
    first = tracker.update(_straight(cfg, y0=0.05, yaw=-1.6))
    assert first is not None
    held = tracker.update(np.zeros((0, 3)), advance_m=2.0)
    assert held is not None
    assert held.y0_m == pytest.approx(first.y0_m + first.slope * 2.0, abs=1e-6)
    assert held.x_end_m < first.x_end_m


def test_held_frames_do_not_reset_the_measured_history(cfg: Config) -> None:
    tracker = AxisTracker(cfg.axis)
    for _ in range(7):
        tracker.update(_straight(cfg, y0=0.05))
    assert tracker.measured_run_frames == 7
    empty = np.zeros((0, 3))
    for _ in range(3):
        tracker.update(empty, advance_m=1.7)
    assert tracker.measured_run_frames == 7, "удержание обнулило чистую историю"


def test_standing_train_does_not_lose_trusted_range(cfg: Config) -> None:
    tracker = AxisTracker(cfg.axis)
    first = tracker.update(_straight(cfg, y0=0.05))
    assert first is not None
    empty = np.zeros((0, 3))
    for _ in range(30):
        held = tracker.update(empty, advance_m=0.0)
    assert held is not None
    assert held.x_end_m == pytest.approx(first.x_end_m, abs=1e-6)
    assert held.sigma_y_1m == pytest.approx(first.sigma_y_1m, abs=1e-9)


    moving = AxisTracker(cfg.axis)
    start = moving.update(_straight(cfg, y0=0.05))
    held_moving = moving.update(empty, advance_m=1.6)
    assert held_moving is not None
    assert held_moving.sigma_y_1m > start.sigma_y_1m


def test_sigma_does_not_punish_the_rails_segment_for_the_bed_error(cfg: Config) -> None:
    common = dict(
        slope=0.0, y0_m=0.0, x_start_m=5.0, x_traced_m=26.0, residual_max_m=0.01,
        meets_clearance=True, n_slices=8, method="rails", source="measured",
        staleness_frames=0, sigma_y_1m=float(np.tan(np.radians(0.075))),
        clearance_m=cfg.axis.clearance_m,
    )
    extended = AxisEstimate(x_end_m=45.0, cross_check_m=0.08, x_joint_m=26.0,
                            far_slope=0.0, far_y0_m=0.0, far_method="bed", **common)
    prior = cfg.detector.cross_check_prior_deg
    near = float(extended.sigma_edge(20.0, prior))
    far = float(extended.sigma_edge(40.0, prior))
    assert near < far, "рельсовый участок наказан ошибкой полосы"
    assert far == pytest.approx(
        float(np.hypot(np.hypot(extended.sigma_y(40.0), 0.01), 0.08)), abs=1e-6)


    broken = AxisEstimate(x_end_m=45.0, cross_check_m=0.30, x_joint_m=26.0,
                          far_slope=0.0, far_y0_m=0.0, far_method="bed", **common)
    assert float(broken.sigma_edge(20.0, prior)) > near * 3

