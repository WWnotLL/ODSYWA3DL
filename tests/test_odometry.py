# Тесты одометрии по продольному профилю.

from __future__ import annotations

import numpy as np
import pytest

from core.odometry import OdometryError, estimate_shift, longitudinal_profile
from tests.synthetic import landmark_tunnel

DT = 0.1


def _profiles(cfg, travelled_m, **kwargs):
    a = longitudinal_profile(*landmark_tunnel(0.0, **kwargs), cfg.odometry)
    b = longitudinal_profile(*landmark_tunnel(travelled_m, **kwargs), cfg.odometry)
    return a, b


def test_recovers_a_known_displacement(cfg):
    a, b = _profiles(cfg, 1.5)
    estimate = estimate_shift(a, b, DT, cfg.odometry)
    assert not estimate.degenerate, estimate.reason
    assert estimate.shift_m == pytest.approx(1.5, abs=cfg.odometry.bin_m)
    assert estimate.speed_mps == pytest.approx(15.0, abs=1.0)


@pytest.mark.parametrize("travelled", [0.4, 0.9, 1.5, 2.2])
def test_recovers_several_displacements(cfg, travelled):
    a, b = _profiles(cfg, travelled)
    estimate = estimate_shift(a, b, DT, cfg.odometry)
    assert not estimate.degenerate, estimate.reason
    assert estimate.speed_mps == pytest.approx(travelled / DT, rel=0.1)


def test_speed_in_kmh_is_consistent(cfg):
    a, b = _profiles(cfg, 1.5)
    estimate = estimate_shift(a, b, DT, cfg.odometry)
    assert estimate.speed_kmh == pytest.approx(estimate.speed_mps * 3.6)


def test_smooth_tube_is_reported_as_degenerate(cfg):
    a, b = _profiles(cfg, 1.5, smooth=True)
    estimate = estimate_shift(a, b, DT, cfg.odometry)
    assert estimate.degenerate
    assert estimate.speed_mps is None
    assert "однородный" in estimate.reason


def test_standing_train_gives_zero_speed(cfg):
    a, b = _profiles(cfg, 0.0)
    estimate = estimate_shift(a, b, DT, cfg.odometry)
    assert not estimate.degenerate
    assert estimate.speed_mps == pytest.approx(0.0, abs=0.5)


def test_richer_section_gives_a_sharper_peak(cfg):
    poor, _ = _profiles(cfg, 1.5, n_landmarks=4)
    rich, _ = _profiles(cfg, 1.5, n_landmarks=40)
    poor_estimate = estimate_shift(poor, longitudinal_profile(*landmark_tunnel(1.5, n_landmarks=4), cfg.odometry), DT, cfg.odometry)
    rich_estimate = estimate_shift(rich, longitudinal_profile(*landmark_tunnel(1.5, n_landmarks=40), cfg.odometry), DT, cfg.odometry)
    assert rich_estimate.prominence > poor_estimate.prominence


def test_rejects_nonpositive_dt(cfg):
    a, b = _profiles(cfg, 1.0)
    with pytest.raises(OdometryError, match="интервал"):
        estimate_shift(a, b, 0.0, cfg.odometry)


def test_rejects_mismatched_profiles(cfg):
    a, _ = _profiles(cfg, 1.0)
    with pytest.raises(OdometryError, match="разной формы"):
        estimate_shift(a, a[:-5], DT, cfg.odometry)


def test_profile_is_detrended(cfg):
    profile = longitudinal_profile(*landmark_tunnel(0.0), cfg.odometry)
    assert np.abs(profile.mean(axis=0)).max() < 0.3
    assert profile.shape[1] == len(cfg.odometry.channels)

