# Одометрия по лидару: путь за кадр из взаимной корреляции продольных профилей соседних кадров.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d

from core.config import OdometryConfig


class OdometryError(ValueError):
    pass


@dataclass(frozen=True)
class ShiftEstimate:
    shift_m: float | None
    speed_mps: float | None
    prominence: float
    degenerate: bool
    reason: str

    @property
    def speed_kmh(self) -> float | None:
        return None if self.speed_mps is None else self.speed_mps * 3.6


def longitudinal_profile(
    xyz: np.ndarray, intensity: np.ndarray, cfg: OdometryConfig
) -> np.ndarray:
    x = xyz[:, 0]
    inside = (x >= cfg.x_min_m) & (x < cfg.x_max_m)
    x, xyz, intensity = x[inside], xyz[inside], intensity[inside]

    n_bins = int(round((cfg.x_max_m - cfg.x_min_m) / cfg.bin_m))
    if n_bins < 3:
        raise OdometryError(f"слишком мало бинов: {n_bins}")
    index = np.clip(((x - cfg.x_min_m) / cfg.bin_m).astype(np.int64), 0, n_bins - 1)

    count = np.bincount(index, minlength=n_bins).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_intensity = np.bincount(index, weights=intensity, minlength=n_bins) / count
        radius = np.hypot(xyz[:, 1], xyz[:, 2])
        mean_radius = np.bincount(index, weights=radius, minlength=n_bins) / count
    sparse = count < cfg.min_points_per_bin
    mean_intensity[sparse] = 0.0
    mean_radius[sparse] = 0.0

    channels = {
        "count": np.log1p(count),
        "intensity": np.nan_to_num(mean_intensity),
        "radius": np.nan_to_num(mean_radius),
    }
    stacked = np.stack([channels[name] for name in cfg.channels], axis=1)
    return _detrend(stacked, cfg.detrend_window_bins)


def _detrend(profile: np.ndarray, window: int) -> np.ndarray:
    trend = uniform_filter1d(profile, size=window, axis=0, mode="nearest")
    residual = profile - trend
    scale = residual.std(axis=0, keepdims=True)
    return np.divide(residual, scale, out=np.zeros_like(residual), where=scale > 0.0)


def _correlate(previous: np.ndarray, current: np.ndarray, max_shift: int) -> np.ndarray:
    n_bins = previous.shape[0]
    shifts = np.arange(-max_shift, max_shift + 1)
    scores = np.zeros(shifts.size)
    for position, shift in enumerate(shifts):
        if shift >= 0:
            a, b = previous[shift:], current[: n_bins - shift]
        else:
            a, b = previous[: n_bins + shift], current[-shift:]
        if a.shape[0] < max_shift:
            continue
        scores[position] = float(np.mean(a * b))
    return scores


def _subbin_peak(scores: np.ndarray, peak: int) -> float:
    if peak <= 0 or peak >= scores.size - 1:
        return float(peak)
    left, middle, right = scores[peak - 1], scores[peak], scores[peak + 1]
    denominator = left - 2.0 * middle + right
    if denominator == 0.0:
        return float(peak)
    return float(peak) + 0.5 * (left - right) / denominator


def estimate_shift(
    previous: np.ndarray, current: np.ndarray, dt_s: float, cfg: OdometryConfig
) -> ShiftEstimate:
    if previous.shape != current.shape:
        raise OdometryError("профили разной формы")
    if dt_s <= 0.0:
        raise OdometryError(f"неположительный интервал между кадрами: {dt_s}")

    max_shift = int(round(cfg.max_shift_m / cfg.bin_m))
    scores = _correlate(previous, current, max_shift)
    if not np.any(scores):
        return ShiftEstimate(None, None, 0.0, True, "профиль однородный, структуры нет вовсе")

    peak = int(np.argmax(scores))
    spread = float(scores.std())
    prominence = (float(scores[peak]) - float(np.median(scores))) / spread if spread > 0 else 0.0

    if peak in (0, scores.size - 1):
        return ShiftEstimate(None, None, prominence, True, "пик на границе диапазона поиска")
    if prominence < cfg.min_prominence:
        return ShiftEstimate(None, None, prominence, True, "профиль однородный, пик не выражен")


    shift_bins = _subbin_peak(scores, peak) - max_shift
    shift_m = shift_bins * cfg.bin_m
    return ShiftEstimate(shift_m, shift_m / dt_s, prominence, False, "ок")

