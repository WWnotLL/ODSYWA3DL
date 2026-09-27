# Ось пути по кадру: трассеры по рельсам и полосе пола, продление базы, якорь позы и удержание оси.

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from core.config import AxisConfig, AxisMethodError


class AxisError(RuntimeError):
    pass


@dataclass(frozen=True)
class AxisEstimate:
    slope: float
    y0_m: float
    x_start_m: float
    x_end_m: float
    x_traced_m: float
    residual_max_m: float
    cross_check_m: float | None
    meets_clearance: bool
    n_slices: int
    method: str
    source: str
    staleness_frames: int
    sigma_y_1m: float


    cross_check_reason: str = "no_rival"

    x_joint_m: float | None = None
    far_slope: float | None = None
    far_y0_m: float | None = None
    far_method: str | None = None
    far_correction_slope: float | None = None
    far_correction_shift_m: float | None = None
    hold_path_m: float = 0.0
    clearance_m: float = 0.113


    polyline_step_m: float | None = None

    @property
    def yaw_deg(self) -> float:
        return float(np.degrees(np.arctan(self.slope)))

    @property
    def state(self) -> str:
        if not self.meets_clearance:
            return "degraded"
        return "held" if self.source == "stale" else "measured"

    def y_at(self, x: np.ndarray | float) -> np.ndarray:
        near = self.slope * np.asarray(x, dtype=float) + self.y0_m
        if self.x_joint_m is None:
            return near
        far = self.far_slope * np.asarray(x, dtype=float) + self.far_y0_m
        return np.where(np.asarray(x, dtype=float) <= self.x_joint_m, near, far)

    def offset(self, xyz: np.ndarray) -> np.ndarray:
        return xyz[:, 1] - self.y_at(xyz[:, 0])

    def sigma_y(self, distance: np.ndarray | float) -> np.ndarray:
        return self.sigma_y_1m * np.asarray(distance, dtype=float)

    def sigma_edge(self, distance: np.ndarray | float, prior_deg: float) -> np.ndarray:
        distance = np.asarray(distance, dtype=float)
        prior = np.tan(np.radians(prior_deg)) * distance
        if self.cross_check_m is None:
            shift = prior
        elif self.x_joint_m is None:
            shift = np.full_like(distance, self.cross_check_m)
        else:
            near = prior if self.cross_check_m <= self.clearance_m else np.full_like(
                distance, self.cross_check_m)
            shift = np.where(distance <= self.x_joint_m, near, self.cross_check_m)
        return np.hypot(np.hypot(self.sigma_y(distance), self.residual_max_m), shift)

    def trustworthy_range(self, clearance_m: float) -> float:
        if self.sigma_y_1m <= 0.0:
            return self.x_end_m
        return min(self.x_end_m, clearance_m / self.sigma_y_1m)

    def polyline(self) -> dict | None:
        if self.polyline_step_m is None:
            return None
        x = np.arange(self.x_start_m, self.x_end_m, self.polyline_step_m)
        x = np.append(x, self.x_end_m)
        y = self.y_at(x)
        far = (x > self.x_joint_m) if self.x_joint_m is not None else np.zeros(x.size, bool)
        return {
            "step_m": self.polyline_step_m,
            "points_xy_m": [[round(float(a), 4), round(float(b), 4)] for a, b in zip(x, y)],
            "source": [self.far_method if f else self.method for f in far],
        }

    def to_debug(self) -> dict:
        return {
            "source": self.source,
            "method": self.method,
            "staleness_frames": self.staleness_frames,
            "slope_deg": round(self.yaw_deg, 4),
            "y_center_m": round(self.y0_m, 4),
            "sigma_y_1m": round(self.sigma_y_1m, 6),
            "residual_max_m": round(self.residual_max_m, 4),
            "cross_check_m": None if self.cross_check_m is None else round(self.cross_check_m, 4),
            "cross_check_reason": self.cross_check_reason,
            "meets_clearance": self.meets_clearance,
            "state": self.state,
            "hold_path_m": round(self.hold_path_m, 2),


            "x_joint_m": None if self.x_joint_m is None else round(self.x_joint_m, 2),
            "far_method": self.far_method,
            "far_correction": (None if self.far_correction_slope is None else {
                "slope": round(self.far_correction_slope, 6),
                "shift_m": round(self.far_correction_shift_m, 4)}),
            "x_traced_m": [round(self.x_start_m, 2), round(self.x_traced_m, 2)],
            "x_trusted_m": round(self.x_end_m, 2),
            "n_slices": self.n_slices,
            "polyline": self.polyline(),
        }


def _slice_bounds(cfg: AxisConfig) -> np.ndarray:
    return np.arange(cfg.x_min_m, cfg.x_max_m, cfg.slice_m)


def _ridge(lateral: np.ndarray, lo: float, hi: float, cfg: AxisConfig) -> float | None:
    band = lateral[(lateral > lo) & (lateral < hi)]
    if band.size < cfg.rails_min_points:
        return None
    bins = np.arange(lo, hi + cfg.rails_histogram_bin_m, cfg.rails_histogram_bin_m)
    counts, _ = np.histogram(band, bins=bins)
    centres = 0.5 * (bins[:-1] + bins[1:])
    peak = int(np.argmax(counts))
    half = cfg.rails_peak_halfwidth_bins
    window = slice(max(0, peak - half), min(counts.size, peak + half + 1))
    if counts[window].sum() < cfg.rails_min_points:
        return None
    return float(np.average(centres[window], weights=counts[window]))


def _rail_centres(xyz: np.ndarray, cfg: AxisConfig) -> tuple[np.ndarray | None, str]:
    head = xyz[(xyz[:, 2] > cfg.rails_z_min_m) & (xyz[:, 2] < cfg.rails_z_max_m)]
    if head.shape[0] < cfg.rails_min_points:
        return None, "too_few_points"
    starts = _slice_bounds(cfg)

    rows, guess = [], 0.0
    for start in starts:
        inside = (head[:, 0] >= start) & (head[:, 0] < start + cfg.slice_m)
        lateral = head[inside, 1]
        left = _ridge(lateral, guess - cfg.rails_search_far_m, guess - cfg.rails_search_near_m, cfg)
        right = _ridge(lateral, guess + cfg.rails_search_near_m, guess + cfg.rails_search_far_m, cfg)
        if left is None or right is None:
            continue
        if not cfg.rails_gauge_min_m < right - left < cfg.rails_gauge_max_m:
            continue
        guess = 0.5 * (left + right)
        rows.append((float(head[inside, 0].mean()), guess))

    for _ in range(cfg.refine_passes):
        if len(rows) < cfg.min_slices:
            break
        table = np.asarray(rows)
        line = np.polyfit(table[:, 0], table[:, 1], 1)
        found = []
        for start in starts:
            inside = (head[:, 0] >= start) & (head[:, 0] < start + cfg.slice_m)
            lateral = head[inside, 1]
            centre = float(np.polyval(line, start + 0.5 * cfg.slice_m))
            half = cfg.rails_half_gauge_m
            window = cfg.rails_search_halfwidth_m
            left = _ridge(lateral, centre - half - window, centre - half + window, cfg)
            right = _ridge(lateral, centre + half - window, centre + half + window, cfg)
            if left is None or right is None:
                continue
            if not cfg.rails_gauge_min_m < right - left < cfg.rails_gauge_max_m:
                continue
            found.append((float(head[inside, 0].mean()), 0.5 * (left + right)))
        if len(found) <= len(rows):
            break
        rows = found

    if len(rows) >= cfg.min_slices:
        return np.asarray(rows), "ok"
    return None, "too_few_slices"


def rail_head_lines(xyz: np.ndarray, estimate: AxisEstimate, cfg: AxisConfig) -> np.ndarray:
    end = estimate.x_joint_m if estimate.x_joint_m is not None else estimate.x_end_m
    starts = _slice_bounds(cfg)
    starts = starts[(starts + cfg.slice_m > estimate.x_start_m) & (starts < end)]
    rows = np.full((starts.size, 4), np.nan)
    rows[:, 0], rows[:, 1] = starts, np.minimum(starts + cfg.slice_m, end)
    head = xyz[(xyz[:, 2] > cfg.rails_z_min_m) & (xyz[:, 2] < cfg.rails_z_max_m)]
    across = estimate.offset(head)
    half, window = cfg.rails_half_gauge_m, cfg.rails_search_halfwidth_m
    for i, start in enumerate(starts):
        lateral = across[(head[:, 0] >= start) & (head[:, 0] < rows[i, 1])]
        left = _ridge(lateral, -half - window, -half + window, cfg)
        right = _ridge(lateral, half - window, half + window, cfg)
        if left is None or right is None or not cfg.rails_gauge_min_m < right - left < cfg.rails_gauge_max_m:
            continue
        rows[i, 2], rows[i, 3] = left, right
    return rows


def _bed_centres(xyz: np.ndarray, cfg: AxisConfig) -> tuple[np.ndarray | None, str]:
    band = xyz[
        (xyz[:, 2] > cfg.bed_z_min_m)
        & (xyz[:, 2] < cfg.bed_z_max_m)
        & (np.abs(xyz[:, 1]) < cfg.bed_lateral_limit_m)
    ]
    if band.shape[0] < cfg.bed_min_points:
        return None, "too_few_points"
    lo_pct = cfg.bed_edge_percentile
    rows = []
    too_wide = too_narrow = 0
    for start in _slice_bounds(cfg):
        inside = (band[:, 0] >= start) & (band[:, 0] < start + cfg.slice_m)
        if inside.sum() < cfg.bed_min_points:
            continue
        lo, hi = np.percentile(band[inside, 1], [lo_pct, 100.0 - lo_pct])
        width = hi - lo
        if width >= cfg.bed_width_max_m:
            too_wide += 1
            continue
        if width <= cfg.bed_width_min_m:
            too_narrow += 1
            continue
        rows.append((float(band[inside, 0].mean()), 0.5 * (lo + hi)))
    if len(rows) >= cfg.min_slices:
        return np.asarray(rows), "ok"


    if too_wide > too_narrow and too_wide >= cfg.min_slices:
        return None, "too_wide"
    return None, "too_few_slices"


_ESTIMATORS = {"rails": _rail_centres, "bed": _bed_centres}


def _trim_to_clearance(table: np.ndarray, cfg: AxisConfig):
    x, y = table[:, 0], table[:, 1]
    best = None
    for end in range(cfg.min_slices, table.shape[0] + 1):
        slope, y0 = np.polyfit(x[:end], y[:end], 1)
        residual = float(np.abs(y[:end] - (slope * x[:end] + y0)).max())
        if residual <= cfg.clearance_m:
            best = (table[:end], slope, y0, residual)
    if best is not None:
        return best

    end = cfg.min_slices
    slope, y0 = np.polyfit(x[:end], y[:end], 1)
    residual = float(np.abs(y[:end] - (slope * x[:end] + y0)).max())
    return table[:end], slope, y0, residual


def _cross_check(kept: np.ndarray, slope: float, y0: float,
                 others: list[np.ndarray], cfg: AxisConfig) -> float | None:
    x_lo, x_hi = float(kept[0, 0]), float(kept[-1, 0])
    gaps: list[float] = []
    for other in others:
        lo = max(x_lo, float(other[0, 0]))
        hi = min(x_hi, float(other[-1, 0]))
        if hi - lo < cfg.slice_m:
            continue
        rival = np.polyfit(other[:, 0], other[:, 1], 1)
        at = np.array([lo, hi])
        gaps.append(float(np.abs(np.polyval(rival, at) - (slope * at + y0)).max()))
    return max(gaps) if gaps else None


def _extend_base(kept: np.ndarray, slope: float, y0: float,
                 others: dict[str, np.ndarray], cfg: AxisConfig):
    best = None
    x_end = float(kept[-1, 0])
    for name, table in others.items():
        rival_kept, r_slope, r_y0, r_residual = _trim_to_clearance(table, cfg)
        rival_end = float(rival_kept[-1, 0])
        if rival_end <= x_end or r_residual > cfg.clearance_m:
            continue


        ends = np.array([min(float(kept[0, 0]), float(rival_kept[0, 0])), rival_end])
        gap = float(np.abs((r_slope * ends + r_y0) - (slope * ends + y0)).max())
        if gap > cfg.clearance_m:
            continue
        if best is None or rival_end > best[3]:
            best = (name, float(r_slope), float(r_y0), rival_end, gap)
    return best


def extension_correction(near: np.ndarray, far: np.ndarray, cfg: AxisConfig
                         ) -> tuple[float, float, float, float] | None:
    near_idx = np.floor((near[:, 0] - cfg.x_min_m) / cfg.slice_m).astype(int)
    far_idx = np.floor((far[:, 0] - cfg.x_min_m) / cfg.slice_m).astype(int)
    common, ni, fi = np.intersect1d(near_idx, far_idx, return_indices=True)
    if common.size < cfg.min_slices:
        return None
    x = near[ni, 0]
    slope, shift = np.polyfit(x, far[fi, 1] - near[ni, 1], 1)
    return float(slope), float(shift), float(x.min()), float(x.max())


def _cross_check_reason(method: str, tables: dict, reasons: dict,
                        cross: float | None, cfg: AxisConfig) -> str:
    if cross is not None:
        return "ok"
    rivals = [name for name in cfg.methods if name != method]
    if not rivals:
        return "no_rival"
    if any(tables.get(name) is not None for name in rivals):
        return "unavailable_no_overlap"
    if any(reasons.get(name) == "too_wide" for name in rivals):
        return "unavailable_structural"
    return "unavailable_failed"


def estimate_axis(xyz: np.ndarray, cfg: AxisConfig) -> AxisEstimate | None:
    need_all = cfg.cross_check or cfg.extend_base
    tables: dict[str, np.ndarray | None] = {}
    reasons: dict[str, str] = {}
    for method in cfg.methods:
        if method not in _ESTIMATORS:
            raise AxisMethodError(f"неизвестный метод оценки оси: {method!r}")
        if need_all:
            tables[method], reasons[method] = _ESTIMATORS[method](xyz, cfg)

    for method in cfg.methods:
        table = tables.get(method)
        if table is None and not need_all:
            table, reasons[method] = _ESTIMATORS[method](xyz, cfg)
        if table is None:
            continue
        kept, slope, y0, residual = _trim_to_clearance(table, cfg)
        others = {name: other for name, other in tables.items()
                  if name != method and other is not None}
        cross = _cross_check(kept, slope, y0, list(others.values()), cfg)
        reason = _cross_check_reason(method, tables, reasons, cross, cfg)

        x_end = float(kept[-1, 0])
        joint = far_slope = far_y0 = far_method = None
        fix_slope = fix_shift = None
        if cfg.extend_base and others:
            extension = _extend_base(kept, slope, y0, others, cfg)
            fix = None
            if extension is not None and cfg.correct_extension:
                fix = extension_correction(table, others[extension[0]], cfg)
                if fix is None:
                    extension = None
            if extension is not None:
                far_method, far_slope, far_y0, x_end, gap = extension
                joint = float(kept[-1, 0])
                if fix is not None:
                    fix_slope, fix_shift = fix[0], fix[1]
                    far_slope, far_y0 = far_slope - fix_slope, far_y0 - fix_shift
                cross = gap if cross is None else max(cross, gap)
                reason = "ok"
        return AxisEstimate(
            slope=float(slope),
            y0_m=float(y0),
            x_start_m=float(kept[0, 0]),
            x_end_m=x_end,
            x_traced_m=float(table[-1, 0]),
            residual_max_m=float(residual),
            cross_check_m=cross,
            meets_clearance=bool(residual <= cfg.clearance_m),
            n_slices=int(kept.shape[0]),
            method=method,
            source="measured",
            staleness_frames=0,
            sigma_y_1m=float(np.tan(np.radians(cfg.noise_slope_deg))),
            clearance_m=float(cfg.clearance_m),
            polyline_step_m=float(cfg.polyline_step_m),
            cross_check_reason=reason,
            x_joint_m=joint,
            far_slope=far_slope,
            far_y0_m=far_y0,
            far_method=far_method,
            far_correction_slope=fix_slope,
            far_correction_shift_m=fix_shift,
        )
    return None


class PoseAnchor:
    def __init__(self, cfg: AxisConfig) -> None:
        self._cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._values: list[float] = []

    @property
    def ready(self) -> bool:
        return bool(self._values)

    @property
    def value(self) -> float | None:
        if not self._values:
            return None
        return float(np.median(self._values[-self._cfg.anchor_window_frames:]))

    def deviation(self, estimate: AxisEstimate) -> float | None:
        anchor = self.value
        return None if anchor is None else abs(estimate.y0_m - anchor)

    def accepts(self, estimate: AxisEstimate) -> bool:
        cfg = self._cfg
        if self.ready:
            return abs(estimate.y0_m - self.value) <= cfg.gate_m
        if not estimate.meets_clearance:
            return False
        if abs(estimate.y0_m) > cfg.anchor_init_max_offset_m:
            return False
        if estimate.cross_check_m is not None:
            return estimate.cross_check_m <= cfg.clearance_m
        return estimate.cross_check_reason == "unavailable_structural" and \
            estimate.method == "rails"

    def update(self, estimate: AxisEstimate) -> None:
        self._values.append(float(estimate.y0_m))


def _advance(estimate: AxisEstimate, distance_m: float, cfg: AxisConfig) -> AxisEstimate:
    if distance_m <= 0.0:
        return estimate
    far_y0 = (None if estimate.far_y0_m is None
              else estimate.far_y0_m + estimate.far_slope * distance_m)
    return replace(
        estimate,
        y0_m=estimate.y0_m + estimate.slope * distance_m,
        far_y0_m=far_y0,
        x_start_m=max(cfg.x_min_m, estimate.x_start_m - distance_m),
        x_end_m=max(cfg.x_min_m, estimate.x_end_m - distance_m),
        x_joint_m=(None if estimate.x_joint_m is None
                   else max(cfg.x_min_m, estimate.x_joint_m - distance_m)),
    )


class AxisTracker:
    def __init__(self, cfg: AxisConfig) -> None:
        self._cfg = cfg
        self.anchor = PoseAnchor(cfg)
        self.reset()

    def reset(self) -> None:
        self._last: AxisEstimate | None = None
        self._staleness = 0
        self._hold_path_m = 0.0
        self._gap_frames = 0
        self._gap_path_m = 0.0
        self.anchor.reset()

        self.n_measured = 0
        self.n_stale = 0
        self.n_lost = 0


        self.causes: dict[str, int] = {}
        self.n_rejected = 0
        self.n_rejected_gate = 0
        self.n_uninitialised = 0


        self.longest_gap_frames = 0
        self.longest_gap_path_m = 0.0


        self._measured_run = 0


        self.last_cause: str | None = None
        self.last_hold_expired = False

    @property
    def last(self) -> AxisEstimate | None:
        return self._last

    @property
    def measured_run_frames(self) -> int:
        return self._measured_run

    def _close_gap(self) -> None:
        self.longest_gap_frames = max(self.longest_gap_frames, self._gap_frames)
        self.longest_gap_path_m = max(self.longest_gap_path_m, self._gap_path_m)
        self._gap_frames = 0
        self._gap_path_m = 0.0

    def _degraded(self, frame_gap: int, advance_m: float | None,
                  cause: str) -> AxisEstimate | None:
        cfg = self._cfg
        self.causes[cause] = self.causes.get(cause, 0) + 1
        self.last_cause = cause
        self.last_hold_expired = False
        self._gap_frames += frame_gap
        self._gap_path_m += 0.0 if advance_m is None else advance_m
        if self._last is None:
            self.n_lost += 1
            return None
        self._staleness += frame_gap
        if advance_m is None:
            expired = self._staleness > cfg.max_staleness_frames
        else:
            self._hold_path_m += advance_m
            expired = self._hold_path_m > cfg.max_hold_path_m
        if expired:
            self._last = None
            self.n_lost += 1
            self.last_hold_expired = True
            return None
        self.n_stale += 1


        if advance_m is None:
            drift = self._staleness * np.tan(np.radians(cfg.max_slope_step_deg))
        else:
            drift = float(np.tan(self._hold_path_m / cfg.hold_curvature_radius_m))
        base = np.tan(np.radians(cfg.noise_slope_deg))
        sigma = float(np.hypot(base, drift))
        held = _advance(self._last, 0.0 if advance_m is None else advance_m, cfg)
        self._last = held
        trusted = min(held.x_end_m, cfg.clearance_m / sigma) if sigma > 0 else held.x_end_m
        return replace(
            held,
            source="stale",
            staleness_frames=self._staleness,
            sigma_y_1m=sigma,
            hold_path_m=self._hold_path_m,
            x_end_m=float(max(trusted, held.x_start_m)),
        )

    def update(self, xyz: np.ndarray, *, frame_gap: int = 1,
               advance_m: float | None = None) -> AxisEstimate | None:
        if frame_gap < 1:
            raise ValueError(f"frame_gap должен быть положительным, получен {frame_gap}")
        fresh = estimate_axis(xyz, self._cfg)
        if fresh is None:
            return self._degraded(frame_gap, advance_m, "tracer_failed")
        if not self.anchor.accepts(fresh):
            if self.anchor.ready:
                self.n_rejected_gate += 1
                cause = "gate_rejected"
            else:
                self.n_uninitialised += 1
                cause = "anchor_uninitialised"
            return self._degraded(frame_gap, advance_m, cause)
        if self._last is not None:
            step = abs(fresh.yaw_deg - self._last.yaw_deg)
            if step > self._cfg.max_slope_step_deg * (self._staleness + frame_gap):
                self.n_rejected += 1
                return self._degraded(frame_gap, advance_m, "outlier_rejected")
        self.anchor.update(fresh)
        self.last_cause = None
        self.last_hold_expired = False
        if self._gap_frames:
            self._measured_run = 0
        self._measured_run += 1
        self._last = fresh
        self._staleness = 0
        self._hold_path_m = 0.0
        self._close_gap()
        self.n_measured += 1
        return fresh

