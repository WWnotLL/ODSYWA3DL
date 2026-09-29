# Дальний участок оси по стенам тоннеля: смещение стен калибруется на рельсовой оси, ось продлевается, пока обе стены видны подряд.

from __future__ import annotations

from dataclasses import replace

import numpy as np

from core.axis import AxisEstimate
from core.config import WallsConfig


def _slab(xyz: np.ndarray, x_from: float, x_to: float, cfg: WallsConfig, railhead: float) -> np.ndarray:
    bins, step = cfg.height_bins_m, cfg.voxel_m
    keep = ((np.abs(xyz[:, 1]) < cfg.lateral_limit_m)
            & (xyz[:, 0] >= x_from - step) & (xyz[:, 0] < x_to + step)
            & (xyz[:, 2] >= railhead + bins[0] - step) & (xyz[:, 2] < railhead + bins[-1] + step))
    return xyz[keep]


def voxels(points: np.ndarray, cfg: WallsConfig, railhead: float) -> np.ndarray:
    bins, step = cfg.height_bins_m, cfg.voxel_m
    cells = np.floor(points / step).astype(np.int64)
    if cells.shape[0] == 0:
        return np.empty((0, 3))
    cz = cells[:, 2]
    low = int(cz.min())
    levels = np.arange(low, int(cz.max()) + 1, dtype=np.int64)
    table = np.digitize(((levels + 0.5) * step).astype(np.float32).astype(float) - railhead, bins) - 1
    band = table[cz - low]
    cells = cells[(band >= 0) & (band < bins.size - 1)]
    key = ((cells[:, 0] + 1000) * 100000 + (cells[:, 1] + 50000)) * 100000 + (cells[:, 2] + 50000)
    _, first = np.unique(key, return_index=True)
    return ((cells[first] + 0.5) * step).astype(np.float32).astype(float)


def _near_voxels(xyz: np.ndarray, axis: AxisEstimate, cfg: WallsConfig, railhead: float) -> np.ndarray:
    slab = _slab(xyz, axis.x_start_m, axis.x_traced_m, cfg, railhead)
    u = np.abs(slab[:, 1] - (axis.slope * slab[:, 0] + axis.y0_m))
    margin = 2.0 * cfg.voxel_m
    return voxels(slab[(u > cfg.side_min_m - margin) & (u < cfg.side_max_m + margin)], cfg, railhead)


def _height_bin(z: np.ndarray, cfg: WallsConfig, railhead: float) -> np.ndarray:
    return np.digitize(z - railhead, cfg.height_bins_m) - 1


def calibrate(pts: np.ndarray, axis: AxisEstimate, cfg: WallsConfig, railhead: float) -> dict[int, np.ndarray]:
    x0, x1 = axis.x_start_m, axis.x_traced_m
    n_bins = cfg.height_bins_m.size - 1
    band = pts[(pts[:, 0] >= x0) & (pts[:, 0] <= x1)]
    hb = _height_bin(band[:, 2], cfg, railhead)
    ok = (hb >= 0) & (hb < n_bins)
    band, hb = band[ok], hb[ok]
    u = band[:, 1] - (axis.slope * band[:, 0] + axis.y0_m)
    edges = np.arange(cfg.side_min_m, cfg.side_max_m + cfg.cal_bin_m, cfg.cal_bin_m)
    sides = {}
    for side in (1, -1):
        v = side * u
        sel = np.flatnonzero((v > cfg.side_min_m) & (v < cfg.side_max_m))
        order = sel[np.argsort(hb[sel], kind="stable")]
        bounds = np.searchsorted(hb[order], np.arange(n_bins + 1))
        table = np.full(n_bins, np.nan)
        for b in range(n_bins):
            group = order[bounds[b]:bounds[b + 1]]
            if group.size < cfg.cal_min_points:
                continue
            vb = v[group]
            counts = np.histogram(vb, bins=edges)[0]
            peak = int(np.argmax(counts))
            near = vb[np.abs(vb - (edges[peak] + 0.5 * cfg.cal_bin_m)) <= cfg.cal_spread_m]
            if near.size < max(cfg.cal_min_points, cfg.cal_peak_fraction * vb.size):
                continue
            slices = np.floor((band[group, 0] - x0) / cfg.slice_m)
            on_peak = np.abs(vb - np.median(near)) <= cfg.cal_spread_m
            if np.unique(slices[on_peak]).size < max(cfg.cal_min_slices, cfg.cal_slice_fraction * np.unique(slices).size):
                continue
            table[b] = float(np.median(near))
        if np.isfinite(table).sum() >= cfg.cal_min_bins:
            sides[side] = table
    return sides


def _fit(xs: list[float], ys: list[float], at, cfg: WallsConfig):
    xs, ys = np.asarray(xs), np.asarray(ys)
    use = xs >= xs.max() - cfg.fit_window_m
    xs, ys = xs[use], ys[use]
    deg = 2 if np.ptp(xs) >= cfg.fit_quadratic_span_m and xs.size >= cfg.fit_quadratic_points else 1
    value = np.polyval(np.polyfit(xs, ys, deg), at)
    return float(value) if np.ndim(value) == 0 else value


def far_axis(xyz: np.ndarray, axis: AxisEstimate, rails: np.ndarray | None, cfg: WallsConfig,
             railhead: float) -> dict:
    sides = calibrate(_near_voxels(xyz, axis, cfg, railhead), axis, cfg, railhead)
    out = {"sides": sorted(sides), "samples": [], "stop": None}
    if not sides:
        out["stop"] = "no_wall"
        return out
    if rails is not None:
        near = rails[(rails[:, 0] >= axis.x_start_m) & (rails[:, 0] <= axis.x_traced_m)]
        xs, ys = list(near[:, 0]), list(near[:, 1])
    else:
        xs, ys = [], []
    if len(xs) < 2:
        grid = np.arange(axis.x_start_m, axis.x_traced_m + 1e-6, cfg.slice_m)
        xs, ys = list(grid), list(axis.slope * grid + axis.y0_m)
    out["xs"], out["ys"] = xs, ys
    if len(sides) < 2:
        out["stop"] = "one_wall"
        return out
    pts = voxels(_slab(xyz, axis.x_traced_m, cfg.far_end_m + cfg.slice_m, cfg, railhead), cfg, railhead)
    hb_all = _height_bin(pts[:, 2], cfg, railhead)
    ok = (hb_all >= 0) & (hb_all < cfg.height_bins_m.size - 1)
    pts, hb_all = pts[ok], hb_all[ok]
    for start in np.arange(axis.x_traced_m, cfg.far_end_m, cfg.slice_m):
        mid = start + 0.5 * cfg.slice_m
        inside = (pts[:, 0] >= start) & (pts[:, 0] < start + cfg.slice_m)
        block, hb = pts[inside], hb_all[inside]
        pred = _fit(xs, ys, block[:, 0], cfg) if block.size else None
        per_side = {}
        for side, table in sides.items():
            d = table[hb] if block.size else np.empty(0)
            valid = np.isfinite(d)
            if not valid.any():
                continue
            cand = block[valid, 1] - side * d[valid]
            gate = np.abs(cand - np.asarray(pred)[valid]) <= cfg.gate_m
            if gate.sum() >= cfg.sample_min_points:
                per_side[side] = float(np.median(cand[gate] - np.asarray(pred)[valid][gate]))
        if len(per_side) < 2:
            out["stop"] = "one_side" if per_side else "no_side"
            break
        if abs(per_side[1] - per_side[-1]) > cfg.width_tol_m:
            out["stop"] = "width"
            break
        y_mid = _fit(xs, ys, mid, cfg) + float(np.mean(list(per_side.values())))
        xs.append(mid)
        ys.append(y_mid)
        out["samples"].append((mid, y_mid))
    else:
        out["stop"] = "end"
    out["xs"], out["ys"] = xs, ys
    return out


def two_sided_reach(far: dict, start: float) -> float:
    return float(far["samples"][-1][0]) if far["samples"] else float(start)


def y_at(xs: np.ndarray, ys: np.ndarray, x: float, cfg: WallsConfig) -> float | None:
    use = np.abs(xs - x) <= cfg.polyline_window_m
    if use.sum() < cfg.polyline_min_points:
        return None
    deg = 2 if use.sum() >= cfg.polyline_quadratic_points else 1
    return float(np.polyval(np.polyfit(xs[use], ys[use], deg), x))


def extend_by_walls(estimate: AxisEstimate | None, xyz: np.ndarray, cfg: WallsConfig,
                    railhead: float) -> AxisEstimate | None:
    if estimate is None:
        return None
    info = {"enabled": True, "used": False, "reach_two_m": None, "sides": [], "stop": "not_measured"}
    if estimate.source != "measured":
        return replace(estimate, walls_info=info)
    far = far_axis(xyz, estimate, estimate.rail_centres, cfg, railhead)
    reach = two_sided_reach(far, estimate.x_traced_m)
    info.update(reach_two_m=round(reach, 2), sides=["L" if s > 0 else "R" for s in sorted(far["sides"], reverse=True)],
                stop=far["stop"])
    start = estimate.x_end_m
    if "xs" not in far or reach <= start + cfg.step_m:
        return replace(estimate, walls_info=info)
    upto = np.asarray(far["xs"]) <= reach + 1e-6
    xs_all, ys_all = np.asarray(far["xs"])[upto], np.asarray(far["ys"])[upto]
    xs, ys = [start], [float(estimate.y_at(start))]
    for x in np.arange(start + cfg.step_m, reach + 1e-6, cfg.step_m):
        y = y_at(xs_all, ys_all, float(x), cfg)
        if y is None:
            break
        xs.append(float(x))
        ys.append(y)
    if len(xs) < 2:
        return replace(estimate, walls_info=info)
    info["used"] = True
    return replace(estimate, x_end_m=xs[-1], walls_x=tuple(xs), walls_y=tuple(ys), walls_from=start, walls_info=info)
