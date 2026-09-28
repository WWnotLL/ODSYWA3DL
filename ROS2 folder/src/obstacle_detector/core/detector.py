# Слой A: коридор по габариту Ом вокруг оси, вырезы, кластеризация и дистанция до препятствия в одном кадре.

from __future__ import annotations

import numpy as np

from core.axis import AxisEstimate
from core.config import DetectorConfig, GaugeConfig
from core.types import Detection, FrameResult


PER_CLUSTER_DEBUG_KEYS = ("gauge_margin_m", "lateral_offset_m", "depth_m")


_NEIGHBOUR_OFFSETS = np.array(
    [(dx, dy, dz)
     for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
     if (dx, dy, dz) > (0, 0, 0)],
    dtype=np.int64,
)


def _encode(cells: np.ndarray, origin: np.ndarray, dims: np.ndarray) -> np.ndarray:
    local = cells - origin
    return (local[:, 0] * dims[1] + local[:, 1]) * dims[2] + local[:, 2]


def _connected_components(cells: np.ndarray) -> np.ndarray:
    if cells.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    origin = cells.min(axis=0) - 1
    dims = cells.max(axis=0) - origin + 2
    keys = _encode(cells, origin, dims)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]

    sources, targets = [], []
    for offset in _NEIGHBOUR_OFFSETS:
        probe = _encode(cells + offset, origin, dims)
        position = np.searchsorted(sorted_keys, probe)
        inside = position < sorted_keys.size
        hit = np.zeros(probe.shape, dtype=bool)
        hit[inside] = sorted_keys[position[inside]] == probe[inside]
        if not hit.any():
            continue
        sources.append(np.flatnonzero(hit))
        targets.append(order[position[hit]])

    labels = np.arange(cells.shape[0], dtype=np.int64)
    if not sources:
        return labels
    left = np.concatenate(sources)
    right = np.concatenate(targets)

    while True:
        updated = labels.copy()
        np.minimum.at(updated, left, labels[right])
        np.minimum.at(updated, right, labels[left])
        while True:
            jumped = updated[updated]
            if np.array_equal(jumped, updated):
                break
            updated = jumped
        if np.array_equal(updated, labels):
            return labels
        labels = updated


def _cluster(points: np.ndarray, distance: np.ndarray, cfg: DetectorConfig) -> np.ndarray:
    scale = cfg.cluster_eps_base_m + cfg.cluster_eps_per_m * distance
    cells = np.floor(points / scale[:, None]).astype(np.int64)
    unique_cells, inverse = np.unique(cells, axis=0, return_inverse=True)
    return _connected_components(unique_cells)[inverse]


def cluster_depth(depths: np.ndarray, rank: int) -> float:
    ordered = np.sort(np.asarray(depths, dtype=float))[::-1]
    return float(ordered[min(rank, ordered.size) - 1])


def drop_shallow(result: FrameResult, min_depth_m: float | None) -> FrameResult:
    depths = result.debug.get("depth_m")
    if min_depth_m is None or depths is None or not result.detections:
        return result
    keep = [d >= min_depth_m for d in depths]
    detections = [d for d, ok in zip(result.detections, keep) if ok]
    debug = dict(result.debug)
    for key in PER_CLUSTER_DEBUG_KEYS:
        values = result.debug.get(key)
        if values is not None and len(values) == len(keep):
            debug[key] = [v for v, ok in zip(values, keep) if ok]
    nearest = min((d.distance_m for d in detections), default=None)
    return FrameResult(bool(detections), nearest, detections, debug)


def _min_points(distance: float, cfg: DetectorConfig) -> float:
    ratio = cfg.min_points_reference_m / max(distance, cfg.min_points_reference_m)
    return max(cfg.min_points_floor, cfg.min_points_at_reference * ratio ** 2)


def _longest_run(flags: np.ndarray) -> int:
    if flags.size == 0 or not flags.any():
        return 0
    padded = np.concatenate(([0], flags.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    return int((edges[1::2] - edges[::2]).max())


def platform_runs(offset: np.ndarray, above_railhead: np.ndarray, x: np.ndarray,
                  near: float, far: float, gauge: GaugeConfig) -> tuple[int, int]:
    if far <= near:
        return 0, 0
    n_slices = max(1, int(np.ceil((far - near) / gauge.platform_slice_m)))
    magnitude = np.abs(offset)
    band = (
        (magnitude >= gauge.platform_lateral_from_m)
        & (magnitude <= gauge.platform_outer_lateral_m)
        & (above_railhead >= gauge.platform_evidence_height_min_m)
        & (above_railhead <= gauge.platform_height_max_m)
        & (x >= near) & (x <= far)
    )
    if not band.any():
        return 0, 0
    slot = np.clip(((x[band] - near) / gauge.platform_slice_m).astype(np.int64),
                   0, n_slices - 1)
    left = offset[band] > 0


    tall = above_railhead[band] >= gauge.platform_top_min_m
    runs = []
    for side in (left, ~left):
        counts = np.bincount(slot[side], minlength=n_slices)
        tops = np.bincount(slot[side & tall], minlength=n_slices)
        runs.append(_longest_run(
            (counts >= gauge.platform_min_points_per_slice)
            & (tops >= gauge.platform_min_points_above_top)))
    return runs[0], runs[1]


def gauge_bounds(gauge: GaugeConfig, cfg: DetectorConfig) -> tuple[float, float]:
    if gauge.rail_head_offset_m is None:
        return max(gauge.z_min_m, cfg.z_ground_margin_m), gauge.z_max_m
    railhead = gauge.rail_head_offset_m
    if cfg.floor_above_rail_head_m is not None:
        return railhead + cfg.floor_above_rail_head_m, railhead + gauge.z_max_m


    return (
        railhead + gauge.z_min_m + cfg.z_ground_margin_m,
        railhead + gauge.z_max_m,
    )


def rail_mask(xyz: np.ndarray, offset: np.ndarray, above_railhead: np.ndarray,
              gauge: GaugeConfig, heads: np.ndarray | None) -> np.ndarray:
    nominal = gauge.in_rail_mask(offset, above_railhead)
    if heads is None or heads.shape[0] == 0:
        return nominal
    found = np.isfinite(heads[:, 2])
    slot = np.searchsorted(heads[:, 0], xyz[:, 0], side="right") - 1
    inside = (slot >= 0) & (xyz[:, 0] < heads[np.clip(slot, 0, None), 1])
    slot = np.clip(slot, 0, None)
    near = inside & found[slot]
    left, right = heads[slot, 2], heads[slot, 3]
    low = above_railhead <= gauge.rail_mask_height_max_m
    on_head = low & ((np.abs(offset - left) <= gauge.rail_mask_found_half_width_m)
                     | (np.abs(offset - right) <= gauge.rail_mask_found_half_width_m))
    return np.where(near, on_head, nominal)


def detect(
    xyz: np.ndarray,
    axis: AxisEstimate | None,
    gauge: GaugeConfig,
    cfg: DetectorConfig,
    *,
    heads: np.ndarray | None = None,
) -> FrameResult:
    debug: dict = {
        "layer": "A",
        "axis": None if axis is None else axis.to_debug(),
        "n_points": int(xyz.shape[0]),
    }
    if axis is None:
        debug["reason"] = "axis_unavailable"
        return FrameResult(False, None, [], debug)

    far = min(axis.x_end_m, cfg.max_range_m) if cfg.max_range_m is not None else axis.x_end_m
    near = max(cfg.min_range_m, axis.x_start_m - 0.5 * cfg.range_start_slack_m)
    floor, ceiling = gauge_bounds(gauge, cfg)
    debug["corridor"] = {
        "x_m": [round(near, 2), round(far, 2)],
        "profile": "Ом (ГОСТ 23961, рис. 6)",
        "half_width_max_m": float(gauge.profile_half_width_m.max()),
        "z_m": [round(floor, 3), round(ceiling, 3)],
        "z_reference": "УГР",
        "truncated_to_axis_base": cfg.max_range_m is None or far < cfg.max_range_m,
    }
    if gauge.train_enabled:
        debug["corridor"]["train_gauge_m"] = {"half_width": gauge.train_half_width_m,
                                              "top_above_railhead": round(gauge.train_top_m, 3)}

    offset = axis.offset(xyz)
    above_railhead = xyz[:, 2] - (gauge.rail_head_offset_m or 0.0)


    inside = (
        (xyz[:, 0] >= near)
        & (xyz[:, 0] <= far)
        & (np.abs(offset) < gauge.half_width_at(above_railhead))
        & (xyz[:, 2] > floor)
        & (xyz[:, 2] < ceiling)
    )
    n_raw = int(inside.sum())


    in_rail_zone = gauge.in_contact_rail_zone(offset, above_railhead)
    inside &= ~in_rail_zone
    n_after_rail = int(inside.sum())


    left_run, right_run = platform_runs(offset, above_railhead, xyz[:, 0], near, far, gauge)
    required = gauge.platform_slices_required
    left_ok, right_ok = left_run >= required, right_run >= required
    in_platform = gauge.in_platform_zone(offset, above_railhead) & np.where(
        offset > 0.0, left_ok, right_ok)
    inside &= ~in_platform
    n_after_platform = int(inside.sum())
    debug["platform"] = {
        "run_m": [round(left_run * gauge.platform_slice_m, 1),
                  round(right_run * gauge.platform_slice_m, 1)],
        "required_m": gauge.platform_continuous_m,
        "cut": [bool(left_ok), bool(right_ok)],
    }


    inside &= ~rail_mask(xyz, offset, above_railhead, gauge, heads)
    debug["n_in_corridor"] = int(inside.sum())
    debug["n_cut_by_contact_rail_zone"] = n_raw - n_after_rail
    debug["n_cut_by_platform_zone"] = n_after_rail - n_after_platform
    debug["n_cut_by_rail_mask"] = n_after_platform - int(inside.sum())
    if inside.sum() < cfg.cluster_min_points:
        return FrameResult(False, None, [], debug)

    points = xyz[inside]
    lateral = offset[inside]
    distance = np.linalg.norm(points, axis=1)
    labels = _cluster(points, distance, cfg)

    order = np.argsort(labels, kind="stable")
    bounds = np.flatnonzero(np.diff(labels[order])) + 1
    groups = np.split(order, bounds)

    detections, rejected, margins, laterals, depths = [], 0, [], [], []
    for group in groups:
        if group.size < cfg.cluster_min_points:
            rejected += 1
            continue
        block = points[group]


        norm_from_base = float(distance[group].min())
        along = float(block[:, 0].min())
        if group.size < _min_points(norm_from_base, cfg):
            rejected += 1
            continue


        height = block[:, 2] - (gauge.rail_head_offset_m or 0.0)
        edges = gauge.half_width_at(height)
        inward = edges - np.abs(lateral[group])
        margin = float(inward.max())
        penetration = gauge.train_depth(lateral[group], height) if gauge.train_enabled else inward


        sigma = float(axis.sigma_edge(along, cfg.cross_check_prior_deg))
        if sigma <= 0.0:
            score = 1.0
        else:
            score = float(np.clip(margin / (cfg.confidence_sigma * sigma), 0.0, 1.0))
        detections.append(Detection(
            distance_m=along,
            centroid_xyz=tuple(block.mean(axis=0)),
            bbox_min_xyz=tuple(block.min(axis=0)),
            bbox_max_xyz=tuple(block.max(axis=0)),
            n_points=int(group.size),
            score=score,
            source="gauge",
        ))
        margins.append(round(margin, 4))
        depths.append(round(cluster_depth(penetration, cfg.depth_rank), 4))


        laterals.append(round(float(lateral[group].mean()), 4))

    order_by_distance = sorted(range(len(detections)), key=lambda i: detections[i].distance_m)


    debug["gauge_margin_m"] = [margins[i] for i in order_by_distance]
    debug["lateral_offset_m"] = [laterals[i] for i in order_by_distance]
    debug["depth_m"] = [depths[i] for i in order_by_distance]
    detections = [detections[i] for i in order_by_distance]
    debug["n_clusters"] = len(groups)
    debug["n_rejected_clusters"] = rejected
    nearest = detections[0].distance_m if detections else None
    return FrameResult(bool(detections), nearest, detections, debug)

