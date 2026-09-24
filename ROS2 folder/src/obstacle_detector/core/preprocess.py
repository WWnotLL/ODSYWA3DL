# Препроцессинг кадра: пустые лучи, двойное эхо, плоскость пола (RANSAC), поворот облака в систему пути.

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.config import GroundConfig, NumericsConfig, PreprocessConfig


class PreprocessError(ValueError):
    pass


def _as_points(xyz: np.ndarray) -> np.ndarray:
    array = np.asarray(xyz)
    if array.ndim != 2 or array.shape[1] != 3:
        raise PreprocessError(f"ожидался массив точек формы (N, 3), получена {array.shape}")
    return array


def ranges_m(xyz: np.ndarray) -> np.ndarray:
    points = _as_points(xyz)
    return np.sqrt(np.einsum("ij,ij->i", points, points))


def valid_mask(xyz: np.ndarray, min_range_m: float) -> np.ndarray:
    points = _as_points(xyz)
    squared = np.einsum("ij,ij->i", points, points)
    return squared > float(min_range_m) ** 2


def shot_key(ring: np.ndarray, timestamp: np.ndarray) -> np.ndarray:
    ring = np.asarray(ring)
    timestamp = np.asarray(timestamp)
    if ring.shape != timestamp.shape or ring.ndim != 1:
        raise PreprocessError(
            f"ring и timestamp должны быть одномерными одной длины, получено "
            f"{ring.shape} и {timestamp.shape}"
        )
    _, time_index = np.unique(timestamp, return_inverse=True)
    n_times = time_index.max() + 1 if time_index.size else 1
    return ring.astype(np.int64) * np.int64(n_times) + time_index.astype(np.int64)


def echoes_per_shot(ring: np.ndarray, timestamp: np.ndarray) -> np.ndarray:
    if ring.size == 0:
        return np.zeros(0, dtype=np.int64)
    _, counts = np.unique(shot_key(ring, timestamp), return_counts=True)
    return counts


def _regular_pair_stride(key: np.ndarray) -> int | None:
    n = key.size
    if n < 2 or n % 2:
        return None
    same = np.flatnonzero(key == key[0])
    if same.size != 2:
        return None
    stride = int(same[1] - same[0])
    if stride < 1 or n % (2 * stride):
        return None
    blocks = key.reshape(-1, 2, stride)
    if not np.array_equal(blocks[:, 0, :], blocks[:, 1, :]):
        return None
    return stride


def _collapse_by_stride(
    ranges: np.ndarray, stride: int, prefer_smaller: bool
) -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(ranges.size, dtype=np.int64).reshape(-1, 2, stride)
    left, right = index[:, 0, :].ravel(), index[:, 1, :].ravel()
    if prefer_smaller:
        take_left = ranges[left] <= ranges[right]
    else:
        take_left = ranges[left] >= ranges[right]

    return np.where(take_left, left, right), np.where(take_left, right, left)


def _partners_by_sort(order: np.ndarray, is_first: np.ndarray) -> np.ndarray:
    position = np.flatnonzero(is_first)
    following = position + 1
    exists = (following < order.size) & ~is_first[np.clip(following, 0, order.size - 1)]
    partner = np.full(position.size, -1, dtype=np.int64)
    partner[exists] = order[following[exists]]
    return partner


def _collapse_by_sort(
    ranges: np.ndarray, key: np.ndarray, prefer_smaller: bool
) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((ranges if prefer_smaller else -ranges, key))
    ordered_key = key[order]
    is_first = np.empty(ordered_key.size, dtype=bool)
    is_first[0] = True
    np.not_equal(ordered_key[1:], ordered_key[:-1], out=is_first[1:])
    return order[is_first], _partners_by_sort(order, is_first)


def collapse_dual_returns(
    ranges: np.ndarray,
    ring: np.ndarray,
    timestamp: np.ndarray,
    keep: str,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    ranges = np.asarray(ranges)
    if ranges.ndim != 1 or ranges.shape != np.asarray(ring).shape:
        raise PreprocessError("ranges, ring и timestamp должны быть одной длины")
    if ranges.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    if keep == "nearest":
        prefer_smaller = True
    elif keep == "farthest":
        prefer_smaller = False
    else:
        raise PreprocessError(f"dual_return.keep: {keep!r} не поддерживается")

    if valid is not None:
        worst = np.inf if prefer_smaller else -np.inf
        ranges = np.where(np.asarray(valid), ranges.astype(np.float64), worst)

    key = shot_key(ring, timestamp)
    stride = _regular_pair_stride(key)
    if stride is not None:
        return _collapse_by_stride(ranges, stride, prefer_smaller)
    return _collapse_by_sort(ranges, key, prefer_smaller)


def check_proper_rotation(rotation: np.ndarray, numerics: NumericsConfig) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise PreprocessError(f"матрица разворота: ожидалась форма (3, 3), получена {matrix.shape}")
    orthonormality = np.abs(matrix @ matrix.T - np.eye(3)).max()
    if orthonormality > numerics.rotation_orthonormal_tol:
        raise PreprocessError(
            f"матрица разворота не ортонормирована: max|R·Rᵀ − I| = {orthonormality:.3e}"
        )
    determinant = float(np.linalg.det(matrix))
    if abs(determinant - 1.0) > numerics.rotation_det_tol:
        raise PreprocessError(
            f"матрица разворота не является собственным вращением: det = {determinant:.6f}. "
            "При det = −1 это отражение: лево и право поменяются местами."
        )
    return matrix


def to_rep103(xyz: np.ndarray, rotation: np.ndarray, numerics: NumericsConfig) -> np.ndarray:
    points = _as_points(xyz)
    matrix = check_proper_rotation(rotation, numerics)
    return points @ matrix.T.astype(points.dtype, copy=False)


def rotation_between(source: np.ndarray, target: np.ndarray, numerics: NumericsConfig) -> np.ndarray:
    a = np.asarray(source, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    axis = np.cross(a, b)
    cosine = float(a @ b)
    sine = float(np.linalg.norm(axis))
    if sine <= numerics.parallel_eps:
        if cosine > 0.0:
            return np.eye(3)


        helper = np.zeros(3)
        helper[int(np.argmin(np.abs(a)))] = 1.0
        perpendicular = np.cross(a, helper)
        perpendicular /= np.linalg.norm(perpendicular)
        return 2.0 * np.outer(perpendicular, perpendicular) - np.eye(3)
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))


@dataclass(frozen=True, eq=False)
class GroundPlane:
    normal: np.ndarray
    offset: float
    inlier_fraction: float
    up_mass_ratio: float
    up_source: str
    n_zone_points: int
    residual_m: float
    source: str
    n_candidates: int
    n_rejected: int

    def signed_distance(self, xyz: np.ndarray) -> np.ndarray:
        return _as_points(xyz).astype(np.float64) @ self.normal + self.offset


def _fit_plane_least_squares(points: np.ndarray) -> tuple[np.ndarray, float]:
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vt[-1]
    normal = normal / np.linalg.norm(normal)
    return normal, float(-normal @ centroid)


def _up_from_chord(
    points: np.ndarray, normal: np.ndarray, offset: float, cfg: GroundConfig
) -> tuple[int, float]:
    slab = (points[:, 0] >= cfg.chord_slab_min_m) & (points[:, 0] <= cfg.chord_slab_max_m)
    section = points[slab]
    if section.shape[0] < cfg.chord_min_points:
        raise PreprocessError(
            "направление «вверх» неоднозначно, а в срезе для резервного критерия "
            f"всего {section.shape[0]} точек при минимуме {cfg.chord_min_points}"
        )
    y, z = section[:, 1], section[:, 2]

    design = np.stack([2.0 * y, 2.0 * z, np.ones_like(y)], axis=1)
    solution, *_ = np.linalg.lstsq(design, y * y + z * z, rcond=None)
    center_y, center_z, k = solution
    radius_squared = k + center_y * center_y + center_z * center_z
    if radius_squared <= 0.0:
        raise PreprocessError("резервный критерий: окружность в сечение не вписывается")

    center = np.array([float(section[:, 0].mean()), float(center_y), float(center_z)])
    side = float(center @ normal + offset)
    if abs(side) <= cfg.up_margin_m:
        raise PreprocessError(
            f"резервный критерий: центр сечения лежит на самой плоскости (|отступ| = {abs(side):.3f} м)"
        )
    return (1 if side > 0.0 else -1), abs(side)


def _determine_up(
    points: np.ndarray, normal: np.ndarray, offset: float, cfg: GroundConfig
) -> tuple[int, float, str]:
    side = points @ normal + offset
    above = int(np.count_nonzero(side > cfg.up_margin_m))
    below = int(np.count_nonzero(side < -cfg.up_margin_m))
    heavier, lighter = max(above, below), min(above, below)
    ratio = float(heavier) / float(lighter) if lighter > 0 else float("inf")
    if heavier == 0:
        raise PreprocessError("по обе стороны плоскости нет точек за пределами полосы up_margin_m")
    if ratio >= cfg.up_mass_ratio_min:
        return (1 if above > below else -1), ratio, "mass_ratio"
    sign, _ = _up_from_chord(points, normal, offset, cfg)
    return sign, ratio, "chord"


def _vertical_bands(zone: np.ndarray, cfg: GroundConfig) -> list[np.ndarray]:
    vertical = zone[:, 2]
    low = float(np.percentile(vertical, cfg.sample_band_percentile))
    high = float(np.percentile(vertical, 100.0 - cfg.sample_band_percentile))
    return [zone[vertical <= low], zone[vertical >= high]]


def _plane_hypotheses(
    pool: np.ndarray, cfg: GroundConfig, generator: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    triplets = generator.integers(0, pool.shape[0], size=(cfg.ransac_iterations, 3))
    a, b, c = pool[triplets[:, 0]], pool[triplets[:, 1]], pool[triplets[:, 2]]
    normals = np.cross(b - a, c - a)
    lengths = np.linalg.norm(normals, axis=1)
    nondegenerate = lengths > 0.0
    tilt_cosine = np.zeros_like(lengths)
    np.divide(np.abs(normals[:, 2]), lengths, out=tilt_cosine, where=nondegenerate)
    acceptable = nondegenerate & (tilt_cosine >= np.cos(np.radians(cfg.max_tilt_deg)))
    if not acceptable.any():
        return np.zeros((0, 3)), np.zeros(0)
    normals = normals[acceptable] / lengths[acceptable, None]
    return normals, -np.einsum("ij,ij->i", normals, a[acceptable])


def _score_hypotheses(
    zone: np.ndarray, normals: np.ndarray, offsets: np.ndarray, cfg: GroundConfig
) -> np.ndarray:
    counts = np.empty(normals.shape[0], dtype=np.int64)
    for start in range(0, normals.shape[0], cfg.hypothesis_chunk):
        stop = start + cfg.hypothesis_chunk
        distance = np.abs(zone @ normals[start:stop].T + offsets[start:stop])
        counts[start:stop] = np.count_nonzero(distance <= cfg.distance_threshold_m, axis=0)
    return counts


def _residual(points: np.ndarray, normal: np.ndarray, offset: float) -> float:
    if points.shape[0] == 0:
        return float("inf")
    return float(np.sqrt(np.mean((points @ normal + offset) ** 2)))


def _refit(
    zone: np.ndarray, normal: np.ndarray, offset: float, cfg: GroundConfig
) -> tuple[np.ndarray, float, np.ndarray]:
    inliers = np.abs(zone @ normal + offset) <= cfg.distance_threshold_m
    for _ in range(cfg.refit_iterations):
        if int(np.count_nonzero(inliers)) < 3:
            break
        normal, offset = _fit_plane_least_squares(zone[inliers])
        inliers = np.abs(zone @ normal + offset) <= cfg.distance_threshold_m
    return normal, offset, inliers


@dataclass(frozen=True, eq=False)
class PlaneCandidate:
    normal: np.ndarray
    offset: float
    inlier_fraction: float
    residual_m: float


def _same_plane(a: PlaneCandidate, b: PlaneCandidate, cfg: GroundConfig) -> bool:
    angle = np.degrees(np.arccos(np.clip(abs(float(a.normal @ b.normal)), -1.0, 1.0)))
    return angle <= cfg.candidate_merge_angle_deg and abs(a.offset - b.offset) <= cfg.candidate_merge_offset_m


def estimate_ground_plane(
    xyz: np.ndarray,
    cfg: GroundConfig,
    expected_offset: float | None = None,
    expected_normal: np.ndarray | None = None,
) -> GroundPlane:
    points = _as_points(xyz).astype(np.float64)
    in_zone = (points[:, 0] >= cfg.zone_min_m) & (points[:, 0] <= cfg.zone_max_m)
    zone = points[in_zone]
    if zone.shape[0] < cfg.min_zone_points:
        raise PreprocessError(
            f"в ближней зоне {cfg.zone_min_m}–{cfg.zone_max_m} м всего {zone.shape[0]} точек "
            f"при минимуме {cfg.min_zone_points}"
        )

    generator = np.random.default_rng(cfg.seed)
    pools = [pool for pool in _vertical_bands(zone, cfg) if pool.shape[0] >= 3]
    hypotheses = [_plane_hypotheses(pool, cfg, generator) for pool in pools]
    normals = np.concatenate([n for n, _ in hypotheses if n.shape[0]], axis=0) if hypotheses else np.zeros((0, 3))
    offsets = np.concatenate([o for n, o in hypotheses if n.shape[0]], axis=0) if hypotheses else np.zeros(0)
    if normals.shape[0] == 0:
        raise PreprocessError(
            f"ни одна из гипотез RANSAC не прошла ограничение по наклону {cfg.max_tilt_deg}°"
        )

    counts = _score_hypotheses(zone, normals, offsets, cfg)
    ranked = np.argsort(counts)[::-1][: cfg.candidate_top_k]


    top_normal, top_offset, _ = _refit(zone, normals[ranked[0]], offsets[ranked[0]], cfg)
    sign, ratio, up_source = _determine_up(points, top_normal, top_offset, cfg)
    up = sign * top_normal
    if expected_normal is not None and float(up @ expected_normal) < 0.0:
        expected_normal, expected_offset = -expected_normal, -expected_offset

    candidates: list[PlaneCandidate] = []


    if expected_offset is not None and expected_normal is not None:
        near = np.abs(zone @ expected_normal + expected_offset) <= cfg.refit_corridor_m
        if int(np.count_nonzero(near)) >= 3:
            seed_normal, seed_offset = _fit_plane_least_squares(zone[near])
            if float(seed_normal @ expected_normal) < 0.0:
                seed_normal, seed_offset = -seed_normal, -seed_offset
            seed_normal, seed_offset, seed_inliers = _refit(zone, seed_normal, seed_offset, cfg)
            candidates.append(
                PlaneCandidate(
                    normal=seed_normal,
                    offset=seed_offset,
                    inlier_fraction=float(np.count_nonzero(seed_inliers)) / float(zone.shape[0]),
                    residual_m=_residual(zone[seed_inliers], seed_normal, seed_offset),
                )
            )

    for index in ranked:
        normal, offset, inliers = _refit(zone, normals[index], offsets[index], cfg)
        if float(normal @ up) < 0.0:
            normal, offset = -normal, -offset
        candidate = PlaneCandidate(
            normal=normal,
            offset=offset,
            inlier_fraction=float(np.count_nonzero(inliers)) / float(zone.shape[0]),
            residual_m=_residual(zone[inliers], normal, offset),
        )
        if not any(_same_plane(candidate, known, cfg) for known in candidates):
            candidates.append(candidate)

    n_candidates = len(candidates)
    supported = [c for c in candidates if c.inlier_fraction >= cfg.min_inlier_fraction]
    if not supported:
        best = max(candidates, key=lambda c: c.inlier_fraction)
        raise PreprocessError(
            f"плоскость основания не найдена: лучшая доля опорных точек {best.inlier_fraction:.3f} "
            f"ниже порога {cfg.min_inlier_fraction}"
        )

    n_rejected = n_candidates - len(supported)
    if expected_offset is not None:
        def agrees(candidate: PlaneCandidate) -> bool:
            if abs(candidate.offset - expected_offset) > cfg.consistency_halfwidth_m:
                return False
            if expected_normal is None:
                return True
            cosine = abs(float(candidate.normal @ expected_normal))
            angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
            return angle <= cfg.consistency_max_tilt_deg

        consistent = [c for c in supported if agrees(c)]


        if consistent:
            n_rejected += len(supported) - len(consistent)
            supported = consistent


    chosen = max(supported, key=lambda c: c.offset)
    return GroundPlane(
        normal=chosen.normal,
        offset=chosen.offset,
        inlier_fraction=chosen.inlier_fraction,
        up_mass_ratio=ratio,
        up_source=up_source,
        n_zone_points=int(zone.shape[0]),
        residual_m=chosen.residual_m,
        source="ransac",
        n_candidates=n_candidates,
        n_rejected=n_rejected,
    )


def level_to_ground(
    xyz: np.ndarray, plane: GroundPlane, numerics: NumericsConfig
) -> tuple[np.ndarray, np.ndarray]:
    points = _as_points(xyz)
    rotation = rotation_between(plane.normal, np.array([0.0, 0.0, 1.0]), numerics)
    leveled = points @ rotation.T.astype(points.dtype, copy=False)
    leveled[:, 2] += plane.offset
    return leveled, rotation


@dataclass(frozen=True, eq=False)
class FrameTransform:
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, raw: np.ndarray) -> np.ndarray:
        return _as_points(raw).astype(np.float64) @ self.rotation.T + self.translation

    def inverse(self, track: np.ndarray) -> np.ndarray:
        return (_as_points(track).astype(np.float64) - self.translation) @ self.rotation

    def to_debug(self) -> dict:
        return {
            "from": "lidar",
            "to": "track_leveled",
            "convention": "p_track = R @ p_lidar + t",
            "rotation": self.rotation.tolist(),
            "translation_m": self.translation.tolist(),
        }


@dataclass(frozen=True, eq=False)
class PreprocessResult:
    xyz: np.ndarray
    intensity: np.ndarray
    ring: np.ndarray
    t_rel: np.ndarray
    secondary_xyz: np.ndarray
    secondary_intensity: np.ndarray
    secondary_ring: np.ndarray
    secondary_t_rel: np.ndarray
    plane: GroundPlane
    stats: dict[str, Any]
    transform: FrameTransform


def preprocess_frame(
    xyz: np.ndarray,
    intensity: np.ndarray,
    ring: np.ndarray,
    timestamp: np.ndarray,
    cfg: PreprocessConfig,
    *,
    plane: GroundPlane | None = None,
    tracker: "GroundTracker | None" = None,
) -> PreprocessResult:
    if plane is not None and tracker is not None:
        raise PreprocessError("укажите либо plane, либо tracker, но не оба сразу")
    points = _as_points(xyz)
    intensity = np.asarray(intensity)
    ring = np.asarray(ring)
    timestamp = np.asarray(timestamp)
    if not (points.shape[0] == intensity.shape[0] == ring.shape[0] == timestamp.shape[0]):
        raise PreprocessError("xyz, intensity, ring и timestamp должны быть одной длины")

    n_input = int(points.shape[0])
    keep = valid_mask(points, cfg.min_range_m)
    n_valid = int(np.count_nonzero(keep))
    if n_valid == 0:
        raise PreprocessError("после отсева пустых лучей в кадре не осталось точек")


    point_ranges = ranges_m(points)
    primary, partner = collapse_dual_returns(
        point_ranges, ring, timestamp, cfg.dual_return.keep, valid=keep
    )
    alive = keep[primary]
    primary, partner = primary[alive], partner[alive]


    has_partner = partner >= 0
    safe_partner = np.where(has_partner, partner, 0)
    divergent = (
        has_partner
        & keep[safe_partner]
        & (np.abs(point_ranges[safe_partner] - point_ranges[primary])
           > cfg.dual_return.secondary_min_delta_m)
    )
    secondary = partner[divergent]


    time_zero = float(timestamp[keep].min())

    rotated = to_rep103(points, cfg.axes.rotation, cfg.numerics)
    primary_xyz, secondary_xyz = rotated[primary], rotated[secondary]

    if tracker is not None:
        plane = tracker.update(primary_xyz)
    elif plane is None:
        plane = estimate_ground_plane(primary_xyz, cfg.ground)
    primary_xyz, level = level_to_ground(primary_xyz, plane, cfg.numerics)
    secondary_xyz, _ = level_to_ground(secondary_xyz, plane, cfg.numerics)
    transform = FrameTransform(
        rotation=level @ check_proper_rotation(cfg.axes.rotation, cfg.numerics),
        translation=np.array([0.0, 0.0, float(plane.offset)]),
    )

    stats = {
        "n_input": n_input,
        "n_valid": n_valid,
        "empty_fraction": 1.0 - n_valid / n_input if n_input else 0.0,
        "n_primary": int(primary.size),
        "n_secondary_divergent": int(secondary.size),
        "divergent_fraction": float(secondary.size) / float(primary.size) if primary.size else 0.0,
        "plane_candidates": plane.n_candidates,
        "plane_rejected": plane.n_rejected,
        "plane_inlier_fraction": plane.inlier_fraction,
        "plane_up_mass_ratio": plane.up_mass_ratio,
        "plane_up_source": plane.up_source,
        "plane_zone_points": plane.n_zone_points,
    }
    return PreprocessResult(
        xyz=primary_xyz,
        intensity=intensity[primary],
        ring=ring[primary],
        t_rel=timestamp[primary] - time_zero,
        secondary_xyz=secondary_xyz,
        secondary_intensity=intensity[secondary],
        secondary_ring=ring[secondary],
        secondary_t_rel=timestamp[secondary] - time_zero,
        plane=plane,
        stats=stats,
        transform=transform,
    )


class GroundTracker:
    def __init__(self, cfg: GroundConfig) -> None:
        self._cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._plane: GroundPlane | None = None
        self._since_full = 0


        self._history: deque[float] = deque(maxlen=self._cfg.history_length)
        self._normals: deque[np.ndarray] = deque(maxlen=self._cfg.history_length)
        self.n_full = 0
        self.n_refit = 0
        self.n_forced = 0
        self.n_rejected_candidates = 0

    @property
    def expected_offset(self) -> float | None:
        return float(np.median(self._history)) if self._history else None

    @property
    def expected_normal(self) -> np.ndarray | None:
        if not self._normals:
            return None
        median = np.median(np.stack(self._normals), axis=0)
        norm = np.linalg.norm(median)
        return median / norm if norm > 0.0 else None

    @property
    def plane(self) -> GroundPlane | None:
        return self._plane

    def update(self, xyz: np.ndarray) -> GroundPlane:
        cfg = self._cfg
        if self._plane is None or self._since_full >= cfg.replan_every_n_frames:
            return self._full(xyz)

        points = _as_points(xyz).astype(np.float64)
        in_zone = (points[:, 0] >= cfg.zone_min_m) & (points[:, 0] <= cfg.zone_max_m)
        zone = points[in_zone]
        previous = self._plane
        near = np.abs(zone @ previous.normal + previous.offset) <= cfg.refit_corridor_m
        if int(np.count_nonzero(near)) < 3:
            self.n_forced += 1
            return self._full(xyz)

        normal, offset = _fit_plane_least_squares(zone[near])
        if normal @ previous.normal < 0.0:
            normal, offset = -normal, -offset
        normal, offset, inliers = _refit(zone, normal, offset, cfg)
        if normal @ previous.normal < 0.0:
            normal, offset = -normal, -offset


        if int(np.count_nonzero(inliers)) < 3:
            self.n_forced += 1
            return self._full(xyz)
        residual = _residual(zone[inliers], normal, offset)
        inlier_fraction = float(np.count_nonzero(inliers)) / float(max(zone.shape[0], 1))


        if residual > cfg.refit_residual_threshold_m or inlier_fraction < cfg.min_inlier_fraction:
            self.n_forced += 1
            return self._full(xyz)

        self._plane = GroundPlane(
            normal=normal,
            offset=offset,
            inlier_fraction=inlier_fraction,
            up_mass_ratio=previous.up_mass_ratio,
            up_source=previous.up_source,
            n_zone_points=int(zone.shape[0]),
            residual_m=residual,
            source="refit",
            n_candidates=1,
            n_rejected=0,
        )
        self._since_full += 1
        self.n_refit += 1
        self._history.append(offset)
        self._normals.append(normal)
        return self._plane

    def _full(self, xyz: np.ndarray) -> GroundPlane:
        plane = estimate_ground_plane(
            xyz,
            self._cfg,
            expected_offset=self.expected_offset,
            expected_normal=self.expected_normal,
        )
        self._plane = plane
        self._since_full = 0
        self.n_full += 1
        self.n_rejected_candidates += plane.n_rejected
        self._history.append(plane.offset)
        self._normals.append(plane.normal)
        return plane

