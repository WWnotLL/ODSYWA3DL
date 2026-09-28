# Загрузка и проверка configs/default.yaml: все параметры ядра, неизвестный или пропущенный ключ — ошибка.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class ConfigError(ValueError):
    pass


class AxisMethodError(ConfigError):
    pass


class _Section:
    def __init__(self, data: Any, path: str) -> None:
        if not isinstance(data, dict):
            raise ConfigError(f"{path}: ожидался блок ключ-значение, получено {type(data).__name__}")
        self._data = data
        self._path = path
        self._used: set[str] = set()

    def _raw(self, key: str) -> Any:
        if key not in self._data:
            raise ConfigError(
                f"{self._path}.{key}: ключ обязателен "
                f"(значений по умолчанию в коде нет, см. ALGORITHM.md §5)"
            )
        self._used.add(key)
        return self._data[key]

    def num(self, key: str) -> float:
        value = self._raw(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{self._path}.{key}: ожидалось число, получено {value!r}")
        return float(value)

    def opt_num(self, key: str) -> float | None:
        value = self._raw(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{self._path}.{key}: ожидалось число или null, получено {value!r}")
        return float(value)

    def flag(self, key: str) -> bool:
        value = self._raw(key)
        if not isinstance(value, bool):
            raise ConfigError(f"{self._path}.{key}: ожидалось true или false, получено {value!r}")
        return value

    def integer(self, key: str) -> int:
        value = self._raw(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{self._path}.{key}: ожидалось целое, получено {value!r}")
        return int(value)

    def text(self, key: str) -> str:
        value = self._raw(key)
        if not isinstance(value, str):
            raise ConfigError(f"{self._path}.{key}: ожидалась строка, получено {value!r}")
        return value

    def choice(self, key: str, allowed: frozenset[str]) -> str:
        value = self.text(key)
        if value not in allowed:
            raise ConfigError(f"{self._path}.{key}: {value!r} вне списка {sorted(allowed)}")
        return value

    def strings(self, key: str) -> tuple[str, ...]:
        value = self._raw(key)
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ConfigError(f"{self._path}.{key}: ожидался список строк, получено {value!r}")
        return tuple(value)

    def matrix3(self, key: str) -> np.ndarray:
        value = self._raw(key)
        array = np.asarray(value, dtype=float)
        if array.shape != (3, 3):
            raise ConfigError(f"{self._path}.{key}: ожидалась матрица 3x3, получена форма {array.shape}")
        return array

    def pairs(self, key: str) -> np.ndarray:
        value = self._raw(key)
        array = np.asarray(value, dtype=float)
        if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] < 2:
            raise ConfigError(
                f"{self._path}.{key}: ожидался список пар [полуширина, высота], "
                f"минимум две вершины; получена форма {array.shape}"
            )
        return array

    def section(self, key: str) -> "_Section":
        return _Section(self._raw(key), f"{self._path}.{key}")

    def close(self) -> None:
        unknown = sorted(set(self._data) - self._used)
        if unknown:
            raise ConfigError(f"{self._path}: неизвестные ключи {unknown}")


@dataclass(frozen=True)
class NumericsConfig:
    rotation_det_tol: float
    rotation_orthonormal_tol: float
    parallel_eps: float


@dataclass(frozen=True, eq=False)
class AxesConfig:
    rotation: np.ndarray


@dataclass(frozen=True)
class DualReturnConfig:
    keep: str
    secondary_min_delta_m: float
    direction_digits: int
    direction_tolerance: float


@dataclass(frozen=True)
class GroundConfig:
    zone_min_m: float
    zone_max_m: float
    min_zone_points: int
    distance_threshold_m: float
    max_tilt_deg: float
    ransac_iterations: int
    refit_iterations: int
    sample_band_percentile: float
    hypothesis_chunk: int
    seed: int
    min_inlier_fraction: float
    up_margin_m: float
    up_mass_ratio_min: float
    chord_slab_min_m: float
    chord_slab_max_m: float
    chord_min_points: int
    replan_every_n_frames: int
    refit_corridor_m: float
    refit_residual_threshold_m: float
    candidate_top_k: int
    candidate_merge_angle_deg: float
    candidate_merge_offset_m: float
    consistency_halfwidth_m: float
    consistency_max_tilt_deg: float
    history_length: int


@dataclass(frozen=True)
class PlaneGuardConfig:
    refit_limits_enabled: bool
    refit_min_inliers: int
    plausibility_enabled: bool
    max_tilt_from_base_deg: float
    max_tilt_step_deg: float
    max_offset_step_m: float
    hold_enabled: bool
    rail_check_enabled: bool
    rail_x_min_m: float
    rail_x_max_m: float
    rail_head_percentile: float
    rail_head_min_m: float
    rail_head_max_m: float

    @property
    def active(self) -> bool:
        return self.refit_limits_enabled or self.plausibility_enabled or self.rail_check_enabled


@dataclass(frozen=True, eq=False)
class PreprocessConfig:
    min_range_m: float
    dual_return: DualReturnConfig
    axes: AxesConfig
    numerics: NumericsConfig
    ground: GroundConfig
    plane_guard: PlaneGuardConfig


@dataclass(frozen=True)
class GaugeConfig:
    half_width_m: float
    height_reference: str
    z_min_m: float
    z_max_m: float
    y_center_offset_m: float | None
    rail_head_offset_m: float | None
    axis_slope: float | None
    profile_half_width_m: np.ndarray
    profile_height_m: np.ndarray
    contact_rail_half_width_min_m: float
    contact_rail_half_width_max_m: float
    contact_rail_height_max_m: float
    platform_lateral_from_m: float
    platform_height_max_m: float
    rail_mask_half_gauge_m: float
    rail_mask_half_width_m: float
    rail_mask_height_max_m: float
    rail_mask_follow_heads: bool
    rail_mask_found_half_width_m: float
    train_enabled: bool
    train_half_width_m: float
    train_top_m: float
    platform_slice_m: float
    platform_min_points_per_slice: int
    platform_continuous_m: float
    platform_outer_lateral_m: float
    platform_evidence_height_min_m: float
    platform_top_min_m: float
    platform_min_points_above_top: int

    @property
    def platform_slices_required(self) -> int:
        return max(1, int(round(self.platform_continuous_m / self.platform_slice_m)))

    def half_width_at(self, height_above_railhead: np.ndarray) -> np.ndarray:
        z = np.asarray(height_above_railhead, dtype=float)
        width = np.interp(z, self.profile_height_m, self.profile_half_width_m,
                          left=0.0, right=0.0)
        outside = (z < self.profile_height_m[0]) | (z > self.profile_height_m[-1])
        return np.where(outside, 0.0, width)

    def in_contact_rail_zone(self, lateral: np.ndarray,
                             height_above_railhead: np.ndarray) -> np.ndarray:
        magnitude = np.abs(np.asarray(lateral, dtype=float))
        return (
            (magnitude >= self.contact_rail_half_width_min_m)
            & (magnitude <= self.contact_rail_half_width_max_m)
            & (np.asarray(height_above_railhead, dtype=float) <= self.contact_rail_height_max_m)
        )

    def in_rail_mask(self, lateral: np.ndarray,
                     height_above_railhead: np.ndarray) -> np.ndarray:
        magnitude = np.abs(np.asarray(lateral, dtype=float))
        return (
            (np.abs(magnitude - self.rail_mask_half_gauge_m) <= self.rail_mask_half_width_m)
            & (np.asarray(height_above_railhead, dtype=float) <= self.rail_mask_height_max_m)
        )

    def train_depth(self, lateral: np.ndarray, height_above_railhead: np.ndarray) -> np.ndarray:
        return np.minimum(self.train_half_width_m - np.abs(np.asarray(lateral, dtype=float)),
                          self.train_top_m - np.asarray(height_above_railhead, dtype=float))

    def in_platform_zone(self, lateral: np.ndarray,
                         height_above_railhead: np.ndarray) -> np.ndarray:
        magnitude = np.abs(np.asarray(lateral, dtype=float))
        return (
            (magnitude >= self.platform_lateral_from_m)
            & (np.asarray(height_above_railhead, dtype=float) <= self.platform_height_max_m)
        )


@dataclass(frozen=True)
class AxisConfig:
    methods: tuple[str, ...]
    slice_m: float
    x_min_m: float
    x_max_m: float
    min_slices: int
    refine_passes: int

    rails_z_min_m: float
    rails_z_max_m: float
    rails_histogram_bin_m: float
    rails_peak_halfwidth_bins: int
    rails_min_points: int
    rails_search_near_m: float
    rails_search_far_m: float
    rails_half_gauge_m: float
    rails_search_halfwidth_m: float
    rails_gauge_min_m: float
    rails_gauge_max_m: float

    bed_z_min_m: float
    bed_z_max_m: float
    bed_lateral_limit_m: float
    bed_min_points: int
    bed_edge_percentile: float
    bed_width_min_m: float
    bed_width_max_m: float

    max_slope_step_deg: float
    max_slope_step_reference_advance_m: float
    max_staleness_frames: int
    noise_slope_deg: float
    clearance_m: float
    cross_check: bool
    cross_check_prior_deg: float
    extend_base: bool
    correct_extension: bool
    hold_curvature_radius_m: float
    gate_m: float
    anchor_window_frames: int
    anchor_init_max_offset_m: float
    polyline_step_m: float

    @property
    def max_hold_path_m(self) -> float:
        return float(np.sqrt(2.0 * self.hold_curvature_radius_m * self.clearance_m))


@dataclass(frozen=True)
class DetectorConfig:
    use_intensity: bool
    min_range_m: float
    max_range_m: float | None
    range_start_slack_m: float
    z_ground_margin_m: float
    cluster_eps_base_m: float
    cluster_eps_per_m: float
    cluster_min_points: int
    min_points_at_reference: float
    min_points_reference_m: float
    min_points_floor: float
    confidence_sigma: float
    depth_rank: int
    min_depth_m: float | None
    floor_above_rail_head_m: float | None
    cross_check_prior_deg: float


@dataclass(frozen=True)
class VerdictConfig:
    memory_frames: int
    latch_through_unchecked: bool


@dataclass(frozen=True)
class ConfirmConfig:
    enabled: bool
    window_frames: int
    required_frames: int
    gate_along_m: float
    gate_along_per_m: float
    gate_lateral_m: float
    gate_vertical_m: float
    unknown_advance_extra_m: float
    max_missed_frames: int
    verdict: VerdictConfig


@dataclass(frozen=True)
class OdometryConfig:
    x_min_m: float
    x_max_m: float
    bin_m: float
    detrend_window_bins: int
    max_shift_m: float
    min_points_per_bin: int
    min_prominence: float
    channels: tuple[str, ...]


@dataclass(frozen=True)
class BagConfig:
    typestore: str
    message_type: str
    preferred_topics: tuple[str, ...]
    expected_point_step: int
    expected_fields: tuple[str, ...]
    release_page_cache_every_frames: int


@dataclass(frozen=True)
class DataConfig:
    root: Path
    records: tuple[str, ...]
    obstacle_record: str

    def path(self, record: str) -> Path:
        if record not in self.records:
            raise ConfigError(f"data.records: запись {record!r} не объявлена в конфиге")
        return self.root / record

    @property
    def clean_records(self) -> tuple[str, ...]:
        return tuple(r for r in self.records if r != self.obstacle_record)


@dataclass(frozen=True)
class FormatCheckConfig:
    record: str
    frame_index: int
    expected_points: int
    expected_valid_points: int
    expected_echoes_per_shot: int
    valid_points_tol: int


@dataclass(frozen=True, eq=False)
class Config:
    preprocess: PreprocessConfig
    gauge: GaugeConfig
    axis: AxisConfig
    detector: DetectorConfig
    confirm: ConfirmConfig
    odometry: OdometryConfig
    bag: BagConfig
    data: DataConfig
    format_check: FormatCheckConfig
    raw: dict

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"конфиг не найден: {path}")
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Any) -> "Config":
        root = _Section(raw, "config")

        data_s = root.section("data")
        data = DataConfig(
            root=Path(data_s.text("root")),
            records=data_s.strings("records"),
            obstacle_record=data_s.text("obstacle_record"),
        )
        data_s.close()
        if data.obstacle_record not in data.records:
            raise ConfigError("data.obstacle_record должен входить в data.records")

        bag_s = root.section("bag")
        bag = BagConfig(
            typestore=bag_s.text("typestore"),
            message_type=bag_s.text("message_type"),
            preferred_topics=bag_s.strings("preferred_topics"),
            expected_point_step=bag_s.integer("expected_point_step"),
            expected_fields=bag_s.strings("expected_fields"),
            release_page_cache_every_frames=bag_s.integer("release_page_cache_every_frames"),
        )
        bag_s.close()
        if bag.release_page_cache_every_frames < 0:
            raise ConfigError("bag.release_page_cache_every_frames: не может быть отрицательным")

        pre_s = root.section("preprocess")
        dual_s = pre_s.section("dual_return")
        dual = DualReturnConfig(
            keep=dual_s.choice("keep", frozenset({"nearest", "farthest"})),
            secondary_min_delta_m=dual_s.num("secondary_min_delta_m"),
            direction_digits=dual_s.integer("direction_digits"),
            direction_tolerance=dual_s.num("direction_tolerance"),
        )
        dual_s.close()
        if dual.direction_digits < 1:
            raise ConfigError("preprocess.dual_return.direction_digits должен быть не меньше 1")
        if dual.direction_tolerance <= 0.0:
            raise ConfigError("preprocess.dual_return.direction_tolerance должен быть положительным")

        axes_s = pre_s.section("axes")
        axes = AxesConfig(rotation=axes_s.matrix3("rotation"))
        axes_s.close()

        num_s = pre_s.section("numerics")
        numerics = NumericsConfig(
            rotation_det_tol=num_s.num("rotation_det_tol"),
            rotation_orthonormal_tol=num_s.num("rotation_orthonormal_tol"),
            parallel_eps=num_s.num("parallel_eps"),
        )
        num_s.close()

        gr_s = pre_s.section("ground")
        ground = GroundConfig(
            zone_min_m=gr_s.num("zone_min_m"),
            zone_max_m=gr_s.num("zone_max_m"),
            min_zone_points=gr_s.integer("min_zone_points"),
            distance_threshold_m=gr_s.num("distance_threshold_m"),
            max_tilt_deg=gr_s.num("max_tilt_deg"),
            ransac_iterations=gr_s.integer("ransac_iterations"),
            refit_iterations=gr_s.integer("refit_iterations"),
            sample_band_percentile=gr_s.num("sample_band_percentile"),
            hypothesis_chunk=gr_s.integer("hypothesis_chunk"),
            seed=gr_s.integer("seed"),
            min_inlier_fraction=gr_s.num("min_inlier_fraction"),
            up_margin_m=gr_s.num("up_margin_m"),
            up_mass_ratio_min=gr_s.num("up_mass_ratio_min"),
            chord_slab_min_m=gr_s.num("chord_slab_min_m"),
            chord_slab_max_m=gr_s.num("chord_slab_max_m"),
            chord_min_points=gr_s.integer("chord_min_points"),
            replan_every_n_frames=gr_s.integer("replan_every_n_frames"),
            refit_corridor_m=gr_s.num("refit_corridor_m"),
            refit_residual_threshold_m=gr_s.num("refit_residual_threshold_m"),
            candidate_top_k=gr_s.integer("candidate_top_k"),
            candidate_merge_angle_deg=gr_s.num("candidate_merge_angle_deg"),
            candidate_merge_offset_m=gr_s.num("candidate_merge_offset_m"),
            consistency_halfwidth_m=gr_s.num("consistency_halfwidth_m"),
            consistency_max_tilt_deg=gr_s.num("consistency_max_tilt_deg"),
            history_length=gr_s.integer("history_length"),
        )
        gr_s.close()
        if ground.zone_min_m >= ground.zone_max_m:
            raise ConfigError("preprocess.ground: zone_min_m должен быть меньше zone_max_m")

        guard_s = pre_s.section("plane_guard")
        refit_s = guard_s.section("refit_limits")
        plaus_s = guard_s.section("plausibility")
        hold_s = guard_s.section("hold")
        rail_s = guard_s.section("rail_check")
        plane_guard = PlaneGuardConfig(
            refit_limits_enabled=refit_s.flag("enabled"),
            refit_min_inliers=refit_s.integer("min_inliers"),
            plausibility_enabled=plaus_s.flag("enabled"),
            max_tilt_from_base_deg=plaus_s.num("max_tilt_from_base_deg"),
            max_tilt_step_deg=plaus_s.num("max_tilt_step_deg"),
            max_offset_step_m=plaus_s.num("max_offset_step_m"),
            hold_enabled=hold_s.flag("enabled"),
            rail_check_enabled=rail_s.flag("enabled"),
            rail_x_min_m=rail_s.num("x_min_m"),
            rail_x_max_m=rail_s.num("x_max_m"),
            rail_head_percentile=rail_s.num("head_percentile"),
            rail_head_min_m=rail_s.num("head_min_m"),
            rail_head_max_m=rail_s.num("head_max_m"),
        )
        for section in (refit_s, plaus_s, hold_s, rail_s, guard_s):
            section.close()
        if plane_guard.refit_min_inliers < 3:
            raise ConfigError("preprocess.plane_guard.refit_limits.min_inliers должен быть не меньше 3")
        if min(plane_guard.max_tilt_from_base_deg, plane_guard.max_tilt_step_deg,
               plane_guard.max_offset_step_m) <= 0.0:
            raise ConfigError("preprocess.plane_guard.plausibility: пороги должны быть положительными")
        if not plane_guard.rail_x_min_m < plane_guard.rail_x_max_m:
            raise ConfigError("preprocess.plane_guard.rail_check: x_min_m должен быть меньше x_max_m")
        if not plane_guard.rail_head_min_m < plane_guard.rail_head_max_m:
            raise ConfigError("preprocess.plane_guard.rail_check: head_min_m должен быть меньше head_max_m")
        if not 0.0 < plane_guard.rail_head_percentile <= 100.0:
            raise ConfigError("preprocess.plane_guard.rail_check: head_percentile вне (0, 100]")

        preprocess = PreprocessConfig(
            min_range_m=pre_s.num("min_range_m"),
            dual_return=dual,
            axes=axes,
            numerics=numerics,
            ground=ground,
            plane_guard=plane_guard,
        )
        pre_s.close()

        gauge_s = root.section("gauge")
        profile = gauge_s.pairs("profile_mm") / 1000.0
        height_reference = gauge_s.choice("height_reference", frozenset({"rail_head", "base"}))
        rail_head_offset = gauge_s.opt_num("rail_head_offset_m")
        lift = (rail_head_offset or 0.0) if height_reference == "base" else 0.0
        profile[:, 1] -= lift
        rail_s = gauge_s.section("contact_rail_exclusion_mm")
        plat_s = gauge_s.section("platform_exclusion_mm")
        mask_s = gauge_s.section("rail_mask_mm")
        eva_s = gauge_s.section("platform_evidence")
        train_s = gauge_s.section("train")
        train_lift = ((rail_head_offset or 0.0)
                      if train_s.choice("height_reference", frozenset({"rail_head", "base"})) == "base" else 0.0)
        gauge = GaugeConfig(
            half_width_m=gauge_s.num("half_width_m"),
            height_reference=height_reference,


            z_min_m=float(profile[:, 1].min()),
            z_max_m=gauge_s.num("z_max_m") - lift,
            y_center_offset_m=gauge_s.opt_num("y_center_offset_m"),
            rail_head_offset_m=rail_head_offset,
            axis_slope=gauge_s.opt_num("axis_slope"),
            profile_half_width_m=profile[:, 0],
            profile_height_m=profile[:, 1],
            contact_rail_half_width_min_m=rail_s.num("half_width_min") / 1000.0,
            contact_rail_half_width_max_m=rail_s.num("half_width_max") / 1000.0,
            contact_rail_height_max_m=rail_s.num("height_max") / 1000.0,


            platform_lateral_from_m=(plat_s.num("lateral_from")
                                     - plat_s.num("lateral_tolerance")) / 1000.0,
            platform_height_max_m=plat_s.num("height_max") / 1000.0,
            rail_mask_half_gauge_m=mask_s.num("half_gauge") / 1000.0,
            rail_mask_half_width_m=mask_s.num("half_width") / 1000.0,
            rail_mask_height_max_m=mask_s.num("height_max") / 1000.0,
            rail_mask_follow_heads=mask_s.flag("follow_heads"),
            rail_mask_found_half_width_m=mask_s.num("found_half_width") / 1000.0,
            train_enabled=train_s.flag("enabled"),
            train_half_width_m=train_s.num("half_width_m"),
            train_top_m=train_s.num("height_m") - train_lift,
            platform_slice_m=eva_s.num("slice_m"),
            platform_min_points_per_slice=eva_s.integer("min_points_per_slice"),
            platform_continuous_m=eva_s.num("continuous_m"),
            platform_outer_lateral_m=eva_s.num("outer_lateral_mm") / 1000.0,
            platform_evidence_height_min_m=eva_s.num("height_min_mm") / 1000.0,
            platform_top_min_m=eva_s.num("top_min_mm") / 1000.0,
            platform_min_points_above_top=eva_s.integer("min_points_above_top"),
        )
        eva_s.close()
        train_s.close()
        rail_s.close()
        gauge_s.close()
        if np.any(np.diff(gauge.profile_height_m) < 0.0):
            raise ConfigError("gauge.profile_mm: вершины должны идти снизу вверх по высоте")
        if np.any(gauge.profile_half_width_m <= 0.0):
            raise ConfigError("gauge.profile_mm: полуширина должна быть положительной")
        plat_s.close(); mask_s.close()
        if gauge.platform_lateral_from_m <= 0.0:
            raise ConfigError(
                "gauge.platform_exclusion_mm: lateral_from должен быть больше допуска"
            )
        if gauge.contact_rail_half_width_min_m >= gauge.contact_rail_half_width_max_m:
            raise ConfigError(
                "gauge.contact_rail_exclusion_mm: half_width_min должен быть меньше half_width_max"
            )
        if gauge.profile_half_width_m.max() > gauge.half_width_m + 1.0:
            raise ConfigError(
                "gauge.profile_mm шире half_width_m больше чем на метр — вероятно, "
                "перепутаны миллиметры и метры"
            )

        odo_s = root.section("odometry")
        allowed_channels = frozenset({"count", "intensity", "radius"})
        channels = odo_s.strings("channels")
        unknown = sorted(set(channels) - allowed_channels)
        if unknown or not channels:
            raise ConfigError(f"odometry.channels: недопустимые каналы {unknown or '(пусто)'}")
        odometry = OdometryConfig(
            x_min_m=odo_s.num("x_min_m"),
            x_max_m=odo_s.num("x_max_m"),
            bin_m=odo_s.num("bin_m"),
            detrend_window_bins=odo_s.integer("detrend_window_bins"),
            max_shift_m=odo_s.num("max_shift_m"),
            min_points_per_bin=odo_s.integer("min_points_per_bin"),
            min_prominence=odo_s.num("min_prominence"),
            channels=channels,
        )
        odo_s.close()

        axis_s = root.section("axis")
        allowed_methods = frozenset({"rails", "bed"})
        methods = axis_s.strings("methods")
        for method in methods:
            if method not in allowed_methods:
                raise AxisMethodError(
                    f"axis.methods: {method!r} неизвестен, допустимо {sorted(allowed_methods)}"
                )
        if not methods:
            raise ConfigError("axis.methods: нужен хотя бы один метод оценки оси")
        rails_s = axis_s.section("rails")
        bed_s = axis_s.section("bed")
        hist_s = axis_s.section("history")
        axis = AxisConfig(
            methods=methods,
            slice_m=axis_s.num("slice_m"),
            x_min_m=axis_s.num("x_min_m"),
            x_max_m=axis_s.num("x_max_m"),
            min_slices=axis_s.integer("min_slices"),
            refine_passes=axis_s.integer("refine_passes"),
            rails_z_min_m=rails_s.num("z_min_m"),
            rails_z_max_m=rails_s.num("z_max_m"),
            rails_histogram_bin_m=rails_s.num("histogram_bin_m"),
            rails_peak_halfwidth_bins=rails_s.integer("peak_halfwidth_bins"),
            rails_min_points=rails_s.integer("min_points"),
            rails_search_near_m=rails_s.num("search_near_m"),
            rails_search_far_m=rails_s.num("search_far_m"),
            rails_half_gauge_m=rails_s.num("half_gauge_m"),
            rails_search_halfwidth_m=rails_s.num("search_halfwidth_m"),
            rails_gauge_min_m=rails_s.num("gauge_min_m"),
            rails_gauge_max_m=rails_s.num("gauge_max_m"),
            bed_z_min_m=bed_s.num("z_min_m"),
            bed_z_max_m=bed_s.num("z_max_m"),
            bed_lateral_limit_m=bed_s.num("lateral_limit_m"),
            bed_min_points=bed_s.integer("min_points"),
            bed_edge_percentile=bed_s.num("edge_percentile"),
            bed_width_min_m=bed_s.num("width_min_m"),
            bed_width_max_m=bed_s.num("width_max_m"),
            max_slope_step_deg=hist_s.num("max_slope_step_deg"),
            max_slope_step_reference_advance_m=hist_s.num(
                "max_slope_step_reference_advance_m"),
            max_staleness_frames=hist_s.integer("max_staleness_frames"),
            noise_slope_deg=hist_s.num("noise_slope_deg"),
            clearance_m=axis_s.num("clearance_m"),
            cross_check=axis_s.flag("cross_check"),
            cross_check_prior_deg=axis_s.num("cross_check_prior_deg"),
            extend_base=axis_s.flag("extend_base"),
            correct_extension=axis_s.flag("correct_extension"),
            hold_curvature_radius_m=hist_s.num("hold_curvature_radius_m"),
            gate_m=hist_s.num("gate_m"),
            anchor_window_frames=hist_s.integer("anchor_window_frames"),
            anchor_init_max_offset_m=hist_s.num("anchor_init_max_offset_m"),
            polyline_step_m=axis_s.num("polyline_step_m"),
        )
        rails_s.close(); bed_s.close(); hist_s.close(); axis_s.close()
        if axis.polyline_step_m <= 0.0:
            raise ConfigError("axis.polyline_step_m должен быть положительным")
        if axis.bed_width_min_m >= axis.bed_width_max_m:
            raise ConfigError("axis.bed: width_min_m должен быть меньше width_max_m")

        det_s = root.section("detector")
        detector = DetectorConfig(
            use_intensity=det_s.flag("use_intensity"),
            min_range_m=det_s.num("min_range_m"),
            max_range_m=det_s.opt_num("max_range_m"),
            range_start_slack_m=det_s.num("range_start_slack_m"),
            z_ground_margin_m=det_s.num("z_ground_margin_m"),
            cluster_eps_base_m=det_s.num("cluster_eps_base_m"),
            cluster_eps_per_m=det_s.num("cluster_eps_per_m"),
            cluster_min_points=det_s.integer("cluster_min_points"),
            min_points_at_reference=det_s.num("min_points_at_reference"),
            min_points_reference_m=det_s.num("min_points_reference_m"),
            min_points_floor=det_s.num("min_points_floor"),
            confidence_sigma=det_s.num("confidence_sigma"),
            depth_rank=det_s.integer("depth_rank"),
            min_depth_m=det_s.opt_num("min_depth_m"),
            floor_above_rail_head_m=det_s.opt_num("floor_above_rail_head_m"),
            cross_check_prior_deg=axis.cross_check_prior_deg,
        )
        det_s.close()
        if detector.depth_rank < 1 or detector.depth_rank > detector.cluster_min_points:
            raise ConfigError("detector.depth_rank должен быть от 1 до cluster_min_points")
        if detector.min_depth_m is not None and detector.min_depth_m < 0.0:
            raise ConfigError("detector.min_depth_m не может быть отрицательным")
        if gauge.train_enabled and (detector.min_depth_m is None or gauge.rail_head_offset_m is None):
            raise ConfigError("gauge.train включается только вместе с detector.min_depth_m (≥ 0) и gauge.rail_head_offset_m")
        if gauge.train_half_width_m <= 0.0 or gauge.train_top_m <= 0.0:
            raise ConfigError("gauge.train: half_width_m и высота над УГР должны быть положительными")
        if gauge.rail_mask_found_half_width_m <= 0.0:
            raise ConfigError("gauge.rail_mask_mm.found_half_width должна быть положительной")
        if detector.floor_above_rail_head_m is not None and gauge.rail_head_offset_m is None:
            raise ConfigError("detector.floor_above_rail_head_m задаётся только вместе с gauge.rail_head_offset_m")
        if detector.use_intensity:
            raise ConfigError(
                "detector.use_intensity: ядро детекции не должно использовать интенсивность — "
                "световозвращающие жилетки выучились бы вместо препятствий (ALGORITHM.md §6 п.3)"
            )

        conf_s = root.section("confirm")
        verdict_s = conf_s.section("verdict")
        confirm = ConfirmConfig(
            enabled=conf_s.flag("enabled"),
            window_frames=conf_s.integer("window_frames"),
            required_frames=conf_s.integer("required_frames"),
            gate_along_m=conf_s.num("gate_along_m"),
            gate_along_per_m=conf_s.num("gate_along_per_m"),
            gate_lateral_m=conf_s.num("gate_lateral_m"),
            gate_vertical_m=conf_s.num("gate_vertical_m"),
            unknown_advance_extra_m=conf_s.num("unknown_advance_extra_m"),
            max_missed_frames=conf_s.integer("max_missed_frames"),
            verdict=VerdictConfig(memory_frames=verdict_s.integer("memory_frames"),
                                  latch_through_unchecked=verdict_s.flag("latch_through_unchecked")),
        )
        verdict_s.close(); conf_s.close()
        if not 1 <= confirm.required_frames <= confirm.window_frames:
            raise ConfigError(
                "confirm: required_frames должно лежать между 1 и window_frames "
                f"(получено {confirm.required_frames} из {confirm.window_frames})"
            )

        fc_s = root.section("format_check")
        format_check = FormatCheckConfig(
            record=fc_s.text("record"),
            frame_index=fc_s.integer("frame_index"),
            expected_points=fc_s.integer("expected_points"),
            expected_valid_points=fc_s.integer("expected_valid_points"),
            expected_echoes_per_shot=fc_s.integer("expected_echoes_per_shot"),
            valid_points_tol=fc_s.integer("valid_points_tol"),
        )
        fc_s.close()

        root.close()
        guard = preprocess.plane_guard
        if guard.plausibility_enabled and not (guard.hold_enabled and confirm.verdict.latch_through_unchecked):
            raise ConfigError(
                "preprocess.plane_guard.plausibility включается только вместе с plane_guard.hold "
                "и confirm.verdict.latch_through_unchecked: без них отказ плоскости снимает стоп")
        return cls(
            preprocess=preprocess,
            gauge=gauge,
            axis=axis,
            detector=detector,
            confirm=confirm,
            odometry=odometry,
            bag=bag,
            data=data,
            format_check=format_check,
            raw=raw,
        )

    def snapshot(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.raw, handle, allow_unicode=True, sort_keys=False)

