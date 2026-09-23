# Контракт выхода ядра (Detection, FrameResult), который получает нода на каждый кадр.

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np


SOURCES = frozenset({"gauge", "background", "fused"})

Vec3 = tuple[float, float, float]


def _as_vec3(value: Any, name: str) -> Vec3:
    seq = tuple(float(v) for v in value)
    if len(seq) != 3:
        raise ValueError(f"{name}: ожидались три координаты, получено {len(seq)}")
    return seq


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


@dataclass(frozen=True)
class Detection:
    distance_m: float
    centroid_xyz: Vec3
    bbox_min_xyz: Vec3
    bbox_max_xyz: Vec3
    n_points: int
    score: float
    source: str

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        set_(self, "distance_m", float(self.distance_m))
        set_(self, "centroid_xyz", _as_vec3(self.centroid_xyz, "centroid_xyz"))
        set_(self, "bbox_min_xyz", _as_vec3(self.bbox_min_xyz, "bbox_min_xyz"))
        set_(self, "bbox_max_xyz", _as_vec3(self.bbox_max_xyz, "bbox_max_xyz"))
        set_(self, "n_points", int(self.n_points))
        set_(self, "score", float(self.score))
        if self.source not in SOURCES:
            raise ValueError(
                f"Detection.source: {self.source!r} вне контракта, допустимо {sorted(SOURCES)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "distance_m": self.distance_m,
            "centroid_xyz": list(self.centroid_xyz),
            "bbox_min_xyz": list(self.bbox_min_xyz),
            "bbox_max_xyz": list(self.bbox_max_xyz),
            "n_points": self.n_points,
            "score": self.score,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Detection":
        return cls(**data)


@dataclass(frozen=True)
class FrameResult:
    obstacle_found: bool
    nearest_distance_m: float | None
    detections: list[Detection]
    debug: dict

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        set_(self, "obstacle_found", bool(self.obstacle_found))
        if self.nearest_distance_m is not None:
            set_(self, "nearest_distance_m", float(self.nearest_distance_m))
        set_(self, "detections", list(self.detections))
        set_(self, "debug", dict(self.debug))

    def to_dict(self) -> dict[str, Any]:
        return {
            "obstacle_found": self.obstacle_found,
            "nearest_distance_m": self.nearest_distance_m,
            "detections": [d.to_dict() for d in self.detections],
            "debug": to_jsonable(self.debug),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrameResult":
        return cls(
            obstacle_found=data["obstacle_found"],
            nearest_distance_m=data["nearest_distance_m"],
            detections=[Detection.from_dict(d) for d in data["detections"]],
            debug=data["debug"],
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "FrameResult":
        return cls.from_dict(json.loads(text))

