# Перевод выхода ядра (рамки, ось, коридор) в систему лидара для 3D-просмотра поверх облака.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.config import Config, GaugeConfig
from core.pipeline import ObstacleDetector
from core.preprocess import FrameTransform
from tools.bag_reader import iter_frames


BOX_EDGES = np.array([[0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], [7, 6], [6, 4],
                      [0, 4], [1, 5], [2, 6], [3, 7]])


@dataclass
class Lines:
    points: np.ndarray
    edges: np.ndarray
    color: tuple[float, float, float]
    name: str


@dataclass
class ViewFrame:
    index: int
    points: np.ndarray
    intensity: np.ndarray
    result: dict
    lines: list[Lines]


def _box_corners(lo, hi) -> np.ndarray:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return np.array([[x, y, z] for x in (lo[0], hi[0])
                     for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def _polyline(points: np.ndarray) -> np.ndarray:
    return np.column_stack([np.arange(len(points) - 1), np.arange(1, len(points))])


def frame_lines(result: dict, gauge: GaugeConfig, colors: dict) -> list[Lines]:
    debug = result["debug"]
    info = debug.get("frame_transform")
    if info is None:
        return []
    transform = FrameTransform(np.asarray(info["rotation"]), np.asarray(info["translation_m"]))
    railhead = gauge.rail_head_offset_m or 0.0
    out: list[Lines] = []

    def boxes(detections: list[dict], color, name) -> None:
        if not detections:
            return
        corners = np.vstack([_box_corners(d["bbox_min_xyz"], d["bbox_max_xyz"])
                             for d in detections])
        edges = np.vstack([BOX_EDGES + 8 * i for i in range(len(detections))])
        out.append(Lines(transform.inverse(corners), edges, color, name))

    boxes((debug.get("candidates_raw") or {}).get("detections", []),
          colors["candidate"], "кандидаты")
    boxes(result["detections"], colors["alarm"], "тревоги")

    axis = debug.get("axis") or {}
    line = axis.get("polyline")
    corridor = debug.get("corridor")
    if line and corridor:
        xy = np.asarray(line["points_xy_m"], float)

        on_rails = np.column_stack([xy, np.full(len(xy), railhead)])
        out.append(Lines(transform.inverse(on_rails), _polyline(on_rails),
                         colors["axis"], "ось"))

        x0, x1 = corridor["x_m"]
        inside = (xy[:, 0] >= x0) & (xy[:, 0] <= x1)
        floor = corridor["z_m"][0]
        half = float(gauge.half_width_at(np.array([floor - railhead]))[0])
        for sign in (1.0, -1.0):
            edge = np.column_stack([xy[inside, 0], xy[inside, 1] + sign * half,
                                    np.full(inside.sum(), floor)])
            if len(edge) > 1:
                out.append(Lines(transform.inverse(edge), _polyline(edge),
                                 colors["corridor"], "коридор"))


        h, w = gauge.profile_height_m, gauge.profile_half_width_m
        ring_y = np.concatenate([w, -w[::-1]])
        ring_z = np.concatenate([h, h[::-1]]) + railhead
        for x in (x0, x1):
            y_axis = float(np.interp(x, xy[:, 0], xy[:, 1]))
            ring = np.column_stack([np.full(len(ring_y), x), ring_y + y_axis, ring_z])
            edges = np.vstack([_polyline(ring), [[len(ring) - 1, 0]]])
            out.append(Lines(transform.inverse(ring), edges, colors["corridor"], "Ом"))
    return out


def load(cfg: Config, record: str, first: int, last: int, stride: int,
         colors: dict) -> list[ViewFrame]:
    detector = ObstacleDetector(cfg)
    frames: list[ViewFrame] = []
    bag_dir = Path(record) if Path(record).is_dir() else cfg.data.path(record)
    for f in iter_frames(bag_dir, cfg.bag, limit=last + 1):
        result = detector.process(f.xyz, f.intensity, f.ring, f.timestamp, f.stamp_ns)
        if f.index < first:
            continue
        keep = np.linalg.norm(f.xyz, axis=1) > cfg.preprocess.min_range_m
        as_dict = result.to_dict()
        frames.append(ViewFrame(
            f.index, np.asarray(f.xyz[keep][::stride]),
            np.asarray(f.intensity[keep][::stride]), as_dict,
            frame_lines(as_dict, cfg.gauge, colors)))
        print(f"\rядро: кадр {f.index} / {last}", end="", flush=True)
    print()
    return frames

