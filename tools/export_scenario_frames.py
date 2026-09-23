# Выгрузка кадров со вставленной фигурой и выхода ядра в PLY для 3D-просмотра.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from core.config import Config
from core.pipeline import ObstacleDetector
from tools.scenarios import build_parser, iter_scenario_frames
from tools.view_geometry import Lines, frame_lines
from tools.visualize_run import load_viz


COLOR_SCALE = 255


def write_cloud_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    data = np.empty(xyz.shape[0], dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                         ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    colors = np.clip(np.round(rgb * COLOR_SCALE), 0, COLOR_SCALE).astype(np.uint8)
    data["red"], data["green"], data["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {xyz.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(data.tobytes())


def write_lines_ply(path: Path, groups: list[Lines]) -> None:
    points, colors, edges, offset = [], [], [], 0
    for group in groups:
        points.append(group.points)
        colors.append(np.tile(np.asarray(group.color, float), (len(group.points), 1)))
        edges.append(group.edges + offset)
        offset += len(group.points)
    xyz = np.vstack(points) if points else np.empty((0, 3))
    rgb = np.vstack(colors) if colors else np.empty((0, 3))
    idx = np.vstack(edges) if edges else np.empty((0, 2), dtype=int)
    rgb8 = np.clip(np.round(rgb * COLOR_SCALE), 0, COLOR_SCALE).astype(int)
    lines = [
        "ply", "format ascii 1.0", f"element vertex {len(xyz)}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
        f"element edge {len(idx)}", "property int vertex1", "property int vertex2",
        "end_header",
    ]
    lines += [f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}" for p, c in zip(xyz, rgb8)]
    lines += [f"{a} {b}" for a, b in idx]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def cloud_colors(intensity: np.ndarray, body: np.ndarray, body_color) -> np.ndarray:
    grey = np.clip(np.asarray(intensity, float) / COLOR_SCALE, 0.0, 1.0)
    rgb = np.repeat(grey[:, None], 3, axis=1)
    rgb[body] = np.asarray(body_color, float)
    return rgb


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--viz", type=Path, required=True)
    parser.add_argument("--story", required=True)
    parser.add_argument("--first", type=int, default=None, help="перекрыть начало диапазона")
    parser.add_argument("--last", type=int, default=None, help="перекрыть конец диапазона")
    parser.add_argument("--every", type=int, default=None, help="перекрыть frames_every")
    args = parser.parse_args(argv)

    viz = load_viz(args.viz)
    story, export = viz["stories"][args.story], viz["export"]
    if "scenario_argv" not in story:
        raise SystemExit(f"сюжет {args.story} — исходная запись без вставки; "
                         "его смотреть через view.py --detect")
    cfg = Config.from_yaml(viz["core_config"])
    first = story["frames"][0] if args.first is None else args.first
    last = story["frames"][1] if args.last is None else args.last
    every = export["frames_every"] if args.every is None else args.every
    colors = {k: tuple(v) for k, v in viz["view3d"]["line_colors"].items()}
    scenario = build_parser().parse_args(story["scenario_argv"])
    out_dir = Path(export["output_root"]) / args.story
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = ObstacleDetector(cfg)
    index = []
    for item in iter_scenario_frames(cfg, scenario, raw=True):
        f = item.frame
        result = detector.process(item.injected, f.intensity, f.ring, f.timestamp, f.stamp_ns)
        if f.index > last:
            break
        if f.index < first or (f.index - first) % every:
            continue
        raw = np.asarray(item.injected, dtype=float)
        valid = np.linalg.norm(raw, axis=1) > cfg.preprocess.min_range_m
        body = (np.linalg.norm(raw - np.asarray(f.xyz, float), axis=1)
                > export["moved_tolerance_m"])
        as_dict = result.to_dict()
        write_cloud_ply(out_dir / f"frame_{f.index:04d}.ply", raw[valid],
                        cloud_colors(f.intensity[valid], body[valid], export["body_color"]))
        write_lines_ply(out_dir / f"frame_{f.index:04d}_lines.ply",
                        frame_lines(as_dict, cfg.gauge, colors))
        axis = as_dict["debug"].get("axis") or {}
        index.append({
            "frame": int(f.index),
            "figure_centre_track_m": None if item.centre is None
            else [round(float(v), 2) for v in item.centre],
            "figure_points": int(body[valid].sum()),
            "run": item.run_index,
            "axis_state": axis.get("state"),
            "axis_reason": axis.get("reason"),
            "candidates": len((as_dict["debug"].get("candidates_raw") or {}).get("detections", [])),
            "alarms": len(as_dict["detections"]),
            "nearest_distance_m": as_dict["nearest_distance_m"],
        })
        print(f"\rкадр {f.index}: точек фигуры {index[-1]['figure_points']}, "
              f"тревог {index[-1]['alarms']}", end="", flush=True)
    (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    size = sum(p.stat().st_size for p in out_dir.glob("*.ply")) / 2 ** 20
    print(f"\nзаписано кадров: {len(index)} → {out_dir} ({size:.0f} МБ)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

