# Офлайн-разметка целевой записи по световозвращающим жилетам — только для оценки, не для детекции.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN

from core.config import Config
from core.preprocess import GroundTracker, preprocess_frame
from tools.bag_reader import count_frames, iter_frames


def detect(cfg: Config, record: str, threshold: float, min_x: float) -> list[dict]:
    tracker = GroundTracker(cfg.preprocess.ground)
    found = []
    for frame in iter_frames(cfg.data.path(record), cfg.bag):
        result = preprocess_frame(
            frame.xyz, frame.intensity, frame.ring, frame.timestamp,
            cfg.preprocess, tracker=tracker,
        )
        bright = (result.intensity > threshold) & (result.xyz[:, 0] > min_x)
        points = result.xyz[bright]
        if points.shape[0] < 3:
            continue
        labels = DBSCAN(eps=1.5, min_samples=3).fit_predict(points)
        for label in set(labels) - {-1}:
            group = points[labels == label]
            found.append({
                "frame": int(frame.index),
                "stamp_ns": int(frame.stamp_ns),
                "x": float(group[:, 0].mean()),
                "y": float(group[:, 1].mean()),
                "z": float(group[:, 2].mean()),
                "n_points": int(group.shape[0]),
            })
    return found


def split_static(found: list[dict], tolerance: float, min_share: float, n_frames: int) -> tuple:
    if not found:
        return [], []
    positions = np.array([[row["x"], row["y"]] for row in found])
    labels = DBSCAN(eps=tolerance, min_samples=3).fit_predict(positions)

    static, moving = [], []
    for label in set(labels):
        members = [row for row, lab in zip(found, labels) if lab == label]
        if label == -1:
            moving.extend(members)
            continue
        group = np.array([[row["x"], row["y"]] for row in members])
        spread = float(np.hypot(*(group.max(axis=0) - group.min(axis=0))))
        seen = len({row["frame"] for row in members})
        if seen >= min_share * n_frames and spread <= tolerance:
            static.append({
                "x": float(group[:, 0].mean()),
                "y": float(group[:, 1].mean()),
                "seen": seen,
                "spread_m": round(spread, 3),
            })
        else:
            moving.extend(members)
    return sorted(static, key=lambda i: i["x"]), sorted(moving, key=lambda r: (r["frame"], r["x"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--record", required=True)
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--threshold", type=float, default=200.0)
    parser.add_argument("--min-x", type=float, default=5.0)
    parser.add_argument("--static-tolerance", type=float, default=1.0)
    parser.add_argument("--static-share", type=float, default=0.8)
    parser.add_argument("--out", default="runs/oracle")
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    if cfg.detector.use_intensity:
        raise SystemExit("конфиг разрешает интенсивность детектору — оракул смысла не имеет")

    calibration_path = Path(args.calibration or f"runs/calibration/{args.record}.json")
    if not calibration_path.is_file():
        raise SystemExit(f"нет калибровки {calibration_path}: сначала tools.calibrate_track")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    axis = np.poly1d([calibration["axis_slope"], calibration["axis_y0_m"]])

    n_frames = count_frames(cfg.data.path(args.record), cfg.bag)
    found = detect(cfg, args.record, args.threshold, args.min_x)
    static, moving = split_static(found, args.static_tolerance, args.static_share, max(n_frames, 1))

    print(f"ярких объектов найдено: {len(found)}")
    print(f"неподвижная инфраструктура ({len(static)}):")
    for item in static:
        print(f"  x={item['x']:6.1f}  y={item['y']:+6.2f}  в {item['seen']} наблюдениях")

    half = cfg.gauge.half_width_m
    for row in moving:
        row["offset_from_axis_m"] = float(row["y"] - axis(row["x"]))
        row["in_gauge"] = bool(abs(row["offset_from_axis_m"]) <= half)
    in_gauge = [r for r in moving if r["in_gauge"]]

    print(f"\nфигуры: {len(moving)} наблюдений, из них в габарите своего пути {len(in_gauge)}")
    if moving:
        near = [r for r in moving if r["x"] < 30]
        far = [r for r in moving if r["x"] >= 30]
        for name, group in (("ближняя", near), ("дальняя", far)):
            if not group:
                continue
            frames = [r["frame"] for r in group]
            offsets = [r["offset_from_axis_m"] for r in group]
            print(f"  {name}: кадры {min(frames)}–{max(frames)}, "
                  f"дистанция {min(r['x'] for r in group):.1f}–{max(r['x'] for r in group):.1f} м, "
                  f"смещение от оси {min(offsets):+.2f}…{max(offsets):+.2f} м, "
                  f"в габарите {sum(r['in_gauge'] for r in group)} из {len(group)}")

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{args.record}.jsonl"
    with target.open("w", encoding="utf-8") as handle:
        for row in moving:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nразметка: {target}  ({len(moving)} строк)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

