# Калибровка геометрии пути по рельсам: колея, рыск, высота головок.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from core.config import Config
from core.preprocess import GroundTracker, preprocess_frame
from tools.bag_reader import iter_frames


def accumulate(cfg: Config, record: str, stride: int, x_max: float) -> np.ndarray:
    tracker = GroundTracker(cfg.preprocess.ground)
    chunks = []
    for frame in iter_frames(cfg.data.path(record), cfg.bag, stride=stride):
        result = preprocess_frame(
            frame.xyz, frame.intensity, frame.ring, frame.timestamp,
            cfg.preprocess, tracker=tracker,
        )
        keep = (result.xyz[:, 0] > 5.0) & (result.xyz[:, 0] < x_max) & (result.xyz[:, 2] < 1.2)
        chunks.append(result.xyz[keep])
    return np.concatenate(chunks)


def _ridge(points: np.ndarray, lo: float, hi: float, width: float, min_points: int):
    band = points[(points[:, 1] > lo) & (points[:, 1] < hi)]
    if band.shape[0] < min_points:
        return None
    bins = np.arange(lo, hi + width, width)
    counts, _ = np.histogram(band[:, 1], bins=bins)
    centres = 0.5 * (bins[:-1] + bins[1:])
    peak = int(np.argmax(counts))
    window = slice(max(0, peak - 3), min(counts.size, peak + 4))
    if counts[window].sum() < min_points:
        return None
    return float(np.average(centres[window], weights=counts[window]))


def _greedy_slices(head: np.ndarray, step: float, x_end: float, min_points: int) -> list:
    rows, guess = [], 0.0
    for start in np.arange(5.0, x_end, step):
        window = head[(head[:, 0] >= start) & (head[:, 0] < start + step)]
        left = _ridge(window, guess - 1.30, guess - 0.35, 0.04, min_points)
        right = _ridge(window, guess + 0.35, guess + 1.30, 0.04, min_points)
        if left is None or right is None or not 1.40 < right - left < 1.72:
            continue
        guess = 0.5 * (left + right)
        rows.append((start + 0.5 * step, left, right, guess))
    return rows


def _refine_slices(head: np.ndarray, rows: list, step: float, x_end: float,
                   min_points: int, half_window: float = 0.45, passes: int = 2) -> list:
    for _ in range(passes):
        if len(rows) < 3:
            return rows
        table = np.array([(r[0], r[3]) for r in rows])
        axis = np.poly1d(np.polyfit(table[:, 0], table[:, 1], 1))
        found = []
        for start in np.arange(5.0, x_end, step):
            window = head[(head[:, 0] >= start) & (head[:, 0] < start + step)]
            centre = float(axis(start + 0.5 * step))
            left = _ridge(window, centre - 0.78 - half_window, centre - 0.78 + half_window, 0.04, min_points)
            right = _ridge(window, centre + 0.78 - half_window, centre + 0.78 + half_window, 0.04, min_points)
            if left is None or right is None or not 1.40 < right - left < 1.72:
                continue
            found.append((start + 0.5 * step, left, right, 0.5 * (left + right)))
        if len(found) <= len(rows):
            break
        rows = found
    return rows


def calibrate(points: np.ndarray, cfg: Config, step: float = 5.0) -> dict:
    head = points[(points[:, 2] > 0.04) & (points[:, 2] < 0.32)]
    half = 0.5 * 1.52 + 0.04

    x_end = float(points[:, 0].max()) - step
    rows = _greedy_slices(head, step, x_end, 60)
    rows = _refine_slices(head, rows, step, x_end, 60)
    if len(rows) < 3:
        raise SystemExit("рельсы не прослеживаются: слишком мало слоёв прошло проверку")

    table = np.array(rows)
    slope, y0 = np.polyfit(table[:, 0], table[:, 3], 1)
    axis = np.poly1d([slope, y0])


    spacings = []
    for x_mid, *_ in rows:
        window = head[(head[:, 0] >= x_mid - 0.5 * step) & (head[:, 0] < x_mid + 0.5 * step)]
        base = axis(x_mid)
        left = _ridge(window, base + 2.4, base + 3.9, 0.03, 60)
        right = _ridge(window, base + 3.9, base + 5.4, 0.03, 60)
        if left is None or right is None or not 1.40 < right - left < 1.75:
            continue
        spacings.append(0.5 * (left + right) - base)

    heights = []
    for x_mid, left, right, centre in rows:
        window = points[(points[:, 0] >= x_mid - 0.5 * step) & (points[:, 0] < x_mid + 0.5 * step)]
        bed = window[(np.abs(window[:, 1] - centre) < 0.35) & (np.abs(window[:, 2]) < 0.15)]
        if bed.shape[0] < 30:
            continue
        bed_level = float(np.median(bed[:, 2]))
        tops = []
        for rail in (left, right):
            strip = window[np.abs(window[:, 1] - rail) < 0.05]


            above = strip[(strip[:, 2] > bed_level + 0.06) & (strip[:, 2] < bed_level + 0.35)]
            if above.shape[0] >= 20:
                tops.append(float(np.percentile(above[:, 2], 75)))
        if tops:
            heights.append(float(np.mean(tops) - bed_level))

    return {
        "axis_y0_m": float(y0),
        "axis_slope": float(slope),
        "axis_yaw_deg": float(np.degrees(np.arctan(slope))),
        "gauge_m": float(np.median(table[:, 2] - table[:, 1])),
        "gauge_spread_m": float(np.std(table[:, 2] - table[:, 1])),
        "track_spacing_m": float(np.median(spacings)) if spacings else None,
        "rail_head_offset_m": float(np.median(heights)) if heights else None,
        "rail_head_spread_m": float(np.std(heights)) if heights else None,
        "x_traced_m": [float(table[0, 0]), float(table[-1, 0])],
        "n_slices": len(rows),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--record", required=True)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--x-max", type=float, default=80.0)
    parser.add_argument("--out", default="runs/calibration")
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    points = accumulate(cfg, args.record, args.stride, args.x_max)
    print(f"накоплено {points.shape[0]} точек")
    result = calibrate(points, cfg)
    for key, value in result.items():
        print(f"  {key:<22} {value}")

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{args.record}.json"
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nсохранено: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

