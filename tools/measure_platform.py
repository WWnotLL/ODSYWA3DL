# Замер геометрии края платформы для параметров её выреза.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from core.axis import AxisTracker
from core.config import Config
from core.preprocess import GroundTracker, preprocess_frame
from tools.bag_reader import iter_frames


SEARCH_Z = (0.7, 1.6)


SEARCH_LATERAL = (1.1, 6.0)


WALK_BIN_M = 0.10


WALK_MIN_POINTS = 3


BIN_M = 0.02


def _profile(lateral: np.ndarray, height: np.ndarray) -> dict | None:
    inside = (
        (lateral >= SEARCH_LATERAL[0]) & (lateral <= SEARCH_LATERAL[1])
        & (height >= SEARCH_Z[0]) & (height <= SEARCH_Z[1])
    )
    if inside.sum() < 30:
        return None
    lateral, height = lateral[inside], height[inside]

    bins = np.arange(SEARCH_Z[0], SEARCH_Z[1] + BIN_M, BIN_M)
    counts, _ = np.histogram(height, bins=bins)
    peak = int(np.argmax(counts))
    if counts[peak] < 20:
        return None
    z_surface = float(bins[peak] + 0.5 * BIN_M)

    on_surface = np.abs(height - z_surface) <= 3 * BIN_M
    if on_surface.sum() < 20:
        return None
    surface_lateral = lateral[on_surface]
    edge = float(np.percentile(surface_lateral, 2))


    walk = np.arange(edge, SEARCH_LATERAL[1] + WALK_BIN_M, WALK_BIN_M)
    counts_out, _ = np.histogram(surface_lateral, bins=walk)
    gap = np.flatnonzero(counts_out < WALK_MIN_POINTS)
    reach = int(gap[0]) if gap.size else counts_out.size
    return {
        "z_surface_m": z_surface,
        "edge_m": edge,
        "span_m": float(np.percentile(surface_lateral, 98) - edge),
        "continuous_m": round(reach * WALK_BIN_M, 2),
        "n_points": int(on_surface.sum()),
        "share_of_band": float(counts[peak] / counts.sum()),
    }


FACE_LATERAL = (1.15, 1.95)


FACE_Z = (0.35, 1.10)


SLICE_M = 2.0


FACE_MIN_POINTS = 20
FACE_MIN_HEIGHT_SPAN_M = 0.55


FACE_MAX_SPREAD_M = 0.30


def _face(lateral: np.ndarray, height: np.ndarray, x: np.ndarray,
          near: float, far: float) -> dict:
    edges = np.arange(near, far + SLICE_M, SLICE_M)
    if edges.size < 3:
        return {"slices": 0, "with_face": 0, "longest_run": 0, "lateral": []}
    inside = (
        (lateral >= FACE_LATERAL[0]) & (lateral <= FACE_LATERAL[1])
        & (height >= FACE_Z[0]) & (height <= FACE_Z[1])
    )
    index = np.digitize(x[inside], edges) - 1
    lateral, height = lateral[inside], height[inside]
    present, positions = [], []
    for slot in range(edges.size - 1):
        block = index == slot
        if block.sum() < FACE_MIN_POINTS:
            present.append(False)
            continue
        h, lat = height[block], lateral[block]
        span = float(np.percentile(h, 90) - np.percentile(h, 10))
        spread = float(np.percentile(lat, 90) - np.percentile(lat, 10))
        ok = span >= FACE_MIN_HEIGHT_SPAN_M and spread <= FACE_MAX_SPREAD_M
        present.append(ok)
        if ok:
            positions.append(float(np.median(lat)))
    run_len = best = 0
    for ok in present:
        run_len = run_len + 1 if ok else 0
        best = max(best, run_len)
    return {
        "slices": len(present),
        "with_face": int(sum(present)),
        "longest_run": best,
        "longest_run_m": round(best * SLICE_M, 1),
        "lateral": positions,
    }


def run(cfg: Config, record: str, stride: int, limit: int | None) -> list[dict]:
    ground = GroundTracker(cfg.preprocess.ground)
    tracker = AxisTracker(cfg.axis)
    railhead = cfg.gauge.rail_head_offset_m or 0.0
    rows: list[dict] = []
    for frame in iter_frames(cfg.data.path(record), cfg.bag, stride=stride, limit=limit):
        result = preprocess_frame(
            frame.xyz, frame.intensity, frame.ring, frame.timestamp,
            cfg.preprocess, tracker=ground,
        )
        axis = tracker.update(result.xyz)
        if axis is None:
            continue
        xyz = result.xyz
        near = max(cfg.detector.min_range_m, axis.x_start_m)
        window = (xyz[:, 0] >= near) & (xyz[:, 0] <= axis.x_end_m)
        offset = axis.offset(xyz)
        height = xyz[:, 2] - railhead
        row = {"frame": int(frame.index), "x_m": [round(near, 1), round(axis.x_end_m, 1)]}
        for name, side in (("left", offset > 0), ("right", offset < 0)):
            mask = window & side
            row[name] = _profile(np.abs(offset[mask]), height[mask])
            row[name + "_face"] = _face(
                np.abs(offset[mask]), height[mask], xyz[mask, 0], near, axis.x_end_m)
        rows.append(row)
    return rows


def summarise(record: str, rows: list[dict]) -> None:
    print(f"\n=== {record}: кадров {len(rows)}")
    for side in ("left", "right"):
        found = [r[side] for r in rows if r[side] is not None]
        if not found:
            print(f"  {side}: поверхность не найдена ни в одном кадре")
            continue
        z = np.array([f["z_surface_m"] for f in found])
        edge = np.array([f["edge_m"] for f in found])
        span = np.array([f["span_m"] for f in found])
        n = np.array([f["n_points"] for f in found])
        print(f"  {side}: поверхность в {len(found)} кадрах из {len(rows)} "
              f"({100 * len(found) / len(rows):.0f} %)")
        cont = np.array([f["continuous_m"] for f in found])
        for label, values in (("высота над УГР", z), ("кромка от оси", edge),
                              ("ширина поверхности", span),
                              ("НЕПРЕРЫВНО наружу", cont)):
            print(f"    {label:22s} p10 {np.percentile(values, 10):6.3f}  "
                  f"медиана {np.median(values):6.3f}  p90 {np.percentile(values, 90):6.3f}  "
                  f"разброс p90−p10 {np.percentile(values, 90) - np.percentile(values, 10):.3f}")
        print(f"    точек на поверхности   медиана {np.median(n):.0f}, "
              f"p10 {np.percentile(n, 10):.0f}")
    print("  вертикальная грань кромки (продольные срезы по 2 м):")
    for side in ("left", "right"):
        faces = [r[side + "_face"] for r in rows if r.get(side + "_face")]
        if not faces:
            continue
        run_m = np.array([f["longest_run_m"] for f in faces])
        share = np.array([f["with_face"] / max(f["slices"], 1) for f in faces])
        lat = np.concatenate([f["lateral"] for f in faces if f["lateral"]]) \
            if any(f["lateral"] for f in faces) else np.array([])
        print(f"    {side}: непрерывная грань, м — p50 {np.median(run_m):.0f}, "
              f"p90 {np.percentile(run_m, 90):.0f}, max {run_m.max():.0f}; "
              f"кадров с гранью ≥ 8 м: {100 * (run_m >= 8).mean():.0f} %")
        if lat.size:
            print(f"        положение грани от оси: p5 {np.percentile(lat, 5):.3f}, "
                  f"медиана {np.median(lat):.3f}, p95 {np.percentile(lat, 95):.3f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--records", nargs="+", required=True)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="runs/platform")
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for record in args.records:
        rows = run(cfg, record, args.stride, args.limit)
        summarise(record, rows)
        target = out_dir / f"{record}.jsonl"
        with target.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

