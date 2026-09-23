# Положение лица конструкции у кромки габарита и направление оси в тех же кадрах.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from core.axis import AxisTracker, _ridge
from core.config import Config
from core.detector import detect
from core.preprocess import GroundTracker, preprocess_frame
from tools.bag_reader import iter_frames


SLICE_M = 2.0


BIN_M = 0.02


MIN_POINTS = 40


X_MARGIN_M = 4.0


BAND_M = 0.20


def _histogram(offset_abs: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    bins = np.arange(window[0], window[1] + BIN_M, BIN_M)
    counts, _ = np.histogram(offset_abs, bins=bins)
    return counts


def _face_from(counts: np.ndarray, window: tuple[float, float],
               fraction: float) -> float | None:
    if counts.sum() < MIN_POINTS or counts.max() == 0:
        return None
    level = fraction * counts.max()
    dense = counts >= level
    pair = dense[:-1] & dense[1:]
    hit = np.flatnonzero(pair)
    if hit.size == 0:
        return None
    return float(window[0] + hit[0] * BIN_M)


def _bands(offset_abs: np.ndarray, height: np.ndarray, window: tuple[float, float],
           z_window: tuple[float, float]) -> list[dict]:
    edges = np.arange(z_window[0], z_window[1] + BAND_M, BAND_M)
    index = np.digitize(height, edges) - 1
    out = []
    for slot in range(edges.size - 1):
        block = offset_abs[index == slot]
        if block.size < 5:
            continue
        counts = _histogram(block, window)
        out.append({
            "z_m": round(float(edges[slot]), 2),
            "n": int(block.size),
            "min_m": round(float(block.min()), 3),
            "p5_m": round(float(np.percentile(block, 5)), 3),
            "p50_m": round(float(np.median(block)), 3),
            "peak_m": round(float(window[0] + BIN_M * int(np.argmax(counts))), 3),
            "counts": counts.tolist(),
        })
    return out


def _face(offset_abs: np.ndarray, window: tuple[float, float],
          fraction: float) -> dict | None:
    if offset_abs.size < MIN_POINTS:
        return None
    counts = _histogram(offset_abs, window)
    face = _face_from(counts, window, fraction)
    if face is None:
        return None
    inside = offset_abs < face
    return {
        "face_m": round(face, 3),
        "peak_m": round(float(window[0] + BIN_M * int(np.argmax(counts))), 3),
        "n_points": int(offset_abs.size),
        "n_inside_face": int(inside.sum()),
        "share_inside_face": round(float(inside.mean()), 4),
        "deepest_m": round(float(face - offset_abs.min()), 3),
        "min_offset_m": round(float(offset_abs.min()), 3),
        "counts": counts.tolist(),
    }


def _rails(xyz: np.ndarray, axis, cfg: Config, near: float, far: float) -> dict | None:
    a = cfg.axis
    band = xyz[(xyz[:, 2] > a.rails_z_min_m) & (xyz[:, 2] < a.rails_z_max_m)
              & (xyz[:, 0] >= near) & (xyz[:, 0] <= far)]
    if band.shape[0] < a.rails_min_points:
        return None
    offset = axis.offset(band)
    half, window = a.rails_half_gauge_m, a.rails_search_halfwidth_m
    edges = np.arange(near, far + a.slice_m, a.slice_m)
    index = np.digitize(band[:, 0], edges) - 1
    rows = []
    for slot in range(edges.size - 1):
        lateral = offset[index == slot]
        if lateral.size < a.rails_min_points // len(edges) + 1:
            continue
        left = _ridge(lateral, half - window, half + window, a)
        right = _ridge(lateral, -half - window, -half + window, a)
        if left is None or right is None:
            continue
        if not a.rails_gauge_min_m < left - right < a.rails_gauge_max_m:
            continue
        rows.append((left, right))
    if not rows:
        return None
    table = np.asarray(rows)
    centre = 0.5 * (table[:, 0] + table[:, 1])
    return {
        "slices": int(table.shape[0]),


        "centre_bias_m": round(float(np.median(centre)), 4),
        "centre_max_abs_m": round(float(np.abs(centre).max()), 4),
        "left_m": round(float(np.median(table[:, 0])), 4),
        "right_m": round(float(np.median(table[:, 1])), 4),
        "gauge_m": round(float(np.median(table[:, 0] - table[:, 1])), 4),
    }


def run(cfg: Config, record: str, first: int, last: int,
        lateral_window: tuple[float, float], z_window: tuple[float, float],
        fraction: float) -> list[dict]:
    ground = GroundTracker(cfg.preprocess.ground)
    tracker = AxisTracker(cfg.axis)
    railhead = cfg.gauge.rail_head_offset_m or 0.0
    rows: list[dict] = []
    for frame in iter_frames(cfg.data.path(record), cfg.bag, limit=last + 1):
        prepared = preprocess_frame(
            frame.xyz, frame.intensity, frame.ring, frame.timestamp,
            cfg.preprocess, tracker=ground,
        ).xyz
        axis = tracker.update(prepared)
        if frame.index < first or axis is None:
            continue
        outcome = detect(prepared, axis, cfg.gauge, cfg.detector)
        if not outcome.detections:
            continue
        offset = axis.offset(prepared)
        height = prepared[:, 2] - railhead
        for detection, lateral in zip(outcome.detections,
                                      outcome.debug["lateral_offset_m"]):
            lo, hi = detection.bbox_min_xyz, detection.bbox_max_xyz
            near = max(lo[0] - X_MARGIN_M, axis.x_start_m)
            far = min(hi[0] + X_MARGIN_M, axis.x_end_m)
            side = np.sign(lateral) if lateral else 1.0
            window = ((prepared[:, 0] >= near) & (prepared[:, 0] <= far)
                      & (np.abs(offset) >= lateral_window[0])
                      & (np.abs(offset) <= lateral_window[1])
                      & (height >= z_window[0]) & (height <= z_window[1]))
            band = window & (np.sign(offset) == side)


            mirror = window & (np.sign(offset) == -side)
            rows.append({
                "frame": int(frame.index),
                "lateral_offset_m": round(float(lateral), 3),
                "distance_m": round(float(detection.distance_m), 2),
                "n_points": int(detection.n_points),
                "x_window_m": [round(near, 1), round(far, 1)],
                "axis_state": axis.state,
                "axis_residual_m": round(float(axis.residual_max_m), 4),
                "face": _face(np.abs(offset[band]), lateral_window, fraction),
                "by_height": _bands(np.abs(offset[band]), height[band],
                                    lateral_window, z_window),
                "face_other_side": _face(np.abs(offset[mirror]), lateral_window, fraction),
                "rails": _rails(prepared, axis, cfg, near, far),
            })
    return rows


def summarise(rows: list[dict], window: tuple[float, float], fraction: float) -> None:
    faces = [r for r in rows if r["face"]]
    print(f"\nкадров с кандидатом: {len(rows)}, с измеримым лицом: {len(faces)}")
    print(f"окно поиска по поперечнику: {window[0]:.2f}…{window[1]:.2f} м, "
          f"карман {1000 * BIN_M:.0f} мм, порог плотности {fraction:.0%} от пика")
    if not faces:
        return

    print("\n| кадр | дальность | кластер | лицо, м | пик, м | точек | внутри лица | глубина, мм |")
    print("|---:|---:|---:|---:|---:|---:|---|---:|")
    for r in faces:
        f = r["face"]
        print(f"| {r['frame']} | {r['distance_m']:.1f} | {r['lateral_offset_m']:+.3f} | "
              f"**{f['face_m']:.3f}** | {f['peak_m']:.3f} | {f['n_points']} | "
              f"{f['n_inside_face']} ({100 * f['share_inside_face']:.1f} %) | "
              f"{1000 * f['deepest_m']:.0f} |")

    face = np.array([r["face"]["face_m"] for r in faces])
    share = np.array([r["face"]["share_inside_face"] for r in faces])
    deep = np.array([r["face"]["deepest_m"] for r in faces])
    print(f"\nлицо по кадрам: медиана {np.median(face):.3f} м, "
          f"разброс {face.min():.3f}…{face.max():.3f}")
    print(f"доля точек внутри лица: {100 * share.min():.1f}…{100 * share.max():.1f} %, "
          f"медиана {100 * np.median(share):.1f} %")
    print(f"глубина захода внутрь: {1000 * deep.min():.0f}…{1000 * deep.max():.0f} мм, "
          f"медиана {1000 * np.median(deep):.0f} мм")


    total = np.sum([r["face"]["counts"] for r in faces], axis=0)
    aggregate = _face_from(total, window, fraction)
    print(f"\nСуммарный профиль по всем кадрам (лицо {aggregate:.3f} м):"
          if aggregate is not None else "\nСуммарный профиль:")
    peak = total.max()
    for index, count in enumerate(total):
        lo = window[0] + index * BIN_M
        if count == 0 and (index == 0 or total[index - 1] == 0):
            continue
        bar = "#" * int(round(40 * count / peak))
        mark = " ← лицо" if aggregate is not None and abs(lo - aggregate) < 1e-9 else ""
        print(f"  {lo:5.3f}…{lo + BIN_M:5.3f}  {count:5d}  {bar}{mark}")


    bands: dict[float, np.ndarray] = {}
    for r in faces:
        for band in r["by_height"]:
            key = band["z_m"]
            bands[key] = bands.get(key, 0) + np.asarray(band["counts"])
    if bands:
        print("\nПрофиль по полосам высоты (лицо каждой полосы отдельно):")
        print("\n| высота над УГР, м | точек | min | p5 | пик | лицо |")
        print("|---|---:|---:|---:|---:|---:|")
        for z in sorted(bands):
            counts = bands[z]
            if counts.sum() < MIN_POINTS:
                continue
            values = np.repeat(window[0] + BIN_M * (np.arange(counts.size) + 0.5), counts)
            face_z = _face_from(counts, window, fraction)
            peak = window[0] + BIN_M * int(np.argmax(counts))
            print(f"| {z:.2f}…{z + BAND_M:.2f} | {counts.sum()} | {values.min():.3f} | "
                  f"{np.percentile(values, 5):.3f} | {peak:.3f} | "
                  f"{'—' if face_z is None else f'{face_z:.3f}'} |")

    other = [r["face_other_side"] for r in rows if r["face_other_side"]]
    print(f"\nТа же полоса с другой стороны: измерима в {len(other)} кадрах из {len(rows)}")
    if other:
        near_side = np.array([o["min_offset_m"] for o in other])
        print(f"  ближайшая точка там: медиана {np.median(near_side):.3f} м, "
              f"минимум {near_side.min():.3f} м, точек медиана "
              f"{np.median([o['n_points'] for o in other]):.0f}")

    rails = [r for r in rows if r["rails"]]
    print(f"\nголовки рельсов в том же продольном окне: {len(rails)} кадров из {len(rows)}")
    if rails:
        print("\n| кадр | дальность кластера | левая, м | правая, м | "
              "середина колеи, м | колея, м | срезов | невязка оси, м |")
        print("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in rails:
            g = r["rails"]
            print(f"| {r['frame']} | {r['distance_m']:.1f} | {g['left_m']:+.4f} | "
                  f"{g['right_m']:+.4f} | **{g['centre_bias_m']:+.4f}** | {g['gauge_m']:.4f} | "
                  f"{g['slices']} | {r['axis_residual_m']:.4f} |")
        bias = np.array([r["rails"]["centre_bias_m"] for r in rails])
        gauge = np.array([r["rails"]["gauge_m"] for r in rails])
        print(f"\nсмещение оси от середины колеи: медиана {1000 * np.median(bias):+.0f} мм, "
              f"max |{1000 * np.abs(bias).max():.0f}| мм")
        print(f"измеренная колея: медиана {np.median(gauge):.4f} м "
              f"({gauge.min():.4f}…{gauge.max():.4f})")
        print("Знак: плюс — ось левее середины колеи, то есть объект СПРАВА "
              "на самом деле дальше от пути, чем показывает ось.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--record", required=True)
    parser.add_argument("--frames", type=int, nargs=2, required=True, metavar=("FIRST", "LAST"))
    parser.add_argument("--lateral", type=float, nargs=2, default=(1.20, 3.20),
                        metavar=("MIN", "MAX"),
                        help="окно поиска по поперечнику, м от оси; "
                             "берётся заведомо шире моды с обеих сторон")
    parser.add_argument("--height", type=float, nargs=2, default=(0.25, 1.70),
                        metavar=("MIN", "MAX"), help="окно по высоте над УГР, м")
    parser.add_argument("--fraction", type=float, default=0.5,
                        help="доля от пика профиля, с которой карман считается "
                             "частью конструкции, а не хвостом")
    parser.add_argument("--out", default="runs/intrusion")
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    rows = run(cfg, args.record, args.frames[0], args.frames[1],
               tuple(args.lateral), tuple(args.height), args.fraction)
    summarise(rows, tuple(args.lateral), args.fraction)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{args.record}.jsonl"
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nзаписано: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

