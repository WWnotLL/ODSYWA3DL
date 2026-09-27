# Вставка виртуального тела в кадр трассировкой лучей — основа позитивных проверок.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from core.axis import AxisEstimate, AxisTracker
from core.config import Config
from core.detector import detect
from core.preprocess import GroundTracker, preprocess_frame
from tools.bag_reader import iter_frames


BODY_RADIUS_M = 0.25
BODY_HEIGHT_M = 1.70


def inject_body(
    xyz: np.ndarray,
    sensor_origin: np.ndarray,
    centre_xy: tuple[float, float],
    z_bottom: float,
    radius_m: float = BODY_RADIUS_M,
    height_m: float = BODY_HEIGHT_M,
) -> tuple[np.ndarray, int]:
    direction = xyz - sensor_origin
    length = np.linalg.norm(direction, axis=1)
    alive = length > 1e-6
    unit = np.zeros_like(direction)
    unit[alive] = direction[alive] / length[alive, None]


    offset = sensor_origin[:2] - np.asarray(centre_xy, dtype=float)
    a = (unit[:, :2] ** 2).sum(axis=1)
    b = 2.0 * (unit[:, :2] @ offset)
    c = float(offset @ offset) - radius_m * radius_m
    disc = b * b - 4.0 * a * c
    hit = alive & (disc > 0.0) & (a > 1e-12)

    t_hit = np.full(xyz.shape[0], np.inf)
    root = np.sqrt(np.where(hit, disc, 0.0))


    denom = np.where(hit, 2.0 * a, 1.0)
    near = np.where(hit, (-b - root) / denom, np.inf)
    far = np.where(hit, (-b + root) / denom, np.inf)
    t_hit = np.where(near > 0.0, near, far)

    z_at_hit = sensor_origin[2] + unit[:, 2] * t_hit
    hit &= np.isfinite(t_hit) & (t_hit > 0.0)
    hit &= (z_at_hit >= z_bottom) & (z_at_hit <= z_bottom + height_m)

    hit &= t_hit < length

    out = xyz.copy()
    out[hit] = sensor_origin + unit[hit] * t_hit[hit, None]
    return out, int(hit.sum())


def inject_tilted_body(
    xyz: np.ndarray,
    sensor_origin: np.ndarray,
    bottom_xyz: tuple[float, float, float],
    direction: np.ndarray,
    radius_m: float,
    length_m: float,
) -> tuple[np.ndarray, int]:
    ray = xyz - sensor_origin
    length = np.linalg.norm(ray, axis=1)
    alive = length > 1e-6
    unit = np.zeros_like(ray)
    unit[alive] = ray[alive] / length[alive, None]

    along = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
    start = sensor_origin - np.asarray(bottom_xyz, dtype=float)
    unit_along = unit @ along
    unit_across = unit - unit_along[:, None] * along
    start_across = start - float(start @ along) * along
    a = (unit_across ** 2).sum(axis=1)
    b = 2.0 * (unit_across @ start_across)
    c = float(start_across @ start_across) - radius_m * radius_m
    disc = b * b - 4.0 * a * c
    hit = alive & (disc > 0.0) & (a > 1e-12)

    root = np.sqrt(np.where(hit, disc, 0.0))
    denom = np.where(hit, 2.0 * a, 1.0)
    near = np.where(hit, (-b - root) / denom, np.inf)
    far = np.where(hit, (-b + root) / denom, np.inf)
    t_hit = np.where(near > 0.0, near, far)

    position = float(start @ along) + unit_along * np.where(np.isfinite(t_hit), t_hit, 0.0)
    hit &= np.isfinite(t_hit) & (t_hit > 0.0)
    hit &= (position >= 0.0) & (position <= length_m)
    hit &= t_hit < length

    out = xyz.copy()
    out[hit] = sensor_origin + unit[hit] * t_hit[hit, None]
    return out, int(hit.sum())


def _overlaps(detections: list, lo: np.ndarray, hi: np.ndarray, tolerance: float) -> bool:
    for d in detections:
        d_lo = np.asarray(d.bbox_min_xyz) - tolerance
        d_hi = np.asarray(d.bbox_max_xyz) + tolerance
        if np.all(d_hi >= lo) and np.all(d_lo <= hi):
            return True
    return False


def _near(detections: list, point: np.ndarray, tolerance: float) -> bool:
    for d in detections:
        lo = np.asarray(d.bbox_min_xyz) - tolerance
        hi = np.asarray(d.bbox_max_xyz) + tolerance
        if np.all(point >= lo) and np.all(point <= hi):
            return True
    return False


def real_figure_track(path: Path) -> dict[int, np.ndarray]:
    track: dict[int, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["x"] > 30.0:
            continue
        track[int(row["frame"])] = np.array([row["x"], row["y"], row["z"]])
    return track


def run(cfg: Config, tolerance: float, radius_m: float, height_m: float,
        advance_m: float | None) -> list[dict]:
    record = cfg.data.obstacle_record
    figures = real_figure_track(Path("runs/oracle") / f"{record}.jsonl")
    if not figures:
        raise SystemExit("разметка оракула не найдена — сначала python -m tools.oracle")
    first, last = min(figures), max(figures)

    ground = GroundTracker(cfg.preprocess.ground)


    clean_tracker = AxisTracker(cfg.axis)
    body_tracker = AxisTracker(cfg.axis)
    railhead = cfg.gauge.rail_head_offset_m or 0.0
    rows: list[dict] = []

    for frame in iter_frames(cfg.data.path(record), cfg.bag, limit=last + 1):
        result = preprocess_frame(
            frame.xyz, frame.intensity, frame.ring, frame.timestamp,
            cfg.preprocess, tracker=ground,
        )
        clean_axis = clean_tracker.update(result.xyz, advance_m=advance_m)


        injected, n_points, centre = result.xyz, 0, None
        if frame.index >= first and frame.index in figures and clean_axis is not None:
            distance = float(figures[frame.index][0])
            centre = (distance, float(clean_axis.y_at(distance)))
            sensor = np.array([0.0, 0.0, float(result.plane.offset)])
            injected, n_points = inject_body(
                result.xyz, sensor, centre, railhead, radius_m, height_m,
            )


        body_axis = body_tracker.update(injected, advance_m=advance_m)
        if centre is None or clean_axis is None:
            continue

        figure = figures[frame.index]
        body_lo = np.array([centre[0] - radius_m, centre[1] - radius_m, railhead])
        body_hi = np.array([centre[0] + radius_m, centre[1] + radius_m, railhead + height_m])


        with_body = (detect(injected, body_axis, cfg.gauge, cfg.detector)
                     if body_axis is not None else None)
        clean = detect(result.xyz, clean_axis, cfg.gauge, cfg.detector)

        front_face = centre[0] - radius_m
        rows.append({
            "frame": int(frame.index),
            "distance_m": round(centre[0], 2),
            "front_face_m": round(front_face, 2),


            "axis_start_m": None if body_axis is None else round(body_axis.x_start_m, 1),
            "corridor_m": None if with_body is None else with_body.debug["corridor"]["x_m"],
            "axis_base_m": None if body_axis is None else round(body_axis.x_end_m, 1),
            "axis_base_clean_m": round(clean_axis.x_end_m, 1),
            "base_minus_front_m": (None if body_axis is None
                                   else round(body_axis.x_end_m - front_face, 1)),
            "axis_traced_m": None if body_axis is None else round(body_axis.x_traced_m, 1),
            "axis_source": "none" if body_axis is None else body_axis.source,
            "axis_state": "lost" if body_axis is None else body_axis.state,
            "body_points": n_points,
            "body_detected": (with_body is not None
                              and _overlaps(with_body.detections, body_lo, body_hi, tolerance)),
            "body_score": (0.0 if with_body is None
                           else max((d.score for d in with_body.detections), default=0.0)),
            "figure_offset_m": round(float(clean_axis.offset(figure[None, :])[0]), 2),
            "figure_detected": _near(clean.detections, figure, tolerance),


            "figure_detected_footprint": _overlaps(
                clean.detections,
                np.array([figure[0] - BODY_RADIUS_M, figure[1] - BODY_RADIUS_M, railhead]),
                np.array([figure[0] + BODY_RADIUS_M, figure[1] + BODY_RADIUS_M,
                          railhead + BODY_HEIGHT_M]),
                tolerance,
            ),
            "clean_alarms": len(clean.detections),
        })
    return rows


def summarise(rows: list[dict]) -> None:
    if not rows:
        print("нет кадров для сверки")
        return
    seen = [r for r in rows if r["body_points"] > 0]
    hit = [r for r in rows if r["body_detected"]]
    miss = [r for r in rows if not r["body_detected"]]
    figure_hit = [r for r in rows if r["figure_detected"]]
    figure_hit_fp = [r for r in rows if r["figure_detected_footprint"]]
    d = np.array([r["distance_m"] for r in rows])

    print(f"  кадров в паре: {len(rows)}, дальность {d.min():.1f}…{d.max():.1f} м")
    if seen:
        print(f"  тело видно лучами в {len(seen)} кадрах "
              f"(точек: медиана {np.median([r['body_points'] for r in seen]):.0f})")


    no_axis = [r for r in rows if r["axis_base_m"] is None]
    with_axis = [r for r in rows if r["axis_base_m"] is not None]
    before = [r for r in with_axis if r["distance_m"] < r["corridor_m"][0]]
    inside = [r for r in with_axis
              if r["corridor_m"][0] <= r["distance_m"] <= r["corridor_m"][1]]
    beyond = [r for r in with_axis if r["distance_m"] > r["corridor_m"][1]]
    pick = lambda g: sum(1 for r in g if r["body_detected"])
    print(f"  ТЕЛО НА ОСИ обнаружено:      {len(hit)}/{len(rows)} "
          f"({100 * len(hit) / len(rows):.0f} %)")
    print(f"    ПЕРЕД коридором:             {pick(before)}/{len(before)}")
    print(f"    внутри коридора:             {pick(inside)}/{len(inside)} "
          f"({100 * pick(inside) / max(len(inside), 1):.0f} %)")
    print(f"    за коридором:                {pick(beyond)}/{len(beyond)}")
    print(f"    ось снесена телом:           0/{len(no_axis)}")
    print(f"  ФИГУРА В МЕЖДУПУТЬЕ вызвала: {len(figure_hit)}/{len(rows)} "
          f"по точке, {len(figure_hit_fp)}/{len(rows)} по габариту")


    if with_axis:
        gap = np.array([r["base_minus_front_m"] for r in with_axis])
        loss = np.array([r["axis_base_clean_m"] - r["axis_base_m"] for r in with_axis])
        cut = int((gap < 0.0).sum())
        print(f"  база относительно передней грани тела: медиана {np.median(gap):+.1f} м, "
              f"p10 {np.percentile(gap, 10):+.1f}, min {gap.min():+.1f}")
        print(f"    кадров, где база обрывается ДО тела: {cut}/{len(with_axis)}")
        print(f"  потеря базы из-за тела (чистая − с телом): медиана {np.median(loss):+.1f} м, "
              f"max {loss.max():+.1f}")

    if miss:
        far = max(miss, key=lambda r: r["distance_m"])
        print(f"  пропуски на дальностях: "
              f"{sorted({round(r['distance_m']) for r in miss})}, дальний {far['distance_m']} м "
              f"(точек на теле {far['body_points']})")
    if hit:
        s = np.array([r["body_score"] for r in hit])
        print(f"  уверенность на теле: медиана {np.median(s):.2f}, min {s.min():.2f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tolerance", type=float, default=0.6,
                        help="допуск сопоставления срабатывания с целью, м")
    parser.add_argument("--body-radius", type=float, default=BODY_RADIUS_M,
                        help="полуширина тела, м: 0.25 — человек, 0.75 — во всю колею")
    parser.add_argument("--body-height", type=float, default=BODY_HEIGHT_M,
                        help="высота тела, м")
    parser.add_argument("--advance-m", type=float, default=None,
                        help="пройденный путь за кадр, м; не задан = одометрии нет")
    parser.add_argument("--out", default="runs/positive_control")
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    rows = run(cfg, args.tolerance, args.body_radius, args.body_height, args.advance_m)
    print(f"\n=== положительный контроль на {cfg.data.obstacle_record}: "
          f"тело r={args.body_radius:.2f} м, h={args.body_height:.2f} м")
    summarise(rows)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)


    target = out_dir / f"{cfg.data.obstacle_record}_r{args.body_radius:.2f}.jsonl"
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  сохранено: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

