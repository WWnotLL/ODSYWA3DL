# Замер разброса смещения оси у сенсора — основа порога гейта позы.

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from core.axis import _ESTIMATORS, AxisTracker
from core.config import Config
from core.preprocess import GroundTracker, preprocess_frame
from tools.bag_reader import iter_frames


def _curvature(xyz: np.ndarray, cfg: Config, method: str) -> float | None:
    table = _ESTIMATORS[method](xyz, cfg.axis)
    if table is None or table.shape[0] < 8:
        return None
    c2, _, _ = np.polyfit(table[:, 0], table[:, 1], 2)
    return float(c2)


def scan(cfg: Config, record: str, stride: int, limit: int | None) -> list[dict]:
    ground = GroundTracker(cfg.preprocess.ground)
    tracker = AxisTracker(cfg.axis)
    rows: list[dict] = []
    for frame in iter_frames(cfg.data.path(record), cfg.bag, stride=stride, limit=limit):
        result = preprocess_frame(
            frame.xyz, frame.intensity, frame.ring, frame.timestamp,
            cfg.preprocess, tracker=ground,
        )
        axis = tracker.update(result.xyz, frame_gap=stride)


        if axis is None or axis.source != "measured":
            continue
        rows.append({
            "record": record,
            "frame": int(frame.index),
            "y0_m": float(axis.y0_m),
            "yaw_deg": float(axis.yaw_deg),
            "x_start_m": float(axis.x_start_m),
            "x_end_m": float(axis.x_end_m),
            "base_m": float(axis.x_end_m - axis.x_start_m),
            "residual_max_m": float(axis.residual_max_m),
            "method": axis.method,
            "meets_clearance": bool(axis.meets_clearance),
            "c2": _curvature(result.xyz, cfg, axis.method),
            "cross_check_m": axis.cross_check_m,
        })
    return rows


def predicted_offset(a: float, b: float, radius_m: float) -> float:
    return -(a * a + 4.0 * a * b + b * b) / (12.0 * radius_m)


def report(rows: list[dict], radius_m: float) -> None:
    if not rows:
        print("  измеренной оси нет ни в одном кадре")
        return
    y0 = np.array([r["y0_m"] for r in rows])
    base = np.array([r["x_end_m"] for r in rows])
    print(f"  кадров с измеренной осью: {len(rows)}")
    print(f"  y0: медиана {np.median(y0):+.3f} м, p5 {np.percentile(y0, 5):+.3f}, "
          f"p95 {np.percentile(y0, 95):+.3f}, размах {y0.min():+.3f}…{y0.max():+.3f}, "
          f"σ {y0.std():.3f}")
    jump = np.abs(np.diff(y0))
    if jump.size:
        print(f"  межкадровый скачок |Δy0|: медиана {np.median(jump):.3f} м, "
              f"p95 {np.percentile(jump, 95):.3f}, max {jump.max():.3f}")


    edges = [0, 12.0, 20.0, 28.0, 1e9]
    print(f"  разрез по дальнему краю базы (предсказание при R = {radius_m:.0f} м):")
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (base >= lo) & (base < hi)
        if mask.sum() < 3:
            continue
        a = np.array([r["x_start_m"] for r in rows])[mask].mean()
        b = base[mask].mean()
        print(f"    база {lo:.0f}…{hi if hi < 1e8 else 99:.0f} м: {int(mask.sum()):4d} кадров, "
              f"y0 медиана {np.median(y0[mask]):+.3f}, σ {y0[mask].std():.3f} | "
              f"предсказано смещение {predicted_offset(a, b, radius_m):+.3f}")


def by_curvature(rows: list[dict], floor: float) -> None:
    have = [r for r in rows if r["c2"] is not None]
    if not have:
        print("  кривизна не оценена ни в одном кадре")
        return
    c2 = np.array([r["c2"] for r in have])
    y0 = np.array([r["y0_m"] for r in have])
    groups = {
        "левая  (c2 > 0)": c2 > floor,
        "прямая (|c2| мал)": np.abs(c2) <= floor,
        "правая (c2 < 0)": c2 < -floor,
    }
    print(f"  разрез по знаку кривизны (порог |c2| = {floor:.1e}, R = {1/(2*floor):.0f} м):")
    for name, mask in groups.items():
        if mask.sum() < 3:
            print(f"    {name}: {int(mask.sum())} кадров — мало")
            continue
        radius = 1.0 / (2.0 * np.abs(np.median(c2[mask]))) if np.abs(np.median(c2[mask])) > 0 else np.inf
        print(f"    {name}: {int(mask.sum()):4d} кадров, y0 медиана {np.median(y0[mask]):+.3f} м, "
              f"σ {y0[mask].std():.3f}, |R| медиана {radius:6.0f} м")


def honest_envelope(rows: list[dict], gate_m: float, limit_m: float) -> None:
    y0 = np.array([r["y0_m"] for r in rows])
    cc = np.array([np.nan if r["cross_check_m"] is None else float(r["cross_check_m"])
                   for r in rows])
    accepted, dev = [], np.zeros(y0.size)
    for i, value in enumerate(y0):
        if accepted:
            dev[i] = abs(value - float(np.median(accepted[-20:])))
        accepted.append(float(value))
    confirmed = np.isfinite(cc) & (cc <= limit_m)
    diverged = np.isfinite(cc) & (cc > limit_m)
    print(f"  разрез по сверке (порог расхождения {limit_m:.3f} м):")
    for name, mask in (("подтверждён сверкой", confirmed),
                       ("сверка разошлась", diverged),
                       ("сверки нет", ~np.isfinite(cc))):
        if mask.sum() == 0:
            print(f"    {name}: 0 кадров")
            continue
        print(f"    {name}: {int(mask.sum()):4d} кадров, |y0 − якорь| p95 "
              f"{np.percentile(dev[mask], 95):.3f}, max {dev[mask].max():.3f}, "
              f"за гейт {gate_m:.2f} м выходит {int((dev[mask] > gate_m).sum())}")


def replay_anchor(rows: list[dict], windows: tuple[int, ...], gate_m: float) -> None:
    y0 = np.array([r["y0_m"] for r in rows])
    print(f"  онлайн-якорь (гейт {gate_m:.2f} м), отклонение от бегущей медианы:")
    for w in windows:
        accepted: list[float] = []
        worst, rejected, run, longest = 0.0, 0, 0, 0
        for value in y0:
            if accepted:
                anchor = float(np.median(accepted[-w:]))
                deviation = abs(value - anchor)
                if deviation > gate_m:
                    rejected += 1
                    run += 1
                    longest = max(longest, run)
                    continue
                run = 0
                worst = max(worst, deviation)
            accepted.append(float(value))
        print(f"    окно {w:3d} кадров: max |y0 − якорь| = {worst:.3f} м, "
              f"отвергнуто {rejected} кадров (серия max {longest}), "
              f"запас до гейта {gate_m - worst:+.3f} м")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--records", nargs="+", required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--radius-m", type=float, default=300.0,
                        help="радиус кривой для предсказания, м (p10 измеренных)")
    parser.add_argument("--curvature-floor", type=float, default=1.0 / (2 * 2000.0),
                        help="ниже этого |c2| участок считается прямым (R = 2000 м)")
    parser.add_argument("--gate-m", type=float, default=0.5,
                        help="порог гейта позы, м")
    parser.add_argument("--out", default="runs/axis_intercept")
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    everything: list[dict] = []
    for record in args.records:
        print(f"\n=== {record}", flush=True)
        rows = scan(cfg, record, args.stride, args.limit)
        report(rows, args.radius_m)
        by_curvature(rows, args.curvature_floor)
        honest_envelope(rows, args.gate_m, cfg.axis.clearance_m)
        replay_anchor(rows, (10, 20, 50), args.gate_m)
        everything.extend(rows)

    if everything:
        target = out_dir / "intercept.csv"
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(everything[0].keys()))
            writer.writeheader()
            writer.writerows(everything)
        print(f"\nсохранено: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

