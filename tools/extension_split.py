# Какие кандидаты снимает отказ от продления оси и видны ли они под рельсовой осью позже.

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.false_alarm_table import load
from tools.positive_control import _overlaps


def _candidates(row: dict) -> list[dict]:
    raw = (row.get("debug") or {}).get("candidates_raw") or {}
    out = []
    for det, lateral in zip(raw.get("detections", []), raw.get("lateral_offset_m", [])):
        out.append({**det, "lateral_offset_m": lateral})
    return out


def _as_box(det: dict) -> SimpleNamespace:
    return SimpleNamespace(bbox_min_xyz=det["bbox_min_xyz"], bbox_max_xyz=det["bbox_max_xyz"])


def _corridor_end(row: dict) -> float | None:
    corridor = (row.get("debug") or {}).get("corridor")
    return None if not corridor else float(corridor["x_m"][1])


def _axis_source_at(row: dict, x: float) -> str | None:
    axis = (row.get("debug") or {}).get("axis") or {}
    line = axis.get("polyline")
    if not line:
        return None
    xs = np.asarray(line["points_xy_m"], float)[:, 0]
    source = line["source"]
    index = int(np.argmin(np.abs(xs - x)))
    return source[index] if isinstance(source, list) else source


def follow(rows: list[dict], frame: int, lo: np.ndarray, hi: np.ndarray,
           min_range: float, tolerance: float, max_frames: int, inside_m: float,
           check_frames: int) -> str:
    by_frame = {r["frame"]: r for r in rows if "debug" in r}
    travelled = 0.0
    checked: list[tuple[int, bool, float, float, str | None]] = []
    stop = None
    for g in range(frame + 1, frame + 1 + max_frames):
        row = by_frame.get(g)
        if row is None:
            stop = "запись кончилась"
            break
        advance = row["debug"].get("advance_m")
        if advance is None:
            stop = f"путь неизвестен в кадре {g}"
            break
        travelled += float(advance)
        end = _corridor_end(row)
        box_lo, box_hi = lo.copy(), hi.copy()
        box_lo[0] -= travelled
        box_hi[0] -= travelled
        if box_hi[0] < min_range:
            stop = "проехал ближнюю границу"
            break
        if end is None or box_lo[0] > end - inside_m:
            if checked:
                break
            continue
        box_lo[0] = max(box_lo[0], min_range)
        box_hi[0] = min(box_hi[0], end)
        seen = _overlaps([_as_box(c) for c in _candidates(row)], box_lo, box_hi, tolerance)
        checked.append((g, seen, box_lo[0], box_hi[0], _axis_source_at(row, box_lo[0])))
        if len(checked) == check_frames:
            break
    if not checked:
        return stop or f"не въехал за {max_frames} кадров (прошли {travelled:.1f} м)"
    hits = sum(c[1] for c in checked)
    sources = sorted({c[4] or "—" for c in checked})
    return (f"{'ЕСТЬ' if hits else 'нет'}: {hits}/{len(checked)} кадров "
            f"{checked[0][0]}–{checked[-1][0]}, x {checked[0][2]:.1f}…{checked[-1][2]:.1f}, "
            f"ось {'/'.join(sources)}")


def split(with_rows: list[dict], without_rows: list[dict], tolerance: float) -> list[dict]:
    other = {r["frame"]: r for r in without_rows if "debug" in r}
    out = []
    for row in with_rows:
        if "debug" not in row or row["frame"] not in other:
            continue
        twin = other[row["frame"]]
        mine, theirs = _candidates(row), _candidates(twin)
        end_with, end_without = _corridor_end(row), _corridor_end(twin)
        for det in mine:
            lo, hi = np.asarray(det["bbox_min_xyz"]), np.asarray(det["bbox_max_xyz"])
            if _overlaps([_as_box(t) for t in theirs], lo, hi, tolerance):
                continue
            if end_without is None:
                group = "без коридора"
            elif lo[0] >= end_without:
                group = "а"
            elif hi[0] > end_without:
                group = "граница"
            else:
                group = "б"
            out.append({"frame": row["frame"], "kind": "снят", "group": group,
                        "box_y": (float(lo[1]), float(hi[1])),
                        "x_m": (round(float(lo[0]), 2), round(float(hi[0]), 2)),
                        "y_offset_m": round(float(det["lateral_offset_m"]), 3),
                        "z_m": (round(float(lo[2]), 2), round(float(hi[2]), 2)),
                        "n_points": det["n_points"],
                        "base_with_m": end_with, "base_without_m": end_without,
                        "axis_source": _axis_source_at(row, float(lo[0])),
                        "alarm_with": bool(row["obstacle_found"])})
        for det in theirs:
            lo, hi = np.asarray(det["bbox_min_xyz"]), np.asarray(det["bbox_max_xyz"])
            if _overlaps([_as_box(m) for m in mine], lo, hi, tolerance):
                continue
            out.append({"frame": row["frame"], "kind": "появился", "group": "—",
                        "x_m": (round(float(lo[0]), 2), round(float(hi[0]), 2)),
                        "y_offset_m": round(float(det["lateral_offset_m"]), 3),
                        "z_m": (round(float(lo[2]), 2), round(float(hi[2]), 2)),
                        "n_points": det["n_points"],
                        "base_with_m": end_with, "base_without_m": end_without,
                        "axis_source": _axis_source_at(twin, float(lo[0])),
                        "alarm_with": bool(row["obstacle_found"])})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--with", dest="with_run", type=Path, required=True)
    parser.add_argument("--without", dest="without_run", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.3,
                        help="допуск пересечения рамок одного кадра двух прогонов, м")
    parser.add_argument("--follow", action="store_true",
                        help="проследить снятого кандидата до базы по рельсам")
    parser.add_argument("--follow-tolerance", type=float, default=1.0,
                        help="допуск пересечения перенесённой рамки, м (ошибка пути)")
    parser.add_argument("--follow-frames", type=int, default=60)
    parser.add_argument("--check-frames", type=int, default=5,
                        help="сколько кадров в базе смотреть после въезда")
    parser.add_argument("--inside-m", type=float, default=1.0,
                        help="сколько длины кандидата должно въехать в базу, м")
    parser.add_argument("--min-range", type=float, default=5.0,
                        help="ближняя граница коридора, м (detector.min_range_m)")
    args = parser.parse_args(argv)

    totals: dict[str, int] = {}
    print("| запись | кадр | что | группа | x, м | y от оси, м | z, м | точек "
          "| конец базы с Б′ → без, м | ось у кандидата (с Б′) | тревога с Б′ |"
          + (" в базе по рельсам |" if args.follow else ""))
    print("|---|---:|---|---|---|---:|---|---:|---|---|---|" + ("---|" if args.follow else ""))
    for path in sorted(args.with_run.glob("*.jsonl")):
        twin = args.without_run / path.name
        if not twin.is_file():
            continue
        with_rows, without_rows = load(path)[0], load(twin)[0]
        rows = split(with_rows, without_rows, args.tolerance)
        for r in rows:
            tail = ""
            if args.follow and r["kind"] == "снят":
                lo = np.array([r["x_m"][0], r["box_y"][0], r["z_m"][0]])
                hi = np.array([r["x_m"][1], r["box_y"][1], r["z_m"][1]])
                verdict = follow(without_rows, r["frame"], lo, hi, args.min_range,
                                 args.follow_tolerance, args.follow_frames, args.inside_m,
                                 args.check_frames)
                totals["в базе по рельсам: " + verdict.split(":")[0]] = totals.get(
                    "в базе по рельсам: " + verdict.split(":")[0], 0) + 1
                tail = f" {verdict} |"
            elif args.follow:
                tail = " |"
            key = f"{r['kind']} ({r['group']})"
            totals[key] = totals.get(key, 0) + 1
            fmt = lambda v: "—" if v is None else f"{v:.1f}"
            print(f"| `{path.stem}` | {r['frame']} | {r['kind']} | {r['group']} "
                  f"| {r['x_m'][0]}–{r['x_m'][1]} | {r['y_offset_m']:+.2f} "
                  f"| {r['z_m'][0]}–{r['z_m'][1]} | {r['n_points']} "
                  f"| {fmt(r['base_with_m'])} → {fmt(r['base_without_m'])} "
                  f"| {r['axis_source'] or '—'} | {'да' if r['alarm_with'] else ''} |" + tail)
    print("\nИтого кандидатов:", ", ".join(f"{k}: {v}" for k, v in sorted(totals.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())

