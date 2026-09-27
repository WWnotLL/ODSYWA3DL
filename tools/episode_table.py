# Эпизоды ложной тревоги построчно: кадры, путь в метрах, дистанции.

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np


EPISODE_GAP_S = 1.0
FILL_HALF_FRAMES = 12


def _advance(row: dict) -> float | None:
    return (row.get("debug") or {}).get("advance_m")


def group_by_gap(items: list[dict], gap_s: float = EPISODE_GAP_S) -> list[list[dict]]:
    groups: list[list[dict]] = []
    for item in sorted(items, key=lambda r: r["stamp_ns"]):
        if groups and (item["stamp_ns"] - groups[-1][-1]["stamp_ns"]) / 1e9 <= gap_s:
            groups[-1].append(item)
        else:
            groups.append([item])
    return groups


def filled_advances(rows: list[dict]) -> dict[int, float]:
    ordered = sorted(rows, key=lambda f: f["frame"])
    measured = np.array([np.nan if _advance(f) is None else _advance(f) for f in ordered])
    filled = measured.copy()
    for i in np.flatnonzero(~np.isfinite(measured)):
        window = measured[max(0, i - FILL_HALF_FRAMES):i + FILL_HALF_FRAMES + 1]
        window = window[np.isfinite(window)]
        filled[i] = np.median(window) if window.size else 0.0
    return {f["frame"]: float(v) for f, v in zip(ordered, filled)}


def episode_path(rows: list[dict], first: int, last: int,
                 filled: dict[int, float] | None = None) -> dict:
    by_frame = {f["frame"]: f for f in rows}
    inner = [k for k in range(first + 1, last + 1) if k in by_frame]
    advances = [_advance(by_frame[k]) for k in inner]
    known = [a for a in advances if a is not None]
    out = {"path_m": sum(known), "unknown_advance": len(advances) - len(known)}
    if filled is not None:
        out["path_filled_m"] = sum(filled[k] for k in inner)
    return out


def episodes(frames: list[dict], flag: Callable[[dict], bool] | None = None,
             *, fill: bool = False) -> list[dict]:
    rows = sorted((f for f in frames if "debug" in f), key=lambda f: f["frame"])
    test = flag or (lambda f: bool(f["obstacle_found"]))
    filled = filled_advances(rows) if fill else None
    out = []
    for group in group_by_gap([f for f in rows if test(f)]):
        first, last = group[0]["frame"], group[-1]["frame"]
        out.append({
            "frames": (first, last),
            "alarm_frames": len(group),
            "duration_s": (group[-1]["stamp_ns"] - group[0]["stamp_ns"]) / 1e9,
            **episode_path(rows, first, last, filled),
            "distance_first_m": group[0]["nearest_distance_m"],
            "distance_last_m": group[-1]["nearest_distance_m"],
        })
    return out


def detail(frames: list[dict], first: int, last: int) -> list[str]:
    rows = {f["frame"]: f for f in frames if "debug" in f}
    lines = ["| кадр | путь от начала, м | тревога | кандидат: ближняя x | центр x | мир: ближняя | мир: центр "
             "| y от оси | z | попаданий | подтверждён |",
             "|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---|"]
    path, known = 0.0, True
    for k in range(first, last + 1):
        row = rows.get(k)
        if row is None:
            continue
        debug = row["debug"]
        if k > first:
            advance = debug.get("advance_m")
            if advance is None:
                known = False
            else:
                path += advance
        raw = debug.get("candidates_raw") or {"detections": [], "lateral_offset_m": []}
        marks = (debug.get("confirm") or {}).get("candidates") or [{}] * len(raw["detections"])
        where = f"{path:.2f}" if known else f"{path:.2f}?"
        alarm = "да" if row["obstacle_found"] else ""
        if not raw["detections"]:
            lines.append(f"| {k} | {where} | {alarm} | — | | | | | | | |")
        for d, lateral, mark in zip(raw["detections"], raw["lateral_offset_m"], marks):
            lines.append(
                f"| {k} | {where} | {alarm} | {d['distance_m']:.2f} | {d['centroid_xyz'][0]:.2f} "
                f"| {path + d['distance_m']:.2f} | {path + d['centroid_xyz'][0]:.2f} | {lateral:+.2f} "
                f"| {d['bbox_min_xyz'][2]:.2f}–{d['bbox_max_xyz'][2]:.2f} | {mark.get('hits', '')} "
                f"| {'да' if mark.get('confirmed') else ''} |")
    return lines


def main(argv: list[str] | None = None) -> int:
    from tools.false_alarm_table import load

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--detail", default=None,
                        help="запись: покадровый разбор кандидатов каждого эпизода в мировых координатах")
    parser.add_argument("--before", type=int, default=5, help="сколько кадров до эпизода показывать")
    args = parser.parse_args(argv)

    if args.detail:
        for run in args.runs:
            frames, _ = load(run / f"{args.detail}.jsonl")
            for e in episodes(frames):
                start = max(0, e["frames"][0] - args.before)
                print(f"\n`{args.detail}`, `{run.name}`, эпизод {e['frames'][0]}–{e['frames'][1]}; "
                      f"мир — путь от кадра {start} плюс x\n")
                print("\n".join(detail(frames, start, e["frames"][1])))
        return 0

    records = sorted({p.stem for run in args.runs for p in run.glob("*.jsonl")})
    print("| запись | прогон | кадров с тревогой | эпизод: кадры | кадров | длит., с "
          "| путь, м | путь неизв., кадров | дистанция первой → последней, м |")
    print("|---|---|---:|---|---:|---:|---:|---:|---|")
    for record in records:
        for run in args.runs:
            path = run / f"{record}.jsonl"
            if not path.is_file():
                continue
            frames, _ = load(path)
            eps = episodes(frames)
            total = sum(e["alarm_frames"] for e in eps)
            if not eps:
                print(f"| `{record}` | `{run.name}` | 0 | — | | | | | |")
            for e in eps:
                print(f"| `{record}` | `{run.name}` | {total} | {e['frames'][0]}–{e['frames'][1]} "
                      f"| {e['alarm_frames']} | {e['duration_s']:.1f} | {e['path_m']:.1f} "
                      f"| {e['unknown_advance']} "
                      f"| {e['distance_first_m']:.1f} → {e['distance_last_m']:.1f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())

