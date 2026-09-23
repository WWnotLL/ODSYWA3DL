# Эпизоды ложной тревоги построчно: кадры, путь в метрах, дистанции.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tools.false_alarm_table import EPISODE_GAP_S, load


def episodes(frames: list[dict]) -> list[dict]:
    rows = sorted((f for f in frames if "debug" in f), key=lambda f: f["frame"])
    by_frame = {f["frame"]: f for f in rows}
    alarms = [f for f in rows if f["obstacle_found"]]
    groups: list[list[dict]] = []
    for f in alarms:
        if groups and (f["stamp_ns"] - groups[-1][-1]["stamp_ns"]) / 1e9 <= EPISODE_GAP_S:
            groups[-1].append(f)
        else:
            groups.append([f])
    out = []
    for group in groups:
        first, last = group[0]["frame"], group[-1]["frame"]
        advances = [by_frame[k]["debug"].get("advance_m") for k in range(first + 1, last + 1)
                    if k in by_frame]
        known = [a for a in advances if a is not None]
        out.append({
            "frames": (first, last),
            "alarm_frames": len(group),
            "duration_s": (group[-1]["stamp_ns"] - group[0]["stamp_ns"]) / 1e9,
            "path_m": sum(known),
            "unknown_advance": len(advances) - len(known),
            "distance_first_m": group[0]["nearest_distance_m"],
            "distance_last_m": group[-1]["nearest_distance_m"],
        })
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    args = parser.parse_args(argv)

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

