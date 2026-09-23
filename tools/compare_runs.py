# Покадровая сверка двух прогонов: что изменила правка.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _rows(path: Path) -> dict[int, dict]:
    return {r["frame"]: r for r in map(json.loads, path.open(encoding="utf-8"))
            if "debug" in r}


def _axis(row: dict) -> tuple:
    axis = row["debug"].get("axis")
    if not axis or axis.get("state") == "lost":
        return ("lost",)
    return (axis["state"], axis["x_trusted_m"], axis["y_center_m"], axis["slope_deg"])


def _strip(items: list[dict] | None, ignore: set[str]) -> list[dict] | None:
    if items is None:
        return None
    return [{k: v for k, v in item.items() if k not in ignore} for item in items]


def _fields(row: dict, ignore: set[str]) -> dict:
    confirm = row["debug"].get("confirm") or {}
    fields = {
        "obstacle_found": row["obstacle_found"],
        "n_detections": len(row["detections"]),
        "detections": _strip(row["detections"], ignore),
        "candidates": _strip(confirm.get("candidates"), ignore),
        "axis": _axis(row),
    }
    if "distance_m" not in ignore:
        fields["nearest_distance_m"] = row["nearest_distance_m"]
    return fields


def _shifts(before: dict, after: dict, key: str) -> list[float]:
    out = []
    for frame in set(before) & set(after):
        a = {tuple(d["centroid_xyz"]): d for d in before[frame]["detections"]}
        for d in after[frame]["detections"]:
            match = a.get(tuple(d["centroid_xyz"]))
            if match is not None:
                out.append(d[key] - match[key])
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--ignore", nargs="*", default=[],
                        choices=["distance_m", "score"])
    args = parser.parse_args(argv)
    ignore = set(args.ignore)
    total_diff = 0
    for after_path in sorted(args.after.glob("*.jsonl")):
        before_path = args.before / after_path.name
        if not before_path.exists():
            continue
        before, after = _rows(before_path), _rows(after_path)
        differ = {}
        for frame in sorted(set(before) | set(after)):
            if frame not in before or frame not in after:
                differ.setdefault("кадр есть только в одном прогоне", []).append(frame)
                continue
            a, b = _fields(before[frame], ignore), _fields(after[frame], ignore)
            for key in a:
                if a[key] != b[key]:
                    differ.setdefault(key, []).append(frame)
        n = sum(len(v) for v in differ.values())
        total_diff += n
        print(f"{after_path.stem}: кадров {len(after)}, расхождений {n}"
              + "".join(f"\n  {k}: {v[:10]}{' …' if len(v) > 10 else ''}"
                        for k, v in differ.items()))
        for key in sorted(ignore):
            shift = _shifts(before, after, key)
            if shift:
                v = np.asarray(shift)
                print(f"  сдвиг {key} по {v.size} детекциям: медиана {np.median(v):+.4f}, "
                      f"min {v.min():+.4f}, max {v.max():+.4f}")
    print(f"\nвсего расхождений: {total_diff}")
    return 0 if total_diff == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

