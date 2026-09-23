# Позитив в полосе дальности: два конфига ядра на одних и тех же положениях тела.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.positive_control import _overlaps
from tools.scenarios import build_parser as scenario_parser


def _rows(path: Path) -> dict[int, dict]:
    return {r["frame"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def _hit(row: dict, tolerance: float) -> bool:
    if row["body_box"] is None:
        return False
    lo, hi = (np.asarray(v, float) for v in row["body_box"])
    raw = row.get("candidates_raw") or {"detections": []}
    boxes = [SimpleNamespace(bbox_min_xyz=d["bbox_min_xyz"], bbox_max_xyz=d["bbox_max_xyz"])
             for d in raw["detections"]]
    return _overlaps(boxes, lo, hi, tolerance)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", type=Path, required=True, help="прогон A (например, с Б′)")
    parser.add_argument("--b", type=Path, required=True, help="прогон B (например, без Б′)")
    parser.add_argument("--band", type=float, nargs=2, required=True, metavar=("ОТ", "ДО"))
    parser.add_argument("--tolerance", type=float,
                        default=scenario_parser().get_default("tolerance"))
    args = parser.parse_args(argv)

    a, b = _rows(args.a), _rows(args.b)
    lo, hi = args.band
    for name, inside in (("в полосе", True), ("вне полосы (контроль)", False)):
        frames = [f for f, r in a.items()
                  if r["body_box"] is not None and f in b
                  and (lo <= r["body_centre_x_m"] <= hi) == inside]
        both = sum(_hit(a[f], args.tolerance) and _hit(b[f], args.tolerance) for f in frames)
        only_a = sum(_hit(a[f], args.tolerance) and not _hit(b[f], args.tolerance) for f in frames)
        only_b = sum(_hit(b[f], args.tolerance) and not _hit(a[f], args.tolerance) for f in frames)
        hits_a, hits_b = both + only_a, both + only_b
        print(f"{name} [{lo:g}; {hi:g}] м: кадров с телом {len(frames)}; "
              f"A {hits_a}, B {hits_b}; оба {both}, только A {only_a}, только B {only_b}")
        if inside and frames:
            ends_a = [a[f]["corridor_end_m"] for f in frames if a[f]["corridor_end_m"]]
            ends_b = [b[f]["corridor_end_m"] for f in frames if b[f]["corridor_end_m"]]
            covered_b = sum(1 for f in frames if b[f]["corridor_end_m"]
                            and b[f]["corridor_end_m"] > a[f]["body_centre_x_m"])
            print(f"  конец коридора, медиана: A {np.median(ends_a):.1f} м, "
                  f"B {np.median(ends_b):.1f} м; у B центр тела внутри коридора "
                  f"в {covered_b} кадрах из {len(frames)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

