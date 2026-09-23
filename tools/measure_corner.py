# Лицо грани у тревоги от середины колеи по рельсам, послойно по высоте против кромки Ом.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from core.config import Config
from tools.false_alarm_table import load
from tools.measure_wall import run


def alarm_spans(run_file: Path, first: int, last: int) -> dict[int, list[tuple[float, float]]]:
    frames, _ = load(run_file)
    out: dict[int, list[tuple[float, float]]] = {}
    for row in frames:
        if first <= row["frame"] <= last and row.get("detections"):
            out[row["frame"]] = [(d["bbox_min_xyz"][0], d["bbox_max_xyz"][0])
                                 for d in row["detections"]]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--record", required=True)
    parser.add_argument("--frames", type=int, nargs=2, required=True)
    parser.add_argument("--run", type=Path, required=True, help="прогон с тревогами")
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--lateral", type=float, nargs=2, default=(1.20, 2.10))
    parser.add_argument("--heights", type=float, nargs="+",
                        default=[2.17, 2.37, 2.57, 2.77, 2.97],
                        help="границы слоёв по высоте над УГР, м")
    parser.add_argument("--x-pad", type=float, default=1.0)
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    first, last = args.frames
    spans = alarm_spans(args.run / f"{args.record}.jsonl", first, last)
    sign = -1.0 if args.side == "right" else 1.0
    print(f"кадров с тревогой в {first}–{last}: {len(spans)}")
    print("\n| над УГР, м | срезов | лицо от рельсов: медиана | p10 | min | лицо от оси ядра, медиана "
          "| полуширина Ом | запас «лицо − кромка», медиана | лиц у границы окна |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for lo, hi in zip(args.heights, args.heights[1:]):
        rows, _ = run(cfg, args.record, first, last, sign, tuple(args.lateral), (lo, hi), 44.0)
        picked = [r for r in rows if r["frame"] in spans and "face_rails_m" in r
                  and any(a - args.x_pad <= r["x_m"] <= b + args.x_pad
                          for a, b in spans[r["frame"]])]
        mid = 0.5 * (lo + hi)
        half = float(cfg.gauge.half_width_at(np.array([mid]))[0])
        if not picked:
            print(f"| {lo:.2f}–{hi:.2f} | 0 | | | | | {half:.3f} | | |")
            continue
        rails = np.array([r["face_rails_m"] for r in picked])
        axis = np.array([r["face_axis_m"] for r in picked])
        at_bound = int(np.sum(rails <= args.lateral[0] + 0.01))
        print(f"| {lo:.2f}–{hi:.2f} | {len(picked)} | {np.median(rails):.3f} "
              f"| {np.percentile(rails, 10):.3f} | {rails.min():.3f} | {np.median(axis):.3f} "
              f"| {half:.3f} | {1000 * (np.median(rails) - half):+.0f} мм | {at_bound} |")
        target = Path("out/corner") / f"{args.record}_{first}_{last}_{lo:.2f}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(picked, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

