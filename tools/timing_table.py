# Время обработки кадра по прогону: медиана и хвосты вне кадров полного RANSAC и отдельно на них.

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from core.config import Config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    args = parser.parse_args(argv)
    period = Config.from_yaml(args.config).preprocess.ground.replan_every_n_frames + 1
    rows = list(csv.DictReader((args.runs / "timings.csv").open(encoding="utf-8")))
    print(f"полный RANSAC — на кадрах, кратных {period}")
    print("| запись | точек в кадре | кадров | медиана, мс | p90 | p99 | кадры полного RANSAC, мс |")
    print("|---|---:|---:|---:|---:|---:|---|")
    for record in sorted({r["record"] for r in rows}):
        mine = [r for r in rows if r["record"] == record]
        frames = np.array([int(r["frame"]) for r in mine])
        total = np.array([float(r["total_ms"]) for r in mine])
        replan = frames % period == 0
        rest = total[~replan]
        peaks = ", ".join(f"{f}: {t:.0f}" for f, t in zip(frames[replan], total[replan]))
        print(f"| `{record}` | {mine[0]['n_points_raw']} | {len(mine)} | {np.median(rest):.1f} "
              f"| {np.percentile(rest, 90):.1f} | {np.percentile(rest, 99):.1f} | {peaks} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
