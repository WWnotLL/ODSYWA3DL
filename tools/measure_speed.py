# Замер скорости состава по записям через одометрию ядра.

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.config import Config
from core.odometry import estimate_shift, longitudinal_profile
from core.preprocess import GroundTracker, preprocess_frame
from tools.bag_reader import iter_frames


def run(cfg: Config, record: str, stride: int) -> list[dict]:
    tracker = GroundTracker(cfg.preprocess.ground)
    rows: list[dict] = []
    previous_profile, previous_stamp = None, None
    for frame in iter_frames(cfg.data.path(record), cfg.bag, stride=stride):
        result = preprocess_frame(
            frame.xyz, frame.intensity, frame.ring, frame.timestamp,
            cfg.preprocess, tracker=tracker,
        )
        profile = longitudinal_profile(result.xyz, result.intensity, cfg.odometry)
        if previous_profile is not None:
            dt = (frame.stamp_ns - previous_stamp) / 1e9
            estimate = estimate_shift(previous_profile, profile, dt, cfg.odometry)
            rows.append({
                "frame": frame.index,
                "dt_s": round(dt, 5),
                "speed_mps": None if estimate.speed_mps is None else round(estimate.speed_mps, 3),
                "speed_kmh": None if estimate.speed_kmh is None else round(estimate.speed_kmh, 2),
                "prominence": round(estimate.prominence, 2),
                "degenerate": int(estimate.degenerate),
                "reason": estimate.reason,
            })
        previous_profile, previous_stamp = profile, frame.stamp_ns
    return rows


def summarise(rows: list[dict], n_segments: int) -> None:
    good = np.array([r["speed_kmh"] for r in rows if not r["degenerate"]], dtype=float)
    share = 100.0 * (1.0 - good.size / max(len(rows), 1))
    print(f"  кадров сопоставлено {len(rows)}, вырожденных {share:.0f} %")
    if good.size:
        print(f"  скорость: медиана {np.median(good):.1f} км/ч, "
              f"p25 {np.percentile(good, 25):.1f}, p75 {np.percentile(good, 75):.1f}")
    print(f"  {'участок':<16}{'кадров':>8}{'вырожд.':>9}{'медиана':>10}{'разброс':>10}")
    edges = np.linspace(0, len(rows), n_segments + 1).astype(int)
    for start, stop in zip(edges[:-1], edges[1:]):
        chunk = rows[start:stop]
        if not chunk:
            continue
        values = np.array([r["speed_kmh"] for r in chunk if not r["degenerate"]], dtype=float)
        bad = 100.0 * (1.0 - values.size / len(chunk))
        span = f"{np.percentile(values, 75) - np.percentile(values, 25):.1f}" if values.size else "—"
        median = f"{np.median(values):.1f}" if values.size else "—"
        label = f"{chunk[0]['frame']}–{chunk[-1]['frame']}"
        print(f"  {label:<16}{len(chunk):>8}{bad:>8.0f}%{median:>10}{span:>10}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--records", nargs="+", default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--segments", type=int, default=6)
    parser.add_argument("--out", default="runs/speed")
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    records = args.records or list(cfg.data.records)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(records), 1, figsize=(11, 3.2 * len(records)), squeeze=False)
    for row_index, record in enumerate(records):
        print(f"\n{record}")
        rows = run(cfg, record, args.stride)
        if not rows:
            print("  нет сопоставимых кадров"); continue
        summarise(rows, args.segments)

        with (out_dir / f"{record}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

        ax = axes[row_index][0]
        good = [(r["frame"], r["speed_kmh"]) for r in rows if not r["degenerate"]]
        bad = [r["frame"] for r in rows if r["degenerate"]]
        if good:
            ax.plot(*zip(*good), ".-", lw=1, ms=4, label="оценка")
        for frame in bad:
            ax.axvline(frame, color="crimson", alpha=0.25, lw=1)
        ax.set_title(f"{record} — красным вырожденные кадры", fontsize=10)
        ax.set_xlabel("кадр"); ax.set_ylabel("км/ч"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "speed.png", dpi=110)
    print(f"\nграфики: {out_dir / 'speed.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

