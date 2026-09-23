# Замер поведения нормали плоскости пола вдоль записи.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.config import Config
from core.preprocess import (collapse_dual_returns, estimate_ground_plane, ranges_m,
                             to_rep103, valid_mask)
from tools.bag_reader import iter_frames


def collect(cfg: Config, record: str, stride: int) -> dict:
    index, roll, pitch, height, tilt = [], [], [], [], []
    for frame in iter_frames(cfg.data.path(record), cfg.bag, stride=stride):
        keep = valid_mask(frame.xyz, cfg.preprocess.min_range_m)
        primary, _ = collapse_dual_returns(
            ranges_m(frame.xyz), frame.ring, frame.timestamp,
            cfg.preprocess.dual_return.keep, valid=keep,
        )
        primary = primary[keep[primary]]
        xyz = to_rep103(frame.xyz, cfg.preprocess.axes.rotation, cfg.preprocess.numerics)[primary]
        plane = estimate_ground_plane(xyz, cfg.preprocess.ground)
        n = plane.normal
        index.append(frame.index)

        roll.append(np.degrees(np.arctan2(n[1], n[2])))
        pitch.append(np.degrees(np.arctan2(-n[0], n[2])))
        tilt.append(np.degrees(np.arccos(np.clip(n[2], -1.0, 1.0))))
        height.append(plane.offset)
    return {
        "index": np.array(index), "roll": np.array(roll), "pitch": np.array(pitch),
        "tilt": np.array(tilt), "height": np.array(height),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--records", nargs="+", default=None)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--out", default="runs/ground_normal")
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    records = args.records or [cfg.data.obstacle_record, "roundT_squareT_pressureGate_squareT"]
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, len(records), figsize=(7 * len(records), 10), squeeze=False)
    print(f"{'запись':<40} {'крен, °':>22} {'тангаж, °':>22} {'высота, м':>18}")
    print(f"{'':<40} {'среднее размах':>22} {'среднее размах':>22} {'среднее размах':>18}")
    for column, record in enumerate(records):
        data = collect(cfg, record, stride=max(1, 201 // args.samples))
        r, p, h = data["roll"], data["pitch"], data["height"]
        print(f"{record:<40} {r.mean():10.2f} {r.max()-r.min():10.2f} "
              f"{p.mean():11.2f} {p.max()-p.min():10.2f} {h.mean():10.3f} {h.max()-h.min():7.3f}")

        for row, (values, label) in enumerate(
            ((r, "крен, °"), (p, "тангаж, °"), (h, "высота лидара над основанием, м"))
        ):
            ax = axes[row][column]
            ax.plot(data["index"], values, lw=1.2)
            ax.axhline(values.mean(), color="crimson", ls="--", lw=0.9,
                       label=f"среднее {values.mean():.2f}")
            ax.set_xlabel("кадр"); ax.set_ylabel(label); ax.grid(alpha=0.3); ax.legend(fontsize=8)
            if row == 0:
                ax.set_title(record, fontsize=10)
    plt.tight_layout()
    target = out_dir / "ground_normal.png"
    plt.savefig(target, dpi=110)
    print(f"\nграфик: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

