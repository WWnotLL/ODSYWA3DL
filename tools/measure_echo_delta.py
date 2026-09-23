# Замер разницы дальностей двойного эха.

from __future__ import annotations

import argparse
import sys

import numpy as np

from core.config import Config
from core.preprocess import ranges_m, shot_key, valid_mask
from tools.bag_reader import iter_frames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--record", default=None)
    parser.add_argument("--frames", type=int, default=5)
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    record = args.record or cfg.format_check.record
    print(f"запись: {record}, кадров: {args.frames}\n")

    deltas: list[np.ndarray] = []
    for frame in iter_frames(cfg.data.path(record), cfg.bag, limit=args.frames):
        key = shot_key(frame.ring, frame.timestamp)
        _, counts = np.unique(key, return_counts=True)
        if counts.min() != 2 or counts.max() != 2:
            print(f"кадр {frame.index}: эхо на выстрел от {counts.min()} до {counts.max()},"
                  " парная раскладка не подтверждается")
            return 1


        order = np.argsort(key, kind="stable").reshape(-1, 2)
        stride = np.unique(order[:, 1] - order[:, 0])

        rng = ranges_m(frame.xyz)
        near_idx, far_idx = order[:, 0], order[:, 1]
        pair_rng = np.stack([rng[near_idx], rng[far_idx]], axis=1)
        keep = valid_mask(frame.xyz, cfg.preprocess.min_range_m)
        both_valid = keep[near_idx] & keep[far_idx]
        one_valid = keep[near_idx] ^ keep[far_idx]

        alive = pair_rng[both_valid]
        delta = np.abs(alive[:, 1] - alive[:, 0])
        deltas.append(delta)

        if frame.index == 0:
            same_xyz = np.all(frame.xyz[near_idx][both_valid] == frame.xyz[far_idx][both_valid], axis=1)
            same_int = frame.intensity[near_idx][both_valid] == frame.intensity[far_idx][both_valid]
            print(f"сдвиг между эхо одного выстрела:       {stride.tolist()} индексов")
            print(f"выстрелов в кадре:                     {order.shape[0]}")
            print(f"  оба эха с возвратом:                 {int(both_valid.sum())}")
            print(f"  ровно одно эхо с возвратом:          {int(one_valid.sum())}")
            print(f"  координаты эхо совпадают побитово:   {100 * same_xyz.mean():.2f} %")
            print(f"  интенсивности совпадают:             {100 * same_int.mean():.2f} %\n")

    delta = np.concatenate(deltas)
    print(f"разность дальностей внутри пары, {delta.size} пар:")
    edges = [0.0, 1e-9, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, np.inf]
    counts, _ = np.histogram(delta, bins=edges)
    for low, high, count in zip(edges[:-1], edges[1:], counts):
        share = 100 * count / delta.size
        bar = "#" * int(round(share / 2))
        print(f"  {low:>8.3g} … {high:<8.3g} {count:>9d}  {share:6.2f} %  {bar}")

    nonzero = delta[delta > 0]
    print(f"\n  ровно ноль:      {100 * (delta == 0).mean():.3f} %")
    print(f"  больше 1 см:     {100 * (delta > 0.01).mean():.3f} %")
    if nonzero.size:
        print(f"  ненулевые: медиана {np.median(nonzero):.4f} м, p99 {np.percentile(nonzero, 99):.3f} м")
    return 0


if __name__ == "__main__":
    sys.exit(main())

