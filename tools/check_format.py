# Сверка разбора формата PointCloud2 с фактами о данных.

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from core.config import Config
from core.preprocess import (
    collapse_dual_returns,
    echoes_per_shot,
    preprocess_frame,
    ranges_m,
    valid_mask,
)
from tools.bag_reader import read_frame

OK, FAIL, INFO = "  ок  ", " ОШИБКА ", " данные "


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--record", default=None, help="по умолчанию — из format_check конфига")
    parser.add_argument("--frame", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    check = cfg.format_check
    record = args.record or check.record
    frame_index = check.frame_index if args.frame is None else args.frame
    bag_dir = cfg.data.path(record)

    print(f"запись:  {bag_dir}")
    print(f"кадр:    {frame_index}\n")

    started = time.perf_counter()
    frame = read_frame(bag_dir, cfg.bag, frame_index)
    read_seconds = time.perf_counter() - started

    failures: list[str] = []

    def verdict(passed: bool, label: str, actual: str, expected: str) -> None:
        mark = OK if passed else FAIL
        print(f"[{mark}] {label:<44} {actual:>24}   ожидалось {expected}")
        if not passed:
            failures.append(label)

    def note(label: str, actual: str) -> None:
        print(f"[{INFO}] {label:<44} {actual:>24}")


    verdict(
        frame.width == check.expected_points,
        "число точек в кадре (DATA.md §1)",
        f"{frame.width}",
        f"{check.expected_points}",
    )

    keep = valid_mask(frame.xyz, cfg.preprocess.min_range_m)
    n_valid = int(np.count_nonzero(keep))
    verdict(
        abs(n_valid - check.expected_valid_points) <= check.valid_points_tol,
        "точек с возвратом (DATA.md §3.1)",
        f"{n_valid}",
        f"{check.expected_valid_points} ± {check.valid_points_tol}",
    )
    note("доля пустых лучей", f"{100 * (1 - n_valid / frame.width):.1f} %")
    verdict(
        int(np.count_nonzero(np.isnan(frame.xyz))) == 0,
        "NaN в координатах (их не должно быть)",
        f"{int(np.count_nonzero(np.isnan(frame.xyz)))}",
        "0",
    )

    counts_all = echoes_per_shot(frame.ring, frame.timestamp)
    verdict(
        bool(np.all(counts_all == check.expected_echoes_per_shot)),
        "эхо на выстрел, все точки (DATA.md §3.2)",
        f"min {counts_all.min()} / max {counts_all.max()}",
        f"ровно {check.expected_echoes_per_shot}",
    )
    note("выстрелов в кадре", f"{counts_all.size}")
    note("колец", f"{np.unique(frame.ring).size}")


    xyz_valid = frame.xyz[keep]
    counts_valid = echoes_per_shot(frame.ring[keep], frame.timestamp[keep])
    primary, secondary = collapse_dual_returns(
        ranges_m(xyz_valid), frame.ring[keep], frame.timestamp[keep], cfg.preprocess.dual_return.keep
    )
    note("выстрелов среди точек с возвратом", f"{counts_valid.size}")
    note("  из них с двумя эхо", f"{int(np.count_nonzero(counts_valid == 2))}")
    note("  из них с одним эхо", f"{int(np.count_nonzero(counts_valid == 1))}")
    note("после схлопывания (основные эхо)", f"{primary.size}")
    note("во втором эхо", f"{secondary.size}")

    started = time.perf_counter()
    result = preprocess_frame(
        frame.xyz, frame.intensity, frame.ring, frame.timestamp, cfg.preprocess
    )
    preprocess_seconds = time.perf_counter() - started

    plane = result.plane
    up_is_plus_z = plane.normal[2] > 0
    print()
    note("нормаль основания (в осях REP-103)", np.array2string(plane.normal, precision=4))
    note("опорных точек RANSAC", f"{plane.inlier_fraction:.3f}")
    note("перевес массы по сторонам", f"{plane.up_mass_ratio:.1f}x")
    note("критерий направления «вверх»", plane.up_source)
    print(
        f"[{INFO}] {'ВЫВОД: в исходных данных +Z смотрит':<44} "
        f"{('вверх' if up_is_plus_z else 'ВНИЗ'):>24}"
    )
    if not up_is_plus_z:
        print(
            "         внимание: нужен доворот на 180° вокруг X (переворачиваются Y и Z),\n"
            "         одиночная смена знака Z дала бы отражение"
        )

    z = result.xyz[:, 2]
    note("высота над основанием, p1 … p99", f"{np.percentile(z, 1):.2f} … {np.percentile(z, 99):.2f} м")
    note("вперёд по X, максимум", f"{result.xyz[:, 0].max():.1f} м")
    note("чтение кадра", f"{read_seconds * 1000:.0f} мс")
    note("препроцессинг", f"{preprocess_seconds * 1000:.0f} мс")

    print()
    if failures:
        print(f"СВЕРКА НЕ ПРОЙДЕНА: {len(failures)} расхождений — {failures}")
        return 1
    print("Сверка пройдена: формат разобран так, как описано в DATA.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())

