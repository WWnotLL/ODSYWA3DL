# Восстановление ring по углу места и пар двойного эха по «луч × азимут» для облаков без полей ring и timestamp; проверка точности на записях, где эти поля есть.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from core.config import Config
from tools.bag_reader import iter_frames

N_RINGS = 128
LEVEL_GAP_DEG = 0.05
LEVEL_TOLERANCE_DEG = 0.03
SHARP_SHARE = 0.9
MIN_RAY_POINTS = 100
DIRECTION_DIGITS = 5


def angles(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz, dtype=np.float64)
    rng = np.linalg.norm(xyz, axis=1)
    valid = rng > 1e-6
    elevation = np.zeros(rng.size)
    elevation[valid] = np.degrees(np.arcsin(xyz[valid, 2] / rng[valid]))
    azimuth = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    return elevation, azimuth, valid


def levels(elevation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values, counts = np.unique(np.round(elevation, 4), return_counts=True)
    breaks = np.flatnonzero(np.diff(values) > LEVEL_GAP_DEG) + 1
    centres, sizes, sharp = [], [], []
    for v, c in zip(np.split(values, breaks), np.split(counts, breaks)):
        top = int(np.argmax(c))
        centres.append(float(v[top]))
        sizes.append(int(c.sum()))
        sharp.append(bool(c[top] >= SHARP_SHARE * c.sum()))
    return np.array(centres), np.array(sizes), np.array(sharp)


def ring_table(elevation: np.ndarray) -> np.ndarray:
    centres, sizes, sharp = levels(elevation)
    keep = sharp & (sizes >= MIN_RAY_POINTS)
    return np.sort(centres[keep])[::-1]


def recover_ring(elevation: np.ndarray, table: np.ndarray) -> np.ndarray:
    ascending = table[::-1]
    pos = np.clip(np.searchsorted(ascending, elevation), 1, ascending.size - 1)
    left, right = ascending[pos - 1], ascending[pos]
    nearest = np.where(np.abs(elevation - left) <= np.abs(elevation - right), pos - 1, pos)
    ring = (ascending.size - 1 - nearest).astype(np.int64)
    ring[np.abs(elevation - ascending[nearest]) > LEVEL_TOLERANCE_DEG] = -1
    return ring


def azimuth_step(azimuth: np.ndarray, ring: np.ndarray) -> np.ndarray:
    steps = np.full(N_RINGS, np.nan)
    for r in np.unique(ring[ring >= 0]):
        a = np.unique(np.round(azimuth[ring == r], 4))
        d = np.round(np.diff(a), 3)
        d = d[d > 0]
        if d.size:
            values, counts = np.unique(d, return_counts=True)
            steps[r] = values[np.argmax(counts)]
    fallback = np.nanmedian(steps) if np.isfinite(steps).any() else 1.0
    return np.where(np.isfinite(steps), steps, fallback)


def pair_groups(key: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    return inverse, counts[inverse]


def pairs_as_set(key: np.ndarray) -> set[tuple[int, int]]:
    order = np.argsort(key, kind="stable")
    k = key[order]
    same = np.flatnonzero(k[1:] == k[:-1])
    return {(int(order[i]), int(order[i + 1])) for i in same}


def direction_key(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    unit = np.round(xyz / np.linalg.norm(xyz, axis=1)[:, None], DIRECTION_DIGITS)
    _, key = np.unique(unit, axis=0, return_inverse=True)
    return key.ravel()


def recovered_key(elevation: np.ndarray, azimuth: np.ndarray, table: np.ndarray,
                  step: float) -> tuple[np.ndarray, np.ndarray]:
    ring = recover_ring(elevation, table)
    column = np.round(azimuth / step[np.clip(ring, 0, N_RINGS - 1)]).astype(np.int64)
    key = np.where(ring >= 0, ring * 100_000 + (column - column.min()), -1 - np.arange(ring.size))
    return ring, key


def evaluate_record(cfg: Config, record: str, frames: list[int]) -> dict:
    wanted = set(frames)
    out = {"frames": 0, "points": 0, "ring_ok": 0, "unassigned": 0,
           "pairs_true": 0, "pairs_found": 0, "pairs_hit": 0, "table_error_deg": 0.0,
           "dir_found": 0, "dir_hit": 0}
    for f in iter_frames(cfg.data.path(record), cfg.bag, limit=max(frames) + 1):
        if f.index not in wanted:
            continue
        elevation, azimuth, valid = angles(f.xyz)
        el, az = elevation[valid], azimuth[valid]
        true_ring = f.ring[valid].astype(np.int64)
        table = ring_table(el)
        truth_table = np.array([np.median(el[true_ring == r]) for r in range(N_RINGS)])
        out["table_error_deg"] = max(out["table_error_deg"],
                                     float(np.abs(np.sort(truth_table)[::-1] - table).max()))
        step = azimuth_step(az, recover_ring(el, table))
        ring, key = recovered_key(el, az, table, step)
        _, t_index = np.unique(f.timestamp[valid], return_inverse=True)
        true_key = true_ring * 10_000_000 + t_index
        truth, found = pairs_as_set(true_key), pairs_as_set(key)
        by_direction = pairs_as_set(direction_key(np.asarray(f.xyz)[valid]))
        out["dir_found"] += len(by_direction)
        out["dir_hit"] += len(truth & by_direction)
        out["frames"] += 1
        out["points"] += int(el.size)
        out["ring_ok"] += int((ring == true_ring).sum())
        out["unassigned"] += int((ring < 0).sum())
        out["pairs_true"] += len(truth)
        out["pairs_found"] += len(found)
        out["pairs_hit"] += len(truth & found)
        out["step_deg"] = "/".join(f"{v:.1f}" for v in np.unique(np.round(step, 1)))
    return out


def describe_cloud(cfg: Config, bag: Path, n_frames: int, reference: str) -> None:
    ref_frame = next(iter_frames(cfg.data.path(reference), cfg.bag, limit=1))
    e_ref, _, v_ref = angles(ref_frame.xyz)
    table = ring_table(e_ref[v_ref])
    print(f"таблица лучей из `{reference}`: {table.size} лучей, {table[0]:+.2f}…{table[-1]:+.2f}°")
    print("\n| кадр | точек | пустых | на луче таблицы: до 0.0005° | до 0.03° | дальше 0.03° "
          "| пар по направлению | пар «луч × азимут» | x, м (p1…p99) | y, м | z, м |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|")
    counts, steps_seen = [], set()
    for f in iter_frames(bag, cfg.bag, limit=n_frames, partial=True):
        elevation, azimuth, valid = angles(f.xyz)
        el, az = elevation[valid], azimuth[valid]
        ascending = table[::-1]
        pos = np.clip(np.searchsorted(ascending, el), 1, ascending.size - 1)
        gap = np.minimum(np.abs(el - ascending[pos - 1]), np.abs(el - ascending[pos]))
        ring = recover_ring(el, table)
        step = azimuth_step(az, ring)
        steps_seen |= set(np.round(step[np.unique(ring[ring >= 0])], 2).tolist())
        _, key = recovered_key(el, az, table, step)
        _, size = pair_groups(key)
        _, dsize = pair_groups(direction_key(np.asarray(f.xyz)[valid]))
        xyz = np.asarray(f.xyz, dtype=np.float64)[valid]
        lo, hi = np.percentile(xyz, 1, axis=0), np.percentile(xyz, 99, axis=0)
        counts.append(f.xyz.shape[0])
        print(f"| {f.index} | {f.xyz.shape[0]} | {100 * (1 - valid.mean()):.1f} % "
              f"| {100 * (gap <= 5e-4).mean():.2f} % | {100 * (gap <= LEVEL_TOLERANCE_DEG).mean():.2f} % "
              f"| {100 * (gap > LEVEL_TOLERANCE_DEG).mean():.2f} % "
              f"| {100 * (dsize == 2).mean():.1f} % | {100 * (size == 2).mean():.1f} % "
              f"| {lo[0]:+.1f}…{hi[0]:+.1f} | {lo[1]:+.1f}…{hi[1]:+.1f} | {lo[2]:+.1f}…{hi[2]:+.1f} |")
    print(f"\nточек в кадре: {min(counts)}…{max(counts)}; шаги азимута по лучам: "
          f"{'/'.join(f'{v:.2f}' for v in sorted(steps_seen))}°")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    sub = parser.add_subparsers(dest="mode", required=True)
    cloud = sub.add_parser("describe")
    cloud.add_argument("--bag", type=Path, required=True)
    cloud.add_argument("--frames", type=int, default=5)
    cloud.add_argument("--reference", default="roundT_doubleT",
                       help="запись, из которой берётся таблица лучей")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--records", nargs="+", required=True)
    ev.add_argument("--frames", type=int, nargs="+", default=[0, 100, 200])
    args = parser.parse_args(argv)
    cfg = Config.from_yaml(args.config)

    if args.mode == "describe":
        describe_cloud(cfg, args.bag, args.frames, args.reference)
        return 0
    print("| запись | кадров | точек | ring верно | не отнесено к лучу | таблица лучей: max ошибка "
          "| шаг азимута | пар настоящих | «луч × азимут»: верных среди найденных / найдено из настоящих "
          "| по направлению: верных / найдено |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for record in args.records:
        r = evaluate_record(cfg, record, args.frames)
        pct = lambda a, b: f"{100 * a / max(b, 1):.3f} %"
        print(f"| `{record}` | {r['frames']} | {r['points']} | {pct(r['ring_ok'], r['points'])} "
              f"| {pct(r['unassigned'], r['points'])} | {r['table_error_deg']:.4f}° | {r['step_deg']}° "
              f"| {r['pairs_true']} | {pct(r['pairs_hit'], r['pairs_found'])} / "
              f"{pct(r['pairs_hit'], r['pairs_true'])} | {pct(r['dir_hit'], r['dir_found'])} / "
              f"{pct(r['dir_hit'], r['pairs_true'])} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
