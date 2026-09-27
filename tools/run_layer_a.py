# Прогон ядра по записям с сохранением покадрового выхода и времени обработки в runs/.

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info

from core.config import Config
from core.pipeline import ObstacleDetector
from tools.bag_reader import iter_frames


PROGRESS_EVERY = 50


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def blas_threads() -> int:
    return max((pool["num_threads"] for pool in threadpool_info() if pool["user_api"] == "blas"),
               default=0)


def run(cfg: Config, record: str, stride: int, limit: int | None,
        timings: list[dict], confirm: bool, without_ring_time: bool = False) -> list[dict]:
    detector = ObstacleDetector(cfg, confirm=confirm)
    threads = blas_threads()
    rows: list[dict] = []
    for frame in iter_frames(cfg.data.path(record), cfg.bag, stride=stride, limit=limit,
                             partial=without_ring_time):
        ring, timestamp = (None, None) if without_ring_time else (frame.ring, frame.timestamp)
        outcome = detector.process(
            frame.xyz, frame.intensity, ring, timestamp,
            frame.stamp_ns, frame_gap=stride,
        )

        row = outcome.to_dict()
        row["frame"] = int(frame.index)
        row["stamp_ns"] = int(frame.stamp_ns)
        if not outcome.debug["checked"]:
            row.update(staleness_frames=0, axis_source="none", axis_method=None, x_traced_m=None,
                       axis_meets_clearance=None, axis_state="unchecked",
                       axis_cross_check_reason=None, axis_reason=outcome.debug["reason"],
                       candidates_found=False, raw=None, advance_m=None, stop_axis_loss=False)
            rows.append(row)
            continue

        axis = outcome.debug["axis"]
        if axis["state"] == "lost":
            axis = None
        row["staleness_frames"] = 0 if axis is None else axis["staleness_frames"]
        row["axis_source"] = "none" if axis is None else axis["source"]
        row["axis_method"] = None if axis is None else axis["method"]
        row["x_traced_m"] = None if axis is None else axis["x_trusted_m"]


        row["axis_meets_clearance"] = None if axis is None else bool(axis["meets_clearance"])
        row["axis_state"] = "lost" if axis is None else axis["state"]
        row["axis_cross_check_reason"] = None if axis is None else axis["cross_check_reason"]
        row["axis_reason"] = outcome.debug["axis"]["reason"]
        candidates = outcome.debug["candidates_raw"]
        row["candidates_found"] = bool(candidates["detections"])
        row["raw"] = candidates if candidates["detections"] else None
        row["advance_m"] = outcome.debug["advance_m"]
        row["stop_axis_loss"] = bool(outcome.debug["verdict"]["stop"])
        rows.append(row)

        stage = outcome.debug["timings_ms"]
        timings.append({
            "record": record,
            "frame": int(frame.index),
            "n_points_raw": int(frame.xyz.shape[0]),
            "n_points_kept": int(outcome.debug["n_points"]),
            "preprocess_ms": stage["preprocess"],
            "odometry_ms": stage["odometry"],
            "axis_ms": stage["axis"],
            "detect_ms": stage["detect"],
            "confirm_ms": stage["confirm"],
            "total_ms": stage["total"],
            "blas_threads": threads,
            "cpu_count": os.cpu_count(),
        })
        if len(rows) % PROGRESS_EVERY == 0:
            print(f"    {record}: {len(rows)} кадров, {timings[-1]['total_ms']:.0f} мс/кадр",
                  flush=True)

    tracker = detector.axis_tracker
    span_ns = (rows[-1]["stamp_ns"] - rows[0]["stamp_ns"]) if len(rows) > 1 else 0
    rows.append({"_summary": {
        "record": record,
        "span_s": round(span_ns / 1e9, 2),
        "stride": stride,
        "axis_measured": tracker.n_measured,
        "axis_stale": tracker.n_stale,
        "axis_rejected": tracker.n_rejected,
        "axis_lost": tracker.n_lost,
        "axis_rejected_gate": tracker.n_rejected_gate,
        "axis_uninitialised": tracker.n_uninitialised,

        "axis_causes": dict(tracker.causes),

        "longest_gap_frames": tracker.longest_gap_frames,
        "longest_gap_path_m": round(tracker.longest_gap_path_m, 2),
        "confirm": None if detector.confirmer is None else {
            "required_frames": cfg.confirm.required_frames,
            "window_frames": cfg.confirm.window_frames,
            "n_confirmed_tracks": detector.confirmer.n_confirmed_tracks,
            "latency_frames": sorted(detector.confirmer.latencies),
        },
        "stop_axis_loss": detector.verdict.n_triggers,
    }})
    return rows


def summarise(rows: list[dict]) -> None:
    frames = [r for r in rows if "_summary" not in r]
    summary = next(r["_summary"] for r in rows if "_summary" in r)
    n = len(frames)
    with_axis = [r for r in frames if r["axis_source"] != "none"]
    fired = [r for r in frames if r["obstacle_found"]]
    stale_fired = [r for r in fired if r["staleness_frames"] > 0]
    print(f"  кадров {n}")


    print(f"  исход: измерена {summary['axis_measured']}, удержана "
          f"{summary['axis_stale']}, потеряна {summary['axis_lost']} "
          f"(сумма {summary['axis_measured'] + summary['axis_stale'] + summary['axis_lost']} "
          f"из {n})")
    print(f"  причины для неизмеренных: {summary['axis_causes']}")
    print(f"  максимальная серия без измеренной оси: {summary['longest_gap_frames']} кадров")
    states = {}
    for r in frames:
        states[r["axis_state"]] = states.get(r["axis_state"], 0) + 1
    print(f"  состояния оси: {states}")
    if with_axis:
        base = np.array([r["x_traced_m"] for r in with_axis], dtype=float)
        print(f"  база оси: медиана {np.median(base):.1f} м, p10 {np.percentile(base, 10):.1f}, "
              f"max {base.max():.1f}")
    print(f"  сработало в {len(fired)} кадрах из {n} ({100 * len(fired) / max(n, 1):.1f} %), "
          f"из них на устаревшей оси {len(stale_fired)}")
    confirm = summary.get("confirm")
    if confirm is not None:
        raw = sum(1 for r in frames if r.get("candidates_found"))
        latency = confirm["latency_frames"]
        print(f"  подтверждение {confirm['required_frames']} из "
              f"{confirm['window_frames']}: кандидатов в {raw} кадрах, "
              f"подтверждённых треков {confirm['n_confirmed_tracks']}, "
              f"задержка медиана "
              f"{np.median(latency) if latency else float('nan'):.0f} кадров")
    print(f"  остановок по В₂: {summary.get('stop_axis_loss', 0)}")
    if fired:
        d = np.array([r["nearest_distance_m"] for r in fired], dtype=float)
        s = np.array([det["score"] for r in fired for det in r["detections"]], dtype=float)
        print(f"  ближайшая дистанция: медиана {np.median(d):.1f} м, min {d.min():.1f}, max {d.max():.1f}")
        print(f"  уверенность: медиана {np.median(s):.2f}, p10 {np.percentile(s, 10):.2f}")


def write_timings(path: Path, timings: list[dict]) -> None:
    if not timings:
        return
    fresh = {row["record"] for row in timings}
    kept = []
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as handle:
            kept = [row for row in csv.DictReader(handle) if row["record"] not in fresh]
    fields = list(timings[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in kept)
        writer.writerows(timings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--records", nargs="+", required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--confirm", action="store_true", default=None,
                        help="включить подтверждение K из M (по умолчанию из конфига)")
    parser.add_argument("--no-confirm", dest="confirm", action="store_false",
                        help="только слой A: строки прогона остаются сырыми")
    parser.add_argument("--without-ring-time", action="store_true",
                        help="не передавать ядру ring и timestamp: пары эха ищутся по направлению")
    parser.add_argument("--out", default="runs/layer_a")
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    if args.confirm is None:
        args.confirm = cfg.confirm.enabled
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)


    cfg.snapshot(out_dir / "config.snapshot.yaml")

    timings: list[dict] = []
    started = time.perf_counter()
    for record in args.records:
        print(f"\n=== {record}", flush=True)
        before = peak_rss_mb()
        rows = run(cfg, record, args.stride, args.limit, timings, args.confirm,
                   args.without_ring_time)
        summarise(rows)


        print(f"  пик RSS: {peak_rss_mb():.0f} МБ (до записи {before:.0f})", flush=True)
        target = out_dir / f"{record}.jsonl"
        with target.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  сохранено: {target}", flush=True)
        write_timings(out_dir / "timings.csv", timings)

    elapsed = time.perf_counter() - started
    print(f"\nвсего {len(timings)} кадров за {elapsed:.1f} с, "
          f"пик RSS {peak_rss_mb():.0f} МБ", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

