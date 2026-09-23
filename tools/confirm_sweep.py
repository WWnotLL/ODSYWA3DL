# Перебор параметров подтверждения «K из M» по готовым прогонам: ложные эпизоды против позитивов.

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from core.confirm import AxisLossVerdict, Confirmer
from core.config import Config
from core.types import Detection, FrameResult
from tools.false_alarm_table import count_episodes, load
from tools.positive_control import _overlaps
from tools.scenarios import build_parser as scenario_parser


def advances(speed_csv: Path) -> dict[int, float | None]:
    if not speed_csv.is_file():
        return {}
    out: dict[int, float | None] = {}
    with speed_csv.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            if int(row["degenerate"]) or not row["speed_mps"]:
                out[frame] = None
            else:
                out[frame] = abs(float(row["speed_mps"])) * float(row["dt_s"])
    return out


def _as_candidates(row: dict) -> FrameResult:
    debug_raw = row.get("candidates_raw", (row.get("debug") or {}).get("candidates_raw"))
    if debug_raw is not None or "candidates_raw" in row:
        raw = debug_raw or {"detections": [], "lateral_offset_m": []}
        debug = {"lateral_offset_m": raw["lateral_offset_m"],
                 "axis": (row.get("debug") or {}).get("axis", {"state": "measured"})}
        return FrameResult(bool(raw["detections"]), None,
                           [Detection.from_dict(d) for d in raw["detections"]], debug)
    raw = row.get("raw")
    if raw is None:
        if row.get("candidates_found"):
            raise ValueError(
                f"кадр {row.get('frame')}: кандидаты были, но не сохранены — "
                "прогон сделан до появления поля `raw`, перебор по нему неверен"
            )
        return FrameResult.from_dict({**row, "detections": [], "obstacle_found": False})
    debug = dict(row["debug"])
    debug["lateral_offset_m"] = raw["lateral_offset_m"]
    debug.pop("confirm", None)
    return FrameResult(True, None, [Detection.from_dict(d) for d in raw["detections"]], debug)


def _checked(row: dict) -> bool:
    return bool(row.get("checked", (row.get("debug") or {}).get("checked", True)))


def debug_advances(frames: list[dict]) -> dict[int, float | None]:
    return {int(r["frame"]): r.get("advance_m", (r.get("debug") or {}).get("advance_m"))
            for r in frames}


def replay_positive(frames: list[dict], cfg: Config, required: int, window: int,
                    tolerance: float) -> dict:
    confirmer = Confirmer(replace(cfg.confirm, required_frames=required, window_frames=window))
    shifts = debug_advances(frames)
    hit_frames, body_frames = 0, 0
    first_seen: dict[int, float] = {}
    runs: set[int] = set()
    for row in frames:
        if not _checked(row):
            continue
        out = confirmer.update(_as_candidates(row), advance_m=shifts.get(int(row["frame"])))
        if row["body_box"] is None:
            continue
        body_frames += 1
        runs.add(row["run"])
        lo, hi = (np.asarray(v, float) for v in row["body_box"])
        if _overlaps(out.detections, lo, hi, tolerance):
            hit_frames += 1
            first_seen.setdefault(row["run"], row["body_centre_x_m"])
    return {"frames": body_frames, "hit_frames": hit_frames, "runs": len(runs),
            "runs_found": len(first_seen),
            "first_m": float(np.median(list(first_seen.values()))) if first_seen else None}


def _cell(p: dict) -> str:
    first = "—" if p["first_m"] is None else f"{p['first_m']:.1f}"
    return f"{p['hit_frames']}/{p['frames']} / {p['runs_found']}/{p['runs']} / {first}"


def replay(frames: list[dict], cfg: Config, shifts: dict[int, float | None],
           required: int, window: int) -> dict:
    confirm_cfg = replace(cfg.confirm, required_frames=required, window_frames=window)
    confirmer = Confirmer(confirm_cfg)
    verdict = AxisLossVerdict(cfg.confirm.verdict)

    alarm_stamps: list[int] = []
    raw_stamps: list[int] = []
    n_confirmed_frames = 0
    for row in frames:
        if not _checked(row):
            continue
        result = _as_candidates(row)
        raw = bool(result.detections)
        out = confirmer.update(result, advance_m=shifts.get(int(row["frame"])))


        verdict.update(out, had_candidate=raw)
        if raw:
            raw_stamps.append(int(row["stamp_ns"]))
        if out.obstacle_found:
            n_confirmed_frames += 1
            alarm_stamps.append(int(row["stamp_ns"]))
    return {
        "frames": len(frames),
        "raw_frames": len(raw_stamps),
        "raw_episodes": count_episodes(raw_stamps),
        "alarm_frames": n_confirmed_frames,
        "episodes": count_episodes(alarm_stamps),
        "verdict_v2": verdict.n_triggers,
        "latencies": list(confirmer.latencies),
        "residuals": np.array(confirmer.residuals) if confirmer.residuals else np.empty((0, 3)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runs", default="runs/layer_a_confirm")
    parser.add_argument("--speed", default="runs/speed")
    parser.add_argument("--required", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--window", type=int, nargs="+", default=[5])
    parser.add_argument("--records", nargs="+", default=None,
                        help="по умолчанию пять чистых записей")
    parser.add_argument("--advance-from", choices=["speed", "debug"], default="speed",
                        help="путь за кадр: замер скорости или одометрия ядра из прогона")
    parser.add_argument("--positives", nargs="*", default=[],
                        help="имя=файл: выход ядра на сценарии (scenarios --dump-results)")
    parser.add_argument("--tolerance", type=float,
                        default=scenario_parser().get_default("tolerance"),
                        help="допуск пересечения с телом, по умолчанию — как у tools.scenarios")
    args = parser.parse_args(argv)
    positives = {}
    for item in args.positives:
        name, path = item.split("=", 1)
        positives[name] = [json.loads(line) for line in Path(path).read_text(
            encoding="utf-8").splitlines()]

    cfg = Config.from_yaml(args.config)
    runs, speed_dir = Path(args.runs), Path(args.speed)
    records = tuple(args.records) if args.records else cfg.data.clean_records

    loaded: dict[str, tuple[list[dict], dict]] = {}
    for record in records:
        path = runs / f"{record}.jsonl"
        if not path.is_file():
            print(f"нет прогона: {path}", file=sys.stderr)
            continue
        loaded[record] = load(path)
    if not loaded:
        return 1

    minutes = sum(s.get("span_s", 0.0) for _, s in loaded.values()) / 60.0
    print(f"записей {len(loaded)}, суммарно {minutes:.1f} мин\n")

    residuals = np.empty((0, 3))
    def shifts_of(record: str, frames: list[dict]) -> dict[int, float | None]:
        if args.advance_from == "debug":
            return debug_advances(frames)
        return advances(speed_dir / f"{record}.csv")

    pos_head = "".join(f" {name}: кадров / заходов / первая, м |" for name in positives)
    print("| K из M | кадров с тревогой | эпизодов | эпизодов/мин | В₂ | задержка, кадров |"
          + pos_head)
    print("|---|---:|---:|---:|---:|---:|" + "---|" * len(positives))
    grid: dict[tuple[int, int], tuple] = {}
    for window in args.window:
        for required in args.required:
            if required > window:
                continue
            totals = {"alarm_frames": 0, "episodes": 0, "verdict_v2": 0, "frames": 0}
            latencies: list[int] = []
            for record, (frames, _) in loaded.items():
                shifts = shifts_of(record, frames)
                out = replay(frames, cfg, shifts, required, window)
                for key in totals:
                    totals[key] += out[key]
                latencies += out["latencies"]
                if required == 1 and residuals.size == 0:
                    residuals = out["residuals"]
                elif required == 1:
                    residuals = np.vstack([residuals, out["residuals"]])
            share = 100.0 * totals["alarm_frames"] / max(totals["frames"], 1)
            delay = f"{np.median(latencies):.0f}" if latencies else "—"
            pos = {name: replay_positive(rows, cfg, required, window, args.tolerance)
                   for name, rows in positives.items()}
            grid[(required, window)] = (totals["episodes"], totals["alarm_frames"], pos)
            cells = "".join(f" {_cell(p)} |" for p in pos.values())
            print(f"| {required} из {window} | {totals['alarm_frames']} ({share:.2f} %) | "
                  f"{totals['episodes']} | {totals['episodes'] / minutes:.1f} | "
                  f"{totals['verdict_v2']} | {delay} |" + cells)

    if residuals.size:
        print("\nНевязки сопоставленных пар (K=1), метры — из них и берутся ворота:")
        print("| | p50 | p90 | p99 | max |")
        print("|---|---:|---:|---:|---:|")
        for name, column in zip(("продольная", "поперечная", "вертикальная"), residuals.T):
            print(f"| {name} | {np.percentile(column, 50):.2f} | "
                  f"{np.percentile(column, 90):.2f} | {np.percentile(column, 99):.2f} | "
                  f"{column.max():.2f} |")

    current = (cfg.confirm.required_frames, cfg.confirm.window_frames)
    if positives and current in grid:
        base_episodes, _, base_pos = grid[current]

        def not_worse(p: dict, b: dict) -> bool:
            return (p["hit_frames"] >= b["hit_frames"] and p["runs_found"] >= b["runs_found"]
                    and (b["first_m"] is None
                         or (p["first_m"] is not None and p["first_m"] >= b["first_m"])))

        better = [key for key, (episodes, _, pos) in grid.items()
                  if key != current and episodes <= base_episodes
                  and all(not_worse(pos[n], base_pos[n]) for n in pos)
                  and (episodes < base_episodes or any(pos[n] != base_pos[n] for n in pos))]
        print(f"\nСтрого не хуже {current[0]} из {current[1]} (эпизодов не больше; на каждом "
              "теле кадров, заходов и дистанции первой детекции не меньше; хоть в чём-то "
              f"лучше): {', '.join(f'{k} из {m}' for k, m in better) or 'нет'}")

    print("\nПо записям при текущем конфиге "
          f"({cfg.confirm.required_frames} из {cfg.confirm.window_frames}):")
    print("| запись | кадров | тревог было | стало | эпизодов было | стало | В₂ |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for record, (frames, _) in loaded.items():
        shifts = shifts_of(record, frames)
        out = replay(frames, cfg, shifts,
                     cfg.confirm.required_frames, cfg.confirm.window_frames)
        print(f"| `{record}` | {out['frames']} | {out['raw_frames']} | {out['alarm_frames']} | "
              f"{out['raw_episodes']} | {out['episodes']} | {out['verdict_v2']} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())

