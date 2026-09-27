# Сводная таблица ложных срабатываний и эпизодов по прогонам записей.

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from core.config import Config
from tools.episode_table import EPISODE_GAP_S, group_by_gap


STANDING_PATH_M = 1.0


POSITION_BUCKETS = 10


def load(path: Path) -> tuple[list[dict], dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    frames = [r for r in rows if "_summary" not in r]
    summary = next((r["_summary"] for r in rows if "_summary" in r), {})
    return frames, summary


def shape_of(detection: dict, elongation: float) -> str:
    lo = detection["bbox_min_xyz"]
    hi = detection["bbox_max_xyz"]
    length = hi[0] - lo[0]
    height = max(hi[2] - lo[2], 1e-3)
    return "протяжённый" if length / height > elongation else "компактный"


def count_episodes(stamps_ns: list[int], gap_s: float = EPISODE_GAP_S) -> int:
    return len(group_by_gap([{"stamp_ns": s} for s in stamps_ns], gap_s))


def travelled_path(speed_csv: Path) -> tuple[float | None, float]:
    if not speed_csv.is_file():
        return None, float("nan")
    path_m, n_rows, n_degenerate = 0.0, 0, 0
    with speed_csv.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            n_rows += 1
            if int(row["degenerate"]):
                n_degenerate += 1
                continue
            if not row["speed_mps"]:
                continue
            path_m += abs(float(row["speed_mps"])) * float(row["dt_s"])
    if n_rows == 0:
        return None, float("nan")
    return path_m, 100.0 * n_degenerate / n_rows


def describe(record: str, frames: list[dict], summary: dict,
             elongation: float, speed_dir: Path) -> dict:
    n = len(frames)
    fired = [r for r in frames if r["obstacle_found"]]
    stale_fired = [r for r in fired if r["staleness_frames"] > 0]
    detections = [d for r in frames for d in r["detections"]]
    shapes = [shape_of(d, elongation) for d in detections]
    compact = [d for d, s in zip(detections, shapes) if s == "компактный"]

    alarm_frames = [
        r for r in frames
        if any(shape_of(d, elongation) == "компактный" for d in r["detections"])
    ]


    valid = [r for r in frames if r.get("axis_meets_clearance") is True]
    valid_alarms = [r for r in alarm_frames if r.get("axis_meets_clearance") is True]
    broken = [r for r in frames if r.get("axis_meets_clearance") is False]
    broken_alarms = [r for r in alarm_frames if r.get("axis_meets_clearance") is False]
    episodes = count_episodes([r["stamp_ns"] for r in alarm_frames])


    candidate_frames = [r for r in frames if r.get("candidates_found")]
    candidate_episodes = count_episodes([r["stamp_ns"] for r in candidate_frames])


    fired_episodes = count_episodes([r["stamp_ns"] for r in fired])

    base = np.array([r["x_traced_m"] for r in frames if r["x_traced_m"] is not None], dtype=float)
    minutes = summary.get("span_s", 0.0) / 60.0
    path_m, degenerate_share = travelled_path(speed_dir / f"{record}.csv")
    standing = path_m is not None and path_m < STANDING_PATH_M

    return {
        "record": record,
        "frames": n,
        "axis_valid": len(valid),
        "axis_broken": len(broken),
        "alarm_valid": len(valid_alarms),
        "alarm_valid_share": 100.0 * len(valid_alarms) / max(len(valid), 1),
        "alarm_broken": len(broken_alarms),
        "alarm_broken_share": 100.0 * len(broken_alarms) / max(len(broken), 1),
        "minutes": minutes,
        "no_axis": sum(1 for r in frames if r["axis_source"] == "none"),
        "base_median_m": float(np.median(base)) if base.size else float("nan"),
        "fired": len(fired),
        "fired_share": 100.0 * len(fired) / max(n, 1),
        "fired_stale": len(stale_fired),
        "detections": len(detections),
        "elongated": sum(1 for s in shapes if s == "протяжённый"),
        "compact": len(compact),
        "alarm_frames": len(alarm_frames),
        "alarm_share": 100.0 * len(alarm_frames) / max(n, 1),
        "episodes": episodes,
        "candidate_frames": len(candidate_frames),
        "candidate_episodes": candidate_episodes,
        "fired_episodes": fired_episodes,
        "confirmed": bool(candidate_frames),
        "stop_axis_loss": summary.get("stop_axis_loss", 0),
        "episodes_per_min": episodes / minutes if minutes > 0 else float("nan"),
        "path_m": path_m,
        "standing": standing,
        "degenerate_share": degenerate_share,
        "episodes_per_km": (
            None if standing or not path_m else 1000.0 * episodes / path_m
        ),
        "compact_distance_median": (
            float(np.median([d["distance_m"] for d in compact])) if compact else float("nan")
        ),
        "positions": position_profile(frames, elongation),
    }


def position_profile(frames: list[dict], elongation: float) -> list[float]:
    if not frames:
        return []
    flags = np.array([
        any(shape_of(d, elongation) == "компактный" for d in r["detections"])
        for r in frames
    ], dtype=float)
    chunks = np.array_split(flags, min(POSITION_BUCKETS, flags.size))
    return [100.0 * float(chunk.mean()) for chunk in chunks]


def render(table: list[dict], cfg: Config, elongation: float) -> str:
    out: list[str] = []
    if any(row["confirmed"] for row in table):
        out.append("### До и после подтверждения «K из M»")
        out.append("")
        out.append("Кандидат — кластер, выживший в слое A. Тревога — кандидат, "
                   "подтверждённый во времени.\nБез первой колонки нельзя отличить "
                   "«детектор стал точнее» от «подтверждение всё съело».")
        out.append("")
        out.append("| запись | кадров с кандидатом | кадров с тревогой | "
                   "эпизодов до | эпизодов после | остановок по В₂ |")
        out.append("|---|---:|---:|---:|---:|---:|")
        for row in table:
            out.append(
                f"| `{row['record']}` | {row['candidate_frames']} | {row['fired']} | "
                f"{row['candidate_episodes']} | {row['fired_episodes']} | "
                f"{row['stop_axis_loss']} |")
        out.append("")
    out.append("| запись | кадров | мин | без оси | база | кадров с тревогой | "
               "компактных (тревога) | эпизодов | эпизодов/мин | эпизодов/км |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in table:
        per_km = "н/п" if row["episodes_per_km"] is None else f"{row['episodes_per_km']:.1f}"
        out.append(
            f"| `{row['record']}` | {row['frames']} | {row['minutes']:.2f} | {row['no_axis']} | "
            f"{row['base_median_m']:.1f} м | {row['fired']} ({row['fired_share']:.1f} %) | "
            f"{row['alarm_frames']} ({row['alarm_share']:.1f} %) | {row['episodes']} | "
            f"{row['episodes_per_min']:.1f} | {per_km} |"
        )
    out.append("")
    out.append("### Раздельно: валидная ось против отказа модели оси")
    out.append("")
    out.append("Ось, не укладывающаяся в нормативный зазор 0.113 м даже на минимальной базе, "
               "— это отказ\nМОДЕЛИ оси, а не срабатывание детектора. На стрелочном переводе "
               "рельсы расходятся и прямая\nтам не определена в принципе. Смешивать эти "
               "кадры с валидными значит выдавать одно за другое.")
    out.append("")
    out.append("| запись | кадров с валидной осью | тревог на них | кадров вне норматива | тревог на них |")
    out.append("|---|---:|---:|---:|---:|")
    for row in table:
        out.append(
            f"| `{row['record']}` | {row['axis_valid']} | "
            f"{row['alarm_valid']} ({row['alarm_valid_share']:.1f} %) | "
            f"{row['axis_broken']} | {row['alarm_broken']} ({row['alarm_broken_share']:.1f} %) |"
        )
    out.append("")
    out.append(f"«Компактный» = длина/высота ≤ {elongation:.0f}. Тревогой считается кадр "
               "с компактным кластером: протяжённая\nконструкция (контактный рельс, бортик, "
               "стена на кривой) — другой отказ, и число её кластеров зависит\nот порогов, "
               "режущих её на куски, а не от числа отказов.")
    out.append("")
    out.append(f"Эпизод — подряд идущие кадры с тревогой, разрыв до {EPISODE_GAP_S:.0f} с "
               "не разрывает эпизод; это одна ложная\nкоманда на торможение. Допуск "
               "зафиксирован в коде метрики и намеренно не взят из параметров слоя C.")
    out.append("")
    out.append("Колонка «эпизодов/км» справочная: знаменатель берётся из замера скорости "
               "и вырождается там, где\nучасток беден ориентирами. Для стоящего состава — "
               "«н/п».")
    out.append("")
    out.append("| запись | путь, м | вырожденных кадров одометрии |")
    out.append("|---|---:|---:|")
    for row in table:
        path = "стоит" if row["standing"] else (
            "—" if row["path_m"] is None else f"{row['path_m']:.0f}")
        deg = "—" if np.isnan(row["degenerate_share"]) else f"{row['degenerate_share']:.0f} %"
        out.append(f"| `{row['record']}` | {path} | {deg} |")

    out.append("")
    out.append("### Где именно сыпется: доля кадров с тревогой по позиции вдоль записи")
    out.append("")
    out.append("Десятые доли записи, от начала к концу, проценты. Разметки участков нет — "
               "это позиция по ходу\nзаписи, а не название участка.")
    out.append("")
    out.append("| запись | " + " | ".join(f"{i + 1}" for i in range(POSITION_BUCKETS)) + " |")
    out.append("|---|" + "---:|" * POSITION_BUCKETS)
    for row in table:
        cells = " | ".join(f"{v:.0f}" for v in row["positions"])
        out.append(f"| `{row['record']}` | {cells} |")

    obstacle = next((r for r in table if r["record"] == cfg.data.obstacle_record), None)
    if obstacle is not None:
        out.append("")
        out.append(f"**Полнота на `{cfg.data.obstacle_record}`: НЕПРИМЕНИМО.** Единственный "
                   f"положительный пример лежит на ≈56 м,\nбаза оси "
                   f"{obstacle['base_median_m']:.0f} м. Позитив за пределами ближней зоны "
                   "по построению, и ноль здесь был бы\nсвойством геометрии, а не детектора. "
                   "Измеримая часть этой записи — избирательность: ближняя\nбегущая фигура "
                   "в междупутье (кадры 165–200) целиком внутри ближней зоны и целиком "
                   "негативна.")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runs", default="runs/layer_a")
    parser.add_argument("--speed", default="runs/speed",
                        help="каталог с замерами скорости для колонки «эпизодов/км»")
    parser.add_argument("--elongation", type=float, default=6.0,
                        help="длина/высота выше этого — протяжённая конструкция, не объект")
    parser.add_argument("--summary", action="store_true",
                        help="записать таблицу в <runs>/summary.md")
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    runs = Path(args.runs)
    speed_dir = Path(args.speed)
    table = []
    for record in cfg.data.records:
        path = runs / f"{record}.jsonl"
        if not path.is_file():
            continue
        frames, summary = load(path)
        table.append(describe(record, frames, summary, args.elongation, speed_dir))

    if not table:
        print(f"нет результатов в {runs}", file=sys.stderr)
        return 1

    text = render(table, cfg, args.elongation)
    print(text)
    if args.summary:
        target = runs / "summary.md"
        target.write_text(
            f"# Слой A: ложные срабатывания\n\nПрогон `{runs}`, конфиг "
            f"`{runs / 'config.snapshot.yaml'}`.\n\n{text}\n", encoding="utf-8")
        print(f"\nсохранено: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

