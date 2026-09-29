# Сценарии с виртуальным телом в реальных кадрах (сближение, появление, у кромки) через всё ядро.

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from core.axis import AxisTracker, estimate_axis
from core.confirm import AxisLossVerdict, Confirmer
from core.config import Config, VerdictConfig
from core.detector import detect
from core.pipeline import ObstacleDetector
from core.preprocess import (GroundTracker, PreprocessResult, check_proper_rotation,
                             preprocess_frame, rotation_between)
from tools.bag_reader import RawFrame, iter_frames
from tools.positive_control import (BODY_HEIGHT_M, BODY_RADIUS_M, _overlaps,
                                    inject_body, inject_tilted_body, real_figure_track)


class VerdictCounter:
    def __init__(self, cfg: VerdictConfig) -> None:
        self._v2 = AxisLossVerdict(cfg)
        self.v1 = 0
        self._seen_measured = False
        self._in_failure = False

    @property
    def v2(self) -> int:
        return self._v2.n_triggers

    def feed(self, state: str, had_cluster: bool) -> None:
        self._v2.feed(state, had_cluster)
        if state in ("measured", "held"):
            self._in_failure = False
            self._seen_measured = self._seen_measured or state == "measured"
            return
        if not self._in_failure:
            self._in_failure = True
            if self._seen_measured:
                self.v1 += 1


def _to_leveled(plane, cfg: Config) -> np.ndarray:
    axes = check_proper_rotation(cfg.preprocess.axes.rotation, cfg.preprocess.numerics)
    level = rotation_between(plane.normal, np.array([0.0, 0.0, 1.0]), cfg.preprocess.numerics)
    return level @ axes


def _placement(scenario: str, cfg: Config, axis, distance_m: float,
               args) -> tuple[tuple[float, float], float, float, float]:
    railhead = cfg.gauge.rail_head_offset_m or 0.0
    if scenario == "edge":
        half = float(cfg.gauge.half_width_at(np.array([args.edge_height_m]))[0])
        radius = args.edge_radius_m
        offset = half - args.edge_penetration_m + radius
        z_bottom = railhead + args.edge_height_m - 0.5 * args.edge_body_height_m
        return ((distance_m, float(axis.y_at(distance_m)) - offset),
                z_bottom, radius, args.edge_body_height_m)


    base = railhead if args.body_base_m is None else args.body_base_m
    return ((distance_m, float(axis.y_at(distance_m)) + args.body_offset_m),
            base, args.body_radius, args.body_height)


def _tilt(tilt_deg: float) -> np.ndarray:
    angle = np.radians(tilt_deg)
    return np.array([0.0, np.sin(angle), np.cos(angle)])


def _inject(xyz: np.ndarray, sensor: np.ndarray, geometry, tilt_deg: float) -> tuple[np.ndarray, int]:
    centre, z_bottom, radius, height = geometry
    if tilt_deg == 0.0:
        return inject_body(xyz, sensor, centre, z_bottom, radius, height)
    return inject_tilted_body(xyz, sensor, (centre[0], centre[1], z_bottom), _tilt(tilt_deg),
                              radius, height)


def body_box(geometry, tilt_deg: float) -> tuple[np.ndarray, np.ndarray]:
    centre, z_bottom, radius, height = geometry
    direction = _tilt(tilt_deg)
    top_y = centre[1] + height * direction[1]
    spread = radius * abs(direction[1])
    lo = np.array([centre[0] - radius, min(centre[1], top_y) - radius * direction[2], z_bottom - spread])
    hi = np.array([centre[0] + radius, max(centre[1], top_y) + radius * direction[2],
                   z_bottom + height * direction[2] + spread])
    return lo, hi


@dataclass(frozen=True, eq=False)
class ScenarioFrame:
    frame: RawFrame
    clean: PreprocessResult
    injected: np.ndarray
    n_points: int
    centre: tuple[float, float] | None
    geometry: tuple | None
    run_index: int


def scenario_range(cfg: Config, args) -> tuple[dict, int, int]:
    record = args.record or cfg.data.obstacle_record
    if args.scenario in ("appearance", "cold_start", "edge"):
        figures = real_figure_track(Path("runs/oracle") / f"{record}.jsonl")
        if not figures:
            raise SystemExit("нет разметки оракула — сначала python -m tools.oracle")
        return figures, min(figures), max(figures)
    return {}, args.first_frame, args.last_frame


def _extension_correction(xyz: np.ndarray, cfg: Config) -> tuple[float, float] | None:
    from core.axis import _bed_centres, _rail_centres
    from tools.measure_extension_residual import extension_correction
    rails, _ = _rail_centres(xyz, cfg.axis)
    bed, _ = _bed_centres(xyz, cfg.axis)
    if rails is None or bed is None:
        return None
    fix = extension_correction(rails, bed, cfg.axis)
    return None if fix is None else (fix[0], fix[1])


def iter_scenario_frames(cfg: Config, args, *, raw: bool) -> Iterator[ScenarioFrame]:
    record = args.record or cfg.data.obstacle_record
    figures, first, last = scenario_range(cfg, args)
    ground = GroundTracker(cfg.preprocess.ground)
    clean_tracker = AxisTracker(cfg.axis)
    approach: float | None = None
    run_index = 0

    for frame in iter_frames(cfg.data.path(record), cfg.bag, limit=last + 1):
        result = preprocess_frame(
            frame.xyz, frame.intensity, frame.ring, frame.timestamp,
            cfg.preprocess, tracker=ground,
        )
        clean_axis = clean_tracker.update(result.xyz, advance_m=args.advance_m)
        active = frame.index >= first

        if args.scenario == "cold_start" and not active:
            continue


        injected, n_points, centre, geometry = (frame.xyz if raw else result.xyz), 0, None, None
        if active and clean_axis is not None:
            if figures:
                if frame.index not in figures:
                    continue
                distance = float(figures[frame.index][0])
            else:
                if clean_axis.residual_max_m > cfg.axis.clearance_m:
                    continue
                far = clean_axis.x_end_m - args.base_margin_m
                if args.scenario == "approach":
                    if approach is None or approach < cfg.detector.min_range_m:
                        approach = min(args.distance_m, far)
                        run_index += 1
                    distance = min(approach, far)
                    approach -= args.advance_m
                else:
                    distance = min(args.distance_m, far)
                if distance < cfg.detector.min_range_m:
                    continue
            geometry = _placement(args.scenario, cfg, clean_axis, distance, args)
            if (args.placement_axis == "corrected" and clean_axis.x_joint_m is not None
                    and clean_axis.far_method == "bed"
                    and distance > clean_axis.x_joint_m):
                fix = _extension_correction(result.xyz, cfg)
                if fix is None:
                    yield ScenarioFrame(frame, result, injected, 0, None, None, run_index)
                    continue
                (cx, cy), z_bottom, radius, height = geometry
                geometry = ((cx, cy - (fix[0] * cx + fix[1])), z_bottom, radius, height)
            centre = geometry[0]
            sensor = np.array([0.0, 0.0, float(result.plane.offset)])
            if not raw:
                injected, n_points = _inject(result.xyz, sensor, geometry, args.body_tilt_deg)
            else:
                matrix = _to_leveled(result.plane, cfg)
                leveled = frame.xyz @ matrix.T
                leveled[:, 2] += float(result.plane.offset)
                leveled, n_points = _inject(leveled, sensor, geometry, args.body_tilt_deg)
                leveled[:, 2] -= float(result.plane.offset)
                injected = leveled @ matrix
        yield ScenarioFrame(frame, result, injected, n_points, centre, geometry, run_index)


def _dump_row(frame, run_index, centre, geometry, outcome, tilt_deg) -> dict:
    box = None
    if centre is not None:
        lo, hi = body_box(geometry, tilt_deg)
        box = [lo.tolist(), hi.tolist()]
    debug = outcome.debug
    return {
        "frame": int(frame.index),
        "stamp_ns": int(frame.stamp_ns),
        "run": run_index,
        "body_centre_x_m": None if centre is None else round(float(centre[0]), 3),
        "body_box": box,
        "checked": bool(debug.get("checked", False)),
        "advance_m": debug.get("advance_m"),
        "candidates_raw": debug.get("candidates_raw"),
        "corridor_end_m": (debug.get("corridor") or {}).get("x_m", [None, None])[1],
        "axis_state": (debug.get("axis") or {}).get("state"),
    }


def run(cfg: Config, args) -> list[dict]:
    if args.via_pipeline and args.fail_every:
        raise SystemExit("--fail-every подаёт трассеру пустое облако в обход ядра "
                         "и с --via-pipeline несовместим")


    body_tracker = None if args.via_pipeline else AxisTracker(cfg.axis)


    if args.detector_config and not args.via_pipeline:
        raise SystemExit("--detector-config относится к ядру и требует --via-pipeline")
    det_cfg = Config.from_yaml(args.detector_config) if args.detector_config else cfg
    detector = ObstacleDetector(det_cfg, confirm=args.confirm) if args.via_pipeline else None


    shadow_ground = (GroundTracker(cfg.preprocess.ground)
                     if args.trace_anchor and args.via_pipeline else None)
    verdict = VerdictCounter(cfg.confirm.verdict)


    confirmer = Confirmer(cfg.confirm) if args.confirm and not args.via_pipeline else None
    rows: list[dict] = []
    if args.dump_results and not args.via_pipeline:
        raise SystemExit("--dump-results пишет выход ядра и требует --via-pipeline")
    dump = open(args.dump_results, "w", encoding="utf-8") if args.dump_results else None

    for item in iter_scenario_frames(cfg, args, raw=detector is not None):
        frame, result, injected = item.frame, item.clean, item.injected
        n_points, centre, geometry = item.n_points, item.centre, item.geometry
        run_index = item.run_index


        forced = (args.fail_every > 0
                  and (frame.index % args.fail_every) < args.fail_length)
        trace = None
        if args.trace_anchor:
            if shadow_ground is not None:
                seen = preprocess_frame(injected, frame.intensity, frame.ring,
                                        frame.timestamp, cfg.preprocess,
                                        tracker=shadow_ground).xyz
            else:
                seen = np.zeros((0, 3)) if forced else injected
            trace = (estimate_axis(seen, cfg.axis), estimate_axis(result.xyz, cfg.axis))
        if detector is not None:
            outcome = detector.process(injected, frame.intensity, frame.ring,
                                       frame.timestamp, frame.stamp_ns)
            body_axis = detector.axis_tracker.last
            state = outcome.debug["axis"]["state"]


            verdict.feed(state, bool(outcome.debug["candidates_raw"]["detections"]))
            if dump is not None:
                dump.write(json.dumps(_dump_row(frame, run_index, centre, geometry, outcome,
                                                args.body_tilt_deg),
                                      ensure_ascii=False) + "\n")
        else:
            body_axis = body_tracker.update(
                np.zeros((0, 3)) if forced else injected, advance_m=args.advance_m)
            outcome = (detect(injected, body_axis, cfg.gauge, cfg.detector)
                       if body_axis is not None else None)
            state = "lost" if body_axis is None else body_axis.state


            verdict.feed(state, bool(outcome and outcome.detections))
            if outcome is not None and confirmer is not None:
                outcome = confirmer.update(outcome, advance_m=args.advance_m or None)
        if centre is None:
            continue

        lo, hi = body_box(geometry, args.body_tilt_deg)
        corridor = (None if outcome is None
                    else outcome.debug.get("corridor", {}).get("x_m"))
        row_trace = {}
        if trace is not None:
            fresh, clean = trace

            row_trace = {
                "cross_check_m": None if fresh is None or fresh.cross_check_m is None
                else round(fresh.cross_check_m, 5),
                "clean_cross_check_m": None if clean is None or clean.cross_check_m is None
                else round(clean.cross_check_m, 5),
            }
        rows.append({
            **row_trace,
            "frame": int(frame.index),
            "distance_m": round(centre[0], 2),
            "axis_state": state,
            "corridor_m": corridor,
            "body_points": n_points,


            "points_stage": "raw" if detector is not None else "preprocessed",


            "advance_m": (args.advance_m if detector is None
                          else outcome.debug.get("advance_m")),
            "detected": bool(outcome is not None
                             and _overlaps(outcome.detections, lo, hi, args.tolerance)),
            "score": 0.0 if outcome is None else max(
                (d.score for d in outcome.detections
                 if _overlaps([d], lo, hi, args.tolerance)), default=0.0),
            "n_detections": 0 if outcome is None else len(outcome.detections),


            "matched_points": 0 if outcome is None else max(
                (d.n_points for d in outcome.detections
                 if _overlaps([d], lo, hi, args.tolerance)), default=0),
            "forced_failure": bool(forced),
            "hold_path_m": 0.0 if body_axis is None else round(body_axis.hold_path_m, 2),
            "run": run_index,


            "body_base_m": round(geometry[1], 3),
            "body_height_m": round(geometry[3], 3),
        })
    if dump is not None:
        dump.close()
    return rows, verdict


def summarise(rows: list[dict], verdict: VerdictCounter, args) -> None:
    if not rows:
        print("  ни одного кадра с телом")
        return
    seen = [r for r in rows if r["body_points"] > 0]
    hit = [r for r in rows if r["detected"]]
    states: dict[str, int] = {}
    for r in rows:
        states[r["axis_state"]] = states.get(r["axis_state"], 0) + 1
    d = np.array([r["distance_m"] for r in rows])
    print(f"  кадров с телом: {len(rows)}, дальность {d.min():.1f}…{d.max():.1f} м")
    print(f"  тело видно лучами: {len(seen)} кадров "
          f"(точек медиана {np.median([r['body_points'] for r in seen]):.0f})" if seen else
          "  тело лучами не видно")
    print(f"  ОБНАРУЖЕНО: {len(hit)}/{len(rows)} ({100 * len(hit) / len(rows):.0f} %)")
    print(f"  состояния оси: {states}")
    if hit:
        s = np.array([r["score"] for r in hit])
        print(f"  уверенность на теле: медиана {np.median(s):.2f}, min {s.min():.2f}, "
              f"p10 {np.percentile(s, 10):.2f}")
    print(f"  правило В: В₁ {verdict.v1}, В₂ {verdict.v2}")


    if seen and hit:
        delay = rows.index(hit[0]) - rows.index(seen[0])
        first = seen[0]["distance_m"]
        print(f"  задержка до первой детекции: {delay} кадров "
              f"(тело видно с {first:.1f} м)")
    runs = sorted({r["run"] for r in rows if r.get("run")})
    if len(runs) > 1:
        found, delays, first_seen = 0, [], []
        for index in runs:
            block = [r for r in rows if r["run"] == index]
            hits = [i for i, r in enumerate(block) if r["detected"]]
            if not hits:
                continue
            found += 1
            delays.append(hits[0])
            first_seen.append(block[hits[0]]["distance_m"])
        print(f"  заходов {len(runs)}, тело найдено в {found} "
              f"({100 * found / len(runs):.0f} %)")
        if delays:
            print(f"    задержка внутри захода: медиана {np.median(delays):.0f} кадров, "
                  f"max {max(delays)}; первая детекция на "
                  f"{np.median(first_seen):.1f} м (медиана)")
    forced = [r for r in rows if r.get("forced_failure")]
    if forced:
        hit = [r for r in forced if r["detected"]]
        hp = [r["hold_path_m"] for r in forced]
        print(f"  из них на ПРИНУДИТЕЛЬНОМ отказе трассера: {len(forced)} кадров, "
              f"обнаружено {len(hit)} ({100 * len(hit) / len(forced):.0f} %), "
              f"путь удержания до {max(hp):.1f} м")
        if hit:
            s2 = np.array([r["score"] for r in hit])
            print(f"    уверенность на удержании: медиана {np.median(s2):.2f}, min {s2.min():.2f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scenario", required=True,
                        choices=["appearance", "cold_start", "edge", "moving", "approach"])
    parser.add_argument("--record", default=None)
    parser.add_argument("--advance-m", type=float, default=0.0,
                        help="путь за кадр, м; 0 = состав стоит")
    parser.add_argument("--tolerance", type=float, default=0.6)
    parser.add_argument("--body-radius", type=float, default=BODY_RADIUS_M)
    parser.add_argument("--body-height", type=float, default=BODY_HEIGHT_M)
    parser.add_argument("--body-offset-m", type=float, default=0.0,
                        help="смещение тела от оси, м")
    parser.add_argument("--body-base-m", type=float, default=None,
                        help="высота основания тела над путевым основанием, м; "
                             "по умолчанию уровень головки рельса (0.23 м), "
                             "0.0 = предмет лежит на основании между рельсами")
    parser.add_argument("--body-tilt-deg", type=float, default=0.0,
                        help="наклон тела от вертикали в поперечной плоскости, градусы; "
                             "верх уходит влево (+y), низ стоит в точке размещения")
    parser.add_argument("--edge-penetration-m", type=float, default=0.05)
    parser.add_argument("--edge-height-m", type=float, default=1.0)
    parser.add_argument("--edge-radius-m", type=float, default=0.15)
    parser.add_argument("--edge-body-height-m", type=float, default=0.4)
    parser.add_argument("--distance-m", type=float, default=20.0)
    parser.add_argument("--base-margin-m", type=float, default=2.0)
    parser.add_argument("--first-frame", type=int, default=20)
    parser.add_argument("--last-frame", type=int, default=200)
    parser.add_argument("--fail-every", type=int, default=0,
                        help="период инъекции отказа трассера, кадров (0 = без отказов)")
    parser.add_argument("--fail-length", type=int, default=0,
                        help="длительность каждого отказа, кадров")
    parser.add_argument("--confirm", action="store_true", default=None,
                        help="включить подтверждение K из M поверх слоя A")
    parser.add_argument("--no-confirm", dest="confirm", action="store_false",
                        help="только слой A, без подтверждения")
    parser.add_argument("--via-pipeline", action="store_true",
                        help="вставлять тело в СЫРОЙ кадр и гнать его через "
                             "core.pipeline.ObstacleDetector целиком")
    parser.add_argument("--trace-anchor", action="store_true",
                        help="писать в строку сверку трассеров в кадре с телом "
                             "и в том же кадре без тела")
    parser.add_argument("--out", default="runs/scenarios")
    parser.add_argument("--placement-axis", choices=["clean", "corrected"], default="clean",
                        help="по какой оси ставить тело за стыком: сырая продлённая или "
                             "с поправкой сдвига и наклона полосы относительно рельсов")
    parser.add_argument("--detector-config", default=None,
                        help="конфиг ядра, если он отличается от конфига размещения тела")
    parser.add_argument("--dump-results", default=None,
                        help="JSONL: покадрово кандидаты ядра, путь и габарит тела — "
                             "для переигрывания подтверждения (tools.confirm_sweep)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = Config.from_yaml(args.config)
    if args.confirm is None:
        args.confirm = cfg.confirm.enabled
    rows, verdict = run(cfg, args)
    record = args.record or cfg.data.obstacle_record
    print(f"\n=== сценарий {args.scenario} на {record}")
    summarise(rows, verdict, args)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_pipeline" if args.via_pipeline else ""
    target = out_dir / f"{record}_{args.scenario}{suffix}.jsonl"
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  сохранено: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

