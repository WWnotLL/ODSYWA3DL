# Глубина захода кластеров за кромку Ом (4-я глубочайшая точка) от оси ядра, от рельсов и от поправленной оси — ложные против позитивов, по зонам «рельсы» и «Б′».

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import core.pipeline as pipeline
from core.axis import _bed_centres, _rail_centres, _trim_to_clearance
from core.config import Config
from core.detector import _cluster, cluster_depth, gauge_bounds, platform_runs
from tools.bag_reader import iter_frames
from tools.episode_table import group_by_gap
from tools.measure_extension_residual import _correction, _raw_far
from tools.positive_control import _overlaps
from tools.scenarios import build_parser, iter_scenario_frames


_captured: dict = {}
_original_detect = pipeline.detect


def _capturing_detect(xyz, axis, gauge, cfg, **kwargs):
    _captured["xyz"], _captured["axis"] = xyz, axis
    return _original_detect(xyz, axis, gauge, cfg, **kwargs)


_RANK: list[int] = []


def _kth_deepest(depth: np.ndarray) -> float | None:
    if depth.size < _RANK[0]:
        return None
    return cluster_depth(depth, _RANK[0])


def clusters(xyz: np.ndarray, axis, cfg: Config) -> list[np.ndarray]:
    gauge, det = cfg.gauge, cfg.detector
    far = min(axis.x_end_m, det.max_range_m) if det.max_range_m is not None else axis.x_end_m
    near = max(det.min_range_m, axis.x_start_m - 0.5 * det.range_start_slack_m)
    floor, ceiling = gauge_bounds(gauge, det)
    offset = axis.offset(xyz)
    above = xyz[:, 2] - (gauge.rail_head_offset_m or 0.0)
    inside = ((xyz[:, 0] >= near) & (xyz[:, 0] <= far)
              & (np.abs(offset) < gauge.half_width_at(above))
              & (xyz[:, 2] > floor) & (xyz[:, 2] < ceiling))
    inside &= ~gauge.in_contact_rail_zone(offset, above)
    left_run, right_run = platform_runs(offset, above, xyz[:, 0], near, far, gauge)
    required = gauge.platform_slices_required
    inside &= ~(gauge.in_platform_zone(offset, above)
                & np.where(offset > 0.0, left_run >= required, right_run >= required))
    inside &= ~gauge.in_rail_mask(offset, above)
    if inside.sum() < det.cluster_min_points:
        return []
    index = np.flatnonzero(inside)
    points = xyz[inside]
    labels = _cluster(points, np.linalg.norm(points, axis=1), det)
    order = np.argsort(labels, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(labels[order])) + 1)
    return [index[g] for g in groups if g.size >= det.cluster_min_points]


def measure(xyz: np.ndarray, axis, candidates: list[dict], cfg: Config) -> list[dict]:
    gauge = cfg.gauge
    railhead = gauge.rail_head_offset_m or 0.0
    rails, _ = _rail_centres(xyz, cfg.axis)
    rails_line = None
    if rails is not None:
        kept, r_slope, r_y0, _ = _trim_to_clearance(rails, cfg.axis)
        half = 0.5 * cfg.axis.slice_m
        rails_line = (r_slope, r_y0, float(kept[0, 0]) - half, float(kept[-1, 0]) + half)
    extended = axis.x_joint_m is not None and axis.far_method == "bed"
    fix = None
    if extended and rails is not None:
        bed, _ = _bed_centres(xyz, cfg.axis)
        if bed is not None:
            fix = _correction(rails, bed, cfg.axis.x_min_m, cfg.axis.slice_m, cfg.axis.min_slices)
    groups = {}
    for g in clusters(xyz, axis, cfg):
        block = xyz[g]
        lo, hi = block.min(axis=0).astype(np.float64), block.max(axis=0).astype(np.float64)
        groups[(tuple(np.round(lo, 5)), tuple(np.round(hi, 5)))] = g
    rows = []
    for cand in candidates:
        key = (tuple(np.round(np.asarray(cand["bbox_min_xyz"], np.float64), 5)),
               tuple(np.round(np.asarray(cand["bbox_max_xyz"], np.float64), 5)))
        g = groups.get(key)
        if g is None:
            rows.append({"matched": False, "bbox_min_xyz": cand["bbox_min_xyz"],
                         "bbox_max_xyz": cand["bbox_max_xyz"]})
            continue
        block = xyz[g]
        edge = gauge.half_width_at(block[:, 2] - railhead)
        depth_core = edge - np.abs(axis.offset(block))
        centre_x = float(block[:, 0].mean())
        zone = "Б′" if extended and centre_x > axis.x_joint_m else "рельсы"
        row = {"matched": True, "zone": zone, "x_near_m": round(float(block[:, 0].min()), 2),
               "x_centre_m": round(centre_x, 2),
               "lateral_m": round(float(axis.offset(block).mean()), 3),
               "z_m": [round(float(block[:, 2].min()), 2), round(float(block[:, 2].max()), 2)],
               "n_points": int(g.size),
               "depth_k4_core_m": _kth_deepest(depth_core),
               "depth_max_core_m": round(float(depth_core.max()), 4),
               "depth_k4_rails_m": None, "depth_k4_fixed_m": None,
               "bbox_min_xyz": cand["bbox_min_xyz"], "bbox_max_xyz": cand["bbox_max_xyz"]}
        if rails_line is not None:
            slope, y0, lo, hi = rails_line
            covered = (block[:, 0] >= lo) & (block[:, 0] <= hi)
            lat = block[covered, 1] - (slope * block[covered, 0] + y0)
            row["depth_k4_rails_m"] = _kth_deepest(edge[covered] - np.abs(lat))
        if zone == "Б′" and fix is not None:
            c_slope, c_shift = fix[0], fix[1]
            raw_slope, raw_y0 = _raw_far(axis)
            line = np.where(block[:, 0] > axis.x_joint_m,
                            raw_slope * block[:, 0] + raw_y0
                            - (c_slope * block[:, 0] + c_shift),
                            axis.y_at(block[:, 0]))
            row["depth_k4_fixed_m"] = _kth_deepest(edge - np.abs(block[:, 1] - line))
        rows.append(row)
    return rows


def _frame_rows(outcome, frame_index: int, stamp_ns: int, cfg: Config) -> list[dict]:
    debug = outcome.debug
    raw = (debug.get("candidates_raw") or {}).get("detections", [])
    if not raw or _captured.get("axis") is None:
        return []
    confirmed = [(d.bbox_min_xyz, d.bbox_max_xyz) for d in outcome.detections]
    rows = measure(_captured["xyz"], _captured["axis"], raw, cfg)
    for row in rows:
        row.update({"frame": frame_index, "stamp_ns": stamp_ns,
                    "axis_state": debug["axis"]["state"],
                    "confirmed": any(np.allclose(row["bbox_min_xyz"], lo)
                                     and np.allclose(row["bbox_max_xyz"], hi)
                                     for lo, hi in confirmed)})
    return rows


def run_record(cfg: Config, record: str) -> list[dict]:
    detector = pipeline.ObstacleDetector(cfg)
    rows = []
    for f in iter_frames(cfg.data.path(record), cfg.bag):
        _captured.clear()
        outcome = detector.process(f.xyz, f.intensity, f.ring, f.timestamp, f.stamp_ns)
        for row in _frame_rows(outcome, f.index, f.stamp_ns, cfg):
            rows.append({"source": record, **row})
        print(f"\r{record}: кадр {f.index}", end="", flush=True)
    print()
    return rows


def run_scenario(cfg: Config, name: str, argv: list[str]) -> list[dict]:
    args = build_parser().parse_args(argv)
    detector = pipeline.ObstacleDetector(cfg)
    rows = []
    for item in iter_scenario_frames(cfg, args, raw=True):
        f = item.frame
        _captured.clear()
        outcome = detector.process(item.injected, f.intensity, f.ring, f.timestamp, f.stamp_ns)
        box = None
        if item.centre is not None:
            (cx, cy), z_bottom, radius, height = item.centre, *item.geometry[1:]
            box = (np.array([cx - radius, cy - radius, z_bottom]),
                   np.array([cx + radius, cy + radius, z_bottom + height]))
        for row in _frame_rows(outcome, f.index, f.stamp_ns, cfg):
            on_body = False
            if box is not None:
                d = argparse.Namespace(bbox_min_xyz=row["bbox_min_xyz"], bbox_max_xyz=row["bbox_max_xyz"])
                on_body = _overlaps([d], box[0], box[1], args.tolerance)
            rows.append({"source": name, "on_body": on_body, "run": item.run_index,
                         "body_x_m": None if box is None else round(float(item.centre[0]), 2), **row})
        print(f"\r{name}: кадр {f.index}", end="", flush=True)
    print()
    return rows


def _episodes(rows: list[dict]) -> list[list[dict]]:
    return group_by_gap([r for r in rows if r.get("confirmed") and r["matched"]])


def _mm(values: list[float | None]) -> list[float]:
    return [1000 * v for v in values if v is not None]


def _stats(values: list[float]) -> str:
    if not values:
        return "—"
    v = np.asarray(values)
    return (f"{len(v)} / {v.min():+.0f} / {np.percentile(v, 5):+.0f} / {np.median(v):+.0f} "
            f"/ {v.max():+.0f}")


def summarise(paths: list[Path]) -> None:
    rows = [json.loads(line) for p in paths for line in p.read_text(encoding="utf-8").splitlines()]
    unmatched = sum(1 for r in rows if not r["matched"])
    print(f"кластеров {len(rows)}, не сопоставлено с рамкой ядра: {unmatched}")
    rows = [r for r in rows if r["matched"]]
    false = [r for r in rows if "on_body" not in r]
    pos = [r for r in rows if r.get("on_body")]
    train = [r for r in rows if "on_body" in r and not r["on_body"] and r["axis_state"] == "held"]

    print("\n### Ложные по эпизодам (тревоги), глубина 4-й точки, мм")
    print("| источник | эпизод, кадры | зона | кластеров | от оси ядра: max / медиана "
          "| от рельсов: max / медиана | от поправленной оси: max / медиана |")
    print("|---|---|---|---:|---|---|---|")
    by_source: dict[str, list[dict]] = {}
    for r in false:
        by_source.setdefault(r["source"], []).append(r)
    episode_max: dict[str, list[float]] = {"рельсы": [], "Б′": []}
    for source, group in sorted(by_source.items()):
        for ep in _episodes(group):
            zones = sorted({r["zone"] for r in ep})
            for zone in zones:
                part = [r for r in ep if r["zone"] == zone]
                core_v = _mm([r["depth_k4_core_m"] for r in part])
                rails_v = _mm([r["depth_k4_rails_m"] for r in part])
                fix_v = _mm([r["depth_k4_fixed_m"] for r in part])
                fmt = lambda v: "—" if not v else f"{max(v):+.0f} / {np.median(v):+.0f}"
                print(f"| `{source}` | {part[0]['frame']}–{part[-1]['frame']} | {zone} | {len(part)} "
                      f"| {fmt(core_v)} | {fmt(rails_v)} | {fmt(fix_v)} |")
                if core_v:
                    episode_max[zone].append(max(core_v))

    print("\n### Все кандидаты ложной стороны (до подтверждения), глубина 4-й точки от оси ядра, мм")
    print("| зона | подтверждён | n / min / p5 / медиана / max |")
    print("|---|---|---|")
    for zone in ("рельсы", "Б′"):
        for flag in (True, False):
            part = [r for r in false if r["zone"] == zone and r["confirmed"] == flag]
            print(f"| {zone} | {'да' if flag else 'нет'} | {_stats(_mm([r['depth_k4_core_m'] for r in part]))} |")

    print("\n### Кластер, привязанный к поезду (кадры held, мимо тела), мм")
    print("| источник | кадр | x | y | от оси ядра | от рельсов |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in train:
        rails = "—" if r["depth_k4_rails_m"] is None else f"{1000 * r['depth_k4_rails_m']:+.0f}"
        print(f"| `{r['source']}` | {r['frame']} | {r['x_near_m']} | {r['lateral_m']:+.2f} "
              f"| {1000 * r['depth_k4_core_m']:+.0f} | {rails} |")

    print("\n### Позитивы: кадры, где кластер на теле, глубина 4-й точки, мм (n / min / p5 / медиана / max)")
    print("| тело | зона | от оси ядра | от рельсов | от поправленной оси |")
    print("|---|---|---|---|---|")
    pos_min: dict[str, list[float]] = {"рельсы": [], "Б′": []}
    for source in sorted({r["source"] for r in pos}):
        for zone in ("рельсы", "Б′"):
            part = [r for r in pos if r["source"] == source and r["zone"] == zone]
            if not part:
                continue
            core_v = _mm([r["depth_k4_core_m"] for r in part])
            pos_min[zone] += core_v
            print(f"| `{source}` | {zone} | {_stats(core_v)} "
                  f"| {_stats(_mm([r['depth_k4_rails_m'] for r in part]))} "
                  f"| {_stats(_mm([r['depth_k4_fixed_m'] for r in part]))} |")

    print("\n### Пороги: эпизод снят, если все его тревоги не глубже порога; позитив потерян, "
          "если кластер на теле не глубже порога (оценка по готовым кластерам, без перепрогона "
          "подтверждения)")
    print("| порог, мм | глубина от | эпизодов снято из | кадров позитива потеряно: рельсы | Б′ |")
    print("|---:|---|---|---|---|")
    episodes_all = [ep for source, group in sorted(by_source.items()) for ep in _episodes(group)]
    for threshold in (31, 45, 62, 83):
        for metric in ("core", "fixed"):
            key = "depth_k4_core_m"

            def depth(r: dict) -> float | None:
                if metric == "fixed" and r["zone"] == "Б′":
                    return r["depth_k4_fixed_m"]
                return r[key]

            removed = sum(all((depth(r) is not None and 1000 * depth(r) <= threshold) for r in ep)
                          for ep in episodes_all)
            lost = {}
            for zone in ("рельсы", "Б′"):
                vals = [1000 * depth(r) for r in pos if r["zone"] == zone and depth(r) is not None]
                lost[zone] = f"{sum(v <= threshold for v in vals)} из {len(vals)}"
            label = "оси ядра" if metric == "core" else "поправленной оси в Б′"
            print(f"| {threshold} | {label} | {removed} из {len(episodes_all)} | {lost['рельсы']} | {lost['Б′']} |")

    print("\n### Разделение по зонам (от оси ядра)")
    for zone in ("рельсы", "Б′"):
        f_max = max(episode_max[zone]) if episode_max[zone] else None
        p = np.asarray(pos_min[zone]) if pos_min[zone] else None
        if f_max is None or p is None:
            print(f"{zone}: нет данных с одной из сторон")
            continue
        below = int((p <= f_max).sum())
        verdict = (f"порог в интервале ({f_max:+.0f}; {p.min():+.0f}) мм разделяет"
                   if p.min() > f_max else f"нет разделения: {below} из {p.size} кадров позитива "
                   f"не глубже самого глубокого эпизода ложных ({f_max:+.0f} мм)")
        print(f"{zone}: эпизоды ложных — max {f_max:+.0f} мм; позитив — min {p.min():+.0f}, "
              f"p5 {np.percentile(p, 5):+.0f} мм; {verdict}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    rec = sub.add_parser("records")
    rec.add_argument("--config", type=Path, required=True)
    rec.add_argument("--records", nargs="+", required=True)
    rec.add_argument("--out", type=Path, required=True)
    sce = sub.add_parser("scenario")
    sce.add_argument("--config", type=Path, required=True)
    sce.add_argument("--name", required=True)
    sce.add_argument("--out", type=Path, required=True)
    sce.add_argument("scenario_argv", nargs=argparse.REMAINDER)
    summ = sub.add_parser("summary")
    summ.add_argument("files", type=Path, nargs="+")
    args = parser.parse_args(argv)

    if args.mode == "summary":
        summarise(args.files)
        return 0
    pipeline.detect = _capturing_detect
    cfg = Config.from_yaml(args.config)
    _RANK[:] = [cfg.detector.depth_rank]
    if args.mode == "records":
        rows = [row for record in args.records for row in run_record(cfg, record)]
    else:
        argv_s = [a for a in args.scenario_argv if a != "--"]
        rows = run_scenario(cfg, args.name, argv_s)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    unmatched = sum(1 for r in rows if not r["matched"])
    print(f"кластеров {len(rows)}, не сопоставлено с рамкой ядра: {unmatched}; записано {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
