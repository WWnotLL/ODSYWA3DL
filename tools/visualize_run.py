# Визуализация работы ядра на записях: кадры PNG и gif по сюжетам из configs/viz.yaml.

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import yaml

from core.config import Config, GaugeConfig
from core.pipeline import ObstacleDetector
from core.preprocess import FrameTransform
from tools.bag_reader import iter_frames
from tools.false_alarm_table import count_episodes
from tools.scenarios import build_parser, iter_scenario_frames


VIZ_SCHEMA = {
    "core_config": None, "output_root": None, "check_against_runs": None,
    "drawing": {"point_stride": None, "point_size": None, "point_alpha": None,
                "fps": None, "dpi": None, "figure_size_in": None,
                "row_heights": None, "column_widths": None},
    "top_view": {"x_range_m": None, "y_range_m": None, "corridor_step_m": None},
    "section": {"window_m": None, "default_distance_m": None,
                "lateral_range_m": None, "height_range_m": None},
    "style": {"line_width": None, "thin_line_width": None, "zone_alpha": None,
              "box_alpha": None, "font_size": None, "legend_font_size": None,
              "title_font_size": None, "section_point_scale": None,
              "platform_hatch": None},
    "colors": None,
    "view3d": {"point_stride": None, "point_size": None, "window_size": None,
               "line_colors": None},
    "export": {"output_root": None, "frames_every": None, "moved_tolerance_m": None,
               "body_color": None},
    "stories": None,
}


TIMELINE_ROWS = ("ось", "кандидат", "тревога", "стоп В₂", "не проверен")


class VizConfigError(ValueError):
    pass


def _check(data: dict, schema: dict, path: str) -> None:
    missing = sorted(set(schema) - set(data))
    unknown = sorted(set(data) - set(schema))
    if missing or unknown:
        raise VizConfigError(f"{path}: нет ключей {missing}, лишние {unknown}")
    for key, sub in schema.items():
        if sub is not None:
            _check(data[key], sub, f"{path}.{key}")


def load_viz(path: Path) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    _check(data, VIZ_SCHEMA, str(path))
    return data


@dataclass
class FrameView:
    index: int
    stamp_ns: int
    result: dict
    points: np.ndarray


def record_source(cfg: Config, record: str, last: int) -> Iterator[tuple]:
    for f in iter_frames(cfg.data.path(record), cfg.bag, limit=last + 1):
        yield f.index, f.stamp_ns, f.xyz, f.intensity, f.ring, f.timestamp


def scenario_source(cfg: Config, argv: list[str], last: int) -> Iterator[tuple]:
    args = build_parser().parse_args(argv)
    for item in iter_scenario_frames(cfg, args, raw=True):
        f = item.frame
        yield f.index, f.stamp_ns, item.injected, f.intensity, f.ring, f.timestamp
        if f.index >= last:
            return


def collect(cfg: Config, source: Iterable[tuple], first: int, last: int,
            stride: int, region: tuple[tuple[float, float], tuple[float, float]]
            ) -> list[FrameView]:
    detector = ObstacleDetector(cfg)
    (x_lo, x_hi), (y_lo, y_hi) = region
    views: list[FrameView] = []
    for index, stamp_ns, xyz, intensity, ring, t_rel in source:
        result = detector.process(xyz, intensity, ring, t_rel, stamp_ns)
        if index < first:
            continue
        if index > last:
            break
        points = np.empty((0, 3))
        info = result.debug.get("frame_transform")
        if info is not None:
            transform = FrameTransform(np.asarray(info["rotation"]),
                                       np.asarray(info["translation_m"]))
            raw = np.asarray(xyz)
            raw = raw[np.linalg.norm(raw, axis=1) > cfg.preprocess.min_range_m][::stride]
            track = transform.apply(raw)
            keep = ((track[:, 0] >= x_lo) & (track[:, 0] <= x_hi)
                    & (track[:, 1] >= y_lo) & (track[:, 1] <= y_hi))
            points = track[keep].astype(np.float32)
        views.append(FrameView(int(index), int(stamp_ns), result.to_dict(), points))
    return views


def _axis_label(debug: dict) -> str:
    axis = debug.get("axis")
    if axis is None:
        return "не проверен"
    if axis["state"] == "lost" and axis.get("reason") == "anchor_uninitialised":
        return "lost: anchor_uninitialised"
    if axis["state"] == "lost":
        return f"lost: {axis.get('reason')}"
    return axis["state"]


def _has_candidate(debug: dict) -> bool:
    raw = debug.get("candidates_raw")
    return bool(raw and raw["detections"])


def summarise(views: list[FrameView]) -> dict:
    alarms = [v for v in views if v.result["obstacle_found"]]
    lost: dict[str, int] = {}
    for v in views:
        label = _axis_label(v.result["debug"])
        if label.startswith("lost") or label == "не проверен":
            lost[label] = lost.get(label, 0) + 1
    return {
        "frames": len(views),
        "range": [views[0].index, views[-1].index] if views else None,
        "with_candidate": sum(_has_candidate(v.result["debug"]) for v in views),
        "with_alarm": len(alarms),
        "episodes": count_episodes([v.stamp_ns for v in alarms]),
        "without_axis": lost,
        "unchecked": sum(not v.result["debug"].get("checked", False) for v in views),
    }


def check_against_run(views: list[FrameView], run_file: Path) -> list[str]:
    rows = {r["frame"]: r for r in map(json.loads, run_file.open(encoding="utf-8"))
            if "debug" in r}
    problems = []
    for v in views:
        row = rows.get(v.index)
        if row is None:
            problems.append(f"кадр {v.index}: нет в прогоне")
            continue
        mine = (v.result["obstacle_found"], _has_candidate(v.result["debug"]),
                v.result["debug"]["axis"]["state"])
        theirs = (row["obstacle_found"], bool(row.get("candidates_found")),
                  row["debug"]["axis"]["state"])
        if mine != theirs:
            problems.append(f"кадр {v.index}: визуализация {mine}, прогон {theirs}")
    return problems


def _polyline(debug: dict) -> tuple[np.ndarray, list[str]] | None:
    axis = debug.get("axis") or {}
    line = axis.get("polyline")
    if not line:
        return None
    return np.asarray(line["points_xy_m"], dtype=float), list(line["source"])


def _axis_y(line: np.ndarray, x: np.ndarray | float) -> np.ndarray:
    return np.interp(x, line[:, 0], line[:, 1])


def _band(ax, xs: np.ndarray, centre: np.ndarray, inner: float, outer: np.ndarray,
          color: str, label: str, alpha: float) -> None:
    for sign in (1.0, -1.0):
        ax.fill_between(xs, centre + sign * inner, centre + sign * outer,
                        color=color, alpha=alpha, linewidth=0,
                        label=label if sign > 0 else None)


def _draw_top(ax, view: FrameView, gauge: GaugeConfig, viz: dict) -> None:
    debug, colors, top = view.result["debug"], viz["colors"], viz["top_view"]
    d, st = viz["drawing"], viz["style"]
    if view.points.size:
        ax.scatter(view.points[:, 0], view.points[:, 1], s=d["point_size"],
                   c=colors["points"], alpha=d["point_alpha"], linewidths=0)
    poly = _polyline(debug)
    corridor = debug.get("corridor")
    railhead = gauge.rail_head_offset_m or 0.0
    if poly is not None:
        line, source = poly
        for kind, style in (("rails", "-"), ("bed", "--")):
            mask = np.array([s == kind for s in source])
            if mask.any():
                ax.plot(line[mask, 0], line[mask, 1], style, lw=st["line_width"],
                        color=colors[f"axis_{kind}"], label=f"ось: {kind}")
    if poly is not None and corridor is not None:
        line, _ = poly
        x0, x1 = corridor["x_m"]
        xs = np.arange(x0, x1 + 1e-9, top["corridor_step_m"])
        centre = _axis_y(line, xs)
        floor = corridor["z_m"][0] - railhead
        half = float(gauge.half_width_at(np.array([floor]))[0])
        for sign in (1.0, -1.0):
            ax.plot(xs, centre + sign * half, color=colors["corridor"], lw=st["thin_line_width"],
                    label="Ом у пола" if sign > 0 else None)
        ax.axvline(x1, color=colors["corridor"], lw=st["thin_line_width"], ls=":")
        outer = np.full_like(xs, min(gauge.contact_rail_half_width_max_m, half))
        _band(ax, xs, centre, gauge.contact_rail_half_width_min_m, outer,
              colors["cut_contact_rail"], "вырез: контактный рельс", st["zone_alpha"])
        cut = (debug.get("platform") or {}).get("cut", [False, False])
        for sign, active in ((1.0, cut[0]), (-1.0, cut[1])):
            if active:
                ax.fill_between(xs, centre + sign * gauge.platform_lateral_from_m,
                                centre + sign * half, facecolor="none",
                                edgecolor=colors["cut_platform"], hatch=st["platform_hatch"],
                                linewidth=0, label="вырез: платформа")
    for det in (debug.get("candidates_raw") or {}).get("detections", []):
        lo, hi = det["bbox_min_xyz"], det["bbox_max_xyz"]
        ax.add_patch(_rect(lo[0], lo[1], hi[0] - lo[0], hi[1] - lo[1],
                           edge=colors["candidate"], fill=None, style=st))
    for det in view.result["detections"]:
        lo, hi = det["bbox_min_xyz"], det["bbox_max_xyz"]
        ax.add_patch(_rect(lo[0], lo[1], hi[0] - lo[0], hi[1] - lo[1],
                           edge=colors["alarm"], fill=colors["alarm"], style=st))
        ax.annotate(f"{det['distance_m']:.1f} м", (hi[0], hi[1]),
                    color=colors["alarm"], fontsize=st["font_size"], weight="bold")
    ax.set_xlim(*top["x_range_m"])
    ax.set_ylim(*top["y_range_m"])
    ax.set_xlabel("вдоль пути, м")
    ax.set_ylabel("поперёк, м (+ влево)")
    ax.set_aspect("auto")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=st["legend_font_size"])


def _rect(x: float, y: float, w: float, h: float, *, edge: str, fill: str | None,
          style: dict):
    from matplotlib.patches import Rectangle
    return Rectangle((x, y), w, h, facecolor=fill or "none", edgecolor=edge,
                     alpha=style["box_alpha"] if fill else 1.0, lw=style["line_width"])


def _draw_section(ax, view: FrameView, gauge: GaugeConfig, viz: dict) -> None:
    debug, colors, sec = view.result["debug"], viz["colors"], viz["section"]
    st = viz["style"]
    railhead = gauge.rail_head_offset_m or 0.0
    candidates = (debug.get("candidates_raw") or {}).get("detections", [])
    x_c = (min(candidates, key=lambda d: d["distance_m"])["centroid_xyz"][0]
           if candidates else sec["default_distance_m"])
    poly = _polyline(debug)
    y_c = float(_axis_y(poly[0], x_c)) if poly is not None else 0.0
    if view.points.size:
        near = np.abs(view.points[:, 0] - x_c) <= sec["window_m"]
        ax.scatter(view.points[near, 1] - y_c, view.points[near, 2] - railhead,
                   s=viz["drawing"]["point_size"] * st["section_point_scale"],
                   c=colors["points"], linewidths=0)
    h, w = gauge.profile_height_m, gauge.profile_half_width_m
    ax.plot(np.concatenate([w, -w[::-1], [w[0]]]), np.concatenate([h, h[::-1], [h[0]]]),
            color=colors["corridor"], lw=st["line_width"], label="Ом")
    from matplotlib.patches import Rectangle
    for sign in (1.0, -1.0):
        x0 = sign * gauge.contact_rail_half_width_min_m
        width = sign * (gauge.contact_rail_half_width_max_m - gauge.contact_rail_half_width_min_m)
        ax.add_patch(Rectangle((min(x0, x0 + width), h[0]), abs(width),
                               gauge.contact_rail_height_max_m - h[0],
                               color=colors["cut_contact_rail"], alpha=st["zone_alpha"], lw=0))
    cut = (debug.get("platform") or {}).get("cut", [False, False])
    for sign, active in ((1.0, cut[0]), (-1.0, cut[1])):
        if active:
            inner = gauge.platform_lateral_from_m
            outer = float(w.max())
            ax.add_patch(Rectangle((min(sign * inner, sign * outer), h[0]), outer - inner,
                                   gauge.platform_height_max_m - h[0], facecolor="none",
                                   edgecolor=colors["cut_platform"],
                                   hatch=st["platform_hatch"], lw=0))
    for det, filled in ([(d, False) for d in candidates]
                        + [(d, True) for d in view.result["detections"]]):
        lo, hi = det["bbox_min_xyz"], det["bbox_max_xyz"]
        if hi[0] < x_c - sec["window_m"] or lo[0] > x_c + sec["window_m"]:
            continue
        ax.add_patch(_rect(lo[1] - y_c, lo[2] - railhead, hi[1] - lo[1], hi[2] - lo[2],
                           edge=colors["alarm"], fill=colors["alarm"] if filled else None,
                           style=st))
    ax.set_xlim(*sec["lateral_range_m"])
    ax.set_ylim(*sec["height_range_m"])
    ax.set_xlabel("от оси, м (+ влево)")
    ax.set_ylabel("над УГР, м")
    ax.set_title(f"сечение x = {x_c:.1f} ± {sec['window_m']:.1f} м", fontsize=st["font_size"])
    ax.set_aspect("equal")


def _timeline_colors(views: list[FrameView], colors: dict) -> np.ndarray:
    from matplotlib.colors import to_rgb
    white = (1.0, 1.0, 1.0)
    grid = np.ones((len(TIMELINE_ROWS), len(views), 3))
    for column, v in enumerate(views):
        debug = v.result["debug"]
        label = _axis_label(debug)
        if label == "не проверен":
            state = colors["unchecked"]
        elif label == "lost: anchor_uninitialised":
            state = colors["state_anchor_uninitialised"]
        else:
            state = colors[f"state_{label.split(':')[0]}"]
        cells = (
            state,
            colors["candidate"] if _has_candidate(debug) else None,
            colors["alarm"] if v.result["obstacle_found"] else None,
            colors["verdict_stop"] if (debug.get("verdict") or {}).get("stop") else None,
            colors["unchecked"] if not debug.get("checked", False) else None,
        )
        for row, color in enumerate(cells):
            grid[row, column] = to_rgb(color) if color else white
    return grid


def _draw_timeline(ax, views: list[FrameView], current: int, grid: np.ndarray,
                   colors: dict, style: dict) -> None:
    ax.imshow(grid, aspect="auto", interpolation="nearest",
              extent=(views[0].index - 0.5, views[-1].index + 0.5, len(TIMELINE_ROWS), 0))
    ax.set_yticks(np.arange(len(TIMELINE_ROWS)) + 0.5, TIMELINE_ROWS,
                  fontsize=style["font_size"])
    ax.axvline(current, color=colors["cursor"], lw=style["line_width"])
    ax.set_xlabel("кадр")


def render(views: list[FrameView], gauge: GaugeConfig, viz: dict, out_dir: Path,
           title: str) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = viz["drawing"]
    grid = _timeline_colors(views, viz["colors"])
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for view in views:
        fig = plt.figure(figsize=tuple(d["figure_size_in"]), dpi=d["dpi"])
        spec = fig.add_gridspec(2, 2, height_ratios=d["row_heights"],
                                width_ratios=d["column_widths"])
        _draw_top(fig.add_subplot(spec[0, 0]), view, gauge, viz)
        _draw_section(fig.add_subplot(spec[0, 1]), view, gauge, viz)
        _draw_timeline(fig.add_subplot(spec[1, :]), views, view.index, grid,
                       viz["colors"], viz["style"])
        debug = view.result["debug"]
        verdict = (debug.get("verdict") or {})
        nearest = view.result["nearest_distance_m"]
        total = (debug.get("timings_ms") or {}).get("total")
        fig.suptitle(
            f"{title} · кадр {view.index} · ось: {_axis_label(debug)} · "
            f"В₂: {'СТОП' if verdict.get('stop') else 'нет'} · "
            f"ближайшая тревога: {'—' if nearest is None else f'{nearest:.2f} м'} · "
            f"обработка {'—' if total is None else f'{total:.0f} мс'}",
            fontsize=viz["style"]["title_font_size"])
        fig.tight_layout()
        path = out_dir / f"frame_{view.index:04d}.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def write_gif(paths: list[Path], target: Path, fps: float) -> None:
    from PIL import Image
    frames = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in paths]
    frames[0].save(target, save_all=True, append_images=frames[1:],
                   duration=int(round(1000.0 / fps)), loop=0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--viz", type=Path, required=True)
    parser.add_argument("--story", required=True)
    parser.add_argument("--point-stride", type=int, default=None,
                        help="перекрыть шаг прореживания точек (проверка: сводка не меняется)")
    parser.add_argument("--no-render", action="store_true",
                        help="только сводка и сверка, без картинок")
    args = parser.parse_args(argv)

    viz = load_viz(args.viz)
    story = viz["stories"][args.story]
    cfg = Config.from_yaml(viz["core_config"])
    first, last = story["frames"]
    if "scenario_argv" in story:
        source = scenario_source(cfg, story["scenario_argv"], last)
        name = "approach_" + story["scenario_argv"][story["scenario_argv"].index("--record") + 1]
        record = None
    else:
        record = story["record"]
        source = record_source(cfg, record, last)
        name = record
    stride = args.point_stride or viz["drawing"]["point_stride"]
    top = viz["top_view"]
    region = (tuple(top["x_range_m"]), tuple(top["y_range_m"]))
    views = collect(cfg, source, first, last, stride, region)

    summary = summarise(views)
    if record is None:
        del summary["episodes"]
    print(f"\n=== сюжет {args.story} ({name}), кадры {first}–{last}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if record is not None:
        run_file = Path(viz["check_against_runs"]) / f"{record}.jsonl"
        problems = check_against_run(views, run_file)
        print(f"сверка с {run_file}: расхождений {len(problems)}")
        for line in problems[:20]:
            print("  " + line)
        summary["check_against_run"] = {"file": str(run_file), "mismatches": len(problems)}

    out_dir = Path(viz["output_root"]) / name / args.story
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    if args.no_render:
        return 0
    gauge = cfg.gauge
    paths = render(views, gauge, viz, out_dir / "frames", f"{name} [{args.story}]")
    gif = out_dir / f"{args.story}.gif"
    write_gif(paths, gif, viz["drawing"]["fps"])
    print(f"кадров: {len(paths)} → {out_dir / 'frames'}; gif: {gif}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

