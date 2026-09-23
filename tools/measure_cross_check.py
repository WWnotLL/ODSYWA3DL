# Разложение расхождения двух трассеров оси: смещение опоры, разные базы, шум.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from core.axis import (_bed_centres, _rail_centres, _ridge, _trim_to_clearance,
                       estimate_axis)
from core.config import AxisConfig, Config
from core.preprocess import GroundTracker, preprocess_frame
from tools.bag_reader import iter_frames


NEIGHBOUR_WINDOW_M = (3.2, 5.2)


NEIGHBOUR_X_M = (5.0, 25.0)


def _slice_index(x: np.ndarray, cfg: AxisConfig) -> np.ndarray:
    return np.floor((x - cfg.x_min_m) / cfg.slice_m).astype(np.int64)


def _bed_edges(xyz: np.ndarray, cfg: AxisConfig) -> dict[int, tuple[float, float]]:
    band = xyz[(xyz[:, 2] > cfg.bed_z_min_m) & (xyz[:, 2] < cfg.bed_z_max_m)
               & (np.abs(xyz[:, 1]) < cfg.bed_lateral_limit_m)]
    edges: dict[int, tuple[float, float]] = {}
    if band.shape[0] == 0:
        return edges
    index = _slice_index(band[:, 0], cfg)
    for slot in np.unique(index):
        lateral = band[index == slot, 1]
        if lateral.size < cfg.bed_min_points:
            continue
        lo, hi = np.percentile(lateral, [cfg.bed_edge_percentile,
                                         100.0 - cfg.bed_edge_percentile])
        edges[int(slot)] = (float(lo), float(hi))
    return edges


def _gauge(xyz: np.ndarray, rails: np.ndarray, cfg: AxisConfig) -> float | None:
    head = xyz[(xyz[:, 2] > cfg.rails_z_min_m) & (xyz[:, 2] < cfg.rails_z_max_m)]
    slots = _slice_index(rails[:, 0], cfg)
    index = _slice_index(head[:, 0], cfg)
    gauges = []
    for slot, centre in zip(slots, rails[:, 1]):
        lateral = head[index == slot, 1]
        half, window = cfg.rails_half_gauge_m, cfg.rails_search_halfwidth_m
        left = _ridge(lateral, centre - half - window, centre - half + window, cfg)
        right = _ridge(lateral, centre + half - window, centre + half + window, cfg)
        if left is not None and right is not None:
            gauges.append(right - left)
    return float(np.median(gauges)) if gauges else None


def _radius(table: np.ndarray) -> float | None:
    if table.shape[0] < 4:
        return None
    a = float(np.polyfit(table[:, 0], table[:, 1], 2)[0])
    return None if abs(a) < 1e-9 else 1.0 / (2.0 * a)


def _neighbour(xyz: np.ndarray, rails_line: np.ndarray, cfg: AxisConfig) -> dict:
    head = xyz[(xyz[:, 2] > cfg.rails_z_min_m) & (xyz[:, 2] < cfg.rails_z_max_m)
               & (xyz[:, 0] >= NEIGHBOUR_X_M[0]) & (xyz[:, 0] <= NEIGHBOUR_X_M[1])]
    lateral = head[:, 1] - np.polyval(rails_line, head[:, 0])
    lo, hi = NEIGHBOUR_WINDOW_M
    return {
        "left": int(((lateral >= lo) & (lateral <= hi)).sum()),
        "right": int(((lateral <= -lo) & (lateral >= -hi)).sum()),
    }


def decompose(xyz: np.ndarray, cfg: AxisConfig) -> dict:
    rails, rails_reason = _rail_centres(xyz, cfg)
    bed, bed_reason = _bed_centres(xyz, cfg)
    estimate = estimate_axis(xyz, cfg)
    row: dict = {
        "rails_reason": rails_reason,
        "bed_reason": bed_reason,
        "method": None if estimate is None else estimate.method,
        "y0_m": None if estimate is None else round(estimate.y0_m, 4),
        "cross_check_m": (None if estimate is None or estimate.cross_check_m is None
                          else round(estimate.cross_check_m, 4)),
        "cross_check_reason": None if estimate is None else estimate.cross_check_reason,
        "meets_clearance": None if estimate is None else estimate.meets_clearance,
        "x_joint_m": None if estimate is None else estimate.x_joint_m,
    }
    if rails is None:
        return row

    kept, slope, y0, _ = _trim_to_clearance(rails, cfg)
    line = np.array([slope, y0])
    row["rails_x_m"] = [round(float(kept[0, 0]), 1), round(float(kept[-1, 0]), 1)]
    gauge = _gauge(xyz, rails, cfg)
    row["gauge_m"] = None if gauge is None else round(gauge, 4)
    row["neighbour_points"] = _neighbour(xyz, line, cfg)
    if bed is None:
        return row

    row["bed_x_m"] = [round(float(bed[0, 0]), 1), round(float(bed[-1, 0]), 1)]
    lo = max(float(kept[0, 0]), float(bed[0, 0]))
    hi = min(float(kept[-1, 0]), float(bed[-1, 0]))
    at = np.array([lo, hi])
    own = np.polyval(line, at)


    full = np.polyfit(bed[:, 0], bed[:, 1], 1)
    row["gap_as_coded_m"] = round(float(np.abs(np.polyval(full, at) - own).max()), 4)


    local = (bed[:, 0] >= lo) & (bed[:, 0] <= hi)
    if local.sum() >= 2:
        fit = np.polyfit(bed[local, 0], bed[local, 1], 1)
        row["gap_local_m"] = round(float(np.abs(np.polyval(fit, at) - own).max()), 4)

    rail_slots = _slice_index(rails[:, 0], cfg)
    bed_slots = _slice_index(bed[:, 0], cfg)
    common, rail_at, bed_at = np.intersect1d(rail_slots, bed_slots, return_indices=True)
    if common.size:
        diff = bed[bed_at, 1] - rails[rail_at, 1]


        row["slice_gap_by_x"] = [[round(float(x), 1), round(float(g), 4)]
                                 for x, g in zip(rails[rail_at, 0], diff)]
        row["slice_gap_m"] = {"median": round(float(np.median(diff)), 4),
                              "min": round(float(diff.min()), 4),
                              "max": round(float(diff.max()), 4),
                              "n": int(common.size)}
        edges = _bed_edges(xyz, cfg)
        left, right = [], []
        for slot, centre in zip(common, rails[rail_at, 1]):
            if int(slot) in edges:
                e_lo, e_hi = edges[int(slot)]
                left.append(e_hi - centre)
                right.append(centre - e_lo)
        if left:
            row["bed_edge_left_m"] = round(float(np.median(left)), 4)
            row["bed_edge_right_m"] = round(float(np.median(right)), 4)
    row["radius_bed_m"] = _radius(bed)
    row["radius_rails_m"] = _radius(rails)
    return row


def run(cfg: Config, record: str, first: int, last: int) -> list[dict]:
    ground = GroundTracker(cfg.preprocess.ground)
    rows = []
    for frame in iter_frames(cfg.data.path(record), cfg.bag, limit=last + 1):
        prepared = preprocess_frame(frame.xyz, frame.intensity, frame.ring,
                                    frame.timestamp, cfg.preprocess, tracker=ground).xyz
        if frame.index < first:
            continue
        row = decompose(prepared, cfg.axis)
        row["frame"] = int(frame.index)
        rows.append(row)
    return rows


def _stats(values: list[float]) -> str:
    v = np.asarray(values, dtype=float)
    return (f"медиана {np.median(v):+.3f}, p10 {np.percentile(v, 10):+.3f}, "
            f"p90 {np.percentile(v, 90):+.3f}  (n={v.size})")


def summarise(rows: list[dict], clearance: float) -> None:
    print("\n| кадр | метод | сверка ядра | сверка локально | слой к слою, медиана | "
          "край слева | край справа | колея | рельсы, м | полоса, м | R полосы, м | сосед L/R |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|---|---|---:|---|")
    for r in rows:
        sg = r.get("slice_gap_m")
        nb = r.get("neighbour_points")
        radius = r.get("radius_bed_m")
        slice_gap = "—" if sg is None else f"{sg['median']:+.3f}"
        neighbour = "—" if nb is None else f"{nb['left']}/{nb['right']}"
        curve = "—" if radius is None else f"{radius:+.0f}"
        print(f"| {r['frame']} | {r['method']} | {r.get('gap_as_coded_m', '—')} | "
              f"{r.get('gap_local_m', '—')} | {slice_gap} | "
              f"{r.get('bed_edge_left_m', '—')} | {r.get('bed_edge_right_m', '—')} | "
              f"{r.get('gauge_m', '—')} | {r.get('rails_x_m', '—')} | {r.get('bed_x_m', '—')} | {curve} | {neighbour} |")

    both = [r for r in rows if "gap_as_coded_m" in r]
    print(f"\nкадров всего {len(rows)}, с обеими таблицами {len(both)}")
    if not both:
        return
    coded = [r["gap_as_coded_m"] for r in both]
    local = [r["gap_local_m"] for r in both if "gap_local_m" in r]
    slices = [r["slice_gap_m"]["median"] for r in both if "slice_gap_m" in r]
    left = [r["bed_edge_left_m"] for r in both if "bed_edge_left_m" in r]
    right = [r["bed_edge_right_m"] for r in both if "bed_edge_right_m" in r]
    print(f"сверка как в ядре:        {_stats(coded)}")
    if local:
        print(f"сверка, соперник локально: {_stats(local)}")
    if slices:
        print(f"полоса − рельсы по слоям:  {_stats(slices)}   (+ значит полоса левее)")
    gauges = [r["gauge_m"] for r in rows if r.get("gauge_m") is not None]
    if gauges:
        print(f"колея по рельсовой опоре:  {_stats(gauges)}")
    if left:
        print(f"край полосы слева:         {_stats(left)}")
        print(f"край полосы справа:        {_stats(right)}")
    pairs = np.array([p for r in both for p in r.get("slice_gap_by_x", [])])
    if pairs.size:
        print("\nполоса − рельсы по дальности (все кадры с обеими таблицами):")
        edges = np.arange(5.0, pairs[:, 0].max() + 2.5, 2.5)
        slot = np.digitize(pairs[:, 0], edges) - 1
        for k in range(edges.size - 1):
            g = pairs[slot == k, 1]
            if g.size >= 5:
                print(f"  x {edges[k]:4.1f}…{edges[k + 1]:4.1f} м: медиана {1000 * np.median(g):+4.0f} мм, "
                      f"p10 {1000 * np.percentile(g, 10):+4.0f}, p90 {1000 * np.percentile(g, 90):+4.0f} "
                      f"(n={g.size})")
    print(f"ниже зазора {clearance:.3f}: как в ядре "
          f"{sum(c <= clearance for c in coded)}/{len(coded)}"
          + (f", локально {sum(c <= clearance for c in local)}/{len(local)}" if local else ""))


def _pct(values: list[float], unit: float = 1.0, fmt: str = "{:+.0f}") -> str:
    if not values:
        return "—"
    v = np.asarray(values, dtype=float) * unit
    return " / ".join(fmt.format(q) for q in np.percentile(v, [10, 50, 90]))


def summarise_runs(directory: Path, clearance: float) -> None:
    print("\n| запись | кадров | холодный старт, кадров | колея p10/p50/p90, мм | "
          "полоса − рельсы, все кадры, мм | то же в кадрах с Б′, мм | кадров с Б′ | "
          "сверка ≤ зазора |")
    print("|---|---:|---:|---|---|---|---:|---:|")
    for path in sorted(directory.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        gauges = [r["gauge_m"] for r in rows if r.get("gauge_m") is not None]
        gaps = [r["slice_gap_m"]["median"] for r in rows if r.get("slice_gap_m")]
        joined = [r["slice_gap_m"]["median"] for r in rows
                  if r.get("slice_gap_m") and r.get("x_joint_m") is not None]
        n_joined = sum(r.get("x_joint_m") is not None for r in rows)
        checked = [r["cross_check_m"] for r in rows if r.get("cross_check_m") is not None]


        cold = next((i for i, r in enumerate(rows) if r.get("method") is not None
                     and r["meets_clearance"]
                     and (r["cross_check_m"] is not None and r["cross_check_m"] <= clearance
                          or r["cross_check_m"] is None
                          and r["cross_check_reason"] == "unavailable_structural"
                          and r["method"] == "rails")), None)
        print(f"| `{path.stem}` | {len(rows)} | {cold if cold is not None else '—'} | "
              f"{_pct(gauges, 1000, '{:.0f}')} | {_pct(gaps, 1000)} | {_pct(joined, 1000)} | "
              f"{n_joined} | {sum(c <= clearance for c in checked)}/{len(checked)} |")
    print("\nКолонки p10 / p50 / p90. Знак «полоса − рельсы»: минус — полоса правее "
          "середины колеи. Холодный старт без учёта |y0| — он во всех кадрах в пределе.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--record")
    parser.add_argument("--frames", type=int, nargs=2, metavar=("FIRST", "LAST"))
    parser.add_argument("--summarise-runs", type=Path, default=None,
                        help="не считать заново, а свести готовые прогоны из каталога")
    parser.add_argument("--out", default="runs/cross_check")
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    if args.summarise_runs is not None:
        summarise_runs(args.summarise_runs, cfg.axis.clearance_m)
        return 0
    if args.record is None or args.frames is None:
        parser.error("нужны --record и --frames либо --summarise-runs")
    rows = run(cfg, args.record, args.frames[0], args.frames[1])
    summarise(rows, cfg.axis.clearance_m)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{args.record}.jsonl"
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nзаписано: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

