# Ошибка продлённой оси против рельсов по стенам с двух сторон — до и после поправки.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from types import SimpleNamespace

from core.axis import (AxisTracker, _bed_centres, _rail_centres, _trim_to_clearance,
                       extension_correction)
from core.config import Config
from core.odometry import estimate_shift, longitudinal_profile
from core.preprocess import GroundTracker, preprocess_frame
from tools.bag_reader import iter_frames
from tools.measure_wall import FACE_RANK, SLICE_M, _kth


FACE_BAND_M = 0.05


def _faces(values: np.ndarray) -> tuple[float, float] | None:
    kth = _kth(values, FACE_RANK)
    if kth is None:
        return None
    band = values[(values >= kth) & (values <= kth + FACE_BAND_M)]
    return kth, float(np.median(band))


def _correction(rails: np.ndarray, bed: np.ndarray, x_min: float, slice_m: float,
                min_slices: int) -> tuple[float, float, float, float] | None:
    cfg = SimpleNamespace(x_min_m=x_min, slice_m=slice_m, min_slices=min_slices)
    return extension_correction(rails, bed, cfg)


def _raw_far(axis) -> tuple[float, float]:
    if axis.far_correction_slope is None:
        return axis.far_slope, axis.far_y0_m
    return (axis.far_slope + axis.far_correction_slope,
            axis.far_y0_m + axis.far_correction_shift_m)


def run(cfg: Config, record: str, lateral: tuple[float, float], height: tuple[float, float],
        far: tuple[float, float], near_min_m: float) -> tuple[list[dict], int]:
    ground = GroundTracker(cfg.preprocess.ground)
    tracker = AxisTracker(cfg.axis)
    railhead = cfg.gauge.rail_head_offset_m or 0.0
    previous_profile, previous_stamp = None, None
    path_m, last_advance, substituted = 0.0, None, 0
    rows: list[dict] = []
    for frame in iter_frames(cfg.data.path(record), cfg.bag):
        prepared = preprocess_frame(frame.xyz, frame.intensity, frame.ring,
                                    frame.timestamp, cfg.preprocess, tracker=ground)
        xyz = prepared.xyz
        profile = longitudinal_profile(xyz, prepared.intensity, cfg.odometry)
        advance = None
        if previous_profile is not None:
            dt_s = (frame.stamp_ns - previous_stamp) / 1e9
            shift = estimate_shift(previous_profile, profile, dt_s, cfg.odometry)
            advance = None if shift.shift_m is None else abs(shift.shift_m)
        previous_profile, previous_stamp = profile, frame.stamp_ns
        axis = tracker.update(xyz, advance_m=advance)


        if advance is None and frame.index > 0:
            advance = last_advance or 0.0
            substituted += 1
        path_m += advance or 0.0
        last_advance = advance if advance else last_advance

        rails, _ = _rail_centres(xyz, cfg.axis)
        if rails is None:
            continue
        kept, r_slope, r_y0, _ = _trim_to_clearance(rails, cfg.axis)
        rails_end = float(kept[-1, 0]) + 0.5 * cfg.axis.slice_m
        h = xyz[:, 2] - railhead
        in_height = (h >= height[0]) & (h <= height[1])


        for start in np.arange(near_min_m, rails_end - SLICE_M, SLICE_M):
            in_slice = in_height & (xyz[:, 0] >= start) & (xyz[:, 0] < start + SLICE_M)
            offset = xyz[in_slice, 1] - (r_slope * xyz[in_slice, 0] + r_y0)
            for sign, side in ((1.0, "left"), (-1.0, "right")):
                values = sign * offset
                faces = _faces(values[(values >= lateral[0]) & (values <= lateral[1])])
                if faces is None:
                    continue
                centre = start + 0.5 * SLICE_M
                rows.append({"kind": "near", "frame": frame.index, "side": side,
                             "x_m": round(centre, 1), "world_m": round(path_m + centre, 1),
                             "face_kth_m": round(faces[0], 4), "face_med_m": round(faces[1], 4)})


        if (axis is None or axis.source != "measured" or axis.x_joint_m is None
                or axis.far_method != "bed"):
            continue
        bed, _ = _bed_centres(xyz, cfg.axis)
        if bed is None:
            continue
        fix = _correction(rails, bed, cfg.axis.x_min_m, cfg.axis.slice_m, cfg.axis.min_slices)
        if fix is None:
            continue
        c_slope, c_shift, c_from, c_to = fix
        lo = max(far[0], axis.x_joint_m)
        for start in np.arange(lo, min(far[1], axis.x_end_m) - SLICE_M + 1e-6, SLICE_M):
            in_slice = in_height & (xyz[:, 0] >= start) & (xyz[:, 0] < start + SLICE_M)
            x, y = xyz[in_slice, 0], xyz[in_slice, 1]
            raw_slope, raw_y0 = _raw_far(axis)
            raw_axis = raw_slope * x + raw_y0
            fixed_axis = raw_axis - (c_slope * x + c_shift)


            shift_axis = raw_axis - (c_slope * c_to + c_shift)
            centre = start + 0.5 * SLICE_M
            for sign, side in ((1.0, "left"), (-1.0, "right")):
                out = {}
                for name, line in (("raw", raw_axis), ("fixed", fixed_axis),
                                   ("shift", shift_axis)):
                    values = sign * (y - line)
                    faces = _faces(values[(values >= lateral[0]) & (values <= lateral[1])])
                    if faces is not None:
                        out[name] = faces
                if len(out) < 3:
                    continue
                rows.append({"kind": "far", "frame": frame.index, "side": side,
                             "x_m": round(centre, 1), "world_m": round(path_m + centre, 1),
                             "raw_kth_m": round(out["raw"][0], 4),
                             "raw_med_m": round(out["raw"][1], 4),
                             "fixed_kth_m": round(out["fixed"][0], 4),
                             "fixed_med_m": round(out["fixed"][1], 4),
                             "shift_kth_m": round(out["shift"][0], 4),
                             "fix_slope": round(c_slope, 5), "fix_shift_m": round(c_shift, 4),
                             "fix_span_m": [round(c_from, 1), round(c_to, 1)],
                             "joint_m": round(axis.x_joint_m, 1)})
        print(f"\rкадр {frame.index}", end="", flush=True)
    print()
    return rows, substituted


def _axis_error(far_face: float, near_face: float, side: str) -> float:
    sign = 1.0 if side == "left" else -1.0
    return -sign * (far_face - near_face)


def summarise(rows: list[dict], bin_m: float, bands: list[tuple[float, float]],
              near_bands: tuple[tuple[float, float], tuple[float, float]]) -> None:
    near = [r for r in rows if r["kind"] == "near"]
    far = [r for r in rows if r["kind"] == "far"]
    key = lambda r: (r["side"], float(np.floor(r["world_m"] / bin_m) * bin_m))
    near_by: dict[tuple, list[dict]] = {}
    for r in near:
        near_by.setdefault(key(r), []).append(r)

    print(f"\nсрезов издалека {len(far)}, вблизи {len(near)}")
    print("\nОшибка продлённой оси, мм (y оси − y правды; правда — лицо стены от рельсов "
          "вблизи на том же метре пути). Лицо: k-я точка / медиана полосы.")
    print("\n| дальность, м | сторона | метров пути | сырая Б′: медиана | |ош.| p90 "
          "| с поправкой (в): медиана | |ош.| p90 | способы лица расходятся, медиана |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for lo, hi in bands:
        for side in ("left", "right"):
            raw, fixed, spread = [], [], []
            groups: dict[tuple, list[dict]] = {}
            for r in far:
                if r["side"] == side and lo <= r["x_m"] < hi:
                    groups.setdefault(key(r), []).append(r)
            for k, group in groups.items():
                ref = near_by.get(k)
                if not ref:
                    continue
                near_kth = float(np.median([r["face_kth_m"] for r in ref]))
                near_med = float(np.median([r["face_med_m"] for r in ref]))
                raw.append(_axis_error(np.median([g["raw_kth_m"] for g in group]), near_kth, side))
                fixed.append(_axis_error(np.median([g["fixed_kth_m"] for g in group]),
                                         near_kth, side))
                spread.append(_axis_error(np.median([g["raw_med_m"] for g in group]),
                                          near_med, side) - raw[-1])
            if not raw:
                print(f"| {lo:g}–{hi:g} | {side} | 0 | | | | | |")
                continue
            raw_a, fix_a = 1000 * np.asarray(raw), 1000 * np.asarray(fixed)
            print(f"| {lo:g}–{hi:g} | {side} | {len(raw)} | {np.median(raw_a):+.0f} "
                  f"| {np.percentile(np.abs(raw_a), 90):.0f} | {np.median(fix_a):+.0f} "
                  f"| {np.percentile(np.abs(fix_a), 90):.0f} "
                  f"| {1000 * np.median(spread):+.0f} |")


    print("\nПара сторон на одном метре пути, мм: ошибка оси = полусумма, "
          "симметричный сдвиг лица = полуразность.")
    print("\n| дальность, м | метров с обеими сторонами | сырая Б′: медиана | |ош.| p90 "
          "| (в) сдвиг + наклон: медиана | |ош.| p90 | только сдвиг: медиана | |ош.| p90 "
          "| полуразность, медиана |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for lo, hi in bands:
        per: dict[str, dict[float, tuple[float, float]]] = {"left": {}, "right": {}}
        for side in ("left", "right"):
            groups: dict[tuple, list[dict]] = {}
            for r in far:
                if r["side"] == side and lo <= r["x_m"] < hi:
                    groups.setdefault(key(r), []).append(r)
            for k, group in groups.items():
                ref = near_by.get(k)
                if not ref:
                    continue
                near_kth = float(np.median([r["face_kth_m"] for r in ref]))
                per[side][k[1]] = (
                    _axis_error(np.median([g["raw_kth_m"] for g in group]), near_kth, side),
                    _axis_error(np.median([g["fixed_kth_m"] for g in group]), near_kth, side),
                    _axis_error(np.median([g["shift_kth_m"] for g in group]), near_kth, side))
        both = sorted(set(per["left"]) & set(per["right"]))
        if not both:
            print(f"| {lo:g}–{hi:g} | 0 | | | | | | | |")
            continue
        raw = 1000 * np.array([0.5 * (per["left"][w][0] + per["right"][w][0]) for w in both])
        fixed = 1000 * np.array([0.5 * (per["left"][w][1] + per["right"][w][1]) for w in both])
        shift = 1000 * np.array([0.5 * (per["left"][w][2] + per["right"][w][2]) for w in both])
        half = 1000 * np.array([0.5 * (per["left"][w][0] - per["right"][w][0]) for w in both])
        print(f"| {lo:g}–{hi:g} | {len(both)} | {np.median(raw):+.0f} "
              f"| {np.percentile(np.abs(raw), 90):.0f} | {np.median(fixed):+.0f} "
              f"| {np.percentile(np.abs(fixed), 90):.0f} | {np.median(shift):+.0f} "
              f"| {np.percentile(np.abs(shift), 90):.0f} | {np.median(half):+.0f} |")


    (a_lo, a_hi), (b_lo, b_hi) = near_bands
    floor = []
    for side in ("left", "right"):
        by_a: dict[tuple, list[float]] = {}
        by_b: dict[tuple, list[float]] = {}
        for r in near:
            if r["side"] != side:
                continue
            if a_lo <= r["x_m"] < a_hi:
                by_a.setdefault(key(r), []).append(r["face_kth_m"])
            elif b_lo <= r["x_m"] < b_hi:
                by_b.setdefault(key(r), []).append(r["face_kth_m"])
        for k in set(by_a) & set(by_b):
            floor.append(1000 * (np.median(by_a[k]) - np.median(by_b[k])))
    paired_floor = []
    per_side: dict[str, dict[float, float]] = {"left": {}, "right": {}}
    for side in ("left", "right"):
        by_a: dict[tuple, list[float]] = {}
        by_b: dict[tuple, list[float]] = {}
        for r in near:
            if r["side"] != side:
                continue
            if a_lo <= r["x_m"] < a_hi:
                by_a.setdefault(key(r), []).append(r["face_kth_m"])
            elif b_lo <= r["x_m"] < b_hi:
                by_b.setdefault(key(r), []).append(r["face_kth_m"])
        for k in set(by_a) & set(by_b):
            per_side[side][k[1]] = _axis_error(np.median(by_a[k]), np.median(by_b[k]), side)
    for w in set(per_side["left"]) & set(per_side["right"]):
        paired_floor.append(500 * (per_side["left"][w] + per_side["right"][w]))
    if paired_floor:
        pf = np.asarray(paired_floor)
        print(f"\nпол метода для пары сторон: медиана {np.median(pf):+.0f} мм, |.| p90 "
              f"{np.percentile(np.abs(pf), 90):.0f} мм, n = {pf.size}")
    if floor:
        f = np.asarray(floor)
        print(f"\nпол метода (лицо от рельсов на {a_lo:g}–{a_hi:g} м против {b_lo:g}–{b_hi:g} м, "
              f"один метр пути): медиана {np.median(f):+.0f} мм, |.| p90 "
              f"{np.percentile(np.abs(f), 90):.0f} мм, n = {f.size}")
    fixes = {(r["frame"]): (r["fix_slope"], r["fix_shift_m"], r["fix_span_m"]) for r in far}
    if fixes:
        slopes = np.array([v[0] for v in fixes.values()]) * 1000
        shifts = np.array([v[1] for v in fixes.values()]) * 1000
        spans = np.array([v[2][1] - v[2][0] for v in fixes.values()])
        print(f"поправка (в) по {len(fixes)} кадрам: наклон {np.median(slopes):+.1f} мм/м "
              f"(p10…p90 {np.percentile(slopes, 10):+.1f}…{np.percentile(slopes, 90):+.1f}), "
              f"сдвиг {np.median(shifts):+.0f} мм, перекрытие {np.median(spans):.1f} м")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--record", required=True)
    parser.add_argument("--lateral", type=float, nargs=2, default=(1.65, 2.80),
                        help="окно лица стены по поперечнику, м: снаружи кромки платформы "
                             "(1.45) и Ом (1.62)")
    parser.add_argument("--height", type=float, nargs=2, default=(0.90, 1.80),
                        help="окно по высоте над УГР, м: выше контактного рельса и платформы")
    parser.add_argument("--far", type=float, nargs=2, default=(25.0, 39.0))
    parser.add_argument("--near-min", type=float, default=8.0)
    parser.add_argument("--bin-m", type=float, default=2.0)
    parser.add_argument("--out", default="out/extension_residual")
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    rows, substituted = run(cfg, args.record, tuple(args.lateral), tuple(args.height),
                            tuple(args.far), args.near_min)
    print(f"кадров с подставленным путём: {substituted}")
    summarise(rows, args.bin_m, [(25.0, 29.0), (29.0, 33.0), (33.0, 39.0)],
              ((18.0, 24.0), (8.0, 14.0)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{args.record}.jsonl"
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"записано: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

