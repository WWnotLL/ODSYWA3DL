# Лицо продольной конструкции на одних метрах пути — от оси ядра издалека и от рельсов вблизи.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from core.axis import AxisTracker, _rail_centres, _trim_to_clearance
from core.config import Config
from core.odometry import estimate_shift, longitudinal_profile
from core.preprocess import GroundTracker, preprocess_frame
from tools.bag_reader import iter_frames


SLICE_M = 2.0


FACE_RANK = 4


TOP_BAND_M = 0.20


def _kth(values: np.ndarray, k: int) -> float | None:
    return None if values.size < k else float(np.partition(values, k - 1)[k - 1])


def _face_and_top(lateral: np.ndarray, height: np.ndarray) -> tuple[float | None, float | None]:
    face = _kth(lateral, FACE_RANK)
    if face is None:
        return None, None
    near_face = height[lateral <= face + TOP_BAND_M]
    return face, float(np.percentile(near_face, 98))


def run(cfg: Config, record: str, first: int, last: int, sign: float,
        lateral: tuple[float, float], z_window: tuple[float, float],
        x_max: float) -> tuple[list[dict], int]:
    ground = GroundTracker(cfg.preprocess.ground)
    tracker = AxisTracker(cfg.axis)
    railhead = cfg.gauge.rail_head_offset_m or 0.0
    previous_profile, previous_stamp = None, None
    path_m, last_advance, substituted = 0.0, None, 0
    rows: list[dict] = []
    for frame in iter_frames(cfg.data.path(record), cfg.bag, limit=last + 1):
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
            substituted += frame.index >= first
        path_m += advance or 0.0
        last_advance = advance if advance else last_advance
        if frame.index < first or axis is None:
            continue

        rails, _ = _rail_centres(xyz, cfg.axis)
        rails_line, rails_span = None, None
        if rails is not None:
            kept, slope, y0, _ = _trim_to_clearance(rails, cfg.axis)
            rails_line = (float(slope), float(y0))
            rails_span = (float(kept[0, 0]) - 0.5 * cfg.axis.slice_m,
                          float(kept[-1, 0]) + 0.5 * cfg.axis.slice_m)

        height = xyz[:, 2] - railhead
        in_height = (height >= z_window[0]) & (height <= z_window[1])
        side_axis = sign * axis.offset(xyz)
        joint = axis.x_joint_m
        for start in np.arange(axis.x_start_m, min(axis.x_end_m, x_max), SLICE_M):
            in_slice = in_height & (xyz[:, 0] >= start) & (xyz[:, 0] < start + SLICE_M)
            band = in_slice & (side_axis >= lateral[0]) & (side_axis <= lateral[1])
            if band.sum() < FACE_RANK:
                continue
            centre = start + 0.5 * SLICE_M
            face_axis, top = _face_and_top(side_axis[band], height[band])
            row = {
                "frame": int(frame.index),
                "x_m": round(float(centre), 1),
                "world_m": round(float(path_m + centre), 1),
                "n_points": int(band.sum()),
                "face_axis_m": round(face_axis, 3),
                "top_m": round(top, 3),


                "segment": ("bed" if joint is not None and centre > joint
                            else "rails"),


                "sensor_above_railhead_m": round(float(prepared.plane.offset) - railhead, 3),
            }
            if rails_line is not None and rails_span[0] <= centre <= rails_span[1]:
                side_rails = sign * (xyz[:, 1] - (rails_line[0] * xyz[:, 0] + rails_line[1]))
                band_r = in_slice & (side_rails >= lateral[0]) & (side_rails <= lateral[1])
                face_rails = _kth(side_rails[band_r], FACE_RANK)
                if face_rails is not None:
                    row["face_rails_m"] = round(face_rails, 3)
            rows.append(row)
    return rows, substituted


def summarise(rows: list[dict], substituted: int, bin_m: float) -> None:
    print(f"\nсрезов со стенкой: {len(rows)}; кадров с подставленным путём: {substituted}")
    if not rows:
        print("в окне нет ни одного среза со стенкой")
        return
    by_world: dict[float, list[dict]] = {}
    for r in rows:
        key = float(np.floor(r["world_m"] / bin_m) * bin_m)
        by_world.setdefault(key, []).append(r)

    print("\nОдин и тот же метр пути: издалека (от оси ядра, за стыком) и вблизи "
          "(от середины колеи по рельсам).")
    print("\n| путь, м | издалека: кадры | дальность | лицо от оси | "
          "вблизи: кадры | дальность | лицо от рельсов | лицо от оси | разница |")
    print("|---|---|---|---:|---|---|---:|---:|---:|")
    deltas = []
    for key in sorted(by_world):
        group = by_world[key]
        far = [r for r in group if r["segment"] == "bed"]
        near = [r for r in group if "face_rails_m" in r]
        if not far or not near:
            continue
        far_face = float(np.median([r["face_axis_m"] for r in far]))
        near_rails = float(np.median([r["face_rails_m"] for r in near]))
        near_axis = float(np.median([r["face_axis_m"] for r in near]))
        deltas.append(near_rails - far_face)
        print(f"| {key:.0f}…{key + bin_m:.0f} | {min(r['frame'] for r in far)}–"
              f"{max(r['frame'] for r in far)} | "
              f"{min(r['x_m'] for r in far):.0f}–{max(r['x_m'] for r in far):.0f} | "
              f"{far_face:.3f} | {min(r['frame'] for r in near)}–{max(r['frame'] for r in near)} | "
              f"{min(r['x_m'] for r in near):.0f}–{max(r['x_m'] for r in near):.0f} | "
              f"**{near_rails:.3f}** | {near_axis:.3f} | {1000 * (near_rails - far_face):+.0f} мм |")
    if deltas:
        d = np.asarray(deltas)
        print(f"\nлицо вблизи от рельсов минус лицо издалека от оси: медиана "
              f"{1000 * np.median(d):+.0f} мм, {1000 * d.min():+.0f}…{1000 * d.max():+.0f} мм "
              f"по {d.size} метрам пути. Плюс — издалека конструкция казалась ближе к пути.")

    rails_faces = np.array([r["face_rails_m"] for r in rows if "face_rails_m" in r])
    if rails_faces.size:
        print(f"\nлицо от середины колеи по всем ближним срезам: медиана "
              f"{np.median(rails_faces):.3f}, p5 {np.percentile(rails_faces, 5):.3f}, "
              f"min {rails_faces.min():.3f} (n={rails_faces.size})")
    sensor = np.array([r["sensor_above_railhead_m"] for r in rows])
    print(f"сенсор над УГР: медиана {np.median(sensor):.3f} м")
    tops = np.array([r["top_m"] for r in rows])
    print(f"верх у лица: медиана {np.median(tops):.3f} м над УГР, "
          f"p10 {np.percentile(tops, 10):.3f}, p90 {np.percentile(tops, 90):.3f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--record", required=True)
    parser.add_argument("--frames", type=int, nargs=2, required=True, metavar=("FIRST", "LAST"))
    parser.add_argument("--side", choices=["left", "right"], required=True,
                        help="сторона конструкции; right — отрицательное смещение по REP-103")
    parser.add_argument("--lateral", type=float, nargs=2, default=(1.20, 2.10),
                        metavar=("MIN", "MAX"), help="окно по поперечнику от опоры, м")
    parser.add_argument("--height", type=float, nargs=2, default=(0.55, 1.20),
                        metavar=("MIN", "MAX"),
                        help="окно по высоте над УГР, м; по умолчанию выше потолка "
                             "выреза контактного рельса, чтобы он не попал в лицо")
    parser.add_argument("--x-max", type=float, default=44.0)
    parser.add_argument("--bin-m", type=float, default=2.0)
    parser.add_argument("--out", default="runs/wall")
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    sign = -1.0 if args.side == "right" else 1.0
    rows, substituted = run(cfg, args.record, args.frames[0], args.frames[1], sign,
                            tuple(args.lateral), tuple(args.height), args.x_max)
    summarise(rows, substituted, args.bin_m)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{args.record}_{args.side}.jsonl"
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nзаписано: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

