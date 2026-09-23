# Проверка числом, в какой системе координат ядро отдаёт выход.

from __future__ import annotations

import argparse
import sys

import numpy as np

from core.axis import _ridge
from core.config import AxisConfig, Config
from core.detector import detect
from core.pipeline import ObstacleDetector
from core.preprocess import GroundTracker, preprocess_frame
from tools.bag_reader import iter_frames
from tools.scenarios import _to_leveled, build_parser, iter_scenario_frames


def _inside(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.all((points >= lo) & (points <= hi), axis=1)


def _frames(cfg: Config, args):
    if args.scenario_argv:
        scenario = build_parser().parse_args(args.scenario_argv)
        for item in iter_scenario_frames(cfg, scenario, raw=True):
            f = item.frame
            yield f.index, item.injected, f.intensity, f.ring, f.timestamp, f.stamp_ns
            if f.index >= max(args.frames):
                return
    else:
        for f in iter_frames(cfg.data.path(args.record), cfg.bag, limit=max(args.frames) + 1):
            yield f.index, f.xyz, f.intensity, f.ring, f.timestamp, f.stamp_ns


def _rails_from_axis(points: np.ndarray, axis: dict, cfg: AxisConfig) -> str:
    slope = np.tan(np.radians(axis["slope_deg"]))
    offset = points[:, 1] - (slope * points[:, 0] + axis["y_center_m"])
    half, window = cfg.rails_half_gauge_m, cfg.rails_search_halfwidth_m
    left = _ridge(offset, half - window, half + window, cfg)
    right = _ridge(offset, -half - window, -half + window, cfg)
    if left is None or right is None:
        return "гребней головок нет"
    return f"середина колеи {1000 * 0.5 * (left + right):+.0f} мм, колея {1000 * (left - right):.0f} мм"


def run(cfg: Config, args) -> None:
    detector = ObstacleDetector(cfg)
    shadow = GroundTracker(cfg.preprocess.ground)
    wanted = set(args.frames)
    a = cfg.axis
    for index, xyz, intensity, ring, t_rel, stamp in _frames(cfg, args):
        outcome = detector.process(xyz, intensity, ring, t_rel, stamp)
        prepared = preprocess_frame(xyz, intensity, ring, t_rel, cfg.preprocess, tracker=shadow)
        if index not in wanted:
            continue
        candidates = outcome.debug["candidates_raw"]["detections"]
        replay = detect(prepared.xyz, detector.axis_tracker.last, cfg.gauge, cfg.detector)
        same = [d.to_dict() for d in replay.detections] == candidates

        matrix = _to_leveled(prepared.plane, cfg)
        shift = np.array([0.0, 0.0, float(prepared.plane.offset)])
        leveled = prepared.xyz.astype(np.float64)
        raw_back = (leveled - shift) @ matrix
        round_trip = float(np.abs(raw_back @ matrix.T + shift - leveled).max())
        angle = float(np.degrees(np.arccos(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))))

        print(f"\n=== кадр {index}: кандидатов {len(candidates)}, тревог "
              f"{len(outcome.detections)}, тень совпала с ядром: {same}")
        print(f"  сырая → выровненная: поворот {angle:.3f}°, сдвиг z {shift[2]:+.3f} м, "
              f"ошибка круга {round_trip:.1e} м")

        axis = outcome.debug.get("axis")
        if axis is not None and axis["state"] != "lost":
            x_lo, x_hi = axis["x_traced_m"]
            head = ((leveled[:, 2] > a.rails_z_min_m) & (leveled[:, 2] < a.rails_z_max_m)
                    & (leveled[:, 0] >= x_lo) & (leveled[:, 0] <= x_hi))
            print(f"  ось, точек полосы головок {int(head.sum())}: в выровненной — "
                  f"{_rails_from_axis(leveled[head], axis, a)}; те же точки в сырой — "
                  f"{_rails_from_axis(raw_back[head], axis, a)}")
        corridor = outcome.debug.get("corridor")

        for number, det in enumerate(candidates):
            lo = np.asarray(det["bbox_min_xyz"])
            hi = np.asarray(det["bbox_max_xyz"])
            own = _inside(leveled, lo, hi)
            mine, back = leveled[own], raw_back[own]
            centroid = np.asarray(det["centroid_xyz"])
            raw_centroid = (centroid - shift) @ matrix
            print(f"  кандидат {number}: distance_m {det['distance_m']:.3f}, "
                  f"n_points {det['n_points']}, точек в рамке {int(own.sum())}")
            print(f"    рамка: доля внутри в выровненной {float(_inside(mine, lo, hi).mean()):.3f}, "
                  f"те же точки в сырой {float(_inside(back, lo, hi).mean()):.3f}")
            if corridor is not None:
                (cx0, cx1), (cz0, cz1) = corridor["x_m"], corridor["z_m"]

                def share(points: np.ndarray) -> float:
                    return float(((points[:, 0] >= cx0) & (points[:, 0] <= cx1)
                                  & (points[:, 2] > cz0) & (points[:, 2] < cz1)).mean())
                print(f"    коридор x {cx0}…{cx1}, z {cz0}…{cz1}: доля точек кандидата "
                      f"в выровненной {share(mine):.3f}, в сырой {share(back):.3f}")
            print(f"    центр как отдан {np.round(centroid, 3).tolist()}, "
                  f"в сырой {np.round(raw_centroid, 3).tolist()}")


            print(f"    ближайшая точка: норма в выровненной {np.linalg.norm(mine, axis=1).min():.3f}, "
                  f"x в выровненной {mine[:, 0].min():.3f}, "
                  f"дальность от лидара {np.linalg.norm(back, axis=1).min():.3f} м")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    scenario_argv = []
    if "--" in argv:
        split = argv.index("--")
        argv, scenario_argv = argv[:split], argv[split + 1:]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--record", help="запись; без неё кадры берутся из сценария после --")
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    args = parser.parse_args(argv)
    args.scenario_argv = scenario_argv
    if not args.record and not scenario_argv:
        parser.error("нужна --record или ключи сценария после --")
    run(Config.from_yaml(args.config), args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

