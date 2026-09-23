# Сверка выхода ядра с разметкой вставленной фигуры на одном знаменателе кадров.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from core.config import Config
from core.pipeline import ObstacleDetector
from tools.bag_reader import iter_frames
from tools.positive_control import _overlaps
from tools.scenarios import build_parser


def figure_box(entry: dict) -> tuple[np.ndarray, np.ndarray]:
    (cx, cy), r = entry["track"]["centre_xy_m"], entry["radius_m"]
    lo = np.array([cx - r, cy - r, entry["track"]["z_bottom_m"]])
    hi = np.array([cx + r, cy + r, entry["track"]["z_top_m"]])
    return lo, hi


def split(boxes: list, box: tuple | None, tolerance: float) -> tuple[int, list[dict]]:
    on, off = 0, []
    for b in boxes:
        d = SimpleNamespace(bbox_min_xyz=b["bbox_min_xyz"], bbox_max_xyz=b["bbox_max_xyz"])
        if box is not None and _overlaps([d], box[0], box[1], tolerance):
            on += 1
        else:
            off.append({"bbox_min_xyz": np.round(b["bbox_min_xyz"], 2).tolist(),
                        "bbox_max_xyz": np.round(b["bbox_max_xyz"], 2).tolist()})
    return on, off


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--last", type=int, required=True)
    parser.add_argument("--tolerance", type=float,
                        default=build_parser().get_default("tolerance"),
                        help="допуск пересечения, по умолчанию — как у tools.scenarios")
    parser.add_argument("--out", type=Path, default=None, help="покадровая таблица JSON")
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    truth = json.loads((args.bag / "figure_truth.json").read_text(encoding="utf-8"))
    by_frame = {e["frame"]: e for e in truth["frames"]}
    detector = ObstacleDetector(cfg)
    rows = []
    for f in iter_frames(args.bag, cfg.bag, limit=args.last + 1):
        result = detector.process(f.xyz, f.intensity, f.ring, f.timestamp, f.stamp_ns).to_dict()
        if f.index < args.first:
            continue
        entry = by_frame.get(f.index)
        box = None if entry is None else figure_box(entry)
        raw = (result["debug"].get("candidates_raw") or {}).get("detections", [])
        cand_on, cand_off = split(raw, box, args.tolerance)
        alarm_on, alarm_off = split(result["detections"], box, args.tolerance)
        rows.append({"frame": f.index, "figure": entry is not None,
                     "figure_points": 0 if entry is None else entry["points_on_figure"],
                     "axis_state": (result["debug"].get("axis") or {}).get("state"),
                     "figure_centre_xy_m": None if entry is None
                     else entry["track"]["centre_xy_m"],
                     "cand_on": cand_on, "cand_off": len(cand_off), "cand_off_boxes": cand_off,
                     "alarm_on": alarm_on, "alarm_off": len(alarm_off)})
        print(f"\rкадр {f.index} / {args.last}", end="", flush=True)
    print()

    n = len(rows)

    def count(pred) -> str:
        k = sum(1 for r in rows if pred(r))
        return f"{k}/{n}"

    print(f"кадров в диапазоне {args.first}–{args.last}: {n}; допуск {args.tolerance} м")
    table = [
        ("фигура в кадре (разметка)", lambda r: r["figure"]),
        ("без оси (state = lost)", lambda r: r["axis_state"] == "lost"),
        ("кадров с кандидатом, любым", lambda r: r["cand_on"] + r["cand_off"] > 0),
        ("  кандидат на фигуре", lambda r: r["cand_on"] > 0),
        ("  кандидат только мимо фигуры", lambda r: r["cand_on"] == 0 and r["cand_off"] > 0),
        ("  кандидат мимо фигуры, любой", lambda r: r["cand_off"] > 0),
        ("кадров с тревогой, любой", lambda r: r["alarm_on"] + r["alarm_off"] > 0),
        ("  тревога на фигуре", lambda r: r["alarm_on"] > 0),
        ("  тревога только мимо фигуры", lambda r: r["alarm_on"] == 0 and r["alarm_off"] > 0),
        ("  тревога мимо фигуры, любая", lambda r: r["alarm_off"] > 0),
    ]
    for label, pred in table:
        print(f"  {label:34s} {count(pred)}")
    for r in rows:
        for b in r["cand_off_boxes"]:
            print(f"  кадр {r['frame']}: кандидат мимо фигуры {b['bbox_min_xyz']} … "
                  f"{b['bbox_max_xyz']}, фигура {r['figure_centre_xy_m']}, "
                  f"на фигуре в кадре {r['cand_on']}")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"покадрово: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

