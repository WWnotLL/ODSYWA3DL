# Копия записи rosbag2 со вписанной фигурой и разметкой — позитив для проигрывания через ноду.

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np
from rosbags.typesys import Stores, get_typestore

from core.config import Config
from core.pipeline import ObstacleDetector
from tools.bag_reader import iter_frames, point_dtype
from tools.scenarios import build_parser, iter_scenario_frames
from tools.visualize_run import load_viz


def _summary(result) -> tuple:
    debug = result.debug
    raw = debug.get("candidates_raw") or {}
    return (result.obstacle_found, len(raw.get("detections", [])),
            (debug.get("axis") or {}).get("state"))


def _release(path: Path) -> None:
    if hasattr(os, "posix_fadvise") and path.exists():
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--viz", type=Path, required=True)
    parser.add_argument("--story", required=True)
    parser.add_argument("--out", type=Path, default=Path("out/bags"))
    args = parser.parse_args(argv)

    viz = load_viz(args.viz)
    story = viz["stories"][args.story]
    if "scenario_argv" not in story:
        raise SystemExit(f"сюжет {args.story} — исходная запись без вставки")
    cfg = Config.from_yaml(viz["core_config"])
    scenario = build_parser().parse_args(story["scenario_argv"])
    record = scenario.record
    source = cfg.data.path(record)
    target = args.out / f"{record}_{args.story}_figure"
    if target.exists():
        raise SystemExit(f"{target} уже есть — удалите его сами, молча не перезаписываю")
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    db_path = next(target.glob("*.db3"))
    print(f"копия: {source} → {target}")

    typestore = get_typestore(Stores[cfg.bag.typestore])
    con = sqlite3.connect(db_path)


    ids = [row[0] for row in con.execute("SELECT id FROM messages ORDER BY timestamp, id")]
    msgtype = con.execute("SELECT type FROM topics").fetchone()[0]

    memory = ObstacleDetector(cfg)
    in_memory: dict[int, tuple] = {}
    truth = []
    changed = 0
    for item in iter_scenario_frames(cfg, scenario, raw=True):
        f = item.frame
        in_memory[f.index] = _summary(memory.process(
            item.injected, f.intensity, f.ring, f.timestamp, f.stamp_ns))
        moved = np.linalg.norm(np.asarray(item.injected, float) - np.asarray(f.xyz, float),
                               axis=1) > viz["export"]["moved_tolerance_m"]
        if item.geometry is None or not moved.any():
            continue
        message_id = ids[f.index]
        blob = con.execute("SELECT data FROM messages WHERE id = ?", (message_id,)).fetchone()[0]
        message = typestore.deserialize_cdr(blob, msgtype)
        count = int(message.width) * int(message.height)
        points = np.frombuffer(message.data, dtype=point_dtype(message, cfg.bag), count=count).copy()
        injected = np.asarray(item.injected, dtype=np.float32)
        points["x"][moved], points["y"][moved], points["z"][moved] = (
            injected[moved, 0], injected[moved, 1], injected[moved, 2])
        payload = np.frombuffer(points.tobytes(), dtype=np.uint8)
        if payload.size != len(message.data):
            raise SystemExit(f"кадр {f.index}: размер облака изменился — так быть не должно")
        new_blob = typestore.serialize_cdr(dataclasses.replace(message, data=payload), msgtype)
        con.execute("UPDATE messages SET data = ? WHERE id = ?", (bytes(new_blob), message_id))
        changed += 1

        (cx, cy), z_bottom, radius, height = item.centre, *item.geometry[1:]
        base_lidar = item.clean.transform.inverse(np.array([[cx, cy, z_bottom],
                                                            [cx, cy, z_bottom + height]]))
        truth.append({
            "frame": int(f.index),
            "stamp_ns": int(f.stamp_ns),
            "run": item.run_index,
            "points_on_figure": int(moved.sum()),
            "radius_m": radius,
            "height_m": height,

            "track": {"centre_xy_m": [round(cx, 3), round(cy, 3)],
                      "z_bottom_m": round(z_bottom, 3), "z_top_m": round(z_bottom + height, 3)},

            "lidar": {"axis_bottom_m": np.round(base_lidar[0], 3).tolist(),
                      "axis_top_m": np.round(base_lidar[1], 3).tolist()},
        })
        if changed % viz["export"]["frames_every"] == 0:
            con.commit()
            _release(db_path)
        print(f"\rкадр {f.index}: точек на фигуре {int(moved.sum())}", end="", flush=True)
    con.commit()
    con.close()
    _release(db_path)
    (target / "figure_truth.json").write_text(
        json.dumps({"source_record": record, "scenario_argv": story["scenario_argv"],
                    "frames": truth}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nизменено сообщений: {changed} из {len(ids)}; разметка: {target / 'figure_truth.json'}")


    replay = ObstacleDetector(cfg)
    differ, seen = [], 0
    for f in iter_frames(target, cfg.bag):
        got = _summary(replay.process(f.xyz, f.intensity, f.ring, f.timestamp, f.stamp_ns))
        if f.index in in_memory:
            seen += 1
            if got != in_memory[f.index]:
                differ.append((f.index, in_memory[f.index], got))
    print(f"проверка копии: сравнено кадров {seen}, расхождений {len(differ)}")
    for frame, mem, disk in differ[:10]:
        print(f"  кадр {frame}: в памяти {mem}, из копии {disk}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

