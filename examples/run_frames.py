#!/usr/bin/env python3
# Минимальный пример без ROS: кадры PointCloud2 из записи rosbag2 → ObstacleDetector → FrameResult.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import Config
from core.pipeline import ObstacleDetector

POINTCLOUD2 = "sensor_msgs/msg/PointCloud2"
POINTFIELD_DTYPES = {1: "<i1", 2: "<u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4", 7: "<f4", 8: "<f8"}


def cloud_to_arrays(msg) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if msg.is_bigendian:
        raise ValueError("ожидается little-endian PointCloud2")
    dtype = np.dtype({
        "names": [f.name for f in msg.fields],
        "formats": [POINTFIELD_DTYPES[f.datatype] for f in msg.fields],
        "offsets": [f.offset for f in msg.fields],
        "itemsize": msg.point_step,
    })
    points = np.frombuffer(msg.data, dtype=dtype, count=msg.width * msg.height)
    xyz = np.stack([points["x"], points["y"], points["z"]], axis=1)
    return xyz, points["intensity"], points["ring"], points["timestamp"]


def stamp_ns(msg) -> int:
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Прогон ядра по кадрам записи rosbag2 без ROS")
    parser.add_argument("bag", type=Path, help="каталог записи rosbag2")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--frames", type=int, default=5, help="сколько кадров; 0 — все")
    parser.add_argument("--jsonl", type=Path, default=None,
                        help="записать выход покадрово (формат tools.compare_runs)")
    args = parser.parse_args(argv)

    detector = ObstacleDetector(Config.from_yaml(args.config))
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    if args.jsonl:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    out = args.jsonl.open("w", encoding="utf-8") if args.jsonl else None
    with AnyReader([args.bag], default_typestore=typestore) as reader:
        connections = [c for c in reader.connections if c.msgtype == POINTCLOUD2]
        for index, (connection, _, raw) in enumerate(reader.messages(connections=connections)):
            if args.frames and index >= args.frames:
                break
            msg = reader.deserialize(raw, connection.msgtype)
            xyz, intensity, ring, t_rel = cloud_to_arrays(msg)
            result = detector.process(xyz, intensity, ring, t_rel, stamp_ns(msg))
            debug = result.debug
            axis = debug.get("axis") or {}
            corridor = (debug.get("corridor") or {}).get("x_m")
            print(f"кадр {index}: checked={debug['checked']} obstacle_found={result.obstacle_found} "
                  f"nearest_distance_m={result.nearest_distance_m} detections={len(result.detections)} "
                  f"axis={axis.get('state')}/{axis.get('reason')} corridor_x_m={corridor} "
                  f"stop={(debug.get('verdict') or {}).get('stop')} "
                  f"total_ms={debug['timings_ms']['total']}")
            for d in result.detections:
                print(f"    Detection distance_m={d.distance_m:.2f} centroid_xyz="
                      f"{np.round(d.centroid_xyz, 2).tolist()} n_points={d.n_points} score={d.score:.2f}")
            if out is not None:
                out.write(json.dumps({"frame": index, "stamp_ns": stamp_ns(msg), **result.to_dict()},
                                     ensure_ascii=False) + "\n")
    if out is not None:
        out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
