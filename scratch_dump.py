#!/usr/bin/env python3
# Разовый разбор первого кадра doubleT_obstacle: границы, дальности, поля точек; кадр сохраняется в out/scratch.

import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

BAG = Path("/home/denimele/hackathon/for_hackathon/doubleT_obstacle")
OUT = Path("/home/denimele/hackathon/for_hackathon/out/scratch")
OUT.mkdir(parents=True, exist_ok=True)
DT = np.dtype(
    {
        "names": ["x", "y", "z", "intensity", "ring", "timestamp"],
        "formats": ["<f4", "<f4", "<f4", "<f4", "<u2", "<f8"],
        "offsets": [0, 4, 8, 12, 16, 18],
        "itemsize": 26,
    }
)

ts = get_typestore(Stores.ROS2_HUMBLE)
with AnyReader([BAG], default_typestore=ts) as r:
    for c, t, raw in r.messages(connections=list(r.connections)):
        m = r.deserialize(raw, c.msgtype)
        a = np.frombuffer(m.data, dtype=DT, count=m.width * m.height)
        xyz = np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)
        ok = np.isfinite(xyz).all(1)
        print("всего", len(a), "валидных", int(ok.sum()), "NaN", int((~ok).sum()))
        print("bbox min", xyz[ok].min(0), "\nbbox max", xyz[ok].max(0))
        print(
            "дальность м:",
            np.linalg.norm(xyz[ok], axis=1).min(),
            "…",
            np.linalg.norm(xyz[ok], axis=1).max(),
        )
        print("intensity", a["intensity"][ok].min(), "…", a["intensity"][ok].max())
        print(
            "ring",
            a["ring"].min(),
            "…",
            a["ring"].max(),
            "уникальных",
            len(np.unique(a["ring"])),
        )
        print(
            "timestamp точек", a["timestamp"][ok].min(), "…", a["timestamp"][ok].max()
        )
        np.save(OUT / "frame0.npy", xyz[ok])
        break
