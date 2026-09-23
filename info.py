#!/usr/bin/env python3
# Сводка по записям в корне проекта: число кадров, точек в кадре, длительность и частота.

from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

ROOT = Path("/home/denimele/hackathon/for_hackathon")
ts = get_typestore(Stores.ROS2_HUMBLE)
for d in sorted(p for p in ROOT.iterdir() if (p / "metadata.yaml").exists()):
    with AnyReader([d], default_typestore=ts) as r:
        c = list(r.connections)[0]
        _, _, raw = next(r.messages(connections=[c]))
        m = r.deserialize(raw, c.msgtype)
        sec = r.duration / 1e9
        print(
            f"{d.name:42s} {c.msgcount:4d} кадр  {m.width:7d} точек  "
            f"{sec:5.1f} с  {c.msgcount / sec:4.1f} Гц  {d.stat().st_size / 1e9:.1f} ГБ"
            if False
            else f"{d.name:42s} {c.msgcount:4d} кадр  {m.width:7d} точек/кадр  {sec:5.1f} с  {c.msgcount / sec:4.1f} Гц"
        )
