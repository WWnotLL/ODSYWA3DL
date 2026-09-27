# Потоковое чтение кадров из rosbag2 без ROS с освобождением страничного кэша.

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from core.config import BagConfig


POINTFIELD_DTYPES = {
    1: "<i1", 2: "<u1", 3: "<i2", 4: "<u2",
    5: "<i4", 6: "<u4", 7: "<f4", 8: "<f8",
}


class BagReadError(RuntimeError):
    pass


@dataclass(frozen=True, eq=False)
class RawFrame:
    index: int
    stamp_ns: int
    xyz: np.ndarray
    intensity: np.ndarray
    ring: np.ndarray | None
    timestamp: np.ndarray | None
    frame_id: str
    width: int


REQUIRED_FIELDS = ("x", "y", "z")


def point_dtype(message, cfg: BagConfig, *, partial: bool = False) -> np.dtype:
    names, formats, offsets = [], [], []
    for field in message.fields:
        if field.count != 1:
            raise BagReadError(f"поле {field.name}: count={field.count}, ожидалось 1")
        if field.datatype not in POINTFIELD_DTYPES:
            raise BagReadError(f"поле {field.name}: неизвестный код типа {field.datatype}")
        names.append(field.name)
        formats.append(POINTFIELD_DTYPES[field.datatype])
        offsets.append(int(field.offset))

    required = REQUIRED_FIELDS if partial else cfg.expected_fields
    missing = [name for name in required if name not in names]
    if missing:
        raise BagReadError(f"в сообщении нет ожидаемых полей {missing}, есть {names}")
    if not partial and int(message.point_step) != cfg.expected_point_step:
        raise BagReadError(
            f"point_step={message.point_step}, в конфиге ожидается {cfg.expected_point_step}"
        )
    return np.dtype(
        {"names": names, "formats": formats, "offsets": offsets, "itemsize": int(message.point_step)}
    )


def select_connections(connections, cfg: BagConfig) -> list:
    matching = [c for c in connections if c.msgtype == cfg.message_type]
    if not matching:
        available = sorted({(c.topic, c.msgtype) for c in connections})
        raise BagReadError(f"в записи нет сообщений типа {cfg.message_type!r}, есть {available}")
    if len({c.topic for c in matching}) == 1:
        return matching
    for topic in cfg.preferred_topics:
        chosen = [c for c in matching if c.topic == topic]
        if chosen:
            return chosen
    topics = sorted({c.topic for c in matching})
    raise BagReadError(
        f"в записи несколько топиков типа {cfg.message_type}: {topics}. "
        "Ни один не указан в bag.preferred_topics — выбор не делается наугад."
    )


def _release_page_cache(bag_dir: Path) -> None:
    if not hasattr(os, "posix_fadvise"):
        return
    for path in bag_dir.glob("*.db3*"):
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def iter_frames(
    bag_dir: str | Path, cfg: BagConfig, *, limit: int | None = None, stride: int = 1,
    partial: bool = False,
) -> Iterator[RawFrame]:
    bag_dir = Path(bag_dir)
    if not (bag_dir / "metadata.yaml").is_file():
        raise BagReadError(f"{bag_dir}: не похоже на rosbag2 — нет metadata.yaml")
    if stride < 1:
        raise BagReadError(f"stride должен быть положительным, получен {stride}")
    try:
        store = Stores[cfg.typestore]
    except KeyError as error:
        raise BagReadError(f"неизвестный typestore {cfg.typestore!r}") from error


    typestore = get_typestore(store)
    with AnyReader([bag_dir], default_typestore=typestore) as reader:
        connections = select_connections(reader.connections, cfg)

        emitted = 0
        every = cfg.release_page_cache_every_frames
        try:
            for index, (connection, stamp_ns, raw) in enumerate(
                reader.messages(connections=connections)
            ):
                if every and index % every == every - 1:
                    _release_page_cache(bag_dir)
                if index % stride:
                    continue
                if limit is not None and emitted >= limit:
                    return
                message = reader.deserialize(raw, connection.msgtype)
                dtype = point_dtype(message, cfg, partial=partial)
                count = int(message.width) * int(message.height)
                record = np.frombuffer(message.data, dtype=dtype, count=count)
                names = record.dtype.names
                yield RawFrame(
                    index=index,
                    stamp_ns=int(stamp_ns),
                    xyz=np.stack([record["x"], record["y"], record["z"]], axis=1),
                    intensity=(np.asarray(record["intensity"]) if "intensity" in names
                               else np.zeros(count, dtype=np.float32)),
                    ring=np.asarray(record["ring"]) if "ring" in names else None,
                    timestamp=np.asarray(record["timestamp"]) if "timestamp" in names else None,
                    frame_id=str(message.header.frame_id),
                    width=count,
                )
                emitted += 1
        finally:
            if every:
                _release_page_cache(bag_dir)


def count_frames(bag_dir: str | Path, cfg: BagConfig) -> int:
    bag_dir = Path(bag_dir)
    if not (bag_dir / "metadata.yaml").is_file():
        raise BagReadError(f"{bag_dir}: не похоже на rosbag2 — нет metadata.yaml")
    typestore = get_typestore(Stores[cfg.typestore])
    with AnyReader([bag_dir], default_typestore=typestore) as reader:
        return sum(c.msgcount for c in select_connections(reader.connections, cfg))


def read_frame(bag_dir: str | Path, cfg: BagConfig, index: int) -> RawFrame:
    for frame in iter_frames(bag_dir, cfg, limit=index + 1):
        if frame.index == index:
            return frame
    raise BagReadError(f"{bag_dir}: кадр {index} не найден")

