# Тесты выгрузки кадров в PLY.

from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.export_scenario_frames import cloud_colors, write_cloud_ply, write_lines_ply
from tools.view_geometry import Lines


def _read_binary_cloud(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    dtype = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    return np.frombuffer(raw[end:], dtype=dtype)


def test_cloud_round_trips_with_the_body_coloured(tmp_path: Path) -> None:
    xyz = np.array([[1.0, -2.0, 0.5], [3.0, 4.0, -1.0], [0.1, 0.2, 0.3]])
    body = np.array([False, True, False])
    rgb = cloud_colors(np.array([0.0, 100.0, 255.0]), body, (1.0, 0.0, 0.0))
    path = tmp_path / "frame.ply"
    write_cloud_ply(path, xyz, rgb)
    data = _read_binary_cloud(path)
    assert np.allclose(np.column_stack([data["x"], data["y"], data["z"]]), xyz)
    assert (data["red"][1], data["green"][1], data["blue"][1]) == (255, 0, 0)
    assert data["red"][2] == data["green"][2] == data["blue"][2] == 255


def test_lines_file_lists_every_vertex_and_edge(tmp_path: Path) -> None:
    groups = [Lines(np.zeros((3, 3)), np.array([[0, 1], [1, 2]]), (0, 1, 0), "ось"),
              Lines(np.ones((2, 3)), np.array([[0, 1]]), (1, 0, 0), "тревоги")]
    path = tmp_path / "frame_lines.ply"
    write_lines_ply(path, groups)
    text = path.read_text(encoding="ascii").splitlines()
    assert "element vertex 5" in text and "element edge 3" in text

    assert text[-1] == "3 4"

