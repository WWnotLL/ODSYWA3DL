# Вставка тела трассировкой лучей: наклонное тело против вертикального и геометрия попаданий.

from __future__ import annotations

import numpy as np

from tools.positive_control import inject_body, inject_tilted_body
from tools.scenarios import body_box


def _fan(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    sensor = np.array([0.0, 0.0, 1.3])
    targets = np.column_stack([rng.uniform(5.0, 40.0, 20000), rng.uniform(-3.0, 3.0, 20000),
                               rng.uniform(-1.0, 4.0, 20000)])
    return sensor, sensor + 1.5 * (targets - sensor)


def test_vertical_tilted_body_matches_vertical_injection():
    sensor, xyz = _fan(np.random.default_rng(0))
    plain, n_plain = inject_body(xyz, sensor, (15.0, 0.2), 0.5, 0.3, 1.0)
    tilted, n_tilted = inject_tilted_body(xyz, sensor, (15.0, 0.2, 0.5), np.array([0.0, 0.0, 1.0]), 0.3, 1.0)
    assert n_plain > 0
    assert n_tilted == n_plain
    assert np.allclose(tilted, plain)


def test_tilted_body_hits_lie_on_its_surface_and_inside_its_box():
    sensor, xyz = _fan(np.random.default_rng(1))
    direction = np.array([0.0, np.sin(np.radians(45.0)), np.cos(np.radians(45.0))])
    out, n = inject_tilted_body(xyz, sensor, (12.0, -0.3, 1.0), direction, 0.1, 1.0)
    moved = np.any(out != xyz, axis=1)
    assert n == int(moved.sum()) > 0
    rel = out[moved] - np.array([12.0, -0.3, 1.0])
    along = rel @ direction
    across = np.linalg.norm(rel - along[:, None] * direction, axis=1)
    assert np.all((along >= -1e-9) & (along <= 1.0 + 1e-9))
    assert np.allclose(across, 0.1)
    lo, hi = body_box(((12.0, -0.3), 1.0, 0.1, 1.0), 45.0)
    assert np.all(out[moved] >= lo - 1e-9) and np.all(out[moved] <= hi + 1e-9)


def test_body_box_without_tilt_is_the_vertical_cylinder_box():
    lo, hi = body_box(((10.0, 0.5), 0.23, 0.15, 0.1), 0.0)
    assert lo.tolist() == [9.85, 0.35, 0.23]
    assert hi.tolist() == [10.15, 0.65, 0.23 + 0.1]
