# Синтетические сцены тоннеля с известным ответом для тестов.

from __future__ import annotations

import numpy as np


REP103_FROM_SOURCE = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


SENSOR_EPOCH_S = 946687297.2


SHOT_PERIOD_S = 1.0 / 36000.0


def to_source_frame(xyz_rep103: np.ndarray) -> np.ndarray:
    return xyz_rep103 @ REP103_FROM_SOURCE


def tunnel_rep103(
    x_min: float = 2.0,
    x_max: float = 60.0,
    half_width: float = 4.0,
    wall_height: float = 5.0,
    n_x: int = 120,
    n_y: int = 30,
    n_z: int = 20,
    n_theta: int = 40,
    noise_m: float = 0.0,
    seed: int = 4242,
) -> np.ndarray:
    x = np.linspace(x_min, x_max, n_x)

    y = np.linspace(-half_width, half_width, n_y)
    gx, gy = np.meshgrid(x, y, indexing="ij")
    floor = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)

    z = np.linspace(0.0, wall_height, n_z)
    gx, gz = np.meshgrid(x, z, indexing="ij")
    left = np.stack([gx.ravel(), np.full(gx.size, half_width), gz.ravel()], axis=1)
    right = np.stack([gx.ravel(), np.full(gx.size, -half_width), gz.ravel()], axis=1)

    theta = np.linspace(0.0, np.pi, n_theta)
    gx, gt = np.meshgrid(x, theta, indexing="ij")
    vault = np.stack(
        [
            gx.ravel(),
            (half_width * np.cos(gt)).ravel(),
            (wall_height + half_width * np.sin(gt)).ravel(),
        ],
        axis=1,
    )
    cloud = np.concatenate([floor, left, right, vault], axis=0)
    if noise_m > 0.0:
        cloud = cloud + np.random.default_rng(seed).normal(0.0, noise_m, cloud.shape)
    return cloud


def circular_section_rep103(
    radius: float = 3.0,
    chord_z: float = -2.0,
    x_min: float = 5.0,
    x_max: float = 15.0,
    n_x: int = 60,
    n_theta: int = 120,
    n_chord: int = 60,
) -> np.ndarray:
    x = np.linspace(x_min, x_max, n_x)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    y_arc, z_arc = radius * np.cos(theta), radius * np.sin(theta)
    on_arc = z_arc > chord_z
    gx, gy = np.meshgrid(x, y_arc[on_arc], indexing="ij")
    _, gz = np.meshgrid(x, z_arc[on_arc], indexing="ij")
    arc = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    half_chord = float(np.sqrt(max(radius * radius - chord_z * chord_z, 0.0)))
    y_chord = np.linspace(-half_chord, half_chord, n_chord)
    gx, gy = np.meshgrid(x, y_chord, indexing="ij")
    chord = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, chord_z)], axis=1)
    return np.concatenate([arc, chord], axis=0)


def source_frame(
    scene_rep103: np.ndarray | None = None,
    n_rings: int = 8,
    n_times: int = 800,
    empty_fraction: float = 0.25,
    secondary_scale: float = 1.2,
    seed: int = 12345,
    layout: str = "blocks",
) -> dict:
    scene = tunnel_rep103() if scene_rep103 is None else scene_rep103
    n_shots = n_rings * n_times

    picked = scene[(np.arange(n_shots) * scene.shape[0]) // n_shots]

    near = to_source_frame(picked)
    far = near * secondary_scale

    shot = np.arange(n_shots)
    column, ring_of_shot = shot // n_rings, shot % n_rings
    near_slot = shot % 2

    if layout in ("blocks", "shuffled"):
        base = column * (2 * n_rings) + ring_of_shot
        index_near = base + near_slot * n_rings
        index_far = base + (1 - near_slot) * n_rings
        ring = np.tile(np.arange(n_rings, dtype=np.uint16), 2 * n_times)
        times = SENSOR_EPOCH_S + np.arange(n_times) * SHOT_PERIOD_S
        timestamp = np.repeat(times, 2 * n_rings)
    elif layout == "interleaved":
        index_near = 2 * shot + near_slot
        index_far = 2 * shot + (1 - near_slot)
        ring = np.repeat(ring_of_shot.astype(np.uint16), 2)
        timestamp = np.repeat(SENSOR_EPOCH_S + column * SHOT_PERIOD_S, 2)
    else:
        raise ValueError(f"неизвестная раскладка {layout!r}")

    xyz = np.zeros((2 * n_shots, 3), dtype=np.float32)
    xyz[index_near] = near
    xyz[index_far] = far
    intensity = np.zeros(2 * n_shots, dtype=np.float32)
    intensity[index_near] = 120.0
    intensity[index_far] = 40.0

    generator = np.random.default_rng(seed)
    n_empty = int(round(empty_fraction * n_shots))
    empty_shots = generator.choice(n_shots, size=n_empty, replace=False)
    xyz[index_near[empty_shots]] = 0.0
    xyz[index_far[empty_shots]] = 0.0

    if layout == "shuffled":
        permutation = generator.permutation(xyz.shape[0])
        xyz, intensity = xyz[permutation], intensity[permutation]
        ring, timestamp = ring[permutation], timestamp[permutation]

    return {
        "xyz": xyz,
        "intensity": intensity,
        "ring": ring,
        "timestamp": timestamp,
        "n_shots": n_shots,
        "n_empty_shots": n_empty,
        "n_live_shots": n_shots - n_empty,
        "near_rep103": picked,
        "pair_stride": n_rings if layout == "blocks" else (1 if layout == "interleaved" else None),
    }


def station_rep103(
    platform_height: float = 1.0,
    platform_width: float = 2.5,
    density_ratio: float = 1.0,
    noise_m: float = 0.01,
    seed: int = 909,
) -> np.ndarray:
    tunnel = tunnel_rep103(noise_m=noise_m, seed=seed)
    x = np.linspace(5.0, 40.0, int(120 * density_ratio))
    y = np.linspace(-platform_width - 4.0, -4.0, int(30 * density_ratio))
    gx, gy = np.meshgrid(x, y, indexing="ij")
    platform = np.stack(
        [gx.ravel(), gy.ravel(), np.full(gx.size, platform_height)], axis=1
    )
    platform = platform + np.random.default_rng(seed).normal(0.0, noise_m, platform.shape)
    return np.concatenate([tunnel, platform], axis=0)


def spurious_low_plane_rep103(depth: float = 0.6, n_points: int = 400, seed: int = 77) -> np.ndarray:
    tunnel = tunnel_rep103(noise_m=0.01, seed=seed)
    generator = np.random.default_rng(seed)
    x = generator.uniform(6.0, 38.0, n_points)
    y = generator.uniform(-0.4, 0.4, n_points)
    trough = np.stack([x, y, np.full(n_points, -depth)], axis=1)
    return np.concatenate([tunnel, trough], axis=0)


def landmark_tunnel(
    travelled_m: float = 0.0,
    length_m: float = 90.0,
    half_width: float = 2.2,
    wall_height: float = 4.0,
    n_x: int = 900,
    n_ring: int = 60,
    n_landmarks: int = 22,
    smooth: bool = False,
    seed: int = 31337,
) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    world_x = np.linspace(0.0, length_m, n_x)
    angle = np.linspace(0.0, np.pi, n_ring)

    gx, ga = np.meshgrid(world_x, angle, indexing="ij")
    radius = np.full(gx.shape, half_width)
    intensity = np.full(gx.shape, 60.0)

    if not smooth:
        positions = np.sort(generator.uniform(2.0, length_m - 2.0, n_landmarks))
        for position, kind in zip(positions, generator.integers(0, 2, n_landmarks)):
            hit = np.abs(world_x - position) < 0.12
            if kind == 0:
                radius[hit, 10:22] -= 0.35
            else:
                intensity[hit, 26:34] = 250.0

    points = np.stack(
        [
            (gx - travelled_m).ravel(),
            (radius * np.cos(ga)).ravel(),
            (wall_height * 0.5 + radius * np.sin(ga)).ravel(),
        ],
        axis=1,
    )
    values = intensity.ravel()
    visible = points[:, 0] > 0.0
    return points[visible], values[visible]


def track_rep103(
    yaw_deg: float = 0.0,
    y0: float = 0.0,
    radius_m: float | None = None,
    x_min: float = 5.0,
    x_max: float = 60.0,
    bed_half_width: float = 1.95,
    wall_offset: float | None = None,
    gauge: float = 1.52,
    rail_head: float = 0.23,
    wall_height: float = 4.0,
    second_track_at: float | None = None,
    n_x: int = 400,
    n_lat: int = 40,
    n_z: int = 30,
) -> np.ndarray:
    slope = np.tan(np.radians(yaw_deg))
    x = np.linspace(x_min, x_max, n_x)
    axis_y = slope * x + y0
    if radius_m is not None:
        axis_y = axis_y + x * x / (2.0 * radius_m)

    def strip(offsets: np.ndarray, z: float) -> np.ndarray:
        gx, go = np.meshgrid(x, offsets, indexing="ij")
        ga = np.repeat(axis_y[:, None], offsets.size, axis=1)
        return np.stack([gx.ravel(), (ga + go).ravel(), np.full(gx.size, z)], axis=1)

    outer = bed_half_width if second_track_at is None else second_track_at + bed_half_width


    wall = outer + 0.65 if wall_offset is None else wall_offset
    bed = strip(np.linspace(-bed_half_width, outer, n_lat), 0.0)
    rails = np.concatenate([
        strip(np.full(6, -0.5 * gauge) + np.linspace(-0.02, 0.02, 6), rail_head),
        strip(np.full(6, +0.5 * gauge) + np.linspace(-0.02, 0.02, 6), rail_head),
    ])
    parts = [bed, rails]
    if second_track_at is not None:
        parts.append(np.concatenate([
            strip(np.full(6, second_track_at - 0.5 * gauge) + np.linspace(-0.02, 0.02, 6), rail_head),
            strip(np.full(6, second_track_at + 0.5 * gauge) + np.linspace(-0.02, 0.02, 6), rail_head),
        ]))

    z = np.linspace(0.05, wall_height, n_z)
    gx, gz = np.meshgrid(x, z, indexing="ij")
    ga = np.repeat(axis_y[:, None], z.size, axis=1)
    parts.append(np.stack([gx.ravel(), (ga - wall).ravel(), gz.ravel()], axis=1))
    parts.append(np.stack([gx.ravel(), (ga + wall).ravel(), gz.ravel()], axis=1))
    return np.concatenate(parts, axis=0)


def box_rep103(
    centre: tuple[float, float, float],
    size: tuple[float, float, float] = (0.5, 0.5, 1.7),
    n: int = 12,
) -> np.ndarray:
    axes = [np.linspace(c - 0.5 * s, c + 0.5 * s, n) for c, s in zip(centre, size)]
    grid = np.meshgrid(*axes, indexing="ij")
    return np.stack([g.ravel() for g in grid], axis=1)

