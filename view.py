#!/usr/bin/env python3
# 3D-просмотр записей и выхода ядра поверх исходного облака в open3d.

import sys
import os

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("GALLIUM_DRIVER", "llvmpipe")
os.environ["WAYLAND_DISPLAY"] = "no-wayland"
os.environ.setdefault("DISPLAY", ":0")
import numpy as np
import open3d as o3d
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


DETECT = len(sys.argv) > 1 and sys.argv[1] == "--detect"
FILES = len(sys.argv) > 1 and sys.argv[1] == "--files"
if FILES:
    if len(sys.argv) != 3:
        raise SystemExit("view.py --files <каталог с frame_NNNN.ply>")
    BAG = Path(sys.argv[2])
    N = 0
elif DETECT:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from core.config import Config
    from tools.view_geometry import load as load_with_detections
    from tools.visualize_run import load_viz

    if len(sys.argv) != 5:
        raise SystemExit("view.py --detect <запись> <первый кадр> <последний кадр>")
    RECORD, FIRST, LAST = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    VIZ = load_viz(Path(__file__).resolve().parent / "configs" / "viz.yaml")
    V3 = VIZ["view3d"]
    CFG = Config.from_yaml(Path(__file__).resolve().parent / VIZ["core_config"])

    BAG = Path(RECORD) if Path(RECORD).is_dir() else CFG.data.path(RECORD)
    N = LAST - FIRST + 1
else:
    BAG = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "/home/denimele/hackathon/for_hackathon/doubleT_obstacle"
    )
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 30

DT = np.dtype(
    {
        "names": ["x", "y", "z", "intensity", "ring", "timestamp"],
        "formats": ["<f4", "<f4", "<f4", "<f4", "<u2", "<f8"],
        "offsets": [0, 4, 8, 12, 16, 18],
        "itemsize": 26,
    }
)

frames = []
views = []
file_lines = []
file_info = []
ts = get_typestore(Stores.ROS2_HUMBLE)
if FILES:
    import json as _json
    info = {}
    if (BAG / "index.json").is_file():
        info = {r["frame"]: r for r in _json.loads((BAG / "index.json").read_text(encoding="utf-8"))}
    for path in sorted(p for p in BAG.glob("frame_*.ply") if not p.stem.endswith("_lines")):
        cloud = o3d.io.read_point_cloud(str(path))
        frames.append((np.asarray(cloud.points), None, np.asarray(cloud.colors)))
        lines_path = path.with_name(path.stem + "_lines.ply")
        file_lines.append(o3d.io.read_line_set(str(lines_path)) if lines_path.is_file() else None)
        number = int(path.stem.split("_")[1])
        file_info.append((number, info.get(number)))
        print(f"\rзагрузка {len(frames)}", end="", flush=True)
elif DETECT:
    views = load_with_detections(CFG, RECORD, FIRST, LAST, V3["point_stride"],
                                 {k: tuple(v) for k, v in V3["line_colors"].items()})
    frames = [(v.points, v.intensity / 255.0) for v in views]
else:
    with AnyReader([BAG], default_typestore=ts) as r:
        for i, (c, t, raw) in enumerate(r.messages(connections=list(r.connections))):
            if i >= N:
                break
            m = r.deserialize(raw, c.msgtype)
            a = np.frombuffer(m.data, dtype=DT, count=m.width * m.height)
            xyz = np.stack([a["x"], a["y"], a["z"]], 1)
            keep = np.linalg.norm(xyz, axis=1) > 1e-6
            frames.append((xyz[keep], a["intensity"][keep] / 255.0))
            print(f"\rзагрузка {i + 1}/{N}", end="", flush=True)
print(f"\nготово: {len(frames)} кадров")
if not frames:
    raise SystemExit(f"{BAG}: нечего показывать — кадров не найдено")

idx = [0]
pcd = o3d.geometry.PointCloud()
overlays = []


def _line_sets(i):
    if FILES:
        return [file_lines[i]] if file_lines[i] is not None else []
    out = []
    for group in views[i].lines if views else []:
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(group.points),
            lines=o3d.utility.Vector2iVector(group.edges.astype(np.int32)))
        ls.paint_uniform_color(group.color)
        out.append(ls)
    return out


def show(i):
    if FILES:
        xyz, _, rgb = frames[i]
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        number, meta = file_info[i]
        if meta:
            print(f"кадр {number}  точек {len(xyz)}  точек фигуры {meta['figure_points']}  "
                  f"ось: {meta['axis_state']}  кандидатов {meta['candidates']}  "
                  f"тревог {meta['alarms']}  ближайшая {meta['nearest_distance_m']}")
        else:
            print(f"кадр {number}  точек {len(xyz)}")
        return
    xyz, inten = frames[i]
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    g = np.clip(inten, 0, 1)
    pcd.colors = o3d.utility.Vector3dVector(np.stack([g, 0.4 + 0.6 * g, 1 - g], 1))
    if views:
        r = views[i].result
        axis = r["debug"].get("axis") or {}
        nearest = r["nearest_distance_m"]
        print(f"кадр {views[i].index}  точек {len(xyz)}  ось: {axis.get('state')}"
              f"{'' if not axis.get('reason') else ' (' + axis['reason'] + ')'}  "
              f"кандидатов {len((r['debug'].get('candidates_raw') or {}).get('detections', []))}  "
              f"тревог {len(r['detections'])}"
              f"{'' if nearest is None else f'  ближайшая {nearest:.2f} м вдоль пути'}")
    else:
        print(f"кадр {i + 1}/{len(frames)}  точек {len(xyz)}")


def _swap(v, i):
    for ls in overlays:
        v.remove_geometry(ls, reset_bounding_box=False)
    overlays[:] = _line_sets(i)
    for ls in overlays:
        v.add_geometry(ls, reset_bounding_box=False)


def nxt(v):
    idx[0] = (idx[0] + 1) % len(frames)
    show(idx[0])
    v.update_geometry(pcd)
    _swap(v, idx[0])
    return False


def prv(v):
    idx[0] = (idx[0] - 1) % len(frames)
    show(idx[0])
    v.update_geometry(pcd)
    _swap(v, idx[0])
    return False


show(0)
vis = o3d.visualization.VisualizerWithKeyCallback()
width, height = (V3["window_size"] if DETECT else (1400, 900))
vis.create_window(window_name=BAG.name, width=width, height=height)
vis.add_geometry(pcd)
overlays[:] = _line_sets(0)
for ls in overlays:
    vis.add_geometry(ls)
vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0))
opt = vis.get_render_option()
if opt is None:
    raise SystemExit("OpenGL не поднялся: окно не создалось")
opt.point_size = V3["point_size"] if DETECT else 1.5
opt.background_color = np.array([0.05, 0.05, 0.08])
vis.register_key_callback(ord("N"), nxt)
vis.register_key_callback(ord("P"), prv)
print("N — следующий кадр, P — предыдущий, мышь — вращение, колесо — зум, Q — выход")
vis.run()
vis.destroy_window()

