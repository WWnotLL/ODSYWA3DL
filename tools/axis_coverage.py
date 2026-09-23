# Покрытие осью по прогонам: состояния оси, холодный старт, доля кадров с продлением по основанию и дальность коридора.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from tools.false_alarm_table import load


def coverage(frames: list[dict]) -> dict:
    rows = sorted((f for f in frames if "debug" in f), key=lambda f: f["frame"])
    states: dict[str, int] = {}
    reasons: dict[str, int] = {}
    cold = 0
    started = False
    extended, ends, ends_rails, ends_ext = 0, [], [], []
    for row in rows:
        debug = row["debug"]
        axis = debug.get("axis") or {}
        state = axis.get("state", "unchecked") if debug.get("checked", False) else "unchecked"
        states[state] = states.get(state, 0) + 1
        if state != "measured" and axis.get("reason"):
            key = f"{state}: {axis['reason']}"
            reasons[key] = reasons.get(key, 0) + 1
        if state == "measured":
            started = True
        elif not started:
            cold += 1
        corridor = debug.get("corridor")
        if not corridor:
            continue
        end = float(corridor["x_m"][1])
        ends.append(end)
        if axis.get("far_method") == "bed" and axis.get("x_joint_m") is not None:
            extended += 1
            ends_ext.append(end)
        else:
            ends_rails.append(end)
    return {"frames": len(rows), "states": states, "reasons": reasons, "cold_start": cold,
            "extended": extended, "with_corridor": len(ends),
            "end_median": float(np.median(ends)) if ends else None,
            "end_rails_median": float(np.median(ends_rails)) if ends_rails else None,
            "ends_ext": ends_ext}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    args = parser.parse_args(argv)
    total_frames = total_ext = 0
    all_ext: list[float] = []
    print("| запись | кадров | measured | held | degraded | lost | не проверен | холодный старт, кадров "
          "| с продлением по основанию | конец коридора, медиана | без продления, медиана |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    details = []
    for path in sorted(args.runs.glob("*.jsonl")):
        c = coverage(load(path)[0])
        s = c["states"]
        total_frames += c["frames"]
        total_ext += c["extended"]
        all_ext += c["ends_ext"]
        fmt = lambda v: "—" if v is None else f"{v:.1f} м"
        print(f"| `{path.stem}` | {c['frames']} | {s.get('measured', 0)} | {s.get('held', 0)} "
              f"| {s.get('degraded', 0)} | {s.get('lost', 0)} | {s.get('unchecked', 0)} "
              f"| {c['cold_start']} | {c['extended']} ({100 * c['extended'] / max(c['frames'], 1):.0f} %) "
              f"| {fmt(c['end_median'])} | {fmt(c['end_rails_median'])} |")
        details.append((path.stem, c["reasons"]))
    print(f"\nкадров с продлением по основанию: {total_ext} из {total_frames} "
          f"({100 * total_ext / max(total_frames, 1):.1f} %)")
    if all_ext:
        e = np.asarray(all_ext)
        print(f"конец коридора в этих кадрах: медиана {np.median(e):.1f} м, "
              f"p10 {np.percentile(e, 10):.1f}, p90 {np.percentile(e, 90):.1f}, max {e.max():.1f} м")
    print("\nпричины, когда ось не измерена:")
    for name, reasons in details:
        if reasons:
            print(f"  {name}: " + ", ".join(f"{k} — {v}" for k, v in sorted(reasons.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
