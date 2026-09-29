# ObstacleDetector: сборка всех этапов ядра в один вызов process() на кадр — точка входа для ноды.

from __future__ import annotations

import time

import numpy as np

from core.axis import AxisTracker, rail_head_lines
from core.confirm import LATERAL_KEY, AxisLossVerdict, Confirmer
from core.config import Config
from core.detector import detect, drop_shallow
from core.odometry import estimate_shift, longitudinal_profile
from core.preprocess import GroundTracker, PreprocessError, preprocess_frame
from core.types import FrameResult
from core.walls import extend_by_walls


class ObstacleDetector:
    def __init__(self, config: Config, *, confirm: bool | None = None) -> None:
        self._cfg = config
        self._ground = GroundTracker(config.preprocess.ground, config.preprocess.plane_guard, config.axis)
        self._axis = AxisTracker(config.axis)


        enabled = config.confirm.enabled if confirm is None else confirm
        self._confirmer = Confirmer(config.confirm) if enabled else None
        self._verdict = AxisLossVerdict(config.confirm.verdict)
        self.reset()

    def reset(self) -> None:
        self._ground.reset()
        self._axis.reset()
        if self._confirmer is not None:
            self._confirmer.reset()
        self._verdict.reset()
        self._previous_profile: np.ndarray | None = None
        self._previous_stamp_ns: int | None = None
        self._pending_gap = 0


    @property
    def axis_tracker(self) -> AxisTracker:
        return self._axis

    @property
    def confirmer(self) -> Confirmer | None:
        return self._confirmer

    @property
    def verdict(self) -> AxisLossVerdict:
        return self._verdict


    def process(self, xyz: np.ndarray, intensity: np.ndarray, ring: np.ndarray | None,
                t_rel: np.ndarray | None, stamp_ns: int, *, frame_gap: int = 1) -> FrameResult:
        marks = [time.perf_counter()]
        try:
            prepared = preprocess_frame(
                xyz, intensity, ring, t_rel, self._cfg.preprocess, tracker=self._ground,
                frame_gap=frame_gap + self._pending_gap)
        except PreprocessError as error:
            self._pending_gap += frame_gap
            return self._unchecked(str(error), marks[0])
        marks.append(time.perf_counter())
        frame_gap += self._pending_gap
        self._pending_gap = 0

        advance_m = self._advance(prepared.xyz, prepared.intensity, stamp_ns)
        self._ground.note_advance(advance_m, frame_gap)
        marks.append(time.perf_counter())

        axis = self._axis.update(prepared.xyz, frame_gap=frame_gap, advance_m=advance_m)
        if self._cfg.axis.walls.enabled:
            axis = extend_by_walls(axis, prepared.xyz, self._cfg.axis.walls, self._cfg.gauge.rail_head_offset_m)
        marks.append(time.perf_counter())

        heads = (rail_head_lines(prepared.xyz, axis, self._cfg.axis)
                 if axis is not None and self._cfg.gauge.rail_mask_follow_heads else None)
        outcome = detect(prepared.xyz, axis, self._cfg.gauge, self._cfg.detector, heads=heads)
        marks.append(time.perf_counter())


        outcome.debug["frame_transform"] = prepared.transform.to_debug()
        if self._ground.guarded:
            outcome.debug["plane_guard"] = self._ground.last_guard
        outcome.debug["dual_return"] = {
            "pairing": prepared.stats["dual_return_pairing"],
            "n_unpaired": prepared.stats["n_unpaired"],
        }


        if outcome.debug["axis"] is None:
            outcome.debug["axis"] = {"state": "lost"}
        outcome.debug["axis"]["reason"] = self._axis.last_cause
        outcome.debug["axis"]["hold_expired"] = self._axis.last_hold_expired


        candidates = {
            "detections": [d.to_dict() for d in outcome.detections],
            LATERAL_KEY: list(outcome.debug.get(LATERAL_KEY, [])),
            "depth_m": list(outcome.debug.get("depth_m", [])),
            "min_depth_m": self._cfg.detector.min_depth_m,
        }
        had_candidate = bool(outcome.detections)
        outcome = drop_shallow(outcome, self._cfg.detector.min_depth_m)
        if self._confirmer is not None:
            outcome = self._confirmer.update(outcome, advance_m=advance_m)
        self._verdict.update(outcome, had_candidate=had_candidate)
        marks.append(time.perf_counter())

        outcome.debug["advance_m"] = None if advance_m is None else round(advance_m, 3)
        outcome.debug["candidates_raw"] = candidates
        outcome.debug["timings_ms"] = {
            name: round(1e3 * (b - a), 3)
            for name, a, b in zip(
                ("preprocess", "odometry", "axis", "detect", "confirm"),
                marks[:-1], marks[1:])
        }
        outcome.debug["timings_ms"]["total"] = round(1e3 * (marks[-1] - marks[0]), 3)
        outcome.debug["checked"] = True
        outcome.debug["frame_gap"] = int(frame_gap)
        return outcome

    def _unchecked(self, reason: str, started: float) -> FrameResult:
        elapsed = round(1e3 * (time.perf_counter() - started), 3)
        debug = {
            "layer": "A",
            "checked": False,
            "reason": "frame_unusable",
            "detail": reason,
            "pending_gap_frames": self._pending_gap,
            "timings_ms": {"total": elapsed},
        }
        if self._ground.guarded:
            debug["plane_guard"] = self._ground.last_guard
        if self._cfg.confirm.verdict.latch_through_unchecked and self._verdict.stopped:
            debug["verdict"] = {"rule": "В₂", "stop": True, "reason": "latched_through_unchecked",
                                "axis_state": "unchecked"}
        return FrameResult(False, None, [], debug)

    def _advance(self, xyz: np.ndarray, intensity: np.ndarray,
                 stamp_ns: int) -> float | None:
        profile = longitudinal_profile(xyz, intensity, self._cfg.odometry)
        advance_m = None
        if self._previous_profile is not None and self._previous_stamp_ns is not None:
            dt_s = (stamp_ns - self._previous_stamp_ns) / 1e9
            if dt_s > 0.0:
                shift = estimate_shift(
                    self._previous_profile, profile, dt_s, self._cfg.odometry)
                advance_m = None if shift.shift_m is None else abs(shift.shift_m)
        self._previous_profile, self._previous_stamp_ns = profile, stamp_ns
        return advance_m

