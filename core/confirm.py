# Слой C: подтверждение кластеров во времени («K из M») и правило остановки В₂ при потере оси.

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.config import ConfirmConfig, VerdictConfig
from core.detector import PER_CLUSTER_DEBUG_KEYS
from core.types import FrameResult


LATERAL_KEY = "lateral_offset_m"


_ALIVE_STATES = frozenset({"measured", "held"})


@dataclass
class _Track:
    x_m: float
    lateral_m: float
    z_m: float
    first_frame: int
    last_frame: int
    hits: list[int] = field(default_factory=list)
    confirmed_at: int | None = None


    extra_gate_m: float = 0.0

    @property
    def confirmed(self) -> bool:
        return self.confirmed_at is not None


def _state_of(result: FrameResult) -> str:
    axis = result.debug.get("axis")
    return "lost" if not axis else str(axis.get("state", "lost"))


def _observations(result: FrameResult) -> np.ndarray:
    n = len(result.detections)
    if n == 0:
        return np.empty((0, 3))
    lateral = result.debug.get(LATERAL_KEY)
    if lateral is None or len(lateral) != n:
        raise ValueError(
            f"debug['{LATERAL_KEY}']: слой подтверждения сопоставляет кластеры "
            "по смещению от оси, и слой A обязан его выдать по одному числу "
            f"на кластер (получено {None if lateral is None else len(lateral)} "
            f"на {n} кластеров)"
        )
    centroids = np.array([d.centroid_xyz for d in result.detections], dtype=float)
    return np.column_stack([centroids[:, 0], np.asarray(lateral, dtype=float), centroids[:, 2]])


class Confirmer:
    def __init__(self, cfg: ConfirmConfig) -> None:
        self._cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._tracks: list[_Track] = []
        self._frame = -1
        self.n_confirmed_tracks = 0

        self.latencies: list[int] = []

        self.residuals: list[tuple[float, float, float]] = []


    def _gates(self, x: np.ndarray, advance_known: bool) -> np.ndarray:
        cfg = self._cfg
        along = cfg.gate_along_m + cfg.gate_along_per_m * np.abs(x)
        if not advance_known:
            along = along + cfg.unknown_advance_extra_m
        return along

    def _match(self, observations: np.ndarray, advance_m: float | None
               ) -> tuple[dict[int, int], np.ndarray]:
        cfg = self._cfg
        n_obs = observations.shape[0]


        for track in self._tracks:
            if advance_m is None:
                track.extra_gate_m += cfg.unknown_advance_extra_m
            else:
                track.x_m -= float(advance_m)
        gates = self._gates(observations[:, 0], advance_m is not None)
        if not self._tracks or n_obs == 0:
            return {}, gates

        predicted = np.array([[t.x_m, t.lateral_m, t.z_m] for t in self._tracks], dtype=float)


        along_gates = (self._gates(observations[:, 0], True)[None, :]
                       + np.array([t.extra_gate_m for t in self._tracks])[:, None])

        d_along = np.abs(observations[None, :, 0] - predicted[:, None, 0])
        d_lat = np.abs(observations[None, :, 1] - predicted[:, None, 1])
        d_vert = np.abs(observations[None, :, 2] - predicted[:, None, 2])
        cost = np.maximum(
            d_along / along_gates,
            np.maximum(d_lat / cfg.gate_lateral_m, d_vert / cfg.gate_vertical_m),
        )
        cost[cost > 1.0] = np.inf

        assigned: dict[int, int] = {}
        taken_tracks: set[int] = set()
        order = np.argsort(cost, axis=None)
        for flat in order:
            track_index, obs_index = divmod(int(flat), n_obs)
            if not np.isfinite(cost[track_index, obs_index]):
                break
            if track_index in taken_tracks or obs_index in assigned:
                continue
            assigned[obs_index] = track_index
            taken_tracks.add(track_index)
            self.residuals.append((
                float(d_along[track_index, obs_index]),
                float(d_lat[track_index, obs_index]),
                float(d_vert[track_index, obs_index]),
            ))
        return assigned, gates


    def update(self, result: FrameResult, *, advance_m: float | None = None) -> FrameResult:
        cfg = self._cfg
        self._frame += 1
        observations = _observations(result)
        assigned, gates = self._match(observations, advance_m)

        confirmed_flags = [False] * observations.shape[0]
        hit_counts = [0] * observations.shape[0]
        for obs_index in range(observations.shape[0]):
            x, lateral, z = observations[obs_index]
            track = self._tracks[assigned[obs_index]] if obs_index in assigned else None
            if track is None:
                track = _Track(x, lateral, z, self._frame, self._frame)
                self._tracks.append(track)
            else:
                track.x_m, track.lateral_m, track.z_m = float(x), float(lateral), float(z)
                track.last_frame = self._frame
                track.extra_gate_m = 0.0
            track.hits.append(self._frame)
            track.hits = [f for f in track.hits if self._frame - f < cfg.window_frames]
            if not track.confirmed and len(track.hits) >= cfg.required_frames:
                track.confirmed_at = self._frame
                self.n_confirmed_tracks += 1
                self.latencies.append(self._frame - track.first_frame)
            confirmed_flags[obs_index] = track.confirmed
            hit_counts[obs_index] = len(track.hits)

        self._tracks = [t for t in self._tracks
                        if self._frame - t.last_frame <= cfg.max_missed_frames]

        detections = [d for d, ok in zip(result.detections, confirmed_flags) if ok]
        debug = dict(result.debug)


        for key in PER_CLUSTER_DEBUG_KEYS:
            values = result.debug.get(key)
            if values is not None and len(values) == len(confirmed_flags):
                debug[key] = [v for v, ok in zip(values, confirmed_flags) if ok]
        debug["confirm"] = {
            "required_frames": cfg.required_frames,
            "window_frames": cfg.window_frames,
            "advance_m": None if advance_m is None else round(float(advance_m), 3),
            "n_candidates": len(result.detections),
            "n_confirmed": len(detections),
            "n_tracks": len(self._tracks),
            "gate_along_m": [round(float(g), 3) for g in gates],


            "candidate_lateral_offset_m": list(result.debug.get(LATERAL_KEY, [])),
            "candidates": [
                {
                    "distance_m": round(d.distance_m, 2),
                    "hits": hits,
                    "confirmed": bool(ok),
                }
                for d, hits, ok in zip(result.detections, hit_counts, confirmed_flags)
            ],
        }
        nearest = min((d.distance_m for d in detections), default=None)
        return FrameResult(bool(detections), nearest, detections, debug)


class AxisLossVerdict:
    def __init__(self, cfg: VerdictConfig) -> None:
        self._cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.n_triggers = 0
        self._seen_measured = False
        self._in_failure = False
        self._stopped = False
        self._since_measured: list[bool] = []

    @property
    def stopped(self) -> bool:
        return self._stopped

    def update(self, result: FrameResult, *, had_candidate: bool | None = None) -> dict:
        state = _state_of(result)
        if had_candidate is None:
            candidates = result.debug.get("confirm", {}).get("n_candidates")
            had_candidate = bool(candidates) if candidates is not None else bool(result.detections)
        verdict = self.feed(state, bool(had_candidate))
        result.debug["verdict"] = verdict
        return verdict

    def feed(self, state: str, had_candidate: bool) -> dict:
        if state in _ALIVE_STATES:
            self._in_failure = False
            self._stopped = False
            if state == "measured":
                self._seen_measured = True
                self._since_measured = []
            self._since_measured.append(had_candidate)
            return {"rule": "В₂", "stop": False, "reason": None, "axis_state": state}

        if not self._in_failure:
            self._in_failure = True
            window = self._since_measured[-self._cfg.memory_frames:]
            if self._seen_measured and any(window):
                self._stopped = True
                self.n_triggers += 1
        reason = None
        if self._stopped:
            reason = "axis_lost_after_clusters"
        elif not self._seen_measured:
            reason = "cold_start"
        else:
            reason = "axis_lost_without_clusters"
        return {"rule": "В₂", "stop": self._stopped, "reason": reason, "axis_state": state}

