"""Per-track, per-zone state machine: OUTSIDE/INSIDE/UNKNOWN, entry/exit
debounce, dwell tracking. Deliberately has zero cv2/ultralytics/torch
dependency (pure Python) so it's unit-testable without real video or CV
models - same philosophy as core.target_lock.

One ZoneMonitor instance owns exactly one zone's state for every track
independently (a dict keyed by track_id), so multiple simultaneous objects
never share or clobber each other's state, and multiple zones are just
multiple ZoneMonitor instances (each carries its own zone_id).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from math import hypot

from backend.modules.zone.geometry import (
    distance_to_polygon, point_in_polygon, predicted_entry_eta, velocity_from,
)

UNKNOWN, OUTSIDE, INSIDE = "UNKNOWN", "OUTSIDE", "INSIDE"


@dataclass
class ZoneEvent:
    type: str  # ZONE_APPROACH | ZONE_ENTRY | ZONE_EXIT | LONG_DWELL
    zone_id: str
    track_id: int
    frame_index: int
    timestamp_sec: float
    data: dict = field(default_factory=dict)


@dataclass
class _TrackState:
    state: str = UNKNOWN
    pending_inside: int = 0
    pending_outside: int = 0
    absence_count: int = 0
    entry_ts: float | None = None
    dwell_emitted: bool = False
    # Recent (timestamp, point) samples, used to estimate heading for the
    # approach prediction. Bounded by elapsed time rather than sample count,
    # so it spans the same amount of real motion at any frame rate.
    history: list[tuple[float, tuple[float, float]]] = field(default_factory=list)
    approach_pending: int = 0
    approach_warned_ts: float | None = None


class ZoneMonitor:
    def __init__(
        self,
        zone_id: str,
        entry_grace_frames: int = 3,
        exit_grace_frames: int = 15,
        track_absence_grace_frames: int = 45,
        dwell_threshold_sec: float = 15.0,
        approach_prediction_sec: float = 2.5,
        approach_grace_frames: int = 2,
        approach_min_speed_px_per_sec: float = 12.0,
        approach_velocity_window_sec: float = 1.5,
        approach_cooldown_sec: float = 20.0,
    ):
        self.zone_id = zone_id
        self.entry_grace_frames = max(1, entry_grace_frames)
        self.exit_grace_frames = max(1, exit_grace_frames)
        self.track_absence_grace_frames = max(1, track_absence_grace_frames)
        self.dwell_threshold_sec = dwell_threshold_sec
        # A horizon of 0 turns prediction off entirely, which is the right
        # behaviour for a camera where approach warnings are unwanted.
        self.approach_prediction_sec = max(0.0, approach_prediction_sec)
        self.approach_grace_frames = max(1, approach_grace_frames)
        self.approach_min_speed_px_per_sec = max(0.0, approach_min_speed_px_per_sec)
        self.approach_velocity_window_sec = max(0.1, approach_velocity_window_sec)
        self.approach_cooldown_sec = max(0.0, approach_cooldown_sec)
        self._tracks: dict[int, _TrackState] = {}
        self.events: list[ZoneEvent] = []

    def state_of(self, track_id: int) -> str:
        st = self._tracks.get(track_id)
        return st.state if st else UNKNOWN

    def update(
        self,
        active_points: dict[int, tuple[float, float]],
        frame_index: int,
        timestamp_sec: float,
        polygon: list[tuple[float, float]],
    ) -> None:
        """Advance one frame. active_points maps track_id -> the point
        (footpoint/centroid, in the SAME coordinate space as polygon) to test
        for every track currently detected this frame. Tracks known from a
        previous frame but absent from active_points are treated as a
        detection gap, not an instant exit.
        """
        seen = set(active_points.keys())

        for tid, st in list(self._tracks.items()):
            if tid in seen:
                continue
            st.absence_count += 1
            if st.state == INSIDE and st.absence_count > self.track_absence_grace_frames:
                self._commit_exit(tid, st, frame_index, timestamp_sec)
            elif st.absence_count > self.track_absence_grace_frames * 4:
                del self._tracks[tid]  # bound memory for long-gone tracks

        for tid, point in active_points.items():
            st = self._tracks.setdefault(tid, _TrackState())
            st.absence_count = 0
            inside = point_in_polygon(point, polygon)

            if st.state != INSIDE:
                if inside:
                    st.pending_inside += 1
                    st.pending_outside = 0
                    if st.pending_inside >= self.entry_grace_frames:
                        self._commit_entry(tid, st, frame_index, timestamp_sec)
                else:
                    st.pending_inside = 0
                    if st.state == UNKNOWN:
                        st.state = OUTSIDE  # safe immediate default, no event
                    self._check_approach(tid, st, point, frame_index, timestamp_sec, polygon)
            else:
                if inside:
                    st.pending_outside = 0
                else:
                    st.pending_outside += 1
                    if st.pending_outside >= self.exit_grace_frames:
                        self._commit_exit(tid, st, frame_index, timestamp_sec)
                        continue
                if st.state == INSIDE and not st.dwell_emitted and st.entry_ts is not None:
                    dwell = timestamp_sec - st.entry_ts
                    if dwell >= self.dwell_threshold_sec:
                        st.dwell_emitted = True
                        self.events.append(ZoneEvent(
                            "LONG_DWELL", self.zone_id, tid, frame_index, timestamp_sec,
                            {"duration_sec": round(dwell, 3)},
                        ))

    def _check_approach(self, tid: int, st: _TrackState, point: tuple[float, float],
                        frame_index: int, ts: float,
                        polygon: list[tuple[float, float]]) -> None:
        """Warn when this track's recent heading points into the zone.

        Only ever raised for a track that is currently outside: once it is in,
        the crossing alarm is the truth and a prediction about it is noise.
        """
        if self.approach_prediction_sec <= 0:
            return

        # A long enough gap means this track was absent, and where it stood
        # before that says nothing about where it is heading now - so the
        # baseline starts again rather than measuring across the gap. The
        # threshold is deliberately longer than the window itself: at live
        # frame rates one ordinary frame interval can approach the window's
        # length, and treating that as an absence would clear the history on
        # every frame and leave the heading permanently unmeasurable.
        absence_gap = self.approach_velocity_window_sec * 2.0
        if st.history and ts - st.history[-1][0] > absence_gap:
            st.history.clear()
        st.history.append((ts, point))
        while len(st.history) > 2 and ts - st.history[0][0] > self.approach_velocity_window_sec:
            st.history.pop(0)

        velocity = velocity_from(st.history)
        speed = hypot(*velocity) if velocity else 0.0
        eta = None
        if speed >= self.approach_min_speed_px_per_sec:
            eta = predicted_entry_eta(point, velocity, polygon, self.approach_prediction_sec)

        if eta is None:
            st.approach_pending = 0
            return

        st.approach_pending += 1
        if st.approach_pending < self.approach_grace_frames:
            return
        if (st.approach_warned_ts is not None
                and ts - st.approach_warned_ts < self.approach_cooldown_sec):
            return

        st.approach_warned_ts = ts
        st.approach_pending = 0
        self.events.append(ZoneEvent(
            "ZONE_APPROACH", self.zone_id, tid, frame_index, ts,
            {"eta_sec": eta, "speed_px_per_sec": round(speed, 1)},
        ))

    def approaching(self, track_id: int) -> bool:
        """Whether this track is under a live approach warning - lets the UI
        show a zone's status without replaying its event history."""
        st = self._tracks.get(track_id)
        if st is None or st.approach_warned_ts is None or st.state == INSIDE:
            return False
        latest = st.history[-1][0] if st.history else st.approach_warned_ts
        return latest - st.approach_warned_ts <= self.approach_prediction_sec

    def _commit_entry(self, tid: int, st: _TrackState, frame_index: int, ts: float) -> None:
        st.state = INSIDE
        # The prediction has been overtaken by events. Clearing it means
        # leaving and returning counts as a fresh approach, not a stale one.
        st.approach_pending = 0
        st.history.clear()
        st.entry_ts = ts
        st.dwell_emitted = False
        st.pending_inside = 0
        self.events.append(ZoneEvent("ZONE_ENTRY", self.zone_id, tid, frame_index, ts, {}))

    def _commit_exit(self, tid: int, st: _TrackState, frame_index: int, ts: float) -> None:
        dwell = (ts - st.entry_ts) if st.entry_ts is not None else 0.0
        st.state = OUTSIDE
        st.pending_outside = 0
        st.absence_count = 0
        st.entry_ts = None
        st.approach_pending = 0
        st.history.clear()
        self.events.append(ZoneEvent(
            "ZONE_EXIT", self.zone_id, tid, frame_index, ts,
            {"dwell_duration_sec": round(dwell, 3)},
        ))


def anyone_near_zone(points: dict[int, tuple[float, float]], monitors: list[ZoneMonitor],
                     polygons_px: list, near_px: float) -> bool:
    """Whether any tracked point is inside a zone, under its live approach
    warning, or within near_px of its edge - what the adaptive detection
    stride treats as worth detecting on every frame."""
    for tid, point in points.items():
        for monitor, polygon in zip(monitors, polygons_px):
            if monitor.state_of(tid) == INSIDE or monitor.approaching(tid):
                return True
            if distance_to_polygon(point, polygon) <= near_px:
                return True
    return False
