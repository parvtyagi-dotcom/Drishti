"""Directional line crossing - a tripwire that cares which way you went.

A zone answers "is this person inside an area". A border needs a different
question: "did this person cross the line, and in which direction". Those are
not the same test, and the difference is operationally decisive - someone
walking *out* of the country and someone walking *in* produce identical zone
events but mean opposite things.

Pure geometry and state, with no cv2/torch dependency, so the crossing rules
are unit-testable without video - the same approach as zone/state.py and
core/target_lock.py.

Direction is decided by which side of the line a track moved from and to. The
side of a point is the sign of the 2D cross product of the line's direction
vector with the vector to the point, which is exact, has no thresholds to
tune, and is independent of how the line was drawn.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# The line divides the frame into two sides. Which is "A" and which is "B" is
# fixed by the sign of the cross product with the line's direction vector:
# side A is where that sign is positive. That is arbitrary but *stable* - it
# depends only on the order the endpoints were given, never on the camera
# angle or the shape of the scene.
#
# Naming the sides rather than the endpoints matters: an operator drawing a
# line thinks in terms of "the far side" and "our side", and a direction named
# after endpoints reads as travel *along* the line instead of across it. The
# UI resolves the remaining ambiguity by showing which side is which on the
# drawn line, and by letting the operator flip the forbidden direction after
# watching one real crossing.
A_TO_B = "A_TO_B"  # from the positive side to the negative side
B_TO_A = "B_TO_A"  # from the negative side to the positive side


@dataclass
class TripwireEvent:
    type: str  # LINE_CROSSED | WRONG_WAY_CROSSING
    tripwire_id: str
    track_id: int
    frame_index: int
    timestamp_sec: float
    direction: str
    data: dict = field(default_factory=dict)


@dataclass
class Tripwire:
    """A directional line in normalized [0,1] coordinates."""

    tripwire_id: str
    camera_id: str
    label: str
    # Two endpoints; the segment between them is the line.
    points: list[tuple[float, float]]
    # Direction that counts as a violation. None means report both ways.
    forbidden_direction: str | None = None

    def __post_init__(self):
        if len(self.points) != 2:
            raise ValueError("a tripwire is defined by exactly 2 points")
        if self.points[0] == self.points[1]:
            raise ValueError("a tripwire's two points must differ")
        for x, y in self.points:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(f"tripwire points must be normalized to [0,1], got ({x}, {y})")
        if self.forbidden_direction not in (None, A_TO_B, B_TO_A):
            raise ValueError(f"unknown direction: {self.forbidden_direction}")

    def to_dict(self) -> dict:
        return {
            "tripwire_id": self.tripwire_id, "camera_id": self.camera_id,
            "label": self.label, "points": [list(p) for p in self.points],
            "forbidden_direction": self.forbidden_direction,
        }

    @staticmethod
    def from_dict(d: dict) -> "Tripwire":
        return Tripwire(
            tripwire_id=d["tripwire_id"], camera_id=d.get("camera_id", ""),
            label=d.get("label", ""), points=[tuple(p) for p in d["points"]],
            forbidden_direction=d.get("forbidden_direction"),
        )


def side_of_line(line: tuple[tuple[float, float], tuple[float, float]],
                 point: tuple[float, float]) -> int:
    """-1, 0 or +1 for which side of the line the point lies on."""
    (x1, y1), (x2, y2) = line
    px, py = point
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    if cross > 0:
        return 1
    if cross < 0:
        return -1
    return 0


def segments_intersect(a1, a2, b1, b2) -> bool:
    """Whether segment a1-a2 crosses segment b1-b2.

    A sign change alone is not enough: a track can move from one side to the
    other while passing *beyond the end* of a finite line, which is not a
    crossing. Both segments must straddle each other.
    """
    d1 = side_of_line((b1, b2), a1)
    d2 = side_of_line((b1, b2), a2)
    d3 = side_of_line((a1, a2), b1)
    d4 = side_of_line((a1, a2), b2)
    return d1 != d2 and d3 != d4 and d1 != 0 and d2 != 0


class TripwireMonitor:
    """Tracks which side of a line each track is on and reports crossings."""

    def __init__(self, tripwire: Tripwire, cooldown_sec: float = 3.0):
        self.tripwire = tripwire
        self.cooldown_sec = cooldown_sec
        self._last_point: dict[int, tuple[float, float]] = {}
        self._last_crossing: dict[int, float] = {}
        self.events: list[TripwireEvent] = []

    def update(self, points: dict[int, tuple[float, float]], frame_index: int,
               timestamp_sec: float) -> list[TripwireEvent]:
        """Advance one frame. `points` are normalized positions per track id."""
        line = (tuple(self.tripwire.points[0]), tuple(self.tripwire.points[1]))
        new_events: list[TripwireEvent] = []

        for track_id, point in points.items():
            previous = self._last_point.get(track_id)
            self._last_point[track_id] = point
            if previous is None:
                continue  # need two positions before movement exists

            if not segments_intersect(previous, point, line[0], line[1]):
                continue

            # A track that jitters across the line would otherwise report a
            # crossing every frame; one crossing per track per cooldown is what
            # an operator can actually act on.
            last = self._last_crossing.get(track_id)
            if last is not None and timestamp_sec - last < self.cooldown_sec:
                continue
            self._last_crossing[track_id] = timestamp_sec

            # Side A is the positive side (see the constants above), so leaving
            # it is A_TO_B.
            direction = A_TO_B if side_of_line(line, previous) > 0 else B_TO_A
            forbidden = self.tripwire.forbidden_direction
            violating = forbidden is not None and direction == forbidden

            event = TripwireEvent(
                type="WRONG_WAY_CROSSING" if violating else "LINE_CROSSED",
                tripwire_id=self.tripwire.tripwire_id,
                track_id=track_id, frame_index=frame_index, timestamp_sec=timestamp_sec,
                direction=direction,
                data={
                    "direction": direction,
                    "label": self.tripwire.label,
                    "forbidden_direction": forbidden,
                },
            )
            new_events.append(event)

        # Forget tracks that have gone, so the position map cannot grow for the
        # lifetime of a continuously running camera.
        for stale in set(self._last_point) - set(points):
            self._last_point.pop(stale, None)
            self._last_crossing.pop(stale, None)

        self.events.extend(new_events)
        return new_events
