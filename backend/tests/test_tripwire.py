"""Directional line crossing.

Pure geometry and state, so every case here is exact - there are no
thresholds being approximated and no video needed.
"""
from __future__ import annotations

import pytest

from backend.modules.zone.tripwire import (
    A_TO_B, B_TO_A, Tripwire, TripwireMonitor, segments_intersect, side_of_line,
)

# A vertical line down the middle of the frame.
MIDLINE = [(0.5, 0.0), (0.5, 1.0)]


def wire(**kwargs) -> Tripwire:
    params = {"tripwire_id": "tw1", "camera_id": "cam-01", "label": "BORDER LINE",
              "points": MIDLINE}
    params.update(kwargs)
    return Tripwire(**params)


def test_rejects_malformed_definitions():
    with pytest.raises(ValueError):
        wire(points=[(0.5, 0.0)])                       # needs two points
    with pytest.raises(ValueError):
        wire(points=[(0.5, 0.5), (0.5, 0.5)])           # zero-length
    with pytest.raises(ValueError):
        wire(points=[(0.5, 0.0), (1.5, 1.0)])           # not normalized
    with pytest.raises(ValueError):
        wire(forbidden_direction="SIDEWAYS")


def test_side_of_line_is_symmetric_about_the_line():
    line = (MIDLINE[0], MIDLINE[1])
    assert side_of_line(line, (0.2, 0.5)) == -side_of_line(line, (0.8, 0.5))
    assert side_of_line(line, (0.5, 0.5)) == 0  # exactly on the line


def test_crossing_requires_straddling_the_finite_segment():
    """Moving between sides *past the end* of a short line is not a crossing -
    the common false positive if only the side change is checked."""
    short = ((0.5, 0.0), (0.5, 0.3))
    # Crosses the infinite line, but well below the segment's end.
    assert not segments_intersect((0.3, 0.9), (0.7, 0.9), *short)
    # Crosses the segment itself.
    assert segments_intersect((0.3, 0.1), (0.7, 0.1), *short)


def test_reports_direction_of_travel():
    monitor = TripwireMonitor(wire())
    monitor.update({1: (0.2, 0.5)}, 0, 0.0)
    events = monitor.update({1: (0.8, 0.5)}, 1, 0.1)
    assert len(events) == 1
    assert events[0].type == "LINE_CROSSED"
    assert events[0].direction == A_TO_B

    monitor2 = TripwireMonitor(wire())
    monitor2.update({1: (0.8, 0.5)}, 0, 0.0)
    back = monitor2.update({1: (0.2, 0.5)}, 1, 0.1)
    assert back[0].direction == B_TO_A


def test_no_event_without_a_crossing():
    monitor = TripwireMonitor(wire())
    monitor.update({1: (0.2, 0.5)}, 0, 0.0)
    assert monitor.update({1: (0.3, 0.6)}, 1, 0.1) == []


def test_first_sighting_alone_never_counts_as_a_crossing():
    # A track appearing already on the far side has not been observed crossing.
    monitor = TripwireMonitor(wire())
    assert monitor.update({1: (0.9, 0.5)}, 0, 0.0) == []


def test_only_the_forbidden_direction_is_escalated():
    monitor = TripwireMonitor(wire(forbidden_direction=A_TO_B))
    monitor.update({1: (0.2, 0.5)}, 0, 0.0)
    violating = monitor.update({1: (0.8, 0.5)}, 1, 0.1)
    assert violating[0].type == "WRONG_WAY_CROSSING"

    allowed = TripwireMonitor(wire(forbidden_direction=A_TO_B))
    allowed.update({1: (0.8, 0.5)}, 0, 0.0)
    events = allowed.update({1: (0.2, 0.5)}, 1, 0.1)
    assert events[0].type == "LINE_CROSSED"  # permitted direction, still logged


def test_jitter_across_the_line_does_not_spam():
    monitor = TripwireMonitor(wire(), cooldown_sec=3.0)
    monitor.update({1: (0.49, 0.5)}, 0, 0.0)
    first = monitor.update({1: (0.51, 0.5)}, 1, 0.1)
    second = monitor.update({1: (0.49, 0.5)}, 2, 0.2)
    third = monitor.update({1: (0.51, 0.5)}, 3, 0.3)
    assert len(first) == 1
    assert second == [] and third == []


def test_crossing_again_after_the_cooldown_is_reported():
    monitor = TripwireMonitor(wire(), cooldown_sec=1.0)
    monitor.update({1: (0.2, 0.5)}, 0, 0.0)
    assert len(monitor.update({1: (0.8, 0.5)}, 1, 0.1)) == 1
    assert len(monitor.update({1: (0.2, 0.5)}, 2, 5.0)) == 1


def test_tracks_are_independent():
    monitor = TripwireMonitor(wire())
    monitor.update({1: (0.2, 0.5), 2: (0.8, 0.5)}, 0, 0.0)
    events = monitor.update({1: (0.8, 0.5), 2: (0.9, 0.5)}, 1, 0.1)
    assert [e.track_id for e in events] == [1]


def test_departed_tracks_are_forgotten():
    """A camera runs for weeks; per-track state must not accumulate forever."""
    monitor = TripwireMonitor(wire())
    monitor.update({i: (0.2, 0.5) for i in range(50)}, 0, 0.0)
    monitor.update({1: (0.2, 0.5)}, 1, 0.1)
    assert set(monitor._last_point) == {1}


def test_diagonal_line_and_round_trip():
    monitor = TripwireMonitor(wire(points=[(0.0, 0.0), (1.0, 1.0)]), cooldown_sec=0.0)
    monitor.update({1: (0.8, 0.2)}, 0, 0.0)          # below the diagonal
    out = monitor.update({1: (0.2, 0.8)}, 1, 0.1)    # above it
    back = monitor.update({1: (0.8, 0.2)}, 2, 0.2)
    assert out[0].direction != back[0].direction


def test_serialisation_round_trip():
    original = wire(forbidden_direction=B_TO_A)
    assert Tripwire.from_dict(original.to_dict()).to_dict() == original.to_dict()
