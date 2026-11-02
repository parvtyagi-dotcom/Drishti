"""Adaptive detection stride: what counts as worth detecting every frame."""
from __future__ import annotations

import numpy as np
import pytest

from backend.config import TrackerConfig
from backend.core.target_lock import TargetLock
from backend.modules.zone.geometry import distance_to_polygon
from backend.modules.zone.state import ZoneMonitor, anyone_near_zone
from backend.orchestrator import _following

SQUARE = [(100, 100), (200, 100), (200, 200), (100, 200)]


def test_adaptive_stride_is_on_by_default():
    assert TrackerConfig().adaptive_stride is True


@pytest.mark.parametrize("point, expected", [
    ((150, 150), 0.0),      # inside
    ((250, 150), 50.0),     # right of the right edge
    ((150, 40), 60.0),      # above the top edge
    ((230, 240), 50.0),     # off the corner: 30-40-50 triangle
])
def test_distance_to_polygon(point, expected):
    assert distance_to_polygon(point, SQUARE) == pytest.approx(expected)


def test_anyone_near_zone_uses_the_margin():
    monitor = ZoneMonitor("z")
    assert anyone_near_zone({1: (240, 150)}, [monitor], [SQUARE], near_px=50)
    assert not anyone_near_zone({1: (260, 150)}, [monitor], [SQUARE], near_px=50)
    assert not anyone_near_zone({}, [monitor], [SQUARE], near_px=50)


def test_anyone_near_zone_checks_every_zone():
    far_zone = [(1000, 1000), (1100, 1000), (1100, 1100)]
    monitors = [ZoneMonitor("a"), ZoneMonitor("b")]
    assert anyone_near_zone({1: (1050, 990)}, monitors, [SQUARE, far_zone], near_px=20)


def test_following_only_while_the_lock_holds_a_track():
    lock = TargetLock(confirmation_observations=1, gap_tolerance_frames=2)
    assert not _following(None)
    assert not _following(lock)                       # nothing confirmed yet
    lock.register_match(7, 0, 0.0, votes=1, similarity=0.9)
    assert lock.confirmed and _following(lock)        # locked on track 7
    for frame in range(1, 6):                         # target gone past its gap tolerance
        lock.step_visibility(set(), frame, frame / 30)
    assert lock.confirmed and not _following(lock)    # still "confirmed", but no longer followed
