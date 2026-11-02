"""Box propagation between detections in the shared tracker.

Covers the cause of a visibly jittering target box. Running the detector
every Nth frame and extrapolating in between is only invisible if the
extrapolation moves at the object's actual speed; when it moved at a whole
stride's worth per frame, the box raced ahead of the vehicle and was yanked
back at each detection, several times a second.
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.config import TrackerConfig
from backend.core.tracker import ObjectTracker, Track

STRIDE = 3
SPEED_PX = 9  # the object's true speed, per frame
WIDTH = 100


@pytest.fixture
def tracker(monkeypatch):
    """A tracker whose detector is a stand-in moving at a known constant speed.

    The real model is never loaded: what is under test is the arithmetic
    between detections, and a stub makes the expected positions exact.
    """
    monkeypatch.setattr("backend.core.tracker.YOLO", lambda *a, **k: object())
    instance = ObjectTracker(TrackerConfig(detection_stride=STRIDE), class_ids=(2,))

    def fake_detect(image):
        left = (instance._frame_index - 1) * SPEED_PX
        return [Track(track_id=1, bbox=(left, 0, left + WIDTH, 50), class_id=2, confidence=0.9)]

    monkeypatch.setattr(instance, "_detect", fake_detect)
    return instance


def _left_edges(tracker: ObjectTracker, frames: int = 12) -> list[int]:
    image = np.zeros((200, 4000, 3), dtype=np.uint8)
    edges = []
    for _ in range(frames):
        tracks = tracker.update(image)
        edges.append(tracks[0].bbox[0] if tracks else None)
    return edges


def test_the_box_never_outruns_the_object_between_detections(tracker):
    edges = _left_edges(tracker)
    # Skip the opening interval: the first detection has nothing to measure
    # speed against, so the box correctly holds still until the second one.
    steps = [b - a for a, b in zip(edges[STRIDE:], edges[STRIDE + 1:])]
    assert steps, "no propagation happened"
    worst = max(steps)
    assert worst <= SPEED_PX * 1.5, (
        f"an extrapolated box moved {worst}px in one frame when the object moves "
        f"{SPEED_PX}px - it is running ahead and will be snapped back"
    )


def test_the_box_is_never_yanked_backwards(tracker):
    """Snapping back is the other half of what reads as jitter."""
    edges = _left_edges(tracker)
    steps = [b - a for a, b in zip(edges, edges[1:])]
    assert min(steps) >= 0, f"the box jumped backwards by {abs(min(steps))}px"


def test_propagated_positions_track_the_real_object(tracker):
    """Smoothness is worthless if the box is smoothly in the wrong place."""
    edges = _left_edges(tracker)
    for frame, left in enumerate(edges[STRIDE:], start=STRIDE):
        truth = frame * SPEED_PX
        assert abs(left - truth) <= SPEED_PX, (
            f"frame {frame}: box at {left}px, object at {truth}px"
        )


def test_a_missed_detection_does_not_send_the_box_running(tracker, monkeypatch):
    """A track the detector drops for one round must not overshoot afterwards.

    Velocity spans the frames actually elapsed rather than an assumed stride,
    so a longer gap cannot inflate the apparent speed. Re-syncing to the true
    position on the detection frame itself is expected and correct - the box
    belongs where the vehicle is - so only extrapolated frames are measured.
    """
    seen = {"calls": 0}

    def flaky_detect(image):
        seen["calls"] += 1
        left = (tracker._frame_index - 1) * SPEED_PX
        if seen["calls"] == 2:      # this detection loses the track entirely
            return []
        return [Track(track_id=1, bbox=(left, 0, left + WIDTH, 50), class_id=2, confidence=0.9)]

    monkeypatch.setattr(tracker, "_detect", flaky_detect)

    image = np.zeros((200, 4000, 3), dtype=np.uint8)
    samples = []
    for _ in range(15):
        tracks = tracker.update(image)
        samples.append((tracks[0].bbox[0], tracks[0].propagated) if tracks else None)

    steps = [
        b[0] - a[0]
        for a, b in zip(samples, samples[1:])
        if a is not None and b is not None and b[1]  # b was extrapolated
    ]
    assert steps, "nothing was extrapolated after the missed detection"
    assert max(steps) <= SPEED_PX * 1.5, (
        f"an extrapolated box moved {max(steps)}px in one frame for an object "
        f"moving {SPEED_PX}px - the gap inflated its speed"
    )
