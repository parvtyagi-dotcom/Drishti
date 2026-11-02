"""Unit tests for ObjectTracker's detection_stride + propagation logic.

_detect() (the real YOLO call) is monkeypatched to return controlled,
deterministic Track lists - these tests verify the propagation ARITHMETIC
(velocity extrapolation, frame clipping, stride cadence) in isolation, not
YOLO's own accuracy. Real-content accuracy is verified separately in
scripts/benchmark_pipeline.py's A/B comparison.
"""
import numpy as np
import pytest

from backend.config import TrackerConfig
from backend.core.tracker import ObjectTracker, Track


def make_tracker(stride: int) -> ObjectTracker:
    cfg = TrackerConfig(detection_stride=stride)
    tracker = ObjectTracker.__new__(ObjectTracker)  # bypass __init__ (no real YOLO load)
    tracker.config = cfg
    tracker.class_ids = None
    tracker._model = None
    tracker._frame_index = 0
    tracker._last_detect_index = None
    tracker.detection_count = 0
    tracker._last_tracks = {}
    tracker._last_detected = {}
    tracker._velocity = {}
    return tracker


def test_default_stride_detects_every_frame(monkeypatch):
    tracker = make_tracker(stride=1)
    calls = []
    monkeypatch.setattr(tracker, "_detect", lambda image: calls.append(1) or [])
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    for _ in range(5):
        tracker.update(image)
    assert len(calls) == 5  # every single frame ran real detection


def test_stride_3_detects_every_third_frame(monkeypatch):
    tracker = make_tracker(stride=3)
    calls = []
    monkeypatch.setattr(tracker, "_detect", lambda image: calls.append(1) or [Track(1, (10, 10, 30, 30), 0, 0.9)])
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    for _ in range(9):
        tracker.update(image)
    assert len(calls) == 3  # frames 0, 3, 6 out of 9


def test_propagated_frames_extrapolate_velocity(monkeypatch):
    tracker = make_tracker(stride=3)
    boxes = [(10, 10, 30, 30), (16, 10, 36, 30)]  # moved +6px in x between two real detections
    call_iter = iter(boxes)

    def fake_detect(image):
        x1, y1, x2, y2 = next(call_iter)
        return [Track(1, (x1, y1, x2, y2), 0, 0.9)]

    monkeypatch.setattr(tracker, "_detect", fake_detect)
    image = np.zeros((200, 200, 3), dtype=np.uint8)

    frame0 = tracker.update(image)  # real detection: (10,10,30,30)
    assert frame0[0].bbox == (10, 10, 30, 30)
    assert frame0[0].propagated is False

    frame1 = tracker.update(image)  # propagated: no velocity yet (first detection), so unchanged
    assert frame1[0].bbox == (10, 10, 30, 30)
    assert frame1[0].propagated is True

    frame2 = tracker.update(image)  # still propagated frame (stride=3 -> frames 0,3,6 detect)
    assert frame2[0].bbox == (10, 10, 30, 30)

    # The two real detections are 3 frames apart and 6px apart, so the object
    # is moving 2px per frame - NOT 6. This originally asserted +6 per
    # propagated frame, which is the overshoot that made the box visibly
    # jitter: it ran three frames ahead of the vehicle and was snapped back at
    # every detection. See test_tracker_propagation.py.
    frame3 = tracker.update(image)  # real detection #2 -> velocity now (+2,0,+2,0) per frame
    assert frame3[0].bbox == (16, 10, 36, 30)

    frame4 = tracker.update(image)  # propagated one frame of real motion
    assert frame4[0].bbox == (18, 10, 38, 30)

    frame5 = tracker.update(image)  # and another - continuous with the next detection
    assert frame5[0].bbox == (20, 10, 40, 30)


def test_propagation_clips_to_frame_bounds(monkeypatch):
    tracker = make_tracker(stride=2)
    # Detections 2 frames apart move (+10,+10,+8,+8), so per frame that is
    # (+5,+5,+4,+4). Propagating once pushes x2/y2 past the 100x100 frame edge
    # while x1/y1 stay in bounds, so the clipped result should be a valid
    # (non-degenerate) box, not dropped entirely.
    boxes = [(50, 50, 90, 90), (60, 60, 98, 98)]
    call_iter = iter(boxes)
    monkeypatch.setattr(tracker, "_detect", lambda image: [Track(1, next(call_iter), 0, 0.9)])
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    tracker.update(image)  # real: (50,50,90,90)
    tracker.update(image)  # propagated using velocity 0 (first detection had none) -> unchanged, in bounds
    tracker.update(image)  # real: (60,60,98,98), velocity now (+10,+10,+8,+8)
    frame3 = tracker.update(image)  # propagated: raw (65,65,102,102) -> x2/y2 clipped to frame (100x100)

    assert len(frame3) == 1  # clipped, not dropped - still a valid non-degenerate box
    x1, y1, x2, y2 = frame3[0].bbox
    assert x2 == 100 and y2 == 100  # clipped exactly to the frame edge
    assert x1 == 65 and y1 == 65  # unaffected (was already in bounds)


def test_track_disappearing_stops_after_next_real_detection(monkeypatch):
    tracker = make_tracker(stride=3)
    detect_results = [[Track(1, (10, 10, 30, 30), 0, 0.9)], []]  # track vanishes on the 2nd real detection
    call_iter = iter(detect_results)
    monkeypatch.setattr(tracker, "_detect", lambda image: next(call_iter))
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    tracker.update(image)  # real: track present
    p1 = tracker.update(image)  # propagated: still shown (stale, expected - bridges at most stride-1 frames)
    assert len(p1) == 1
    tracker.update(image)  # propagated
    real2 = tracker.update(image)  # real detection #2: track genuinely gone
    assert real2 == []
    p2 = tracker.update(image)  # propagation now has nothing to extrapolate from
    assert p2 == []


def test_stride_is_clamped_to_at_least_one():
    tracker = make_tracker(stride=0)
    assert max(1, tracker.config.detection_stride) == 1


def _detected_frames(tracker, monkeypatch, urgent_on: set[int], n: int) -> list[int]:
    detected = []
    monkeypatch.setattr(tracker, "_detect", lambda image: detected.append(i) or [Track(1, (10, 10, 30, 30), 0, 0.9)])
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(n):
        tracker.update(image, urgent=i in urgent_on)
    return detected


def test_urgent_frames_always_detect(monkeypatch):
    tracker = make_tracker(stride=5)
    assert _detected_frames(tracker, monkeypatch, urgent_on={6, 7, 8}, n=10) == [0, 5, 6, 7, 8]
    assert tracker.detection_count == 5


def test_stride_restarts_from_the_last_urgent_detection(monkeypatch):
    # After the last urgent frame (8) the next routine detection is a full
    # stride later (13), not whenever the frame number is a multiple of 5.
    tracker = make_tracker(stride=5)
    assert _detected_frames(tracker, monkeypatch, urgent_on={7, 8}, n=19) == [0, 5, 7, 8, 13, 18]


def test_never_urgent_keeps_the_old_cadence(monkeypatch):
    tracker = make_tracker(stride=5)
    assert _detected_frames(tracker, monkeypatch, urgent_on=set(), n=16) == [0, 5, 10, 15]


def test_propagation_after_urgent_frames_uses_per_frame_velocity(monkeypatch):
    # Two urgent detections one frame apart: +4px is one frame of motion,
    # so the next (propagated) frame moves +4px, not +4px per stride.
    tracker = make_tracker(stride=5)
    boxes = iter([(10, 10, 30, 30), (14, 10, 34, 30)])
    monkeypatch.setattr(tracker, "_detect", lambda image: [Track(1, next(boxes), 0, 0.9)])
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    tracker.update(image, urgent=True)
    tracker.update(image, urgent=True)
    (t,) = tracker.update(image)
    assert t.propagated and t.bbox == (18, 10, 38, 30)
