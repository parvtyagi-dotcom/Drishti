"""Sentinel: tamper detection on synthetic frames, and the sabotage timing rule."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from backend.core.events import describe_event, severity_for
from backend.live.camera_manager import CameraManager
from backend.live.tamper import BLURRED, COVERED, MOVED, TamperMonitor, TamperReading, TappedReader


def scene(seed: int = 1, brightness: float = 1.0) -> np.ndarray:
    """A textured outdoor-like view: shapes and edges a camera would see."""
    rng = np.random.default_rng(seed)
    img = np.full((360, 640, 3), 70, np.uint8)
    for _ in range(40):
        x, y = rng.integers(0, 600), rng.integers(0, 330)
        w, h = rng.integers(10, 120), rng.integers(10, 80)
        colour = tuple(int(c) for c in rng.integers(20, 235, 3))
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), colour, -1)
    return np.clip(img.astype(np.float32) * brightness, 0, 255).astype(np.uint8)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def feed(monitor, clock, frame, seconds):
    results = []
    for _ in range(seconds):
        clock.t += 1.0
        results.append(monitor.observe(frame))
    return [r for r in results if r is not None]


def warmed_up():
    clock = Clock()
    monitor = TamperMonitor(clock=clock)
    assert feed(monitor, clock, scene(), 12) == []
    return monitor, clock


@pytest.mark.parametrize("make, kind", [
    (lambda: np.full((360, 640, 3), 40, np.uint8), COVERED),
    (lambda: cv2.GaussianBlur(scene(), (0, 0), 12), BLURRED),
    (lambda: scene(seed=99), MOVED),
])
def test_each_kind_of_tampering_is_reported_once_it_persists(make, kind):
    monitor, clock = warmed_up()
    reports = feed(monitor, clock, make(), 6)
    assert len(reports) == 1 and isinstance(reports[0], TamperReading) and reports[0].kind == kind
    assert monitor.tampered == kind


def test_a_brief_obstruction_is_ignored():
    monitor, clock = warmed_up()
    assert feed(monitor, clock, np.zeros((360, 640, 3), np.uint8), 2) == []
    assert feed(monitor, clock, scene(), 5) == []


def test_restoring_the_view_is_reported_after_it_holds():
    monitor, clock = warmed_up()
    feed(monitor, clock, np.full((360, 640, 3), 40, np.uint8), 6)
    assert feed(monitor, clock, scene(), 6) == ["restored"]
    assert monitor.tampered is None


def test_a_person_lingering_close_to_the_lens_is_not_a_move():
    monitor, clock = warmed_up()
    near = scene()
    near[:, 60:580] = scene(seed=7)[:, 60:580]  # a figure filling most of the frame
    assert feed(monitor, clock, near, 20) == []


def test_slow_dusk_is_not_tampering():
    monitor, clock = warmed_up()
    reports = []
    for step in range(80):  # the scene darkens to a third over 80 seconds
        reports += feed(monitor, clock, scene(brightness=1.0 - step * 0.0085), 1)
    assert reports == []


def test_frames_are_sampled_not_inspected_every_time():
    clock = Clock()
    monitor = TamperMonitor(clock=clock)
    calls = []
    original = monitor._learn
    monitor._learn = lambda *a, **k: (calls.append(1), original(*a, **k))
    for _ in range(25):  # 25 frames within one second
        monitor.observe(scene())
    assert len(calls) == 1


def test_tapped_reader_shows_raw_frames_and_passes_them_on():
    class Frame:
        def __init__(self, image):
            self.image = image

    class Reader:
        info = "info"

        def __iter__(self):
            yield Frame("a")
            yield Frame("b")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

    seen = []
    tapped = TappedReader(Reader(), seen.append)
    with tapped as reader:
        assert reader.info == "info"
        assert [f.image for f in reader] == ["a", "b"]
    assert seen == ["a", "b"]


# --- sabotage timing -----------------------------------------------------------

class FakeSession:
    def __init__(self):
        self.emitted = []

    def emit_event(self, event):
        self.emitted.append(event)
        return event


class FakeFleet:
    SABOTAGE_WINDOW_SEC = CameraManager.SABOTAGE_WINDOW_SEC
    _ACTIVITY_SEVERITIES = CameraManager._ACTIVITY_SEVERITIES

    def __init__(self, events):
        self.events = events

    def recent_events(self, limit=100):
        return self.events[-limit:]


def test_camera_dying_right_after_activity_is_flagged_as_possible_sabotage():
    fleet = FakeFleet([
        {"type": "ZONE_ENTRY", "severity": "CRITICAL", "wall_time": 950.0, "camera_name": "Gate 1"},
        {"type": "CAMERA_OFFLINE", "severity": "HIGH", "wall_time": 990.0, "camera_name": "Gate 2"},
        {"type": "VEHICLE_DETECTED", "severity": "INFO", "wall_time": 995.0, "camera_name": "Gate 2"},
    ])
    session = FakeSession()
    event = CameraManager.check_sabotage(fleet, session, now=1000.0)
    assert event["type"] == "CAMERA_SABOTAGE_SUSPECTED"
    assert event["data"]["recent_alerts"] == 1 and event["data"]["last_alert_camera"] == "Gate 1"
    assert event["data"]["seconds_since_last_alert"] == 50


def test_a_quiet_outage_is_just_an_outage():
    fleet = FakeFleet([{"type": "ZONE_ENTRY", "severity": "CRITICAL", "wall_time": 500.0, "camera_name": "Gate 1"}])
    session = FakeSession()
    assert CameraManager.check_sabotage(fleet, session, now=1000.0) is None
    assert session.emitted == []


def test_sentinel_events_have_alarm_severity_and_plain_reasons():
    assert severity_for({"type": "CAMERA_TAMPERED"}) == "HIGH"
    assert severity_for({"type": "CAMERA_SABOTAGE_SUSPECTED"}) == "CRITICAL"
    assert describe_event({"type": "CAMERA_TAMPERED", "data": {"detail": "view blocked or blacked out"}}) == \
        "Possible camera tampering: view blocked or blacked out"
    assert describe_event({"type": "CAMERA_SABOTAGE_SUSPECTED", "data": {
        "recent_alerts": 3, "seconds_since_last_alert": 20, "last_alert_camera": "Gate 1"}}) == \
        "Camera went offline after 3 alerts on the post, the last 20s earlier (last on Gate 1): possible sabotage"
