"""Sensor fusion: a sensor trip is checked against the linked camera."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from backend.core.events import describe_event, severity_for, should_alarm
from backend.live.sensors import SensorFusion, SensorRegistry, _spread


def _jpeg() -> bytes:
    ok, buf = cv2.imencode(".jpg", np.zeros((48, 64, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


class FakeEvidence:
    def __init__(self, times):
        self.frames = [(t, _jpeg()) for t in times]

    def frames_since(self, since):
        return [f for f in self.frames if f[0] >= since]


class FakeTamper:
    tampered = None


class FakeCamera:
    def __init__(self, name):
        self.name = name


class FakeSession:
    def __init__(self, name="Gate 1", state="online", frames=(98.0, 100.0, 102.0, 104.0)):
        self.camera = FakeCamera(name)
        self.state = state
        self.tamper = FakeTamper()
        self.evidence = FakeEvidence(frames)
        self.emitted = []

    def status_dict(self):
        return {"state": self.state, "degraded": False}

    def emit_event(self, event):
        severity = severity_for(event)
        out = {**event, "severity": severity, "reason": describe_event(event),
               "alarm": should_alarm({"severity": severity})}
        self.emitted.append(out)
        return out


class FakeManager:
    def __init__(self, sessions=()):
        self.sessions = {s.camera.name.lower(): s for s in sessions}
        self.unattached = []

    def find_by_name(self, name):
        return self.sessions.get(name.lower())

    def emit_unattached(self, event, source_id, source_name):
        out = {**event, "camera_id": source_id, "severity": severity_for(event)}
        self.unattached.append(out)
        return out


def make(tmp_path, sessions=(), look=lambda image: (0, 0, 0.0)):
    reg = SensorRegistry(tmp_path / "sensors.json")
    fusion = SensorFusion(reg, FakeManager(sessions), look=look,
                          clock=lambda: 100.0, wait=lambda s: None)
    return reg, fusion


def types(session):
    return [e["type"] for e in session.emitted]


def test_person_on_camera_confirms_and_sounds(tmp_path):
    cam = FakeSession()
    reg, fusion = make(tmp_path, [cam], look=lambda img: (2, 0, 0.8))
    sensor = reg.add("Fence A", "fence", "gate 1")
    fusion.trigger(sensor, background=False)
    assert types(cam) == ["SENSOR_TRIGGERED", "SENSOR_ALARM_CONFIRMED"]
    first, final = cam.emitted
    assert not first["alarm"]
    assert final["severity"] == "CRITICAL" and final["alarm"]
    assert "saw 2 persons" in final["reason"]


def test_empty_camera_is_logged_quietly(tmp_path):
    cam = FakeSession()
    reg, fusion = make(tmp_path, [cam])
    sensor = reg.add("Fence A", "fence", "Gate 1")
    fusion.trigger(sensor, background=False)
    final = cam.emitted[-1]
    assert final["type"] == "SENSOR_ALARM_UNCONFIRMED"
    assert final["severity"] == "LOW" and not final["alarm"]
    assert "animal, wind or fault" in final["reason"]


def test_repeated_empty_trips_escalate(tmp_path):
    cam = FakeSession()
    reg, fusion = make(tmp_path, [cam])
    sensor = reg.add("Fence A", "fence", "Gate 1")
    for _ in range(3):
        fusion.trigger(sensor, background=False)
    final = cam.emitted[-1]
    assert final["data"]["repeat_count"] == 3
    assert final["severity"] == "MEDIUM"
    assert "3 times" in final["reason"]


@pytest.mark.parametrize("setup,why", [
    (lambda cam: setattr(cam, "state", "offline"), "offline"),
    (lambda cam: setattr(cam.tamper, "tampered", "covered"), "blocked or moved"),
    (lambda cam: setattr(cam.evidence, "frames", []), "no video"),
])
def test_camera_that_cannot_look_never_dismisses(tmp_path, setup, why):
    cam = FakeSession()
    setup(cam)
    reg, fusion = make(tmp_path, [cam])
    sensor = reg.add("Fence A", "fence", "Gate 1")
    fusion.trigger(sensor, background=False)
    final = cam.emitted[-1]
    assert final["type"] == "SENSOR_ALARM_UNVERIFIED"
    assert final["alarm"]
    assert why in final["reason"]


def test_missing_camera_still_raises_the_alarm(tmp_path):
    reg, fusion = make(tmp_path)
    sensor = reg.add("Ground 3", "ground", "Gone camera")
    fusion.trigger(sensor, background=False)
    final = fusion.manager.unattached[-1]
    assert final["type"] == "SENSOR_ALARM_UNVERIFIED"
    assert final["camera_id"] == "sensors"
    assert final["severity"] == "HIGH"


def test_detector_failure_counts_as_unverified(tmp_path):
    def broken(image):
        raise RuntimeError("model missing")
    cam = FakeSession()
    reg, fusion = make(tmp_path, [cam], look=broken)
    sensor = reg.add("Fence A", "fence", "Gate 1")
    fusion.trigger(sensor, background=False)
    assert cam.emitted[-1]["type"] == "SENSOR_ALARM_UNVERIFIED"


def test_drone_alert_sounds_without_a_camera_check(tmp_path):
    cam = FakeSession()
    looked = []
    reg, fusion = make(tmp_path, [cam], look=lambda img: looked.append(1) or (0, 0, 0.0))
    sensor = reg.add("Mast mic", "acoustic_drone", "Gate 1")
    result = fusion.trigger(sensor, background=False)
    assert result["status"] == "alarm"
    assert types(cam) == ["DRONE_SUSPECTED"]
    assert cam.emitted[0]["alarm"] and not looked
    assert "look up" in cam.emitted[0]["reason"]


def test_test_trips_say_so(tmp_path):
    cam = FakeSession()
    reg, fusion = make(tmp_path, [cam], look=lambda img: (1, 0, 0.9))
    sensor = reg.add("Fence A", "fence", "Gate 1")
    fusion.trigger(sensor, source="simulated", background=False)
    assert all("(test trigger)" in e["reason"] for e in cam.emitted)


def test_only_frames_around_the_trip_are_checked(tmp_path):
    cam = FakeSession(frames=[80.0, 90.0, 97.0, 101.0])
    reg, fusion = make(tmp_path, [cam], look=lambda img: (1, 0, 0.9))
    sensor = reg.add("Fence A", "fence", "Gate 1")
    fusion.trigger(sensor, background=False)
    assert cam.emitted[-1]["data"]["frames_checked"] == 2


def test_registry_persists_and_checks_keys(tmp_path):
    reg = SensorRegistry(tmp_path / "sensors.json")
    sensor = reg.add("Fence A", "fence", None)
    again = SensorRegistry(tmp_path / "sensors.json")
    assert again.get(sensor.sensor_id).name == "Fence A"
    assert again.check_key(sensor.sensor_id, sensor.key) is not None
    assert again.check_key(sensor.sensor_id, "wrong") is None
    assert again.check_key(sensor.sensor_id, None) is None
    assert "key" not in sensor.public()
    with pytest.raises(ValueError):
        reg.add("X", "laser", None)


def test_spread_keeps_both_ends():
    items = list(range(20))
    picked = _spread(items, 5)
    assert picked[0] == 0 and picked[-1] == 19 and len(picked) == 5


def test_digging_alert_sounds_without_a_camera_check(tmp_path):
    cam = FakeSession()
    reg, fusion = make(tmp_path, [cam], look=lambda img: (0, 0, 0.0))
    sensor = reg.add("Fence post 4", "seismic", "Gate 1")
    assert not sensor.public()["checkable"]
    result = fusion.trigger(sensor, detail="6 steady strikes about 1.2 s apart", background=False)
    assert result["status"] == "alarm"
    event = cam.emitted[0]
    assert event["type"] == "DIGGING_SUSPECTED" and event["alarm"]
    assert "Digging sensor 'Fence post 4'" in event["reason"] and "tunnelling" in event["reason"]
