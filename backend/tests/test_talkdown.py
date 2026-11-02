"""Talk-down: warnings go out only when a speaker is listening, and not too often."""
from __future__ import annotations

from backend.core.events import describe_event, severity_for
from backend.live.talkdown import TalkDown


class FakeCamera:
    def __init__(self, name):
        self.name = name


class FakeSession:
    def __init__(self, name="Gate 1"):
        self.camera = FakeCamera(name)
        self.emitted = []

    def emit_event(self, event):
        out = {**event, "camera_name": self.camera.name, "reason": describe_event(event),
               "severity": severity_for(event)}
        self.emitted.append(out)
        return out


class FakeManager:
    def __init__(self, session):
        self.session = session

    def find_by_name(self, name):
        return self.session if name.lower() == self.session.camera.name.lower() else None

    def subscribe(self, cb):
        pass

    def unsubscribe(self, cb):
        pass


def make():
    now = [0.0]
    cam = FakeSession()
    td = TalkDown(FakeManager(cam), clock=lambda: now[0])
    return td, cam, now


def approach():
    return {"type": "ZONE_APPROACH", "camera_name": "Gate 1", "track_id": 4}


def test_no_speaker_means_no_warning_and_no_record():
    td, cam, _ = make()
    td.on_event(approach())
    assert cam.emitted == []
    assert td.warn(cam, trigger="operator", by="op") is None


def test_approach_warns_once_per_cooldown():
    td, cam, now = make()
    td.speaker_connected("gate  1")
    td.on_event(approach())
    td.on_event(approach())
    assert len(cam.emitted) == 1
    event = cam.emitted[0]
    assert event["type"] == "TALKDOWN_ISSUED" and event["severity"] == "INFO"
    assert "automatically, after zone approach" in event["reason"]
    assert "रुकिए" in event["data"]["lines"]["hi"]
    now[0] = 25.0
    td.on_event(approach())
    assert len(cam.emitted) == 2


def test_routine_events_and_auto_off_do_not_warn():
    td, cam, _ = make()
    td.speaker_connected("Gate 1")
    td.on_event({"type": "VEHICLE_DETECTED", "camera_name": "Gate 1"})
    td.auto = False
    td.on_event(approach())
    assert cam.emitted == []


def test_operator_warning_ignores_cooldown_and_names_operator():
    td, cam, _ = make()
    td.speaker_connected("Gate 1")
    td.on_event(approach())
    event = td.warn(cam, trigger="operator", by="sgt.rao")
    assert event is not None and "by sgt.rao" in event["reason"]
    assert len(cam.emitted) == 2


def test_speaker_count_follows_connections():
    td, cam, _ = make()
    td.speaker_connected("Gate 1")
    td.speaker_connected("Gate 1")
    td.speaker_disconnected("Gate 1")
    assert td.speakers_for("Gate 1") == 1
    td.speaker_disconnected("Gate 1")
    td.speaker_disconnected("Gate 1")
    assert td.speakers_for("Gate 1") == 0
