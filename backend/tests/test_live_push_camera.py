"""Live camera sessions fed by a pushed source (a phone acting as a camera).

These cover the failure that made a working phone camera look broken: the
session kept dropping offline - sounding the siren and filling the evidence
panel each time - while the phone was still transmitting normally. Every check
here is about the seam between a long-lived sending socket and a session that
replaces its reader whenever it restarts.
"""
from __future__ import annotations

import json
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.core.video import PUSH_SCHEME, VideoOpenError
from backend.live.camera_manager import AnalysisSpec, Camera, LiveSession, camera_manager
from backend.live.evidence import EvidenceConfig, EvidenceRecorder

client = TestClient(app)

FRAME = cv2.imencode(".jpg", np.full((120, 160, 3), 90, dtype=np.uint8))[1].tobytes()


def _wait_until(predicate, timeout_sec: float = 5.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


@pytest.fixture
def phone_camera():
    """A registered push camera, removed again however the test ends."""
    response = client.post("/api/live/phone-cameras", json={"name": "Test phone"})
    assert response.status_code == 201
    camera_id = response.json()["camera_id"]
    yield camera_id
    camera_manager.remove(camera_id)


def test_pushed_frames_follow_the_session_when_it_swaps_readers(phone_camera):
    """Reconfiguring a camera must not orphan the socket already feeding it.

    A session builds a fresh reader every time it restarts, so an ingest
    socket that resolved the reader once went on filling a buffer nothing was
    reading. The live reader then starved and the camera was declared offline
    with the phone still sending - the reason a phone camera dropped out every
    time anyone pressed Configure.
    """
    session = camera_manager.get(phone_camera)

    with client.websocket_connect(f"/api/live/ws/ingest/{phone_camera}") as socket:
        socket.send_bytes(FRAME)
        assert _wait_until(lambda: session.reader.status.last_frame_at is not None), \
            "the first reader never received the pushed frame"

        original = session.reader
        session.configure_analysis(AnalysisSpec())  # restarts in the background
        assert _wait_until(lambda: session.reader is not original),             "this test is meaningless unless the reader was replaced"

        # The phone knows nothing about the reconfigure and keeps sending.
        for _ in range(10):
            socket.send_bytes(FRAME)
            time.sleep(0.02)

        assert _wait_until(lambda: session.reader.status.last_frame_at is not None), \
            "frames were still going to the discarded reader"

    status = session.status_dict()
    assert status["state"] == "online"
    assert status["frame_age_sec"] < 5.0, "the replacement reader is being starved"


def test_a_pushed_camera_waits_for_its_phone_instead_of_dying(monkeypatch):
    """A phone that stops sending is an ordinary gap, not a dead session.

    The session used to let its capture thread exit, leaving recovery to
    whenever the phone happened to reconnect. Staying alive and retrying is
    what makes the feed resume on its own.
    """
    camera = Camera(camera_id="wait-test", name="Phone", source=f"{PUSH_SCHEME}wait-test")
    session = LiveSession(camera, analysis=AnalysisSpec(person_id=True))
    session.SENDER_RETRY_SEC = 0.05
    attempts: list[float] = []

    def never_connects():
        attempts.append(time.monotonic())
        raise VideoOpenError("no frames pushed to push://wait-test within 30.0s")

    monkeypatch.setattr(session, "_run_with_analysis", never_connects)

    session.start()
    try:
        assert _wait_until(lambda: len(attempts) >= 3, timeout_sec=5.0), \
            f"session gave up after {len(attempts)} attempt(s) instead of waiting for the sender"
        assert session.is_running

        assert _wait_until(
            lambda: session.status_dict().get("error") == "waiting for the sending device",
        ), "a phone that is simply not sending yet is being reported as a crash"
        # analysis_error is the banner shown to an operator; waiting for a
        # handset is not something they need to act on.
        assert session.status_dict()["analysis_error"] is None
    finally:
        session.stop()


def test_a_genuine_failure_is_still_reported_as_one(monkeypatch):
    """The waiting path must not swallow real faults."""
    camera = Camera(camera_id="fault-test", name="Phone", source=f"{PUSH_SCHEME}fault-test")
    session = LiveSession(camera, analysis=AnalysisSpec(person_id=True))
    session.SENDER_RETRY_SEC = 0.05

    monkeypatch.setattr(
        session, "_run_with_analysis",
        lambda: (_ for _ in ()).throw(RuntimeError("reference photo could not be read")),
    )

    session.start()
    try:
        assert _wait_until(lambda: session.status_dict()["analysis_error"] is not None), \
            "a real failure was hidden as if the phone were merely quiet"
        assert "reference photo" in session.status_dict()["analysis_error"]
    finally:
        session.stop()


def test_an_alert_with_no_footage_resolves_instead_of_pending_forever(tmp_path):
    """A camera going down has no recording of itself.

    Without a manifest the bundle is indistinguishable from one still being
    assembled, so the UI showed "clip pending" on alerts that could never have
    a clip - which reads as a system that lost the footage.
    """
    recorder = EvidenceRecorder(
        "cam-x", tmp_path, EvidenceConfig(post_event_sec=0.2, pre_event_sec=0.2),
    )
    bundle = recorder.capture({"event_id": "abc123", "type": "CAMERA_OFFLINE"})
    assert bundle is not None

    manifest = tmp_path / "cam-x" / bundle / "clip.json"
    try:
        assert _wait_until(manifest.is_file, timeout_sec=10.0), \
            "no manifest was written, so this bundle would sit at 'clip pending' forever"
    finally:
        recorder.stop()

    body = json.loads(manifest.read_text(encoding="utf-8"))
    assert body["clip"] is None
    assert "no footage" in body["note"]


def test_a_bundle_with_footage_still_reports_a_clip(tmp_path):
    """The no-footage path must not mask ordinary evidence capture."""
    recorder = EvidenceRecorder(
        "cam-y", tmp_path, EvidenceConfig(post_event_sec=0.2, pre_event_sec=2.0),
    )
    for _ in range(6):
        recorder.add_frame(FRAME)

    bundle = recorder.capture({"event_id": "def456", "type": "TARGET_CONFIRMED"})
    manifest = tmp_path / "cam-y" / bundle / "clip.json"
    try:
        assert _wait_until(manifest.is_file, timeout_sec=10.0)
    finally:
        recorder.stop()

    body = json.loads(manifest.read_text(encoding="utf-8"))
    assert body["clip"] is not None
    assert body["clip"]["frames"] > 0
    assert (tmp_path / "cam-y" / bundle / "clip.mp4").is_file()
