"""What Live Monitor shows before an operator connects anything.

The server used to register stand-in cameras that looped sample footage from
the moment it booted. They looked like real feeds, kept analysis running, and
raised alarms - sounding the siren over whatever the operator was actually
doing, including work in Forensic Analysis that had nothing to do with them.
These pin the two halves of the fix: nothing runs unless it was connected,
and a camera with no footage behind it says so plainly.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.api.live import _source_missing, _status_for
from backend.app import app
from backend.core.video import PUSH_SCHEME
from backend.live.camera_manager import Camera, CameraManager, LiveSession, seed_demo_cameras

client = TestClient(app)


def test_the_server_starts_with_no_cameras():
    """Nothing plays until an operator connects it."""
    cameras = client.get("/api/live/cameras").json()
    assert cameras == [], f"the server registered cameras nobody connected: {cameras}"


def test_nothing_is_analysing_on_a_fresh_server():
    """No analysis means no detections, so no alarms and no siren."""
    for camera in client.get("/api/live/cameras").json():
        assert camera["modules"] == [], (
            f"{camera['name']} is running {camera['modules']} without being asked"
        )


def test_the_sample_feed_is_opt_in(tmp_path):
    """It still exists for a rehearsal - it just has to be asked for."""
    manager = CameraManager()
    (tmp_path / "smoke_person_id.mp4").write_bytes(b"not really a video")

    seeded = seed_demo_cameras(tmp_path, manager=manager, autostart=False)

    assert len(seeded) == 1
    session = manager.get(seeded[0].camera_id)
    # Registered, but watching for nothing until someone chooses what to watch.
    assert session.analysis.module_names() == []
    assert not session.analysis.any_enabled


def test_seeding_twice_does_not_duplicate_the_camera(tmp_path):
    manager = CameraManager()
    (tmp_path / "smoke_person_id.mp4").write_bytes(b"not really a video")

    seed_demo_cameras(tmp_path, manager=manager, autostart=False)
    again = seed_demo_cameras(tmp_path, manager=manager, autostart=False)

    assert again == []
    assert len(manager.list()) == 1


# --- no signal ------------------------------------------------------------

def _session(source: str) -> LiveSession:
    return LiveSession(Camera(camera_id="t", name="T", source=source))


def test_a_file_camera_with_no_footage_reports_no_signal(tmp_path):
    session = _session(str(tmp_path / "gone.mp4"))
    assert _source_missing(session) is True
    headline, detail = _status_for(session)
    assert headline == "NO SIGNAL"
    assert "no footage" in detail.lower()


def test_a_camera_whose_footage_exists_does_not_report_no_signal(tmp_path):
    clip = tmp_path / "present.mp4"
    clip.write_bytes(b"not really a video")
    session = _session(str(clip))
    assert _source_missing(session) is False
    assert _status_for(session)[0] != "NO SIGNAL"


@pytest.mark.parametrize("source", [
    f"{PUSH_SCHEME}abc123",
    "rtsp://192.168.0.50:554/stream1",
    "http://192.168.0.50/video.mjpg",
])
def test_a_camera_that_is_reached_over_the_network_is_never_no_signal(source):
    """An unreachable URL is a connection problem, which may fix itself. Only
    a file that is not on disk is permanently absent."""
    assert _source_missing(_session(source)) is False
