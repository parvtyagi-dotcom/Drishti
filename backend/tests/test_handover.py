"""The shift handover report."""
from __future__ import annotations

from fastapi.testclient import TestClient

from backend.agent.patterns import Pattern
from backend.api.handover import build_report
from backend.app import app

NOW = 1_800_000_000.0


def report(**overrides):
    args = dict(hours=12, events=[], cameras=[], chain={"valid": True, "summary": "All 5 records verified."},
                briefs=[], proposals=[], patterns=[], generated_by="commander", now=NOW)
    args.update(overrides)
    return build_report(**args)


def test_quiet_shift_says_nothing_is_outstanding():
    page = report(cameras=[{"name": "Gate 1", "state": "online", "modules": ["zone"]}])
    assert "Nothing outstanding" in page and "No alarms this shift." in page and "All 5 records verified." in page


def test_alarms_open_items_and_patterns_are_listed():
    events = [
        {"type": "ZONE_ENTRY", "alarm": True, "severity": "CRITICAL", "wall_time": NOW - 600, "camera_name": "Gate 1", "reason": "Track #7 crossed into 'Restricted'"},
        {"type": "CAMERA_TAMPERED", "alarm": True, "severity": "HIGH", "wall_time": NOW - 300, "camera_name": "Gate 2"},
        {"type": "ZONE_ENTRY", "alarm": True, "severity": "CRITICAL", "wall_time": NOW - 13 * 3600, "camera_name": "Old"},
    ]
    page = report(
        events=events,
        cameras=[{"name": "Gate 2", "state": "offline"}],
        proposals=[{"status": "pending", "action": "dispatch_patrol"}],
        patterns=[Pattern(kind="plate", key="TEST123", summary="Plate TEST123 read 3 times on 2 days.", sightings=3, days=2)],
    )
    assert "Track #7 crossed into &#x27;Restricted&#x27;" in page  # escaped, never raw HTML
    assert "Old" not in page  # outside the 12-hour window
    assert "1 AI proposal still waiting" in page
    assert "Gate 2 is offline" in page and "Gate 2 reported tampering" in page
    assert "Plate TEST123 read 3 times on 2 days." in page


def test_restored_camera_is_not_listed_as_tampered():
    events = [{"type": "CAMERA_TAMPERED", "wall_time": NOW - 300, "camera_name": "Gate 2"},
              {"type": "CAMERA_VIEW_RESTORED", "wall_time": NOW - 200, "camera_name": "Gate 2"}]
    assert "reported tampering" not in report(events=events)


def test_broken_log_is_called_out():
    page = report(chain={"valid": False, "summary": "Integrity check FAILED at record 4: hash mismatch"})
    assert "Integrity check FAILED at record 4" in page and "class=\"bad\"" in page


def test_camera_names_cannot_inject_markup():
    page = report(cameras=[{"name": "<script>alert(1)</script>", "state": "online"}])
    assert "<script>alert(1)</script>" not in page and "&lt;script&gt;" in page


def test_handover_route_serves_a_page():
    response = TestClient(app).get("/api/live/handover", params={"hours": 500})
    assert response.status_code == 200 and "text/html" in response.headers["content-type"]
    assert "Last 72 hours" in response.text  # capped
