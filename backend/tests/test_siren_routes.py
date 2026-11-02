"""The siren phone's picture and brief: recent alarms only."""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from backend.api import agent as agent_api
from backend.api import siren_routes
from backend.app import app

client = TestClient(app)


class FakeManager:
    def __init__(self, events):
        self.events = events

    def recent_events(self, limit=100):
        return self.events


class FakeService:
    def brief_for_event(self, event_id):
        return {"status": "ready", "text": "Two people at the fence.", "model_kind": "local"} if event_id == "e1" else None


@pytest.fixture
def events(tmp_path, monkeypatch):
    bundle = tmp_path / "live" / "cam1" / "b1"
    bundle.mkdir(parents=True)
    (bundle / "snapshot.jpg").write_bytes(b"\xff\xd8jpeg")
    now = time.time()
    items = [
        {"event_id": "e1", "alarm": True, "wall_time": now, "camera_id": "cam1", "evidence_bundle": "b1"},
        {"event_id": "e2", "alarm": False, "wall_time": now, "camera_id": "cam1", "evidence_bundle": "b1"},
        {"event_id": "e3", "alarm": True, "wall_time": now - 3600, "camera_id": "cam1", "evidence_bundle": "b1"},
        {"event_id": "e4", "alarm": True, "wall_time": now, "camera_id": "cam1"},
    ]
    monkeypatch.setattr(siren_routes, "camera_manager", FakeManager(items))
    monkeypatch.setattr(siren_routes, "EVIDENCE_DIR", tmp_path)
    monkeypatch.setattr(agent_api, "_service", FakeService())


def test_recent_alarm_picture_is_served_without_login(events):
    reply = client.get("/api/live/siren/alert/e1/snapshot.jpg")
    assert reply.status_code == 200 and reply.content == b"\xff\xd8jpeg"


@pytest.mark.parametrize("event_id", ["e2", "e3", "nope"])
def test_routine_old_or_unknown_events_are_not_served(events, event_id):
    assert client.get(f"/api/live/siren/alert/{event_id}/snapshot.jpg").status_code == 404
    assert client.get(f"/api/live/siren/alert/{event_id}/brief").status_code == 404


def test_alarm_without_picture(events):
    assert client.get("/api/live/siren/alert/e4/snapshot.jpg").status_code == 404


def test_brief_ready_or_pending(events):
    assert client.get("/api/live/siren/alert/e1/brief").json()["text"] == "Two people at the fence."
    assert client.get("/api/live/siren/alert/e4/brief").json() == {"status": "pending"}
