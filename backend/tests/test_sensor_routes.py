"""Sensor routes: admins manage sensors, devices report trips with their key."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.api import sensor_routes
from backend.api.auth import ADMIN, AUDITOR, OPERATOR, User, current_user
from backend.app import app
from backend.live.sensors import SensorFusion, SensorRegistry

client = TestClient(app)


class RecordingFusion(SensorFusion):
    def __init__(self, registry):
        super().__init__(registry, manager=None, look=lambda image: (0, 0, 0.0))
        self.trips = []

    def trigger(self, sensor, source="sensor", detail=None, background=True):
        self.trips.append((sensor.sensor_id, source, detail))
        return {"status": "checking", "event": {"type": "SENSOR_TRIGGERED", "reason": "tripped"}}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    reg = SensorRegistry(tmp_path / "sensors.json")
    fusion = RecordingFusion(reg)
    monkeypatch.setattr(sensor_routes, "_registry", reg)
    monkeypatch.setattr(sensor_routes, "_fusion", fusion)

    def use(name: str):
        app.dependency_overrides[current_user] = lambda: User(username="u", role=name)
    yield use, fusion
    app.dependency_overrides.pop(current_user, None)


def test_admin_adds_sensor_and_sees_key_operator_does_not(setup):
    role, _ = setup
    role(ADMIN)
    made = client.post("/api/sensors", json={"name": "Fence A", "kind": "fence", "camera": "Gate 1"}).json()
    assert made["key"] and made["label"] == "Fence vibration sensor"
    assert client.get("/api/sensors").json()["sensors"][0]["key"] == made["key"]
    role(OPERATOR)
    assert "key" not in client.get("/api/sensors").json()["sensors"][0]
    assert client.post("/api/sensors", json={"name": "B", "kind": "fence"}).status_code == 403


def test_unknown_kind_is_rejected(setup):
    role, _ = setup
    role(ADMIN)
    assert client.post("/api/sensors", json={"name": "B", "kind": "laser"}).status_code == 400


def test_operator_can_test_but_auditor_cannot(setup):
    role, fusion = setup
    role(ADMIN)
    sid = client.post("/api/sensors", json={"name": "Fence A", "kind": "fence"}).json()["sensor_id"]
    role(OPERATOR)
    assert client.post(f"/api/sensors/{sid}/test").json()["status"] == "checking"
    assert fusion.trips == [(sid, "simulated", None)]
    role(AUDITOR)
    assert client.post(f"/api/sensors/{sid}/test").status_code == 403


def test_hook_needs_the_right_key(setup):
    role, fusion = setup
    role(ADMIN)
    made = client.post("/api/sensors", json={"name": "Mic", "kind": "acoustic_drone"}).json()
    app.dependency_overrides.pop(current_user, None)
    sid, key = made["sensor_id"], made["key"]
    assert client.post(f"/api/sensors/hook/{sid}").status_code == 403
    assert client.post(f"/api/sensors/hook/{sid}", headers={"X-Sensor-Key": "nope"}).status_code == 403
    reply = client.post(f"/api/sensors/hook/{sid}", headers={"X-Sensor-Key": key}, json={"detail": "hum 180 Hz"})
    assert reply.status_code == 200
    assert fusion.trips == [(sid, "sensor", "hum 180 Hz")]
    info = client.get(f"/api/sensors/hook/{sid}/info", headers={"X-Sensor-Key": key}).json()
    assert info["label"] == "Acoustic drone sensor"


def test_admin_removes_sensor(setup):
    role, _ = setup
    role(ADMIN)
    sid = client.post("/api/sensors", json={"name": "Fence A", "kind": "fence"}).json()["sensor_id"]
    assert client.delete(f"/api/sensors/{sid}").status_code == 200
    assert client.delete(f"/api/sensors/{sid}").status_code == 404
