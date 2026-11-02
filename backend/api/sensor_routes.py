"""Sensors: fence, ground, radar and acoustic sensors whose alarms the cameras check.

    GET    /api/sensors                      the post's sensors and the kinds on offer
    POST   /api/sensors                      add one (admin); the reply carries its key
    DELETE /api/sensors/{id}                 remove one (admin)
    POST   /api/sensors/{id}/test            fire a test trip (admin, operator)

    POST   /api/sensors/hook/{id}            a real sensor reports a trip, with
                                             its key in the X-Sensor-Key header
    GET    /api/sensors/hook/{id}/info       the sensor's name, for a device holding its key
    GET    /api/sensors/hook/page            the phone page that turns a phone into an
                                             acoustic sensor

The hook paths need no login: a fence controller or a phone in a hut cannot
sign in. Each sensor gets its own random key instead, which only an admin can
see, and which only lets that one sensor report a trip.
"""
from __future__ import annotations

import threading
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.api.auth import ADMIN, OPERATOR, User, current_user, require_role
from backend.live.camera_manager import camera_manager
from backend.live.sensors import SENSOR_KINDS, SensorFusion, SensorRegistry

router = APIRouter(prefix="/api/sensors", tags=["sensors"])

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

_registry: SensorRegistry | None = None
_fusion: SensorFusion | None = None


def registry() -> SensorRegistry:
    global _registry
    if _registry is None:
        _registry = SensorRegistry()
    return _registry


def fusion() -> SensorFusion:
    global _fusion
    if _fusion is None:
        _fusion = SensorFusion(registry(), camera_manager)
    return _fusion


def warm_up() -> None:
    """Load the detector in the background when the post has sensors."""
    if not registry().list():
        return
    look = getattr(fusion().look, "warm", None)
    if look is not None:
        threading.Thread(target=look, name="sensor-warmup", daemon=True).start()


class NewSensor(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    kind: str
    camera: str | None = Field(default=None, max_length=80)


class Trip(BaseModel):
    detail: str | None = Field(default=None, max_length=200)


@router.get("")
def list_sensors(user: User = Depends(current_user)) -> dict:
    show_keys = user.role == ADMIN
    return {
        "sensors": [s.public(with_key=show_keys) for s in registry().list()],
        "kinds": [{"kind": k, "label": v} for k, v in SENSOR_KINDS.items()],
    }


@router.post("")
def add_sensor(req: NewSensor, _: User = Depends(require_role(ADMIN))) -> dict:
    try:
        sensor = registry().add(req.name, req.kind, req.camera)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return sensor.public(with_key=True)


@router.delete("/{sensor_id}")
def remove_sensor(sensor_id: str, _: User = Depends(require_role(ADMIN))) -> dict:
    if not registry().remove(sensor_id):
        raise HTTPException(404, "no such sensor")
    return {"removed": sensor_id}


@router.post("/{sensor_id}/test")
def test_sensor(sensor_id: str, _: User = Depends(require_role(ADMIN, OPERATOR))) -> dict:
    """A rehearsal trip. Runs the real check, but every event it produces is
    marked as a test so the log never mistakes it for a real alarm."""
    sensor = registry().get(sensor_id)
    if sensor is None:
        raise HTTPException(404, "no such sensor")
    return _summary(fusion().trigger(sensor, source="simulated"))


# --- device side (no login; the sensor's key instead) ---------------------------

@router.get("/hook/page", include_in_schema=False)
def sensor_page() -> FileResponse:
    path = STATIC_DIR / "sensor.html"
    if not path.is_file():
        raise HTTPException(404, "sensor page not installed")
    return FileResponse(path, media_type="text/html", headers={"Cache-Control": "no-cache"})


@router.get("/hook/{sensor_id}/info")
def sensor_info(sensor_id: str, x_sensor_key: str | None = Header(default=None)) -> dict:
    sensor = registry().check_key(sensor_id, x_sensor_key)
    if sensor is None:
        raise HTTPException(403, "unknown sensor or wrong key")
    return {"name": sensor.name, "kind": sensor.kind, "label": sensor.label, "camera": sensor.camera}


@router.post("/hook/{sensor_id}")
def sensor_trip(sensor_id: str, req: Trip | None = None,
                x_sensor_key: str | None = Header(default=None)) -> dict:
    sensor = registry().check_key(sensor_id, x_sensor_key)
    if sensor is None:
        raise HTTPException(403, "unknown sensor or wrong key")
    return _summary(fusion().trigger(sensor, source="sensor", detail=req.detail if req else None))


def _summary(result: dict) -> dict:
    event = result.get("event") or {}
    return {"status": result["status"], "event_type": event.get("type"), "reason": event.get("reason")}
