"""Guards' verdicts on alarms, which teach the false-alarm memory.

    POST   /api/live/events/{event_id}/verdict   {"verdict": "false" | "real", "note": "..."}
    GET    /api/live/false-alarms                what has been learnt
    DELETE /api/live/false-alarms                forget everything (admin)

Each verdict is also written to the tamper-evident log, so who silenced what
is always on record.
"""
from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.api.auth import ADMIN, OPERATOR, User, require_role
from backend.live.camera_manager import camera_manager
from backend.live.false_alarms import memory

router = APIRouter(prefix="/api/live", tags=["live"])


class Verdict(BaseModel):
    verdict: str = Field(pattern="^(false|real)$")
    note: str | None = Field(default=None, max_length=120)


@router.post("/events/{event_id}/verdict")
def give_verdict(event_id: str, req: Verdict, user: User = Depends(require_role(ADMIN, OPERATOR))) -> dict:
    event = next((e for e in reversed(camera_manager.recent_events(limit=500))
                  if e.get("event_id") == event_id), None)
    if event is None:
        raise HTTPException(404, "that event is no longer in the recent list")
    if event.get("severity") not in ("HIGH", "CRITICAL"):
        raise HTTPException(400, "only alarms can be marked")
    result = memory.mark(event, req.verdict, by=user.username, note=req.note)
    camera_manager.store.record({
        "event_id": uuid.uuid4().hex[:12], "type": "ALARM_VERDICT",
        "camera_id": event.get("camera_id"), "camera_name": event.get("camera_name"),
        "severity": "INFO", "alarm": False, "wall_time": time.time(),
        "reason": f"{user.username} marked {event.get('type')} as {req.verdict}"
                  + (f": {req.note}" if req.note else ""),
        "data": {"alarm_event_id": event_id, **result},
    })
    return result


@router.get("/false-alarms")
def learnt() -> dict:
    return {"patterns": memory.patterns(), "min_marks": memory.MIN_MARKS, "keep_days": memory.KEEP_DAYS}


@router.delete("/false-alarms")
def forget(_: User = Depends(require_role(ADMIN))) -> dict:
    return {"forgotten": memory.clear()}
