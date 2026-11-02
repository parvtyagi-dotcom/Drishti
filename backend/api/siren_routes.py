"""What the siren phone shows beside the alarm: the picture and the AI brief.

    GET /api/live/siren/alert/{event_id}/snapshot.jpg
    GET /api/live/siren/alert/{event_id}/brief

Public, like the siren page, because the phone cannot sign in. To keep that
narrow, only alarm events from the last 15 minutes are served, looked up by
their random event id, and only the snapshot (never the clip or the files
around it). After that window the picture is only in Evidence, behind a login.
"""
from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.api import agent as agent_api
from backend.config import EVIDENCE_DIR
from backend.live.camera_manager import camera_manager

router = APIRouter(prefix="/api/live/siren/alert", tags=["live"])

WINDOW_SEC = 15 * 60


def _recent_alarm(event_id: str) -> dict:
    now = time.time()
    for event in reversed(camera_manager.recent_events(limit=500)):
        if event.get("event_id") == event_id:
            if event.get("alarm") and now - (event.get("wall_time") or 0) <= WINDOW_SEC:
                return event
            break
    raise HTTPException(404, "no recent alarm with that id")


@router.get("/{event_id}/snapshot.jpg", include_in_schema=False)
def alarm_snapshot(event_id: str) -> FileResponse:
    event = _recent_alarm(event_id)
    bundle = event.get("evidence_bundle")
    if not bundle:
        raise HTTPException(404, "this alarm has no picture")
    path = EVIDENCE_DIR / "live" / Path(str(event.get("camera_id"))).name / Path(bundle).name / "snapshot.jpg"
    if not path.is_file():
        raise HTTPException(404, "picture not written yet")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/{event_id}/brief")
def alarm_brief(event_id: str) -> dict:
    _recent_alarm(event_id)
    service = agent_api._service
    if service is None:
        return {"status": "off"}
    brief = service.brief_for_event(event_id)
    if brief is None:
        return {"status": "pending"}
    return {"status": brief["status"], "text": brief["text"], "model_kind": brief["model_kind"]}
