"""Talk-down warnings: speaker devices near cameras, and the operator's button.

    GET  /api/live/speaker                    the speaker page (no login: a phone
                                              taped to a pole cannot sign in)
    WS   /api/live/ws/speaker?camera=NAME     a speaker waiting for warnings
    GET  /api/live/talkdown                   automatic mode and connected speakers
    PUT  /api/live/talkdown                   turn automatic warnings on or off
    POST /api/live/cameras/{id}/talkdown      warn now, at that camera's speakers
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.api.auth import ADMIN, OPERATOR, User, require_role
from backend.live.camera_manager import camera_manager
from backend.live.talkdown import TalkDown

logger = logging.getLogger("backend.api.talkdown")
router = APIRouter(prefix="/api/live", tags=["live"])
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

talkdown = TalkDown(camera_manager)
CAN_WARN = Depends(require_role(ADMIN, OPERATOR))


class AutoMode(BaseModel):
    auto: bool


@router.get("/speaker", include_in_schema=False)
def speaker_page() -> FileResponse:
    path = STATIC_DIR / "speaker.html"
    if not path.is_file():
        raise HTTPException(404, "speaker page not installed")
    return FileResponse(path, media_type="text/html", headers={"Cache-Control": "no-cache"})


@router.get("/talkdown")
def talkdown_status() -> dict:
    return {
        "auto": talkdown.auto,
        "speakers": {s.camera.name: talkdown.speakers_for(s.camera.name) for s in camera_manager.list()},
    }


@router.put("/talkdown")
def set_auto(req: AutoMode, _: User = CAN_WARN) -> dict:
    talkdown.auto = req.auto
    return talkdown_status()


@router.post("/cameras/{camera_id}/talkdown")
def warn_now(camera_id: str, user: User = CAN_WARN) -> dict:
    session = camera_manager.get(camera_id)
    if session is None:
        raise HTTPException(404, "camera not found")
    event = talkdown.warn(session, trigger="operator", by=user.username)
    if event is None:
        raise HTTPException(409, f"no speaker is connected for {session.camera.name}; "
                                 "open the speaker page on a device near the camera")
    return {"reason": event["reason"], "speakers": event["data"]["speakers"]}


@router.websocket("/ws/speaker")
async def speaker_websocket(websocket: WebSocket, camera: str = ""):
    await websocket.accept()
    session = camera_manager.find_by_name(camera) if camera else None
    if session is None:
        await websocket.send_json({"type": "error", "detail": f"no camera named '{camera}'"})
        await websocket.close(code=4404)
        return
    name = session.camera.name
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=20)

    def enqueue(event: dict) -> None:
        if event.get("type") == "TALKDOWN_ISSUED" and event.get("camera_name") == name:
            loop.call_soon_threadsafe(lambda: queue.full() or queue.put_nowait(event))

    camera_manager.subscribe(enqueue)
    talkdown.speaker_connected(name)
    try:
        await websocket.send_json({"type": "ready", "camera": name})
        while True:
            # Wake every 20 s even when quiet, so a dead connection is noticed
            # and the speaker count stays honest.
            try:
                event = await asyncio.wait_for(queue.get(), timeout=20)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
                continue
            await websocket.send_json({"type": "warn", "lines": event["data"]["lines"],
                                       "reason": event["reason"]})
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("speaker websocket failed")
    finally:
        talkdown.speaker_disconnected(name)
        camera_manager.unsubscribe(enqueue)
