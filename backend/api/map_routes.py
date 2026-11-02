"""The site map: where each camera is, drawn over a plan of the post.

    GET    /api/live/map                 one site's positions, plan state and the list of sites
    PUT    /api/live/map/positions       place (or remove) one camera
    POST   /api/live/map/image           upload the site plan or satellite picture
    GET    /api/live/map/image           the plan itself
    POST   /api/live/map/sites           add a site
    PATCH  /api/live/map/sites/{id}      rename a site
    DELETE /api/live/map/sites/{id}      remove a site and its plan

A deployment can watch more than one site (two campuses, a post and its
outstation), so each site has its own plan and its own camera positions. The
calls above take ``?site=<id>``; without it they act on the default site,
which is how the single-map API before sites behaved and still behaves.

Positions are keyed by camera name, not id: cameras are registered again
after a restart and get new ids, while the name an operator gave a camera
("Gate 3") stays the same. Coordinates are fractions of the plan's width and
height, so they survive a plan being replaced by a sharper picture.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.api.auth import ADMIN, OPERATOR, User, require_role
from backend.config import DATA_DIR
from backend.live.camera_manager import camera_manager
from backend.live.prediction import predict

router = APIRouter(prefix="/api/live/map", tags=["live"])

CAN_EDIT = Depends(require_role(ADMIN, OPERATOR))
MAX_IMAGE_BYTES = 15 * 1024 * 1024
# Long side of the stored plan. Larger adds bytes to every dashboard load
# without making a camera marker any easier to place.
MAX_IMAGE_SIDE = 2400
# The site that existed before there were sites. Its plan keeps the old file
# name, so upgrading moves nothing on disk.
DEFAULT_SITE = "main"

_lock = threading.Lock()


def _state_path() -> Path:
    return DATA_DIR / "site_map.json"


def _image_path(site_id: str) -> Path:
    if site_id == DEFAULT_SITE:
        return DATA_DIR / "site_map.jpg"
    return DATA_DIR / "site_maps" / f"{site_id}.jpg"


def _load() -> dict:
    try:
        state = json.loads(_state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    if "sites" not in state:
        # The single-map file from before sites becomes the default site.
        state = {
            "default": DEFAULT_SITE,
            "sites": {
                DEFAULT_SITE: {
                    "name": "Main site",
                    "image_version": state.get("image_version", 0),
                    "positions": state.get("positions", {}),
                }
            },
        }
    state.setdefault("default", next(iter(state["sites"])))
    return state


def _save(state: dict) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write then rename, so a crash mid-write never leaves a half-written map.
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def _site(state: dict, site_id: str | None) -> tuple[str, dict]:
    sid = site_id or state["default"]
    if sid not in state["sites"]:
        raise HTTPException(404, "no such site")
    return sid, state["sites"][sid]


def _summary(state: dict) -> list[dict]:
    return [
        {"id": sid, "name": s["name"], "has_image": _image_path(sid).is_file()}
        for sid, s in state["sites"].items()
    ]


class Placement(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    x: float | None = Field(default=None, ge=0.0, le=1.0)
    y: float | None = Field(default=None, ge=0.0, le=1.0)


class SiteName(BaseModel):
    name: str = Field(min_length=1, max_length=80)


@router.get("")
def get_map(site: str | None = None) -> dict:
    with _lock:
        state = _load()
    sid, s = _site(state, site)
    return {
        "site_id": sid,
        "site_name": s["name"],
        "has_image": _image_path(sid).is_file(),
        "image_version": s["image_version"],
        "positions": s["positions"],
        "sites": _summary(state),
    }


@router.put("/positions")
def place(req: Placement, site: str | None = None, _: User = CAN_EDIT) -> dict:
    """Set a camera's position; send x and y as null to take it off the map."""
    with _lock:
        state = _load()
        _, s = _site(state, site)
        if req.x is None or req.y is None:
            s["positions"].pop(req.name, None)
        else:
            s["positions"][req.name] = [round(req.x, 4), round(req.y, 4)]
        _save(state)
    return {"positions": s["positions"]}


@router.post("/image")
async def upload_image(file: UploadFile, site: str | None = None, _: User = CAN_EDIT) -> dict:
    with _lock:
        sid, _s = _site(_load(), site)
    data = await file.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "the picture is larger than 15 MB; use a smaller export")
    # Decode and re-encode rather than storing the upload as-is: it must
    # really be a picture, and only plain JPEG pixels are ever served back.
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "that file is not a picture this system can read (use JPG or PNG)")
    height, width = image.shape[:2]
    scale = MAX_IMAGE_SIDE / max(height, width)
    if scale < 1:
        image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise HTTPException(500, "could not save the picture")
    image_path = _image_path(sid)
    with _lock:
        state = _load()
        if sid not in state["sites"]:
            raise HTTPException(404, "no such site")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(encoded.tobytes())
        state["sites"][sid]["image_version"] += 1
        _save(state)
    return {
        "has_image": True,
        "image_version": state["sites"][sid]["image_version"],
        "width": image.shape[1],
        "height": image.shape[0],
    }


@router.get("/image")
def get_image(site: str | None = None) -> FileResponse:
    with _lock:
        sid, _s = _site(_load(), site)
    image_path = _image_path(sid)
    if not image_path.is_file():
        raise HTTPException(404, "no site plan uploaded yet")
    return FileResponse(image_path, media_type="image/jpeg")


@router.post("/sites")
def add_site(req: SiteName, _: User = CAN_EDIT) -> dict:
    sid = uuid.uuid4().hex[:8]
    with _lock:
        state = _load()
        state["sites"][sid] = {"name": req.name.strip(), "image_version": 0, "positions": {}}
        _save(state)
    return {"id": sid, "name": req.name.strip(), "sites": _summary(state)}


@router.patch("/sites/{site_id}")
def rename_site(site_id: str, req: SiteName, _: User = CAN_EDIT) -> dict:
    with _lock:
        state = _load()
        _, s = _site(state, site_id)
        s["name"] = req.name.strip()
        _save(state)
    return {"sites": _summary(state)}


@router.delete("/sites/{site_id}")
def delete_site(site_id: str, _: User = CAN_EDIT) -> dict:
    with _lock:
        state = _load()
        _site(state, site_id)
        if len(state["sites"]) == 1:
            raise HTTPException(409, "the last site cannot be removed")
        del state["sites"][site_id]
        if state["default"] == site_id:
            state["default"] = next(iter(state["sites"]))
        _save(state)
        _image_path(site_id).unlink(missing_ok=True)
    return {"sites": _summary(state)}


@router.get("/prediction")
def prediction(site: str | None = None) -> dict:
    """The intruder's likely next camera and where to send the patrol, from
    the order of recent alerts across the site's placed cameras."""
    import time
    with _lock:
        _, chosen = _site(_load(), site)
        positions = {k: tuple(v) for k, v in chosen["positions"].items()}
    return {"prediction": predict(camera_manager.recent_events(limit=500), positions, time.time())}
