"""Routes for adding IP cameras by ONVIF scan instead of typing an RTSP URL.

    POST /api/live/onvif/scan       cameras that answered on this network
    POST /api/live/onvif/profiles   that camera's video streams, after login
    POST /api/live/onvif/add        resolve the stream address and add the camera

The stream address, which carries the camera's login, is built and stored on
the server and never sent back to the browser. Only addresses on private
local networks are contacted: the scan finds cameras on the post's own
network, and nothing here should become a way to make the server fetch
arbitrary URLs.
"""
from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.api.auth import ADMIN, OPERATOR, User, require_role
from backend.api.network import lan_addresses
from backend.core.video import DEFAULT_MAX_ANALYSIS_WIDTH
from backend.live.camera_manager import camera_manager
from backend.live.onvif import OnvifCamera, OnvifError, discover

router = APIRouter(prefix="/api/live/onvif", tags=["live"])

CAN_ADD = Depends(require_role(ADMIN, OPERATOR))


class CameraLogin(BaseModel):
    xaddr: str = Field(min_length=1, max_length=300)
    username: str = Field(default="", max_length=100)
    password: str = Field(default="", max_length=200)


class AddOnvifCamera(CameraLogin):
    profile_token: str = Field(min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=80)
    location: str | None = Field(default=None, max_length=120)
    max_width: int = Field(default=DEFAULT_MAX_ANALYSIS_WIDTH, ge=320, le=3840)


def _local_camera_address(xaddr: str) -> str:
    """Accept only http(s) addresses on a private network."""
    parts = urlsplit(xaddr)
    try:
        host = ipaddress.ip_address(parts.hostname or "")
    except ValueError:
        raise HTTPException(400, "use the camera address found by the scan (an IP address on this network)")
    if parts.scheme not in ("http", "https") or not (host.is_private and not host.is_loopback):
        raise HTTPException(400, "only cameras on this local network can be added this way")
    return xaddr


def _camera(req: CameraLogin) -> OnvifCamera:
    return OnvifCamera(_local_camera_address(req.xaddr), req.username, req.password)


@router.post("/scan")
def scan(_: User = CAN_ADD) -> dict:
    interfaces = lan_addresses()
    cameras = discover(interfaces)
    return {"interfaces": interfaces, "cameras": [c.to_dict() for c in cameras]}


@router.post("/profiles")
def profiles(req: CameraLogin, _: User = CAN_ADD) -> list[dict]:
    try:
        return [p.to_dict() for p in _camera(req).profiles()]
    except OnvifError as exc:
        raise HTTPException(502, str(exc)) from exc


@router.post("/add", status_code=201)
def add(req: AddOnvifCamera, _: User = CAN_ADD) -> dict:
    try:
        source = _camera(req).stream_uri(req.profile_token)
    except OnvifError as exc:
        raise HTTPException(502, str(exc)) from exc
    camera = camera_manager.add(name=req.name, source=source, location=req.location, max_width=req.max_width)
    session = camera_manager.get(camera.camera_id)
    session.start()
    # status_dict reports the source kind, never the URL with its login.
    return session.status_dict()
