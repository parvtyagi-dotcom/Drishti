"""Live endpoints for devices at a post: siren, phone camera and pairing.

Split out of backend.api.live, which had grown past the size it could be
reviewed at. These are the pages a phone opens by scanning a QR code, the
frames a phone camera pushes in, the Android APK, and the addresses and
pairing keys the Devices page turns into those QR codes. Same /api/live
prefix and paths as before.
"""
from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.api import device_auth
from backend.api.auth import ADMIN, OPERATOR
from backend.api.network import lan_addresses, reachable_base_url
from backend.core.video import PUSH_SCHEME, PushStreamReader
from backend.live.camera_manager import camera_manager

logger = logging.getLogger("backend.api.live_devices")

router = APIRouter(prefix="/api/live", tags=["live"])

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


class PhoneCameraRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    location: str | None = Field(default=None, max_length=120)


@router.get("/siren", include_in_schema=False)
def siren_page() -> FileResponse:
    """Standalone alarm panel, served straight from the backend.

    Deliberately not part of the React app: the siren runs on a second device
    (a phone at the post), and that device should need nothing but the
    backend's address - no frontend build, no dev server, no bundle. It is
    also the one screen that must still work if the dashboard does not.
    """
    path = STATIC_DIR / "siren.html"
    if not path.is_file():
        raise HTTPException(404, "siren page not installed")
    return FileResponse(path, media_type="text/html")


# The siren installs to a phone's home screen as a PWA, which needs a
# manifest, icons, and a service worker served under a scope that covers the
# page. Building a real APK would mean an Android toolchain and a signed
# binary to distribute; a PWA gets the same result here - own icon, fullscreen,
# vibration, wake lock - with nothing to install from a store.
_SIREN_ASSETS = {
    "manifest.webmanifest": ("siren.webmanifest", "application/manifest+json"),
    "icon-192.png": ("siren-192.png", "image/png"),
    "icon-512.png": ("siren-512.png", "image/png"),
    "icon-maskable.png": ("siren-maskable-512.png", "image/png"),
}


@router.get("/siren/{asset}", include_in_schema=False)
def siren_asset(asset: str) -> FileResponse:
    entry = _SIREN_ASSETS.get(asset)
    if entry is None:
        raise HTTPException(404, "unknown siren asset")
    filename, media_type = entry
    path = STATIC_DIR / filename
    if not path.is_file():
        raise HTTPException(404, "siren asset not installed")
    return FileResponse(path, media_type=media_type)


@router.get("/sw.js", include_in_schema=False)
def siren_service_worker() -> FileResponse:
    """Served from /api/live/ so its scope covers the siren page below it - a
    worker registered deeper than the page it controls would never activate."""
    path = STATIC_DIR / "sw.js"
    if not path.is_file():
        raise HTTPException(404, "service worker not installed")
    return FileResponse(
        path,
        media_type="application/javascript",
        headers={"Service-Worker-Allowed": "/api/live/", "Cache-Control": "no-cache"},
    )


@router.post("/phone-cameras", status_code=201)
def create_phone_camera(req: PhoneCameraRequest) -> dict:
    """Create a camera that a phone browser will push frames into.

    Registered up front so the operator can position and configure it before
    the phone connects; the session simply waits for the first frame.
    """
    camera = camera_manager.add(
        name=req.name, source=f"{PUSH_SCHEME}{uuid.uuid4().hex[:8]}",
        loop=False, location=req.location,
    )
    session = camera_manager.get(camera.camera_id)
    # Rebuild the source id so it matches the camera id the phone will use.
    session.camera.source = f"{PUSH_SCHEME}{camera.camera_id}"
    key = device_auth.device_key(device_auth.ingest_scope(camera.camera_id))
    return {**session.status_dict(), "join_path": f"/api/live/phone?camera_id={camera.camera_id}&key={key}"}


@router.get("/phone", include_in_schema=False)
def phone_capture_page() -> FileResponse:
    """Page a phone opens to become a camera.

    Browsers only expose a device camera on a secure origin, so over a LAN
    address this page needs HTTPS - see scripts/make_dev_cert.py. Where that
    is inconvenient, an IP-camera app pointed at /api/live/cameras works
    instead, with no browser security involved at all.
    """
    path = STATIC_DIR / "phone.html"
    if not path.is_file():
        raise HTTPException(404, "phone capture page not installed")
    return FileResponse(path, media_type="text/html")


@router.websocket("/ws/ingest/{camera_id}")
async def ingest_websocket(websocket: WebSocket, camera_id: str):
    """Receive camera frames pushed from a phone.

    Each websocket message is one JPEG. Frames are dropped rather than queued
    if they cannot be decoded - a phone on flaky WiFi should degrade to a lower
    rate, never stall the analysis thread waiting on it.
    """
    # Anyone who could push frames here could feed the analysis fake video -
    # raising alarms, or hiding a real intruder behind a still picture.
    if not await device_auth.authorize_websocket(
        websocket, device_scope=device_auth.ingest_scope(camera_id), session_roles=(ADMIN, OPERATOR),
    ):
        return
    await websocket.accept()
    session = camera_manager.get(camera_id)
    if session is None:
        await websocket.close(code=4404, reason="camera not found")
        return

    if not isinstance(session.reader, PushStreamReader):
        await websocket.close(code=4400, reason="camera does not accept pushed frames")
        return

    if not session.is_running:
        session.start()

    received = rejected = 0
    try:
        while True:
            data = await websocket.receive_bytes()
            # The reader is resolved per frame rather than bound once, because
            # a session swaps in a fresh one whenever it restarts - an operator
            # pressing Configure, or recovery after an outage. A socket holding
            # the previous reader would go on filling a buffer nothing reads
            # while the live reader starved, and the camera would be declared
            # offline with the phone still happily transmitting.
            reader = session.reader
            if isinstance(reader, PushStreamReader) and reader.push(data):
                received += 1
            else:
                rejected += 1
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("phone ingest failed for %s", camera_id)
    finally:
        logger.info("phone ingest closed for %s: %d frames, %d rejected", camera_id, received, rejected)


APK_PATH = Path(__file__).resolve().parent.parent.parent / "dist" / "Drishti-Siren.apk"


@router.get("/siren-app", include_in_schema=False)
def download_siren_apk() -> FileResponse:
    """Serve the Android siren app for sideloading.

    Public, like the siren page itself: a phone being set up as an alarm has
    no session yet, and requiring one would mean signing in on a device whose
    entire purpose is to sit on a table and make noise.
    """
    if not APK_PATH.is_file():
        raise HTTPException(404, "the Android app has not been built - see siren-app/README")
    return FileResponse(
        APK_PATH, media_type="application/vnd.android.package-archive",
        filename="Drishti-Siren.apk",
    )


@router.get("/connect-info")
def connect_info(request: Request) -> dict:
    """Addresses a phone should use to reach this server.

    Built from the server's own interfaces rather than the browser's URL, so
    the link works even when the dashboard is open on localhost - which would
    otherwise produce a QR code telling the phone to connect to itself.
    """
    scheme = request.url.scheme
    port = request.url.port or (443 if scheme == "https" else 80)
    base = reachable_base_url(scheme, port)
    siren_key = device_auth.device_key(device_auth.SIREN)

    # A phone's browser will not grant camera access over plain http on a LAN
    # address, so pairing needs an https URL. That is independent of how the
    # operator happens to be viewing this page: the server commonly listens on
    # both, and telling someone to enable TLS that is already running is worse
    # than useless. So the https listener is reported separately.
    https_port = os.environ.get("BORDERWATCH_HTTPS_PORT")
    if scheme == "https":
        secure_base, camera_ready = base, True
    elif https_port:
        secure_base = reachable_base_url("https", int(https_port))
        camera_ready = secure_base is not None
    else:
        secure_base, camera_ready = None, False

    return {
        "base_url": base,
        "secure_base_url": secure_base,
        "addresses": lan_addresses(),
        "scheme": scheme,
        "port": port,
        # The pairing key rides in the link the QR code encodes, so a phone
        # that scans it is paired without ever seeing a login form.
        "siren_url": f"{base}/api/live/siren?key={siren_key}" if base else None,
        "siren_key": siren_key,
        "speaker_key": device_auth.device_key(device_auth.SPEAKER),
        "apk_url": f"{base}/api/live/siren-app" if base else None,
        "camera_capable": camera_ready,
        "camera_warning": None if camera_ready else (
            "Pairing a phone as a camera needs https, because phones refuse "
            "camera access to an unencrypted page. Generate a certificate with "
            "scripts/make_dev_cert.py and start the server with scripts/serve.py, "
            "which runs both. The siren and the Android app work over http."
        ),
        "on_network": base is not None,
    }
