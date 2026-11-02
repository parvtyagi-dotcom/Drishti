"""Live camera HTTP API: registration, control, health and video streaming.

Video reaches the browser as multipart MJPEG rather than a finished MP4. That
is the whole point of Live Mode: Forensic Mode can only show you a result
after the job completes, because the thing it produces *is* a file. An MJPEG
response is an open connection the server keeps appending frames to, so the
operator sees the current frame, not a finished artifact.

MJPEG (over WebRTC) is a deliberate trade: it renders in a plain <img> tag
with no player library, no signalling and no codec negotiation, which is what
makes it viable to build and demo reliably. WebRTC would cut bandwidth and
latency further and is the natural upgrade once the rest is stable.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from backend.api.storage import NotFoundError, find_reference_photos, load_zone, new_id, save_zone
from backend.api.network import lan_addresses, reachable_base_url
from backend.config import EVIDENCE_DIR
from backend.core.calibration import (
    CalibrationError, GroundCalibration, delete_calibration, load_calibration, save_calibration,
)
from backend.core.video import DEFAULT_MAX_ANALYSIS_WIDTH, PUSH_SCHEME, PushStreamReader
from backend.live.camera_manager import AnalysisSpec, camera_manager
from backend.modules.zone.geometry import Zone

logger = logging.getLogger("backend.api.live")

router = APIRouter(prefix="/api/live", tags=["live"])

_BOUNDARY = "frame"
DEFAULT_STREAM_FPS = 15.0
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# How long a viewer waits for a real frame before being shown a status card
# instead. Restarting a camera's analysis reloads the face and appearance
# models, which takes tens of seconds on CPU; sending nothing during that time
# leaves the browser holding the last stale image, which is indistinguishable
# from a crashed feed.
_STATUS_FRAME_AFTER_SEC = 1.5
_STATUS_FRAME_INTERVAL_SEC = 1.0
_status_frame_cache: dict[tuple[str, str, int, int], bytes] = {}


def _status_frame(headline: str, detail: str, width: int = 960, height: int = 540) -> bytes:
    """A rendered 'what is happening' card, encoded as a JPEG.

    Produced server-side rather than by the browser so that anything consuming
    the stream - the dashboard, a plain <img>, a recording - sees the same
    explanation instead of a frozen picture.
    """
    key = (headline, detail, width, height)
    cached = _status_frame_cache.get(key)
    if cached is not None:
        return cached

    import cv2
    import numpy as np

    canvas = np.full((height, width, 3), 12, np.uint8)
    canvas[:, :, 0] = 22  # slight blue cast, matching the dashboard
    cv2.putText(canvas, headline, (40, height // 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (235, 235, 235), 2, cv2.LINE_AA)
    if detail:
        cv2.putText(canvas, detail, (40, height // 2 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 160, 170), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    encoded = buf.tobytes() if ok else b""
    _status_frame_cache[key] = encoded
    return encoded


def _source_missing(session) -> bool:
    """Whether this camera's footage simply is not there.

    Only meaningful for a file-backed camera: a URL or a phone that cannot be
    reached is a connection problem, which may fix itself, while a file that
    does not exist never will. Worth separating so the feed can say which.
    """
    source = session.camera.source
    if source.startswith(PUSH_SCHEME) or "://" in source or source.strip().isdigit():
        return False
    return not Path(source).is_file()


def _status_for(session) -> tuple[str, str]:
    """What to tell a viewer when no live frame is arriving."""
    status = session.status_dict()
    # Checked before the error below, because a missing file surfaces as a
    # capture failure whose message explains far less than saying plainly
    # that there is nothing connected.
    if _source_missing(session):
        return "NO SIGNAL", "No footage is connected to this camera"
    if session.analysis_error:
        return "Analysis failed to start", str(session.analysis_error)[:70]
    if not status["running"]:
        return "Camera stopped", "Press Start to resume this feed"
    if status["state"] in ("offline", "reconnecting"):
        return "Reconnecting to camera", status.get("error") or "The feed dropped; retrying"
    if status["modules"]:
        return "Starting analysis", "Loading models for: " + ", ".join(status["modules"])
    return "Connecting to camera", "Waiting for the first frame"


class CameraCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    source: str = Field(min_length=1, description="rtsp:// URL, or a local file path used as a stand-in camera")
    loop: bool = Field(default=False, description="Restart a file source at EOF so it behaves like a continuous camera")
    location: str | None = Field(default=None, max_length=120)
    start: bool = Field(default=True, description="Begin capturing immediately")
    max_width: int = Field(
        default=DEFAULT_MAX_ANALYSIS_WIDTH, ge=320, le=3840,
        description="Frames are downscaled to this width before analysis. Raise it for ANPR, "
                    "where readable plates need pixels; lower it for more speed.",
    )
    rotate: Literal[0, 90, 180, 270] = Field(
        default=0, description="Clockwise rotation for a sideways picture, e.g. a phone used as a webcam",
    )


_rotate_lock = threading.Lock()


class RotateRequest(BaseModel):
    degrees: Literal[0, 90, 180, 270]


class AnalysisConfigRequest(BaseModel):
    """Which modules this camera should run. Mirrors a Forensic Mode job
    request, because a camera is the same analysis pointed at a stream."""

    person_id: bool = False
    reference_set_id: str | None = Field(
        default=None, description="Reference photo set from POST /api/references; required when person_id is on",
    )
    anpr: bool = False
    target_plate: str | None = None
    zone_ids: list[str] = Field(default_factory=list)
    behavior: bool = False
    lowlight: bool = False
    fog: bool = False


class PhoneCameraRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    location: str | None = Field(default=None, max_length=120)


class ZoneCreateOnCameraRequest(BaseModel):
    label: str = Field(default="RESTRICTED ZONE", max_length=80)
    shape: str = Field(default="polygon", pattern="^(rectangle|polygon)$")
    points: list[tuple[float, float]] = Field(
        description="Normalized [0,1] coordinates; rectangle takes 2 opposite corners, polygon 3+",
    )


def _require(camera_id: str):
    session = camera_manager.get(camera_id)
    if session is None:
        raise HTTPException(404, f"camera not found: {camera_id}")
    return session


@router.get("/cameras")
def list_cameras() -> list[dict]:
    return [s.status_dict() for s in camera_manager.list()]


@router.post("/cameras", status_code=201)
def create_camera(req: CameraCreateRequest) -> dict:
    camera = camera_manager.add(
        name=req.name, source=req.source, loop=req.loop, location=req.location,
        max_width=req.max_width, rotate=req.rotate,
    )
    session = camera_manager.get(camera.camera_id)
    if req.start:
        session.start()
    return session.status_dict()


@router.post("/cameras/{camera_id}/rotate")
def rotate_camera(camera_id: str, req: RotateRequest) -> dict:
    """Turn a sideways picture upright. Restarts capture so the new rotation
    applies from the next frame; analysis settings are kept."""
    session = _require(camera_id)
    # Two rotate clicks arriving together would each stop and reopen the same
    # webcam at once, and a USB webcam opened twice in parallel crashed the
    # whole process inside the driver.
    with _rotate_lock:
        if session.camera.rotate == req.degrees:
            return session.status_dict()
        was_running = session.is_running
        session.stop()
        session.camera.rotate = req.degrees
        if was_running:
            session.start()
    return session.status_dict()


# Accuracy: more pixels for reading plates and far faces, at fewer frames.
# Speed: fewer pixels, more frames, for a busy CPU or close-up faces.
QUALITY_PRESETS = {"accuracy": (1280, 10.0), "balanced": (DEFAULT_MAX_ANALYSIS_WIDTH, 12.0), "speed": (640, 15.0)}


class QualityRequest(BaseModel):
    mode: Literal["accuracy", "balanced", "speed"]


@router.post("/cameras/{camera_id}/quality")
def set_camera_quality(camera_id: str, req: QualityRequest) -> dict:
    """Trade analysis resolution against frame rate for one camera.
    Restarts capture in the background; analysis settings are kept."""
    session = _require(camera_id)
    session.camera.max_width, session.camera.target_fps = QUALITY_PRESETS[req.mode]
    session.configure_analysis(session.analysis)
    return session.status_dict()


@router.get("/cameras/{camera_id}")
def get_camera(camera_id: str) -> dict:
    return _require(camera_id).status_dict()


@router.post("/cameras/{camera_id}/start")
def start_camera(camera_id: str) -> dict:
    session = _require(camera_id)
    session.start()
    return session.status_dict()


@router.post("/cameras/{camera_id}/stop")
def stop_camera(camera_id: str) -> dict:
    session = _require(camera_id)
    session.stop()
    return session.status_dict()


@router.delete("/cameras/{camera_id}", status_code=204)
def delete_camera(camera_id: str) -> Response:
    if not camera_manager.remove(camera_id):
        raise HTTPException(404, f"camera not found: {camera_id}")
    return Response(status_code=204)


@router.post("/cameras/{camera_id}/simulate-fault")
def simulate_fault(camera_id: str, seconds: float = 20.0) -> dict:
    """Take a camera's feed down for real, then let it recover on its own.

    For rehearsing and demonstrating the outage path. The stream is genuinely
    severed and the ordinary reconnect logic runs, so what you see is the real
    recovery behaviour rather than a staged status change.
    """
    session = _require(camera_id)
    if not session.is_running:
        raise HTTPException(409, "camera is not running")
    applied = session.inject_fault(seconds)
    return {**session.status_dict(), "fault_seconds": applied}


@router.put("/cameras/{camera_id}/analysis")
def configure_analysis(camera_id: str, req: AnalysisConfigRequest) -> dict:
    """Set which analysis modules run on this camera.

    Applied by restarting the session: the pipeline builds trackers, reference
    galleries and zone monitors once at startup, so modules cannot be swapped
    mid-run without leaving half-initialised state behind.
    """
    session = _require(camera_id)

    reference_paths = []
    if req.person_id:
        if not req.reference_set_id:
            raise HTTPException(400, "person_id requires a reference_set_id - upload photos via POST /api/references")
        try:
            reference_paths = find_reference_photos(req.reference_set_id)
        except NotFoundError as exc:
            raise HTTPException(404, str(exc))

    zones = []
    for zone_id in req.zone_ids:
        try:
            zones.append(load_zone(zone_id))
        except NotFoundError as exc:
            raise HTTPException(404, str(exc))

    spec = AnalysisSpec(
        person_id=req.person_id, reference_photo_paths=reference_paths,
        reference_set_id=req.reference_set_id if req.person_id else None,
        anpr=req.anpr, target_plate=req.target_plate,
        zones=zones, behavior=req.behavior, lowlight=req.lowlight, fog=req.fog,
    )
    session.configure_analysis(spec)
    return session.status_dict()


@router.post("/cameras/{camera_id}/zones", status_code=201)
def create_camera_zone(camera_id: str, req: ZoneCreateOnCameraRequest) -> dict:
    """Draw a restricted zone on a camera's field of view.

    Points are normalized, so a zone drawn on a snapshot stays correct
    regardless of the resolution the stream is later analysed at.
    """
    _require(camera_id)
    try:
        zone = Zone(
            zone_id=new_id(), video_id=camera_id, label=req.label,
            shape=req.shape, points=[tuple(p) for p in req.points],
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    save_zone(zone)
    return zone.to_dict()


class CalibrationRequest(BaseModel):
    points: list[tuple[float, float]] = Field(min_length=4, max_length=4)
    width_m: float
    depth_m: float


@router.get("/cameras/{camera_id}/calibration")
def get_camera_calibration(camera_id: str) -> dict:
    """The camera's ground calibration, or {"calibrated": false}."""
    _require(camera_id)
    calibration = load_calibration(camera_id)
    return {"calibrated": False} if calibration is None else {"calibrated": True, **calibration.to_dict()}


@router.put("/cameras/{camera_id}/calibration")
def set_camera_calibration(camera_id: str, req: CalibrationRequest) -> dict:
    """Mark a rectangle on the ground and give its real size, so behaviour
    analytics measures speed and distance in metres. Takes effect the next
    time the camera's analysis is applied (or restarted)."""
    _require(camera_id)
    try:
        calibration = GroundCalibration(points=[tuple(p) for p in req.points], width_m=req.width_m, depth_m=req.depth_m)
    except CalibrationError as exc:
        raise HTTPException(400, str(exc))
    save_calibration(camera_id, calibration)
    return {"calibrated": True, **calibration.to_dict()}


@router.delete("/cameras/{camera_id}/calibration")
def clear_camera_calibration(camera_id: str) -> dict:
    _require(camera_id)
    delete_calibration(camera_id)
    return {"calibrated": False}


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
    return {**session.status_dict(), "join_path": f"/api/live/phone?camera_id={camera.camera_id}"}


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
        "siren_url": f"{base}/api/live/siren" if base else None,
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


@router.get("/audit")
def audit_log(limit: int = 100, camera_id: str | None = None,
              severity: str | None = None, alarms_only: bool = False) -> list[dict]:
    """The durable event record, newest first.

    Unlike /events - which serves the in-memory tail for the live dashboard -
    this reads from the store and survives restarts.
    """
    return camera_manager.store.query(
        limit=limit, camera_id=camera_id, severity=severity, alarms_only=alarms_only,
    )


@router.get("/audit/verify")
def verify_audit_chain() -> dict:
    """Recompute the hash chain over every stored event.

    Answers one question: has this record been altered since it was written?
    A modified, deleted or reordered row cannot keep the chain consistent, so
    the first inconsistency is reported with the record it occurs at.
    """
    status = camera_manager.store.verify_chain()
    anchor = camera_manager.store.read_anchor()
    if status.valid:
        summary = f"All {status.total_events} records verified - the log has not been altered."
    elif status.broken_at_id is not None:
        summary = f"Chain broken at record {status.broken_at_id}: {status.detail}"
    else:
        summary = status.detail or "The log failed verification."
    return {
        **status.as_dict(),
        "summary": summary,
        "anchor": anchor,
        # Stated plainly rather than implied. An integrity feature that is
        # vague about its limits invites more trust than it has earned.
        "guarantees": {
            "detects": [
                "any record altered after it was written",
                "a record removed or inserted in the middle of the log",
                "records reordered",
                "the most recent records deleted (via the separate head anchor)",
            ],
            "does_not_detect": [
                "an attacker with write access to both the database and the "
                "head anchor who recomputes the whole chain consistently",
            ],
            "note": "Detecting a fully recomputed chain needs the head hash "
                    "published somewhere outside this machine - an append-only "
                    "external service, a second site, or a permissioned ledger. "
                    "That is the point at which a distributed ledger genuinely "
                    "adds something a local hash chain cannot.",
        },
    }


@router.get("/evidence")
def list_evidence(camera_id: str | None = None, limit: int = 50) -> list[dict]:
    """Evidence bundles on disk, newest first."""
    root = EVIDENCE_DIR / "live"
    if not root.is_dir():
        return []
    camera_dirs = [root / camera_id] if camera_id else [d for d in root.iterdir() if d.is_dir()]
    # Only this account's cameras (see backend.api.ownership).
    camera_dirs = [d for d in camera_dirs if camera_manager.get(d.name) is not None]

    bundles = []
    for camera_dir in camera_dirs:
        if not camera_dir.is_dir():
            continue
        for bundle in camera_dir.iterdir():
            if not bundle.is_dir():
                continue
            event = {}
            event_file = bundle / "event.json"
            if event_file.is_file():
                try:
                    event = json.loads(event_file.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    event = {}
            clip = bundle / "clip.mp4"
            # Three outcomes, not two. A clip that is still being assembled and
            # one that can never exist - a camera-down alert has no footage of
            # itself - look identical from the filesystem unless the manifest
            # is consulted, and showing both as "pending" leaves bundles
            # looking permanently stuck.
            if clip.is_file():
                clip_state = "ready"
            elif (bundle / "clip.json").is_file():
                clip_state = "unavailable"
            else:
                clip_state = "pending"
            bundles.append({
                "bundle_id": bundle.name,
                "camera_id": camera_dir.name,
                "created_at": bundle.stat().st_mtime,
                "event_type": event.get("type"),
                "severity": event.get("severity"),
                "reason": event.get("reason"),
                "camera_name": event.get("camera_name"),
                "has_snapshot": (bundle / "snapshot.jpg").is_file(),
                # A clip appears a few seconds after the event: the tail has to
                # be recorded before it can be assembled.
                "has_clip": clip.is_file(),
                "clip_state": clip_state,
                "clip_size_bytes": clip.stat().st_size if clip.is_file() else 0,
            })

    bundles.sort(key=lambda b: b["created_at"], reverse=True)
    return bundles[:limit]


@router.get("/evidence/{camera_id}/{bundle_id}/{asset}", include_in_schema=False)
def get_evidence_asset(camera_id: str, bundle_id: str, asset: str) -> FileResponse:
    allowed = {
        "snapshot.jpg": "image/jpeg",
        "clip.mp4": "video/mp4",
        "event.json": "application/json",
        "clip.json": "application/json",
    }
    media_type = allowed.get(asset)
    if media_type is None:
        raise HTTPException(404, "unknown evidence asset")
    # Basename-only on every segment: these are URL path parameters, and an
    # evidence store is exactly the kind of place a traversal would be aimed at.
    _require(camera_id)
    path = EVIDENCE_DIR / "live" / Path(camera_id).name / Path(bundle_id).name / asset
    if not path.is_file():
        raise HTTPException(404, "evidence asset not found")
    return FileResponse(path, media_type=media_type)


@router.get("/events")
def list_events(limit: int = 100, camera_id: str | None = None) -> list[dict]:
    """Recent events across the fleet - the backlog a dashboard renders on
    open, before the websocket starts delivering new ones."""
    return camera_manager.recent_events(limit=limit, camera_id=camera_id)


@router.websocket("/ws/events")
async def events_websocket(websocket: WebSocket):
    """Push events to operators as they happen.

    Analysis runs on camera threads while this endpoint is async, so events
    cross that boundary through a queue handed over with
    call_soon_threadsafe. The queue is bounded and drops oldest-first: a
    stalled browser must never be able to block the analysis thread behind it.
    """
    await websocket.accept()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=200)

    def enqueue(event: dict) -> None:
        def put() -> None:
            if queue.full():
                try:
                    queue.get_nowait()  # drop the oldest; newest matters most
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(event)

        loop.call_soon_threadsafe(put)

    # Subscribe before reading the backlog so nothing that fires in between is
    # missed. That ordering can deliver an event twice - once in the backlog,
    # once from the queue - so ids already sent are skipped exactly once.
    camera_manager.subscribe(enqueue)
    try:
        backlog = camera_manager.recent_events(limit=25)
        already_sent = {e.get("event_id") for e in backlog if e.get("event_id")}
        for event in backlog:
            await websocket.send_json(event)
        while True:
            event = await queue.get()
            event_id = event.get("event_id")
            if event_id in already_sent:
                already_sent.discard(event_id)
                continue
            await websocket.send_json(event)
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("event websocket failed")
    finally:
        camera_manager.unsubscribe(enqueue)


@router.get("/cameras/{camera_id}/snapshot")
def camera_snapshot(camera_id: str) -> Response:
    """Most recent frame as a single JPEG - used for thumbnails and as the
    still image the operator draws zone polygons on."""
    session = _require(camera_id)
    jpeg, _seq = session.wait_for_frame(last_seq=-1)
    if jpeg is None:
        raise HTTPException(409, "camera has not produced a frame yet")
    return Response(content=jpeg, media_type="image/jpeg")


@router.get("/cameras/{camera_id}/stream")
def camera_stream(camera_id: str, fps: float = DEFAULT_STREAM_FPS) -> StreamingResponse:
    """Live MJPEG feed.

    `fps` throttles what is *sent*, not what is captured or analysed - the
    session keeps running at full rate for the AI regardless. Monitoring
    eyes gain nothing from 60fps, and an uncapped feed at source rate costs
    several MB/s per viewer, so the default is deliberately well below camera
    rate and can be raised per viewer when someone wants smoother motion.
    """
    session = _require(camera_id)
    min_interval = 1.0 / max(1.0, min(fps, 60.0))

    def part(jpeg: bytes) -> bytes:
        return (
            f"--{_BOUNDARY}\r\n"
            f"Content-Type: image/jpeg\r\n"
            f"Content-Length: {len(jpeg)}\r\n\r\n"
        ).encode() + jpeg + b"\r\n"

    def generate():
        seq = -1  # -1 so the very first iteration returns the current frame immediately
        last_sent = 0.0
        last_real_frame = time.monotonic()
        last_status_sent = 0.0

        while True:
            jpeg, seq = session.wait_for_frame(seq)
            now = time.monotonic()

            if jpeg is None or now - last_real_frame > _STATUS_FRAME_AFTER_SEC:
                # Either nothing has ever arrived, or the feed has gone quiet.
                # Keep sending something so the connection stays alive and the
                # viewer is told why the picture is not moving, rather than
                # being left with a frozen image and no explanation.
                if jpeg is None and now - last_status_sent >= _STATUS_FRAME_INTERVAL_SEC:
                    last_status_sent = now
                    headline, detail = _status_for(session)
                    yield part(_status_frame(headline, detail))
                if jpeg is None:
                    # Only give up once the session has actually stopped; a
                    # restarting camera is expected to be quiet for a while.
                    if not session.is_running:
                        break
                    continue

            last_real_frame = now
            if now - last_sent < min_interval:
                continue  # drop, don't queue: stale frames are worthless live
            last_sent = now
            yield part(jpeg)

    return StreamingResponse(
        generate(),
        media_type=f"multipart/x-mixed-replace; boundary={_BOUNDARY}",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
            # Proxies that buffer would defeat the point of a live stream.
            "X-Accel-Buffering": "no",
        },
    )
