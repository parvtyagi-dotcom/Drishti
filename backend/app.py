"""Drishti API - thin HTTP layer over the module pipelines.

Two job creation paths:
  - POST /api/zone-jobs: the original Phase 3 zone-only job (kept exactly
    as-is for backward compatibility - existing callers/tests are unaffected).
  - POST /api/jobs: the unified Phase 6 path - enable any subset of
    {person_id, anpr, zone(s), behavior, lowlight} and get ONE job that runs
    them all through backend.orchestrator.CombinedPipeline (one video decode,
    one tracker pass, one output video, one event stream), rather than
    requiring N separate jobs/pipeline runs for N modules.

Errors are translated to clean HTTP responses (404/400/409) with a short
message - never a raw Python traceback. Full tracebacks go to the server
log via `logger.exception` in the unhandled-exception handler.
"""
from __future__ import annotations

import contextlib
import dataclasses
import logging
import os
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.agent.models import build_chain
from backend.agent.service import AgentService
from backend.api.agent import router as agent_router
from backend.api.agent import set_service as set_agent_service
from backend.api.auth import _token_from, decode_token, google_client_id, has_users, is_public_path, no_login
from backend.api.auth_routes import router as auth_router
from backend.api.jobs import job_manager
from backend.api.live import router as live_router
from backend.api.handover import router as handover_router
from backend.api.map_routes import router as map_router
from backend.api.onvif_routes import router as onvif_router
from backend.api.webcam_routes import router as webcam_router
from backend.api.ownership import ownership_middleware
from backend.live.warmup import warm_up_models
from backend.api.sensor_routes import router as sensor_router
from backend.api.sensor_routes import warm_up as warm_up_sensors
from backend.api.talkdown_routes import router as talkdown_router
from backend.api.siren_routes import router as siren_router
from backend.api import search_routes
from backend.api.feedback_routes import router as feedback_router
from backend.api.satellite_routes import router as satellite_router
from backend.api.talkdown_routes import talkdown
from backend.api.schemas import (
    AnprModuleRequest, BehaviorModuleRequest, CombinedJobRequest, EvidenceItemOut,
    JobStatusOut, LowlightModuleRequest, PersonIdModuleRequest, PlateSearchRequest,
    ReferenceUploadOut, SystemStatusOut, VideoInfoOut, VideoUploadOut, ZoneCreateRequest,
    ZoneJobRequest, ZoneOut, ZoneUpdateRequest,
)
from backend.api.storage import (
    NotFoundError, find_reference_photos, find_video_path, load_zone, new_id,
    save_reference_photos, save_uploaded_video, save_zone,
)
from backend.core.events import severity_for, should_alarm
from backend.config import (
    AnprConfig, BehaviorConfig, EVIDENCE_DIR, LowLightConfig, OUTPUTS_DIR, PersonIDConfig,
    SFACE_MODEL_PATH, UPLOADS_DIR, YUNET_MODEL_PATH, YOLO_MODEL_PATH, ZoneConfig, detect_device, get_config,
)
from backend.live.camera_manager import camera_manager, seed_demo_cameras
from backend.core.video import VideoReader, extract_frame
from backend.modules.anpr.ocr import PlateOcr
from backend.modules.zone.geometry import Zone
from backend.modules.zone.pipeline import ZonePipeline
from backend.orchestrator import CombinedPipeline, OrchestratorError, OrchestratorRequest

logger = logging.getLogger("backend.app")


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Live Monitor starts empty and shows only the cameras an operator has
    # actually connected. A stand-in camera looping sample footage looks like
    # a real feed, keeps analysis running, and raises alarms over whatever
    # else someone is doing - including work in Forensic Analysis that has
    # nothing to do with it. Set BORDERWATCH_DEMO_CAMERAS=1 to register the
    # sample feed deliberately, for a rehearsal or a walkthrough.
    if os.environ.get("BORDERWATCH_DEMO_CAMERAS"):
        seeded = seed_demo_cameras(UPLOADS_DIR)
        if seeded:
            logger.info("seeded %d demo camera(s): %s", len(seeded), ", ".join(c.name for c in seeded))
    # Camera health is only useful if something is watching it continuously;
    # this is what turns a dropout into a CAMERA_OFFLINE event.
    camera_manager.start_monitor()
    warm_up_models()
    warm_up_sensors()
    talkdown.start()
    search_routes.start()
    # The AI analyst only reads events and proposes actions, so it starts
    # after the cameras and stops before them; detection never waits on it.
    agent_service = None
    agent_config = get_config().agent
    if agent_config.enabled:
        agent_service = AgentService(agent_config, build_chain(agent_config),
                                     store=camera_manager.store, manager=camera_manager)
        agent_service.start()
        set_agent_service(agent_service)
        logger.info("AI analyst on: %s, falling back to local %s", agent_config.cloud_model, agent_config.local_model)
    yield
    if agent_service is not None:
        set_agent_service(None)
        agent_service.stop()
    talkdown.stop()
    search_routes.stop()
    camera_manager.stop_monitor()
    # Capture threads are daemons, but stopping them explicitly releases the
    # captures instead of leaving them to be torn down mid-read.
    camera_manager.stop_all()


app = FastAPI(title="Drishti API", lifespan=lifespan)
app.include_router(auth_router)
app.include_router(live_router)
app.include_router(onvif_router)
app.include_router(webcam_router)
app.include_router(map_router)
app.include_router(handover_router)
app.include_router(agent_router)
app.include_router(sensor_router)
app.include_router(talkdown_router)
app.include_router(siren_router)
app.include_router(search_routes.router)
app.include_router(feedback_router)
app.include_router(satellite_router)
app.middleware("http")(ownership_middleware)



@app.middleware("http")
async def require_authentication(request: Request, call_next):
    """Gate every endpoint behind a session, with a narrow public list.

    Enforced centrally rather than per-route: with surveillance video and an
    evidence trail behind this API, a route added later must be protected by
    default. An allowlist that someone forgets to extend fails closed; a
    per-route decorator that someone forgets to add fails open.

    Disable only for local development, never on a bound interface:
        BORDERWATCH_DISABLE_AUTH=1
    """
    path = request.url.path
    if (
        os.environ.get("BORDERWATCH_DISABLE_AUTH")
        or no_login()
        or request.method == "OPTIONS"
        or not path.startswith("/api/")
        or is_public_path(path)
    ):
        return await call_next(request)

    token = _token_from(request)
    if token is not None and decode_token(token) is not None:
        return await call_next(request)
    if not has_users() and not google_client_id():
        return JSONResponse(
            {"detail": "no accounts configured - create the first administrator"}, status_code=401,
        )
    return JSONResponse({"detail": "authentication required"}, status_code=401)


@app.exception_handler(NotFoundError)
def _not_found_handler(request: Request, exc: NotFoundError):
    return Response(content=str(exc), status_code=404)


@app.exception_handler(OrchestratorError)
def _orchestrator_error_handler(request: Request, exc: OrchestratorError):
    return JSONResponse({"detail": str(exc)}, status_code=400)


@app.exception_handler(Exception)
def _unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse({"detail": "internal server error"}, status_code=500)


def _job_status_out(job) -> JobStatusOut:
    return JobStatusOut(**job.progress_info())


# --- videos -----------------------------------------------------------------

@app.post("/api/videos", response_model=VideoUploadOut)
async def upload_video(file: UploadFile):
    data = await file.read()
    if not data:
        raise HTTPException(400, "uploaded file is empty")
    video_id, path = save_uploaded_video(file.filename or "upload.mp4", data)
    try:
        with VideoReader(path) as reader:
            info = reader.info
    except Exception:
        path.unlink(missing_ok=True)
        raise HTTPException(400, "could not be read as a video - unsupported or corrupted file")
    if info.frame_count <= 0:
        path.unlink(missing_ok=True)
        raise HTTPException(400, "video has no readable frames")
    return VideoUploadOut(
        video_id=video_id, filename=file.filename or path.name,
        video_info=VideoInfoOut(fps=info.fps, width=info.width, height=info.height, frame_count=info.frame_count, duration_sec=info.duration_sec),
    )


@app.get("/api/videos/{video_id}/frame")
def get_representative_frame(video_id: str, at: float = 0.5):
    path = find_video_path(video_id)
    image, _info = extract_frame(path, fraction=at)
    ok, buf = cv2.imencode(".jpg", image)
    if not ok:
        raise HTTPException(500, "failed to encode representative frame")
    return Response(content=buf.tobytes(), media_type="image/jpeg")


# --- reference photos (Target Person ID) -------------------------------------

@app.post("/api/references", response_model=ReferenceUploadOut)
async def upload_references(files: list[UploadFile]):
    if not files:
        raise HTTPException(400, "at least one reference photo is required")
    if len(files) > 6:
        raise HTTPException(400, "at most 6 reference photos are supported")
    payload = [(f.filename or "ref.jpg", await f.read()) for f in files]
    if any(not data for _, data in payload):
        raise HTTPException(400, "one or more uploaded reference photos were empty")
    ref_id, paths = save_reference_photos(payload)
    # Check the faces now: a set with no usable face would otherwise only fail
    # once a camera starts matching against it. Sideways photos are turned
    # upright on disk so every later use sees the face the right way up.
    from backend.modules.person_id.detector import FaceDetector
    from backend.modules.person_id.recognizer import upright_face
    from backend.config import YUNET_MODEL_PATH

    detector = FaceDetector(YUNET_MODEL_PATH)
    no_face = []
    for (name, _), path in zip(payload, paths):
        image = cv2.imread(str(path))
        upright, face = upright_face(image, detector) if image is not None else (None, None)
        if face is None:
            no_face.append(name)
        elif upright is not image:
            cv2.imwrite(str(path), upright)
    if len(no_face) == len(paths):
        raise HTTPException(
            400, f"no face found in {', '.join(no_face)} - use a clear photo with the face visible",
        )
    return ReferenceUploadOut(reference_set_id=ref_id, photo_count=len(paths) - len(no_face))


# --- zones --------------------------------------------------------------------

@app.post("/api/zones", response_model=ZoneOut)
def create_zone(req: ZoneCreateRequest):
    find_video_path(req.video_id)  # 404s if the video doesn't exist
    zone = Zone(zone_id=new_id(), video_id=req.video_id, label=req.label, shape=req.shape, points=req.points)
    save_zone(zone)
    return ZoneOut(**zone.to_dict())


@app.get("/api/zones/{zone_id}", response_model=ZoneOut)
def get_zone(zone_id: str):
    zone = load_zone(zone_id)
    return ZoneOut(**zone.to_dict())


@app.put("/api/zones/{zone_id}", response_model=ZoneOut)
def update_zone(zone_id: str, req: ZoneUpdateRequest):
    existing = load_zone(zone_id)
    zone = Zone(
        zone_id=zone_id, video_id=existing.video_id,
        label=req.label if req.label is not None else existing.label,
        shape=req.shape if req.shape is not None else existing.shape,
        points=req.points if req.points is not None else existing.points,
    )
    save_zone(zone)
    return ZoneOut(**zone.to_dict())


# --- zone-only jobs (kept for backward compatibility) ------------------------

@app.post("/api/zone-jobs", response_model=JobStatusOut)
def start_zone_job(req: ZoneJobRequest):
    video_path = find_video_path(req.video_id)
    zone = load_zone(req.zone_id)

    zone_config = ZoneConfig()
    if req.config is not None:
        overrides = {k: v for k, v in req.config.model_dump().items() if v is not None}
        zone_config = dataclasses.replace(zone_config, **overrides)

    with VideoReader(video_path) as reader:
        total_frames = reader.info.frame_count
    job = job_manager.create(kind="zone", total_frames=total_frames)
    output_path = OUTPUTS_DIR / f"zone_{job.job_id}.mp4"
    job.output_path = str(output_path)

    def _run(progress_cb):
        result = ZonePipeline(config=zone_config).run(video_path, zone, output_path, progress_cb=progress_cb)
        job.events = result.events
        return {
            "zone_id": result.zone_id, "tracks_observed": result.tracks_observed,
            "entries": result.entries, "exits": result.exits, "dwell_events": result.dwell_events,
            "codec_used": result.codec_used, "browser_playable": result.browser_playable,
        }

    job_manager.start(job, _run)
    return _job_status_out(job)


# --- unified combined jobs (Phase 6) ------------------------------------------

@app.post("/api/jobs", response_model=JobStatusOut)
def start_combined_job(req: CombinedJobRequest):
    video_path = find_video_path(req.video_id)
    if not (req.person_id or req.anpr or req.zone_ids or req.behavior or req.lowlight or req.fog):
        raise HTTPException(400, "at least one module must be enabled")

    orch_req = OrchestratorRequest()

    if req.person_id is not None:
        orch_req.enable_person_id = True
        orch_req.reference_photo_paths = find_reference_photos(req.person_id.reference_set_id)
        if req.person_id.config:
            orch_req.person_id_config = dataclasses.replace(PersonIDConfig(), **req.person_id.config)

    if req.anpr is not None:
        orch_req.enable_anpr = True
        orch_req.target_plate = req.anpr.target_plate
        if req.anpr.config:
            orch_req.anpr_config = dataclasses.replace(AnprConfig(), **req.anpr.config)

    if req.zone_ids:
        orch_req.zones = [load_zone(zid) for zid in req.zone_ids]

    if req.behavior is not None:
        orch_req.enable_behavior = True
        if req.behavior.config:
            orch_req.behavior_config = dataclasses.replace(BehaviorConfig(), **req.behavior.config)

    if req.lowlight is not None:
        orch_req.enable_lowlight = True
        if req.lowlight.config:
            orch_req.lowlight_config = dataclasses.replace(LowLightConfig(), **req.lowlight.config)

    if req.fog is not None:
        orch_req.enable_dehaze = True

    with VideoReader(video_path) as reader:
        total_frames = reader.info.frame_count
    job = job_manager.create(kind="combined", total_frames=total_frames)
    output_path = OUTPUTS_DIR / f"job_{job.job_id}.mp4"
    evidence_dir = EVIDENCE_DIR / job.job_id
    orch_req.evidence_dir = evidence_dir
    job.output_path = str(output_path)
    job.evidence_dir = str(evidence_dir)

    def _run(progress_cb):
        result = CombinedPipeline(orch_req).run(video_path, output_path, progress_cb=progress_cb)
        job.events = result.events
        return {
            "modules_run": result.modules_run, "codec_used": result.codec_used,
            "browser_playable": result.browser_playable,
            "video_info": {
                "fps": result.video_info.fps, "width": result.video_info.width,
                "height": result.video_info.height, "frame_count": result.video_info.frame_count,
                "duration_sec": result.video_info.duration_sec,
            },
            "person_id": result.person_id, "anpr": result.anpr, "zones": result.zones,
            "behavior": result.behavior, "lowlight": result.lowlight,
        }

    job_manager.start(job, _run)
    return _job_status_out(job)


@app.post("/api/anpr/search", response_model=JobStatusOut)
def start_plate_search(req: PlateSearchRequest):
    """Convenience wrapper: 'is this plate present in the video?' - equivalent
    to POST /api/jobs with only anpr enabled and target_plate set."""
    video_path = find_video_path(req.video_id)
    anpr_module = AnprModuleRequest(target_plate=req.plate, config=req.config)
    return start_combined_job(CombinedJobRequest(video_id=req.video_id, anpr=anpr_module))


# --- jobs (shared by both job kinds) ------------------------------------------

@app.get("/api/jobs", response_model=list[JobStatusOut])
def list_jobs():
    jobs = sorted(job_manager.list(), key=lambda j: j.created_at, reverse=True)
    return [_job_status_out(j) for j in jobs]


@app.get("/api/jobs/{job_id}", response_model=JobStatusOut)
def get_job(job_id: str):
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(404, f"job not found: {job_id}")
    return _job_status_out(job)


@app.post("/api/jobs/{job_id}/cancel", response_model=JobStatusOut)
def cancel_job(job_id: str):
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(404, f"job not found: {job_id}")
    ok = job_manager.cancel(job)
    if not ok:
        raise HTTPException(409, f"job cannot be cancelled (status={job.status})")
    return _job_status_out(job)


@app.get("/api/jobs/{job_id}/events")
def get_job_events(job_id: str):
    """A finished job's events, each carrying the severity its rule earned.

    Live events are stamped with severity as they are emitted; forensic ones
    were not, so anything consuming them had to keep its own opinion about
    which types matter. Two lists of "what counts as an alarm" inevitably
    drift - and did: raising a zone crossing to CRITICAL made the siren sound
    on a live camera while the same crossing stayed silent in Forensic Mode,
    because the page was still working from a hardcoded list that predated it.
    Stamping severity here leaves one source of truth for both modes.
    """
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(404, f"job not found: {job_id}")
    if job.status != "done":
        raise HTTPException(409, f"job is not finished (status={job.status})")
    return [
        {**event, "severity": severity_for(event), "alarm": should_alarm(
            {**event, "severity": severity_for(event)})}
        for event in job.events
    ]


@app.get("/api/jobs/{job_id}/output")
def get_job_output(job_id: str):
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(404, f"job not found: {job_id}")
    if job.status != "done":
        raise HTTPException(409, f"job is not finished (status={job.status})")
    if job.output_path is None or not Path(job.output_path).is_file():
        raise HTTPException(404, "output video not found")
    return FileResponse(job.output_path, media_type="video/mp4")


@app.get("/api/jobs/{job_id}/evidence", response_model=list[EvidenceItemOut])
def list_job_evidence(job_id: str):
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(404, f"job not found: {job_id}")
    if not job.evidence_dir or not Path(job.evidence_dir).is_dir():
        return []
    items = []
    for path in sorted(Path(job.evidence_dir).glob("*.jpg")):
        parts = path.stem.split("_")
        track_id = int(parts[0]) if parts and parts[0].isdigit() else None
        frame_index = int(parts[-1]) if parts and parts[-1].isdigit() else None
        event_type = "_".join(parts[1:-1]) if len(parts) > 2 else None
        items.append(EvidenceItemOut(filename=path.name, track_id=track_id, event_type=event_type, frame_index=frame_index))
    return items


@app.get("/api/jobs/{job_id}/evidence/{filename}")
def get_job_evidence_file(job_id: str, filename: str):
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(404, f"job not found: {job_id}")
    if not job.evidence_dir:
        raise HTTPException(404, "no evidence for this job")
    path = Path(job.evidence_dir) / Path(filename).name  # basename only - no path traversal
    if not path.is_file():
        raise HTTPException(404, "evidence file not found")
    return FileResponse(path, media_type="image/jpeg")


# --- system status -------------------------------------------------------

@app.get("/api/system/status", response_model=SystemStatusOut)
def system_status():
    ocr = PlateOcr()
    jobs = job_manager.list()
    return SystemStatusOut(
        device=detect_device(),
        ocr_available=ocr.available, ocr_status=ocr.status.to_dict(),
        models_present={
            "yunet": YUNET_MODEL_PATH.is_file(), "sface": SFACE_MODEL_PATH.is_file(), "yolo": YOLO_MODEL_PATH.exists(),
        },
        active_jobs=sum(1 for j in jobs if j.status in ("pending", "running")),
        total_jobs=len(jobs),
    )


# --- dashboard ----------------------------------------------------------------
#
# The built frontend is served by this app rather than a separate dev server,
# so a deployment - and a phone on the same network - needs exactly one address
# and one port. Without this, the API answers at the root with "Not Found",
# which looks like a broken server rather than a missing frontend.

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

if FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_dashboard(full_path: str):
        """Return the app shell for any non-API path.

        The dashboard routes client-side, so /live and /audit must both serve
        index.html and let the browser resolve them - returning 404 would break
        every deep link and page refresh.
        """
        if full_path.startswith("api/"):
            raise HTTPException(404, "not found")
        asset = FRONTEND_DIST / full_path
        if full_path and asset.is_file():
            return FileResponse(asset)
        # The shell names the current build's hashed bundle, so it must never
        # be cached: a browser holding an old copy keeps showing the old
        # dashboard after an update (new pages missing from the sidebar).
        # The bundles themselves are named by content hash and cache safely.
        return FileResponse(FRONTEND_DIST / "index.html", headers={"Cache-Control": "no-cache"})
else:
    logger.warning(
        "frontend not built (%s missing) - the API will run but the dashboard "
        "will not be served. Run: cd frontend && npm run build", FRONTEND_DIST,
    )
