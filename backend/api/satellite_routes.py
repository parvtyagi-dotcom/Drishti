"""Satellite check: weekly before/after comparison of saved Sentinel-2 pairs.

    GET  /api/satellite/pairs                     saved pairs (any signed-in user)
    POST /api/satellite/pairs/{pair_id}/check     find what changed (admin, operator)
    GET  /api/satellite/pairs/{pair_id}/check     the last result, if any
    GET  /api/satellite/pairs/{pair_id}/{which}.png   before/after image or location map
    GET  /api/satellite/places?q=...              look a place up by name (needs internet)
    POST /api/satellite/fetch                     fetch a new pair for a place (needs internet)
    GET  /api/satellite/fetch/{job_id}            progress of that fetch

Saved pairs are checked offline. Only the place search and the fetch go to
the internet, and only when an operator asks for them.
"""
from __future__ import annotations

import re
import threading
import uuid
from datetime import date, timedelta

import requests
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.api.auth import ADMIN, AUDITOR, OPERATOR, User, require_role
from backend.live import satellite, sentinel

router = APIRouter(prefix="/api/satellite", tags=["satellite"])

ANY = require_role(ADMIN, OPERATOR, AUDITOR)

# How far either side of each chosen date to look for a cloud-free picture.
# Sentinel-2 passes every 5 days at best, so a narrower window often finds none.
SEARCH_WINDOW_DAYS = 15
MIN_GAP_DAYS = 7


class FetchRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    lat: float = Field(ge=-85, le=85)
    lon: float = Field(ge=-180, le=180)
    before: date
    after: date
    size_km: float = Field(default=5, ge=1, le=10)


_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def date_windows(before: date, after: date, today: date) -> tuple[str, str]:
    """Search windows around the two chosen dates that never overlap, so the
    same picture cannot be used for both."""
    middle = before + (after - before) / 2
    b0, b1 = before - timedelta(days=SEARCH_WINDOW_DAYS), min(before + timedelta(days=SEARCH_WINDOW_DAYS), middle)
    a0 = max(after - timedelta(days=SEARCH_WINDOW_DAYS), middle + timedelta(days=1))
    a1 = min(after + timedelta(days=SEARCH_WINDOW_DAYS), today)
    return f"{b0}/{b1}", f"{a0}/{a1}"


def _pair_id(name: str, after: date) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.split(",")[0].lower()).strip("-")[:40] or "place"
    return f"{slug}-{after:%Y%m%d}"


def _run_fetch(job_id: str, req: FetchRequest, pair_id: str, before: str, after: str) -> None:
    job = _jobs[job_id]
    bbox = sentinel.bbox_around(req.lat, req.lon, req.size_km)
    try:
        sentinel.fetch_pair(bbox, before, after, req.name, pair_id,
                            note=f"Searched by place: {req.name}.", log=job["log"].append)
        job["status"] = "done"
    except sentinel.FetchError as exc:
        job.update(status="failed", error=str(exc))
    except requests.RequestException:
        job.update(status="failed", error="Could not reach the satellite archive. This needs an internet connection.")
    except Exception as exc:  # noqa: BLE001 - surface anything else to the operator, not a silent hang
        job.update(status="failed", error=f"Fetch failed: {exc}")


@router.get("/places")
def places(q: str, _: User = Depends(ANY)) -> dict:
    q = q.strip()
    if not q:
        return {"places": []}
    try:
        return {"places": sentinel.geocode(q)}
    except requests.RequestException:
        raise HTTPException(502, "Place search needs an internet connection.")


@router.post("/fetch", status_code=202)
def start_fetch(req: FetchRequest, _: User = Depends(require_role(ADMIN, OPERATOR))) -> dict:
    today = date.today()
    if req.after > today:
        raise HTTPException(400, "The 'after' date cannot be in the future.")
    if (req.after - req.before).days < MIN_GAP_DAYS:
        raise HTTPException(400, f"Pick dates at least {MIN_GAP_DAYS} days apart.")
    with _jobs_lock:
        if any(j["status"] == "running" for j in _jobs.values()):
            raise HTTPException(409, "Another fetch is still running. Wait for it to finish.")
        job_id = uuid.uuid4().hex[:12]
        pair_id = _pair_id(req.name, req.after)
        _jobs[job_id] = {"job_id": job_id, "status": "running", "pair_id": pair_id, "log": [], "error": None}
    before, after = date_windows(req.before, req.after, today)
    threading.Thread(target=_run_fetch, args=(job_id, req, pair_id, before, after), daemon=True).start()
    return _jobs[job_id]


@router.get("/fetch/{job_id}")
def fetch_status(job_id: str, _: User = Depends(ANY)) -> dict:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "no such fetch")
    return job


@router.get("/pairs")
def pairs(_: User = Depends(ANY)) -> dict:
    return {"pairs": satellite.list_pairs()}


@router.post("/pairs/{pair_id}/check")
async def check(pair_id: str, refresh: bool = False, _: User = Depends(require_role(ADMIN, OPERATOR))) -> dict:
    result = await run_in_threadpool(satellite.run_check, pair_id, refresh)
    if result is None:
        raise HTTPException(404, "no such saved pair")
    return result


@router.get("/pairs/{pair_id}/check")
def last_check(pair_id: str, _: User = Depends(ANY)) -> dict:
    result = satellite._cache.get(pair_id)
    if result is None:
        raise HTTPException(404, "this pair has not been checked yet")
    return result


@router.get("/pairs/{pair_id}/{which}.png")
def image(pair_id: str, which: str, _: User = Depends(ANY)) -> FileResponse:
    path = satellite.image_path(pair_id, which)
    if path is None:
        raise HTTPException(404, "no such image")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})
