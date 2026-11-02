"""Appearance search over what the live cameras have seen.

    GET /api/search/appearance/status        whether search is available, and how much is indexed
    GET /api/search/appearance?q=...         ranked matches (kind, camera, hours, limit optional)
    GET /api/search/appearance/{id}/crop.jpg the picture a match was ranked on
"""
from __future__ import annotations

import threading
import time

from fastapi import APIRouter, HTTPException, Query, Response

from backend.live.appearance import AppearanceIndex, model_available
from backend.live.camera_manager import camera_manager

router = APIRouter(prefix="/api/search/appearance", tags=["search"])

_index: AppearanceIndex | None = None

UNAVAILABLE = ("Appearance search is not installed on this machine. Run "
               "python scripts/download_search_model.py once (about 600 MB), then restart.")


def start() -> AppearanceIndex | None:
    """Index every camera's tracked people and vehicles, if the model is here."""
    global _index
    if _index is not None or not model_available():
        return _index
    _index = AppearanceIndex()
    camera_manager.add_track_observer(_index.observe)
    # Load the model now, not on the first query an operator is waiting on.
    threading.Thread(target=lambda: _index.model.text(["warm up"]), name="appearance-warmup", daemon=True).start()
    return _index


def stop() -> None:
    if _index is not None:
        _index.stop()


def _require() -> AppearanceIndex:
    if _index is None:
        raise HTTPException(503, UNAVAILABLE)
    return _index


@router.get("/status")
def status() -> dict:
    if _index is None:
        return {"available": False, "message": UNAVAILABLE}
    return {"available": True, **_index.stats()}


@router.get("")
def search(q: str = Query(min_length=2, max_length=120), kind: str | None = None,
           camera: str | None = None, hours: float = Query(default=24, gt=0, le=24 * 7),
           limit: int = Query(default=24, ge=1, le=100)) -> dict:
    index = _require()
    if kind not in (None, "", "person", "vehicle"):
        raise HTTPException(400, "kind must be person or vehicle")
    results = index.search(q.strip(), kind=kind or None, camera=camera or None,
                           since=time.time() - hours * 3600, limit=limit)
    return {"query": q, "results": results}


@router.get("/{sighting_id}/crop.jpg", include_in_schema=False)
def crop(sighting_id: int) -> Response:
    data = _require().crop(sighting_id)
    if data is None:
        raise HTTPException(404, "no such picture")
    return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})
