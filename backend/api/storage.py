"""File-backed storage for uploaded videos and zone definitions.

No database for this phase - videos live under uploads/{video_id}{ext},
zones are one JSON file per zone under zones/{zone_id}.json. Simple, and
matches the top-level uploads/ / zones/ directories the project already
uses for on-disk state.
"""
from __future__ import annotations

import io
import json
import re
import shutil
import uuid
from pathlib import Path
from typing import BinaryIO

from backend.config import REFERENCES_DIR, UPLOADS_DIR, ZONES_DIR
from backend.modules.zone.geometry import Zone


class NotFoundError(Exception):
    pass


class UploadTooLargeError(ValueError):
    pass


# Ids reach these functions straight from request paths and JSON bodies. They
# were joined onto directories and fed to glob() as-is, so "../users" escaped
# the folder and "*" matched any file. Every id this system issues is a uuid
# hex; the wider pattern keeps hand-made ids like "smoke_target" working.
_SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")

VIDEO_SUFFIXES = frozenset({".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm", ".mpg", ".mpeg", ".ts"})
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
MAX_VIDEO_BYTES = 4 * 1024 ** 3     # a long CCTV export, not an unbounded stream
MAX_PHOTO_BYTES = 20 * 1024 ** 2
_COPY_CHUNK = 1024 * 1024


def _checked_id(value: str, kind: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise NotFoundError(f"{kind} not found: {value!r}")
    return value


def safe_suffix(filename: str, allowed: frozenset[str], default: str) -> str:
    """The upload's extension if it is a known media type, else `default`.
    A client-chosen ".html" or ".exe" never lands on disk."""
    suffix = Path(filename or "").suffix.lower()
    return suffix if suffix in allowed else default


def _copy_limited(source: BinaryIO, path: Path, max_bytes: int) -> int:
    """Stream an upload to disk in chunks, refusing it past max_bytes.

    Reading the whole upload into memory first meant one large file could
    exhaust the RAM the live cameras need.
    """
    written = 0
    try:
        with path.open("wb") as out:
            while chunk := source.read(_COPY_CHUNK):
                written += len(chunk)
                if written > max_bytes:
                    raise UploadTooLargeError(f"file is larger than {max_bytes // 1024 ** 2} MB")
                out.write(chunk)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return written


def new_id() -> str:
    return uuid.uuid4().hex


def save_uploaded_video(filename: str, data: bytes | BinaryIO, max_bytes: int = MAX_VIDEO_BYTES) -> tuple[str, Path]:
    video_id = new_id()
    ext = safe_suffix(filename, VIDEO_SUFFIXES, ".mp4")
    path = UPLOADS_DIR / f"{video_id}{ext}"
    _copy_limited(io.BytesIO(data) if isinstance(data, bytes) else data, path, max_bytes)
    return video_id, path


def find_video_path(video_id: str) -> Path:
    _checked_id(video_id, "video")
    matches = list(UPLOADS_DIR.glob(f"{video_id}.*"))
    if not matches:
        raise NotFoundError(f"video not found: {video_id}")
    return matches[0]


def save_zone(zone: Zone) -> None:
    path = ZONES_DIR / f"{_checked_id(zone.zone_id, 'zone')}.json"
    path.write_text(json.dumps(zone.to_dict(), indent=2))


def load_zone(zone_id: str) -> Zone:
    path = ZONES_DIR / f"{_checked_id(zone_id, 'zone')}.json"
    if not path.is_file():
        raise NotFoundError(f"zone not found: {zone_id}")
    return Zone.from_dict(json.loads(path.read_text()))


def save_reference_photos(files: list[tuple[str, bytes | BinaryIO]],
                          max_bytes: int = MAX_PHOTO_BYTES) -> tuple[str, list[Path]]:
    """Saves 1-6 uploaded reference photos under references/{reference_set_id}/
    and returns (reference_set_id, [paths]). Validation of the 1-6 count and
    of "does at least one contain a usable face" is the pipeline's job
    (ReferenceSet already handles per-photo rejection without crashing) -
    this layer just persists bytes to disk."""
    ref_id = new_id()
    ref_dir = REFERENCES_DIR / ref_id
    ref_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    try:
        for i, (filename, data) in enumerate(files):
            path = ref_dir / f"{i}{safe_suffix(filename, IMAGE_SUFFIXES, '.jpg')}"
            _copy_limited(io.BytesIO(data) if isinstance(data, bytes) else data, path, max_bytes)
            paths.append(path)
    except BaseException:
        shutil.rmtree(ref_dir, ignore_errors=True)
        raise
    return ref_id, paths


def find_reference_photos(reference_set_id: str) -> list[Path]:
    ref_dir = REFERENCES_DIR / _checked_id(reference_set_id, "reference set")
    if not ref_dir.is_dir():
        raise NotFoundError(f"reference set not found: {reference_set_id}")
    return sorted(ref_dir.glob("*"))
