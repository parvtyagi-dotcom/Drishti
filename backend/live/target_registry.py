"""Shared target appearance across cameras.

A target confirmed on one camera should be recognisable on the next one, even
where no face is ever visible there. That is the difference between several
cameras running the same search independently and a system that actually
tracks a person across a site.

How it works: face recognition confirms *who* someone is on the camera that
can see their face, and every confident observation contributes an appearance
embedding to a gallery shared by every camera searching for that same target.
A second camera can then re-acquire on appearance alone - the thing face
recognition cannot do once someone has turned away or walked out of range.

Two deliberate limits, because appearance matching is weaker evidence than a
face:

  * Only observations the owning camera was already confident about are
    shared. The gallery gating that prevents identity drift on one camera is
    exactly what keeps a bad observation from being broadcast to all of them.
  * Shared appearance goes stale. Someone matched by clothing an hour later is
    a much weaker claim than a minute later, so entries expire.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("backend.live.target_registry")

# How long a shared appearance stays usable. Appearance is clothing and build,
# not identity: over hours people change coats, lighting changes, and the
# evidence decays from weak to misleading.
DEFAULT_TTL_SEC = 30 * 60

# Cap per target so a long run cannot grow without bound, and so one camera
# with a clear view cannot dominate the shared view of the target.
MAX_SHARED_EMBEDDINGS = 40


@dataclass
class _Observation:
    embedding: np.ndarray
    camera_id: str
    at: float = field(default_factory=time.time)


@dataclass
class TargetSighting:
    """A camera reporting that it has the target in view."""

    camera_id: str
    camera_name: str
    at: float
    confirmed_by: str  # "face" or "appearance"


class TargetRegistry:
    """Appearance galleries shared between cameras, keyed by enrolled target."""

    def __init__(self, ttl_sec: float = DEFAULT_TTL_SEC):
        self.ttl_sec = ttl_sec
        self._observations: dict[str, list[_Observation]] = {}
        self._sightings: dict[str, TargetSighting] = {}
        self._lock = threading.Lock()

    # --- contributing --------------------------------------------------------

    def contribute(self, target_key: str, embedding: np.ndarray, camera_id: str) -> None:
        """Share one confident appearance observation of a target."""
        if embedding is None or target_key is None:
            return
        with self._lock:
            entries = self._observations.setdefault(target_key, [])
            entries.append(_Observation(embedding=embedding, camera_id=camera_id))
            self._prune(target_key)

    def record_sighting(self, target_key: str, camera_id: str, camera_name: str,
                        confirmed_by: str) -> TargetSighting | None:
        """Note that a camera currently has the target, and report the previous
        camera if this is a handover from somewhere else."""
        with self._lock:
            previous = self._sightings.get(target_key)
            self._sightings[target_key] = TargetSighting(
                camera_id=camera_id, camera_name=camera_name,
                at=time.time(), confirmed_by=confirmed_by,
            )
            if previous is None or previous.camera_id == camera_id:
                return None
            return previous

    # --- consuming -----------------------------------------------------------

    def embeddings_for(self, target_key: str, exclude_camera: str | None = None) -> list[np.ndarray]:
        """Appearances of this target contributed by *other* cameras.

        Excluding the calling camera matters: matching against a camera's own
        observations tells it nothing new, and would let one camera's mistake
        reinforce itself.
        """
        with self._lock:
            self._prune(target_key)
            return [
                o.embedding for o in self._observations.get(target_key, [])
                if exclude_camera is None or o.camera_id != exclude_camera
            ]

    def cameras_holding(self, target_key: str) -> set[str]:
        with self._lock:
            self._prune(target_key)
            return {o.camera_id for o in self._observations.get(target_key, [])}

    def last_sighting(self, target_key: str) -> TargetSighting | None:
        with self._lock:
            return self._sightings.get(target_key)

    def forget(self, target_key: str) -> None:
        with self._lock:
            self._observations.pop(target_key, None)
            self._sightings.pop(target_key, None)

    def stats(self) -> dict:
        with self._lock:
            return {
                "targets": len(self._observations),
                "observations": sum(len(v) for v in self._observations.values()),
                "sightings": {
                    key: {
                        "camera_id": s.camera_id, "camera_name": s.camera_name,
                        "at": s.at, "confirmed_by": s.confirmed_by,
                    }
                    for key, s in self._sightings.items()
                },
            }

    # --- internals -----------------------------------------------------------

    def _prune(self, target_key: str) -> None:
        """Caller must hold the lock."""
        entries = self._observations.get(target_key)
        if not entries:
            return
        cutoff = time.time() - self.ttl_sec
        fresh = [o for o in entries if o.at >= cutoff]
        if len(fresh) > MAX_SHARED_EMBEDDINGS:
            # Keep the most recent: current appearance beats older appearance
            # when deciding whether the person in front of a camera now is the
            # same one seen elsewhere.
            fresh = fresh[-MAX_SHARED_EMBEDDINGS:]
        self._observations[target_key] = fresh


target_registry = TargetRegistry()
