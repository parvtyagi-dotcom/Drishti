"""Live camera registry and per-camera capture sessions.

This is the live counterpart to backend.api.jobs.JobManager. The difference
is the lifecycle: a job is created, runs to completion and is *done*, whereas
a camera session is started and then simply runs - there is no terminal
success state, only online/reconnecting/offline and an eventual explicit
stop. Modelling a camera as a job was the thing that made "live" impossible
in the Forensic Mode design, so it is deliberately not reused here.

One session owns one capture thread. Frames are JPEG-encoded once per frame
inside that thread and handed to every viewer, rather than each viewer
encoding independently - with several operators watching the same camera,
per-viewer encoding is the difference between one encode per frame and N.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque

from backend.api.ownership import current_owner, visible
from backend.config import EVENT_DB_PATH
from backend.core.events import ALARM_SEVERITIES, SILENT_TYPES, describe_event, severity_for
from backend.core.video import DEFAULT_MAX_ANALYSIS_WIDTH
from backend.live.store import EventStore
# LiveSession and its configuration moved to their own modules; re-exported
# so every existing `from backend.live.camera_manager import ...` still works.
from backend.live.live_config import (  # noqa: F401
    DEFAULT_CAPTURE_FPS, AnalysisSpec, Camera, live_anpr_config, live_behavior_config,
    live_person_id_config, live_tracker_config,
)
from backend.live.live_session import LiveSession  # noqa: F401

logger = logging.getLogger("backend.live")


class CameraManager:
    """Registry of configured cameras and their running sessions."""

    # How often camera health is sampled. Fast enough that an operator sees a
    # dropout within a couple of seconds, slow enough to be free.
    HEALTH_POLL_SEC = 2.0

    def __init__(self):
        self._sessions: dict[str, LiveSession] = {}
        self._lock = threading.Lock()
        self._global_subscribers: list = []
        self._subscriber_lock = threading.Lock()
        self._monitor: threading.Thread | None = None
        self._monitor_stop = threading.Event()
        # Events raised by something other than a camera (a fence sensor with
        # no camera linked, say). Kept here so dashboards backfill them too.
        self._unattached: deque[dict] = deque(maxlen=LiveSession.MAX_RECENT_EVENTS)
        # Called with (session, clean image, tracks) on every analysed frame of
        # every camera, including cameras added later.
        self._track_observers: list = []
        # One store for the whole fleet: a single chain over all cameras means
        # removing a camera's records breaks the shared sequence too.
        self.store = EventStore(EVENT_DB_PATH)

    # --- health monitoring ---------------------------------------------------

    def start_monitor(self) -> None:
        """Watch every camera's health and turn changes into events."""
        if self._monitor is not None and self._monitor.is_alive():
            return
        self._monitor_stop.clear()
        self.store.start()
        self._monitor = threading.Thread(target=self._monitor_loop, name="camera-health", daemon=True)
        self._monitor.start()

    def stop_monitor(self) -> None:
        self._monitor_stop.set()
        self.store.stop()
        if self._monitor is not None:
            self._monitor.join(timeout=3.0)
            self._monitor = None

    def _monitor_loop(self) -> None:
        while not self._monitor_stop.wait(self.HEALTH_POLL_SEC):
            for session in self.list():
                try:
                    event = session.check_health()
                    if event is not None and event.get("type") == "CAMERA_OFFLINE":
                        self.check_sabotage(session)
                except Exception:
                    # Health monitoring must never be the thing that takes the
                    # fleet down.
                    logger.exception("health check failed for %s", session.camera.camera_id)

    # A camera that goes dark this soon after an alert anywhere on the post
    # is treated as possibly cut, not merely faulty.
    SABOTAGE_WINDOW_SEC = 120.0
    _ACTIVITY_SEVERITIES = frozenset({"MEDIUM", "HIGH", "CRITICAL"})

    def check_sabotage(self, session: "LiveSession", now: float | None = None) -> dict | None:
        """Sentinel: a camera failing right after activity nearby is how an
        intruder would blind a post, so that timing is raised on its own.

        Cameras also fail for dull reasons, which is why this needs recent
        alerts as well as the outage, and why it reports a suspicion with the
        evidence rather than a conclusion.
        """
        now = time.time() if now is None else now
        recent = [
            e for e in self.recent_events(limit=500)
            if now - (e.get("wall_time") or 0) <= self.SABOTAGE_WINDOW_SEC
            and e.get("severity") in self._ACTIVITY_SEVERITIES
            and not str(e.get("type", "")).startswith(("CAMERA_", "AGENT_"))
        ]
        if not recent:
            return None
        last = max(recent, key=lambda e: e.get("wall_time") or 0)
        return session.emit_event({"type": "CAMERA_SABOTAGE_SUSPECTED", "data": {
            "recent_alerts": len(recent),
            "window_sec": int(self.SABOTAGE_WINDOW_SEC),
            "last_alert_type": last.get("type"),
            "last_alert_camera": last.get("camera_name"),
            "seconds_since_last_alert": round(now - (last.get("wall_time") or now)),
        }})

    # --- fleet-wide event subscription ---------------------------------------
    #
    # A control room watches every camera at once, so consumers subscribe here
    # rather than to each session - including cameras added after they
    # subscribed, which per-session subscription would miss.

    def subscribe(self, callback) -> None:
        with self._subscriber_lock:
            self._global_subscribers.append(callback)

    def unsubscribe(self, callback) -> None:
        with self._subscriber_lock:
            if callback in self._global_subscribers:
                self._global_subscribers.remove(callback)

    def _fanout(self, event: dict) -> None:
        with self._subscriber_lock:
            subscribers = list(self._global_subscribers)
        for callback in subscribers:
            try:
                callback(event)
            except Exception:
                logger.exception("global event subscriber failed")

    def emit_unattached(self, event: dict, source_id: str, source_name: str) -> dict:
        """Publish an event that no camera raised, through the same log,
        siren and dashboards as a camera's. Severity and reason come from the
        same rules, so a sensor alarm is never treated differently from a
        camera alarm of the same weight."""
        severity = severity_for(event)
        enriched = {
            "track_id": None, "frame_index": None, "timestamp_sec": 0.0,
            **event,
            "event_id": uuid.uuid4().hex[:12],
            "seq": None,
            "camera_id": source_id,
            "camera_name": source_name,
            "severity": severity,
            "reason": describe_event(event),
            "alarm": severity in ALARM_SEVERITIES and event.get("type") not in SILENT_TYPES,
            "wall_time": time.time(),
        }
        self._unattached.append(enriched)
        self.store.record(enriched)
        self._fanout(enriched)
        return enriched

    def add_track_observer(self, observer) -> None:
        self._track_observers.append(observer)
        for session in self.list():
            session.track_observers.append(observer)

    def find_by_name(self, name: str) -> LiveSession | None:
        """The camera an operator calls `name`, ignoring case and spacing.
        Sensors, like the site map, refer to cameras by name because ids
        change when a camera is registered again."""
        wanted = " ".join(name.split()).lower()
        for session in self.list():
            if " ".join(session.camera.name.split()).lower() == wanted:
                return session
        return None

    def recent_events(self, limit: int = 100, camera_id: str | None = None) -> list[dict]:
        """Recent events across the fleet, newest last."""
        sessions = [self.get(camera_id)] if camera_id else self.list()
        collected: list[dict] = [] if camera_id or not visible(None) else list(self._unattached)
        for session in sessions:
            if session is not None:
                collected.extend(session.recent_events(limit))
        collected.sort(key=lambda e: e.get("wall_time", 0))
        return collected[-limit:]

    def add(self, name: str, source: str, loop: bool = False, location: str | None = None,
            camera_id: str | None = None, analysis: AnalysisSpec | None = None,
            max_width: int = DEFAULT_MAX_ANALYSIS_WIDTH, rotate: int = 0) -> Camera:
        camera = Camera(
            camera_id=camera_id or uuid.uuid4().hex[:8],
            name=name, source=source, loop=loop, location=location, max_width=max_width, rotate=rotate,
            owner=current_owner.get() or None,
        )
        session = LiveSession(camera, analysis=analysis, store=self.store)
        session.subscribe(self._fanout)
        session.track_observers.extend(self._track_observers)
        with self._lock:
            self._sessions[camera.camera_id] = session
        return camera

    def get(self, camera_id: str) -> LiveSession | None:
        with self._lock:
            session = self._sessions.get(camera_id)
        return session if session is not None and visible(session.camera.owner) else None

    def list(self) -> list[LiveSession]:
        with self._lock:
            return [s for s in self._sessions.values() if visible(s.camera.owner)]

    def remove(self, camera_id: str) -> bool:
        with self._lock:
            session = self._sessions.pop(camera_id, None)
        if session is None:
            return False
        session.stop()
        return True

    def stop_all(self) -> None:
        for session in self.list():
            session.stop()


camera_manager = CameraManager()


# Stand-in cameras for development and demos: local clips looped so they
# behave like continuously running border-post feeds. A real deployment
# registers rtsp:// cameras through the API instead; nothing downstream can
# tell the difference, which is exactly why the stand-ins are useful.
# One camera slot, so Live Monitor opens with somewhere for footage to appear
# rather than an empty page. It is registered whether or not the footage is
# actually on disk: a camera with nothing behind it reports no signal, which
# is a truthful state an operator recognises, while hiding it would make an
# absent feed indistinguishable from a system that never had one.
DEMO_CAMERA = ("cam-01", "CAMERA 01", "smoke_person_id.mp4", "Sector 7 - Perimeter")


def seed_demo_cameras(uploads_dir, manager: CameraManager | None = None,
                      autostart: bool = True) -> list[Camera]:
    """Register the camera slot Live Monitor opens on.

    It starts with no analysis armed. Arming it here would mean detections,
    and therefore alarms, running from the moment the server boots - which
    sounds the siren over whatever the operator is actually doing, including
    work in Forensic Analysis that has nothing to do with this camera. What
    to watch for is the operator's choice, made in Configure.
    """
    manager = manager or camera_manager
    camera_id, name, filename, location = DEMO_CAMERA
    if manager.get(camera_id) is not None:
        return []
    camera = manager.add(
        name=name, source=str(uploads_dir / filename), loop=True,
        location=location, camera_id=camera_id,
    )
    if autostart:
        manager.get(camera.camera_id).start()
    return [camera]
