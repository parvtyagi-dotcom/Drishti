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
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from backend.config import (
    AnprConfig, BehaviorConfig, EVENT_DB_PATH, EVIDENCE_DIR, PersonIDConfig, TrackerConfig,
)
from backend.core.events import ALARM_SEVERITIES, SILENT_TYPES, describe_event, severity_for
from backend.core.video import (
    DEFAULT_MAX_ANALYSIS_WIDTH, LiveStreamReader, OFFLINE, ONLINE, PUSH_SCHEME,
    PushStreamReader, RECONNECTING, STOPPED, VideoOpenError,
)
from backend.core.motion import MotionGate
from backend.live.crowd import PERSON_CLASS, GroupWatch
from backend.live.evidence import EvidenceRecorder
from backend.live.frame_feed import LiveFrameFeed
from backend.live.false_alarms import memory as false_alarm_memory
from backend.live.tamper import TamperMonitor, TamperReading, TappedReader
from backend.live.target_registry import target_registry
from backend.live.store import EventStore
from backend.core.calibration import load_calibration
from backend.orchestrator import CombinedPipeline, OrchestratorRequest
from backend.api.ownership import current_owner, visible

logger = logging.getLogger("backend.live")


def live_behavior_config() -> BehaviorConfig:
    """Behaviour thresholds retuned for live analysis rates.

    The defaults in config.BehaviorConfig assume Forensic Mode's conditions:
    every frame of a 30-60fps file, so a track's path is sampled almost
    continuously and an 8px movement floor with a 0.3s checkpoint reliably
    separates real movement from detector jitter.

    Live analysis on CPU runs at roughly 3-6fps, which breaks those
    assumptions in a specific way: consecutive observations are ~0.25s and
    tens of pixels apart, so *every* frame clears an 8px floor and becomes a
    new movement segment, and any tracker wobble between two distant
    observations reads as a direction reversal. Measured on a sparse test
    clip, the file-tuned defaults produced ~120 behaviour events in 30
    seconds - which is noise, not intelligence, and an alert stream nobody can
    act on is worse than none.

    These values restore the intent by spanning a longer baseline per segment
    and demanding more evidence before calling something suspicious.
    """
    return BehaviorConfig(
        # A segment now spans ~1s of real motion rather than a single frame,
        # so direction and speed are measured against a stable baseline.
        sample_interval_sec=1.0,
        min_displacement_px=25.0,
        # Only near-reversals count, not the ordinary course corrections a
        # person makes walking around others. Set this high because the
        # remaining false positives at low frame rates come from tracker ID
        # churn - one identity handed to two people looks like an impossible
        # turn - and only a near-180 degree threshold reliably rejects that.
        direction_change_degrees=120.0,
        reversal_angle_degrees=150.0,
        # Pacing means repeatedly retracing a path, which needs a longer
        # window and more repetitions to distinguish from walking past twice.
        reversal_count=4,
        reversal_window_sec=30.0,
        # One alert per track per 20s: enough to stay informed, not enough to
        # bury the operator.
        behavior_cooldown_seconds=20.0,
        # The same ~1s baseline, in metres, for a calibrated camera.
        min_displacement_m=0.8,
        vehicle_min_displacement_m=2.0,
    )


def live_tracker_config() -> TrackerConfig:
    """Live keeps the person tracker at its normal stride when a target is
    locked - see TrackerConfig.adaptive_person_stride. The zone urgency
    stays on: it only runs the shared detector more often near a zone."""
    return TrackerConfig(adaptive_person_stride=False)


def live_anpr_config() -> AnprConfig:
    """ANPR sampling retuned for live analysis rates.

    ocr_sample_stride exists to keep an expensive step off every frame, and
    5 is right for Forensic Mode: a 30fps file yields ~6 OCR looks a second,
    so a plate is read a dozen times or more while a vehicle crosses the view
    and vote_min_observations agreeing reads is an easy bar to clear.

    Live analysis already drops frames to stay in step with the wall clock,
    running at roughly 3-6fps on CPU. Striding again on top of that decimates
    twice: the same setting yields well under one look per second, so a
    vehicle passing in four seconds is read two or three times - never enough
    agreeing reads to confirm a plate. Measured on real road footage, plates
    were located 46 times and confirmed zero times.

    Sampling every analysed frame restores the intended cadence in time
    rather than in frames. It costs nothing extra per frame, because the
    frames it now samples are ones the pipeline already decoded and detected
    vehicles in.
    """
    return AnprConfig(ocr_sample_stride=1)


def live_person_id_config() -> PersonIDConfig:
    """Face and ReID sampling retuned for live analysis rates.

    Same reasoning as live_anpr_config: face_sample_stride=5 over frames that
    have already been thinned to a few per second leaves only a fraction of a
    face look per second, so a target has to stand still for many seconds
    before confirmation_observations can be met. A person who merely walks
    through the view is read once, or not at all.

    Every second frame, not every frame: at the 12fps live capture rate that
    is still six face looks a second, and checking every frame was what
    capped a camera with people close in view at ~4fps - so its video lagged.
    Measured on the street demo clip: every frame 3.6fps, every 2nd 7.4fps,
    target confirmed both ways.
    """
    return PersonIDConfig(face_sample_stride=2, reid_sample_stride=2)


# Capture rate for live analysis. Deliberately far below what the demo clips
# (30-60fps) or a modern camera can deliver: analytics gains nothing from
# 60fps, and a lower rate means consecutive analysed frames sit closer
# together in scene time whenever inference cannot keep up - which on CPU it
# cannot. Matches how analytics cameras are configured in practice.
DEFAULT_CAPTURE_FPS = 12.0


def _source_kind(source: str) -> str:
    """Coarse description of where a camera's frames come from. Returned to
    clients instead of the source itself, which can embed RTSP credentials."""
    lowered = source.lower()
    if lowered.startswith(PUSH_SCHEME):
        return "phone"
    if lowered.startswith(("rtsp://", "rtsps://")):
        return "rtsp"
    if lowered.startswith(("http://", "https://")):
        return "http"
    if lowered.strip().isdigit():
        return "webcam"
    return "file"


@dataclass
class Camera:
    camera_id: str
    name: str
    source: str
    loop: bool = False
    location: str | None = None
    target_fps: float = DEFAULT_CAPTURE_FPS
    # Frames are downscaled to this width before analysis. Raise it for a
    # camera doing ANPR, where legibility depends on plate pixels and the
    # default trades resolution for speed.
    max_width: int = DEFAULT_MAX_ANALYSIS_WIDTH
    # Clockwise degrees applied to every frame (0/90/180/270), for a phone
    # webcam that sends its picture sideways.
    rotate: int = 0
    # Account that added it; None for demo cameras (see backend.api.ownership).
    owner: str | None = None

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            # The source can embed RTSP credentials (rtsp://user:pass@host),
            # so it is never returned to clients - only whether it is a
            # network camera or a local stand-in file.
            "source_kind": _source_kind(self.source),
            "loop": self.loop,
            "target_fps": self.target_fps,
            "max_width": self.max_width,
            "rotate": self.rotate,
            "location": self.location,
        }


@dataclass
class AnalysisSpec:
    """Which analysis modules this camera runs.

    Mirrors the fields of a Forensic Mode job request on purpose: a camera is
    just the same analysis pointed at a stream instead of a file, so the two
    stay configured the same way.
    """

    person_id: bool = False
    reference_photo_paths: list[Path] = field(default_factory=list)
    # Kept so the UI can show which enrolment is active and let the operator
    # change other settings without re-uploading the same photos.
    reference_set_id: str | None = None
    anpr: bool = False
    target_plate: str | None = None
    zones: list = field(default_factory=list)
    behavior: bool = False
    behavior_config: BehaviorConfig | None = None  # defaults to live_behavior_config()
    lowlight: bool = False
    fog: bool = False

    @property
    def any_enabled(self) -> bool:
        return bool(self.person_id or self.anpr or self.zones or self.behavior or self.lowlight or self.fog)

    def module_names(self) -> list[str]:
        names = []
        for enabled, name in (
            (self.person_id, "person_id"), (self.anpr, "anpr"), (bool(self.zones), "zone"),
            (self.behavior, "behavior"), (self.lowlight, "lowlight"), (self.fog, "fog"),
        ):
            if enabled:
                names.append(name)
        return names


class LiveSession:
    """One continuously running camera capture, optionally with analysis."""

    # A viewer that has waited this long without a new frame gets the current
    # one re-sent, so an MJPEG connection to a dead camera stays open (showing
    # a frozen last frame) instead of silently hanging forever.
    VIEWER_WAIT_TIMEOUT = 2.0

    # Live runs indefinitely, so the in-memory event list cannot grow without
    # limit. The full record belongs in the event store; this is only the
    # recent tail a newly-opened dashboard backfills from.
    MAX_RECENT_EVENTS = 500

    # Minimum gap between two physical alarms of the same kind on one camera.
    # Keyed on event type alone, not on track id: when tracking hands the same
    # person a new id (routine at live frame rates), a per-track gate would let
    # the siren re-fire for what the operator sees as one continuing incident.
    # The events are all still logged - this only governs the siren.
    ALARM_COOLDOWN_SEC = 30.0
    # Pause between attempts to pick a pushed camera back up. Short, because
    # the attempt itself just blocks waiting for the next frame.
    SENDER_RETRY_SEC = 2.0

    def __init__(self, camera: Camera, analysis: AnalysisSpec | None = None, jpeg_quality: int = 80,
                 store: EventStore | None = None):
        self.camera = camera
        self.store = store
        self.analysis = analysis or AnalysisSpec()
        self.jpeg_quality = jpeg_quality
        self.reader = self._new_reader()

        self._thread: threading.Thread | None = None
        self._latest_jpeg: bytes | None = None
        self._frame_seq = 0
        self._condition = threading.Condition()
        self._started_at: float | None = None
        self._stopping = False
        self._recent_events: deque[dict] = deque(maxlen=self.MAX_RECENT_EVENTS)
        self._event_subscribers: list = []
        self._event_lock = threading.Lock()
        self._analysis_error: str | None = None
        self._alarm_gate: dict[str, float] = {}
        # Live video at camera speed (see _on_camera_frame): the clean frame
        # the analysis is working on, and what it drew last time as a
        # (mask, pixels) overlay to lay over fresh camera frames.
        self._analysis_clean = None
        self._overlay = None
        self._show_analysed = False
        self._last_display_at = 0.0
        self._last_evidence_at = 0.0
        self._analysed_times: deque[float] = deque(maxlen=20)
        self._last_health: str | None = None
        self.evidence = EvidenceRecorder(camera.camera_id, EVIDENCE_DIR / "live")
        self.tamper = TamperMonitor()
        self.groups = GroupWatch()
        self.motion_gate: MotionGate | None = None
        # Extra per-frame observers of what is in view (search crops, say).
        self.track_observers: list = []
        self._fault_until: float = 0.0

    def _new_reader(self):
        """Build the right kind of source for this camera.

        A "push://" camera has no URL to open - a phone browser sends frames to
        it - so it gets a PushStreamReader. Everything else (rtsp://, http://,
        a local file) is pulled by LiveStreamReader. Both present the same
        interface, so nothing downstream branches on which one it got.
        """
        if self.camera.source.startswith(PUSH_SCHEME):
            return PushStreamReader(
                self.camera.camera_id, target_fps=self.camera.target_fps,
                max_width=self.camera.max_width,
            )
        return LiveStreamReader(
            self.camera.source, loop=self.camera.loop, target_fps=self.camera.target_fps,
            max_width=self.camera.max_width, rotate=self.camera.rotate,
        )

    @property
    def _zone_labels(self) -> dict[str, str]:
        """Zone id -> the label the operator gave it, for readable alerts."""
        return {
            z.zone_id: z.label
            for z in self.analysis.zones
            if getattr(z, "label", None)
        }

    # --- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        # A capture that outlived stop()'s wait still holds the device. Opening
        # a USB webcam a second time while it does crashes the process inside
        # the driver, so let the old one finish first.
        draining = getattr(self, "_draining", None)
        if draining is not None:
            draining.join(timeout=20.0)
            self._draining = None
        if self.reader.status.state == STOPPED:
            # A stopped reader's iterator has already returned; starting again
            # needs a fresh one rather than a spent generator.
            self.reader = self._new_reader()
        self._analysis_error = None
        self._stopping = False
        with self._condition:
            self._latest_jpeg = None
            self._condition.notify_all()
        self._started_at = time.time()
        self._thread = threading.Thread(
            target=self._run, name=f"live-{self.camera.camera_id}", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stopping = True
        self.reader.stop()
        self.evidence.stop()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                self._draining = self._thread
            self._thread = None
        # Wake any viewer still blocked waiting for a frame that will never come.
        with self._condition:
            self._condition.notify_all()

    @property
    def analysis_error(self) -> str | None:
        return self._analysis_error

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # --- capture thread ------------------------------------------------------

    def _run(self) -> None:
        # Outer loop so an outage does not end the session. A camera that came
        # back only after someone pressed a button would make "automatic
        # recovery" a claim rather than a behaviour.
        while not self._stopping:
            try:
                if self.analysis.any_enabled:
                    self._run_with_analysis()
                else:
                    for frame in self.reader:
                        self._check_tamper(frame.image)
                        self._publish(frame.image)
                        if self._stopping:
                            break
            except Exception as exc:  # pragma: no cover - defensive
                # A capture thread dying silently would leave the UI showing a
                # camera that looks fine but never updates, so the failure is
                # recorded in the status the health board already reads.
                waiting = self._awaiting_sender(exc)
                if waiting:
                    # A phone that walked out of range is ordinary, not a
                    # crash, and logging a stack trace for it would train
                    # people to ignore the log.
                    logger.info("camera %s is waiting for its sending device",
                                self.camera.camera_id)
                else:
                    logger.exception("live session %s failed", self.camera.camera_id)
                self.reader.status.state = OFFLINE
                self.reader.status.error = (
                    "waiting for the sending device" if waiting
                    else f"capture thread crashed: {exc}"
                )
                self._analysis_error = None if waiting else str(exc)

            if self._stopping:
                break

            if self.faulted:
                # Ride out the injected outage, then reconnect from scratch -
                # the same thing that happens when a cable is plugged back in.
                while self.faulted and not self._stopping:
                    time.sleep(0.25)
                if self._stopping:
                    break
                logger.info("camera %s recovering from injected fault", self.camera.camera_id)
                self.reader = self._new_reader()
                self._analysis_error = None
                continue

            if self.camera.source.startswith(PUSH_SCHEME):
                # A pushed camera has no URL to retry, so the old code let the
                # session thread die and relied on the next incoming websocket
                # to restart it. That left the camera dependent on the phone
                # noticing and reconnecting. Staying alive and waiting means
                # the feed resumes by itself the moment frames come back.
                time.sleep(self.SENDER_RETRY_SEC)
                if self._stopping:
                    break
                self.reader = self._new_reader()
                continue

            break

        with self._condition:
            self._condition.notify_all()

    def _run_with_analysis(self) -> None:
        """Drive the shared analysis pipeline from this camera's stream.

        This is the same CombinedPipeline Forensic Mode runs on an uploaded
        file - not a live-specific reimplementation - so every detection rule,
        threshold and confirmation requirement already tested there applies
        identically here.
        """
        spec = self.analysis
        # Cameras searching for the *same* enrolled target share what that
        # person looks like, so the second camera can pick them up without ever
        # seeing their face. Keyed on the enrolment, not the camera, because
        # that is what makes two cameras a system rather than two searches.
        target_key = spec.reference_set_id if spec.person_id else None
        share = match = None
        if target_key:
            camera_id = self.camera.camera_id

            def share(embedding, _key=target_key, _cam=camera_id):
                target_registry.contribute(_key, embedding, _cam)

            def match(embedding, _key=target_key, _cam=camera_id):
                others = target_registry.embeddings_for(_key, exclude_camera=_cam)
                if not others:
                    return -1.0
                return max(float(np.dot(embedding, other)) for other in others)

        self.motion_gate = MotionGate()
        request = OrchestratorRequest(
            enable_person_id=spec.person_id,
            reference_photo_paths=list(spec.reference_photo_paths),
            share_appearance=share,
            match_shared_appearance=match,
            person_id_config=live_person_id_config(),
            tracker_config=live_tracker_config(),
            enable_anpr=spec.anpr,
            target_plate=spec.target_plate,
            anpr_config=live_anpr_config(),
            zones=list(spec.zones),
            enable_behavior=spec.behavior,
            behavior_config=spec.behavior_config or live_behavior_config(),
            enable_lowlight=spec.lowlight,
            enable_dehaze=spec.fog,
            observe_tracks=self._observe_tracks,
            motion_gate=self.motion_gate,
            ground_calibration=load_calibration(self.camera.camera_id) if spec.behavior else None,
        )
        pipeline = CombinedPipeline(request)
        # Frame dimensions must be known before the pipeline can scale zone
        # polygons, so the stream is opened up front rather than lazily.
        self.reader.connect()
        self._overlay, self._show_analysed = None, False
        pipeline.run(
            # The feed reads the camera at full speed for the display; the
            # pipeline takes the newest frame whenever it is free. Tamper
            # checks still see each analysed frame before anything is drawn.
            reader=TappedReader(LiveFrameFeed(self.reader, self._on_camera_frame), self._before_analysis),
            write_output=False,
            frame_callback=self._on_analysed_frame,
        )

    def _check_tamper(self, image) -> None:
        """Sentinel: report a covered, blurred or turned camera, and when its
        view comes back. Never allowed to interrupt the video itself."""
        try:
            result = self.tamper.observe(image)
        except Exception:
            logger.exception("tamper check failed on %s", self.camera.camera_id)
            return
        if isinstance(result, TamperReading):
            self.emit_event({"type": "CAMERA_TAMPERED", "data": {
                "kind": result.kind, "detail": result.detail,
                "baseline": result.baseline, "current": result.current,
            }})
        elif result == "restored":
            self.emit_event({"type": "CAMERA_VIEW_RESTORED", "data": {}})

    def _observe_tracks(self, image, tracks) -> None:
        people = sum(1 for _tid, cls, _box in tracks if cls == PERSON_CLASS)
        group = self.groups.observe(people)
        if group is not None:
            self.emit_event({"type": "GROUP_GATHERING", "data": group})
        for observer in list(self.track_observers):
            try:
                observer(self, image, tracks)
            except Exception:
                logger.exception("track observer failed on %s", self.camera.camera_id)

    # Above this share of changed pixels the analysis has reworked the whole
    # picture (low-light or fog enhancement), not just drawn on it, so the
    # analysed frames are shown as they are instead of a raw feed.
    WHOLE_FRAME_CHANGE = 0.4
    DISPLAY_MAX_FPS = 25.0

    def _before_analysis(self, image) -> None:
        self._check_tamper(image)
        self._analysis_clean = image.copy()

    def _on_analysed_frame(self, image, new_events: list[dict]) -> None:
        self._analysed_times.append(time.monotonic())
        clean = self._analysis_clean
        if clean is not None and clean.shape == image.shape:
            # Whatever the pipeline drew - boxes, labels, zones, banners - is
            # exactly the pixels that changed. Keep them as the overlay.
            mask = cv2.absdiff(image, clean).max(axis=2) > 0
            if mask.mean() > self.WHOLE_FRAME_CHANGE:
                self._overlay, self._show_analysed = None, True
            else:
                self._overlay, self._show_analysed = (mask, image.copy()), False
        else:
            self._overlay, self._show_analysed = None, True
        if self._show_analysed:
            self._publish(image)
        for event in new_events:
            self.emit_event(event)

    def _on_camera_frame(self, frame) -> None:
        """Every frame the camera delivers, with the analysis's latest
        drawings laid over it - so the video plays at camera speed while the
        boxes update at analysis speed."""
        if self._show_analysed:
            return
        now = time.monotonic()
        if now - self._last_display_at < 1.0 / self.DISPLAY_MAX_FPS:
            return
        self._last_display_at = now
        overlay = self._overlay
        image = frame.image
        if overlay is not None and overlay[0].shape == image.shape[:2]:
            image = image.copy()
            image[overlay[0]] = overlay[1][overlay[0]]
        self._publish(image)

    @property
    def analysis_fps(self) -> float:
        times = self._analysed_times
        if len(times) < 2 or time.monotonic() - times[-1] > 3.0:
            return 0.0
        return round((len(times) - 1) / (times[-1] - times[0]), 2) if times[-1] > times[0] else 0.0

    def emit_event(self, event: dict) -> dict:
        """Stamp an event with camera, severity and reason, then publish it.

        Every event reaches operators through here - detections from the
        pipeline and camera-health transitions alike - so a camera going down
        is delivered, logged, prioritised and alarmed by exactly the same path
        as an intrusion. A failure the operator never hears about is
        indistinguishable from a quiet border.
        """
        severity = severity_for(event)
        # Most events carry the zone at the top level; behaviour escalations
        # carry it inside data, so check both.
        zone_id = event.get("zone") or (event.get("data") or {}).get("zone_id")
        zone_label = self._zone_labels.get(zone_id)
        if zone_label:
            event = {**event, "zone_label": zone_label}
        # An alarm matching a pattern the guards have marked false before is
        # still logged and shown, but does not sound the siren.
        likely_false = None
        if severity in ALARM_SEVERITIES:
            likely_false = false_alarm_memory.check({**event, "camera_name": self.camera.name, "wall_time": time.time()})
        reason = describe_event(event)
        if likely_false:
            reason += (f" [likely false: marked false {likely_false['marks']} times here around"
                       f" {likely_false['around']}" + (f", '{likely_false['note']}'" if likely_false["note"] else "") + "]")
        enriched = {
            "track_id": None, "frame_index": None, "timestamp_sec": 0.0,
            **event,
            # The pipeline numbers events per run ("evt_7"), which repeats
            # across cameras and restarts. Consumers need a stable unique
            # id to deduplicate and to use as a render key, so one is
            # assigned here where the event enters the live world.
            "event_id": uuid.uuid4().hex[:12],
            "seq": event.get("event_id"),
            "camera_id": self.camera.camera_id,
            "camera_name": self.camera.name,
            "severity": severity,
            "reason": reason,
            "alarm": False if likely_false or event.get("type") in SILENT_TYPES else self._claim_alarm(event.get("type", ""), severity),
            "likely_false": likely_false,
            "wall_time": time.time(),
        }
        # Evidence is captured for exactly the events that raise an alarm.
        # Recording every routine detection would bury the incidents that
        # matter under hours of ordinary footage, and cost disk to do it.
        if enriched["alarm"]:
            bundle = self.evidence.capture(enriched)
            if bundle:
                enriched["evidence_bundle"] = bundle

        self._recent_events.append(enriched)
        if self.store is not None:
            self.store.record(enriched)
        self._notify_event(enriched)
        return enriched

    # --- health -------------------------------------------------------------

    def _awaiting_sender(self, exc: Exception) -> bool:
        """Whether this failure is just a pushed camera with nobody sending.

        Worth separating from a genuine fault: a phone carried out of WiFi
        range, or one whose screen locked, is an expected part of using a
        handset as a camera. Reporting it as a crashed capture thread makes an
        ordinary gap look like a defect in the system.
        """
        return (
            isinstance(exc, VideoOpenError)
            and self.camera.source.startswith(PUSH_SCHEME)
        )

    def check_health(self) -> dict | None:
        """Emit an event when this camera's health actually changes.

        Only transitions are reported. Re-announcing "still offline" every
        couple of seconds would bury real detections under status noise, which
        is the same reason the siren is gated.
        """
        status = self.status_dict()
        state = status["state"]
        if status["degraded"]:
            health = "CAMERA_DEGRADED"
        elif state in (ONLINE,):
            health = "CAMERA_ONLINE"
        elif state in (OFFLINE, RECONNECTING):
            health = "CAMERA_OFFLINE"
        else:
            health = None  # connecting/stopped are not incidents

        if health is None or health == self._last_health:
            self._last_health = health if health is not None else self._last_health
            return None

        first_observation = self._last_health is None
        self._last_health = health
        # Coming up for the first time is normal startup, not an incident
        # worth announcing to a control room.
        if first_observation and health == "CAMERA_ONLINE":
            return None

        return self.emit_event({
            "type": health,
            "data": {
                "state": state,
                "frame_age_sec": status["frame_age_sec"],
                "reconnect_count": status["reconnect_count"],
                "error": status["error"],
            },
        })

    def _claim_alarm(self, event_type: str, severity: str) -> bool:
        """Whether this event should physically sound the siren.

        Severity decides whether an event is alarm-worthy at all; this decides
        whether sounding it again right now tells the operator anything new. A
        siren that re-fires every couple of seconds for one ongoing incident
        trains people to ignore it, which is worse than no siren.
        """
        if severity not in ALARM_SEVERITIES:
            return False
        now = time.monotonic()
        last = self._alarm_gate.get(event_type)
        if last is not None and now - last < self.ALARM_COOLDOWN_SEC:
            return False
        self._alarm_gate[event_type] = now
        return True

    # --- event subscription --------------------------------------------------

    def subscribe(self, callback) -> None:
        """Register a callback invoked with each new event dict."""
        with self._event_lock:
            self._event_subscribers.append(callback)

    def unsubscribe(self, callback) -> None:
        with self._event_lock:
            if callback in self._event_subscribers:
                self._event_subscribers.remove(callback)

    def _notify_event(self, event: dict) -> None:
        with self._event_lock:
            subscribers = list(self._event_subscribers)
        for callback in subscribers:
            try:
                callback(event)
            except Exception:
                # One broken subscriber (a dropped websocket, say) must not
                # take down the analysis thread feeding everyone else.
                logger.exception("event subscriber failed")

    def recent_events(self, limit: int = 100) -> list[dict]:
        events = list(self._recent_events)
        return events[-limit:]

    def _publish(self, image) -> None:
        ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return
        jpeg = buf.tobytes()
        with self._condition:
            self._latest_jpeg = jpeg
            self._frame_seq += 1
            self._condition.notify_all()
        # The same encoded frame feeds the evidence buffer, so keeping a
        # rolling window costs no extra encoding. Sampled at the clip's own
        # rate: the display now runs at camera speed, and a clip written at
        # clip_fps from every display frame would play back in slow motion.
        now = time.monotonic()
        if now - self._last_evidence_at >= 1.0 / self.evidence.config.clip_fps:
            self._last_evidence_at = now
            self.evidence.add_frame(jpeg)

    # --- viewer side ---------------------------------------------------------

    def wait_for_frame(self, last_seq: int) -> tuple[bytes | None, int]:
        """Block until a frame newer than `last_seq` is available.

        Returns (jpeg, seq). Viewers pass the seq back on the next call, which
        means a slow viewer naturally skips frames it missed instead of
        building an ever-growing backlog - the right behaviour for live video,
        where the newest frame is the only one that matters.
        """
        with self._condition:
            if self._frame_seq <= last_seq:
                self._condition.wait(timeout=self.VIEWER_WAIT_TIMEOUT)
            return self._latest_jpeg, self._frame_seq

    # --- status --------------------------------------------------------------

    def status_dict(self) -> dict:
        status = self.reader.status.as_dict()
        state = status["state"]
        if not self.is_running and state not in (OFFLINE, STOPPED):
            state = STOPPED
        elif self.is_running and state == STOPPED:
            # The session is alive and swapping in a fresh reader after an
            # outage. "Stopped" would suggest an operator halted it, when in
            # fact recovery is in progress.
            state = RECONNECTING
        if self.faulted:
            # During an injected outage the camera really is unavailable, so
            # it reports unavailable - the point is to exercise the honest
            # path, not to dress up a healthy camera as a broken one.
            state = OFFLINE
            status["error"] = "link down (injected fault for rehearsal)"
        age = status["frame_age_sec"]
        # A camera can be "connected" and still useless: the socket is open and
        # frames stopped arriving. Surfacing that as DEGRADED is what lets the
        # operator tell a frozen feed from a working one at a glance.
        degraded = bool(state == "online" and age is not None and age > 5.0)
        return {
            **self.camera.to_dict(),
            **status,
            "state": state,
            "degraded": degraded,
            "running": self.is_running,
            "uptime_sec": round(time.time() - self._started_at, 1) if self._started_at else None,
            "modules": self.analysis.module_names(),
            "reference_set_id": self.analysis.reference_set_id,
            "target_plate": self.analysis.target_plate,
            "zone_ids": [z.zone_id for z in self.analysis.zones],
            "analysis_error": self._analysis_error,
            # measured_fps is the video (camera) rate; this is how many of
            # those frames the analysis gets through.
            "analysis_fps": self.analysis_fps if self.analysis.any_enabled else None,
            "event_count": len(self._recent_events),
            # Share of frames the detector skipped because nothing moved.
            "ai_idle_share": round(self.motion_gate.idle_share, 2) if self.motion_gate else None,
        }

    def inject_fault(self, seconds: float = 20.0) -> float:
        """Genuinely break this camera's feed for a while.

        Used to rehearse and demonstrate recovery. It does not fake a status:
        the capture is torn down and pointed at an unreachable source, so the
        real reconnect-with-backoff path runs and real CAMERA_OFFLINE /
        CAMERA_ONLINE events are emitted. A simulated outage that only changed
        a label would prove nothing about whether recovery works.
        """
        seconds = max(1.0, min(seconds, 300.0))
        self._fault_until = time.monotonic() + seconds
        # Dropping the reader mid-read is exactly what a yanked cable does.
        self.reader.stop()
        return seconds

    @property
    def faulted(self) -> bool:
        return time.monotonic() < self._fault_until

    def configure_analysis(self, spec: AnalysisSpec) -> None:
        """Change which modules this camera runs.

        Takes effect on the next start: the pipeline builds its trackers,
        reference galleries and monitors once at startup, so swapping modules
        mid-run would leave half-initialised state behind.
        """
        was_running = self.is_running
        self.analysis = spec
        self._analysis_error = None

        def restart() -> None:
            if was_running:
                self.stop()
            # A stopped reader cannot be restarted - its iterator has already
            # exited - so a fresh one is required for the next run.
            self.reader = self._new_reader()
            if was_running:
                self.start()

        # Stopping waits for the capture to release the device and starting
        # reloads models; done inside the request, that froze the dashboard
        # for up to half a minute. The feed shows a status card meanwhile.
        threading.Thread(target=restart, name=f"reconfigure-{self.camera.camera_id}", daemon=True).start()


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
