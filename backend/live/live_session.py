"""One camera's capture session.

Split out of backend.live.camera_manager (which re-exports LiveSession)
once that module passed 1,100 lines: a LiveSession owns one capture thread,
its analysis pipeline, crash recovery, the frames viewers see and the events
it raises. CameraManager - the registry of sessions - stays in camera_manager.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque

import cv2
import numpy as np

from backend.config import EVIDENCE_DIR
from backend.core.events import ALARM_SEVERITIES, SILENT_TYPES, describe_event, severity_for
from backend.core.video import (
    LiveStreamReader, OFFLINE, ONLINE, PUSH_SCHEME, PushStreamReader, RECONNECTING, STOPPED, VideoOpenError,
)
from backend.core.motion import MotionGate
from backend.live.crowd import PERSON_CLASS, GroupWatch
from backend.live.evidence import EvidenceRecorder
from backend.live.frame_feed import LiveFrameFeed
from backend.live.false_alarms import memory as false_alarm_memory
from backend.live.live_config import (
    AnalysisSpec, Camera, live_anpr_config, live_behavior_config, live_person_id_config, live_tracker_config,
)
from backend.live.tamper import TamperMonitor, TamperReading, TappedReader
from backend.live.target_registry import target_registry
from backend.live.store import EventStore
from backend.core.calibration import load_calibration
from backend.orchestrator import CombinedPipeline, OrchestratorError, OrchestratorRequest

logger = logging.getLogger("backend.live")

# Analysis failures that come from the configuration, not a passing fault:
# retrying them cannot succeed, so the camera falls back to plain video.
_UNUSABLE_ANALYSIS_ERRORS = (OrchestratorError, FileNotFoundError)


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
    # Restart after a crash: doubling from MIN up to MAX, reset once a run has
    # stayed up for HEALTHY_RUN seconds.
    RESTART_BACKOFF_MIN_SEC = 1.0
    RESTART_BACKOFF_MAX_SEC = 15.0
    RESTART_HEALTHY_RUN_SEC = 60.0

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
        # Why analysis was switched off for this run, if it was (see _disable_analysis).
        self._disabled_reason: str | None = None
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

    def _sleep_unless_stopping(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while not self._stopping and time.monotonic() < deadline:
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    def _clear_latest_frame(self) -> None:
        """Forget the last frame, so viewers get a status card rather than a
        frozen picture that looks live while the camera is down."""
        with self._condition:
            self._latest_jpeg = None
            self._frame_seq += 1
            self._condition.notify_all()

    def _run_video_only(self) -> None:
        for frame in self.reader:
            self._check_tamper(frame.image)
            self._publish(frame.image)
            if self._stopping:
                break

    def _run(self) -> None:
        # Outer loop so an outage does not end the session. A camera that came
        # back only after someone pressed a button would make "automatic
        # recovery" a claim rather than a behaviour.
        backoff: float | None = self.RESTART_BACKOFF_MIN_SEC
        self._disabled_reason = None
        while not self._stopping:
            run_started = time.monotonic()
            analysing = self.analysis.any_enabled and self._disabled_reason is None
            crashed = False
            try:
                if analysing:
                    self._run_with_analysis()
                else:
                    self._run_video_only()
            except Exception as exc:  # pragma: no cover - defensive
                if analysing and isinstance(exc, _UNUSABLE_ANALYSIS_ERRORS):
                    self._disable_analysis(exc)
                    continue
                crashed = True
                self._record_crash(exc)
            backoff = self._next_attempt(crashed, run_started, backoff)
            if backoff is None:
                break

        with self._condition:
            self._condition.notify_all()

    def _disable_analysis(self, exc: Exception) -> None:
        """The configuration itself cannot work (a reference photo with no
        face, a model file that is not there). Retrying would fail the same
        way forever, so keep the video and say why the analysis is off instead
        of taking the whole camera down. Only an analysis attempt lands here,
        so a failing video-only run still backs off like any crash."""
        logger.error("analysis disabled on %s: %s", self.camera.camera_id, exc)
        self._disabled_reason = f"analysis disabled: {exc}"
        self._analysis_error = self._disabled_reason
        if not self._stopping:
            self.reader.stop()
            self.reader = self._new_reader()

    def _record_crash(self, exc: Exception) -> None:
        """Make a failure visible. A capture thread dying silently would leave
        the UI showing a camera that looks fine but never updates, so the
        failure goes into the status the health board already reads."""
        self._clear_latest_frame()
        waiting = self._awaiting_sender(exc)
        if waiting:
            # A phone that walked out of range is ordinary, not a crash, and
            # logging a stack trace for it would train people to ignore the log.
            logger.info("camera %s is waiting for its sending device", self.camera.camera_id)
        else:
            logger.exception("live session %s failed", self.camera.camera_id)
        self.reader.status.state = OFFLINE
        self.reader.status.error = "waiting for the sending device" if waiting else f"capture thread crashed: {exc}"
        if waiting:
            self._analysis_error = self._disabled_reason
        elif self._disabled_reason:
            # Keep saying why analysis is off; the video fault is extra.
            self._analysis_error = f"{self._disabled_reason}; video: {exc}"
        else:
            self._analysis_error = str(exc)

    def _next_attempt(self, crashed: bool, run_started: float, backoff: float) -> float | None:
        """What to do after a run ended: the backoff to carry into the next
        attempt, or None to end the session."""
        if self._stopping:
            return None
        if self.faulted:
            return backoff if self._ride_out_fault() else None
        if self.camera.source.startswith(PUSH_SCHEME):
            # A pushed camera has no URL to retry, so the old code let the
            # session thread die and relied on the next incoming websocket to
            # restart it. Staying alive and waiting means the feed resumes by
            # itself the moment frames come back.
            time.sleep(self.SENDER_RETRY_SEC)
            if self._stopping:
                return None
            self.reader = self._new_reader()
            return backoff
        if not crashed:
            # The source ended by itself: a recording without loop=True has
            # simply finished, which is not an outage to recover from.
            return None
        return self._restart_after_crash(run_started, backoff)

    def _ride_out_fault(self) -> bool:
        """Wait out an injected outage, then reconnect from scratch - the same
        thing that happens when a cable is plugged back in."""
        while self.faulted and not self._stopping:
            time.sleep(0.25)
        if self._stopping:
            return False
        logger.info("camera %s recovering from injected fault", self.camera.camera_id)
        self.reader = self._new_reader()
        self._analysis_error = self._disabled_reason
        return True

    def _restart_after_crash(self, run_started: float, backoff: float) -> float | None:
        """Any other crash - a detector error, a camera slow to open, a bad
        frame - used to end the session, leaving the feed frozen until someone
        pressed Configure. Retry from a fresh reader and pipeline instead,
        backing off so a persistent fault does not spin the CPU or flood the
        log. Returns the next backoff, or None if the session was stopped."""
        if time.monotonic() - run_started > self.RESTART_HEALTHY_RUN_SEC:
            backoff = self.RESTART_BACKOFF_MIN_SEC
        logger.info("restarting camera %s in %.1fs", self.camera.camera_id, backoff)
        self._sleep_unless_stopping(backoff)
        if self._stopping:
            return None
        self.reader.stop()
        self.reader = self._new_reader()
        if self._stopping:
            # stop() landed while the reader was being rebuilt; it stopped the
            # old one, so stop this one too rather than leak it.
            self.reader.stop()
            return None
        # The crash message is cleared on retry, but a configuration that
        # disabled analysis still applies and must stay visible.
        self._analysis_error = self._disabled_reason
        return min(backoff * 2, self.RESTART_BACKOFF_MAX_SEC)

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


