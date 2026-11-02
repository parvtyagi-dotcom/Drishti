"""Sensor fusion: other sensors raise an alarm, the cameras check it.

A border post is not only cameras. Fence vibration cables, buried ground
sensors, ground radar and acoustic arrays all raise alarms, and on their own
they trip on animals, wind and rain often enough that guards learn to ignore
them. A camera can settle most of those alarms in seconds, so this module puts
the two together:

  1. A sensor reports a trip (over its own key, or an operator's test).
  2. SENSOR_TRIGGERED is logged at once, without the siren.
  3. A few seconds later the linked camera's recent frames, from just before
     the trip to just after, are run through the person and vehicle detector.
  4. The outcome is one of three events:
       SENSOR_ALARM_CONFIRMED    someone or something was seen: siren
       SENSOR_ALARM_UNVERIFIED   the camera could not look: siren, because
                                 an unchecked alarm must never be dismissed
       SENSOR_ALARM_UNCONFIRMED  the camera looked and saw nothing: logged
                                 quietly, raised again if it keeps happening

An acoustic drone sensor is the exception. A small drone a hundred metres up
is a few pixels on a ground camera and is not something the detector knows,
so an empty frame proves nothing and the alert sounds on the sensor's word.
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import tempfile
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from backend.config import DATA_DIR, YOLO_MODEL_PATH

logger = logging.getLogger("backend.live.sensors")

# What each kind of sensor is called on screen and in alerts.
SENSOR_KINDS: dict[str, str] = {
    "fence": "Fence vibration sensor",
    "ground": "Ground sensor",
    "radar": "Ground radar",
    "pir": "Motion sensor",
    "acoustic_drone": "Acoustic drone sensor",
    "seismic": "Digging sensor",
}
# Kinds a ground camera cannot confirm or rule out.
# Kinds a ground camera cannot confirm or rule out, and the event each raises:
# a drone is too small and high, and a tunnel is underground.
UNCHECKABLE_KINDS = {"acoustic_drone": "DRONE_SUSPECTED", "seismic": "DIGGING_SUSPECTED"}

PERSON_CLASS = 0
VEHICLE_CLASSES = (2, 3, 5, 7)

# The source id and name on events from a sensor with no camera linked.
SENSOR_SOURCE_ID = "sensors"
SENSOR_SOURCE_NAME = "Post sensors"


@dataclass
class Sensor:
    sensor_id: str
    name: str
    kind: str
    # Linked camera, by name (ids change when a camera is registered again).
    camera: str | None = None
    # The secret a real sensor sends with each trip.
    key: str = field(default_factory=lambda: secrets.token_urlsafe(18))
    created: float = field(default_factory=time.time)

    @property
    def label(self) -> str:
        return SENSOR_KINDS.get(self.kind, "Sensor")

    def public(self, with_key: bool = False) -> dict:
        out = asdict(self)
        out["label"] = self.label
        out["checkable"] = self.kind not in UNCHECKABLE_KINDS
        if not with_key:
            out.pop("key")
        return out


class SensorRegistry:
    """The post's sensors, saved to a small JSON file next to the site map."""

    def __init__(self, path: Path | None = None):
        self.path = path or DATA_DIR / "sensors.json"
        self._lock = threading.Lock()
        self._sensors: dict[str, Sensor] = {}
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = []
        for item in raw if isinstance(raw, list) else []:
            try:
                sensor = Sensor(**{k: item[k] for k in Sensor.__dataclass_fields__ if k in item})
            except TypeError:
                continue
            self._sensors[sensor.sensor_id] = sensor

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump([asdict(s) for s in self._sensors.values()], f, indent=2)
        os.replace(tmp, self.path)

    def list(self) -> list[Sensor]:
        with self._lock:
            return sorted(self._sensors.values(), key=lambda s: s.created)

    def get(self, sensor_id: str) -> Sensor | None:
        with self._lock:
            return self._sensors.get(sensor_id)

    def add(self, name: str, kind: str, camera: str | None) -> Sensor:
        if kind not in SENSOR_KINDS:
            raise ValueError(f"unknown sensor kind '{kind}'")
        sensor = Sensor(sensor_id=uuid.uuid4().hex[:8], name=name.strip(), kind=kind,
                        camera=(camera or "").strip() or None)
        with self._lock:
            self._sensors[sensor.sensor_id] = sensor
            self._save()
        return sensor

    def remove(self, sensor_id: str) -> bool:
        with self._lock:
            if self._sensors.pop(sensor_id, None) is None:
                return False
            self._save()
        return True

    def check_key(self, sensor_id: str, key: str | None) -> Sensor | None:
        sensor = self.get(sensor_id)
        if sensor is None or not key or not secrets.compare_digest(sensor.key, key):
            return None
        return sensor


class YoloLook:
    """Counts people and vehicles in a frame with the same YOLO model the
    cameras use. Loaded on first use and shared, so sensors cost nothing
    until one trips."""

    def __init__(self, model_path: Path = YOLO_MODEL_PATH, confidence: float = 0.35):
        self.model_path = model_path
        self.confidence = confidence
        self._model = None
        self._lock = threading.Lock()

    def warm(self) -> None:
        """Load the model ahead of the first trip, so the first alarm is not
        the one that waits for it."""
        self(np.zeros((64, 64, 3), dtype=np.uint8))

    def __call__(self, image: np.ndarray) -> tuple[int, int, float]:
        with self._lock:
            if self._model is None:
                from ultralytics import YOLO
                self._model = YOLO(str(self.model_path))
            results = self._model.predict(image, conf=self.confidence, verbose=False,
                                          classes=[PERSON_CLASS, *VEHICLE_CLASSES])
        boxes = results[0].boxes if results else None
        if boxes is None or len(boxes) == 0:
            return 0, 0, 0.0
        classes = boxes.cls.int().cpu().tolist()
        confs = boxes.conf.cpu().tolist()
        persons = sum(1 for c in classes if c == PERSON_CLASS)
        vehicles = sum(1 for c in classes if c in VEHICLE_CLASSES)
        return persons, vehicles, max(confs)


@dataclass
class CheckResult:
    outcome: str  # confirmed | unconfirmed | unverified
    persons: int = 0
    vehicles: int = 0
    confidence: float = 0.0
    frames_checked: int = 0
    seen_offset_sec: float | None = None  # when, relative to the trip
    why: str | None = None


class SensorFusion:
    # The camera's view is checked from a little before the trip (someone may
    # already be in view as the fence moves) to a little after (or still
    # approaching it).
    LOOK_BEFORE_SEC = 4.0
    LOOK_AFTER_SEC = 6.0
    MAX_FRAMES = 8
    # Unconfirmed trips from one sensor inside this window are counted, so a
    # sensor that keeps going off with nothing on camera gets attention.
    REPEAT_WINDOW_SEC = 600.0

    def __init__(self, registry: SensorRegistry, manager, look=None,
                 clock=time.monotonic, wait=time.sleep):
        self.registry = registry
        self.manager = manager
        self.look = look or YoloLook()
        self.clock = clock
        self.wait = wait
        self._checking: set[str] = set()
        self._unconfirmed: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    # --- entry point ---------------------------------------------------------

    def trigger(self, sensor: Sensor, source: str = "sensor", detail: str | None = None,
                background: bool = True) -> dict:
        """Handle one trip. Returns at once with what was logged; the camera
        check runs on its own thread unless `background` is False."""
        with self._lock:
            if sensor.sensor_id in self._checking:
                return {"status": "already_checking"}
            self._checking.add(sensor.sensor_id)

        base = {
            "sensor_id": sensor.sensor_id, "sensor_name": sensor.name,
            "sensor_kind": sensor.kind, "sensor_label": sensor.label,
            "source": source, "camera": sensor.camera,
        }
        if detail:
            base["detail"] = detail[:200]

        if sensor.kind in UNCHECKABLE_KINDS:
            try:
                event = self._emit(sensor, UNCHECKABLE_KINDS[sensor.kind], base)
            finally:
                with self._lock:
                    self._checking.discard(sensor.sensor_id)
            return {"status": "alarm", "event": event}

        tripped_at = self.clock()
        first = self._emit(sensor, "SENSOR_TRIGGERED", base)
        if background:
            threading.Thread(target=self._check_and_report, args=(sensor, base, tripped_at),
                             name=f"sensor-check-{sensor.sensor_id}", daemon=True).start()
            return {"status": "checking", "event": first}
        final = self._check_and_report(sensor, base, tripped_at)
        return {"status": "checked", "event": first, "result": final}

    # --- the check -----------------------------------------------------------

    def _check_and_report(self, sensor: Sensor, base: dict, tripped_at: float) -> dict | None:
        try:
            self.wait(self.LOOK_AFTER_SEC)
            try:
                result = self.check(sensor, tripped_at)
            except Exception:
                logger.exception("sensor check failed for %s", sensor.sensor_id)
                result = CheckResult("unverified", why="the camera check failed")
            return self._report(sensor, base, result)
        finally:
            with self._lock:
                self._checking.discard(sensor.sensor_id)

    def check(self, sensor: Sensor, tripped_at: float) -> CheckResult:
        if not sensor.camera:
            return CheckResult("unverified", why="no camera is linked to this sensor")
        session = self.manager.find_by_name(sensor.camera)
        if session is None:
            return CheckResult("unverified", why=f"camera '{sensor.camera}' is not connected")
        status = session.status_dict()
        if status.get("state") != "online" or status.get("degraded"):
            return CheckResult("unverified", why=f"{session.camera.name} is {status.get('state')}")
        if session.tamper.tampered:
            return CheckResult("unverified", why=f"{session.camera.name}'s view is blocked or moved")

        frames = session.evidence.frames_since(tripped_at - self.LOOK_BEFORE_SEC)
        if not frames:
            return CheckResult("unverified", why=f"{session.camera.name} sent no video")
        picks = _spread(frames, self.MAX_FRAMES)
        best = CheckResult("unconfirmed", frames_checked=len(picks))
        for t, jpeg in picks:
            image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue
            persons, vehicles, conf = self.look(image)
            if persons + vehicles > best.persons + best.vehicles:
                best = CheckResult("confirmed", persons, vehicles, round(conf, 2), len(picks),
                                   round(t - tripped_at, 1))
        return best

    def _report(self, sensor: Sensor, base: dict, result: CheckResult) -> dict:
        data = {**base, "frames_checked": result.frames_checked}
        if result.outcome == "confirmed":
            data.update(persons=result.persons, vehicles=result.vehicles,
                        confidence=result.confidence, seen_offset_sec=result.seen_offset_sec)
            return self._emit(sensor, "SENSOR_ALARM_CONFIRMED", data)
        if result.outcome == "unverified":
            data["why"] = result.why
            return self._emit(sensor, "SENSOR_ALARM_UNVERIFIED", data)
        now = self.clock()
        with self._lock:
            recent = self._unconfirmed.setdefault(sensor.sensor_id, deque())
            recent.append(now)
            while recent and now - recent[0] > self.REPEAT_WINDOW_SEC:
                recent.popleft()
            data["repeat_count"] = len(recent)
        data["repeat_window_min"] = int(self.REPEAT_WINDOW_SEC // 60)
        return self._emit(sensor, "SENSOR_ALARM_UNCONFIRMED", data)

    def _emit(self, sensor: Sensor, etype: str, data: dict) -> dict:
        event = {"type": etype, "data": data}
        session = self.manager.find_by_name(sensor.camera) if sensor.camera else None
        if session is not None:
            return session.emit_event(event)
        return self.manager.emit_unattached(event, SENSOR_SOURCE_ID, SENSOR_SOURCE_NAME)


def _spread(items: list, count: int) -> list:
    """Up to `count` items evenly spaced across the list, ends included."""
    if len(items) <= count:
        return list(items)
    step = (len(items) - 1) / (count - 1)
    return [items[round(i * step)] for i in range(count)]
