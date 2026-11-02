"""Deduplicated event stream shared by all module pipelines.

The point of EventBus is to stop pipelines from emitting hundreds of
near-duplicate events for the same object. Emission is keyed by
(event_type, track_id); a repeat of the same key is dropped unless the
configured cooldown has elapsed, or the emitter explicitly marks a state
transition (e.g. ENTER -> DWELL -> EXIT) via `state`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

INFO = "INFO"
LOW = "LOW"
MEDIUM = "MEDIUM"
HIGH = "HIGH"
CRITICAL = "CRITICAL"

# Severity is a property of the *rule that fired*, not of a model's confidence
# score. A confirmed target crossing a restricted boundary is critical whether
# the match scored 0.71 or 0.94 - what makes it critical is which rule it
# satisfied. Confidence still travels on the event for the operator to weigh;
# it just isn't what decides urgency.
EVENT_SEVERITY: dict[str, str] = {
    # A known target inside a restricted area, or acting on it - the events
    # the whole system exists to surface.
    "TARGET_ZONE_INTRUSION": CRITICAL,
    "TARGET_BEHAVIOR_ALERT": CRITICAL,
    # Crossing into a restricted area is the incident a zone exists to catch,
    # whether or not the person is a known target, so it sounds the siren on
    # its own. ZONE_APPROACH below is the warning that precedes it and stays
    # deliberately below alarm severity: something that has not happened yet
    # must not be able to cry wolf.
    "ZONE_ENTRY": CRITICAL,
    # A watched person or vehicle positively identified on camera.
    "TARGET_CONFIRMED": HIGH,
    "TARGET_REACQUIRED": HIGH,
    "TARGET_VEHICLE_FOUND": HIGH,
    "TARGET_VEHICLE_REACQUIRED": HIGH,
    "TARGET_CROSS_CAMERA_MATCH": HIGH,
    "CAMERA_OFFLINE": HIGH,
    # Sentinel. A blinded camera is as useful to an intruder as a missing one,
    # and one that dies right after activity nearby may have been cut.
    "CAMERA_TAMPERED": HIGH,
    "CAMERA_SABOTAGE_SUSPECTED": CRITICAL,
    # Sensor fusion (live/sensors.py). A fence or ground sensor on its own
    # trips on animals and wind, so its alarm is checked against the camera
    # first: seen on video is an intrusion; a camera that could not look must
    # not dismiss it; nothing seen is logged, not sounded.
    "SENSOR_ALARM_CONFIRMED": CRITICAL,
    "SENSOR_ALARM_UNVERIFIED": HIGH,
    # A drone is too small and high for the ground cameras to confirm, so an
    # acoustic drone alert sounds on the sensor's word alone.
    "DRONE_SUSPECTED": HIGH,
    # Rhythmic ground vibration: someone digging, possibly a tunnel. Cameras
    # cannot see underground, so it sounds on the sensor's word.
    "DIGGING_SUSPECTED": HIGH,
    "SENSOR_TRIGGERED": MEDIUM,
    "SENSOR_ALARM_UNCONFIRMED": LOW,  # MEDIUM when it repeats; see severity_for
    # Something worth an operator's attention, but not an alarm.
    "ZONE_APPROACH": MEDIUM,
    "LONG_DWELL": MEDIUM,
    "LOITERING": MEDIUM,
    "GROUP_GATHERING": MEDIUM,
    "PLATE_CONFIRMED": MEDIUM,
    "TARGET_LOST": MEDIUM,
    "TARGET_VEHICLE_LOST": MEDIUM,
    "CAMERA_DEGRADED": MEDIUM,
    # Context: real, logged, but routine on its own.
    "ZONE_EXIT": LOW,
    "SUDDEN_DIRECTION_CHANGE": LOW,
    "ABNORMAL_SPEED": LOW,
    "REPEATED_BACK_AND_FORTH": LOW,
    "PLATE_DETECTED": LOW,
    # Routine observations that belong in the log, not in anyone's face.
    "VEHICLE_DETECTED": INFO,
    "LOW_LIGHT": INFO,
    "FOG_DETECTED": INFO,
    "CAMERA_ONLINE": INFO,
    "CAMERA_VIEW_RESTORED": INFO,
    # A record that a spoken warning was played, not an alarm in itself.
    "TALKDOWN_ISSUED": INFO,
    "ALARM_VERDICT": INFO,
}

# What an external siren/alarm responds to. Deliberately narrow: an alarm that
# fires on routine events stops being treated as an alarm.
ALARM_SEVERITIES = frozenset({HIGH, CRITICAL})
# Shown and logged as HIGH, but silent: a tamper warning fired on people
# walking close to the lens, and an operator who hears false sirens stops listening.
SILENT_TYPES = frozenset({"CAMERA_TAMPERED"})


def severity_of(event_type: str) -> str:
    """Severity for an event type; unknown types default to INFO rather than
    raising, so adding a new event can never crash the alert path."""
    return EVENT_SEVERITY.get(event_type, INFO)


def severity_for(event: dict) -> str:
    """Severity for a specific event, which for some types depends on what
    actually triggered it rather than on the type alone.

    TARGET_BEHAVIOR_ALERT is raised by behaviour/state.py in two different
    situations: the *confirmed target* behaving oddly, or *anyone* behaving
    oddly inside a restricted zone. Only the first is a person of interest, so
    only the first is allowed to reach alarm severity and sound the siren. The
    zone-driven case is still logged and shown - it just doesn't claim to be
    something it isn't.
    """
    etype = event.get("type", "")
    if etype == "TARGET_BEHAVIOR_ALERT":
        return CRITICAL if (event.get("data") or {}).get("is_target") else MEDIUM
    if etype == "SENSOR_ALARM_UNCONFIRMED":
        # One trip with nothing on camera is usually an animal or the wind.
        # The same sensor tripping again and again is worth a look: someone
        # may be working the fence just outside the camera's view.
        return MEDIUM if (event.get("data") or {}).get("repeat_count", 1) >= 3 else LOW
    return severity_of(etype)


def should_alarm(event: dict) -> bool:
    return event.get("severity", severity_of(event.get("type", ""))) in ALARM_SEVERITIES


def describe_event(event: dict) -> str:
    """One plain-language sentence explaining why this event fired.

    Every alert an operator is expected to act on has to say what triggered
    it. "Suspicious activity: 91%" is not something a guard can verify,
    escalate or dismiss; "entered restricted zone and stayed 18 seconds" is.
    The numbers behind the decision stay on the event either way - this is
    the human-readable summary of them, not a replacement.
    """
    etype = event.get("type", "UNKNOWN")
    data = event.get("data") or {}
    track = event.get("track_id")
    who = f"Track #{track}" if track is not None else "Subject"
    # Prefer the operator's own name for the zone ("BORDER LINE") over its
    # internal id - the id means nothing to the person reading the alert.
    label = event.get("zone_label")
    zone = event.get("zone")
    zone_label = f"'{label}'" if label else (f"zone {zone}" if zone else "restricted zone")

    def num(key, fmt="{:.1f}"):
        value = data.get(key)
        return fmt.format(value) if isinstance(value, (int, float)) else None

    if etype == "TARGET_CONFIRMED":
        votes, sim = data.get("votes"), data.get("similarity")
        detail = []
        if votes is not None:
            detail.append(f"{votes} reference photo match{'es' if votes != 1 else ''}")
        if sim is not None:
            detail.append(f"similarity {sim}")
        return "Target person identified" + (f" ({', '.join(detail)})" if detail else "")
    if etype == "TARGET_CROSS_CAMERA_MATCH":
        sim = data.get("similarity")
        detail = f" (appearance match {sim})" if sim is not None else ""
        return f"Target picked up on this camera from another camera's sighting{detail}"
    if etype == "TARGET_REACQUIRED":
        return f"Target re-identified after being out of view ({who})"
    if etype == "TARGET_LOST":
        return "Target no longer visible on this camera"
    if etype == "TARGET_ZONE_INTRUSION":
        return f"Confirmed target crossed into {zone_label}"
    if etype == "TARGET_BEHAVIOR_ALERT":
        trigger = str(data.get("trigger_behavior", "unusual movement")).replace("_", " ").lower()
        # Same event type, two very different meanings - and claiming a
        # confirmed target when none is enrolled is exactly the kind of
        # overclaim that destroys trust in the whole alert stream.
        if data.get("is_target"):
            return f"Confirmed target showed {trigger}"
        return f"{who} showed {trigger} inside {zone_label}"
    if etype == "ZONE_APPROACH":
        secs = num("eta_sec")
        timing = f" within about {secs} seconds" if secs else " shortly"
        return f"{who} is heading toward {zone_label} and is predicted to enter{timing}"
    if etype == "ZONE_ENTRY":
        return f"{who} crossed into {zone_label}"
    if etype == "ZONE_EXIT":
        return f"{who} left {zone_label}"
    if etype == "LONG_DWELL":
        secs = num("duration_sec")
        return f"{who} stayed in {zone_label}" + (f" for {secs} seconds" if secs else "")
    if etype == "LOITERING":
        secs = num("duration_sec")
        return f"{who} remained in a small area" + (f" for {secs} seconds" if secs else "")
    if etype == "REPEATED_BACK_AND_FORTH":
        count = data.get("reversal_count")
        return f"{who} reversed direction repeatedly" + (f" ({count} times)" if count else "")
    if etype == "SUDDEN_DIRECTION_CHANGE":
        deg = num("angle_degrees", "{:.0f}")
        return f"{who} changed direction sharply" + (f" ({deg} degrees)" if deg else "")
    if etype == "ABNORMAL_SPEED":
        return f"{who} moved unusually fast"
    if etype == "PLATE_CONFIRMED":
        plate = event.get("plate") or data.get("text")
        conf = event.get("confidence", data.get("confidence"))
        detail = f" at {conf:.0f}% confidence" if isinstance(conf, (int, float)) else ""
        return f"Number plate {plate} confirmed across multiple reads{detail}"
    if etype == "PLATE_DETECTED":
        return f"Number plate located on vehicle {who.lower()}, not yet confirmed"
    if etype == "TARGET_VEHICLE_FOUND":
        return "Vehicle matching the searched number plate identified"
    if etype == "TARGET_VEHICLE_REACQUIRED":
        return "Searched vehicle re-identified after being out of view"
    if etype == "TARGET_VEHICLE_LOST":
        return "Searched vehicle no longer visible on this camera"
    if etype == "VEHICLE_DETECTED":
        return f"Vehicle detected ({who})"
    if etype == "LOW_LIGHT":
        return "Low-light conditions detected; image enhancement applied"
    if etype == "FOG_DETECTED":
        return "Fog detected; frames are cleared of haze before detection"
    if etype == "CAMERA_OFFLINE":
        return "Camera stopped responding"
    if etype == "CAMERA_DEGRADED":
        return "Camera is connected but frames have stopped updating"
    if etype == "CAMERA_ONLINE":
        return "Camera connected"
    if etype == "CAMERA_TAMPERED":
        detail = data.get("detail") or "view changed suddenly"
        return f"Possible camera tampering: {detail}"
    if etype == "CAMERA_VIEW_RESTORED":
        return "Camera view back to normal"
    if etype == "CAMERA_SABOTAGE_SUSPECTED":
        count = data.get("recent_alerts")
        secs = data.get("seconds_since_last_alert")
        where = data.get("last_alert_camera")
        alerts = f"{count} alert{'s' if count != 1 else ''}" if count else "recent alerts"
        timing = f", the last {secs}s earlier" if secs is not None else ""
        place = f" (last on {where})" if where else ""
        return f"Camera went offline after {alerts} on the post{timing}{place}: possible sabotage"
    if etype == "GROUP_GATHERING":
        people, secs = data.get("people"), num("duration_sec", "{:.0f}")
        return f"Group of about {people} people stayed together in view" + (f" for {secs} seconds" if secs else "")
    if etype == "TALKDOWN_ISSUED":
        trigger = str(data.get("trigger") or "").replace("_", " ").lower()
        why = f"automatically, after {trigger}" if data.get("auto") else f"by {data.get('by') or 'an operator'}"
        return f"Spoken warning played at the camera's speaker ({why})"
    if etype.startswith(("SENSOR_", "DRONE_", "DIGGING_")):
        return _describe_sensor_event(etype, data)
    return etype.replace("_", " ").capitalize()


def _describe_sensor_event(etype: str, data: dict) -> str:
    kind = data.get("sensor_label") or "Sensor"
    name = data.get("sensor_name")
    sensor = f"{kind} '{name}'" if name else kind
    test = " (test trigger)" if data.get("source") == "simulated" else ""
    camera = data.get("camera") or "the linked camera"
    if etype == "SENSOR_TRIGGERED":
        check = f"checking {camera}" if data.get("camera") else "no camera linked to check it"
        return f"{sensor} triggered{test}; {check}"
    if etype == "SENSOR_ALARM_CONFIRMED":
        seen = []
        for key, word in (("persons", "person"), ("vehicles", "vehicle")):
            count = data.get(key) or 0
            if count:
                seen.append(f"{count} {word}{'s' if count != 1 else ''}")
        what = " and ".join(seen) or "movement"
        return f"{sensor} triggered{test} and {camera} saw {what}: confirmed intrusion"
    if etype == "SENSOR_ALARM_UNVERIFIED":
        why = data.get("why") or "the camera could not check"
        return f"{sensor} triggered{test} but {why}: treat as real until checked"
    if etype == "SENSOR_ALARM_UNCONFIRMED":
        repeat = data.get("repeat_count", 1)
        again = f", {repeat} times in {data.get('repeat_window_min', 10)} minutes" if repeat >= 2 else ""
        return f"{sensor} triggered{test}{again}; nothing seen on {camera} (animal, wind or fault?)"
    if etype == "DIGGING_SUSPECTED":
        detail = f" ({data['detail']})" if data.get("detail") else ""
        return f"{sensor} felt rhythmic ground vibration like digging{test}{detail}; check for tunnelling"
    if etype == "DRONE_SUSPECTED":
        return f"{sensor} heard a drone-like sound{test}; ground cameras cannot confirm a drone, look up"
    return etype.replace("_", " ").capitalize()


@dataclass
class Event:
    type: str
    track_id: int | None
    frame_index: int
    timestamp_sec: float
    data: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "type": self.type,
            "track_id": self.track_id,
            "frame_index": self.frame_index,
            "timestamp_sec": round(self.timestamp_sec, 3),
            **self.data,
        }


class EventBus:
    def __init__(self):
        self._events: list[Event] = []
        # (event_type, track_id) -> (last_emit_timestamp_sec, last_state)
        self._last: dict[tuple[str, int | None], tuple[float, str | None]] = {}

    def emit(
        self,
        event_type: str,
        track_id: int | None,
        frame_index: int,
        timestamp_sec: float,
        cooldown_sec: float = 0.0,
        state: str | None = None,
        **data,
    ) -> Event | None:
        """Emit an event unless it's a duplicate within the cooldown window.

        A change in `state` (e.g. moving from ENTER to DWELL) always bypasses
        the cooldown, since that's a real transition, not a repeat.
        """
        key = (event_type, track_id)
        prev = self._last.get(key)
        if prev is not None:
            prev_ts, prev_state = prev
            same_state = state is None or state == prev_state
            if same_state and (timestamp_sec - prev_ts) < cooldown_sec:
                return None

        event = Event(
            type=event_type,
            track_id=track_id,
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            data=data,
        )
        self._events.append(event)
        self._last[key] = (timestamp_sec, state)
        return event

    @property
    def events(self) -> list[Event]:
        return list(self._events)

    def as_dicts(self) -> list[dict]:
        return [e.as_dict() for e in self._events]
