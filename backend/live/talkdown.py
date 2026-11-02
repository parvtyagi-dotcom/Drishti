"""Talk-down: a spoken warning at the fence before someone crosses it.

Most systems record an intrusion. A warning played at the spot while the
person is still approaching can stop it happening. This module decides when to
warn and logs that it did; the sound comes from a speaker device near the
camera (any phone or PC with a loudspeaker, running /api/live/speaker), which
holds a websocket open for its camera.

A warning is only issued, and only logged, when at least one speaker for that
camera is connected. Logging "warning played" with no speaker to play it would
put a false record in the audit trail.
"""
from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger("backend.live.talkdown")

# What prompts an automatic warning: the moments before a crossing, and
# someone lingering. A crossing itself also warns, telling them to go back.
AUTO_TRIGGERS = frozenset({"ZONE_APPROACH", "LOITERING", "GROUP_GATHERING", "ZONE_ENTRY", "SENSOR_ALARM_CONFIRMED"})

MESSAGE = {
    "hi": "रुकिए! यह प्रतिबंधित सीमा क्षेत्र है। आप कैमरे की निगरानी में हैं। तुरंत पीछे हटें।",
    "en": "Stop. This is a restricted border area. You are being watched on camera. Turn back now.",
}


def _key(name: str) -> str:
    return " ".join(name.split()).lower()


class TalkDown:
    # One warning per camera in this window, however many events arrive: a
    # speaker repeating itself every second is noise, not deterrence.
    COOLDOWN_SEC = 20.0

    def __init__(self, manager, clock=time.monotonic):
        self.manager = manager
        self.clock = clock
        self.auto = True
        self._speakers: dict[str, int] = {}
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def start(self) -> None:
        self.manager.subscribe(self.on_event)

    def stop(self) -> None:
        self.manager.unsubscribe(self.on_event)

    # --- speakers --------------------------------------------------------------

    def speaker_connected(self, camera_name: str) -> None:
        with self._lock:
            key = _key(camera_name)
            self._speakers[key] = self._speakers.get(key, 0) + 1

    def speaker_disconnected(self, camera_name: str) -> None:
        with self._lock:
            key = _key(camera_name)
            count = self._speakers.get(key, 0) - 1
            if count > 0:
                self._speakers[key] = count
            else:
                self._speakers.pop(key, None)

    def speakers_for(self, camera_name: str) -> int:
        with self._lock:
            return self._speakers.get(_key(camera_name), 0)

    # --- warnings --------------------------------------------------------------

    def on_event(self, event: dict) -> None:
        if not self.auto or event.get("type") not in AUTO_TRIGGERS:
            return
        name = event.get("camera_name") or ""
        if not self.speakers_for(name):
            return
        session = self.manager.find_by_name(name)
        if session is None:
            return
        self.warn(session, trigger=event.get("type"), by=None, respect_cooldown=True,
                  track_id=event.get("track_id"))

    def warn(self, session, trigger: str | None, by: str | None,
             respect_cooldown: bool = False, track_id=None) -> dict | None:
        """Issue a warning at this camera's speakers. Returns the logged event,
        or None when no speaker is listening or it is too soon to repeat."""
        name = session.camera.name
        if not self.speakers_for(name):
            return None
        now = self.clock()
        with self._lock:
            last = self._last.get(_key(name))
            if respect_cooldown and last is not None and now - last < self.COOLDOWN_SEC:
                return None
            self._last[_key(name)] = now
        return session.emit_event({
            "type": "TALKDOWN_ISSUED",
            "track_id": track_id,
            "data": {
                "trigger": trigger, "auto": by is None, "by": by,
                "speakers": self.speakers_for(name), "lines": MESSAGE,
            },
        })
