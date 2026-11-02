"""False-alarm memory: the post teaches Drishti its own false alarms.

When a guard marks an alarm as false, the pattern is remembered: which
camera, what kind of alarm, which zone, and the time of day. Shadows that
cross a zone at sunset, cattle that come to the fence at dawn, headlights
that blind one camera at night: these repeat at the same place and hour.

Once the same pattern has been marked false often enough, a new alarm that
matches it is still logged and shown, marked "likely false" with the reason,
but it does not sound the siren. Marking one of those alarms as real wipes
the pattern, so one mistaken lesson cannot silence a camera.

Some alarms are never silenced however often they were marked false: a
watched person or vehicle, and suspected sabotage.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

from backend.config import DATA_DIR

NEVER_SILENCED_PREFIXES = ("TARGET_", "CAMERA_SABOTAGE")


def _key(name: str | None) -> str:
    return " ".join((name or "").split()).lower()


def _hour(ts: float) -> float:
    t = datetime.fromtimestamp(ts)
    return t.hour + t.minute / 60


def _hour_gap(a: float, b: float) -> float:
    gap = abs(a - b) % 24
    return min(gap, 24 - gap)


class FalseAlarmMemory:
    # Marked false this many times at the same camera, type, zone and hour...
    MIN_MARKS = 2
    # ...within this many hours of the time of day...
    HOUR_WINDOW = 1.0
    # ...in this many recent days.
    KEEP_DAYS = 14

    def __init__(self, path: Path | None = None, clock=time.time):
        self.path = path or DATA_DIR / "false_alarms.json"
        self.clock = clock
        self._lock = threading.Lock()
        self._marks: list[dict] = self._load()

    def _load(self) -> list[dict]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (OSError, ValueError):
            return []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(self._marks, f, indent=2)
        os.replace(tmp, self.path)

    def _prune(self) -> None:
        horizon = self.clock() - self.KEEP_DAYS * 86400
        self._marks = [m for m in self._marks if m["time"] >= horizon]

    @staticmethod
    def _pattern(event: dict) -> dict:
        return {
            "camera": _key(event.get("camera_name")),
            "type": event.get("type"),
            "zone": event.get("zone") or (event.get("data") or {}).get("zone_id"),
        }

    def _matching(self, pattern: dict, hour: float) -> list[dict]:
        return [m for m in self._marks
                if m["camera"] == pattern["camera"] and m["type"] == pattern["type"]
                and m["zone"] == pattern["zone"] and _hour_gap(m["hour"], hour) <= self.HOUR_WINDOW]

    # --- what guards teach it ---------------------------------------------------

    def mark(self, event: dict, verdict: str, by: str | None = None, note: str | None = None) -> dict:
        """Record a guard's verdict on an alarm: "false" or "real"."""
        pattern = self._pattern(event)
        when = event.get("wall_time") or self.clock()
        with self._lock:
            self._prune()
            if verdict == "real":
                # A real alarm at this place and hour wipes what was learnt
                # there, so a wrong lesson never keeps a camera quiet.
                before = len(self._marks)
                drop = {id(m) for m in self._matching(pattern, _hour(when))}
                self._marks = [m for m in self._marks if id(m) not in drop]
                forgotten = before - len(self._marks)
            else:
                self._marks.append({**pattern, "hour": round(_hour(when), 2), "time": when,
                                    "by": by, "note": (note or "")[:120] or None,
                                    "camera_name": event.get("camera_name")})
                forgotten = 0
            self._save()
            count = len(self._matching(pattern, _hour(when)))
        return {"verdict": verdict, "pattern_marks": count, "forgotten": forgotten}

    # --- what it does with that -------------------------------------------------

    def check(self, event: dict) -> dict | None:
        """If this alarm matches a learnt false-alarm pattern, say why."""
        etype = str(event.get("type") or "")
        if etype.startswith(NEVER_SILENCED_PREFIXES):
            return None
        pattern = self._pattern(event)
        now = event.get("wall_time") or self.clock()
        with self._lock:
            self._prune()
            hits = self._matching(pattern, _hour(now))
        if len(hits) < self.MIN_MARKS:
            return None
        notes = [h["note"] for h in hits if h.get("note")]
        return {"marks": len(hits), "around": f"{int(_hour(now)):02d}:00", "note": notes[-1] if notes else None}

    def patterns(self) -> list[dict]:
        with self._lock:
            self._prune()
            groups: dict[tuple, dict] = {}
            for m in self._marks:
                k = (m["camera"], m["type"], m["zone"], int(m["hour"]))
                g = groups.setdefault(k, {"camera": m.get("camera_name") or m["camera"], "type": m["type"],
                                          "zone": m["zone"], "hour": int(m["hour"]), "marks": 0, "note": None})
                g["marks"] += 1
                g["note"] = m.get("note") or g["note"]
        return sorted(groups.values(), key=lambda g: -g["marks"])

    def clear(self) -> int:
        with self._lock:
            n = len(self._marks)
            self._marks = []
            self._save()
        return n


memory = FalseAlarmMemory()
