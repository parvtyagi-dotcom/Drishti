"""Pattern: what keeps happening, across days.

A single sighting is an event; the same plate at the gate four nights out of
seven, or someone entering the same zone at the same hour on three days, is
reconnaissance. No camera notices that, because each alert stands alone.
This reads the event log back over days and reports repeats, in plain code
so the counts are exact.

  plate   the same number plate read on two or more days
  hour    alarms on one camera at the same hour of day on two or more days
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime

PLATE_TYPES = ("PLATE_CONFIRMED",)
ACTIVITY_TYPES = ("ZONE_ENTRY", "TARGET_ZONE_INTRUSION", "LOITERING", "LONG_DWELL", "TARGET_BEHAVIOR_ALERT")


@dataclass
class Pattern:
    kind: str           # plate | hour
    key: str            # the plate, or "camera @ hour"
    summary: str        # one sentence for an operator
    sightings: int
    days: int
    cameras: list[str] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _day(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def _hour(ts: float) -> int:
    return datetime.fromtimestamp(ts).hour


def _hours_text(hours: set[int]) -> str:
    lo, hi = min(hours), max(hours)
    return f"around {lo:02d}:00" if lo == hi else f"between {lo:02d}:00 and {hi + 1:02d}:00"


def _plate(event: dict) -> str | None:
    text = event.get("plate") or (event.get("data") or {}).get("text")
    return str(text).strip().upper() if text else None


def find_patterns(events: list[dict], min_days: int = 2) -> list[Pattern]:
    """Repeats in these events (any order), most frequent first."""
    patterns: list[Pattern] = []

    plates: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        if e.get("type") in PLATE_TYPES and (plate := _plate(e)):
            plates[plate].append(e)
    for plate, seen in plates.items():
        days = {_day(e["wall_time"]) for e in seen}
        if len(days) < min_days:
            continue
        cameras = sorted({e.get("camera_name") or e.get("camera_id") or "?" for e in seen})
        hours = {_hour(e["wall_time"]) for e in seen}
        patterns.append(Pattern(
            kind="plate", key=plate,
            summary=f"Plate {plate} read {len(seen)} times on {len(days)} days, {_hours_text(hours)}, at {', '.join(cameras)}.",
            sightings=len(seen), days=len(days), cameras=cameras,
            first_seen=min(e["wall_time"] for e in seen), last_seen=max(e["wall_time"] for e in seen),
        ))

    # Alarms on one camera bucketed by hour of day: the same hour on several
    # days is a routine someone is keeping, whatever the individual tracks.
    slots: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for e in events:
        if e.get("type") in ACTIVITY_TYPES and e.get("wall_time"):
            slots[(e.get("camera_name") or e.get("camera_id") or "?", _hour(e["wall_time"]))].append(e)
    for (camera, hour), seen in slots.items():
        days = {_day(e["wall_time"]) for e in seen}
        if len(days) < min_days:
            continue
        patterns.append(Pattern(
            kind="hour", key=f"{camera} @ {hour:02d}:00",
            summary=f"Zone alarms on {camera} {_hours_text({hour})} on {len(days)} different days ({len(seen)} alarms).",
            sightings=len(seen), days=len(days), cameras=[camera],
            first_seen=min(e["wall_time"] for e in seen), last_seen=max(e["wall_time"] for e in seen),
        ))

    return sorted(patterns, key=lambda p: (-p.days, -p.sightings))


def recent_patterns(store=None, manager=None, days_back: int = 7, now: float | None = None) -> list[Pattern]:
    """Patterns over the last days_back days of the event log."""
    now = time.time() if now is None else now
    since = now - days_back * 86400
    if store is not None:
        # Two queries, so a busy week of zone alarms cannot crowd the far
        # rarer plate reads out of the store's per-query limit.
        events = (store.query(limit=1000, since=since, types=list(PLATE_TYPES))
                  + store.query(limit=1000, since=since, types=list(ACTIVITY_TYPES)))
    elif manager is not None:
        wanted = set(PLATE_TYPES + ACTIVITY_TYPES)
        events = [e for e in manager.recent_events(limit=1000) if e.get("type") in wanted and (e.get("wall_time") or 0) >= since]
    else:
        events = []
    return find_patterns(events)


def patterns_for_incident(patterns: list[Pattern], incident: dict) -> list[Pattern]:
    """The patterns that touch this incident: its camera, or a plate it read."""
    camera = incident.get("camera")
    plates = {str((e.get("data") or {}).get("text") or "").upper() for e in incident.get("events") or []}
    return [p for p in patterns if (p.kind == "plate" and p.key in plates) or camera in p.cameras]
