"""Group related events into incidents, in plain code.

A person approaching a zone, crossing it and loitering inside produces three
alerts that an operator experiences as one thing. Deciding which events
belong together is done here with fixed rules, so it is predictable and
testable; the agents only write the narrative for an incident this produces.

Rules: events are grouped per camera. A MEDIUM or higher event opens an
incident; later events on that camera at LOW or above join it until the
camera has been quiet for gap_sec. INFO events (routine detections, camera
online) never open or join one. Agent records never re-enter the grouping.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from backend.core.events import CRITICAL, HIGH, INFO, LOW, MEDIUM, severity_for

RANK = {INFO: 0, LOW: 1, MEDIUM: 2, HIGH: 3, CRITICAL: 4}
# Incidents worth an AI brief: the ones that sound the siren or identify a target.
BRIEF_SEVERITIES = (HIGH, CRITICAL)


def _scalar_data(data: dict | None) -> dict:
    """Keep only plain values from an event's data: plate text, scores,
    zone ids. Arrays of boxes and embeddings mean nothing to a reader."""
    return {k: v for k, v in (data or {}).items() if isinstance(v, (str, int, float, bool))}


def event_view(event: dict) -> dict:
    """The compact form of an event that agents read."""
    wall = event.get("wall_time")
    view = {
        "time": datetime.fromtimestamp(wall).strftime("%H:%M:%S") if wall else None,
        "camera": event.get("camera_name") or event.get("camera_id"),
        "camera_id": event.get("camera_id"),
        "type": event.get("type"),
        "severity": event.get("severity") or severity_for(event),
        "reason": event.get("reason"),
        "track_id": event.get("track_id"),
        "zone": event.get("zone_label") or event.get("zone"),
        "has_evidence": bool(event.get("evidence_bundle")),
    }
    data = _scalar_data(event.get("data"))
    if data:
        view["data"] = data
    return {k: v for k, v in view.items() if v not in (None, "", {})}


@dataclass
class Incident:
    incident_id: str
    camera_id: str
    camera_name: str
    opened_at: float
    last_at: float
    events: list[dict] = field(default_factory=list)

    @property
    def max_severity(self) -> str:
        return max((e.get("severity") or severity_for(e) for e in self.events), key=lambda s: RANK.get(s, 0), default=INFO)

    @property
    def needs_brief(self) -> bool:
        return self.max_severity in BRIEF_SEVERITIES

    @property
    def track_ids(self) -> list[int]:
        return sorted({e["track_id"] for e in self.events if e.get("track_id") is not None})

    def for_agent(self) -> dict:
        return {
            "incident_id": self.incident_id,
            "camera": self.camera_name or self.camera_id,
            "camera_id": self.camera_id,
            "started": datetime.fromtimestamp(self.opened_at).strftime("%H:%M:%S"),
            "highest_severity": self.max_severity,
            "track_ids": self.track_ids,
            "events": [event_view(e) for e in self.events],
        }


class IncidentGrouper:
    def __init__(self, gap_sec: float = 60.0):
        self.gap_sec = gap_sec
        self._open: dict[str, Incident] = {}

    def add(self, event: dict) -> Incident | None:
        """Place one event; return the incident it joined or opened, if any."""
        etype = event.get("type", "")
        if etype.startswith("AGENT_"):
            return None
        rank = RANK.get(event.get("severity") or severity_for(event), 0)
        if rank < RANK[LOW]:
            return None
        camera_id = event.get("camera_id") or "unknown"
        at = event.get("wall_time") or time.time()

        incident = self._open.get(camera_id)
        if incident is not None and at - incident.last_at > self.gap_sec:
            incident = None
        if incident is None:
            if rank < RANK[MEDIUM]:
                return None  # routine on its own; it may still join a real incident
            incident = Incident(
                incident_id=uuid.uuid4().hex[:10], camera_id=camera_id,
                camera_name=event.get("camera_name") or camera_id, opened_at=at, last_at=at,
            )
            self._open[camera_id] = incident
        incident.events.append(event)
        incident.last_at = max(incident.last_at, at)
        return incident
