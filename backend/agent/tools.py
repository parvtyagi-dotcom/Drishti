"""What the agents can look at, and the one thing they can ask for.

Every tool here only reads. The single exception, propose_action, records a
request that does nothing until an operator approves it. Tool results carry
camera and OCR text (plate readings, zone names) that the agents are told to
treat as data, never as instructions.

Tool inputs are checked here too: Claude's strict mode guarantees the shape,
but a local model does not, and strict mode cannot express number limits.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from backend.agent.incidents import event_view
from backend.agent.models import ToolSpec

TARGET_EVENT_TYPES = (
    "TARGET_CONFIRMED", "TARGET_REACQUIRED", "TARGET_LOST", "TARGET_CROSS_CAMERA_MATCH",
    "TARGET_ZONE_INTRUSION", "TARGET_BEHAVIOR_ALERT",
    "TARGET_VEHICLE_FOUND", "TARGET_VEHICLE_REACQUIRED", "TARGET_VEHICLE_LOST",
)
# What an agent may propose. Nothing here runs without an operator's approval.
ACTIONS = (
    "enable_person_id",     # look for this person on another camera
    "enable_anpr",          # read plates on another camera
    "review_footage",       # a person should look at a clip
    "dispatch_patrol",      # send someone to a location
    "raise_alert_level",    # tell the post to heighten readiness
)


def _nullable(schema: dict) -> dict:
    return {"anyOf": [schema, {"type": "null"}]}


def _int(value, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


class EventSource:
    """Reads events from the durable store when there is one, otherwise from
    the cameras' in-memory recent events (as in tests and short demos)."""

    def __init__(self, store=None, manager=None, clock=time.time):
        self.store = store
        self.manager = manager
        self.clock = clock

    def recent(self, minutes_back: int, camera_id: str | None, limit: int) -> list[dict]:
        since = self.clock() - minutes_back * 60
        if self.store is not None:
            events = self.store.query(limit=1000, camera_id=camera_id, since=since)
        elif self.manager is not None:
            events = [e for e in self.manager.recent_events(limit=1000, camera_id=camera_id)
                      if (e.get("wall_time") or 0) >= since]
        else:
            events = []
        events = [e for e in events if not str(e.get("type", "")).startswith("AGENT_")]
        events.sort(key=lambda e: e.get("wall_time") or 0, reverse=True)
        return events[:limit]


def other_camera_activity(source: EventSource, incident: dict, minutes_back: int = 15, limit: int = 12) -> list[dict]:
    """Notable recent events on every camera except the incident's own.

    Gathered in code and handed to the Supervisor up front: a small local
    model rarely thinks to search for a sighting on another camera, yet that
    link is often the most useful thing a brief can add. Sightings of watched
    people and vehicles, camera outages, and anything MEDIUM or above count.
    """
    from backend.agent.incidents import RANK

    here = incident.get("camera_id")
    notable = [
        e for e in source.recent(minutes_back, None, 1000)
        if e.get("camera_id") != here and (
            e.get("type") in TARGET_EVENT_TYPES or e.get("type") == "CAMERA_OFFLINE"
            or RANK.get(e.get("severity"), 0) >= RANK["MEDIUM"])
    ]
    # Oldest first, so it reads as a timeline leading up to the incident.
    return [event_view(e) for e in reversed(notable[:limit])]


def search_events_tool(source: EventSource) -> ToolSpec:
    def handler(args: dict) -> dict:
        minutes = _int(args.get("minutes_back"), 30, 1, 24 * 60)
        limit = _int(args.get("limit"), 20, 1, 25)
        camera = str(args.get("camera_id") or "").strip().lower()
        types = args.get("event_types") or None
        events = source.recent(minutes, None, 1000)
        if camera:
            # Models (and operators) name cameras the way the dashboard shows
            # them, "CAMERA 01", as often as by id, "cam-01"; accept either.
            events = [e for e in events if camera in (str(e.get("camera_id") or "").lower(),
                                                       str(e.get("camera_name") or "").lower())]
        if types:
            wanted = {str(t).upper() for t in types}
            events = [e for e in events if e.get("type") in wanted]
        return {"count": min(len(events), limit), "events": [event_view(e) for e in events[:limit]]}

    return ToolSpec(
        name="search_events",
        description=(
            "Search recent Drishti events, newest first. Filter by camera and by event type "
            "(for example ZONE_ENTRY, TARGET_CONFIRMED, PLATE_CONFIRMED, CAMERA_OFFLINE). "
            "Use it to check what else happened around an incident."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "camera_id": _nullable({"type": "string", "description": "Only this camera, by id or by name; null for all cameras."}),
                "event_types": _nullable({"type": "array", "items": {"type": "string"}, "description": "Only these event types; null for all."}),
                "minutes_back": {"type": "integer", "description": "How far back to look, 1-1440 minutes."},
                "limit": {"type": "integer", "description": "Most events to return, 1-25."},
            },
            "required": ["camera_id", "event_types", "minutes_back", "limit"],
            "additionalProperties": False,
        },
        handler=handler,
    )


def target_sightings_tool(source: EventSource) -> ToolSpec:
    def handler(args: dict) -> dict:
        minutes = _int(args.get("minutes_back"), 60, 1, 24 * 60)
        events = [e for e in source.recent(minutes, None, 1000) if e.get("type") in TARGET_EVENT_TYPES]
        return {"count": len(events[:60]), "sightings": [event_view(e) for e in events[:60]]}

    return ToolSpec(
        name="find_target_sightings",
        description=(
            "List every sighting of watched people and vehicles across all cameras: confirmed, "
            "reacquired, lost, matched across cameras, or inside a restricted zone. Newest first."
        ),
        input_schema={
            "type": "object",
            "properties": {"minutes_back": {"type": "integer", "description": "How far back to look, 1-1440 minutes."}},
            "required": ["minutes_back"],
            "additionalProperties": False,
        },
        handler=handler,
    )


def patterns_tool(source: EventSource) -> ToolSpec:
    def handler(args: dict) -> dict:
        from backend.agent.patterns import recent_patterns

        days = _int(args.get("days_back"), 7, 1, 30)
        found = recent_patterns(source.store, source.manager, days_back=days, now=source.clock())
        return {"patterns": [p.summary for p in found[:10]]}

    return ToolSpec(
        name="find_patterns",
        description="Repeats across days: the same plate read on several days, or alarms on one camera at the same hour on several days.",
        input_schema={
            "type": "object",
            "properties": {"days_back": {"type": "integer", "description": "How many days to look back, 1-30."}},
            "required": ["days_back"],
            "additionalProperties": False,
        },
        handler=handler,
    )


def list_cameras_tool(manager) -> ToolSpec:
    def handler(args: dict) -> dict:
        cameras = []
        for session in manager.list() if manager is not None else []:
            status = session.status_dict()
            cameras.append({
                "camera_id": status.get("camera_id"), "name": status.get("name"),
                "location": status.get("location"), "state": status.get("state"),
                "analysis": status.get("modules") or [],
            })
        return {"cameras": cameras}

    return ToolSpec(
        name="list_cameras",
        description="List the post's cameras with location, whether each is online, and which analysis is switched on.",
        input_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        handler=handler,
    )


@dataclass
class ProposedAction:
    action: str
    camera_id: str | None
    reason: str


@dataclass
class ProposalCollector:
    """Holds what an agent proposed during one run. Nothing is recorded or
    shown until the run finishes, so a run retried on another model never
    leaves duplicate proposals behind.

    Capped in code, not just in the prompt: a small local model asked for at
    most two proposals made four. An operator facing an alarm needs one or
    two clear choices, not a list."""
    items: list[ProposedAction] = field(default_factory=list)
    limit: int = 2


def propose_action_tool(collector: ProposalCollector) -> ToolSpec:
    def handler(args: dict) -> str:
        action = args.get("action")
        if action not in ACTIONS:
            raise ValueError(f"action must be one of {', '.join(ACTIONS)}")
        reason = str(args.get("reason") or "").strip()[:300]
        if not reason:
            raise ValueError("give a reason the operator can check")
        camera_id = str(args.get("camera_id") or "").strip()
        # Models sometimes spell "no camera" as text instead of null.
        camera_id = None if camera_id.lower() in ("", "null", "none", "-") else camera_id
        if any(p.action == action and p.camera_id == camera_id for p in collector.items):
            return "Already proposed; not added again."
        if len(collector.items) >= collector.limit:
            raise ValueError(f"at most {collector.limit} proposals per answer; this one was not recorded")
        collector.items.append(ProposedAction(action=action, camera_id=camera_id, reason=reason))
        return "Recorded as a proposal. It does nothing until an operator approves it."

    return ToolSpec(
        name="propose_action",
        description=(
            "Propose one action for the operator to approve or reject. Nothing happens without "
            "their approval. Propose only what the evidence supports, one call per action."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": list(ACTIONS)},
                "camera_id": _nullable({"type": "string", "description": "Camera the action concerns, or null."}),
                "reason": {"type": "string", "description": "One sentence the operator can verify."},
            },
            "required": ["action", "camera_id", "reason"],
            "additionalProperties": False,
        },
        handler=handler,
    )
