"""The parts of a brief that rules write, and the whole brief when no AI can.

The Situation and Evidence lines of every brief come from here, read
straight off the incident's events, because a model on a laptop CPU can
misstate facts (calling a person "a vehicle") and a brief with wrong facts is
worse than none. An AI model adds only judgment on top: links to other
cameras and the next step. When no model answers, the standard next step for
that kind of event completes the brief, so the operator never sees a blank.
"""
from __future__ import annotations

from backend.agent.incidents import RANK

LABELS = {
    "ZONE_ENTRY": "Restricted zone entered",
    "ZONE_APPROACH": "Person heading for the restricted zone",
    "ZONE_EXIT": "Restricted zone left",
    "LONG_DWELL": "Long stay in the zone",
    "LOITERING": "Loitering",
    "TARGET_CONFIRMED": "Watched person identified",
    "TARGET_REACQUIRED": "Watched person seen again",
    "TARGET_LOST": "Watched person lost from view",
    "TARGET_CROSS_CAMERA_MATCH": "Watched person matched on another camera",
    "TARGET_ZONE_INTRUSION": "Watched person inside the restricted zone",
    "TARGET_BEHAVIOR_ALERT": "Suspicious behaviour",
    "TARGET_VEHICLE_FOUND": "Watched vehicle found",
    "TARGET_VEHICLE_REACQUIRED": "Watched vehicle seen again",
    "PLATE_CONFIRMED": "Number plate read",
    "CAMERA_OFFLINE": "Camera went offline",
    "CAMERA_DEGRADED": "Camera feed degraded",
}

# Standard next step by the most serious kind of event, checked in order.
NEXT_STEPS = (
    (("TARGET_ZONE_INTRUSION", "ZONE_ENTRY"),
     "Check the live view of {camera} and send the nearest patrol if the person is still inside the zone."),
    (("TARGET_CONFIRMED", "TARGET_REACQUIRED", "TARGET_CROSS_CAMERA_MATCH"),
     "Verify the match on the evidence clip, then alert the patrol with where the person was last seen."),
    (("TARGET_VEHICLE_FOUND", "TARGET_VEHICLE_REACQUIRED"),
     "Verify the plate on the evidence clip and stop the vehicle at the next checkpoint."),
    (("TARGET_BEHAVIOR_ALERT",),
     "Keep the subject on the live view of {camera} and send a patrol to check on them."),
    (("CAMERA_OFFLINE",),
     "Check the camera's power and cable, and treat activity near {camera} as suspicious until it is back."),
)
DEFAULT_NEXT_STEP = "Review the evidence clip and decide whether to send a patrol."


def _label(event_type: str | None) -> str:
    return LABELS.get(event_type or "", (event_type or "Event").replace("_", " ").capitalize())


def template_brief(incident: dict, other_activity: list[dict] | None = None, history: list | None = None) -> str:
    """Brief from an incident in the form Incident.for_agent() returns."""
    situation, evidence = template_facts(incident)
    lines = [f"Situation: {situation}", f"Evidence: {evidence}",
             f"Other cameras: {template_other_cameras(other_activity or [])}"]
    if history:
        lines.append(f"History: {template_history(history)}")
    lines.append(f"Next step: {template_next_step(incident)}")
    return "\n".join(lines)


def template_history(patterns: list) -> str:
    """Repeats from earlier days that touch this incident (Pattern objects)."""
    return " ".join(p.summary for p in patterns[:2])


def template_other_cameras(activity: list[dict]) -> str:
    """What the other cameras saw recently, as a fact rather than a guess.

    A small model told "nothing elsewhere" when a watched person had been
    confirmed on another camera minutes earlier. Listing the activity itself
    means the operator always sees it and can judge the link.
    """
    if not activity:
        return "Nothing notable in the last 15 minutes."
    by_camera: dict[str, list[dict]] = {}
    for e in activity:
        by_camera.setdefault(e.get("camera") or e.get("camera_id") or "?", []).append(e)
    parts = []
    for camera, events in list(by_camera.items())[:3]:
        shown = ", ".join(
            f"{e.get('time') or '—'} {_label(e.get('type'))}" + (f" (track #{e['track_id']})" if e.get("track_id") is not None else "")
            for e in events[:3]
        )
        more = f" and {len(events) - 3} more" if len(events) > 3 else ""
        parts.append(f"{camera}: {shown}{more}")
    return "; ".join(parts) + "."


def template_next_step(incident: dict) -> str:
    """The standard next step for the most serious kind of event present."""
    camera = incident.get("camera") or "the camera"
    types = {e.get("type") for e in incident.get("events") or []}
    step = next((text for kinds, text in NEXT_STEPS if types & set(kinds)), DEFAULT_NEXT_STEP)
    return step.format(camera=camera)


def template_facts(incident: dict) -> tuple[str, str]:
    """The Situation and Evidence lines, read straight off the events.

    These are what a brief must never get wrong, so even when an AI model
    writes the rest of the brief, these two lines come from here.
    """
    events = incident.get("events") or []
    camera = incident.get("camera") or "the camera"
    if not events:
        return f"An alarm was raised on {camera}.", "No event details were recorded."

    # The first event at the incident's highest severity is what the alarm was about.
    top_rank = max(RANK.get(e.get("severity"), 0) for e in events)
    top = next(e for e in events if RANK.get(e.get("severity"), 0) == top_rank)
    times = [e["time"] for e in events if e.get("time")]
    span = f" between {times[0]} and {times[-1]}" if len(times) > 1 and times[0] != times[-1] else ""
    count = len(events)
    situation = (
        f"{_label(top.get('type'))} on {camera}"
        + (f" at {top['time']}" if top.get("time") else "")
        + f"; {count} related event{'s' if count != 1 else ''}{span}."
    )

    # One line per kind of event, most serious first, then in the order they
    # first happened. Four people crossing reads as one line, not four.
    groups: dict[str, list[dict]] = {}
    for e in events:
        groups.setdefault(e.get("type") or "", []).append(e)
    ordered = sorted(groups.values(), key=lambda g: (-max(RANK.get(e.get("severity"), 0) for e in g), events.index(g[0])))
    # The four most serious kinds, then shown in the order they happened.
    lines = []
    for group in sorted(ordered[:4], key=lambda g: events.index(g[0])):
        first = group[0]
        line = f"{first.get('time') or '—'} {_label(first.get('type'))}"
        if len(group) == 1:
            line += f" ({first['reason']})" if first.get("reason") else ""
        else:
            tracks = sorted({e["track_id"] for e in group if e.get("track_id") is not None})
            shown = ", ".join(f"#{t}" for t in tracks[:5]) + (", …" if len(tracks) > 5 else "")
            line += f" ×{len(group)}" + (f" (tracks {shown})" if tracks else "")
        lines.append(line)
    return situation, "; ".join(lines) + "."
