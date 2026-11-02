"""Where the intruder is heading, and where to send the patrol.

Built from what is already known: where each camera sits on the Site Map,
and the order in which cameras raised alerts. Two or more cameras in
sequence give a direction and a speed across the map. The next camera
roughly ahead on that line is the likely next sighting, and the time to get
there at the same speed is the estimate. If no camera lies ahead, the path
leads into a blind spot, which is said plainly with the point to cover.

It is a straight-line guess from the last two sightings, labelled as such:
it tells the patrol where to look first, and every number it gives is shown
with what it was worked out from.
"""
from __future__ import annotations

import math

TRAIL_TYPES = frozenset({
    "TARGET_CONFIRMED", "TARGET_REACQUIRED", "TARGET_CROSS_CAMERA_MATCH", "TARGET_ZONE_INTRUSION",
    "TARGET_BEHAVIOR_ALERT", "TARGET_VEHICLE_FOUND", "TARGET_VEHICLE_REACQUIRED",
    "ZONE_APPROACH", "ZONE_ENTRY", "LOITERING", "GROUP_GATHERING", "SENSOR_ALARM_CONFIRMED",
})
WINDOW_SEC = 30 * 60
# A camera counts as "ahead" within this angle of the direction of travel.
MAX_ANGLE_DEG = 60.0
# How far ahead to point when the path leads where no camera can see.
BLIND_LOOKAHEAD_SEC = 120.0


def trail(events: list[dict], positions: dict, now: float) -> list[dict]:
    """Alert stops in time order; repeats on one camera merge into one stop."""
    stops: list[dict] = []
    for e in sorted(events, key=lambda e: e.get("wall_time") or 0):
        if e.get("type") not in TRAIL_TYPES or now - (e.get("wall_time") or 0) > WINDOW_SEC:
            continue
        name = e.get("camera_name")
        if name not in positions:
            continue
        if stops and stops[-1]["camera"] == name:
            stops[-1]["last"] = e["wall_time"]
            continue
        stops.append({"camera": name, "time": e["wall_time"], "last": e["wall_time"], "type": e.get("type")})
    return stops


def predict(events: list[dict], positions: dict, now: float) -> dict | None:
    stops = trail(events, positions, now)
    if len(stops) < 2:
        return None
    a, b = stops[-2], stops[-1]
    (ax, ay), (bx, by) = positions[a["camera"]], positions[b["camera"]]
    dx, dy = bx - ax, by - ay
    step = math.hypot(dx, dy)
    elapsed = max(b["time"] - a["last"], 1.0)
    if step < 1e-6:
        return None
    speed = step / elapsed                      # map widths per second
    heading = math.degrees(math.atan2(dy, dx))

    best = None
    for name, (cx, cy) in positions.items():
        if name in (a["camera"], b["camera"]):
            continue
        vx, vy = cx - bx, cy - by
        dist = math.hypot(vx, vy)
        if dist < 1e-6:
            continue
        cos = (vx * dx + vy * dy) / (dist * step)
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
        if angle > MAX_ANGLE_DEG:
            continue
        score = dist * (1 + angle / MAX_ANGLE_DEG)
        if best is None or score < best[0]:
            best = (score, name, dist, angle)

    since_last = max(0.0, now - b["last"])
    result = {
        "from": [a["camera"], b["camera"]],
        "trail": [s["camera"] for s in stops],
        "heading_deg": round(heading),
        "last_seen": b["last"],
        "basis": f"straight line from {a['camera']} to {b['camera']} "
                 f"({round(elapsed)} s apart), assuming the same speed",
    }
    if best is None:
        px = min(1.0, max(0.0, bx + dx / elapsed * BLIND_LOOKAHEAD_SEC))
        py = min(1.0, max(0.0, by + dy / elapsed * BLIND_LOOKAHEAD_SEC))
        result.update(next_camera=None, point=[round(px, 3), round(py, 3)],
                      advice=f"No camera ahead: the path leads into a blind spot past {b['camera']}. "
                             f"Send the patrol toward the marked point.")
        return result
    _, name, dist, angle = best
    eta = max(0.0, dist / speed - since_last)
    result.update(next_camera=name, point=list(positions[name]), eta_sec=round(eta),
                  angle_off_deg=round(angle),
                  advice=f"Likely next at {name} in about {_human(eta)}. Send the patrol to {name} now.")
    return result


def _human(sec: float) -> str:
    if sec < 60:
        return f"{max(5, round(sec / 5) * 5)} seconds"
    minutes = round(sec / 60)
    return f"{minutes} minute{'s' if minutes != 1 else ''}"
