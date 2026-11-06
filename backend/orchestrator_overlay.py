"""What the combined pipeline draws on each frame: zones, person-ID face and
candidate boxes, plate boxes, alert boxes and banners.

Pure drawing on the frame image - no analysis state is changed here except
the per-track face sightings that have gone stale, which
draw_person_overlays drops as it goes.
"""
from __future__ import annotations

from backend.core.output import (
    CYAN, MAGENTA, ORANGE, RED, YELLOW, draw_banner, draw_polygon, draw_target_box,
    draw_thin_box, format_timestamp,
)
from backend.core.target_lock import TargetLock
from backend.modules.person_id.face_search import (
    _VetoStreaks, _candidate_body_label, _face_box_style,
)
from backend.modules.zone.state import INSIDE

BEHAVIOR_BANNER_COLOR = {
    "LOITERING": ORANGE, "SUDDEN_DIRECTION_CHANGE": ORANGE,
    "ABNORMAL_SPEED": ORANGE, "REPEATED_BACK_AND_FORTH": ORANGE,
    "TARGET_BEHAVIOR_ALERT": (0, 0, 220),
}
BEHAVIOR_LABEL = {
    "LOITERING": "LOITERING", "SUDDEN_DIRECTION_CHANGE": "SUDDEN DIRECTION CHANGE",
    "ABNORMAL_SPEED": "ABNORMAL SPEED", "REPEATED_BACK_AND_FORTH": "REPEATED PACING",
}
_ZONE_EXIT_COLOR = (60, 200, 60)
_LONG_DWELL_COLOR = (0, 0, 200)


def class_name(track_id, track_class: dict, vehicle_class_ids) -> str:
    return "VEHICLE" if track_class.get(track_id) in vehicle_class_ids else "PERSON"


def zone_banner(ev, cname: str) -> tuple[list[str], tuple[int, int, int]] | None:
    """Banner lines and colour for a zone event, or None for events that get
    no banner."""
    who = f"{cname} #{ev.track_id}"
    if ev.type == "ZONE_APPROACH":
        eta = ev.data.get("eta_sec")
        return ["APPROACHING ZONE", who, f"Predicted entry in {eta}s" if eta else "Predicted entry"], ORANGE
    if ev.type == "ZONE_ENTRY":
        return ["ZONE INTRUSION", who, format_timestamp(ev.timestamp_sec)], RED
    if ev.type == "ZONE_EXIT":
        return ["ZONE EXIT", who, format_timestamp(ev.timestamp_sec)], _ZONE_EXIT_COLOR
    if ev.type == "LONG_DWELL":
        return ["LONG DWELL", who, f"Duration: {ev.data['duration_sec']:.1f} sec"], _LONG_DWELL_COLOR
    return None


def behavior_banner(ev, cname: str) -> tuple[list[str], tuple[int, int, int]]:
    if ev.type == "TARGET_BEHAVIOR_ALERT":
        lines = ["TARGET BEHAVIOR ALERT", f"{cname} #{ev.track_id}", ev.data["trigger_behavior"]]
    else:
        lines = ["BEHAVIOR ALERT", f"{BEHAVIOR_LABEL[ev.type]} - {cname} #{ev.track_id}",
                 format_timestamp(ev.timestamp_sec)]
    return lines, BEHAVIOR_BANNER_COLOR[ev.type]


def draw_person_overlays(frame, lock: TargetLock, tracks_by_id: dict, face_sightings: dict,
                         veto_streaks: _VetoStreaks, cfg) -> None:
    """What person ID is doing before - and besides - the red TARGET box:
    a yellow CANDIDATE n/N body box on every track gathering face matches
    while nothing is confirmed, and a thin box on each recently found
    face (green MATCH for the target, yellow CANDIDATE, faint grey for
    anyone else). Without these an operator saw nothing at all until
    confirmation and reasonably concluded detection was broken."""
    if not lock.confirmed:
        for tid, count in lock.candidates().items():
            track = tracks_by_id.get(tid)
            if track is not None:
                draw_target_box(frame.image, track.bbox,
                                label=_candidate_body_label(count, cfg.confirmation_observations),
                                color=YELLOW, thickness=2)
    veto_streaks.prune(frame.timestamp_sec)
    for tid in list(face_sightings):
        sighting = face_sightings[tid]
        if frame.timestamp_sec - sighting.timestamp_sec > cfg.face_box_hold_sec:
            # Stale: stop drawing it. Its veto streak has its own,
            # longer window (see _VetoStreaks).
            del face_sightings[tid]
            continue
        track = tracks_by_id.get(tid)
        if track is None:
            continue
        color, label, thickness = _face_box_style(tid, sighting.similarity, lock)
        tx1, ty1 = track.bbox[0], track.bbox[1]
        rx1, ry1, rx2, ry2 = sighting.rel_bbox
        draw_thin_box(frame.image, (tx1 + rx1, ty1 + ry1, tx1 + rx2, ty1 + ry2), color,
                      label=label, thickness=thickness, label_below=True)


def draw_zones(image, zone_polygons_px, zones, zone_monitors, tracks_by_id: dict) -> None:
    for polygon_px, zone, monitor in zip(zone_polygons_px, zones, zone_monitors):
        breached = any(monitor.state_of(tid) == INSIDE for tid in tracks_by_id)
        nearing = not breached and any(monitor.approaching(tid) for tid in tracks_by_id)
        label = zone.label or "RESTRICTED ZONE"
        if breached:
            draw_polygon(image, polygon_px, label=f"{label} - INTRUSION", color=RED)
        elif nearing:
            draw_polygon(image, polygon_px, label=f"{label} - APPROACH", color=ORANGE)
        else:
            draw_polygon(image, polygon_px, label=label)


def draw_auto_plates(image, tracks_by_id: dict, aggregator, drawn: set[int], vehicle_class_ids) -> None:
    """Automatic ANPR (no searched plate): a cyan box with the read on every
    vehicle that has one - full box once confirmed, thin and marked '?' while
    the vote is still open, so ANPR is visibly working long before then."""
    for tid, track in tracks_by_id.items():
        if tid in drawn or track.class_id not in vehicle_class_ids:
            continue
        agg = aggregator.aggregate(tid)
        if agg is None:
            continue
        label, thickness = _plate_box_style(agg)
        if agg.confirmed:
            draw_target_box(image, track.bbox, label=label, color=CYAN, thickness=thickness)
            drawn.add(tid)
        else:
            draw_thin_box(image, track.bbox, CYAN, label=label[0], thickness=thickness)


def draw_alert_boxes(image, frame_index: int, tracks_by_id: dict, drawn: set[int],
                     zone_inside_any: dict, behavior_alert_until: dict,
                     track_class: dict, vehicle_class_ids) -> None:
    """Orange box on whoever is inside a zone, magenta on whoever has a live
    behaviour alert; a track already boxed as a target is left alone."""
    for tid in zone_inside_any:
        if tid in drawn or tid not in tracks_by_id:
            continue
        cname = class_name(tid, track_class, vehicle_class_ids)
        draw_target_box(image, tracks_by_id[tid].bbox, label=f"{cname} #{tid}", color=ORANGE, thickness=2)
        drawn.add(tid)
    for tid, until in behavior_alert_until.items():
        if tid in drawn or frame_index >= until or tid not in tracks_by_id:
            continue
        cname = class_name(tid, track_class, vehicle_class_ids)
        draw_target_box(image, tracks_by_id[tid].bbox, label=f"{cname} #{tid}", color=MAGENTA, thickness=2)
        drawn.add(tid)


def draw_banners(image, active_banners: list, frame_index: int) -> list:
    """Draw the banners still in date and return them; expired ones drop out."""
    current = [b for b in active_banners if b[2] > frame_index]
    for slot, (lines, color, _expires) in enumerate(current):
        draw_banner(image, lines, color, slot=slot)
    return current


def _plate_box_style(agg) -> tuple[list[str], int]:
    """(label lines, thickness): a confirmed plate gets the full box, a
    provisional read a thin one marked '?'."""
    if agg.confirmed:
        return [agg.text, f"{round(agg.mean_confidence)}%"], 3
    return [f"{agg.text} ?"], 1
