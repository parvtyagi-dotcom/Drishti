"""The combined pipeline's bookkeeping: evidence crops, TargetLock event
draining and the per-module summaries returned at the end of a run."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2

# Plate width below which OCR cannot reliably resolve characters. Measured on
# real footage: a 42px plate read as "11" where the true plate had 7
# characters. Industry guidance for ANPR sits around 100-150px of plate width.
MIN_LEGIBLE_PLATE_WIDTH_PX = 100


def save_evidence(evidence_dir: Path | None, frame_image, track_id, event_type: str,
                  frame_index: int, bbox=None) -> str | None:
    """Write the event's crop (or the whole frame) and return its path."""
    if evidence_dir is None or track_id is None:
        return None
    if bbox is not None:
        x1, y1, x2, y2 = (max(0, v) for v in bbox)
        crop = frame_image[y1:y2, x1:x2]
    else:
        crop = frame_image
    if crop is None or crop.size == 0:
        return None
    path = evidence_dir / f"{track_id}_{event_type}_{frame_index}.jpg"
    return str(path) if cv2.imwrite(str(path), crop) else None


def drain_lock_events(lock, start: int, all_events: list[dict], next_event_id: Callable[[], str]) -> int:
    """Copy the TargetLock events after `start` into all_events and return
    the new count drained. None for the lock means nothing to drain."""
    if lock is None:
        return start
    for e in lock.events[start:]:
        all_events.append({"event_id": next_event_id(), "type": e.type, "track_id": e.track_id,
                           "frame_index": e.frame_index, "timestamp_sec": e.timestamp_sec, "data": e.data})
    return len(lock.events)


def legibility_warning(plate_widths: list[int]) -> tuple[int | None, str | None]:
    """(median plate width, warning). Plate legibility is governed by how many
    pixels the plate itself covers, and a camera framed too wide produces
    plates that are found but unreadable at any processing resolution.
    Without this the run just reports "no plates confirmed", which looks like
    a broken reader rather than a camera that needs moving or zooming."""
    if not plate_widths:
        return None, None
    ordered = sorted(plate_widths)
    median = ordered[len(ordered) // 2]
    if median >= MIN_LEGIBLE_PLATE_WIDTH_PX:
        return median, None
    return median, (
        f"plates are only ~{median}px wide; "
        f"~{MIN_LEGIBLE_PLATE_WIDTH_PX}px is needed to read characters reliably. "
        "Move the camera closer, zoom in, or raise the analysis resolution."
    )


def lock_summary(lock) -> dict:
    """The person-ID target's lock state, as the run summary reports it."""
    return {
        "confirmed": lock.confirmed, "target_track_id": lock.target_track_id,
        **_sighting_times(lock),
    }


def _sighting_times(lock) -> dict:
    return {
        "first_seen_sec": lock.first_seen_sec, "last_seen_sec": lock.last_seen_sec,
        "visible_duration_sec": lock.visible_duration_sec, "visible_frame_count": lock.visible_frame_count,
    }


def anpr_target_summary(anpr_lock, aggregator, normalized_target) -> dict | None:
    if anpr_lock is None:
        return None
    match_conf = None
    if anpr_lock.target_track_id is not None:
        agg = aggregator.aggregate(anpr_lock.target_track_id)
        match_conf = agg.mean_confidence if agg else None
    return {
        "found": anpr_lock.confirmed, "track_id": anpr_lock.target_track_id,
        "plate_text": normalized_target if anpr_lock.confirmed else None, "match_confidence": match_conf,
        **_sighting_times(anpr_lock),
    }


def detected_plates(aggregator) -> list[dict]:
    return [
        {"track_id": tid, "text": agg.text, "mean_confidence": agg.mean_confidence,
         "observation_count": agg.observation_count, "confirmed": agg.confirmed}
        for tid in aggregator.all_track_ids() if (agg := aggregator.aggregate(tid)) is not None
    ]


def zones_summary(zones, zone_monitors) -> list[dict]:
    summary = []
    for zone, monitor in zip(zones, zone_monitors):
        def count(kind: str) -> int:
            return sum(1 for e in monitor.events if e.type == kind)
        summary.append({"zone_id": zone.zone_id, "label": zone.label, "entries": count("ZONE_ENTRY"),
                        "exits": count("ZONE_EXIT"), "dwell_events": count("LONG_DWELL")})
    return summary


def behavior_summary(behavior_monitor, vehicle_behavior_monitor, tracks_observed: int) -> dict:
    counts: dict[str, int] = {}
    events = behavior_monitor.events + (vehicle_behavior_monitor.events if vehicle_behavior_monitor else [])
    for e in events:
        counts[e.type] = counts.get(e.type, 0) + 1
    return {"event_counts": counts, "tracks_observed": tracks_observed,
            "units": "m" if vehicle_behavior_monitor else "px"}


def lowlight_summary(backend: str, frame_count: int | None, low_light_frames: int, enhanced_frames: int) -> dict:
    total = frame_count or 1
    return {
        "low_light_detected": low_light_frames > 0, "enhancement_applied": enhanced_frames > 0,
        "backend": backend, "low_light_frames": low_light_frames,
        "enhanced_frames": enhanced_frames, "enhanced_percentage": round(100.0 * enhanced_frames / total, 2),
    }
