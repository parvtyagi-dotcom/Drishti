"""Pattern: repeats across days, from the event log."""
from __future__ import annotations

from datetime import datetime

from backend.agent.patterns import find_patterns, patterns_for_incident, recent_patterns
from backend.agent.template import template_brief


def at(day: int, hour: int, minute: int = 0) -> float:
    return datetime(2026, 10, day, hour, minute).timestamp()


def plate(ts, text="TEST123", camera="Gate 1"):
    return {"type": "PLATE_CONFIRMED", "wall_time": ts, "camera_name": camera, "plate": text, "data": {"text": text}}


def zone(ts, camera="CAMERA 01"):
    return {"type": "ZONE_ENTRY", "wall_time": ts, "camera_name": camera}


def test_a_plate_seen_on_several_days_is_a_pattern():
    events = [plate(at(14, 22, 10)), plate(at(15, 22, 40)), plate(at(15, 22, 45)), plate(at(17, 23, 5), camera="Gate 2"),
              plate(at(17, 9), text="ONCE99")]
    [p] = [p for p in find_patterns(events) if p.kind == "plate"]
    assert p.key == "TEST123" and (p.sightings, p.days) == (4, 3)
    assert p.summary == "Plate TEST123 read 4 times on 3 days, between 22:00 and 24:00, at Gate 1, Gate 2."


def test_alarms_at_the_same_hour_on_several_days_are_a_pattern():
    events = [zone(at(14, 2, 5)), zone(at(14, 2, 30)), zone(at(16, 2, 50)), zone(at(16, 14))]
    [p] = find_patterns(events)
    assert p.kind == "hour" and p.key == "CAMERA 01 @ 02:00" and p.days == 2 and p.sightings == 3


def test_one_day_of_activity_is_not_a_pattern():
    assert find_patterns([plate(at(14, 22)), plate(at(14, 23)), zone(at(14, 2)), zone(at(14, 2, 30))]) == []


def test_most_persistent_patterns_come_first():
    events = [plate(at(d, 22)) for d in (14, 15)] + [zone(at(d, 2)) for d in (14, 15, 16)]
    assert [p.kind for p in find_patterns(events)] == ["hour", "plate"]


def test_patterns_are_matched_to_an_incident_by_camera_or_plate():
    found = find_patterns([plate(at(d, 22)) for d in (14, 15)] + [zone(at(d, 2), camera="Gate 9") for d in (14, 15)])
    incident = {"camera": "CAMERA 01", "events": [{"type": "PLATE_CONFIRMED", "data": {"text": "TEST123"}}]}
    assert [p.kind for p in patterns_for_incident(found, incident)] == ["plate"]
    assert patterns_for_incident(found, {"camera": "Gate 9", "events": []})[0].kind == "hour"


def test_store_is_read_in_two_queries_so_plates_are_never_crowded_out():
    class Store:
        def __init__(self):
            self.calls = []

        def query(self, limit, since, types):
            self.calls.append(types)
            if types == ["PLATE_CONFIRMED"]:
                return [plate(at(14, 22)), plate(at(15, 22))]
            return [zone(at(14, 2))] * 1000

    store = Store()
    patterns = recent_patterns(store=store, now=at(18, 12))
    assert len(store.calls) == 2 and any(p.kind == "plate" for p in patterns)


def test_history_line_appears_in_rules_brief_only_when_relevant():
    incident = {"camera": "CAMERA 01", "events": [{"type": "ZONE_ENTRY", "severity": "CRITICAL", "time": "02:10:00"}]}
    history = find_patterns([zone(at(d, 2)) for d in (14, 15)])
    with_history = template_brief(incident, [], history).split("\n")
    assert with_history[3].startswith("History: Zone alarms on CAMERA 01 around 02:00 on 2 different days")
    assert "History:" not in template_brief(incident, [], [])
