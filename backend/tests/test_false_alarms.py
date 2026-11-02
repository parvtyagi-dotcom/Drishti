"""False-alarm memory: learns from guards, never silences the serious ones."""
from datetime import datetime

from backend.live.false_alarms import FalseAlarmMemory


def at(hour, minute=0, day=20):
    return datetime(2026, 10, day, hour, minute).timestamp()


def alarm(hour, minute=0, day=20, etype="ZONE_ENTRY", camera="Gate 1", zone="z1"):
    return {"type": etype, "camera_name": camera, "zone": zone, "wall_time": at(hour, minute, day)}


def make(tmp_path):
    return FalseAlarmMemory(tmp_path / "fa.json", clock=lambda: at(23, day=25))


def test_two_marks_silence_the_same_place_and_hour(tmp_path):
    mem = make(tmp_path)
    mem.mark(alarm(18, 5, day=21), "false", note="tree shadow")
    assert mem.check(alarm(18, 10, day=24)) is None          # one mark is not enough
    mem.mark(alarm(18, 20, day=22), "false")
    hit = mem.check(alarm(18, 40, day=24))
    assert hit["marks"] == 2 and hit["note"] == "tree shadow"


def test_other_hour_camera_zone_or_type_still_sounds(tmp_path):
    mem = make(tmp_path)
    for d in (21, 22):
        mem.mark(alarm(18, day=d), "false")
    assert mem.check(alarm(21, day=24)) is None
    assert mem.check(alarm(18, day=24, camera="Gate 2")) is None
    assert mem.check(alarm(18, day=24, zone="z2")) is None
    assert mem.check(alarm(18, day=24, etype="CAMERA_TAMPERED")) is None


def test_watched_targets_and_sabotage_are_never_silenced(tmp_path):
    mem = make(tmp_path)
    for etype in ("TARGET_ZONE_INTRUSION", "CAMERA_SABOTAGE_SUSPECTED"):
        for d in (21, 22, 23):
            mem.mark(alarm(18, day=d, etype=etype), "false")
        assert mem.check(alarm(18, day=24, etype=etype)) is None


def test_a_real_alarm_wipes_the_lesson(tmp_path):
    mem = make(tmp_path)
    for d in (21, 22):
        mem.mark(alarm(18, day=d), "false")
    result = mem.mark(alarm(18, 15, day=24), "real")
    assert result["forgotten"] == 2
    assert mem.check(alarm(18, day=24)) is None


def test_memory_survives_restart_and_old_marks_expire(tmp_path):
    mem = make(tmp_path)
    for d in (1, 2):           # older than 14 days before the 25th
        mem.mark(alarm(18, day=d), "false")
    for d in (21, 22):
        mem.mark(alarm(6, day=d), "false")
    again = make(tmp_path)
    assert again.check(alarm(18, day=24)) is None
    assert again.check(alarm(6, day=24)) is not None
    assert again.patterns()[0]["marks"] == 2
