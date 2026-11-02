"""Group gathering: enough people, for long enough, alerts once per group."""
from __future__ import annotations

from backend.core.events import describe_event, severity_for
from backend.live.crowd import GroupConfig, GroupWatch


def run(watch, now, counts, step=0.5):
    alerts = []
    for n in counts:
        alert = watch.observe(n)
        if alert:
            alerts.append(alert)
        now[0] += step
    return alerts


def make():
    now = [0.0]
    return GroupWatch(GroupConfig(min_people=4, hold_sec=10, rearm_sec=30), clock=lambda: now[0]), now


def test_group_held_for_ten_seconds_alerts_once():
    watch, now = make()
    alerts = run(watch, now, [5] * 40)
    assert len(alerts) == 1
    assert alerts[0]["people"] == 5 and alerts[0]["duration_sec"] >= 9


def test_brief_crowd_or_small_group_does_not_alert():
    watch, now = make()
    assert run(watch, now, [5] * 10 + [1] * 30) == []
    assert run(watch, now, [3] * 40) == []


def test_one_missed_detection_does_not_reset():
    watch, now = make()
    counts = ([5] * 9 + [2]) * 3
    assert len(run(watch, now, counts)) == 1


def test_rearms_only_after_group_breaks_up():
    watch, now = make()
    run(watch, now, [5] * 30)
    assert run(watch, now, [0] * 20 + [5] * 30) == []   # 10 s apart is not enough
    assert len(run(watch, now, [0] * 70 + [6] * 30)) == 1


def test_event_text_and_severity():
    event = {"type": "GROUP_GATHERING", "data": {"people": 6, "duration_sec": 12.0}}
    assert severity_for(event) == "MEDIUM"
    assert describe_event(event) == "Group of about 6 people stayed together in view for 12 seconds"
