"""Forensic events carry the same severity the live path stamps.

Live events are stamped with severity as they are emitted; forensic ones were
not, so the Results page kept its own hardcoded list of which types deserved
the siren. Two opinions about what counts as an alarm drift, and did: raising
a zone crossing to CRITICAL made a live camera sound while the identical
crossing stayed silent in Forensic Mode, because that list predated the
change. These pin both modes to one source of truth.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.core.events import ALARM_SEVERITIES, severity_for, severity_of

client = TestClient(app)


def test_a_zone_crossing_is_alarm_worthy():
    """The incident a restricted zone exists to catch sounds the siren."""
    assert severity_of("ZONE_ENTRY") in ALARM_SEVERITIES


def test_the_approach_warning_is_not_alarm_worthy():
    """It has not happened yet, so it must not be able to cry wolf."""
    assert severity_of("ZONE_APPROACH") not in ALARM_SEVERITIES


@pytest.mark.parametrize("event_type", [
    "TARGET_CONFIRMED", "TARGET_REACQUIRED", "TARGET_ZONE_INTRUSION",
    "TARGET_VEHICLE_FOUND", "TARGET_CROSS_CAMERA_MATCH", "ZONE_ENTRY",
])
def test_every_finding_worth_interrupting_someone_for_alarms(event_type):
    assert severity_of(event_type) in ALARM_SEVERITIES, (
        f"{event_type} would not sound the siren"
    )


@pytest.mark.parametrize("event_type", [
    "ZONE_APPROACH", "ZONE_EXIT", "PLATE_DETECTED", "VEHICLE_DETECTED",
    "LOW_LIGHT", "CAMERA_ONLINE", "TARGET_LOST",
])
def test_routine_events_stay_silent(event_type):
    """An alarm that fires on everything stops being treated as an alarm."""
    assert severity_of(event_type) not in ALARM_SEVERITIES, (
        f"{event_type} would sound the siren for something routine"
    )


def test_the_events_endpoint_stamps_severity_and_alarm():
    """What the Results page reads in order to decide whether to sound."""
    jobs = client.get("/api/jobs").json()
    jobs = jobs if isinstance(jobs, list) else jobs.get("jobs", [])
    done = [j for j in jobs if j.get("status") == "done"]
    if not done:
        pytest.skip("no completed job available to read events from")

    for job in done:
        events = client.get(f"/api/jobs/{job['job_id']}/events").json()
        if not events:
            continue
        for event in events:
            assert "severity" in event, "an event reached the UI with no severity"
            assert "alarm" in event, "an event reached the UI with no alarm decision"
            # The stamp must agree with the rule, not be decided
            # independently. severity_for rather than severity_of, because a
            # few types earn their severity from what actually triggered them:
            # TARGET_BEHAVIOR_ALERT is CRITICAL for a confirmed target and
            # MEDIUM for anyone else behaving oddly in a zone, and claiming
            # the former when it was the latter is exactly the overclaim that
            # made the siren untrustworthy before.
            assert event["severity"] == severity_for(event)
            assert event["alarm"] is (event["severity"] in ALARM_SEVERITIES)
        return
    pytest.skip("no completed job had any events")
