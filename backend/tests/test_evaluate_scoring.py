"""Scoring rules for scripts/evaluate.py.

The accuracy report is what we put in front of judges, so the arithmetic
behind it is tested directly: a matching bug here would print a confident,
wrong number.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "evaluate", Path(__file__).resolve().parents[2] / "scripts" / "evaluate.py"
)
evaluate = importlib.util.module_from_spec(_SPEC)
# Dataclasses resolve their module through sys.modules while the class body runs.
sys.modules["evaluate"] = evaluate
_SPEC.loader.exec_module(evaluate)


def test_edit_distance_and_char_accuracy():
    assert evaluate.edit_distance("TEST123", "TEST123") == 0
    assert evaluate.edit_distance("TEST123", "TE5T123") == 1
    assert evaluate.char_accuracy("TEST123", "TE5T123") == pytest.approx(6 / 7)
    assert evaluate.char_accuracy("AB", "WXYZ") == 0.0


def test_one_detection_cannot_satisfy_two_crossings():
    matched, false_alarms = evaluate.match_zone_entries([5.0], [(4.0, 6.0), (4.5, 6.5)], tolerance=0.0)
    assert matched == [5.0, None]
    assert false_alarms == []


def test_entries_outside_every_window_are_false_alarms():
    matched, false_alarms = evaluate.match_zone_entries([2.0, 10.0, 30.0], [(9.0, 11.0)], tolerance=1.0)
    assert matched == [10.0]
    assert false_alarms == [2.0, 30.0]


def test_tolerance_widens_the_window():
    matched, _ = evaluate.match_zone_entries([12.8], [(10.0, 12.0)], tolerance=1.0)
    assert matched == [12.8]


def test_warning_lead_time_uses_same_track_since_its_last_entry():
    events = [
        {"type": "ZONE_APPROACH", "track_id": 1, "timestamp_sec": 1.0},
        {"type": "ZONE_ENTRY", "track_id": 1, "timestamp_sec": 3.0},
        {"type": "ZONE_APPROACH", "track_id": 2, "timestamp_sec": 9.5},  # other track: ignored
        {"type": "ZONE_APPROACH", "track_id": 1, "timestamp_sec": 8.0},
        {"type": "ZONE_ENTRY", "track_id": 1, "timestamp_sec": 10.0},
    ]
    assert evaluate.warning_lead_times(events, [3.0, 10.0]) == [2.0, 2.0]


def test_labels_reject_rows_the_scorer_cannot_use(tmp_path):
    bad = tmp_path / "labels.csv"
    bad.write_text("clip,module,expected\nclip.mp4,zone,entry\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="zone needs a zone JSON"):
        evaluate.read_labels(bad)


def test_labels_skip_comments_and_group_runs(tmp_path):
    labels_file = tmp_path / "labels.csv"
    labels_file.write_text(
        "# comment\n"
        "clip,module,expected,start_sec,end_sec,reference,zone,lowlight,condition\n"
        "a.mp4,anpr,TEST123,,,,,no,day\n"
        "a.mp4,anpr,ABC999,,,,,no,day\n"
        "a.mp4,person_id,present,,,refs,,yes,night\n",
        encoding="utf-8",
    )
    labels = evaluate.read_labels(labels_file)
    assert [lab.line for lab in labels] == [3, 4, 5]
    runs = evaluate.group_runs(labels)
    assert len(runs) == 2
    assert len(runs[0].labels) == 2 and runs[1].lowlight is True


def _behavior_run(labels, events):
    run = evaluate.Run(module="behavior", clip="c.mp4", reference="", zone="", lowlight=False)
    run.labels = labels
    run.events = events
    return run


def test_behaviour_occurrences_match_once_and_by_type():
    L = evaluate.Label
    labels = [
        L("c.mp4", "behavior", "LOITERING", 10.0, 20.0),
        L("c.mp4", "behavior", "ABNORMAL_SPEED", 30.0, 32.0),
        L("c.mp4", "behavior", "ABNORMAL_SPEED", 50.0, 52.0),
    ]
    events = [
        {"type": "LOITERING", "timestamp_sec": 18.0},
        {"type": "LOITERING", "timestamp_sec": 19.0},       # second alert, same window: false alarm
        {"type": "ABNORMAL_SPEED", "timestamp_sec": 31.5},
        {"type": "LOITERING", "timestamp_sec": 51.0},       # wrong type for the 50-52 s label
    ]
    s = evaluate.score_behavior(_behavior_run(labels, events), tolerance=1.0)
    caught = {(o["type"], o["label"].start_sec): o["detected_at"] for o in s["occurrences"]}
    assert caught == {("LOITERING", 10.0): 18.0, ("ABNORMAL_SPEED", 30.0): 31.5, ("ABNORMAL_SPEED", 50.0): None}
    assert sorted(s["false_alarms"]) == [("LOITERING", 19.0), ("LOITERING", 51.0)]


def test_behaviour_none_rows_only_count_false_alarms():
    L = evaluate.Label
    s = evaluate.score_behavior(
        _behavior_run([L("c.mp4", "behavior", "none")], [{"type": "SUDDEN_DIRECTION_CHANGE", "timestamp_sec": 3.0}]),
        tolerance=1.0,
    )
    assert s["occurrences"] == [] and s["false_alarms"] == [("SUDDEN_DIRECTION_CHANGE", 3.0)]


def test_behaviour_labels_are_validated(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("clip,module,expected,start_sec,end_sec\nc.mp4,behavior,DANCING,1,2\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        evaluate.read_labels(bad)
    no_window = tmp_path / "nowin.csv"
    no_window.write_text("clip,module,expected,start_sec,end_sec\nc.mp4,behavior,LOITERING,,\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        evaluate.read_labels(no_window)
    good = tmp_path / "good.csv"
    good.write_text("clip,module,expected,start_sec,end_sec,calibration\nc.mp4,behavior,loitering,1,9,data/calibrations/cam.json\n",
                    encoding="utf-8")
    (lab,) = evaluate.read_labels(good)
    assert lab.module == "behavior" and lab.calibration == "data/calibrations/cam.json"
