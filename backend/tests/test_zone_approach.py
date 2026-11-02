"""Predicting a restricted-zone intrusion before the boundary is crossed.

An alarm that fires on the crossing tells an operator what has already
happened. These cover the warning that precedes it: extrapolating a track's
recent heading, deciding whether it is genuinely closing on the zone, and
refusing to warn repeatedly about the same approach.

Pure geometry and state - no video, no models - so the rule itself is what is
under test rather than a detector's accuracy.
"""
from __future__ import annotations

from backend.modules.zone.geometry import predicted_entry_eta, velocity_from
from backend.modules.zone.state import INSIDE, OUTSIDE, ZoneMonitor

# A 100x100 box sitting at x >= 200. Everything below walks the x axis at y=50,
# so "approaching" means moving right and "retreating" means moving left.
ZONE = [(200.0, 0.0), (300.0, 0.0), (300.0, 100.0), (200.0, 100.0)]
FPS = 10.0


def _walk(monitor: ZoneMonitor, start_x: float, step_x: float, frames: int,
          track_id: int = 1, start_frame: int = 0) -> None:
    """Move one track along y=50 at a constant step per frame."""
    for i in range(frames):
        frame = start_frame + i
        monitor.update({track_id: (start_x + step_x * i, 50.0)},
                       frame, frame / FPS, ZONE)


def _types(monitor: ZoneMonitor) -> list[str]:
    return [e.type for e in monitor.events]


# --- the prediction itself ------------------------------------------------

def test_velocity_spans_the_whole_window_not_the_last_pair():
    """Measured across the window, so one noisy sample cannot set the heading."""
    history = [(0.0, (0.0, 0.0)), (0.5, (30.0, 0.0)), (1.0, (40.0, 0.0))]
    vx, vy = velocity_from(history)
    assert vx == 40.0  # 40px over 1.0s, not the 20px/s of the final pair
    assert vy == 0.0


def test_velocity_needs_a_baseline():
    assert velocity_from([]) is None
    assert velocity_from([(1.0, (0.0, 0.0))]) is None
    # Two samples at the same instant carry no information about speed.
    assert velocity_from([(1.0, (0.0, 0.0)), (1.0, (5.0, 0.0))]) is None


def test_eta_is_the_earliest_moment_of_entry():
    # 100px away, closing at 100px/s -> about a second out.
    eta = predicted_entry_eta((100.0, 50.0), (100.0, 0.0), ZONE, horizon_sec=2.5)
    assert eta is not None
    assert 0.9 <= eta <= 1.2


def test_no_eta_when_the_path_misses_the_zone():
    # Moving away from the zone.
    assert predicted_entry_eta((100.0, 50.0), (-100.0, 0.0), ZONE, horizon_sec=2.5) is None
    # Moving fast, but parallel to it and never crossing.
    assert predicted_entry_eta((100.0, 50.0), (0.0, 100.0), ZONE, horizon_sec=2.5) is None


def test_no_eta_beyond_the_horizon():
    # 100px away at 10px/s is ten seconds out - not imminent.
    assert predicted_entry_eta((100.0, 50.0), (10.0, 0.0), ZONE, horizon_sec=2.5) is None


# --- the warning ----------------------------------------------------------

def test_approaching_the_zone_warns_before_the_boundary_is_crossed():
    monitor = ZoneMonitor("z1", approach_prediction_sec=2.5, approach_grace_frames=2)
    _walk(monitor, start_x=120.0, step_x=10.0, frames=6)  # 100px/s, closing

    assert "ZONE_APPROACH" in _types(monitor)
    assert "ZONE_ENTRY" not in _types(monitor), "warned only after it had already happened"
    warning = next(e for e in monitor.events if e.type == "ZONE_APPROACH")
    assert warning.data["eta_sec"] > 0
    assert monitor.state_of(1) == OUTSIDE


def test_the_warning_is_followed_by_the_crossing_alarm():
    monitor = ZoneMonitor("z1", entry_grace_frames=2, approach_grace_frames=2)
    _walk(monitor, start_x=120.0, step_x=15.0, frames=16)

    types = _types(monitor)
    assert types.index("ZONE_APPROACH") < types.index("ZONE_ENTRY")
    assert monitor.state_of(1) == INSIDE


def test_walking_away_from_the_zone_never_warns():
    monitor = ZoneMonitor("z1", approach_grace_frames=2)
    _walk(monitor, start_x=180.0, step_x=-10.0, frames=8)
    assert _types(monitor) == []


def test_walking_past_the_zone_never_warns():
    """Parallel to the boundary at speed - close, but not coming in."""
    monitor = ZoneMonitor("z1", approach_grace_frames=2)
    for i in range(10):
        monitor.update({1: (150.0, 10.0 * i)}, i, i / FPS, ZONE)
    assert _types(monitor) == []


def test_a_stationary_track_does_not_warn_on_jitter():
    """Detector wobble must not extrapolate into a phantom approach."""
    monitor = ZoneMonitor("z1", approach_min_speed_px_per_sec=12.0, approach_grace_frames=2)
    for i in range(12):
        monitor.update({1: (190.0 + (i % 2), 50.0)}, i, i / FPS, ZONE)  # +/-1px wobble
    assert _types(monitor) == []


def test_one_approach_warns_once():
    """Dedup: an operator hears about an approach, not about every frame of it."""
    monitor = ZoneMonitor("z1", approach_grace_frames=2, approach_cooldown_sec=20.0)
    _walk(monitor, start_x=100.0, step_x=5.0, frames=18)
    assert _types(monitor).count("ZONE_APPROACH") == 1


def test_the_cooldown_expires_so_a_later_approach_still_warns():
    monitor = ZoneMonitor("z1", approach_grace_frames=2, approach_cooldown_sec=1.0)
    _walk(monitor, start_x=100.0, step_x=8.0, frames=8, start_frame=0)
    first = _types(monitor).count("ZONE_APPROACH")
    assert first == 1
    # Well past the cooldown, approaching again.
    for i in range(8):
        frame = 200 + i
        monitor.update({1: (100.0 + 8.0 * i, 50.0)}, frame, 40.0 + i / FPS, ZONE)
    assert _types(monitor).count("ZONE_APPROACH") == 2


def test_a_track_already_inside_is_not_warned_about():
    """Once someone is in, predicting they might enter is noise."""
    monitor = ZoneMonitor("z1", entry_grace_frames=1, approach_grace_frames=1)
    for i in range(10):
        monitor.update({1: (250.0 + i, 50.0)}, i, i / FPS, ZONE)  # inside, moving deeper
    assert monitor.state_of(1) == INSIDE
    assert "ZONE_APPROACH" not in _types(monitor)


def test_prediction_can_be_turned_off():
    """A horizon of zero disables the warning without touching crossings."""
    monitor = ZoneMonitor("z1", entry_grace_frames=2, approach_prediction_sec=0.0)
    _walk(monitor, start_x=120.0, step_x=15.0, frames=16)
    assert "ZONE_APPROACH" not in _types(monitor)
    assert "ZONE_ENTRY" in _types(monitor), "disabling the warning broke the alarm"


def test_two_tracks_are_warned_about_independently():
    monitor = ZoneMonitor("z1", approach_grace_frames=2)
    for i in range(8):
        monitor.update(
            {1: (120.0 + 12.0 * i, 50.0),   # closing
             2: (180.0 - 12.0 * i, 50.0)},  # retreating
            i, i / FPS, ZONE,
        )
    warned = {e.track_id for e in monitor.events if e.type == "ZONE_APPROACH"}
    assert warned == {1}


def test_approaching_flag_reports_live_status():
    """What the dashboard and the zone outline read to show state."""
    monitor = ZoneMonitor("z1", approach_grace_frames=2)
    assert monitor.approaching(1) is False
    _walk(monitor, start_x=120.0, step_x=12.0, frames=5)
    assert monitor.approaching(1) is True


# --- live frame rates -----------------------------------------------------
#
# Added after the warning worked in every unit test and never once fired on a
# live camera. Forensic Mode analyses 30fps; live analysis on CPU delivers
# 1-3fps, and the velocity window was shorter than the gap between live
# frames - so every frame looked like a track returning from an absence, the
# history was cleared, and no heading could ever be measured.

def test_it_warns_at_live_frame_rates():
    """One frame per second, which is what a CPU camera actually delivers."""
    monitor = ZoneMonitor("z1", approach_grace_frames=2)
    for i in range(6):
        monitor.update({1: (100.0 + 40.0 * i, 50.0)}, i, float(i), ZONE)  # 40px/s at 1fps
    assert "ZONE_APPROACH" in _types(monitor), "the warning is dead at live frame rates"


def test_it_warns_at_two_frames_per_second():
    monitor = ZoneMonitor("z1", approach_grace_frames=2)
    for i in range(8):
        monitor.update({1: (120.0 + 20.0 * i, 50.0)}, i, i * 0.5, ZONE)
    assert "ZONE_APPROACH" in _types(monitor)


def test_a_real_absence_still_restarts_the_baseline():
    """The gap rule must still do its job for a genuinely absent track."""
    monitor = ZoneMonitor("z1", approach_grace_frames=2)
    monitor.update({1: (300.0, 50.0)}, 0, 0.0, ZONE)     # far side of the zone
    # Gone for a minute, then reappears on the near side moving away. Measuring
    # across that gap would invent a large leftward velocity.
    monitor.update({1: (100.0, 50.0)}, 100, 60.0, ZONE)
    monitor.update({1: (96.0, 50.0)}, 101, 61.0, ZONE)
    assert "ZONE_APPROACH" not in _types(monitor)
