"""Motion check: the detector sleeps on a still scene and wakes on movement."""
import numpy as np

from backend.core.motion import MotionGate


def scene(x=None):
    img = np.full((240, 320, 3), 90, dtype=np.uint8)
    if x is not None:
        img[100:180, x:x + 30] = 230
    return img


def test_still_scene_is_mostly_skipped():
    gate = MotionGate(check_every_sec=5)
    runs = [gate.should_run(scene(), t / 10, tracking=False) for t in range(200)]
    assert sum(runs) <= 6          # first frame, then one safety check every 5 s
    assert gate.idle_share > 0.9


def test_movement_wakes_it_and_it_stays_awake_briefly():
    gate = MotionGate(hold_sec=3, check_every_sec=60)
    for t in range(50):
        gate.should_run(scene(), t / 10, tracking=False)
    assert gate.should_run(scene(40), 5.0, tracking=False)
    assert gate.should_run(scene(40), 7.0, tracking=False)       # within hold
    assert not gate.should_run(scene(40), 9.0, tracking=False)   # still again


def test_someone_tracked_keeps_it_awake():
    gate = MotionGate(hold_sec=1, check_every_sec=60)
    gate.should_run(scene(), 0, tracking=False)
    assert gate.should_run(scene(), 10, tracking=True)
