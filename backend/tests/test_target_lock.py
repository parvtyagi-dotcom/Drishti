"""Unit tests for the pure target-lock state machine (no video/CV involved).

Covers the "proper tests" the spec calls for that don't need real footage:
target confirmed, target temporarily occluded, target not present, and
never replacing a locked target with a different person's box.
"""
from backend.core.target_lock import TargetLock


def make_lock(confirmation_observations=2, gap_tolerance_frames=5):
    return TargetLock(
        confirmation_observations=confirmation_observations,
        gap_tolerance_frames=gap_tolerance_frames,
    )


def test_target_not_present_never_confirms():
    lock = make_lock()
    for i in range(30):
        visible = lock.step_visibility(active_track_ids=set(), frame_index=i, timestamp_sec=i / 30)
        assert visible is None
    assert lock.confirmed is False
    assert lock.target_track_id is None
    assert lock.visible_frame_count == 0


def test_confirmation_requires_multiple_observations():
    lock = make_lock(confirmation_observations=2)
    lock.register_match(track_id=7, frame_index=0, timestamp_sec=0.0, votes=2, similarity=0.5)
    assert lock.confirmed is False  # only one observation so far

    lock.register_match(track_id=7, frame_index=5, timestamp_sec=0.17, votes=2, similarity=0.55)
    assert lock.confirmed is True
    assert lock.target_track_id == 7
    assert lock.first_seen_sec == 0.17


def test_confirmed_target_draws_every_frame_present():
    lock = make_lock()
    lock.register_match(7, 0, 0.0, votes=2, similarity=0.5)
    lock.register_match(7, 1, 0.03, votes=2, similarity=0.5)
    assert lock.confirmed

    for i in range(2, 10):
        visible = lock.step_visibility({7, 9}, i, i / 30)
        assert visible == 7
    assert lock.visible_frame_count == 8


def test_short_occlusion_is_tolerated_without_losing_identity():
    lock = make_lock(gap_tolerance_frames=5)
    lock.register_match(7, 0, 0.0, votes=2, similarity=0.5)
    lock.register_match(7, 1, 0.03, votes=2, similarity=0.5)
    lock.step_visibility({7}, 2, 0.06)

    # target absent for 3 frames (< gap_tolerance_frames=5) - should stay locked
    for i in range(3, 6):
        visible = lock.step_visibility(set(), i, i / 30)
        assert visible is None
    assert lock.target_track_id == 7  # identity retained
    assert not any(e.type == "TARGET_LOST" for e in lock.events)


def test_occlusion_beyond_tolerance_marks_lost_then_reacquires():
    lock = make_lock(gap_tolerance_frames=3)
    lock.register_match(7, 0, 0.0, votes=2, similarity=0.5)
    lock.register_match(7, 1, 0.03, votes=2, similarity=0.5)
    lock.step_visibility({7}, 2, 0.06)

    # absent for 4 frames (> tolerance of 3) -> should become lost
    for i in range(3, 7):
        lock.step_visibility(set(), i, i / 30)
    assert lock.target_track_id is None
    assert any(e.type == "TARGET_LOST" for e in lock.events)

    # a new track (id=42) matches strongly -> re-acquisition, no fresh
    # confirmation_observations needed
    lock.register_match(42, 8, 0.27, votes=2, similarity=0.6)
    assert lock.target_track_id == 42
    assert any(e.type == "TARGET_REACQUIRED" and e.track_id == 42 for e in lock.events)

    visible = lock.step_visibility({42}, 9, 0.3)
    assert visible == 42


def test_locked_target_is_never_replaced_by_a_different_person():
    lock = make_lock()
    lock.register_match(7, 0, 0.0, votes=2, similarity=0.5)
    lock.register_match(7, 1, 0.03, votes=2, similarity=0.5)
    assert lock.confirmed and lock.target_track_id == 7

    # target 7 still actively tracked; a different track (99) also matches -
    # must NOT steal the lock while 7 is present.
    lock.step_visibility({7, 99}, 2, 0.06)
    lock.register_match(99, 2, 0.06, votes=2, similarity=0.9)
    assert lock.target_track_id == 7

    visible = lock.step_visibility({7, 99}, 3, 0.1)
    assert visible == 7


# --- stricter re-acquisition and release (wrong-person fix) -----------------

def _lost_lock(reacquire_observations: int = 3) -> TargetLock:
    lock = TargetLock(confirmation_observations=1, gap_tolerance_frames=1,
                      reacquire_observations=reacquire_observations)
    lock.register_match(1, 0, 0.0, votes=2, similarity=0.9)
    for f in (1, 2, 3):
        lock.step_visibility(set(), f, f / 10)
    assert lock.confirmed and lock.target_track_id is None
    return lock


def test_appearance_alone_needs_several_matches_to_reacquire():
    lock = _lost_lock()
    lock.register_match(7, 4, 0.4, votes=1, similarity=0.7, strong=False)
    lock.register_match(7, 5, 0.5, votes=1, similarity=0.7, strong=False)
    assert lock.target_track_id is None  # two looks are not enough
    lock.register_match(7, 6, 0.6, votes=1, similarity=0.7, strong=False)
    assert lock.target_track_id == 7


def test_appearance_matches_on_different_people_do_not_add_up():
    lock = _lost_lock()
    for tid in (7, 8, 9):
        lock.register_match(tid, 4, 0.4, votes=1, similarity=0.7, strong=False)
    assert lock.target_track_id is None


def test_a_face_match_still_reacquires_at_once():
    lock = _lost_lock()
    lock.register_match(7, 4, 0.4, votes=2, similarity=0.6)
    assert lock.target_track_id == 7


def test_release_drops_a_wrong_lock_and_can_reacquire():
    lock = _lost_lock()
    lock.register_match(7, 4, 0.4, votes=2, similarity=0.6)
    lock.release(5, 0.5, "face does not match the target")
    assert lock.target_track_id is None and lock.confirmed
    assert lock.events[-1].type == "TARGET_LOST" and lock.events[-1].data["reason"].startswith("face")
    lock.register_match(3, 6, 0.6, votes=2, similarity=0.6)
    assert lock.target_track_id == 3


def test_default_keeps_single_match_reacquisition_for_other_modules():
    lock = _lost_lock(reacquire_observations=1)
    lock.register_match(7, 4, 0.4, votes=1, similarity=0.7, strong=False)
    assert lock.target_track_id == 7
