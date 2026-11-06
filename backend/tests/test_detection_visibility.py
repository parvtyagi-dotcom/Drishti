"""Detection visibility and robustness: face/candidate boxes, cheaper per-frame
face search, a safer face veto, vote defaults, the live person stride and
loud ANPR fallbacks. Pure helpers only - no models are loaded here."""
import logging

import cv2
import numpy as np
import pytest

from backend.config import PersonIDConfig, TrackerConfig
from backend.core.output import GREEN, GREY, YELLOW, draw_thin_box
from backend.core.target_lock import TargetLock
from backend.modules.anpr.aggregator import AggregatedPlate


# ---------------------------------------------------------------- TargetLock
def _lock(n=2):
    return TargetLock(confirmation_observations=n, gap_tolerance_frames=10)


def test_candidates_lists_tracks_with_votes_without_mutating_counts():
    lock = _lock(3)
    lock.register_match(7, 0, 0.0, votes=1, similarity=0.5)
    assert lock.candidates() == {7: 1}
    assert lock.candidate_count(99) == 0
    assert 99 not in lock.match_counts  # reading must not create entries
    assert not lock.confirmed


def test_candidates_is_a_copy():
    lock = _lock(3)
    lock.register_match(7, 0, 0.0, votes=1, similarity=0.5)
    lock.candidates()[7] = 50
    assert lock.candidate_count(7) == 1


# ---------------------------------------------------------------- face boxes
from backend.modules.person_id.face_search import (  # noqa: E402
    _FaceOrientation, _VetoStreaks, _candidate_body_label, _face_box_style,
    _face_bbox_in_crop, _face_veto_applies,
)
from backend.orchestrator_overlay import _plate_box_style  # noqa: E402


@pytest.mark.parametrize("rotation", [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE])
def test_face_bbox_maps_back_from_every_rotation(rotation):
    h, w = 120, 80
    image = np.zeros((h, w), np.uint8)
    image[30:50, 10:40] = 255  # the "face": x 10..40, y 30..50
    turned = image if rotation is None else cv2.rotate(image, rotation)
    ys, xs = np.nonzero(turned)
    rotated_bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    assert _face_bbox_in_crop(rotated_bbox, rotation, (h, w), scale=1.0) == (10, 30, 40, 50)


def test_face_bbox_undoes_the_upscale():
    # crop enlarged 2x before detection: a face at 20..80 is at 10..40 originally
    assert _face_bbox_in_crop((20, 60, 80, 100), None, (240, 160), scale=2.0) == (10, 30, 40, 50)


def test_face_box_style_green_for_the_confirmed_target():
    lock = _lock(1)
    lock.register_match(3, 0, 0.0, votes=1, similarity=0.6)
    color, label, _ = _face_box_style(3, 0.613, lock)
    assert color == GREEN and label == "MATCH 0.61"


def test_face_box_style_yellow_for_a_candidate():
    lock = _lock(2)
    lock.register_match(4, 0, 0.0, votes=1, similarity=0.5)
    color, label, _ = _face_box_style(4, 0.5, lock)
    assert color == YELLOW and label == "CANDIDATE 0.50"


def test_face_box_style_faint_grey_without_label_for_a_stranger():
    color, label, thickness = _face_box_style(5, 0.1, _lock(2))
    assert color == GREY and label is None and thickness == 1


def test_candidate_body_label():
    assert _candidate_body_label(1, 2) == "CANDIDATE 1/2"


def test_draw_thin_box_draws_in_place():
    image = np.zeros((100, 100, 3), np.uint8)
    draw_thin_box(image, (20, 40, 60, 80), YELLOW, label="CANDIDATE 0.50")
    assert tuple(image[60, 20]) == YELLOW  # left edge
    assert image[:40].any()  # label sits above the box


# ---------------------------------------------------------------- orientation
_UPRIGHT_RAW = np.array([0, 0, 30, 30, 10, 10, 20, 10, 15, 15, 12, 20, 18, 20, 0.9], np.float32)


class _FakeFace:
    def __init__(self):
        self.bbox = (0, 0, 30, 30)
        self.raw = _UPRIGHT_RAW
        self.width_px = 30.0


class _FakeDetector:
    """Finds a face only in images of `face_shape`."""

    def __init__(self, face_shape):
        self.face_shape = face_shape
        self.calls = []

    def best(self, image):
        self.calls.append(image.shape[:2])
        return _FakeFace() if image.shape[:2] == self.face_shape else None


def test_upright_camera_tries_only_the_unrotated_crop():
    crop = np.zeros((120, 80, 3), np.uint8)
    detector = _FakeDetector((120, 80))
    orientation = _FaceOrientation()
    for _ in range(5):
        _, face, rotation = orientation.find(detector, crop)
        assert face is not None and rotation is None
    assert detector.calls == [(120, 80)] * 5


def test_a_miss_does_not_pay_for_rotations_every_time():
    crop = np.zeros((120, 80, 3), np.uint8)
    detector = _FakeDetector((1, 1))  # no face anywhere
    orientation = _FaceOrientation()
    for _ in range(_FaceOrientation.PROBE_EVERY_MISSES - 1):
        orientation.find(detector, crop)
    assert len(detector.calls) == _FaceOrientation.PROBE_EVERY_MISSES - 1


def test_a_sideways_camera_is_learned_and_its_rotation_tried_first():
    crop = np.zeros((120, 80, 3), np.uint8)
    detector = _FakeDetector((80, 120))  # face only visible once turned 90 degrees
    orientation = _FaceOrientation()
    found = None
    for _ in range(_FaceOrientation.PROBE_EVERY_MISSES):
        _, found, rotation = orientation.find(detector, crop)
    assert found is not None and orientation.sideways
    detector.calls.clear()
    _, face, rotation = orientation.find(detector, crop)
    assert face is not None and rotation is not None
    assert detector.calls == [(80, 120)]  # learned rotation first, nothing else


# ---------------------------------------------------------------- face veto
def test_veto_applies_to_arcface_at_the_normal_width():
    cfg = PersonIDConfig()
    assert _face_veto_applies(45, -0.2, "arcface", cfg.face_reject_threshold, cfg)


def test_veto_ignores_a_mid_size_sface_face():
    cfg = PersonIDConfig()
    assert not _face_veto_applies(60, 0.05, "sface", 0.25, cfg)


def test_veto_applies_to_a_large_sface_face():
    cfg = PersonIDConfig()
    assert cfg.face_reject_min_width_px_sface >= 80
    assert _face_veto_applies(cfg.face_reject_min_width_px_sface, 0.05, "sface", 0.25, cfg)


def test_no_veto_when_similarity_is_not_low():
    cfg = PersonIDConfig()
    assert not _face_veto_applies(200, 0.3, "sface", 0.25, cfg)


def test_forensic_search_still_tries_every_rotation_on_every_crop():
    """Forensic Mode keeps the old exhaustive search: a sideways recording must
    not miss its first faces while the cheap live search learns the angle."""
    crop = np.zeros((120, 80, 3), np.uint8)
    detector = _FakeDetector((80, 120))  # face only visible once turned 90 degrees
    orientation = _FaceOrientation(exhaustive=True)
    _, face, rotation = orientation.find(detector, crop)
    assert face is not None and rotation is not None
    assert len(detector.calls) == 4


def test_veto_streak_outlives_the_face_box_display_hold():
    """Live samples faces about once a second; vetoes 1.5 s apart are still
    consecutive evidence. They used to be forgotten with the 1 s face box."""
    streaks = _VetoStreaks(window_sec=10.0)
    assert streaks.record(1, True, now=0.0) == 1
    streaks.prune(now=1.5)
    assert streaks.record(1, True, now=1.5) == 2


def test_veto_streak_expires_after_its_window():
    streaks = _VetoStreaks(window_sec=10.0)
    streaks.record(1, True, now=0.0)
    assert streaks.record(1, True, now=11.0) == 1
    streaks.prune(now=30.0)
    assert streaks.record(1, False, now=30.0) == 0


def test_veto_streak_needs_consecutive_vetoes():
    streaks = _VetoStreaks()
    assert streaks.record(1, True) == 1
    assert streaks.record(1, False) == 0
    assert streaks.record(1, True) == 1
    assert streaks.record(1, True) == 2
    assert streaks.record(2, True) == 1


def test_confirmed_lock_release_needs_two_vetoes_by_default():
    assert PersonIDConfig().target_release_consecutive_vetoes == 2


# ---------------------------------------------------------------- votes / stride
def test_one_reference_vote_live_but_forensic_keeps_two():
    """Live needs a quick lock from few, small faces; Forensic Mode keeps its
    stricter, previously validated default."""
    from backend.live.live_config import live_person_id_config
    assert PersonIDConfig().min_reference_votes == 2
    live = live_person_id_config()
    assert live.min_reference_votes == 1
    assert live.confirmation_observations == 2
    assert PersonIDConfig.resolve_votes(live.min_reference_votes, 2) == 1


def test_person_detection_stride_defaults_to_the_shared_stride():
    assert TrackerConfig().person_detection_stride is None


def test_live_person_tracker_detects_more_often_than_the_shared_one():
    from backend.live.camera_manager import live_tracker_config

    cfg = live_tracker_config()
    assert cfg.person_detection_stride is not None
    assert cfg.person_detection_stride < cfg.detection_stride
    assert cfg.detection_stride == TrackerConfig().detection_stride


# ---------------------------------------------------------------- ANPR
def test_plate_box_style_confirmed_vs_provisional():
    confirmed = AggregatedPlate("TEST123", 91.4, 4, 4, True)
    provisional = AggregatedPlate("TEST12", 70.0, 1, 1, False)
    label, thickness = _plate_box_style(confirmed)
    assert label == ["TEST123", "91%"] and thickness == 3
    label, thickness = _plate_box_style(provisional)
    assert label == ["TEST12 ?"] and thickness == 1


def test_neural_plate_detector_failure_is_logged_once(monkeypatch, caplog):
    from backend.modules.anpr import plate_detector as pd

    def boom(*a, **k):
        raise RuntimeError("model download failed")

    monkeypatch.setattr(pd, "_NeuralPlateDetector", boom)
    monkeypatch.setattr(pd, "_WARNED", set())
    with caplog.at_level(logging.WARNING, logger=pd.logger.name):
        assert pd.PlateDetector().backend == "classical"
        pd.PlateDetector()
    hits = [r for r in caplog.records if "model download failed" in r.getMessage()]
    assert len(hits) == 1


def test_fast_plate_ocr_failure_and_unavailable_ocr_are_logged(monkeypatch, caplog):
    from backend.modules.anpr import ocr

    def boom(*a, **k):
        raise RuntimeError("no fast_plate_ocr")

    monkeypatch.setattr(ocr, "_FastPlateOcrBackend", boom)
    monkeypatch.setattr(ocr, "configure_tesseract", lambda: ocr.OcrStatus(False, reason="tesseract missing"))
    monkeypatch.setattr(ocr, "_WARNED", set())
    with caplog.at_level(logging.WARNING, logger=ocr.logger.name):
        reader = ocr.PlateOcr()
        ocr.PlateOcr()
    assert not reader.available
    messages = [r.getMessage() for r in caplog.records]
    assert sum("no fast_plate_ocr" in m for m in messages) == 1
    assert sum("tesseract missing" in m for m in messages) == 1
