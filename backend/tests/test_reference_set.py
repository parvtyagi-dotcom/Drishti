"""Reference-photo handling tests using real YuNet+SFace models.

Uses the ground-truth fixtures under references/smoke_target (two photos of
the same person, cropped from different frames of the same real video) and
references/smoke_unrelated (a different person entirely) - see
scratchpad notes: these were built by detecting faces with YuNet in real
footage and cropping around them, so "same person" / "different person" is
known by construction, not assumed.
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.config import ARCFACE_MODEL_PATH, PersonIDConfig, SFACE_MODEL_PATH, YUNET_MODEL_PATH
from backend.modules.person_id.detector import FaceDetector
from backend.modules.person_id.recognizer import ArcFaceRecognizer, FaceRecognizer, ReferenceSet

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TARGET_REF1 = REPO_ROOT / "references" / "smoke_target" / "ref1.jpg"
TARGET_REF2 = REPO_ROOT / "references" / "smoke_target" / "ref2.jpg"
UNRELATED_REF1 = REPO_ROOT / "references" / "smoke_unrelated" / "ref1.jpg"

pytestmark = pytest.mark.skipif(
    not (TARGET_REF1.exists() and TARGET_REF2.exists() and UNRELATED_REF1.exists()),
    reason="ground-truth reference fixtures not present",
)


@pytest.fixture(scope="module")
def detector():
    return FaceDetector(YUNET_MODEL_PATH)


@pytest.fixture(scope="module")
def recognizer():
    return FaceRecognizer(SFACE_MODEL_PATH)


def test_single_reference_photo_is_usable(detector, recognizer):
    refset = ReferenceSet([TARGET_REF1], detector, recognizer)
    assert refset.usable_count == 1
    assert refset.rejected == []
    votes = PersonIDConfig.resolve_votes(configured_min_votes=2, usable_reference_count=refset.usable_count)
    assert votes == 1  # must not demand 2 votes from 1 reference


def test_multiple_reference_photos_all_usable(detector, recognizer):
    refset = ReferenceSet([TARGET_REF1, TARGET_REF2], detector, recognizer)
    assert refset.usable_count == 2
    votes = PersonIDConfig.resolve_votes(configured_min_votes=2, usable_reference_count=refset.usable_count)
    assert votes == 2


def test_unreadable_or_faceless_photo_is_rejected_not_crashed(tmp_path, detector, recognizer):
    blank_path = tmp_path / "blank.jpg"
    cv2.imwrite(str(blank_path), np.full((200, 200, 3), 128, dtype=np.uint8))

    refset = ReferenceSet([TARGET_REF1, blank_path], detector, recognizer)
    assert refset.usable_count == 1  # blank image dropped, not fatal
    assert len(refset.rejected) == 1
    assert refset.rejected[0][0] == blank_path


def test_same_person_scores_above_threshold_different_person_does_not(detector, recognizer):
    cfg = PersonIDConfig()
    refset = ReferenceSet([TARGET_REF1], detector, recognizer)

    probe_same = cv2.imread(str(TARGET_REF2))
    face_same = detector.best(probe_same)
    emb_same = recognizer.embed(probe_same, face_same)
    votes_same, sim_same = refset.vote(emb_same, cfg.similarity_threshold)
    assert votes_same >= 1
    assert sim_same >= cfg.similarity_threshold

    probe_diff = cv2.imread(str(UNRELATED_REF1))
    face_diff = detector.best(probe_diff)
    emb_diff = recognizer.embed(probe_diff, face_diff)
    votes_diff, sim_diff = refset.vote(emb_diff, cfg.similarity_threshold)
    assert votes_diff == 0
    assert sim_diff < cfg.similarity_threshold


@pytest.mark.parametrize("rotation", [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE])
def test_a_sideways_or_upside_down_photo_is_still_usable(tmp_path, detector, recognizer, rotation):
    turned = tmp_path / "turned.jpg"
    cv2.imwrite(str(turned), cv2.rotate(cv2.imread(str(TARGET_REF1)), rotation))
    refs = ReferenceSet([turned], detector, recognizer)
    assert refs.usable_count == 1 and refs.rejected == []
    upright = ReferenceSet([TARGET_REF2], detector, recognizer)
    _, best = upright.vote(refs.embeddings[0].embedding, threshold=PersonIDConfig().similarity_threshold)
    assert best >= PersonIDConfig().similarity_threshold


@pytest.mark.skipif(not ARCFACE_MODEL_PATH.exists(), reason="ArcFace model not downloaded")
def test_arcface_matches_the_same_person_and_rejects_another(detector):
    arc = ArcFaceRecognizer(ARCFACE_MODEL_PATH)
    refs = ReferenceSet([TARGET_REF1], detector, arc)
    probe = ReferenceSet([TARGET_REF2], detector, arc).embeddings[0].embedding
    stranger = ReferenceSet([UNRELATED_REF1], detector, arc).embeddings[0].embedding
    assert refs.vote(probe, arc.match_threshold)[0] == 1
    assert refs.vote(stranger, arc.match_threshold)[0] == 0


def test_a_face_is_found_in_a_sideways_person_crop(detector):
    from backend.orchestrator import _find_face

    crop = cv2.rotate(cv2.imread(str(TARGET_REF1)), cv2.ROTATE_90_CLOCKWISE)
    image, face = _find_face(detector, crop)
    assert face is not None
    r = face.raw
    assert abs(r[6] - r[4]) > abs(r[7] - r[5])  # eyes level: the face was turned upright
