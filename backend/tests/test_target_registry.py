"""Cross-camera appearance sharing.

Matching a person between cameras on appearance alone is the weakest evidence
the system produces, so the guardrails around it are what these tests check:
that a camera never matches against itself, that stale appearances expire, and
that the store stays bounded on a system running for days.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from backend.live.target_registry import MAX_SHARED_EMBEDDINGS, TargetRegistry


def unit(*values: float) -> np.ndarray:
    """Normalised vector, matching what the ReID model emits - similarity is
    a dot product, which is only a cosine if the vectors are unit length."""
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


@pytest.fixture
def registry() -> TargetRegistry:
    return TargetRegistry()


def test_a_camera_never_matches_against_its_own_observations(registry):
    """Otherwise a camera would confirm a cross-camera handover using nothing
    but its own view, and one camera's mistake would reinforce itself."""
    registry.contribute("target-1", unit(1, 0, 0), "cam-01")
    assert registry.embeddings_for("target-1", exclude_camera="cam-01") == []
    assert len(registry.embeddings_for("target-1", exclude_camera="cam-02")) == 1


def test_appearance_is_shared_between_different_cameras(registry):
    registry.contribute("target-1", unit(1, 0, 0), "cam-01")
    registry.contribute("target-1", unit(0.9, 0.1, 0), "cam-01")
    shared = registry.embeddings_for("target-1", exclude_camera="cam-04")
    assert len(shared) == 2


def test_targets_are_kept_separate(registry):
    registry.contribute("target-1", unit(1, 0, 0), "cam-01")
    registry.contribute("target-2", unit(0, 1, 0), "cam-01")
    assert len(registry.embeddings_for("target-1", exclude_camera="cam-02")) == 1
    assert len(registry.embeddings_for("target-2", exclude_camera="cam-02")) == 1


def test_stale_appearances_expire():
    """Appearance is clothing and build, not identity. An hour later it is
    weak evidence, so it must not keep producing confident matches."""
    registry = TargetRegistry(ttl_sec=0.3)
    registry.contribute("target-1", unit(1, 0, 0), "cam-01")
    assert len(registry.embeddings_for("target-1", exclude_camera="cam-02")) == 1
    time.sleep(0.4)
    assert registry.embeddings_for("target-1", exclude_camera="cam-02") == []


def test_storage_stays_bounded(registry):
    # A camera watching a target for hours must not grow this without limit.
    for i in range(MAX_SHARED_EMBEDDINGS * 3):
        registry.contribute("target-1", unit(1, i * 0.001 + 0.001, 0), "cam-01")
    assert len(registry.embeddings_for("target-1", exclude_camera="cam-02")) <= MAX_SHARED_EMBEDDINGS


def test_handover_reports_the_previous_camera(registry):
    assert registry.record_sighting("target-1", "cam-01", "CAM-01", "face") is None
    # Same camera again is not a handover.
    assert registry.record_sighting("target-1", "cam-01", "CAM-01", "face") is None
    previous = registry.record_sighting("target-1", "cam-04", "CAM-04", "appearance")
    assert previous is not None
    assert previous.camera_id == "cam-01"


def test_last_sighting_tracks_the_current_camera(registry):
    registry.record_sighting("target-1", "cam-01", "CAM-01", "face")
    registry.record_sighting("target-1", "cam-04", "CAM-04", "appearance")
    latest = registry.last_sighting("target-1")
    assert latest.camera_id == "cam-04"
    assert latest.confirmed_by == "appearance"


def test_cameras_holding_a_target(registry):
    registry.contribute("target-1", unit(1, 0, 0), "cam-01")
    registry.contribute("target-1", unit(0, 1, 0), "cam-04")
    assert registry.cameras_holding("target-1") == {"cam-01", "cam-04"}


def test_similar_appearance_scores_higher_than_dissimilar(registry):
    """The matching itself is a dot product in the pipeline; this confirms the
    stored vectors behave the way that calculation assumes."""
    registry.contribute("target-1", unit(1, 0, 0), "cam-01")
    stored = registry.embeddings_for("target-1", exclude_camera="cam-04")[0]
    same_person = unit(0.98, 0.02, 0)
    someone_else = unit(0, 1, 0)
    assert float(np.dot(stored, same_person)) > 0.9
    assert float(np.dot(stored, someone_else)) < 0.3


def test_forget_clears_everything_for_a_target(registry):
    registry.contribute("target-1", unit(1, 0, 0), "cam-01")
    registry.record_sighting("target-1", "cam-01", "CAM-01", "face")
    registry.forget("target-1")
    assert registry.embeddings_for("target-1", exclude_camera="cam-02") == []
    assert registry.last_sighting("target-1") is None


def test_unknown_target_is_empty_not_an_error(registry):
    assert registry.embeddings_for("never-seen", exclude_camera="cam-01") == []
    assert registry.last_sighting("never-seen") is None
    assert registry.cameras_holding("never-seen") == set()


def test_contributing_nothing_is_ignored(registry):
    registry.contribute("target-1", None, "cam-01")
    registry.contribute(None, unit(1, 0, 0), "cam-01")
    assert registry.embeddings_for("target-1", exclude_camera="cam-02") == []
