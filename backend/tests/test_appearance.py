"""Appearance search: one clear crop per track, ranked by a text query."""
from __future__ import annotations

import numpy as np

from backend.live.appearance import AppearanceIndex


class ColourModel:
    """Stands in for CLIP: an image's vector is its mean colour (BGR); a text
    query names a colour."""
    WORDS = {"blue": [1, 0, 0], "green": [0, 1, 0], "red": [0, 0, 1]}

    def images(self, crops):
        v = np.array([c.reshape(-1, 3).mean(axis=0) + 1e-3 for c in crops], dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def text(self, phrases):
        v = np.array([self.WORDS.get(p.split()[-1], [1, 1, 1]) for p in phrases], dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


class Cam:
    def __init__(self, cid, name):
        self.camera_id, self.name = cid, name


class Session:
    def __init__(self, cid="c1", name="Gate 1"):
        self.camera = Cam(cid, name)


def frame(colour_boxes):
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    for (x1, y1, x2, y2), bgr in colour_boxes:
        img[y1:y2, x1:x2] = bgr
    return img


def make(tmp_path):
    now = [1000.0]
    index = AppearanceIndex(tmp_path / "a.db", model=ColourModel(), clock=lambda: now[0])
    filed = []
    index._ensure_worker = lambda: None
    index._queue.put_nowait = lambda batch: filed.append(index.add(batch))
    return index, now


RED = (0, 0, 255)
BLUE = (255, 0, 0)


def test_tracks_are_filed_after_leaving_view_and_ranked_by_query(tmp_path):
    index, now = make(tmp_path)
    s = Session()
    red_box, blue_box = (10, 10, 60, 130), (200, 10, 260, 140)
    for _ in range(3):
        index.observe(s, frame([(red_box, RED), (blue_box, BLUE)]), [(1, 0, red_box), (2, 2, blue_box)])
        now[0] += 0.5
    now[0] += 3
    index.observe(s, frame([]), [])
    assert index.stats()["sightings"] == 2
    top = index.search("a man in red")[0]
    assert top["track_id"] == 1 and top["kind"] == "person" and top["camera_name"] == "Gate 1"
    assert index.search("blue")[0]["kind"] == "car"
    assert [r["kind"] for r in index.search("red", kind="vehicle")] == ["car"]
    assert index.crop(top["id"])[:2] == b"\xff\xd8"


def test_tiny_boxes_and_other_classes_are_ignored(tmp_path):
    index, now = make(tmp_path)
    s = Session()
    index.observe(s, frame([]), [(1, 0, (10, 10, 30, 30)), (2, 16, (50, 10, 150, 200))])
    now[0] += 5
    index.observe(s, frame([]), [])
    assert index.stats()["sightings"] == 0


def test_someone_standing_still_is_filed_anyway(tmp_path):
    index, now = make(tmp_path)
    s = Session()
    box = (10, 10, 60, 130)
    for _ in range(25):
        index.observe(s, frame([(box, RED)]), [(1, 0, box)])
        now[0] += 0.5
    assert index.stats()["sightings"] == 1


def test_camera_and_time_filters(tmp_path):
    index, now = make(tmp_path)
    box = (10, 10, 60, 130)
    index.observe(Session("c1", "Gate 1"), frame([(box, RED)]), [(1, 0, box)])
    index.observe(Session("c2", "Gate 2"), frame([(box, RED)]), [(1, 0, box)])
    now[0] += 5
    index.observe(Session("c1", "Gate 1"), frame([]), [])
    index.observe(Session("c2", "Gate 2"), frame([]), [])
    assert len(index.search("red", camera="Gate 2")) == 1
    assert index.search("red", since=now[0] + 1) == []


def test_near_identical_pictures_are_shown_once(tmp_path):
    index, now = make(tmp_path)
    box = (10, 10, 60, 130)
    for track in (1, 2):
        index.observe(Session(), frame([(box, RED)]), [(track, 0, box)])
    now[0] += 5
    index.observe(Session(), frame([]), [])
    assert index.stats()["sightings"] == 2
    assert len(index.search("red")) == 1
