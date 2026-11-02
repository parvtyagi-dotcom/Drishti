"""A webcam must be opened with DirectShow first on every connect.

Two USB webcams were once remembered as Media-Foundation-only after a single
DirectShow failure, and stayed on MSMF for good - where they gave black frames
at 1fps. The failure itself was only two opens colliding.
"""
import os

import cv2
import pytest

from backend.core import video
from backend.core.video import LiveStreamReader

pytestmark = pytest.mark.skipif(os.name != "nt", reason="DirectShow/MSMF are Windows backends")


class FakeCapture:
    def __init__(self, opens: bool):
        self._opens = opens

    def isOpened(self):
        return self._opens

    def set(self, *_):
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: 30.0, cv2.CAP_PROP_FRAME_WIDTH: 1280, cv2.CAP_PROP_FRAME_HEIGHT: 720}.get(prop, 0)

    def release(self):
        pass


def test_directshow_is_tried_again_after_it_failed_once(monkeypatch):
    calls = []
    dshow_works = iter([False, True])

    def fake_capture(index, backend):
        calls.append(backend)
        if backend == cv2.CAP_DSHOW:
            return FakeCapture(next(dshow_works))
        return FakeCapture(True)

    monkeypatch.setattr(video.cv2, "VideoCapture", fake_capture)

    reader = LiveStreamReader("0")
    reader._open()  # DirectShow fails, falls back to MSMF
    assert calls == [cv2.CAP_DSHOW, cv2.CAP_MSMF]

    calls.clear()
    reader._open()  # reconnect: DirectShow again, not straight to MSMF
    assert calls == [cv2.CAP_DSHOW]
