"""Live video at camera speed while the analysis takes the frames it can."""
from __future__ import annotations

import threading
import time

import numpy as np

from backend.core.video import Frame
from backend.live.frame_feed import LiveFrameFeed


class FakeCamera:
    """30 fps for `seconds`, then ends like a stopped reader."""

    def __init__(self, seconds: float = 1.0, fps: float = 30.0, fail: bool = False):
        self.n = int(seconds * fps)
        self.interval = 1.0 / fps
        self.fail = fail
        self.stopped = False
        self.info = "camera-info"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def stop(self):
        self.stopped = True

    def __iter__(self):
        for i in range(self.n):
            if self.stopped:
                return
            time.sleep(self.interval)
            yield Frame(index=i, timestamp_sec=i * self.interval, image=np.full((4, 4, 3), i % 256, np.uint8))
        if self.fail:
            raise RuntimeError("stream read failed")


def test_display_sees_every_frame_while_slow_analysis_takes_the_newest():
    shown = []
    camera = FakeCamera(seconds=1.0)
    analysed = []
    with LiveFrameFeed(camera, lambda f: shown.append(f.index)) as feed:
        for frame in feed:
            analysed.append(frame.index)
            time.sleep(0.2)  # a slow pipeline: ~5 analysed frames a second
    assert len(shown) == camera.n                       # display: camera speed
    assert 3 <= len(analysed) <= 8                      # analysis: its own speed
    assert analysed == sorted(analysed) and analysed[-1] >= camera.n - 8  # always the newest


def test_analysis_gets_a_copy_it_can_draw_on():
    seen = {}
    with LiveFrameFeed(FakeCamera(seconds=0.2), lambda f: seen.setdefault(f.index, f.image)) as feed:
        for frame in feed:
            original = seen.get(frame.index)
            frame.image[:] = 255
            assert original is not None and original is not frame.image
            break


def test_passes_reader_attributes_through():
    assert LiveFrameFeed(FakeCamera(), lambda f: None).info == "camera-info"


def test_a_camera_failure_reaches_the_session():
    feed = LiveFrameFeed(FakeCamera(seconds=0.1, fail=True), lambda f: None)
    try:
        with feed:
            for _ in feed:
                pass
    except RuntimeError as exc:
        assert "stream read failed" in str(exc)
    else:
        raise AssertionError("the camera error was swallowed")


def test_exit_stops_the_camera_thread():
    camera = FakeCamera(seconds=10)
    with LiveFrameFeed(camera, lambda f: None) as feed:
        next(iter(feed))
    assert camera.stopped
    assert not any(t.name == "live-frame-feed" and t.is_alive() for t in threading.enumerate())
