"""A live camera must survive its own analysis failing.

Before this, any exception inside the analysis pipeline (a model error, a
camera that was slow to open, a bad frame) ended the capture thread for every
source except a phone and an injected fault. The feed froze on its last
frame and nothing brought it back until an operator pressed Configure. These
pin the recovery behaviour: retry with backoff, fall back to plain video when
the configuration itself is unusable, and never leave a stale frame posing as
a live one.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from backend.core.video import OFFLINE, ONLINE, STOPPED, Frame
from backend.live.camera_manager import AnalysisSpec, Camera, LiveSession
from backend.orchestrator import OrchestratorError


class _Status:
    def __init__(self):
        self.state = ONLINE
        self.error = None

    def as_dict(self) -> dict:
        return {"state": self.state, "error": self.error, "frame_age_sec": 0.0, "reconnect_count": 0}


class _FakeReader:
    """Yields small frames until stopped, like a camera that never ends."""

    def __init__(self):
        self.status = _Status()
        self._stop = threading.Event()

    def __iter__(self):
        index = 0
        while not self._stop.is_set():
            yield Frame(index=index, timestamp_sec=index / 10, image=np.zeros((8, 8, 3), np.uint8))
            index += 1
            time.sleep(0.01)
        self.status.state = STOPPED

    def stop(self):
        self._stop.set()


class _ScriptedSession(LiveSession):
    """A LiveSession whose analysis run follows a script of failures."""

    RESTART_BACKOFF_MIN_SEC = 0.01
    RESTART_BACKOFF_MAX_SEC = 0.02

    def __init__(self, failures, **kwargs):
        self.failures = list(failures)
        self.analysis_runs = 0
        self.readers_built = 0
        camera = Camera(camera_id="cam1", name="Gate", source="rtsp://camera.invalid/stream")
        super().__init__(camera, analysis=AnalysisSpec(behavior=True), **kwargs)

    def _new_reader(self):
        self.readers_built = getattr(self, "readers_built", 0) + 1
        return _FakeReader()

    def _run_with_analysis(self):
        self.analysis_runs += 1
        if self.failures:
            raise self.failures.pop(0)
        for frame in self.reader:
            self._publish(frame.image)
            if self._stopping:
                break


def _wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_a_crash_in_analysis_restarts_the_camera_instead_of_ending_it():
    session = _ScriptedSession([RuntimeError("detector exploded"), RuntimeError("again")])
    session.start()
    try:
        assert _wait_until(lambda: session.analysis_runs >= 3), "the session gave up after a crash"
        assert session.is_running
        # Recovered: frames flow again from a freshly built reader.
        _jpeg, seq = session.wait_for_frame(-1)
        assert _wait_until(lambda: session.wait_for_frame(seq)[1] > seq)
        assert session.readers_built >= 3
    finally:
        session.stop()


def test_a_crash_clears_the_last_frame_so_viewers_are_not_shown_a_frozen_picture():
    gate = threading.Event()

    class _CrashAfterAFrame(_ScriptedSession):
        RESTART_BACKOFF_MIN_SEC = 5.0
        RESTART_BACKOFF_MAX_SEC = 5.0

        def _run_with_analysis(self):
            self.analysis_runs += 1
            self._publish(np.zeros((8, 8, 3), np.uint8))
            gate.wait(2.0)
            raise RuntimeError("tracker failed")

    session = _CrashAfterAFrame([])
    session.start()
    try:
        assert _wait_until(lambda: session.wait_for_frame(-1)[0] is not None)
        gate.set()
        assert _wait_until(lambda: session.wait_for_frame(-1)[0] is None), (
            "a crashed camera kept serving its last frame as if it were live"
        )
        assert session.status_dict()["state"] == OFFLINE
    finally:
        session.stop()


def test_an_unusable_configuration_falls_back_to_plain_video():
    """A reference photo with no face will never start, however often it is
    retried. The camera keeps showing video and says why analysis is off."""
    session = _ScriptedSession([OrchestratorError("no usable face found in any reference photo")])
    session.start()
    try:
        _jpeg, seq = session.wait_for_frame(-1)
        assert _wait_until(lambda: session.wait_for_frame(seq)[1] > seq), "video stopped with the analysis"
        assert session.analysis_runs == 1, "a configuration error was retried pointlessly"
        assert "no usable face" in (session.analysis_error or "")
        assert session.is_running
    finally:
        session.stop()


def test_a_video_only_failure_backs_off_instead_of_spinning():
    """Once analysis is off, a FileNotFoundError from the video itself is an
    ordinary crash: it must wait out the backoff, not loop flat out."""

    class _VideoAlsoFails(_ScriptedSession):
        RESTART_BACKOFF_MIN_SEC = 0.2
        RESTART_BACKOFF_MAX_SEC = 0.2

        def __init__(self):
            super().__init__([OrchestratorError("no usable face")])
            self.video_runs = 0

        def _run_video_only(self):
            self.video_runs += 1
            raise FileNotFoundError("clip.mp4")

    session = _VideoAlsoFails()
    session.start()
    try:
        time.sleep(0.5)
        assert 1 <= session.video_runs <= 4, f"video-only failures retried {session.video_runs} times in 0.5 s"
        # The later video crash must not wipe out why analysis is off.
        assert _wait_until(lambda: "analysis disabled" in (session.analysis_error or ""))
    finally:
        session.stop()


def test_stop_interrupts_a_restart_backoff():
    class _SlowBackoff(_ScriptedSession):
        RESTART_BACKOFF_MIN_SEC = 30.0
        RESTART_BACKOFF_MAX_SEC = 30.0

    session = _SlowBackoff([RuntimeError("boom")] * 5)
    session.start()
    assert _wait_until(lambda: session.analysis_runs >= 1)
    started = time.monotonic()
    session.stop(timeout=5.0)
    assert time.monotonic() - started < 2.0, "stop() waited out a 30 s backoff"
    assert not session.is_running


def test_a_finite_file_that_ends_normally_is_not_restarted():
    class _EndsAfterAFewFrames(_ScriptedSession):
        def _run_with_analysis(self):
            self.analysis_runs += 1
            self._publish(np.zeros((8, 8, 3), np.uint8))

    session = _EndsAfterAFewFrames([])
    session.start()
    assert _wait_until(lambda: not session.is_running), "a finished recording was looped forever"
    assert session.analysis_runs == 1
