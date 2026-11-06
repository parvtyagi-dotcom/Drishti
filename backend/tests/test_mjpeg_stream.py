"""The MJPEG feed must not pass off a dead camera as a live one.

wait_for_frame returns the last frame again when nothing new arrives within
its timeout. The stream used to treat that repeat as fresh, so a camera that
had stopped kept showing its last picture indefinitely - the "frozen video"
operators saw - instead of the status card explaining what happened.
"""
from __future__ import annotations

import time

from backend.api import live

FRAME = b"real-frame"


class _StalledSession:
    """Delivers one real frame, then only ever repeats it."""

    is_running = True

    def wait_for_frame(self, last_seq):
        time.sleep(0.005)
        return FRAME, 1


class _LiveSession:
    """Delivers a new frame on every call."""

    is_running = True

    def __init__(self):
        self.seq = 0

    def wait_for_frame(self, last_seq):
        time.sleep(0.005)
        self.seq += 1
        return FRAME, self.seq


def _collect(session, seconds, monkeypatch):
    monkeypatch.setattr(live, "_STALE_FEED_AFTER_SEC", 0.05)
    monkeypatch.setattr(live, "_STATUS_FRAME_INTERVAL_SEC", 0.0)
    monkeypatch.setattr(live, "_stall_status", lambda session, quiet: ("Feed stalled", ""))
    monkeypatch.setattr(live, "_status_frame", lambda headline, detail: f"STATUS:{headline}".encode())
    parts = []
    deadline = time.monotonic() + seconds
    for part in live._mjpeg_parts(session, fps=60):
        parts.append(part)
        if time.monotonic() > deadline:
            break
    return parts


def test_a_stalled_camera_switches_to_a_status_card(monkeypatch):
    parts = _collect(_StalledSession(), 0.3, monkeypatch)

    frames = [p for p in parts if FRAME in p]
    statuses = [p for p in parts if b"STATUS:Feed stalled" in p]
    assert len(frames) == 1, "the same frame was re-sent as if it were new"
    assert statuses, "a stalled feed never told the viewer it had stalled"


def test_a_live_camera_keeps_streaming_frames(monkeypatch):
    parts = _collect(_LiveSession(), 0.3, monkeypatch)

    assert any(FRAME in p for p in parts)
    assert not any(b"STATUS:" in p for p in parts), "a healthy feed was reported as stalled"
