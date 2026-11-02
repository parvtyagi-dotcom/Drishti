"""Read a live camera at its own speed, independent of the analysis.

The analysis pipeline pulls one frame at a time and spends as long as it
needs on it - with Person ID and two faces close to the camera that is about
a fifth of a second on a laptop CPU. Pulled straight from the camera, the
video an operator watches then moves at that rate too: five jerky frames a
second from a camera delivering thirty.

LiveFrameFeed puts a thread in between. The thread reads the camera at full
rate and hands every frame to a display callback; the analysis takes the
newest frame whenever it is ready for another and simply never sees the ones
it was too slow for. Live analysis already discarded those frames (see
LiveStreamReader's catch-up), so what is analysed does not change - only
what is shown does.
"""
from __future__ import annotations

import threading
from typing import Callable

from backend.core.video import Frame


class LiveFrameFeed:
    WAIT_SLICE_SEC = 0.5

    def __init__(self, reader, on_frame: Callable[[Frame], None]):
        self._reader = reader
        self._on_frame = on_frame
        self._cond = threading.Condition()
        self._latest: Frame | None = None
        self._seq = 0
        self._done = False
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None

    def __getattr__(self, name):
        # info, metrics, status and the rest come from the real reader.
        return getattr(self._reader, name)

    def __enter__(self) -> "LiveFrameFeed":
        self._reader.__enter__()
        self._thread = threading.Thread(target=self._grab, name="live-frame-feed", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._reader.stop()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return self._reader.__exit__(*exc)

    def _grab(self) -> None:
        try:
            for frame in self._reader:
                with self._cond:
                    self._latest = frame
                    self._seq += 1
                    self._cond.notify_all()
                try:
                    self._on_frame(frame)
                except Exception:  # a display failure must not stop the camera
                    pass
        except BaseException as exc:
            self._error = exc
        finally:
            with self._cond:
                self._done = True
                self._cond.notify_all()

    def __iter__(self):
        last = 0
        while True:
            with self._cond:
                while self._seq == last and not self._done:
                    self._cond.wait(self.WAIT_SLICE_SEC)
                if self._seq == last:
                    break
                frame, last = self._latest, self._seq
            # The pipeline draws on what it is given; the display thread is
            # still using the original.
            yield Frame(index=frame.index, timestamp_sec=frame.timestamp_sec, image=frame.image.copy())
        if self._error is not None:
            raise self._error
