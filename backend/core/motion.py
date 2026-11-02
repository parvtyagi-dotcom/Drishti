"""Motion check: wake the AI only when something in view changes.

A border camera shows an empty scene almost all the time. Running the person
and vehicle detector on those frames is wasted work, so a live camera first
compares each frame with the previous one at thumbnail size, which costs well
under a millisecond. The detector runs while there is movement, for a few
seconds after it, while anything is still being tracked (someone standing
still), and once every few seconds regardless, as a safety net for movement
too slow or small for the frame comparison to catch.

Motion alone never raises an alert: wind, shadows and rain all move pixels.
It only decides when the detector should look.
"""
from __future__ import annotations

import cv2
import numpy as np


class MotionGate:
    def __init__(self, width: int = 160, pixel_delta: int = 25, min_changed: float = 0.002,
                 hold_sec: float = 3.0, check_every_sec: float = 5.0):
        self.width = width
        self.pixel_delta = pixel_delta
        # Share of the thumbnail that must change to count as motion: 0.2% is
        # roughly a person far down a fence line.
        self.min_changed = min_changed
        self.hold_sec = hold_sec
        self.check_every_sec = check_every_sec
        self._prev: np.ndarray | None = None
        self._last_motion = float("-inf")
        self._last_run = float("-inf")
        self.frames = 0
        self.frames_run = 0

    def _thumb(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        small = cv2.resize(image, (self.width, max(1, int(h * self.width / w))), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)

    def should_run(self, image: np.ndarray, now: float, tracking: bool) -> bool:
        """Whether the detector should look at this frame."""
        self.frames += 1
        thumb = self._thumb(image)
        if self._prev is not None and self._prev.shape == thumb.shape:
            changed = np.count_nonzero(cv2.absdiff(thumb, self._prev) > self.pixel_delta) / thumb.size
            if changed >= self.min_changed:
                self._last_motion = now
        # The first frame is looked at by the periodic check below.
        self._prev = thumb
        run = (tracking or now - self._last_motion <= self.hold_sec
               or now - self._last_run >= self.check_every_sec)
        if run:
            self._last_run = now
            self.frames_run += 1
        return run

    @property
    def idle_share(self) -> float:
        """Share of frames the detector skipped."""
        return 0.0 if not self.frames else 1 - self.frames_run / self.frames
