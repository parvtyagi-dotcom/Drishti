"""Sentinel: notice when someone interferes with a camera.

The easiest way past a camera is to blind it: a hand or cloth over the lens,
a smear of mud or spray paint, or a push that points it at the sky. The
camera keeps streaming, so nothing else in the system notices - the feed is
simply useless. This watches each camera's raw frames for exactly that.

It learns what the camera normally sees, then reports a sudden change that
lasts a few seconds:

  covered  the image goes nearly uniform (lens blocked or blacked out)
  blurred  the image loses most of its detail (lens smeared or sprayed)
  moved    the scene no longer matches what the camera was looking at

The baseline follows slow changes, so dusk or a passing cloud never looks
like tampering, while a change within seconds does. One sample per second
on a thumbnail costs almost nothing next to detection.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

COVERED, BLURRED, MOVED = "covered", "blurred", "moved"


@dataclass
class TamperConfig:
    sample_every_sec: float = 1.0
    # Samples of normal view needed before anything is judged.
    warmup_samples: int = 8
    # Covered: brightness spread falls below this, or below this share of normal.
    covered_std: float = 8.0
    covered_ratio: float = 0.25
    # Blurred: sharpness falls below this share of normal.
    blurred_ratio: float = 0.2
    # Moved: similarity to the usual scene falls below this (1.0 = identical).
    moved_similarity: float = 0.35
    # ...and at most this share of the frame's tiles still look the same. A
    # person standing close to the lens covers the middle but leaves the edges
    # of the scene in view; a camera pushed to point elsewhere changes them all.
    moved_max_unchanged: float = 0.25
    # A tile counts as unchanged when its mean difference from the usual
    # scene (in normalised brightness units) is below this.
    tile_unchanged_diff: float = 0.6
    # Moved must hold longer than covered/blurred: someone lingering in front
    # of the camera is common, a real push leaves the view changed for good.
    moved_persist_sec: float = 5.0
    # A condition must hold this long before it is reported, so a bird or a
    # person walking past the lens never raises it.
    persist_sec: float = 3.0
    # How quickly the idea of "normal" follows slow change, per sample.
    adapt_rate: float = 0.03


@dataclass
class TamperReading:
    kind: str
    detail: str
    baseline: dict
    current: dict


def measure(image: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Brightness, spread, sharpness, and a small normalised thumbnail."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32)
    sharpness = float(cv2.Laplacian(small, cv2.CV_32F).var())
    thumb = cv2.resize(small, (32, 18), interpolation=cv2.INTER_AREA)
    thumb = (thumb - thumb.mean()) / (thumb.std() + 1e-6)
    return float(small.mean()), float(small.std()), sharpness, thumb


class TamperMonitor:
    def __init__(self, config: TamperConfig | None = None, clock=time.monotonic):
        self.config = config or TamperConfig()
        self.clock = clock
        self._last_sample = float("-inf")
        self._samples = 0
        self._std = self._sharp = 0.0
        self._scene: np.ndarray | None = None
        self._suspect: str | None = None
        self._suspect_since = 0.0
        self._reported: str | None = None
        self._clear_since: float | None = None

    @property
    def tampered(self) -> str | None:
        return self._reported

    def observe(self, image: np.ndarray) -> TamperReading | str | None:
        """Look at one raw frame (at most once per sample interval).

        Returns a TamperReading when tampering has just been confirmed,
        "restored" when a reported tampering has just cleared, else None.
        """
        now = self.clock()
        if now - self._last_sample < self.config.sample_every_sec:
            return None
        self._last_sample = now
        mean, std, sharp, thumb = measure(image)
        cfg = self.config

        if self._samples < cfg.warmup_samples:
            self._learn(std, sharp, thumb, rate=1.0 / (self._samples + 1))
            self._samples += 1
            return None

        similarity = float((thumb * self._scene).mean()) if self._scene is not None else 1.0
        if std < max(cfg.covered_std, cfg.covered_ratio * self._std):
            kind = COVERED
        elif sharp < cfg.blurred_ratio * self._sharp:
            kind = BLURRED
        elif similarity < cfg.moved_similarity and self._unchanged_share(thumb) <= cfg.moved_max_unchanged:
            kind = MOVED
        else:
            kind = None
        current = {"mean": round(mean, 1), "spread": round(std, 1), "sharpness": round(sharp, 1), "similarity": round(similarity, 2)}

        if kind is None:
            self._suspect = None
            # Only a normal view teaches what normal is.
            self._learn(std, sharp, thumb, rate=cfg.adapt_rate)
            if self._reported is not None:
                if self._clear_since is None:
                    self._clear_since = now
                elif now - self._clear_since >= cfg.persist_sec:
                    self._reported = None
                    self._clear_since = None
                    return "restored"
            return None

        self._clear_since = None
        if kind != self._suspect:
            self._suspect, self._suspect_since = kind, now
            return None
        persist = cfg.moved_persist_sec if kind == MOVED else cfg.persist_sec
        if self._reported is None and now - self._suspect_since >= persist:
            self._reported = kind
            baseline = {"spread": round(self._std, 1), "sharpness": round(self._sharp, 1)}
            return TamperReading(kind=kind, detail=_DETAIL[kind], baseline=baseline, current=current)
        return None

    def _unchanged_share(self, thumb: np.ndarray) -> float:
        """Share of 4x3 tiles across the frame that still match the usual scene."""
        diff = np.abs(thumb - self._scene)
        tiles = [diff[r:r + 6, c:c + 8].mean() for r in range(0, 18, 6) for c in range(0, 32, 8)]
        return sum(d < self.config.tile_unchanged_diff for d in tiles) / len(tiles)

    def _learn(self, std: float, sharp: float, thumb: np.ndarray, rate: float) -> None:
        self._std += rate * (std - self._std)
        self._sharp += rate * (sharp - self._sharp)
        self._scene = thumb.copy() if self._scene is None else self._scene + rate * (thumb - self._scene)
        # Keep the reference normalised, or similarity drifts as it blends.
        self._scene = (self._scene - self._scene.mean()) / (self._scene.std() + 1e-6)


_DETAIL = {
    COVERED: "view blocked or blacked out",
    BLURRED: "image suddenly out of focus",
    MOVED: "camera now points somewhere else",
}


class TappedReader:
    """Passes a camera's frames through unchanged, showing each raw frame to
    a callback first. The analysis pipeline draws boxes onto the frames it
    receives, so tamper checks need the frame before that happens."""

    def __init__(self, reader, tap):
        self._reader = reader
        self._tap = tap

    def __getattr__(self, name):
        return getattr(self._reader, name)

    def __iter__(self):
        for frame in self._reader:
            self._tap(frame.image)
            yield frame

    def __enter__(self):
        self._reader.__enter__()
        return self

    def __exit__(self, *exc):
        return self._reader.__exit__(*exc)
