"""Fog mode: clear haze from a frame before the detectors look at it.

Fog is the border's worst season for cameras: BSF has said gadgets lose much
of their capability from December to February, and smugglers time drone and
foot crossings for it. This uses the dark channel prior (He, Sun and Tang,
CVPR 2009): in a clear outdoor picture almost every small patch has some
pixel that is nearly black in at least one colour, so the grey lift where
that is not true measures how much fog is in the way, and can be removed.

Plain OpenCV, no model, a few milliseconds a frame. It only touches frames
that are actually hazy, so clear video passes through unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DehazeConfig:
    # Fog lifts the darkest pixels toward grey and flattens contrast. A frame
    # counts as foggy when its darkest 5% are at least this bright (0-1)...
    min_black_level: float = 0.4
    # ...and its brightness spread is at most this. Both are needed: a
    # bright sky alone lifts neither the blacks of the ground nor flattens it.
    max_contrast: float = 0.13
    # How much fog to remove; below 1 keeps a trace so depth still reads.
    strength: float = 0.9
    min_transmission: float = 0.15
    work_width: int = 320
    patch: int = 9


def fog_measures(image: np.ndarray) -> tuple[float, float]:
    """(black level, contrast) of a frame, both 0-1."""
    gray = cv2.cvtColor(cv2.resize(image, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    g = gray.astype(np.float32) / 255.0
    return float(np.percentile(g, 5)), float(g.std())


@dataclass
class DehazeInfo:
    haze: float  # black level: how far fog has lifted the darkest pixels
    foggy: bool
    cleared: bool


def _dark_channel(img: np.ndarray, patch: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    return cv2.erode(img.min(axis=2), kernel)


def dehaze(image: np.ndarray, config: DehazeConfig | None = None) -> tuple[np.ndarray, DehazeInfo]:
    """Return the cleared frame (or the original when it is not foggy)."""
    cfg = config or DehazeConfig()
    black, contrast = fog_measures(image)
    if black < cfg.min_black_level or contrast > cfg.max_contrast:
        return image, DehazeInfo(round(black, 3), False, False)
    h, w = image.shape[:2]
    small = cv2.resize(image, (cfg.work_width, max(1, int(h * cfg.work_width / w))),
                       interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    dark = _dark_channel(small, cfg.patch)

    # The fog's colour: the brightest pixels among the haziest 0.1%.
    flat = dark.reshape(-1)
    top = np.argsort(flat)[-max(1, flat.size // 1000):]
    airlight = np.clip(small.reshape(-1, 3)[top].max(axis=0), 0.5, 1.0)

    transmission = 1.0 - cfg.strength * _dark_channel(small / airlight, cfg.patch)
    # Smooth so object edges do not get halos, then scale up to full size.
    transmission = cv2.blur(transmission, (cfg.patch * 2 + 1, cfg.patch * 2 + 1))
    transmission = cv2.resize(transmission, (w, h), interpolation=cv2.INTER_LINEAR)
    transmission = np.maximum(transmission, cfg.min_transmission)[..., None]

    frame = image.astype(np.float32) / 255.0
    cleared = (frame - airlight) / transmission + airlight
    out = (np.clip(cleared, 0, 1) * 255).astype(np.uint8)
    return out, DehazeInfo(round(black, 3), True, True)
