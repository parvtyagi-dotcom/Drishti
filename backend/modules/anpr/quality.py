"""Which plate crops are worth handing to OCR.

The plate OCR reads garbage off a crop too small to resolve characters, and
it does so CONFIDENTLY - so its confidence score cannot be the filter.
Measured on the two ANPR demo clips (371 plate crops, every third frame):
every crop narrower than ~37px produced a wrong string, several at 90%+
confidence ("AZT525" 90.4, "XZY52F" 93.6, "M93PJX" 99.9), while every
correct TEST123 came from a crop at least 75px wide. Those wrong strings
are not harmless: three agreeing ones confirm a plate that does not exist.

Sharpness gets only a floor, not a real threshold: the same OCR model kept
reading TEST123 on 53 of 90 good crops even under a 9x9 Gaussian blur (55
of 90 unblurred), so a strict blur gate would throw away readable frames.
It is measured on the crop scaled to a fixed height and stretched to full
contrast, so neither plate size nor a dark scene reads as "blurry".

Aspect ratio catches plates seen at a steep angle or cut off at the crop
edge: full plates measured 2.6-3.3 wide-to-high, the partial reads ("TC",
"123") 1.3-1.7. The floor sits below two-line Indian plates (~1.6-2.0).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

_SHARPNESS_HEIGHT = 48


@dataclass(frozen=True)
class PlateQuality:
    width: int
    aspect: float
    sharpness: float


def measure(plate_crop: np.ndarray) -> PlateQuality:
    h, w = plate_crop.shape[:2]
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY) if plate_crop.ndim == 3 else plate_crop
    scaled_w = max(1, round(w * _SHARPNESS_HEIGHT / max(1, h)))
    gray = cv2.resize(gray, (scaled_w, _SHARPNESS_HEIGHT), interpolation=cv2.INTER_AREA)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return PlateQuality(width=w, aspect=w / max(1, h), sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()))


def rejection_reason(plate_crop: np.ndarray, cfg) -> str | None:
    """None if the crop is worth reading, else why not: 'too_small',
    'bad_angle' or 'blurred'. cfg is an AnprConfig."""
    if plate_crop is None or plate_crop.size == 0:
        return "too_small"
    q = measure(plate_crop)
    if q.width < cfg.min_plate_width_px:
        return "too_small"
    if not cfg.min_plate_aspect <= q.aspect <= cfg.max_plate_aspect:
        return "bad_angle"
    if q.sharpness < cfg.min_plate_sharpness:
        return "blurred"
    return None
