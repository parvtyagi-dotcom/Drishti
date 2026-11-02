"""Automatic low-light detection + CPU-friendly enhancement.

Shared core capability (like video.py/tracker.py) so any pipeline can opt
into it: analyze_brightness() is cheap enough to run on every frame, and
enhance_frame() only ever touches frames analyze_brightness() classified
LOW_LIGHT - bright footage is never processed.

Backends: "fast" (the default) is CLAHE on LAB's L channel plus adaptive
gamma. "zero_dce" is Zero-DCE++, a small network that predicts a brightening
curve for every pixel - see core/zero_dce.py, and LowLightConfig.backend for
why it is not the default (slower, and worse for detection on the demo
clips). If Zero-DCE++ cannot load (weights file missing, torch unavailable)
the frame is enhanced with "fast" and the reason is reported in
fallback_reason, never a crash.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from backend.config import LowLightConfig

NORMAL, LOW_LIGHT = "NORMAL", "LOW_LIGHT"


@dataclass
class BrightnessStats:
    mean_l: float
    p10_l: float
    std_l: float
    classification: str


@dataclass
class EnhancementInfo:
    classification: str
    enhanced: bool
    backend_used: str | None
    mean_l_before: float
    mean_l_after: float | None = None
    fallback_reason: str | None = None


def analyze_brightness(image: np.ndarray | None, config: LowLightConfig) -> BrightnessStats:
    """Cheap brightness classification: resize to a small working size, LAB
    convert, take mean and 10th-percentile of the L channel. Safe on
    None/empty input (classified NORMAL - nothing to enhance)."""
    if image is None or image.size == 0:
        return BrightnessStats(0.0, 0.0, 0.0, NORMAL)

    h, w = image.shape[:2]
    longer = max(h, w)
    scale = min(1.0, config.sample_max_dim / longer) if longer > 0 else 1.0
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else image

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)
    mean_l = float(l_channel.mean())
    p10_l = float(np.percentile(l_channel, 10))
    std_l = float(l_channel.std())

    is_low = mean_l < config.mean_threshold or p10_l < config.shadow_p10_threshold
    return BrightnessStats(mean_l, p10_l, std_l, LOW_LIGHT if is_low else NORMAL)


def compute_adaptive_gamma(mean_l: float, target_mean: float, gamma_min: float) -> float:
    """Solves 255*(mean_l/255)^gamma ~= target_mean for gamma, clamped to
    [gamma_min, 1.0] so this only ever brightens (gamma<1), never darkens,
    and never over-brightens into a blown-out/noisy result.

    Returns 1.0 (no-op) if mean_l is already >= target_mean or degenerate.
    """
    if mean_l <= 0.0 or mean_l >= target_mean or target_mean <= 0.0 or target_mean >= 255.0:
        return 1.0
    gamma = math.log(target_mean / 255.0) / math.log(mean_l / 255.0)
    return min(1.0, max(gamma_min, gamma))


def _apply_gamma(l_channel: np.ndarray, gamma: float) -> np.ndarray:
    """gamma is used directly as the exponent (out = 255*(in/255)**gamma) -
    matching how compute_adaptive_gamma() solved for it. gamma < 1
    brightens; do NOT invert it here (that would flip the effect to
    darkening, which was a real bug caught by test_dark_frame_triggers_
    enhancement producing an all-black frame)."""
    if abs(gamma - 1.0) < 1e-3:
        return l_channel
    lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(l_channel, lut)


def enhance_fast(image: np.ndarray, config: LowLightConfig) -> np.ndarray:
    """CLAHE on the L channel (contrast, preserves color in a/b channels)
    followed by adaptive gamma brightening if still dark. Cheap: one CLAHE
    call and a 256-entry LUT, both O(pixels)."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=config.clahe_clip_limit, tileGridSize=(config.clahe_tile_size, config.clahe_tile_size))
    l_eq = clahe.apply(l_channel)

    mean_after_clahe = float(l_eq.mean())
    gamma = compute_adaptive_gamma(mean_after_clahe, config.gamma_target_mean, config.gamma_min)
    l_final = _apply_gamma(l_eq, gamma)

    enhanced_lab = cv2.merge([l_final, a_channel, b_channel])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


_zero_dce_failures: dict[str, str] = {}


def _enhance_zero_dce(image: np.ndarray, config: LowLightConfig) -> np.ndarray:
    """Raises if the model cannot be loaded; process_frame() falls back."""
    from backend.core.zero_dce import shared_model

    return shared_model(config.zero_dce_model_path, config.zero_dce_work_dim).enhance(image)


def process_frame(image: np.ndarray | None, config: LowLightConfig) -> tuple[np.ndarray, EnhancementInfo]:
    """The single entry point pipelines should call: classify, and enhance
    only if warranted. Returns (possibly-enhanced image, info). Never
    mutates the input array in place.
    """
    if image is None or image.size == 0:
        return image, EnhancementInfo(NORMAL, False, None, 0.0)

    stats = analyze_brightness(image, config)
    if not config.enabled or stats.classification == NORMAL:
        return image, EnhancementInfo(stats.classification, False, None, stats.mean_l)

    backend = config.backend
    fallback_reason = None
    if backend not in ("fast", "zero_dce"):
        raise ValueError(f"unknown lowlight backend: {config.backend!r}")

    enhanced = None
    if backend == "zero_dce":
        # A model that failed to load once is not retried on every frame.
        key = str(config.zero_dce_model_path)
        fallback_reason = _zero_dce_failures.get(key)
        if fallback_reason is None:
            try:
                enhanced = _enhance_zero_dce(image, config)
            except Exception as exc:
                fallback_reason = _zero_dce_failures[key] = f"zero_dce unavailable ({exc}); used 'fast' instead"
        if enhanced is None:
            backend = "fast"
    if enhanced is None:
        enhanced = enhance_fast(image, config)
    after_stats = analyze_brightness(enhanced, config)
    return enhanced, EnhancementInfo(
        classification=stats.classification, enhanced=True, backend_used=backend,
        mean_l_before=stats.mean_l, mean_l_after=after_stats.mean_l, fallback_reason=fallback_reason,
    )
