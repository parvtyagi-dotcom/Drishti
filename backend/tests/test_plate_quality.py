import cv2
import numpy as np
import pytest

from backend.config import AnprConfig
from backend.modules.anpr.quality import measure, rejection_reason


def plate(width: int, height: int) -> np.ndarray:
    """White plate, black characters - sharp, high-contrast text."""
    img = np.full((height, width, 3), 235, dtype=np.uint8)
    cv2.putText(img, "MH12AB1234", (2, int(height * 0.75)), cv2.FONT_HERSHEY_SIMPLEX,
                height / 40, (10, 10, 10), max(1, height // 20))
    return img


def test_a_clear_plate_is_read():
    assert rejection_reason(plate(140, 45), AnprConfig()) is None


def test_a_tiny_plate_is_skipped():
    # The demo clips' garbage reads all came from crops under ~37px wide.
    assert rejection_reason(plate(36, 12), AnprConfig()) == "too_small"


@pytest.mark.parametrize("w, h", [(60, 60), (140, 130), (300, 30)])
def test_a_plate_seen_at_a_steep_angle_or_cut_off_is_skipped(w, h):
    assert rejection_reason(plate(w, h), AnprConfig()) == "bad_angle"


def test_a_two_line_plate_shape_is_still_read():
    assert rejection_reason(plate(120, 70), AnprConfig()) is None  # aspect ~1.7


def test_only_a_smear_counts_as_blurred():
    clear = plate(140, 45)
    assert rejection_reason(cv2.GaussianBlur(clear, (9, 9), 0), AnprConfig()) is None
    smear = cv2.blur(clear, (25, 25))  # sharpness ~37 against ~227 for the 9x9 Gaussian
    assert rejection_reason(smear, AnprConfig()) == "blurred"


def test_sharpness_does_not_depend_on_brightness():
    clear = plate(140, 45)
    dark = (clear.astype(np.float32) * 0.15).astype(np.uint8)
    assert measure(dark).sharpness == pytest.approx(measure(clear).sharpness, rel=0.25)


def test_empty_crop_is_skipped():
    assert rejection_reason(np.zeros((0, 0, 3), dtype=np.uint8), AnprConfig()) == "too_small"
