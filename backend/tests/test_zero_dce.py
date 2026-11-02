"""Zero-DCE++ on the CPU: the fast table-lookup path must match the paper's
curve, and the model must genuinely brighten a dark real frame."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from backend.config import REPO_ROOT, ZERO_DCE_MODEL_PATH
from backend.core.zero_dce import ALPHA_LEVELS, ZeroDCE, apply_curves, build_curve_table

pytestmark = pytest.mark.skipif(not ZERO_DCE_MODEL_PATH.is_file(), reason="zero_dce_pp.pth not present")
DARK_CLIP = REPO_ROOT / "uploads" / "smoke_person_id_dark.mp4"


def _frame() -> np.ndarray:
    cap = cv2.VideoCapture(str(DARK_CLIP))
    ok, frame = cap.read()
    cap.release()
    assert ok
    return frame


def _mean_l(image: np.ndarray) -> float:
    return float(cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0].mean())


def test_curve_keeps_values_in_range_and_brightens_for_negative_alpha():
    # The released weights were trained with the authors' code's form,
    # x + a*(x^2 - x): the paper's x + a*x*(1 - x) with the sign of a flipped,
    # so here a NEGATIVE alpha is what brightens.
    x = np.linspace(0, 1, 101, dtype=np.float32)
    up = apply_curves(x.copy(), np.full_like(x, -0.8))
    down = apply_curves(x.copy(), np.full_like(x, 0.8))
    assert up.min() >= 0 and up.max() <= 1 and down.min() >= 0 and down.max() <= 1
    assert np.all(up[1:-1] > x[1:-1]) and np.all(down[1:-1] < x[1:-1])
    assert up[0] == 0 and up[-1] == pytest.approx(1)  # black stays black, white stays white


def test_curve_table_matches_the_curve():
    table = build_curve_table().reshape(ALPHA_LEVELS, 256)
    level = 100
    alpha = -1 + 2 * level / (ALPHA_LEVELS - 1)
    pixels = np.arange(256, dtype=np.float32) / 255
    exact = apply_curves(pixels.copy(), np.full_like(pixels, alpha)) * 255
    assert np.abs(table[level].astype(np.float32) - exact).max() <= 0.5 + 1e-3


@pytest.mark.parametrize("use_openvino", [True, False])
def test_dark_frame_is_brightened_and_close_to_the_exact_computation(use_openvino):
    frame = _frame()
    model = ZeroDCE(ZERO_DCE_MODEL_PATH, use_openvino=use_openvino)
    out = model.enhance(frame)
    assert out.shape == frame.shape and out.dtype == np.uint8
    assert _mean_l(out) > _mean_l(frame) + 40  # measured: 13.8 -> ~85

    exact = apply_curves(frame.astype(np.float32) / 255, model.alpha_map(frame)) * 255
    diff = np.abs(out.astype(np.float32) - exact)
    assert diff.mean() < 1.5  # the table rounds alpha to 1/128 of its range


def test_openvino_and_torch_backends_agree():
    frame = _frame()
    a = ZeroDCE(ZERO_DCE_MODEL_PATH, use_openvino=True)
    b = ZeroDCE(ZERO_DCE_MODEL_PATH, use_openvino=False)
    if a.backend != "openvino":
        pytest.skip("openvino not installed")
    assert np.abs(a.alpha_map(frame) - b.alpha_map(frame)).max() < 1e-4


def test_any_frame_size_works():
    model = ZeroDCE(ZERO_DCE_MODEL_PATH)
    for shape in [(64, 64, 3), (37, 91, 3), (1706, 960, 3)]:
        dark = np.full(shape, 15, dtype=np.uint8)
        assert model.enhance(dark).shape == shape
