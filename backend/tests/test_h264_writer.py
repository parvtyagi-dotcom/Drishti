"""Output videos are written as browser-playable H.264, not the mp4v fallback.

The bundled OpenH264 DLL was never on the Windows DLL search path, so every
H.264 write quietly fell back to the next codec (logging a misleading
"Incorrect library version loaded"). backend.config now puts models/ on PATH.
"""
import os

import cv2
import numpy as np
import pytest

from backend.config import OPENH264_DLL_PATH

pytestmark = pytest.mark.skipif(
    os.name != "nt" or not OPENH264_DLL_PATH.is_file(), reason="Windows with the bundled OpenH264 DLL only",
)


def test_avc1_writer_opens(tmp_path):
    path = tmp_path / "probe.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), 30, (64, 64))
    try:
        assert writer.isOpened(), "H.264 (avc1) writer did not open - OpenH264 DLL not found?"
        for _ in range(5):
            writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    finally:
        writer.release()
    assert path.stat().st_size > 0
