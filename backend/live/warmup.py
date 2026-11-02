"""Load the detection and recognition models once when the server starts.

Each camera still builds its own model objects (tracker state cannot be
shared), but the first one used to pay for importing PyTorch, reading the
weights from disk and the first inference all at once - tens of seconds in
which the first camera showed nothing. Doing that here, in the background,
leaves later loads warm. It costs a few seconds of boot time.
"""
from __future__ import annotations

import logging
import threading

from backend.config import MODELS_DIR, YOLO_MODEL_PATH

logger = logging.getLogger("backend.live.warmup")

_ONNX = ("arcface_w600k_r50.onnx", "osnet_x0_25_msmt17.onnx")


def _warm() -> None:
    try:
        import numpy as np
        from ultralytics import YOLO

        YOLO(str(YOLO_MODEL_PATH))(np.zeros((480, 640, 3), np.uint8), verbose=False)
        import onnxruntime as ort

        for name in _ONNX:
            path = MODELS_DIR / name
            if path.is_file():
                ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        logger.info("models warmed up")
    except Exception:  # a failed warm-up only means the first camera is slower
        logger.exception("model warm-up failed")


_lock = threading.Lock()
_done = False


def warm_up_models() -> None:
    """Runs synchronously, before the other startup threads: torchvision
    imported from two threads at once deadlocked Python's import lock.
    scripts/serve.py starts two servers in one process, so the second waits
    for the first rather than warming up again."""
    global _done
    with _lock:
        if not _done:
            _warm()
            _done = True
