"""Routes for adding a webcam plugged into this machine, without typing its index.

    GET  /api/live/webcams         webcams that opened, each with a snapshot

OpenCV has no call that lists cameras, so the scan tries the first few device
indexes and keeps the ones that deliver a frame. A snapshot comes back with
each, because an index means nothing to an operator - the picture is how they
tell the laptop's own camera from the USB one pointing at the gate.
"""
from __future__ import annotations

import base64
import os
import threading
import time

from fastapi import APIRouter, Depends

from backend.api.auth import ADMIN, OPERATOR, User, require_role
from backend.api.ownership import current_owner
from backend.core.video import WEBCAM_OPEN_LOCK
from backend.live.camera_manager import camera_manager

router = APIRouter(prefix="/api/live/webcams", tags=["live"])

CAN_ADD = Depends(require_role(ADMIN, OPERATOR))

MAX_INDEX = 6
# A probe that has not delivered a frame by now is given up on. Media
# Foundation was measured hanging indefinitely on a USB webcam.
PROBE_TIMEOUT_SEC = 6.0
THUMB_WIDTH = 320


def _grab(cap) -> "np.ndarray | None":
    # The first frames from many webcams are black while exposure settles.
    frame = None
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        ok, img = cap.read()
        if ok and img is not None:
            frame = img
            if img.mean() > 8:
                break
    return frame


def _probe(index: int, found: dict[int, dict]) -> None:
    import cv2

    frame = None
    virtual = False
    if os.name == "nt":
        # Same order and lock as the camera itself (core/video.py): DirectShow
        # first, Media Foundation only for devices DirectShow cannot open.
        with WEBCAM_OPEN_LOCK:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
            # Real USB webcams all open with DirectShow; what opens only with
            # MSMF here has been a phone-webcam app (DroidCam), whose picture
            # is a "Start DroidCam" card until the phone connects.
            virtual = True
    else:
        cap = cv2.VideoCapture(index, cv2.CAP_ANY)
    try:
        if cap.isOpened():
            frame = _grab(cap)
    finally:
        cap.release()
    if frame is None:
        return
    h, w = frame.shape[:2]
    thumb = cv2.resize(frame, (THUMB_WIDTH, max(1, int(h * THUMB_WIDTH / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    found[index] = {
        "index": index,
        "width": w,
        "height": h,
        "snapshot": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None,
        "in_use_by": None,
        "virtual": virtual,
    }


@router.get("")
def scan(_: User = CAN_ADD) -> dict:
    # A webcam already feeding a camera here is reported, not reopened:
    # grabbing it a second time can stall or steal the running feed. Every
    # account's cameras count - a webcam is one physical device.
    in_use: dict[int, str] = {}
    token = current_owner.set(None)
    try:
        sessions = camera_manager.list()
    finally:
        current_owner.reset(token)
    for session in sessions:
        source = session.camera.source.strip()
        if source.isdigit():
            in_use[int(source)] = session.camera.name

    # One device at a time: probing them in parallel made DirectShow fail on
    # webcams that work, which is how two USB webcams went missing.
    found: dict[int, dict] = {}
    for index in range(MAX_INDEX):
        if index in in_use:
            found[index] = {"index": index, "width": 0, "height": 0, "snapshot": None, "in_use_by": in_use[index], "virtual": False}
            continue
        t = threading.Thread(target=_probe, args=(index, found), daemon=True)
        t.start()
        t.join(PROBE_TIMEOUT_SEC)
    return {"webcams": [found[i] for i in sorted(found)]}
