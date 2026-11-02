"""Rotating a sideways camera picture upright (phone used as a USB webcam)."""
import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.app import app
from backend.core.video import LiveStreamReader, rotate_frame
from backend.live.camera_manager import camera_manager

client = TestClient(app)


def _clip(path, w=64, h=48, frames=5):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for _ in range(frames):
        img = np.zeros((h, w, 3), np.uint8)
        img[:, : w // 4] = 255  # bright stripe on the left edge, to see where it goes
        out.write(img)
    out.release()


def test_rotate_frame_turns_clockwise():
    img = np.zeros((48, 64, 3), np.uint8)
    img[:, :16] = 255
    turned = rotate_frame(img, 90)
    assert turned.shape == (64, 48, 3)
    assert turned[:16].mean() > 200  # the left edge is now the top
    assert rotate_frame(img, 0) is img
    assert rotate_frame(img, 180).shape == img.shape


def test_reader_reports_and_yields_rotated_size(tmp_path):
    clip = tmp_path / "cam.mp4"
    _clip(clip)
    reader = LiveStreamReader(clip, rotate=90, max_width=None)
    frame = next(iter(reader))
    reader.stop()
    assert frame.image.shape[:2] == (64, 48)
    assert (reader.status.width, reader.status.height) == (48, 64)


def test_unknown_rotation_is_ignored():
    assert LiveStreamReader("0", rotate=45).rotate == 0


def test_rotate_endpoint_sets_rotation_and_rejects_odd_angles(tmp_path):
    clip = tmp_path / "cam.mp4"
    _clip(clip)
    created = client.post("/api/live/cameras", json={"name": "T", "source": str(clip), "start": False}).json()
    cam_id = created["camera_id"]
    try:
        assert client.post(f"/api/live/cameras/{cam_id}/rotate", json={"degrees": 45}).status_code == 422
        resp = client.post(f"/api/live/cameras/{cam_id}/rotate", json={"degrees": 270})
        assert resp.status_code == 200
        assert camera_manager.get(cam_id).camera.rotate == 270
    finally:
        client.delete(f"/api/live/cameras/{cam_id}")
