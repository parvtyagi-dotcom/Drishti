"""Ground calibration: pixels in the picture -> metres on the ground."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from backend.config import BehaviorConfig
from backend.core import calibration as cal
from backend.core.calibration import CalibrationError, GroundCalibration
from backend.modules.behavior.state import BehaviorMonitor

W, H = 1000, 600

# A 4 m x 10 m rectangle on the ground, seen in perspective: the near edge
# (bottom of the picture) is wide, the far edge narrow.
TRAPEZOID = [(0.30, 0.90), (0.70, 0.90), (0.58, 0.40), (0.42, 0.40)]


def ground_rect() -> GroundCalibration:
    return GroundCalibration(points=TRAPEZOID, width_m=4.0, depth_m=10.0)


def test_corners_map_to_the_rectangle():
    c = ground_rect()
    expected = [(0, 0), (4, 0), (4, 10), (0, 10)]
    for (nx, ny), (gx, gy) in zip(TRAPEZOID, expected):
        x, y = c.to_ground((nx * W, ny * H), W, H)
        assert x == pytest.approx(gx, abs=1e-3) and y == pytest.approx(gy, abs=1e-3)


def test_resolution_does_not_matter():
    c = ground_rect()
    a = c.to_ground((0.5 * W, 0.6 * H), W, H)
    b = c.to_ground((0.5 * 1920, 0.6 * 1080), 1920, 1080)
    assert a == pytest.approx(b, abs=1e-6)


def test_the_same_real_step_is_more_pixels_near_the_camera():
    c = ground_rect()
    near = abs(c.to_ground((0.5 * W, 0.88 * H), W, H)[1] - c.to_ground((0.5 * W, 0.86 * H), W, H)[1])
    far = abs(c.to_ground((0.5 * W, 0.44 * H), W, H)[1] - c.to_ground((0.5 * W, 0.42 * H), W, H)[1])
    assert far > near * 2  # the same 12px covers far more ground far away


@pytest.mark.parametrize("points, width, depth", [
    (TRAPEZOID[:3], 4, 10),                                      # three corners
    ([(0.3, 0.9), (0.58, 0.4), (0.7, 0.9), (0.42, 0.4)], 4, 10),  # crossed (bow-tie)
    ([(0.1, 0.1), (0.2, 0.2), (0.3, 0.3), (0.4, 0.4)], 4, 10),    # all on one line
    (TRAPEZOID, 0, 10),                                          # no size
    ([(1.3, 0.9), (0.7, 0.9), (0.58, 0.4), (0.42, 0.4)], 4, 10),  # outside the picture
])
def test_bad_calibrations_are_refused(points, width, depth):
    with pytest.raises(CalibrationError):
        GroundCalibration(points=points, width_m=width, depth_m=depth)


def test_save_load_delete(tmp_path, monkeypatch):
    monkeypatch.setattr(cal, "CALIBRATION_DIR", tmp_path)
    assert cal.load_calibration("cam1") is None
    cal.save_calibration("cam1", ground_rect())
    loaded = cal.load_calibration("cam1")
    assert loaded is not None and loaded.to_dict() == ground_rect().to_dict()
    assert cal.delete_calibration("cam1") and cal.load_calibration("cam1") is None


def test_camera_id_cannot_escape_the_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(cal, "CALIBRATION_DIR", tmp_path)
    with pytest.raises(CalibrationError):
        cal.save_calibration("../evil", ground_rect())


def _pixel_y(c: GroundCalibration, metres_from_near_edge: float) -> float:
    """Picture row (0-1) on the centre line that is this far up the rectangle."""
    ys = np.linspace(0.40, 0.90, 20001)
    dist = np.array([c.to_ground((0.5 * W, y * H), W, H)[1] for y in ys])
    return float(ys[np.argmin(np.abs(dist - metres_from_near_edge))])


def _walk(c: GroundCalibration, m_from: float, m_to: float, seconds: float, fps: int = 10):
    """Someone moving straight up the centre line at a constant speed ON THE
    GROUND - so, because of perspective, unevenly in pixels. Yields
    (frame, t, ground position, pixel position)."""
    n = int(seconds * fps)
    for i in range(n + 1):
        m = m_from + (m_to - m_from) * i / n
        px = (0.5 * W, _pixel_y(c, m) * H)
        yield i, i / fps, c.to_ground(px, W, H), px


def _abnormal_speed(points, config: BehaviorConfig) -> list:
    m = BehaviorMonitor(config)
    for i, ts, ground, _ in points:
        m.update({1: ground}, i, ts)
    return [e for e in m.events if e.type == "ABNORMAL_SPEED"]


def test_metres_make_speed_the_same_near_and_far():
    """A 1.5 m/s walk is calm whether it happens near the camera or far away.

    In pixels the far walk crosses few pixels and the near one many, so a
    pixel threshold cannot be right for both; in metres both measure 1.5 m/s.
    """
    c = ground_rect()
    near = list(_walk(c, 0.0, 3.0, seconds=2.0))   # 3 m in 2 s
    far = list(_walk(c, 7.0, 10.0, seconds=2.0))   # 3 m in 2 s
    px_speed = lambda w: abs(w[-1][3][1] - w[0][3][1]) / 2.0
    assert px_speed(near) > 2 * px_speed(far)  # very different in pixels...

    metric = BehaviorConfig(sample_interval_sec=0.5).in_metres()
    for walk in (near, far):
        speeds = [np.hypot(*np.subtract(walk[i + 5][2], walk[i][2])) / 0.5 for i in range(0, len(walk) - 5, 5)]
        assert np.mean(speeds) == pytest.approx(1.5, rel=0.05)  # ...the same in metres
        assert _abnormal_speed(walk, metric) == []


def test_running_is_flagged_in_metres():
    run = list(_walk(ground_rect(), 0.0, 10.0, seconds=2.0))  # 5 m/s
    events = _abnormal_speed(run, BehaviorConfig(sample_interval_sec=0.5).in_metres())
    assert events and events[0].data["speed_m_per_sec"] == pytest.approx(5.0, rel=0.1)


def test_a_car_at_city_speed_is_not_flagged_but_a_person_would_be():
    drive = list(_walk(ground_rect(), 0.0, 10.0, seconds=1.0))  # 10 m/s = 36 km/h
    assert _abnormal_speed(drive, BehaviorConfig(sample_interval_sec=0.3).in_metres(vehicles=True)) == []
    assert _abnormal_speed(drive, BehaviorConfig(sample_interval_sec=0.3).in_metres()) != []


def test_in_metres_keeps_timing_and_marks_units():
    base = BehaviorConfig(loitering_seconds=12.0, behavior_cooldown_seconds=20.0)
    m = base.in_metres()
    assert m.units == "m" and base.units == "px"
    assert m.loitering_seconds == 12.0 and m.behavior_cooldown_seconds == 20.0
    assert m.speed_threshold_px_per_sec == base.speed_threshold_m_per_sec


# --- API and end-to-end ------------------------------------------------------

from fastapi.testclient import TestClient  # noqa: E402

from backend.app import app  # noqa: E402
from backend.config import REPO_ROOT  # noqa: E402
from backend.live.camera_manager import camera_manager  # noqa: E402
from backend.orchestrator import CombinedPipeline, OrchestratorRequest  # noqa: E402

client = TestClient(app)


@pytest.fixture
def camera(tmp_path, monkeypatch):
    monkeypatch.setattr(cal, "CALIBRATION_DIR", tmp_path)
    response = client.post("/api/live/phone-cameras", json={"name": "Calibration test"})
    assert response.status_code == 201
    camera_id = response.json()["camera_id"]
    yield camera_id
    camera_manager.remove(camera_id)


def test_calibration_api_round_trip(camera):
    url = f"/api/live/cameras/{camera}/calibration"
    assert client.get(url).json() == {"calibrated": False}
    body = {"points": [list(p) for p in TRAPEZOID], "width_m": 4, "depth_m": 10}
    saved = client.put(url, json=body)
    assert saved.status_code == 200 and saved.json()["calibrated"] is True
    assert client.get(url).json()["width_m"] == 4
    assert client.delete(url).json() == {"calibrated": False}
    assert client.get(url).json() == {"calibrated": False}


def test_calibration_api_refuses_a_crossed_rectangle(camera):
    body = {"points": [[0.3, 0.9], [0.58, 0.4], [0.7, 0.9], [0.42, 0.4]], "width_m": 4, "depth_m": 10}
    response = client.put(f"/api/live/cameras/{camera}/calibration", json=body)
    assert response.status_code == 400 and "order" in response.json()["detail"]


def test_calibration_api_unknown_camera():
    assert client.get("/api/live/cameras/nope/calibration").status_code == 404


def test_pipeline_runs_behaviour_in_metres(tmp_path):
    video = REPO_ROOT / "uploads" / "smoke_person_id.mp4"
    if not video.exists():
        pytest.skip("smoke clip not present")
    req = OrchestratorRequest(enable_behavior=True, ground_calibration=ground_rect())
    result = CombinedPipeline(req).run(video, tmp_path / "out.mp4")
    assert result.behavior["units"] == "m"
    speeds = [e["data"] for e in result.events if e["type"] == "ABNORMAL_SPEED"]
    assert all("speed_m_per_sec" in d and "speed_px_per_sec" not in d for d in speeds)
