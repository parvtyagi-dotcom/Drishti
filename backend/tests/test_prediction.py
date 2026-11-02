"""Path prediction: next camera ahead, time to reach it, or a blind spot."""
from backend.live.prediction import predict

POS = {"A": (0.1, 0.5), "B": (0.3, 0.5), "C": (0.6, 0.52), "Behind": (0.0, 0.5), "Side": (0.3, 0.9)}


def ev(cam, t, etype="ZONE_ENTRY"):
    return {"camera_name": cam, "wall_time": t, "type": etype}


def test_next_camera_ahead_with_eta():
    p = predict([ev("A", 1000), ev("B", 1060)], POS, now=1060)
    assert p["next_camera"] == "C"
    assert 80 <= p["eta_sec"] <= 100        # 0.3 away at 0.2 per 60 s
    assert "Send the patrol to C" in p["advice"]
    assert p["trail"] == ["A", "B"]


def test_eta_counts_down_since_last_sighting():
    p = predict([ev("A", 1000), ev("B", 1060)], POS, now=1120)
    assert p["eta_sec"] <= 35


def test_blind_spot_when_nothing_ahead():
    pos = {"A": (0.5, 0.5), "B": (0.5, 0.3), "Far": (0.9, 0.9)}
    p = predict([ev("A", 0), ev("B", 30)], pos, now=30)
    assert p["next_camera"] is None and "blind spot" in p["advice"]
    assert p["point"][1] < 0.3


def test_needs_two_placed_cameras_and_ignores_routine_events():
    assert predict([ev("B", 10)], POS, now=10) is None
    assert predict([ev("A", 0, "VEHICLE_DETECTED"), ev("B", 30)], POS, now=30) is None
    assert predict([ev("Unplaced", 0), ev("B", 30)], POS, now=30) is None
    assert predict([ev("A", 0), ev("B", 30)], POS, now=30 + 3600) is None   # too old
