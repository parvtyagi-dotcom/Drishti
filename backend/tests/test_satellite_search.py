"""Place search and on-demand fetch for Satellite Check, without the network."""
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from backend.api import satellite_routes
from backend.api.auth import OPERATOR, User, current_user
from backend.app import app
from backend.live import sentinel

client = TestClient(app)


@pytest.fixture(autouse=True)
def as_operator():
    app.dependency_overrides[current_user] = lambda: User(username="op", role=OPERATOR)
    yield
    app.dependency_overrides.pop(current_user, None)


def test_windows_around_far_apart_dates_are_fifteen_days_each_side():
    before, after = satellite_routes.date_windows(date(2026, 6, 1), date(2026, 9, 1), today=date(2026, 10, 26))
    assert before == "2026-05-17/2026-06-16"
    assert after == "2026-08-17/2026-09-16"


def test_windows_never_overlap_so_one_picture_cannot_be_both():
    before, after = satellite_routes.date_windows(date(2026, 9, 1), date(2026, 9, 11), today=date(2026, 10, 26))
    assert before.split("/")[1] < after.split("/")[0]


def test_after_window_stops_at_today():
    _, after = satellite_routes.date_windows(date(2026, 9, 1), date(2026, 10, 20), today=date(2026, 10, 26))
    assert after.endswith("/2026-10-26")


def test_pair_id_is_a_safe_folder_name():
    pid = satellite_routes._pair_id("Uri, Baramulla, Jammu and Kashmir, India", date(2026, 10, 21))
    assert pid == "uri-20261021"
    assert satellite_routes._pair_id("../../etc", date(2026, 1, 1)) == "etc-20260101"


def test_box_around_a_point_is_the_requested_size():
    w_km, h_km = sentinel.box_km(sentinel.bbox_around(34.08, 74.03, 5))
    assert abs(w_km - 5) < 0.05 and abs(h_km - 5) < 0.05


def test_fetch_rejects_future_and_too_close_dates():
    body = {"name": "Uri", "lat": 34.08, "lon": 74.03}
    tomorrow = date.today() + timedelta(days=1)
    resp = client.post("/api/satellite/fetch", json={**body, "before": "2026-01-01", "after": str(tomorrow)})
    assert resp.status_code == 400 and "future" in resp.json()["detail"]
    resp = client.post("/api/satellite/fetch", json={**body, "before": "2026-01-01", "after": "2026-01-03"})
    assert resp.status_code == 400 and "apart" in resp.json()["detail"]


def test_empty_place_search_does_not_go_online():
    assert client.get("/api/satellite/places", params={"q": "  "}).json() == {"places": []}


def test_unknown_fetch_job_is_404():
    assert client.get("/api/satellite/fetch/nope").status_code == 404
