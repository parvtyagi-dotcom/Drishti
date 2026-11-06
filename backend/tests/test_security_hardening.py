"""Security fixes from the ECC review, pinned with real requests.

Each test names the hole it closes. They run with authentication switched ON
(conftest disables it for the rest of the suite), against a throwaway user
store and signing key.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend import app as app_module
from backend.api import auth, auth_routes, device_auth, storage
from backend.api.live import CameraCreateRequest
from backend.app import app


@pytest.fixture
def secured(tmp_path, monkeypatch):
    """Login enforced, with an admin, an operator and an auditor."""
    monkeypatch.delenv("BORDERWATCH_DISABLE_AUTH", raising=False)
    monkeypatch.delenv("DRISHTI_NO_LOGIN", raising=False)
    monkeypatch.delenv("DRISHTI_OPEN_DEVICES", raising=False)
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
    monkeypatch.setattr(auth, "USERS_PATH", tmp_path / "users.json")
    monkeypatch.setattr(auth, "SECRET_PATH", tmp_path / "secret")
    monkeypatch.setattr(auth, "REVOKED_PATH", tmp_path / "revoked.json")
    auth_routes.login_throttle.reset()
    auth.create_user("admin", "adminpass123", auth.ADMIN)
    auth.create_user("operator", "operatorpass1", auth.OPERATOR)
    auth.create_user("auditor", "auditorpass1", auth.AUDITOR)
    return tmp_path


def _client_as(username: str, password: str) -> TestClient:
    client = TestClient(app)
    response = client.post("/api/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    return client


# --- login is the default -------------------------------------------------------

def test_login_is_required_unless_explicitly_switched_off(monkeypatch):
    monkeypatch.delenv("DRISHTI_NO_LOGIN", raising=False)
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
    assert not auth.no_login(), "an unconfigured server must not hand out admin to everyone"
    monkeypatch.setenv("DRISHTI_NO_LOGIN", "1")
    assert auth.no_login()


def test_api_rejects_anonymous_requests(secured):
    assert TestClient(app).get("/api/live/cameras").status_code == 401


def test_login_response_does_not_leak_the_token_into_the_body(secured):
    response = TestClient(app).post("/api/auth/login", json={"username": "admin", "password": "adminpass123"})
    assert "token" not in response.json()
    assert response.cookies.get("bw_session")


# --- path traversal in the dashboard fallback -------------------------------------

@pytest.mark.parametrize("path", [
    "../data/session-secret", "../../backend/config.py", "..%2f..%2fREADME.md", "assets/../../README.md",
])
def test_dashboard_fallback_never_serves_files_outside_the_build(tmp_path, path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("shell")
    (tmp_path / "secret.txt").write_text("do not serve")
    assert app_module._dashboard_asset(dist, path) is None
    assert app_module._dashboard_asset(dist, "../secret.txt") is None


def test_dashboard_fallback_still_serves_real_assets(tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "assets" / "app.js").write_text("js")
    assert app_module._dashboard_asset(dist, "assets/app.js") == (dist / "assets" / "app.js").resolve()


# --- websockets require a session or a device key ---------------------------------

def test_event_websocket_refuses_anonymous_viewers(secured):
    with pytest.raises(WebSocketDisconnect):
        with TestClient(app).websocket_connect("/api/live/ws/events") as ws:
            ws.receive_json()


def test_event_websocket_accepts_a_signed_in_operator(secured):
    client = _client_as("operator", "operatorpass1")
    with client.websocket_connect("/api/live/ws/events", headers={"origin": "http://testserver"}):
        pass


def test_event_websocket_refuses_a_cross_site_page_riding_the_cookie(secured):
    client = _client_as("operator", "operatorpass1")
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/live/ws/events", headers={"origin": "http://evil.example"}) as ws:
            ws.receive_json()


def test_event_websocket_accepts_a_paired_siren(secured):
    key = device_auth.device_key("siren")
    with TestClient(app).websocket_connect(f"/api/live/ws/events?key={key}"):
        pass


def test_a_siren_key_cannot_push_camera_frames(secured):
    key = device_auth.device_key("siren")
    with pytest.raises(WebSocketDisconnect):
        with TestClient(app).websocket_connect(f"/api/live/ws/ingest/somecam?key={key}") as ws:
            ws.receive_json()


def test_an_auditor_session_cannot_push_camera_frames(secured):
    client = _client_as("auditor", "auditorpass1")
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/live/ws/ingest/somecam", headers={"origin": "http://testserver"}) as ws:
            ws.receive_json()


def test_a_revocation_survives_an_unreadable_revocation_file(secured):
    client = _client_as("operator", "operatorpass1")
    token = client.cookies.get("bw_session")
    client.post("/api/auth/logout")
    auth.REVOKED_PATH.write_text("{not json", encoding="utf-8")
    assert auth.decode_token(token) is None, "a corrupt revocation file re-enabled a logged-out session"


def test_speaker_websocket_refuses_anonymous_devices(secured):
    with pytest.raises(WebSocketDisconnect):
        with TestClient(app).websocket_connect("/api/live/ws/speaker?camera=Gate") as ws:
            ws.receive_json()


def test_device_keys_are_scoped_and_unforgeable(secured):
    assert device_auth.check_device_key(device_auth.device_key("ingest:cam1"), "ingest:cam1")
    assert not device_auth.check_device_key(device_auth.device_key("ingest:cam1"), "ingest:cam2")
    assert not device_auth.check_device_key("0" * 32, "siren")
    assert not device_auth.check_device_key(None, "siren")


def test_siren_alert_pictures_need_the_siren_key(secured):
    anonymous = TestClient(app).get("/api/live/siren/alert/abc123/brief")
    assert anonymous.status_code == 401
    keyed = TestClient(app).get(f"/api/live/siren/alert/abc123/brief?key={device_auth.device_key('siren')}")
    assert keyed.status_code == 404  # authorised; that alarm simply does not exist


def test_pairing_links_carry_their_key(secured):
    info = _client_as("admin", "adminpass123").get("/api/live/connect-info").json()
    if info["siren_url"]:
        assert f"key={device_auth.device_key('siren')}" in info["siren_url"]
    assert info["siren_key"] == device_auth.device_key("siren")


# --- roles ---------------------------------------------------------------------

def test_an_auditor_cannot_change_anything(secured):
    client = _client_as("auditor", "auditorpass1")
    assert client.get("/api/live/cameras").status_code == 200
    response = client.post("/api/live/cameras", json={"name": "x", "source": "rtsp://10.0.0.5/s", "start": False})
    assert response.status_code == 403


def test_only_an_admin_can_delete(secured):
    client = _client_as("operator", "operatorpass1")
    assert client.delete("/api/live/cameras/nonexistent").status_code == 403
    admin = _client_as("admin", "adminpass123")
    assert admin.delete("/api/live/cameras/nonexistent").status_code == 404


# --- sessions ------------------------------------------------------------------

def test_logout_revokes_the_session(secured):
    client = _client_as("operator", "operatorpass1")
    token = client.cookies.get("bw_session")
    assert client.post("/api/auth/logout").status_code == 200
    assert auth.decode_token(token) is None, "a logged-out token still worked"


def test_repeated_failed_logins_are_throttled(secured):
    client = TestClient(app)
    codes = [client.post("/api/auth/login", json={"username": "admin", "password": "wrong"}).status_code
             for _ in range(auth_routes.LoginThrottle.MAX_FAILURES + 1)]
    assert codes[-1] == 429, codes
    # Even the right password is refused while throttled.
    ok = client.post("/api/auth/login", json={"username": "admin", "password": "adminpass123"})
    assert ok.status_code == 429


def test_google_accounts_default_to_read_only_and_need_an_allowlist(monkeypatch):
    monkeypatch.delenv("GOOGLE_DEFAULT_ROLE", raising=False)
    monkeypatch.delenv("GOOGLE_ALLOWED_EMAILS", raising=False)
    monkeypatch.delenv("GOOGLE_ALLOWED_DOMAINS", raising=False)
    assert auth_routes.google_default_role() == auth.AUDITOR
    assert not auth_routes.google_email_allowed("someone@gmail.com"), "any Google account could sign in"
    monkeypatch.setenv("GOOGLE_ALLOWED_DOMAINS", "bsf.gov.in")
    assert auth_routes.google_email_allowed("officer@bsf.gov.in")
    assert not auth_routes.google_email_allowed("someone@gmail.com")


# --- untrusted identifiers and sources ----------------------------------------------

@pytest.mark.parametrize("bad", ["../users", "..\\users", "a/b", "*", "x?", "", "a" * 65])
def test_storage_ids_cannot_escape_their_directory(bad):
    for lookup in (storage.find_video_path, storage.load_zone, storage.find_reference_photos):
        with pytest.raises(storage.NotFoundError):
            lookup(bad)


def test_uploaded_file_extensions_are_allowlisted():
    assert storage.safe_suffix("clip.MP4", storage.VIDEO_SUFFIXES, ".mp4") == ".mp4"
    assert storage.safe_suffix("evil.html", storage.VIDEO_SUFFIXES, ".mp4") == ".mp4"
    assert storage.safe_suffix("face.png", storage.IMAGE_SUFFIXES, ".jpg") == ".png"


@pytest.mark.parametrize("source", [
    "rtsp://192.168.1.20:554/stream1", "http://10.0.0.7:8080/video", "https://cam.local/mjpeg", "0", "2",
])
def test_camera_sources_that_are_real_cameras_are_accepted(source):
    assert CameraCreateRequest(name="c", source=source).source == source


@pytest.mark.parametrize("source", [
    "file:///etc/passwd", "C:\\Windows\\win.ini", "/etc/passwd", "http://169.254.169.254/latest/meta-data",
    "gopher://10.0.0.1/", "rtsp://", "concat:a.mp4|b.mp4", "push://cam",
    # Other spellings of the metadata address that FFmpeg resolves the same way.
    "http://2852039166/", "http://0xa9fea9fe/", "http://0251.0376.0251.0376/", "http://169.254.43518/",
    "rtsp://[::ffff:169.254.169.254]/", "rtsp://[::ffff:a9fe:a9fe]/", "http://metadata.google.internal/",
    "http://0/", "\\\\fileserver\\share\\clip.mp4",
])
def test_camera_sources_that_are_not_cameras_are_refused(source):
    with pytest.raises(ValueError):
        CameraCreateRequest(name="c", source=source)


def test_a_local_recording_inside_uploads_is_still_accepted(tmp_path, monkeypatch):
    from backend.live import source_policy
    monkeypatch.setattr(source_policy, "UPLOADS_DIR", tmp_path)
    clip = tmp_path / "rehearsal.mp4"
    clip.write_bytes(b"x")
    assert CameraCreateRequest(name="c", source=str(clip)).source == str(clip)


def test_api_docs_are_off_by_default():
    assert app.docs_url is None and app.openapi_url is None
