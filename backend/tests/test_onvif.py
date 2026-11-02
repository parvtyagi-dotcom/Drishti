"""ONVIF discovery and stream lookup, against canned camera responses.

No camera is needed: the responses below follow the ONVIF and WS-Discovery
message shapes, with the namespace prefixes varied the way real vendors
vary them, and the login digest is checked the way a camera checks it.
"""
from __future__ import annotations

import base64
import hashlib
import re

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.api.auth import OPERATOR, User, current_user
from backend.app import app
from backend.live import onvif
from backend.live.camera_manager import camera_manager
from backend.live.onvif import OnvifCamera, OnvifError, parse_probe_match, with_credentials

PROBE_MATCH = b"""<?xml version="1.0" encoding="UTF-8"?>
<SOAP-ENV:Envelope xmlns:SOAP-ENV="http://www.w3.org/2003/05/soap-envelope"
  xmlns:wsdd="http://schemas.xmlsoap.org/ws/2005/04/discovery">
 <SOAP-ENV:Body><wsdd:ProbeMatches><wsdd:ProbeMatch>
  <wsdd:Types>dn:NetworkVideoTransmitter</wsdd:Types>
  <wsdd:Scopes>onvif://www.onvif.org/type/video_encoder onvif://www.onvif.org/name/HIKVISION%20DS-2CD1023
   onvif://www.onvif.org/hardware/DS-2CD1023G0E-I onvif://www.onvif.org/location/city/hangzhou</wsdd:Scopes>
  <wsdd:XAddrs>http://169.254.10.2/onvif/device_service http://192.168.1.64/onvif/device_service</wsdd:XAddrs>
 </wsdd:ProbeMatch></wsdd:ProbeMatches></SOAP-ENV:Body></SOAP-ENV:Envelope>"""


def test_probe_match_names_the_camera_and_prefers_the_address_that_answered():
    camera = parse_probe_match(PROBE_MATCH, "192.168.1.64")
    assert camera.xaddr == "http://192.168.1.64/onvif/device_service"
    assert camera.name == "HIKVISION DS-2CD1023" and camera.hardware == "DS-2CD1023G0E-I"


def test_other_traffic_on_the_discovery_port_is_ignored():
    assert parse_probe_match(b"not xml", "10.0.0.5") is None
    assert parse_probe_match(b"<a><b/></a>", "10.0.0.5") is None


def test_stream_url_gets_the_login_safely_encoded():
    assert with_credentials("rtsp://192.168.1.64:554/Streaming/Channels/101", "admin", "p@ss:word") == \
        "rtsp://admin:p%40ss%3Aword@192.168.1.64:554/Streaming/Channels/101"
    assert with_credentials("rtsp://u:p@cam/x", "admin", "y") == "rtsp://u:p@cam/x"


# --- a fake camera ----------------------------------------------------------------

def envelope(body: str) -> str:
    return ('<env:Envelope xmlns:env="http://www.w3.org/2003/05/soap-envelope" '
            'xmlns:tt="http://www.onvif.org/ver10/schema" xmlns:trt="http://www.onvif.org/ver10/media/wsdl" '
            f'xmlns:tds="http://www.onvif.org/ver10/device/wsdl"><env:Body>{body}</env:Body></env:Envelope>')


CAMERA_TIME = envelope(
    "<tds:GetSystemDateAndTimeResponse><tds:SystemDateAndTime><tt:UTCDateTime>"
    "<tt:Time><tt:Hour>3</tt:Hour><tt:Minute>4</tt:Minute><tt:Second>5</tt:Second></tt:Time>"
    "<tt:Date><tt:Year>2026</tt:Year><tt:Month>1</tt:Month><tt:Day>2</tt:Day></tt:Date>"
    "</tt:UTCDateTime></tds:SystemDateAndTime></tds:GetSystemDateAndTimeResponse>")
CAPABILITIES = envelope(
    "<tds:GetCapabilitiesResponse><tds:Capabilities><tt:Media>"
    "<tt:XAddr>http://192.168.1.64/onvif/Media</tt:XAddr></tt:Media></tds:Capabilities></tds:GetCapabilitiesResponse>")
PROFILES = envelope(
    '<trt:GetProfilesResponse>'
    '<trt:Profiles token="Profile_2"><tt:Name>subStream</tt:Name><tt:VideoEncoderConfiguration>'
    "<tt:Encoding>H264</tt:Encoding><tt:Resolution><tt:Width>640</tt:Width><tt:Height>360</tt:Height></tt:Resolution>"
    "</tt:VideoEncoderConfiguration></trt:Profiles>"
    '<trt:Profiles token="Profile_1"><tt:Name>mainStream</tt:Name><tt:VideoEncoderConfiguration>'
    "<tt:Encoding>H264</tt:Encoding><tt:Resolution><tt:Width>1920</tt:Width><tt:Height>1080</tt:Height></tt:Resolution>"
    "</tt:VideoEncoderConfiguration></trt:Profiles></trt:GetProfilesResponse>")
STREAM_URI = envelope(
    "<trt:GetStreamUriResponse><trt:MediaUri><tt:Uri>rtsp://192.168.1.64:554/Streaming/Channels/101</tt:Uri>"
    "</trt:MediaUri></trt:GetStreamUriResponse>")
NOT_AUTHORIZED = envelope(
    "<env:Fault><env:Code><env:Value>env:Sender</env:Value></env:Code>"
    '<env:Reason><env:Text xml:lang="en">Sender not Authorized</env:Text></env:Reason></env:Fault>')


def digest_is_valid(body: str, password: str) -> bool:
    nonce = base64.b64decode(re.search(r"<Nonce[^>]*>([^<]+)</Nonce>", body).group(1))
    created = re.search(r"<Created[^>]*>([^<]+)</Created>", body).group(1)
    sent = re.search(r"<Password[^>]*>([^<]+)</Password>", body).group(1)
    return sent == base64.b64encode(hashlib.sha1(nonce + created.encode() + password.encode()).digest()).decode()


class FakeCamera:
    def __init__(self, password="secret"):
        self.password = password
        self.requests: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = request.content.decode()
        self.requests.append(body)
        if "GetSystemDateAndTime" in body:
            assert "UsernameToken" not in body  # the clock is read before logging in
            return httpx.Response(200, text=CAMERA_TIME)
        if not digest_is_valid(body, self.password):
            return httpx.Response(400, text=NOT_AUTHORIZED)
        if "GetCapabilities" in body:
            return httpx.Response(200, text=CAPABILITIES)
        if "GetProfiles" in body:
            assert str(request.url) == "http://192.168.1.64/onvif/Media"
            return httpx.Response(200, text=PROFILES)
        if "GetStreamUri" in body:
            assert "<ProfileToken>Profile_1</ProfileToken>" in body
            return httpx.Response(200, text=STREAM_URI)
        return httpx.Response(400, text=NOT_AUTHORIZED)


def camera(fake, password="secret"):
    return OnvifCamera("http://192.168.1.64/onvif/device_service", "admin", password,
                       http=httpx.Client(transport=httpx.MockTransport(fake)))


def test_profiles_and_stream_address_with_a_valid_login():
    fake = FakeCamera()
    cam = camera(fake)
    profiles = cam.profiles()
    assert [p.token for p in profiles] == ["Profile_1", "Profile_2"]  # main stream first
    assert (profiles[0].width, profiles[0].height, profiles[0].encoding) == (1920, 1080, "H264")
    assert cam.stream_uri("Profile_1") == "rtsp://admin:secret@192.168.1.64:554/Streaming/Channels/101"
    # The login timestamp is written in the camera's clock (2026-01-02), not ours.
    assert "<Created" in fake.requests[1] and "2026-01-02T03:04" in fake.requests[1]


def test_wrong_password_is_reported_plainly():
    with pytest.raises(OnvifError, match="rejected the username or password"):
        camera(FakeCamera(password="secret"), password="wrong").profiles()


def test_unreachable_camera_is_reported_plainly():
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)
    with pytest.raises(OnvifError, match="cannot reach the camera"):
        camera(refuse).profiles()


# --- routes ---------------------------------------------------------------------

client = TestClient(app)


@pytest.fixture
def operator():
    app.dependency_overrides[current_user] = lambda: User(username="op", role=OPERATOR)
    yield
    app.dependency_overrides.pop(current_user, None)


@pytest.mark.parametrize("xaddr", ["http://8.8.8.8/onvif/device_service", "http://example.com/x",
                                   "http://127.0.0.1/onvif", "file:///etc/passwd"])
def test_only_cameras_on_the_local_network_are_contacted(operator, xaddr):
    response = client.post("/api/live/onvif/profiles", json={"xaddr": xaddr, "username": "a", "password": "b"})
    assert response.status_code == 400


def test_adding_a_scanned_camera_keeps_its_login_off_the_response(operator, monkeypatch):
    fake = FakeCamera()
    real_init = OnvifCamera.__init__

    def with_fake_transport(self, xaddr, username, password, http=None, timeout_sec=6.0):
        real_init(self, xaddr, username, password, http=httpx.Client(transport=httpx.MockTransport(fake)))

    monkeypatch.setattr(onvif.OnvifCamera, "__init__", with_fake_transport)
    monkeypatch.setattr("backend.live.camera_manager.LiveSession.start", lambda self: None)

    response = client.post("/api/live/onvif/add", json={
        "xaddr": "http://192.168.1.64/onvif/device_service", "username": "admin", "password": "secret",
        "profile_token": "Profile_1", "name": "Gate camera", "location": "Gate 3",
    })
    assert response.status_code == 201
    body = response.json()
    assert body["source_kind"] == "rtsp" and "secret" not in response.text
    session = camera_manager.get(body["camera_id"])
    try:
        assert session.camera.source == "rtsp://admin:secret@192.168.1.64:554/Streaming/Channels/101"
    finally:
        camera_manager.remove(body["camera_id"])


def test_scan_reports_the_interfaces_it_used(operator, monkeypatch):
    monkeypatch.setattr("backend.api.onvif_routes.lan_addresses", lambda: ["192.168.1.10"])
    monkeypatch.setattr("backend.api.onvif_routes.discover", lambda interfaces: [
        parse_probe_match(PROBE_MATCH, "192.168.1.64")])
    body = client.post("/api/live/onvif/scan").json()
    assert body["interfaces"] == ["192.168.1.10"]
    assert body["cameras"][0]["name"] == "HIKVISION DS-2CD1023"
