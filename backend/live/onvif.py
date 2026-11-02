"""Find IP cameras on the local network and get their stream address, any brand.

Every camera vendor's RTSP URL is different (Hikvision, Dahua, CP Plus ...),
which is what makes "integrate with existing cameras" sound hard. ONVIF is
the industry standard that removes the difference: a camera announces itself
when asked over WS-Discovery, and tells you its own stream URL when asked
over its media service. This module does both with the standard library and
httpx, so adding a camera becomes scan, pick, type the password.

Only scan networks you run or are authorised to use. Discovery is one
multicast question on the local segment; it does not probe or log in to
anything, and credentials are only sent to the camera the operator picked.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os
import re
import socket
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import httpx

logger = logging.getLogger("backend.live.onvif")

DISCOVERY_ADDRESS = ("239.255.255.250", 3702)

PROBE = """<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"
            xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"
            xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
            xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
  <e:Header>
    <w:MessageID>uuid:{message_id}</w:MessageID>
    <w:To e:mustUnderstand="true">urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>
    <w:Action e:mustUnderstand="true">http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>
  </e:Header>
  <e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body>
</e:Envelope>"""

WSSE = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd"
WSU = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd"
DIGEST = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest"
BASE64 = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary"


class OnvifError(RuntimeError):
    """A camera could not be reached or refused a request. The message is
    written for the operator: it says what to check."""


# --- small XML helpers ---------------------------------------------------------
# Cameras use different namespace prefixes for the same elements, so matching
# is done on local names only.

def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find_all(root: ET.Element, name: str) -> list[ET.Element]:
    return [el for el in root.iter() if _local(el.tag) == name]


def _find(root: ET.Element, name: str) -> ET.Element | None:
    return next((el for el in root.iter() if _local(el.tag) == name), None)


def _text(root: ET.Element | None, name: str) -> str | None:
    if root is None:
        return None
    el = _find(root, name)
    return el.text.strip() if el is not None and el.text else None


# --- discovery -------------------------------------------------------------------

@dataclass
class DiscoveredCamera:
    address: str          # IP the answer came from
    xaddr: str            # the camera's ONVIF device service URL
    name: str | None = None
    hardware: str | None = None
    xaddrs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"address": self.address, "xaddr": self.xaddr, "name": self.name, "hardware": self.hardware}


def _scope_value(scopes: list[str], kind: str) -> str | None:
    for scope in scopes:
        match = re.match(rf"onvif://www\.onvif\.org/{kind}/(.+)$", scope)
        if match:
            return unquote(match.group(1)).replace("_", " ").strip() or None
    return None


def parse_probe_match(payload: bytes, sender: str) -> DiscoveredCamera | None:
    """One WS-Discovery ProbeMatch answer, or None if it is not one."""
    try:
        root = ET.fromstring(payload)
    except ET.ParseError:
        return None
    match = _find(root, "ProbeMatch")
    if match is None:
        return None
    xaddrs = (_text(match, "XAddrs") or "").split()
    if not xaddrs:
        return None
    # Prefer the address on the same host that answered: cameras with
    # several interfaces list them all, some unreachable from here.
    preferred = next((x for x in xaddrs if urlsplit(x).hostname == sender), xaddrs[0])
    scopes = (_text(match, "Scopes") or "").split()
    return DiscoveredCamera(
        address=sender, xaddr=preferred, xaddrs=xaddrs,
        name=_scope_value(scopes, "name"), hardware=_scope_value(scopes, "hardware"),
    )


def discover(interfaces: list[str], timeout_sec: float = 2.5) -> list[DiscoveredCamera]:
    """Ask every camera on the given local interfaces to answer.

    One probe is sent from each interface, because on a machine with several
    networks (Wi-Fi plus a camera LAN, say) a single multicast only leaves
    through one of them.
    """
    sockets = []
    message = PROBE.format(message_id=uuid.uuid4()).encode()
    for interface in interfaces or ["0.0.0.0"]:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
            sock.bind((interface, 0))
            if interface != "0.0.0.0":
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(interface))
            sock.setblocking(False)
            sock.sendto(message, DISCOVERY_ADDRESS)
            sockets.append(sock)
        except OSError as exc:
            logger.warning("ONVIF probe could not be sent from %s: %s", interface, exc)

    found: dict[str, DiscoveredCamera] = {}
    deadline = time.monotonic() + timeout_sec
    try:
        while sockets and time.monotonic() < deadline:
            for sock in sockets:
                try:
                    payload, (sender, _) = sock.recvfrom(65535)
                except (BlockingIOError, InterruptedError):
                    continue
                except OSError:
                    continue
                camera = parse_probe_match(payload, sender)
                if camera is not None:
                    found.setdefault(camera.xaddr, camera)
            time.sleep(0.05)
    finally:
        for sock in sockets:
            sock.close()
    return sorted(found.values(), key=lambda c: tuple(int(p) if p.isdigit() else 0 for p in c.address.split(".")))


# --- talking to one camera ---------------------------------------------------------

def _security_header(username: str, password: str, clock_offset: timedelta) -> str:
    """WS-Security UsernameToken with a password digest. The camera checks
    the timestamp too, so it is written in the camera's own clock."""
    nonce = os.urandom(16)
    created = (datetime.now(timezone.utc) + clock_offset).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    digest = base64.b64encode(hashlib.sha1(nonce + created.encode() + password.encode()).digest()).decode()
    return (
        f'<Security s:mustUnderstand="1" xmlns="{WSSE}"><UsernameToken>'
        f"<Username>{_xml_escape(username)}</Username>"
        f'<Password Type="{DIGEST}">{digest}</Password>'
        f'<Nonce EncodingType="{BASE64}">{base64.b64encode(nonce).decode()}</Nonce>'
        f'<Created xmlns="{WSU}">{created}</Created>'
        "</UsernameToken></Security>"
    )


def _xml_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _envelope(body: str, header: str = "") -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">'
        f"<s:Header>{header}</s:Header><s:Body>{body}</s:Body></s:Envelope>"
    )


@dataclass
class StreamProfile:
    token: str
    name: str
    width: int | None
    height: int | None
    encoding: str | None

    def to_dict(self) -> dict:
        return {"token": self.token, "name": self.name, "width": self.width, "height": self.height, "encoding": self.encoding}


class OnvifCamera:
    def __init__(self, xaddr: str, username: str, password: str, http: httpx.Client | None = None,
                 timeout_sec: float = 6.0):
        self.xaddr = xaddr
        self.username = username
        self.password = password
        self._http = http or httpx.Client(timeout=timeout_sec)
        self._clock_offset: timedelta | None = None
        self._media_xaddr: str | None = None

    def _post(self, url: str, body: str, authenticated: bool = True) -> ET.Element:
        header = _security_header(self.username, self.password, self._clock()) if authenticated else ""
        try:
            response = self._http.post(url, content=_envelope(body, header).encode(),
                                       headers={"Content-Type": "application/soap+xml; charset=utf-8"})
        except httpx.HTTPError as exc:
            raise OnvifError(f"cannot reach the camera at {urlsplit(url).hostname}; check it is powered and on this network") from exc
        try:
            root = ET.fromstring(response.content)
        except ET.ParseError as exc:
            raise OnvifError(f"the camera answered with something other than ONVIF (HTTP {response.status_code})") from exc
        fault = _find(root, "Fault")
        if fault is not None or response.status_code >= 400:
            reason = ""
            if fault is not None:
                # An Element with no children is falsy, so no `a or b` here.
                source = _find(fault, "Reason")
                reason = " ".join(t.strip() for t in (source if source is not None else fault).itertext() if t.strip())
            if response.status_code in (400, 401) or "auth" in reason.lower() or "NotAuthorized" in reason:
                raise OnvifError("the camera rejected the username or password")
            raise OnvifError(f"the camera refused the request: {reason or f'HTTP {response.status_code}'}")
        return root

    def _clock(self) -> timedelta:
        """Offset between the camera's clock and ours. Cameras on a closed
        network often have the wrong time, and the login digest includes a
        timestamp the camera checks against its own clock."""
        if self._clock_offset is None:
            self._clock_offset = timedelta(0)
            try:
                root = self._post(self.xaddr, '<GetSystemDateAndTime xmlns="http://www.onvif.org/ver10/device/wsdl"/>',
                                  authenticated=False)
                utc = _find(root, "UTCDateTime")
                if utc is not None:
                    camera_now = datetime(
                        int(_text(utc, "Year")), int(_text(utc, "Month")), int(_text(utc, "Day")),
                        int(_text(utc, "Hour")), int(_text(utc, "Minute")), int(_text(utc, "Second")),
                        tzinfo=timezone.utc,
                    )
                    self._clock_offset = camera_now - datetime.now(timezone.utc)
            except (OnvifError, TypeError, ValueError):
                pass  # use our own clock; most cameras accept a small difference
        return self._clock_offset

    def media_xaddr(self) -> str:
        if self._media_xaddr is None:
            root = self._post(self.xaddr, '<GetCapabilities xmlns="http://www.onvif.org/ver10/device/wsdl">'
                                          "<Category>Media</Category></GetCapabilities>")
            media = _find(root, "Media")
            address = _text(media, "XAddr")
            if not address:
                raise OnvifError("the camera did not report a media service")
            self._media_xaddr = address
        return self._media_xaddr

    def profiles(self) -> list[StreamProfile]:
        root = self._post(self.media_xaddr(), '<GetProfiles xmlns="http://www.onvif.org/ver10/media/wsdl"/>')
        out = []
        for profile in _find_all(root, "Profiles"):
            token = profile.get("token")
            if not token:
                continue
            encoder = _find(profile, "VideoEncoderConfiguration")
            resolution = _find(encoder, "Resolution") if encoder is not None else None
            out.append(StreamProfile(
                token=token, name=_text(profile, "Name") or token,
                width=int(_text(resolution, "Width")) if _text(resolution, "Width") else None,
                height=int(_text(resolution, "Height")) if _text(resolution, "Height") else None,
                encoding=_text(encoder, "Encoding"),
            ))
        if not out:
            raise OnvifError("the camera reported no video streams")
        # Highest resolution first: the main stream, which ANPR needs.
        return sorted(out, key=lambda p: -((p.width or 0) * (p.height or 0)))

    def stream_uri(self, profile_token: str) -> str:
        body = (
            '<GetStreamUri xmlns="http://www.onvif.org/ver10/media/wsdl">'
            '<StreamSetup><Stream xmlns="http://www.onvif.org/ver10/schema">RTP-Unicast</Stream>'
            '<Transport xmlns="http://www.onvif.org/ver10/schema"><Protocol>RTSP</Protocol></Transport></StreamSetup>'
            f"<ProfileToken>{_xml_escape(profile_token)}</ProfileToken></GetStreamUri>"
        )
        uri = _text(self._post(self.media_xaddr(), body), "Uri")
        if not uri:
            raise OnvifError("the camera did not return a stream address")
        return with_credentials(uri, self.username, self.password)


def with_credentials(uri: str, username: str, password: str) -> str:
    """The stream URL with the login in it, as RTSP expects. Cameras return
    it without one; the existing camera pipeline reads credentials from the
    URL and keeps them out of everything it reports."""
    parts = urlsplit(uri)
    if not username or "@" in parts.netloc:
        return uri
    netloc = f"{quote(username, safe='')}:{quote(password, safe='')}@{parts.netloc}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
