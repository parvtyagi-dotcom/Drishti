"""Which camera sources an operator may connect.

A camera's source goes straight to cv2.VideoCapture, which is FFmpeg - and
FFmpeg opens far more than cameras: file://, concat:, gopher://, any local
path. Accepting any string let a signed-in user read files off the server
(they came back as "video" frames) or make it fetch internal URLs such as a
cloud metadata endpoint.

Allowed:
  * rtsp / rtsps / http / https URLs whose host is not, and does not resolve
    to, a link-local (169.254.0.0/16, fe80::/10 - where metadata services
    live), unspecified or multicast address. Private LAN addresses and
    loopback stay allowed: that is where real cameras and phone-webcam apps
    are.
  * a bare webcam index, "0" to "9"
  * a recording inside uploads/, used as a stand-in camera for rehearsals

The host check resolves names and refuses the numeric shorthands
(2852039166, 0xa9fea9fe, 0251.0376.0251.0376, 169.254.43518) that FFmpeg's
resolver turns into the metadata address while ipaddress does not parse them.
"""
from __future__ import annotations

import ipaddress
import re
import socket
from pathlib import Path
from urllib.parse import urlsplit

from backend.config import UPLOADS_DIR

NETWORK_SCHEMES = frozenset({"rtsp", "rtsps", "http", "https"})
_METADATA_HOSTNAMES = frozenset({"metadata", "metadata.google.internal", "instance-data"})
_NUMERIC_LABEL = re.compile(r"(0x[0-9a-f]*|[0-9]+)")
_CANONICAL_V4 = re.compile(r"(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}")


def _unsafe_address(text: str) -> bool:
    try:
        address = ipaddress.ip_address(text.split("%", 1)[0])
    except ValueError:
        return False
    mapped = getattr(address, "ipv4_mapped", None)
    if mapped is not None:
        address = mapped  # ::ffff:169.254.169.254 is the IPv4 address in disguise
    return address.is_link_local or address.is_unspecified or address.is_multicast


def _resolved(host: str) -> list[str]:
    try:
        return [info[4][0] for info in socket.getaddrinfo(host, None)]
    except (OSError, UnicodeError):
        return []  # unresolvable now: FFmpeg will fail to open it too


def _blocked_host(host: str) -> bool:
    name = host.strip("[]").lower().rstrip(".")
    if name in _METADATA_HOSTNAMES or name.endswith(".internal"):
        return True
    labels = name.split(".")
    if all(_NUMERIC_LABEL.fullmatch(label) for label in labels) and not _CANONICAL_V4.fullmatch(name):
        return True  # integer, hex, octal or short-dotted IPv4 spelling
    if _unsafe_address(name):
        return True
    return any(_unsafe_address(address) for address in _resolved(name))


def validate_camera_source(source: str) -> str:
    """Return source unchanged if it is allowed, else raise ValueError."""
    value = (source or "").strip()
    if value.isdigit() and len(value) == 1:
        return source
    if "://" in value:
        parts = urlsplit(value)
        if parts.scheme.lower() not in NETWORK_SCHEMES:
            raise ValueError(f"unsupported camera address scheme '{parts.scheme}': use rtsp://, http:// or https://")
        if not parts.hostname:
            raise ValueError("camera address has no host")
        if _blocked_host(parts.hostname):
            raise ValueError(f"camera address {parts.hostname} is not a camera network address")
        return source
    if value.startswith(("\\\\", "//")):
        # A UNC path makes Windows open an SMB connection (and offer the
        # service account's credentials) as soon as it is resolved.
        raise ValueError("network file shares cannot be used as a camera")
    if ":" in value and not Path(value).drive:
        # FFmpeg protocol prefixes without "//" (concat:, subfile:, data:...).
        raise ValueError("unsupported camera source")
    try:
        path = Path(value).resolve()
    except (OSError, ValueError):
        raise ValueError("unsupported camera source")
    if not path.is_relative_to(Path(UPLOADS_DIR).resolve()):
        raise ValueError("a local recording used as a camera must be uploaded first (it must live in uploads/)")
    return source
