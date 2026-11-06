"""Who may open a websocket, and how devices that cannot sign in pair.

Starlette's HTTP middleware never sees a websocket, so the session gate in
backend.app did not cover /ws/events (every alarm), /ws/ingest (camera frames
into the analysis) or /ws/speaker. Each handler now calls authorize_websocket
before accepting.

Phones used as a siren, a camera or a talk-down speaker cannot log in. They
pair instead: an operator who is signed in opens a QR link that carries a
device key. A key is an HMAC of its scope under this installation's signing
secret, so it is:

  * scoped    - a siren key reads alarms; it cannot push frames into a camera
  * stable    - a code printed for a post keeps working after a restart
  * revocable - rotating data/session-secret invalidates every key at once

DRISHTI_OPEN_DEVICES=1 lets devices connect without a key. It exists only so
an Android siren APK built before pairing keys can still be demoed; rebuild
the APK and leave it off.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
from urllib.parse import urlsplit

from fastapi import HTTPException, Request, WebSocket

from backend.api import auth

logger = logging.getLogger("backend.api.device_auth")

SIREN = "siren"
SPEAKER = "speaker"
_KEY_LENGTH = 32
# Close code a browser reports for "not allowed" (4000-4999 is app-defined).
UNAUTHORIZED_CLOSE = 4401


_KEY_IN_URL = re.compile(r"([?&]key=)[0-9a-fA-F]+")


class RedactDeviceKeys(logging.Filter):
    """Keeps pairing keys out of access logs: they travel as ?key= in device
    URLs, and uvicorn logs every request line."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.args:
            record.args = tuple(
                _KEY_IN_URL.sub(r"\1[redacted]", a) if isinstance(a, str) else a for a in record.args
            )
        if isinstance(record.msg, str):
            record.msg = _KEY_IN_URL.sub(r"\1[redacted]", record.msg)
        return True


def install_log_redaction() -> None:
    for name in ("uvicorn.access", "uvicorn.error"):
        log = logging.getLogger(name)
        if not any(isinstance(f, RedactDeviceKeys) for f in log.filters):
            log.addFilter(RedactDeviceKeys())


def ingest_scope(camera_id: str) -> str:
    return f"ingest:{camera_id}"


def device_key(scope: str) -> str:
    digest = hmac.new(auth._secret(), f"device:{scope}".encode(), hashlib.sha256).hexdigest()
    return digest[:_KEY_LENGTH]


def check_device_key(key: str | None, scope: str) -> bool:
    if not key or len(key) != _KEY_LENGTH:
        return False
    return hmac.compare_digest(key, device_key(scope))


def auth_disabled() -> bool:
    return bool(os.environ.get("BORDERWATCH_DISABLE_AUTH")) or auth.no_login()


def open_devices() -> bool:
    return os.environ.get("DRISHTI_OPEN_DEVICES") == "1"


def _same_origin(origin: str | None, host: str | None) -> bool:
    """A cookie rides along on any page's websocket to us, including a
    hostile site the operator happens to have open. Browsers always send
    Origin on websockets, so a session cookie only counts when the page
    opening the socket is our own."""
    if not origin:
        return True  # not a browser (curl, a native app): no ambient cookie risk
    return urlsplit(origin).netloc.lower() == (host or "").lower()


def session_user(websocket: WebSocket) -> auth.User | None:
    token = websocket.cookies.get("bw_session")
    if not token:
        header = websocket.headers.get("authorization", "")
        token = header[7:].strip() if header.lower().startswith("bearer ") else None
    if not token:
        return None
    if not _same_origin(websocket.headers.get("origin"), websocket.headers.get("host")):
        logger.warning("refused cross-origin websocket from %s", websocket.headers.get("origin"))
        return None
    return auth.decode_token(token)


async def authorize_websocket(websocket: WebSocket, device_scope: str | None = None,
                              session_roles: tuple[str, ...] | None = None) -> bool:
    """True if the socket may proceed; otherwise it has been closed.

    session_roles limits which signed-in roles count: pushing frames into a
    camera is a write, so a read-only AUDITOR session must not open ingest.
    """
    if auth_disabled():
        return True
    user = session_user(websocket)
    if user is not None and (session_roles is None or user.role in session_roles):
        return True
    if device_scope is not None:
        if open_devices() or check_device_key(websocket.query_params.get("key"), device_scope):
            return True
    await websocket.close(code=UNAUTHORIZED_CLOSE)
    return False


def require_device_or_session(request: Request, scope: str) -> None:
    """HTTP counterpart, for the few device pages' data endpoints."""
    if auth_disabled() or open_devices():
        return
    if check_device_key(request.query_params.get("key"), scope):
        return
    token = auth._token_from(request)
    if token and auth.decode_token(token) is not None:
        return
    raise HTTPException(401, "this device is not paired - scan the code from the Devices page again")
