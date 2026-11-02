"""Authentication and role-based access for the Drishti API.

Until now any device on the network could view every camera, add cameras, or
delete evidence. That is survivable on localhost and unacceptable the moment
the server binds 0.0.0.0 - which the phone siren and phone camera both
require. Surveillance footage and an evidence trail are exactly the data that
must not be world-readable on a shared network.

Deliberately built on the standard library: PBKDF2 for password hashing and an
HMAC-signed token, rather than pulling in a JWT/bcrypt stack for a single-node
edge box. The security properties that matter here - passwords never stored in
recoverable form, tokens that cannot be forged or edited client-side, and an
expiry - are all provided without new dependencies to audit or update.

Three roles, matching how a control room actually works:
  ADMIN     - full control, including camera configuration and user management
  OPERATOR  - monitor, acknowledge and configure analysis; cannot delete
  AUDITOR   - read-only, including the audit trail
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

from fastapi import Depends, HTTPException, Request

from backend.config import DATA_DIR

logger = logging.getLogger("backend.api.auth")

ADMIN, OPERATOR, AUDITOR = "ADMIN", "OPERATOR", "AUDITOR"
ROLES = (ADMIN, OPERATOR, AUDITOR)

USERS_PATH = DATA_DIR / "users.json"
SECRET_PATH = DATA_DIR / "session-secret"
# A control-room screen and a phone propped at a post are not casual browser
# tabs - they are appliances that must still be working after a weekend.
# Re-authenticating a wall display every twelve hours is how people end up
# disabling authentication altogether, so the session is long by default and
# the operator can end it deliberately with Sign out.
TOKEN_TTL_SEC = 30 * 24 * 3600
PBKDF2_ROUNDS = 240_000

# Endpoints reachable without a token. The siren and phone-camera pages are
# opened by scanning a code on a device that cannot log in first; they are
# read-only surfaces and the data they carry is already on the operator's
# screen. Everything else requires a session.
PUBLIC_PATHS = {
    "/api/auth/login", "/api/auth/bootstrap", "/api/auth/google", "/api/auth/me",
    "/api/live/siren", "/api/live/speaker", "/api/live/sw.js", "/api/live/phone", "/api/live/siren-app",
    "/docs", "/openapi.json", "/redoc",
}
# Sensor hooks carry their own per-sensor key instead of a session; see
# api/sensor_routes.py.
PUBLIC_PREFIXES = ("/api/live/siren/", "/api/sensors/hook/")


def _secret() -> bytes:
    """Per-installation signing key, generated once and kept out of source."""
    if SECRET_PATH.is_file():
        return SECRET_PATH.read_bytes()
    secret = secrets.token_bytes(32)
    SECRET_PATH.parent.mkdir(parents=True, exist_ok=True)
    SECRET_PATH.write_bytes(secret)
    try:
        os.chmod(SECRET_PATH, 0o600)
    except OSError:
        pass  # best effort; Windows ACLs differ
    return secret


def hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), PBKDF2_ROUNDS)
    return f"pbkdf2_sha256${PBKDF2_ROUNDS}${salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _algo, rounds, salt, expected = stored.split("$")
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), int(rounds))
    except (ValueError, TypeError):
        return False
    # Constant-time comparison: a timing-sensitive check would leak the hash
    # one byte at a time.
    return hmac.compare_digest(digest.hex(), expected)


@dataclass
class User:
    username: str
    role: str
    # Filled only for Google sign-ins; password accounts leave them empty.
    name: str = ""
    email: str = ""
    picture: str = ""

    def public(self) -> dict:
        d = {"username": self.username, "role": self.role}
        if self.email:
            d.update(name=self.name, email=self.email, picture=self.picture)
        return d


def _load_users() -> dict:
    if not USERS_PATH.is_file():
        return {}
    try:
        return json.loads(USERS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.exception("could not read user store")
        return {}


def _save_users(users: dict) -> None:
    USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    USERS_PATH.write_text(json.dumps(users, indent=2), encoding="utf-8")


def list_users() -> list[User]:
    return [User(username=name, role=rec.get("role", AUDITOR)) for name, rec in _load_users().items()]


def has_users() -> bool:
    return bool(_load_users())


def create_user(username: str, password: str, role: str) -> User:
    if role not in ROLES:
        raise ValueError(f"unknown role: {role}")
    username = username.strip().lower()
    if not username or len(password) < 8:
        raise ValueError("username is required and password must be at least 8 characters")
    users = _load_users()
    users[username] = {"password": hash_password(password), "role": role}
    _save_users(users)
    return User(username=username, role=role)


def authenticate(username: str, password: str) -> User | None:
    record = _load_users().get(username.strip().lower())
    if record is None:
        # Hash anyway so a missing user and a wrong password take the same
        # time, otherwise the response time reveals which usernames exist.
        hash_password(password)
        return None
    if not verify_password(password, record["password"]):
        return None
    return User(username=username.strip().lower(), role=record.get("role", AUDITOR))


# --- tokens ------------------------------------------------------------------

def issue_token(user: User) -> str:
    payload = {"sub": user.username, "role": user.role, "exp": time.time() + TOKEN_TTL_SEC}
    if user.email:
        payload.update(name=user.name, email=user.email, picture=user.picture)
    raw = json.dumps(payload, separators=(",", ":")).encode()
    body = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    signature = hmac.new(_secret(), body.encode(), hashlib.sha256).hexdigest()
    return f"{body}.{signature}"


def decode_token(token: str) -> User | None:
    try:
        body, signature = token.split(".", 1)
    except ValueError:
        return None
    expected = hmac.new(_secret(), body.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None  # forged or edited
    try:
        padded = body + "=" * (-len(body) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded))
    except (ValueError, TypeError):
        return None
    if payload.get("exp", 0) < time.time():
        return None
    return User(
        username=payload.get("sub", ""), role=payload.get("role", AUDITOR),
        name=payload.get("name", ""), email=payload.get("email", ""), picture=payload.get("picture", ""),
    )


# --- FastAPI dependencies ----------------------------------------------------

def _token_from(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    # Cookie fallback so <img>/<video> tags and the MJPEG stream - which cannot
    # set an Authorization header - still authenticate.
    return request.cookies.get("bw_session")


# Sign-in is off by default: every request acts as a local administrator.
# Set DRISHTI_LOGIN=1 to require accounts again - do that before exposing the
# server to any network you do not fully trust.
NO_LOGIN_USER = User(username="local-admin", role=ADMIN)


def google_client_id() -> str:
    """OAuth Web client ID. Setting it turns sign-in on (Google becomes the front door)."""
    return os.environ.get("GOOGLE_CLIENT_ID", "").strip()


def no_login() -> bool:
    return not os.environ.get("DRISHTI_LOGIN") and not google_client_id()


def current_user(request: Request) -> User:
    if no_login():
        return NO_LOGIN_USER
    token = _token_from(request)
    user = decode_token(token) if token else None
    if user is None and not has_users() and not google_client_id():
        # Before any account exists the system is unconfigured, not open: only
        # the bootstrap endpoint is reachable, and it is what creates the first
        # admin.
        raise HTTPException(401, "no accounts configured - create the first administrator")
    if user is None:
        raise HTTPException(401, "authentication required")
    return user


def require_role(*allowed: str):
    """Dependency enforcing that the caller holds one of `allowed` roles."""

    def dependency(user: User = Depends(current_user)) -> User:
        if user.role not in allowed:
            raise HTTPException(
                403, f"this action requires one of: {', '.join(allowed)} (you are {user.role})",
            )
        return user

    return dependency


def is_public_path(path: str) -> bool:
    return path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES)
