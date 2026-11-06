"""Login, session and user-management endpoints."""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict, deque

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from backend.api.auth import (
    ADMIN, AUDITOR, OPERATOR, ROLES, TOKEN_TTL_SEC, User, authenticate, create_user,
    _token_from, current_user, google_client_id, has_users, issue_token, list_users, require_role,
    revoke_token,
)

logger = logging.getLogger("backend.api.auth_routes")
router = APIRouter(prefix="/api/auth", tags=["auth"])


class GoogleLoginRequest(BaseModel):
    credential: str = Field(min_length=10, max_length=8192)


def _csv_env(name: str) -> set[str]:
    return {v.strip().lower() for v in os.environ.get(name, "").split(",") if v.strip()}


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class CreateUserRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=8, max_length=256)
    role: str = Field(default=OPERATOR)


class LoginThrottle:
    """Refuse further logins from one client after repeated failures.

    Each attempt costs a 240k-round PBKDF2, so an unthrottled login endpoint
    was both a password-guessing oracle and a way to pin the CPU the camera
    analysis needs. Kept in memory: a restart clearing it is acceptable for a
    single-site edge server.
    """

    MAX_FAILURES = 5
    WINDOW_SEC = 300.0
    # Bound on clients tracked at once, so rotating source addresses cannot
    # grow this without limit; the oldest entries are dropped first.
    MAX_TRACKED = 10_000

    def __init__(self):
        self._failures: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _recent(self, key: str, now: float) -> deque[float]:
        attempts = self._failures.get(key)
        if attempts is None:
            return deque()
        while attempts and now - attempts[0] > self.WINDOW_SEC:
            attempts.popleft()
        if not attempts:
            del self._failures[key]  # forget clients whose failures have aged out
            return deque()
        return attempts

    def blocked(self, key: str) -> bool:
        with self._lock:
            return len(self._recent(key, time.monotonic())) >= self.MAX_FAILURES

    def failed(self, key: str) -> None:
        with self._lock:
            now = time.monotonic()
            self._recent(key, now)
            self._failures[key].append(now)
            while len(self._failures) > self.MAX_TRACKED:
                self._failures.pop(next(iter(self._failures)))

    def succeeded(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)

    def reset(self) -> None:
        with self._lock:
            self._failures.clear()


login_throttle = LoginThrottle()


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def google_default_role() -> str:
    """Role for a Google sign-in. Read-only unless configured otherwise: an
    account that passes the allowlist should not become an administrator by
    default."""
    role = os.environ.get("GOOGLE_DEFAULT_ROLE", AUDITOR).strip().upper()
    return role if role in ROLES else AUDITOR


def google_email_allowed(email: str) -> bool:
    """Only listed addresses or domains may sign in. An empty allowlist used to
    mean "every Google account on earth"; now it means nobody, until one is set."""
    emails, domains = _csv_env("GOOGLE_ALLOWED_EMAILS"), _csv_env("GOOGLE_ALLOWED_DOMAINS")
    return email in emails or email.rsplit("@", 1)[-1] in domains


def _set_session_cookie(request: Request, response: Response, token: str) -> None:
    response.set_cookie(
        "bw_session", token,
        max_age=TOKEN_TTL_SEC,
        httponly=True,       # not readable by scripts, so XSS cannot lift it
        samesite="lax",
        # Secure whenever the request itself came over https. Plain-http LAN
        # deployments still work (a secure cookie would never be sent there),
        # but an https deployment no longer exposes the session over http.
        secure=request.url.scheme == "https",
        path="/",
    )


@router.get("/me")
def whoami(request: Request) -> dict:
    """Session state. Deliberately unauthenticated so the UI can tell the
    difference between 'logged out' and 'not set up yet' before showing a form."""
    from backend.api.auth import NO_LOGIN_USER, _token_from, decode_token, no_login

    if no_login():
        return {"configured": True, "authenticated": True,
                "user": {"username": NO_LOGIN_USER.username, "role": NO_LOGIN_USER.role}}
    token = _token_from(request)
    user = decode_token(token) if token else None
    if user is not None:
        return {"configured": True, "authenticated": True, "user": user.public()}
    # "Configured" means a local administrator exists. Google sign-in alone
    # does not count: its accounts are read-only by default and may be
    # restricted to nobody, so the first-run screen must still appear.
    if not has_users():
        return {"configured": False, "authenticated": False, "user": None}
    return {"configured": True, "authenticated": False, "user": None}


@router.post("/bootstrap", status_code=201)
def bootstrap_admin(req: CreateUserRequest, request: Request, response: Response) -> dict:
    """Create the first administrator.

    Only available while no account exists, so it cannot be used later to mint
    an admin on a running system.
    """
    if has_users():
        raise HTTPException(409, "already configured - sign in instead")
    try:
        user = create_user(req.username, req.password, ADMIN)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    _set_session_cookie(request, response, issue_token(user))
    # The session lives only in the httpOnly cookie. Returning the token in
    # the body as well handed it to page scripts, which is what httpOnly is
    # there to prevent.
    return {"user": {"username": user.username, "role": user.role}}


@router.post("/login")
def login(req: LoginRequest, request: Request, response: Response) -> dict:
    client = _client_key(request)
    if login_throttle.blocked(client):
        raise HTTPException(429, "too many failed sign-in attempts - wait a few minutes and try again")
    user = authenticate(req.username, req.password)
    if user is None:
        login_throttle.failed(client)
        # One message for both failure modes, so it cannot be used to discover
        # which usernames are valid.
        raise HTTPException(401, "invalid username or password")
    login_throttle.succeeded(client)
    _set_session_cookie(request, response, issue_token(user))
    return {"user": {"username": user.username, "role": user.role}}


@router.post("/google")
def google_login(req: GoogleLoginRequest, request: Request, response: Response) -> dict:
    """Exchange a Google ID token (from Sign in with Google) for a session.

    The token's signature, expiry and audience are verified against Google's
    public keys; nothing the browser claims about the user is trusted.
    Required allowlist (empty = nobody may sign in with Google):
        GOOGLE_ALLOWED_EMAILS   comma-separated exact addresses
        GOOGLE_ALLOWED_DOMAINS  comma-separated domains, e.g. example.com
    Role for Google users: GOOGLE_DEFAULT_ROLE (default AUDITOR, read-only).
    """
    client_id = google_client_id()
    if not client_id:
        raise HTTPException(503, "Google sign-in is not configured (GOOGLE_CLIENT_ID is unset)")
    try:
        from google.auth.transport import requests as g_requests
        from google.oauth2 import id_token
    except ImportError:
        raise HTTPException(503, "google-auth is not installed on the server")
    try:
        # A PC clock a second or two behind Google's rejects fresh tokens as
        # "used too early" - tolerate small drift.
        info = id_token.verify_oauth2_token(
            req.credential, g_requests.Request(), client_id, clock_skew_in_seconds=60,
        )
    except ValueError as exc:
        logger.warning("rejected Google token: %s", exc)
        raise HTTPException(401, "invalid Google sign-in")
    except Exception:
        logger.exception("could not verify Google token")
        raise HTTPException(502, "could not reach Google to verify sign-in")

    email = str(info.get("email", "")).lower()
    if not email or not info.get("email_verified"):
        raise HTTPException(401, "Google account email is not verified")
    if not google_email_allowed(email):
        raise HTTPException(
            403, "this Google account is not allowed to use Drishti "
                 "(an administrator must list it in GOOGLE_ALLOWED_EMAILS or GOOGLE_ALLOWED_DOMAINS)",
        )

    role = google_default_role()
    user = User(
        username=email, role=role, name=str(info.get("name") or email),
        email=email, picture=str(info.get("picture") or ""),
    )
    _set_session_cookie(request, response, issue_token(user))
    return {"user": user.public()}


@router.post("/logout")
def logout(request: Request, response: Response) -> dict:
    revoke_token(_token_from(request))
    response.delete_cookie("bw_session", path="/")
    return {"ok": True}


@router.get("/users")
def get_users(_: User = Depends(require_role(ADMIN))) -> list[dict]:
    return [{"username": u.username, "role": u.role} for u in list_users()]


@router.post("/users", status_code=201)
def add_user(req: CreateUserRequest, _: User = Depends(require_role(ADMIN))) -> dict:
    if req.role not in ROLES:
        raise HTTPException(400, f"role must be one of: {', '.join(ROLES)}")
    try:
        user = create_user(req.username, req.password, req.role)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"username": user.username, "role": user.role}


@router.get("/roles")
def roles() -> dict:
    return {
        "roles": [
            {"role": ADMIN, "description": "Full control, including cameras, users and deletion"},
            {"role": OPERATOR, "description": "Monitor and configure analysis; cannot delete records"},
            {"role": AUDITOR, "description": "Read-only, including the audit trail"},
        ],
    }
