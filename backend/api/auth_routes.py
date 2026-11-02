"""Login, session and user-management endpoints."""
from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from backend.api.auth import (
    ADMIN, AUDITOR, OPERATOR, ROLES, TOKEN_TTL_SEC, User, authenticate, create_user,
    current_user, google_client_id, has_users, issue_token, list_users, require_role,
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


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        "bw_session", token,
        max_age=TOKEN_TTL_SEC,
        httponly=True,       # not readable by scripts, so XSS cannot lift it
        samesite="lax",
        # Not forced secure: the edge deployment is often plain http on a
        # closed LAN, and a cookie marked secure would simply never be sent.
        secure=False,
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
    if not has_users() and not google_client_id():
        return {"configured": False, "authenticated": False, "user": None}
    return {"configured": True, "authenticated": False, "user": None}


@router.post("/bootstrap", status_code=201)
def bootstrap_admin(req: CreateUserRequest, response: Response) -> dict:
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
    token = issue_token(user)
    _set_session_cookie(response, token)
    return {"token": token, "user": {"username": user.username, "role": user.role}}


@router.post("/login")
def login(req: LoginRequest, response: Response) -> dict:
    user = authenticate(req.username, req.password)
    if user is None:
        # One message for both failure modes, so it cannot be used to discover
        # which usernames are valid.
        raise HTTPException(401, "invalid username or password")
    token = issue_token(user)
    _set_session_cookie(response, token)
    return {"token": token, "user": {"username": user.username, "role": user.role}}


@router.post("/google")
def google_login(req: GoogleLoginRequest, response: Response) -> dict:
    """Exchange a Google ID token (from Sign in with Google) for a session.

    The token's signature, expiry and audience are verified against Google's
    public keys; nothing the browser claims about the user is trusted.
    Optional restrictions (empty = allow everyone):
        GOOGLE_ALLOWED_EMAILS   comma-separated exact addresses
        GOOGLE_ALLOWED_DOMAINS  comma-separated domains, e.g. example.com
    Role for Google users: GOOGLE_DEFAULT_ROLE (default ADMIN, for demos).
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
    emails, domains = _csv_env("GOOGLE_ALLOWED_EMAILS"), _csv_env("GOOGLE_ALLOWED_DOMAINS")
    if (emails or domains) and email not in emails and email.rsplit("@", 1)[-1] not in domains:
        raise HTTPException(403, "this Google account is not allowed to use Drishti")

    role = os.environ.get("GOOGLE_DEFAULT_ROLE", ADMIN).strip().upper()
    if role not in ROLES:
        role = ADMIN
    user = User(
        username=email, role=role, name=str(info.get("name") or email),
        email=email, picture=str(info.get("picture") or ""),
    )
    token = issue_token(user)
    _set_session_cookie(response, token)
    return {"token": token, "user": user.public()}


@router.post("/logout")
def logout(response: Response) -> dict:
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
