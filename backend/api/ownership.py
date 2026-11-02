"""Per-account workspaces: each signed-in account sees only its own cameras and jobs.

The signed-in account is put in a context variable for the duration of each
HTTP request. The camera and job managers filter by it when it is set; code
running outside a request (camera threads, job workers, the health monitor)
has none and sees everything, which is what keeps the analysis itself
unaware of accounts.

Cameras and jobs made before workspaces existed, and the demo cameras created
at startup, have no owner. They belong to the first account that signs in
after this change, remembered in data/legacy_owner.txt.
"""
from __future__ import annotations

import os
from contextvars import ContextVar

from fastapi import Request

from backend.api.auth import NO_LOGIN_USER, _token_from, decode_token, no_login
from backend.config import DATA_DIR

current_owner: ContextVar[str | None] = ContextVar("current_owner", default=None)

_LEGACY_FILE = DATA_DIR / "legacy_owner.txt"
_legacy_owner: str | None = None


def owner_of(request: Request) -> str | None:
    if no_login():
        return NO_LOGIN_USER.username
    token = _token_from(request)
    user = decode_token(token) if token else None
    if user is None:
        return None
    return user.email or user.username


def legacy_owner() -> str | None:
    global _legacy_owner
    if _legacy_owner is None and _LEGACY_FILE.is_file():
        _legacy_owner = _LEGACY_FILE.read_text(encoding="utf-8").strip() or None
    return _legacy_owner


def _claim_legacy(owner: str) -> None:
    global _legacy_owner
    if legacy_owner() is None:
        _LEGACY_FILE.parent.mkdir(parents=True, exist_ok=True)
        _LEGACY_FILE.write_text(owner, encoding="utf-8")
        _legacy_owner = owner


def visible(item_owner: str | None) -> bool:
    """Whether the current request's account may see an item with this owner."""
    viewer = current_owner.get()
    if viewer is None:
        return True  # not inside a request: background work sees everything
    return (item_owner or legacy_owner()) == viewer


async def ownership_middleware(request: Request, call_next):
    owner = owner_of(request)
    if owner:
        _claim_legacy(owner)
    # Signed out is "", not None: it must see nothing, not everything -
    # unless auth is switched off for development, where no one has an account.
    if owner is None and os.environ.get("BORDERWATCH_DISABLE_AUTH"):
        return await call_next(request)
    token = current_owner.set(owner or "")
    try:
        return await call_next(request)
    finally:
        current_owner.reset(token)
