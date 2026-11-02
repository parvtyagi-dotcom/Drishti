"""Authentication, token integrity and role enforcement.

These run against the real functions rather than mocks: a password check or a
signature check that is only verified against a stand-in has not been verified
at all.
"""
from __future__ import annotations

import time

import pytest

from backend.api import auth


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Point the user store and signing key at a temp directory so tests never
    read or overwrite a real deployment's credentials."""
    monkeypatch.setattr(auth, "USERS_PATH", tmp_path / "users.json")
    monkeypatch.setattr(auth, "SECRET_PATH", tmp_path / "secret")
    yield


def test_password_is_not_recoverable_from_its_hash():
    stored = auth.hash_password("border-watch-2026")
    assert "border-watch-2026" not in stored
    assert stored.startswith("pbkdf2_sha256$")


def test_password_verifies_only_against_the_right_password():
    stored = auth.hash_password("correct horse battery")
    assert auth.verify_password("correct horse battery", stored)
    assert not auth.verify_password("Correct horse battery", stored)
    assert not auth.verify_password("", stored)


def test_same_password_hashes_differently_each_time():
    # Distinct salts, so identical passwords do not produce identical hashes
    # and one cracked hash does not reveal every account using that password.
    assert auth.hash_password("repeated") != auth.hash_password("repeated")


def test_malformed_hash_is_rejected_rather_than_crashing():
    assert not auth.verify_password("anything", "not-a-real-hash")


def test_authenticate_rejects_unknown_user_and_wrong_password():
    auth.create_user("guard", "guardpass123", auth.OPERATOR)
    assert auth.authenticate("guard", "guardpass123") is not None
    assert auth.authenticate("guard", "wrong") is None
    assert auth.authenticate("ghost", "guardpass123") is None


def test_username_is_case_insensitive():
    auth.create_user("Commander", "commanderpass", auth.ADMIN)
    assert auth.authenticate("COMMANDER", "commanderpass") is not None


def test_short_password_is_refused():
    with pytest.raises(ValueError):
        auth.create_user("weak", "short", auth.OPERATOR)


def test_unknown_role_is_refused():
    with pytest.raises(ValueError):
        auth.create_user("someone", "longenoughpass", "SUPERUSER")


def test_token_round_trips_identity_and_role():
    user = auth.User(username="commander", role=auth.ADMIN)
    decoded = auth.decode_token(auth.issue_token(user))
    assert decoded is not None
    assert decoded.username == "commander"
    assert decoded.role == auth.ADMIN


def test_edited_token_is_rejected():
    """The payload is readable but signed; changing it must invalidate it -
    otherwise any operator could promote themselves to admin."""
    token = auth.issue_token(auth.User(username="guard", role=auth.OPERATOR))
    body, signature = token.split(".", 1)
    import base64, json

    payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    payload["role"] = auth.ADMIN
    forged_body = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode()
    ).decode().rstrip("=")
    assert auth.decode_token(f"{forged_body}.{signature}") is None


def test_token_with_wrong_signature_is_rejected():
    token = auth.issue_token(auth.User(username="guard", role=auth.OPERATOR))
    body, _ = token.split(".", 1)
    assert auth.decode_token(f"{body}.{'0' * 64}") is None


def test_expired_token_is_rejected(monkeypatch):
    monkeypatch.setattr(auth, "TOKEN_TTL_SEC", -1)
    token = auth.issue_token(auth.User(username="guard", role=auth.OPERATOR))
    assert auth.decode_token(token) is None


def test_garbage_tokens_are_rejected():
    for bad in ("", "no-dot", "a.b.c.d", "...", "abc."):
        assert auth.decode_token(bad) is None


def test_public_paths_cover_devices_that_cannot_log_in():
    # A phone scanning a QR code reaches these before any session exists.
    assert auth.is_public_path("/api/live/siren")
    assert auth.is_public_path("/api/live/siren/icon-192.png")
    assert auth.is_public_path("/api/live/phone")
    # Everything carrying footage or records must not be public.
    assert not auth.is_public_path("/api/live/cameras")
    assert not auth.is_public_path("/api/live/evidence")
    assert not auth.is_public_path("/api/live/audit")
    assert not auth.is_public_path("/api/live/cameras/cam-01/stream")
