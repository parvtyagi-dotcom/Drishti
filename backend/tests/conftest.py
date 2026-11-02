"""Shared test configuration.

The API requires a session for every endpoint. These tests exercise pipeline
and API behaviour rather than access control, so they run with authentication
disabled rather than threading a login through every request - which would
test the login helper repeatedly and the thing under test not at all.

Set here (before the app is imported anywhere) rather than in individual test
modules, so a new test file cannot accidentally run without it and appear to
fail for reasons unrelated to what it checks. Authentication itself is covered
by backend/tests/test_auth.py, which turns this off deliberately.
"""
from __future__ import annotations

import os

os.environ.setdefault("BORDERWATCH_DISABLE_AUTH", "1")
# Sign-in is off by default in the app; switch it on here so test_auth.py
# checks the real login path rather than the local-admin shortcut.
os.environ["DRISHTI_LOGIN"] = "1"
# Demo cameras open video files and start capture threads; a test run should
# not be doing that in the background. They are opt-in, so this only has to
# make sure nothing in the environment has switched them on.
os.environ.pop("BORDERWATCH_DEMO_CAMERAS", None)
