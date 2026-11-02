"""The AI analyst's HTTP routes, through the real FastAPI app.

A service with a scripted model stands in for a live one, so these check the
routes themselves: what an operator, an auditor and a switched-off system each
get back.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.agent.models import ModelChain, ModelUnavailable
from backend.agent.service import AgentService
from backend.api import agent as agent_api
from backend.api.auth import AUDITOR, OPERATOR, User, current_user
from backend.app import app
from backend.config import AgentConfig
from backend.tests.test_agent import FakeStore, ScriptedModel, answer, calls, event

client = TestClient(app)


@pytest.fixture
def as_role():
    def use(role: str):
        app.dependency_overrides[current_user] = lambda: User(username=f"{role.lower()}-1", role=role)
    yield use
    app.dependency_overrides.pop(current_user, None)


@pytest.fixture
def service():
    def build(script):
        svc = AgentService(AgentConfig(enabled=True, brief_delay_sec=0.0), ModelChain([ScriptedModel("cloud", script)]),
                           store=FakeStore())
        agent_api.set_service(svc)
        return svc
    yield build
    agent_api.set_service(None)


def test_switched_off_analyst_says_how_to_turn_it_on(as_role):
    as_role(OPERATOR)
    status = client.get("/api/agent/status").json()
    assert status["enabled"] is False and "DRISHTI_AGENTS=1" in status["message"]
    assert client.get("/api/agent/briefs").json() == []
    response = client.post("/api/agent/ask", json={"question": "Anything at CAM-01?"})
    assert response.status_code == 503 and "DRISHTI_AGENTS=1" in response.json()["detail"]


def test_operator_sees_brief_and_approves_its_proposal(as_role, service):
    as_role(OPERATOR)
    svc = service([
        calls(("propose_action", {"action": "review_footage", "camera_id": "cam-01", "reason": "Clip shows the crossing."})),
        answer("Situation: one person crossed.\nEvidence: ZONE_ENTRY.\nNext step: review the clip.", model="cloud"),
    ])
    svc.brief_incident(svc._grouper.add(event("ZONE_ENTRY", "CRITICAL", 1000.0)))

    briefs = client.get("/api/agent/briefs").json()
    assert briefs[0]["status"] == "ready" and briefs[0]["text"].startswith("Situation:")
    pending = client.get("/api/agent/proposals", params={"status": "pending"}).json()
    assert len(pending) == 1

    pid = pending[0]["proposal_id"]
    decided = client.post(f"/api/agent/proposals/{pid}/approve").json()
    assert decided["status"] == "approved" and decided["decided_by"] == "operator-1"
    again = client.post(f"/api/agent/proposals/{pid}/reject")
    assert again.status_code == 409
    assert client.post("/api/agent/proposals/nope/approve").status_code == 404


def test_auditor_can_read_but_not_ask_or_decide(as_role, service):
    service([])
    as_role(AUDITOR)
    assert client.get("/api/agent/briefs").status_code == 200
    assert client.post("/api/agent/ask", json={"question": "hi"}).status_code == 403
    assert client.post("/api/agent/proposals/x/approve").status_code == 403


def test_copilot_answers_and_reports_when_no_model_is_reachable(as_role, service):
    as_role(OPERATOR)
    service([answer("Nothing unusual in the last hour.", model="cloud")])
    reply = client.post("/api/agent/ask", json={"question": "Anything unusual?"})
    assert reply.status_code == 200 and reply.json()["answer"] == "Nothing unusual in the last hour."

    service([ModelUnavailable("offline")])
    down = client.post("/api/agent/ask", json={"question": "Anything unusual?"})
    assert down.status_code == 503 and "offline" in down.json()["detail"]


def test_patterns_work_even_with_the_analyst_off(as_role, monkeypatch):
    from backend.agent.patterns import Pattern

    as_role(OPERATOR)
    seen = {}

    def fake(store=None, manager=None, days_back=7, now=None):
        seen["days"] = days_back
        return [Pattern(kind="plate", key="TEST123", summary="Plate TEST123 read 3 times on 2 days.", sightings=3, days=2)]

    monkeypatch.setattr("backend.agent.patterns.recent_patterns", fake)
    body = client.get("/api/agent/patterns", params={"days": 90}).json()
    assert body[0]["key"] == "TEST123" and seen["days"] == 30


def test_empty_question_is_rejected(as_role, service):
    as_role(OPERATOR)
    service([])
    assert client.post("/api/agent/ask", json={"question": ""}).status_code == 422
