"""HTTP routes for the AI analyst: incident briefs, the Copilot, approvals.

Reading briefs and proposals needs only a session, like every other /api
route. Asking the Copilot and deciding on a proposal need an operator or
administrator: both can lead to changes at the post, and an auditor's role
is to read the record, not add to it.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.agent.models import ModelUnavailable
from backend.agent.service import AgentService
from backend.api.auth import ADMIN, OPERATOR, User, require_role

router = APIRouter(prefix="/api/agent", tags=["agent"])

_service: AgentService | None = None

OFF_MESSAGE = "The AI analyst is off. Start Drishti with DRISHTI_AGENTS=1 to turn it on."


def set_service(service: AgentService | None) -> None:
    global _service
    _service = service


def _require_service() -> AgentService:
    if _service is None:
        raise HTTPException(503, OFF_MESSAGE)
    return _service


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


@router.get("/status")
def status() -> dict:
    if _service is None:
        return {"enabled": False, "models": [], "message": OFF_MESSAGE}
    # In the order briefs try them; a rules-based brief follows if all fail.
    return {"enabled": True, "models": _service.models(), "message": None}


@router.get("/patterns")
def patterns(days: int = 7) -> list[dict]:
    """Repeats across days. Plain counting over the event log, so it works
    whether or not the AI analyst is switched on."""
    from backend.agent.patterns import recent_patterns
    from backend.live.camera_manager import camera_manager

    found = recent_patterns(store=camera_manager.store, days_back=max(1, min(days, 30)))
    return [p.to_dict() for p in found[:20]]


@router.get("/briefs")
def briefs(limit: int = 50) -> list[dict]:
    return _service.briefs(limit=max(1, min(limit, 200))) if _service is not None else []


@router.get("/proposals")
def proposals(status: str | None = None) -> list[dict]:
    return _service.proposals(status) if _service is not None else []


# Plain def: FastAPI runs it on a worker thread, so a model call that takes
# several seconds never blocks the event loop serving camera streams.
@router.post("/ask")
def ask(req: AskRequest, user: User = Depends(require_role(ADMIN, OPERATOR))) -> dict:
    service = _require_service()
    try:
        return service.ask(req.question, user=user.username)
    except ModelUnavailable as exc:
        raise HTTPException(503, f"No AI model is reachable right now: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def _decide(proposal_id: str, approve: bool, user: User) -> dict:
    service = _require_service()
    try:
        return service.decide(proposal_id, approve=approve, user=user.username).to_dict()
    except KeyError as exc:
        raise HTTPException(404, "no such proposal") from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/proposals/{proposal_id}/approve")
def approve(proposal_id: str, user: User = Depends(require_role(ADMIN, OPERATOR))) -> dict:
    return _decide(proposal_id, True, user)


@router.post("/proposals/{proposal_id}/reject")
def reject(proposal_id: str, user: User = Depends(require_role(ADMIN, OPERATOR))) -> dict:
    return _decide(proposal_id, False, user)
