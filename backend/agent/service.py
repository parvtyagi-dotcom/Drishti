"""Runs the agents beside the live cameras.

The service hears every camera event, groups them into incidents, and when an
incident reaches alarm level it asks the Supervisor for a brief on a
background thread. Camera threads only hand events over and return at once,
so a slow or unreachable model can never delay detection or the siren.

Proposals wait for an operator. Every brief, answer, proposal and decision is
written to the event store as an AGENT_* record at INFO severity: inside the
same hash chain as the detections, and never able to raise an alarm itself.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field

from backend.agent.agents import AgentRun, run_copilot, write_brief
from backend.agent.incidents import Incident, IncidentGrouper
from backend.agent.models import ModelChain, ModelUnavailable
from backend.agent.patterns import patterns_for_incident, recent_patterns
from backend.agent.template import template_brief
from backend.agent.tools import EventSource, other_camera_activity
from backend.core.events import INFO

logger = logging.getLogger("backend.agent.service")

PENDING, APPROVED, REJECTED = "pending", "approved", "rejected"


@dataclass
class Proposal:
    proposal_id: str
    action: str
    camera_id: str | None
    reason: str
    model: str
    source: str  # "incident" or "copilot"
    incident_id: str | None = None
    status: str = PENDING
    created_at: float = field(default_factory=time.time)
    decided_by: str | None = None
    decided_at: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Brief:
    brief_id: str
    incident_id: str
    camera_id: str
    camera_name: str
    # "ready": written by an AI model. "template": no model answered, so the
    # brief was written from fixed rules and `note` says why.
    status: str
    text: str
    model: str
    model_kind: str  # claude | cloud | local | template
    created_at: float
    incident: dict
    proposal_ids: list[str] = field(default_factory=list)
    note: str | None = None
    # The events this brief covers, so a device holding one alarm (the siren
    # phone) can find the brief that explains it.
    event_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class AgentAudit:
    """Writes agent activity into the tamper-evident event log."""

    def __init__(self, store=None, clock=time.time):
        self.store = store
        self.clock = clock

    def record(self, etype: str, reason: str, data: dict, camera_id: str | None = None,
               camera_name: str | None = None) -> dict:
        event = {
            "event_id": uuid.uuid4().hex[:12], "type": etype,
            "camera_id": camera_id or "agent", "camera_name": camera_name or "AI analyst",
            "severity": INFO, "alarm": False, "reason": reason,
            "wall_time": self.clock(), "data": data,
        }
        if self.store is not None:
            self.store.record(event)
        return event


class AgentService:
    def __init__(self, config, chain: ModelChain, store=None, manager=None, clock=time.time):
        self.config = config
        self.chain = chain
        self.manager = manager
        self.clock = clock
        self.source = EventSource(store=store, manager=manager, clock=clock)
        self.audit = AgentAudit(store=store, clock=clock)
        self._grouper = IncidentGrouper(gap_sec=config.incident_gap_sec)
        self._lock = threading.Lock()
        self._scheduled: set[str] = set()
        self._briefs: list[Brief] = []
        self._proposals: dict[str, Proposal] = {}
        self._queue: queue.Queue[Incident | None] = queue.Queue()
        self._worker: threading.Thread | None = None

    # --- lifecycle --------------------------------------------------------------

    def start(self) -> None:
        if self._worker is not None:
            return
        self._worker = threading.Thread(target=self._work, name="agent-briefs", daemon=True)
        self._worker.start()
        if self.manager is not None:
            self.manager.subscribe(self.on_event)

    def stop(self) -> None:
        if self.manager is not None:
            self.manager.unsubscribe(self.on_event)
        if self._worker is not None:
            self._queue.put(None)
            self._worker.join(timeout=5)
            self._worker = None

    # --- incidents ----------------------------------------------------------------

    def on_event(self, event: dict) -> None:
        """Called on camera threads for every event: group it, return at once."""
        with self._lock:
            incident = self._grouper.add(event)
            if incident is None or not incident.needs_brief or incident.incident_id in self._scheduled:
                return
            self._scheduled.add(incident.incident_id)
        # Wait briefly so the brief includes what happened right after the alarm.
        timer = threading.Timer(self.config.brief_delay_sec, self._queue.put, args=(incident,))
        timer.daemon = True
        timer.start()

    def _work(self) -> None:
        while True:
            incident = self._queue.get()
            if incident is None:
                return
            try:
                self.brief_incident(incident)
            except Exception:  # one failed brief must not stop later ones
                logger.exception("brief failed for incident %s", incident.incident_id)

    def brief_incident(self, incident: Incident) -> Brief:
        with self._lock:
            snapshot = incident.for_agent()
        history = patterns_for_incident(
            recent_patterns(self.source.store, self.manager, now=self.clock()), snapshot)
        written = model = None
        note = None
        try:
            written, model = self.chain.run(lambda m: write_brief(
                m, snapshot, self.source, self.manager, max_steps=self.config.max_steps, history=history))
        except ModelUnavailable as exc:
            note = f"No AI model answered: {exc}"
        except Exception:  # the operator still gets a brief; the bug goes to the log
            logger.exception("AI brief failed for incident %s", incident.incident_id)
            note = "The AI brief failed with an internal error (see the server log)"
        if written is not None and not written.model_contributed:
            note = f"{written.run.model} gave no usable judgment ({written.run.stopped})"

        if note is None:
            run = written.run
            proposals = self._keep_proposals(run, "incident", incident.incident_id)
            brief = Brief(
                brief_id=uuid.uuid4().hex[:10], incident_id=incident.incident_id,
                camera_id=incident.camera_id, camera_name=incident.camera_name, status="ready",
                text=written.text, model=run.model, model_kind=getattr(model, "kind", "cloud"),
                created_at=self.clock(), incident=snapshot, proposal_ids=[p.proposal_id for p in proposals],
            )
            data = {"model": run.model, "stopped": run.stopped, "steps": run.steps, "model_answer": run.text}
            reason = f"AI judgment by {run.model} on rules-written facts."
        else:
            brief = Brief(
                brief_id=uuid.uuid4().hex[:10], incident_id=incident.incident_id,
                camera_id=incident.camera_id, camera_name=incident.camera_name, status="template",
                text=template_brief(snapshot, other_camera_activity(self.source, snapshot), history), model="template", model_kind="template",
                created_at=self.clock(), incident=snapshot, note=note,
            )
            data = {"model": "template", "note": note}
            reason = "Brief written from rules; no AI model was available."
        brief.event_ids = [e.get("event_id") for e in incident.events if e.get("event_id")]
        self.audit.record("AGENT_BRIEF", reason,
                          {"incident_id": incident.incident_id, "brief_id": brief.brief_id, "text": brief.text, **data},
                          incident.camera_id, incident.camera_name)
        with self._lock:
            self._briefs.append(brief)
            del self._briefs[:-200]
        return brief

    # --- copilot ----------------------------------------------------------------

    def ask(self, question: str, user: str) -> dict:
        question = question.strip()[:1000]
        if not question:
            raise ValueError("question is empty")
        run, model = self.chain.run(lambda m: run_copilot(
            m, question, self.source, self.manager, max_steps=self.config.max_steps))
        proposals = self._keep_proposals(run, "copilot", None)
        self.audit.record("AGENT_ANSWER", f"Copilot answered {user}.",
                          {"question": question, "answer": run.text, "model": run.model, "user": user,
                           "steps": run.steps, "proposal_ids": [p.proposal_id for p in proposals]})
        return {"answer": run.text, "model": run.model, "model_kind": getattr(model, "kind", "cloud"),
                "proposals": [p.to_dict() for p in proposals]}

    def models(self) -> list[dict]:
        """The models briefs are tried on, in order."""
        return [{"name": m.name, "kind": getattr(m, "kind", "cloud")} for m in self.chain.models]

    # --- proposals ----------------------------------------------------------------

    def _keep_proposals(self, run: AgentRun, source: str, incident_id: str | None) -> list[Proposal]:
        kept = []
        for item in run.proposals:
            proposal = Proposal(
                proposal_id=uuid.uuid4().hex[:10], action=item.action, camera_id=item.camera_id,
                reason=item.reason, model=run.model, source=source, incident_id=incident_id,
                created_at=self.clock(),
            )
            with self._lock:
                self._proposals[proposal.proposal_id] = proposal
            self.audit.record("AGENT_PROPOSAL", f"AI proposed {item.action}: {item.reason}", proposal.to_dict(), item.camera_id)
            kept.append(proposal)
        return kept

    def decide(self, proposal_id: str, approve: bool, user: str) -> Proposal:
        with self._lock:
            proposal = self._proposals.get(proposal_id)
            if proposal is None:
                raise KeyError(proposal_id)
            if proposal.status != PENDING:
                raise ValueError(f"proposal already {proposal.status}")
            proposal.status = APPROVED if approve else REJECTED
            proposal.decided_by = user
            proposal.decided_at = self.clock()
        verb = "approved" if approve else "rejected"
        self.audit.record("AGENT_APPROVED" if approve else "AGENT_REJECTED",
                          f"{user} {verb} AI proposal to {proposal.action}.", proposal.to_dict(), proposal.camera_id)
        return proposal

    # --- reading ----------------------------------------------------------------

    def briefs(self, limit: int = 50) -> list[dict]:
        with self._lock:
            return [b.to_dict() for b in reversed(self._briefs[-limit:])]

    def brief_for_event(self, event_id: str) -> dict | None:
        """The newest brief covering this event, if one has been written."""
        with self._lock:
            for brief in reversed(self._briefs):
                if event_id in brief.event_ids:
                    return brief.to_dict()
        return None

    def proposals(self, status: str | None = None) -> list[dict]:
        with self._lock:
            items = [p for p in self._proposals.values() if status is None or p.status == status]
        return [p.to_dict() for p in sorted(items, key=lambda p: p.created_at, reverse=True)]
