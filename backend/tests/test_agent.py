"""The AI analyst layer, tested without any real model.

A scripted fake model stands in for Claude and the local model, so these
tests check what the agent code itself guarantees: the tool loop, falling
back when a model is unreachable, proposals waiting for approval, and every
agent step landing in the audit log without ever raising an alarm.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import anthropic
import httpx
import httpx2
import pytest

from backend.agent.agents import run_agent, run_supervisor
from backend.agent.incidents import IncidentGrouper
from backend.agent.models import (
    ClaudeModel,
    HostedModel,
    ModelChain,
    ModelUnavailable,
    OllamaModel,
    build_chain,
    ToolCall,
    ToolResult,
    ToolSpec,
    Turn,
)
from backend.agent.service import AgentService
from backend.agent.tools import EventSource
from backend.config import AgentConfig


# --- fakes ----------------------------------------------------------------------

class ScriptedConversation:
    def __init__(self, model, system, tools):
        self.model = model
        self.system = system
        self.tools = tools

    def _next(self) -> Turn:
        step = self.model.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    def send(self, text):
        self.model.sent.append(("user", text))
        return self._next()

    def send_tool_results(self, results):
        self.model.sent.append(("tools", results))
        return self._next()


class ScriptedModel:
    def __init__(self, name, script):
        self.name = name
        self.script = list(script)
        self.sent = []
        self.starts = 0

    def start(self, system, tools):
        self.starts += 1
        return ScriptedConversation(self, system, tools)


def answer(text, model="fake"):
    return Turn(text=text, tool_calls=[], stop="done", model=model)


def calls(*items, model="fake"):
    return Turn(text="", tool_calls=[ToolCall(f"c{i}", name, args) for i, (name, args) in enumerate(items)],
                stop="tool_use", model=model)


class FakeStore:
    def __init__(self, events=()):
        self.recorded = []
        self.events = list(events)

    def record(self, event):
        self.recorded.append(event)

    def query(self, limit=100, camera_id=None, since=None, **_):
        out = [e for e in self.events
               if (camera_id is None or e["camera_id"] == camera_id) and (since is None or e["wall_time"] >= since)]
        return out[:limit]


def event(etype, severity, at, camera="cam-01", track=1, **extra):
    return {"type": etype, "severity": severity, "wall_time": at, "camera_id": camera,
            "camera_name": camera.upper(), "track_id": track, "reason": f"{etype} on {camera}", **extra}


# --- incidents --------------------------------------------------------------------

def test_related_events_on_one_camera_form_one_incident():
    grouper = IncidentGrouper(gap_sec=60)
    first = grouper.add(event("ZONE_APPROACH", "MEDIUM", 100.0))
    second = grouper.add(event("ZONE_ENTRY", "CRITICAL", 105.0))
    third = grouper.add(event("LOITERING", "MEDIUM", 140.0))
    assert first is second is third
    assert first.needs_brief and first.max_severity == "CRITICAL"
    assert [e["type"] for e in first.events] == ["ZONE_APPROACH", "ZONE_ENTRY", "LOITERING"]


def test_quiet_gap_and_other_cameras_start_new_incidents():
    grouper = IncidentGrouper(gap_sec=60)
    a = grouper.add(event("ZONE_ENTRY", "CRITICAL", 100.0))
    b = grouper.add(event("ZONE_ENTRY", "CRITICAL", 101.0, camera="cam-02"))
    c = grouper.add(event("ZONE_ENTRY", "CRITICAL", 200.0))
    assert a is not b and a is not c


def test_routine_and_agent_events_never_open_incidents():
    grouper = IncidentGrouper()
    assert grouper.add(event("VEHICLE_DETECTED", "INFO", 1.0)) is None
    assert grouper.add(event("ABNORMAL_SPEED", "LOW", 2.0)) is None  # LOW alone is routine
    assert grouper.add(event("AGENT_BRIEF", "CRITICAL", 3.0)) is None


def test_low_events_join_an_open_incident():
    grouper = IncidentGrouper()
    incident = grouper.add(event("LOITERING", "MEDIUM", 1.0))
    assert grouper.add(event("SUDDEN_DIRECTION_CHANGE", "LOW", 5.0)) is incident
    assert not incident.needs_brief


# --- the tool loop --------------------------------------------------------------------

def echo_tool(log):
    def handler(args):
        log.append(args)
        return {"seen": args}
    return ToolSpec("echo", "Echo", {"type": "object", "properties": {}, "required": [], "additionalProperties": False}, handler)


def test_agent_runs_requested_tools_and_returns_the_answer():
    log = []
    model = ScriptedModel("fake", [calls(("echo", {"x": 1})), answer("All clear.")])
    run = run_agent(model, "test", "system", [echo_tool(log)], "task")
    assert run.text == "All clear." and run.stopped == "done"
    assert log == [{"x": 1}]
    kind, results = model.sent[1]
    assert kind == "tools" and json.loads(results[0].content) == {"seen": {"x": 1}}


def test_unknown_and_failing_tools_are_reported_back_not_raised():
    def broken(_):
        raise RuntimeError("database locked")
    tools = [ToolSpec("broken", "Fails", {}, broken)]
    model = ScriptedModel("fake", [calls(("missing", {}), ("broken", {})), answer("Done.")])
    run = run_agent(model, "test", "system", tools, "task")
    results = model.sent[1][1]
    assert results[0].is_error and "Unknown tool" in results[0].content
    assert results[1].is_error and "database locked" in results[1].content
    assert run.text == "Done."


def test_agent_stops_after_max_steps():
    model = ScriptedModel("fake", [calls(("echo", {}))] * 4)
    run = run_agent(model, "test", "system", [echo_tool([])], "task", max_steps=3)
    assert run.stopped == "step_limit"


def test_supervisor_delegates_to_pursuit_and_collects_proposals():
    model = ScriptedModel("fake", [
        calls(("ask_pursuit", {"question": "Where did track 1 go?"})),
        answer("Track 1 reached CAM-02 at 22:41:40."),            # pursuit's answer
        calls(("propose_action", {"action": "enable_person_id", "camera_id": "cam-02", "reason": "Seen heading there."})),
        answer("Situation: crossing.\nEvidence: ZONE_ENTRY.\nNext step: watch CAM-02."),
    ])
    incident = {"incident_id": "i1", "events": []}
    run = run_supervisor(model, incident, EventSource(), manager=None)
    assert run.text.startswith("Situation:")
    assert model.starts == 2  # supervisor + pursuit, each its own conversation
    assert len(run.proposals) == 1 and run.proposals[0].camera_id == "cam-02"
    assert any(step.get("agent") == "pursuit" for step in run.steps)


def test_invalid_proposals_are_rejected_back_to_the_agent():
    model = ScriptedModel("fake", [
        calls(("propose_action", {"action": "fire_weapon", "camera_id": None, "reason": "x"})),
        answer("Brief."),
    ])
    run = run_supervisor(model, {"incident_id": "i1"}, EventSource(), manager=None)
    assert run.proposals == []
    assert model.sent[1][1][0].is_error


# --- facts from rules, judgment from the model ---------------------------------------------

def incident_with_history():
    grouper = IncidentGrouper()
    incident = grouper.add(event("ZONE_ENTRY", "CRITICAL", 1000.0, reason="Track #7 crossed into 'Restricted'"))
    earlier = [event("TARGET_CONFIRMED", "HIGH", 700.0, camera="cam-02", track=3, reason="Track #3 matched"),
               event("VEHICLE_DETECTED", "INFO", 800.0, camera="cam-02"),
               event("ZONE_ENTRY", "CRITICAL", 990.0, reason="same camera")]
    source = EventSource(store=FakeStore(earlier), clock=lambda: 1030.0)
    return incident.for_agent(), source


def test_other_camera_activity_lists_only_notable_events_elsewhere():
    from backend.agent.tools import other_camera_activity

    snapshot, source = incident_with_history()
    activity = other_camera_activity(source, snapshot)
    assert [(e["camera_id"], e["type"]) for e in activity] == [("cam-02", "TARGET_CONFIRMED")]


def test_model_lines_are_added_to_rule_written_facts():
    from backend.agent.agents import write_brief

    snapshot, source = incident_with_history()
    model = ScriptedModel("local", [answer(
        "**Situation:** a vehicle entered\n- **Links:** The watched person was confirmed on CAM-02 five minutes earlier.\n"
        "Next step: Turn on Person ID on CAM-01.", model="local")])
    written = write_brief(model, snapshot, source, manager=None)
    lines = written.text.split("\n")
    assert lines[0].startswith("Situation: Restricted zone entered on CAM-01")  # rules, not "a vehicle"
    # What other cameras saw is a rules-written fact; the model's own "Links" claim is not used.
    assert lines[2].startswith("Other cameras: CAM-02: ") and "Watched person identified (track #3)" in lines[2]
    assert lines[3] == "Next step: Turn on Person ID on CAM-01."
    assert written.model_contributed
    task = model.sent[0][1]
    assert "Other cameras: CAM-02" in task and "Facts already shown to the operator" in task


def test_missing_next_step_falls_back_to_the_standard_one():
    from backend.agent.agents import write_brief

    snapshot, source = incident_with_history()
    written = write_brief(ScriptedModel("local", [answer("Links: none found.", model="local")]), snapshot, source, manager=None)
    assert written.text.split("\n")[-1].startswith("Next step: Check the live view of CAM-01")
    assert not written.model_contributed


def test_a_proposal_alone_counts_as_the_model_contributing():
    from backend.agent.agents import write_brief

    snapshot, source = incident_with_history()
    model = ScriptedModel("local", [
        calls(("propose_action", {"action": "enable_person_id", "camera_id": "cam-01", "reason": "Watched person seen on CAM-02."})),
        answer("ok", model="local"),
    ])
    written = write_brief(model, snapshot, source, manager=None)
    assert written.model_contributed and written.run.proposals[0].action == "enable_person_id"


def test_proposals_are_capped_deduplicated_and_null_cameras_cleaned():
    def propose(action, camera):
        return ("propose_action", {"action": action, "camera_id": camera, "reason": "r"})

    model = ScriptedModel("fake", [
        calls(propose("review_footage", "cam-01"), propose("review_footage", "cam-01"),
              propose("dispatch_patrol", "null"), propose("raise_alert_level", None)),
        answer("Next step: x."),
    ])
    run = run_supervisor(model, {"incident_id": "i"}, EventSource(), manager=None)
    assert [(p.action, p.camera_id) for p in run.proposals] == [("review_footage", "cam-01"), ("dispatch_patrol", None)]
    results = model.sent[1][1]
    assert "Already proposed" in results[1].content
    assert results[3].is_error and "at most 2" in results[3].content


def test_local_models_run_lean_without_pursuit():
    from backend.agent.agents import LEAN_MAX_STEPS, write_brief

    snapshot, source = incident_with_history()
    local = ScriptedModel("local", [calls(("ask_pursuit", {"question": "q"}))] * 5)
    local.kind = "local"
    written = write_brief(local, snapshot, source, manager=None)
    assert local.starts == 1  # Pursuit was never started
    assert written.run.stopped == "step_limit" and len(written.run.steps) == LEAN_MAX_STEPS
    assert all(step["error"] for step in written.run.steps)  # ask_pursuit is not offered in lean mode


def test_event_search_accepts_camera_names_as_well_as_ids():
    from backend.agent.tools import search_events_tool

    entry = {**event("ZONE_ENTRY", "CRITICAL", 990.0), "camera_name": "CAMERA 01"}
    store = FakeStore([entry, event("ZONE_ENTRY", "CRITICAL", 995.0, camera="cam-02")])
    tool = search_events_tool(EventSource(store=store, clock=lambda: 1000.0))

    def count(camera):
        return tool.handler({"camera_id": camera, "event_types": None, "minutes_back": 10, "limit": 10})["count"]

    assert count("camera 01") == count("cam-01") == 1
    assert count(None) == 2


def test_copilot_gets_a_counted_summary_with_every_question():
    from backend.agent.agents import run_copilot

    entry = {**event("ZONE_ENTRY", "CRITICAL", 990.0), "camera_name": "CAMERA 01"}
    store = FakeStore([entry, {**entry, "wall_time": 995.0}, event("VEHICLE_DETECTED", "INFO", 996.0)])
    model = ScriptedModel("local", [answer("Two zone entries on CAMERA 01.", model="local")])
    run_copilot(model, "What happened on CAMERA 01?", EventSource(store=store, clock=lambda: 1000.0), manager=None)
    task = model.sent[0][1]
    assert "- CAMERA 01: 2 ZONE_ENTRY" in task and "VEHICLE_DETECTED" not in task


def test_parse_judgment_ignores_other_lines():
    from backend.agent.agents import parse_judgment

    assert parse_judgment("Situation: x\n## Next steps: Send patrol.\nlink: CAM-02 match") == {
        "next_step": "Send patrol.", "links": "CAM-02 match"}


# --- choosing a model ------------------------------------------------------------------

def test_chain_falls_back_when_the_first_model_is_unavailable():
    cloud = ScriptedModel("cloud", [ModelUnavailable("offline")])
    local = ScriptedModel("local", [answer("from local", model="local")])
    chain = ModelChain([cloud, local])
    run, used = chain.run(lambda m: run_agent(m, "t", "s", [], "task"))
    assert used is local and run.text == "from local"


def test_a_specialist_losing_its_model_reruns_the_whole_brief_elsewhere():
    cloud = ScriptedModel("cloud", [calls(("ask_pursuit", {"question": "q"})), ModelUnavailable("dropped")])
    local = ScriptedModel("local", [answer("Situation: from local.", model="local")])
    run, used = ModelChain([cloud, local]).run(
        lambda m: run_supervisor(m, {"incident_id": "i"}, EventSource(), manager=None))
    assert used is local and run.text == "Situation: from local."


def test_recently_failed_model_is_tried_last_then_retried():
    now = [0.0]
    cloud = ScriptedModel("cloud", [ModelUnavailable("offline"), answer("cloud back", model="cloud")])
    local = ScriptedModel("local", [answer("a", model="local"), answer("b", model="local")])
    chain = ModelChain([cloud, local], retry_after_sec=60, clock=lambda: now[0])
    chain.run(lambda m: run_agent(m, "t", "s", [], "x"))
    _, used = chain.run(lambda m: run_agent(m, "t", "s", [], "x"))
    assert used is local and cloud.starts == 1  # cloud skipped while resting
    now[0] = 61.0
    run, used = chain.run(lambda m: run_agent(m, "t", "s", [], "x"))
    assert used is cloud and run.text == "cloud back"


def test_chain_reports_every_failure_when_nothing_answers():
    chain = ModelChain([ScriptedModel("a", [ModelUnavailable("x")]), ScriptedModel("b", [ModelUnavailable("y")])])
    with pytest.raises(ModelUnavailable, match="a: x; b: y"):
        chain.run(lambda m: run_agent(m, "t", "s", [], "task"))


# --- Claude through the SDK ---------------------------------------------------------------

class FakeClaudeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def claude_response(*blocks, stop="end_turn"):
    return SimpleNamespace(content=list(blocks), stop_reason=stop, model="claude-opus-5-5")


def test_claude_request_shape_and_tool_parsing():
    thinking = SimpleNamespace(type="thinking", thinking="", signature="sig")
    tool_use = SimpleNamespace(type="tool_use", id="tu_1", name="echo", input={"x": 1})
    client = FakeClaudeClient([claude_response(thinking, tool_use, stop="tool_use"),
                               claude_response(SimpleNamespace(type="text", text="Brief."))])
    tool = ToolSpec("echo", "Echo", {"type": "object", "properties": {}, "required": [], "additionalProperties": False}, lambda a: a)
    conv = ClaudeModel("claude-opus-5-5", effort="medium", client=client).start("sys", [tool])

    turn = conv.send("task")
    assert turn.stop == "tool_use" and turn.tool_calls[0].input == {"x": 1}
    first = client.requests[0]
    assert first["tools"][0]["strict"] is True
    assert first["fallbacks"] == "default" and first["betas"] == ["server-side-fallback-2026-07-01"]
    assert first["output_config"] == {"effort": "medium"}
    assert "thinking" not in first

    turn = conv.send_tool_results([ToolResult("tu_1", "echo", "{}")])
    assert turn.text == "Brief." and turn.stop == "done"
    messages = client.requests[1]["messages"]
    # The assistant turn is resent exactly as returned, thinking block included.
    assert messages[1]["content"][0] is thinking
    assert messages[2]["content"][0] == {"type": "tool_result", "tool_use_id": "tu_1", "content": "{}", "is_error": False}


def test_claude_refusal_and_haiku_without_fallbacks():
    client = FakeClaudeClient([claude_response(stop="refusal")])
    turn = ClaudeModel("claude-haiku-5-5", client=client).start("sys", []).send("q")
    assert turn.stop == "refusal"
    assert "fallbacks" not in client.requests[0]


def test_claude_without_credentials_falls_back_instead_of_crashing(monkeypatch):
    # With no key the real SDK raises a TypeError when the request is built;
    # it has to become an ordinary "unavailable" so the local model takes over.
    monkeypatch.setattr(anthropic, "Anthropic", lambda **_: SimpleNamespace(api_key=None, auth_token=None, credentials=None))
    cloud = ClaudeModel("claude-opus-5-5")
    local = ScriptedModel("local", [answer("offline brief", model="local")])
    run, used = ModelChain([cloud, local]).run(lambda m: run_agent(m, "t", "s", [], "task"))
    assert used is local and run.text == "offline brief"


def test_claude_network_failure_means_unavailable_but_bad_request_does_not():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    offline = ClaudeModel("claude-opus-5-5", client=FakeClaudeClient([anthropic.APIConnectionError(request=request)]))
    with pytest.raises(ModelUnavailable):
        offline.start("sys", []).send("q")
    bad = anthropic.BadRequestError("bad", response=httpx2.Response(400, request=request), body=None)
    broken = ClaudeModel("claude-opus-5-5", client=FakeClaudeClient([bad]))
    with pytest.raises(anthropic.BadRequestError):
        broken.start("sys", []).send("q")


# --- local model through Ollama -------------------------------------------------------------

def test_ollama_tool_calls_and_results_round_trip():
    seen = []

    def handle(request):
        body = json.loads(request.content)
        seen.append(body)
        if len(seen) == 1:
            return httpx.Response(200, json={"message": {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "echo", "arguments": '{"x": 2}'}}]}})
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "Local brief."}})

    model = OllamaModel("qwen2.5:3b", http=httpx.Client(transport=httpx.MockTransport(handle)))
    tool = ToolSpec("echo", "Echo", {"type": "object", "properties": {}}, lambda a: a)
    conv = model.start("sys", [tool])
    turn = conv.send("task")
    assert turn.tool_calls[0].name == "echo" and turn.tool_calls[0].input == {"x": 2}
    turn = conv.send_tool_results([ToolResult(turn.tool_calls[0].call_id, "echo", "ok")])
    assert turn.text == "Local brief."
    assert seen[0]["messages"][0] == {"role": "system", "content": "sys"}
    assert seen[0]["tools"][0]["function"]["name"] == "echo"
    assert seen[1]["messages"][-1] == {"role": "tool", "tool_name": "echo", "content": "ok"}


def test_ollama_missing_model_or_server_means_unavailable():
    missing = OllamaModel("nope", http=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404, json={}))))
    with pytest.raises(ModelUnavailable, match="ollama pull nope"):
        missing.start("sys", []).send("q")

    def refuse(request):
        raise httpx.ConnectError("refused", request=request)
    down = OllamaModel("m", http=httpx.Client(transport=httpx.MockTransport(refuse)))
    with pytest.raises(ModelUnavailable, match="Ollama running"):
        down.start("sys", []).send("q")


# --- hosted free-tier models ---------------------------------------------------------------

def hosted(handler, key="test-key"):
    return HostedModel("groq", "some-model", "https://api.example/v1", key, "GROQ_API_KEY",
                       http=httpx.Client(transport=httpx.MockTransport(handler)))


def test_hosted_model_tool_round_trip():
    seen = []

    def handle(request):
        seen.append((request.headers.get("authorization"), json.loads(request.content)))
        if len(seen) == 1:
            return httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "echo", "arguments": '{"x": 3}'}}]}}]})
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Cloud brief."}}]})

    tool = ToolSpec("echo", "Echo", {"type": "object", "properties": {}}, lambda a: a)
    conv = hosted(handle).start("sys", [tool])
    turn = conv.send("task")
    assert turn.tool_calls[0].call_id == "call_1" and turn.tool_calls[0].input == {"x": 3}
    turn = conv.send_tool_results([ToolResult("call_1", "echo", "ok")])
    assert turn.text == "Cloud brief." and turn.model == "groq:some-model"
    auth, first = seen[0]
    assert auth == "Bearer test-key" and first["tool_choice"] == "auto" and first["tools"][0]["function"]["name"] == "echo"
    assert seen[1][1]["messages"][-1] == {"role": "tool", "tool_call_id": "call_1", "content": "ok"}


@pytest.mark.parametrize("status, message", [(401, "rejected the key"), (429, "free-tier limit"), (400, "error 400")])
def test_hosted_model_problems_fall_through_to_the_next_model(status, message):
    model = hosted(lambda r: httpx.Response(status, json={}))
    with pytest.raises(ModelUnavailable, match=message):
        model.start("sys", []).send("q")


def test_hosted_model_without_its_key_is_skipped():
    with pytest.raises(ModelUnavailable, match="set GROQ_API_KEY"):
        hosted(lambda r: httpx.Response(200), key=None).start("sys", [])


def test_chain_order_follows_configuration(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    chain = build_chain(AgentConfig(hosted_provider="openrouter", hosted_model=""))
    assert [m.kind for m in chain.models] == ["claude", "cloud", "local"]
    assert chain.models[1].name == "openrouter:openrouter/free"
    assert [m.kind for m in build_chain(AgentConfig(hosted_provider="")).models] == ["claude", "local"]
    with pytest.raises(ValueError, match="DRISHTI_HOSTED_MODEL"):
        build_chain(AgentConfig(hosted_provider="groq", hosted_model=""))
    with pytest.raises(ValueError, match="unknown"):
        build_chain(AgentConfig(hosted_provider="nope"))


# --- the service --------------------------------------------------------------------------

def make_service(script, events=()):
    store = FakeStore(events)
    config = AgentConfig(enabled=True, brief_delay_sec=0.0)
    model = ScriptedModel("cloud", script)
    return AgentService(config, ModelChain([model]), store=store), store


def test_brief_is_written_and_audited_without_raising_an_alarm():
    service, store = make_service([
        calls(("propose_action", {"action": "review_footage", "camera_id": "cam-01", "reason": "Crossing on clip."})),
        answer("Situation: one person crossed.\nEvidence: ZONE_ENTRY 22:41:07.\nNext step: review clip.", model="cloud"),
    ])
    incident = service._grouper.add(event("ZONE_ENTRY", "CRITICAL", 1000.0))
    brief = service.brief_incident(incident)

    assert brief.status == "ready" and brief.model == "cloud"
    assert len(brief.proposal_ids) == 1
    types = [e["type"] for e in store.recorded]
    assert types == ["AGENT_PROPOSAL", "AGENT_BRIEF"]
    assert all(e["severity"] == "INFO" and e["alarm"] is False for e in store.recorded)
    assert service.briefs()[0]["text"].startswith("Situation:")


def test_rules_write_the_brief_when_no_model_answers():
    service, store = make_service([ModelUnavailable("offline")])
    incident = service._grouper.add(event("ZONE_ENTRY", "CRITICAL", 1.0))
    brief = service.brief_incident(incident)
    assert brief.status == "template" and brief.model_kind == "template"
    assert brief.text.startswith("Situation: Restricted zone entered on CAM-01")
    assert "offline" in brief.note
    assert store.recorded[0]["type"] == "AGENT_BRIEF" and store.recorded[0]["data"]["model"] == "template"


def test_an_empty_or_refused_answer_also_falls_back_to_rules():
    service, _ = make_service([Turn(text="", tool_calls=[], stop="refusal", model="cloud")])
    brief = service.brief_incident(service._grouper.add(event("ZONE_ENTRY", "CRITICAL", 1.0)))
    assert brief.status == "template" and "refusal" in brief.note


def test_unexpected_errors_still_leave_a_brief():
    service, store = make_service([RuntimeError("bug")])
    brief = service.brief_incident(service._grouper.add(event("ZONE_ENTRY", "CRITICAL", 1.0)))
    assert brief.status == "template" and "internal error" in brief.note
    assert service.briefs()[0]["brief_id"] == brief.brief_id
    assert store.recorded[0]["type"] == "AGENT_BRIEF"


# --- rules-based brief ------------------------------------------------------------

def test_template_brief_leads_with_the_alarm_and_suggests_the_standard_step():
    from backend.agent.template import template_brief

    grouper = IncidentGrouper()
    incident = grouper.add(event("ZONE_APPROACH", "MEDIUM", 1_700_000_000.0, reason="Track #1 heading for zone"))
    grouper.add(event("ZONE_ENTRY", "CRITICAL", 1_700_000_005.0, reason="Track #1 entered zone"))
    grouper.add(event("LOITERING", "MEDIUM", 1_700_000_030.0, reason="Track #1 loitering 25s"))
    text = template_brief(incident.for_agent())
    situation, evidence, elsewhere, step = text.split("\n")
    assert elsewhere == "Other cameras: Nothing notable in the last 15 minutes."
    assert situation.startswith("Situation: Restricted zone entered on CAM-01 at ")
    assert "3 related events between" in situation
    assert evidence.startswith("Evidence: ") and evidence.index("Restricted zone entered") < evidence.index("Loitering")
    assert step == "Next step: Check the live view of CAM-01 and send the nearest patrol if the person is still inside the zone."


def test_template_brief_groups_repeated_events():
    from backend.agent.template import template_brief

    events = [{"type": "ZONE_ENTRY", "severity": "CRITICAL", "time": "11:04:28", "track_id": t} for t in (6, 7, 10)]
    events.append({"type": "ZONE_EXIT", "severity": "LOW", "time": "11:04:40", "track_id": 6, "reason": "Track #6 left"})
    evidence = template_brief({"camera": "CAM-01", "events": events}).split("\n")[1]
    assert evidence == "Evidence: 11:04:28 Restricted zone entered ×3 (tracks #6, #7, #10); 11:04:40 Restricted zone left (Track #6 left)."


def test_template_brief_handles_unknown_types_and_no_events():
    from backend.agent.template import template_brief

    assert "Next step: Review the evidence clip" in template_brief({"camera": "CAM-09", "events": []})
    text = template_brief({"camera": "CAM-09", "events": [{"type": "NEW_KIND_OF_ALERT", "severity": "HIGH", "time": "10:00:00"}]})
    assert text.startswith("Situation: New kind of alert on CAM-09 at 10:00:00")


def test_each_alarm_incident_is_scheduled_once():
    service, _ = make_service([])
    service.on_event(event("ZONE_ENTRY", "CRITICAL", 1.0))
    service.on_event(event("LOITERING", "MEDIUM", 2.0))
    service.on_event(event("ZONE_ENTRY", "CRITICAL", 3.0))
    assert len(service._scheduled) == 1
    queued = service._queue.get(timeout=2)
    assert [e["type"] for e in queued.events] == ["ZONE_ENTRY", "LOITERING", "ZONE_ENTRY"]


def test_proposals_wait_for_one_operator_decision():
    service, store = make_service([
        calls(("propose_action", {"action": "dispatch_patrol", "camera_id": None, "reason": "Person inside zone."})),
        answer("Brief.", model="cloud"),
    ])
    brief = service.brief_incident(service._grouper.add(event("ZONE_ENTRY", "CRITICAL", 1.0)))
    pid = brief.proposal_ids[0]
    assert service.proposals("pending")[0]["proposal_id"] == pid

    decided = service.decide(pid, approve=True, user="commander")
    assert decided.status == "approved" and decided.decided_by == "commander"
    assert store.recorded[-1]["type"] == "AGENT_APPROVED"
    with pytest.raises(ValueError):
        service.decide(pid, approve=False, user="guard")
    with pytest.raises(KeyError):
        service.decide("nope", approve=True, user="guard")


def test_copilot_answers_from_events_and_audits_the_question():
    events = [event("PLATE_CONFIRMED", "MEDIUM", 9_999_999_000.0, data={"plate_text": "TEST123"})]
    service, store = make_service([
        calls(("search_events", {"camera_id": None, "event_types": ["PLATE_CONFIRMED"], "minutes_back": 60, "limit": 10})),
        answer("TEST123 was read on CAM-01.", model="cloud"),
    ], events=events)
    service.source.clock = lambda: 9_999_999_100.0
    result = service.ask("Did TEST123 appear?", user="guard")
    assert result["answer"] == "TEST123 was read on CAM-01."
    tool_output = json.loads(service.chain.models[0].sent[1][1][0].content)
    assert tool_output["events"][0]["data"]["plate_text"] == "TEST123"
    assert store.recorded[-1]["type"] == "AGENT_ANSWER" and store.recorded[-1]["data"]["user"] == "guard"
