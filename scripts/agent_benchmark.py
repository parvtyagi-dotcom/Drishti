"""Time how long the AI Analyst takes to brief an incident on this machine.

Runs the real Supervisor agent (tools, Pursuit delegation and all) on a
sample zone-intrusion incident, several times, and reports seconds per brief.
For a local model it also measures raw generation speed in tokens/second,
which is what decides whether briefs arrive in seconds or minutes.

    python scripts/agent_benchmark.py                 # the local Ollama model
    python scripts/agent_benchmark.py --kind cloud    # the hosted free tier
    python scripts/agent_benchmark.py --kind claude --runs 1

The incident is sample data shaped like real Drishti events; no camera runs.
Every call to a cloud model counts against that provider's limits or bill.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from backend.agent.agents import write_brief  # noqa: E402
from backend.agent.incidents import IncidentGrouper  # noqa: E402
from backend.agent.models import ModelUnavailable, OllamaModel, build_chain  # noqa: E402
from backend.agent.tools import EventSource  # noqa: E402
from backend.config import get_config  # noqa: E402


class SampleEvents:
    """Stands in for the event store, holding the sample incident plus some
    earlier activity on another camera for the agents' tools to find."""

    def __init__(self, events):
        self.events = events

    def query(self, limit=100, camera_id=None, since=None, **_):
        out = [e for e in self.events
               if (camera_id is None or e["camera_id"] == camera_id) and (since is None or e["wall_time"] >= since)]
        return sorted(out, key=lambda e: e["wall_time"], reverse=True)[:limit]


def sample_events(now: float) -> list[dict]:
    def ev(offset, etype, severity, camera, track, reason, **extra):
        return {"type": etype, "severity": severity, "wall_time": now + offset, "camera_id": camera,
                "camera_name": camera.upper().replace("-", " "), "track_id": track, "reason": reason, **extra}
    return [
        ev(-300, "TARGET_CONFIRMED", "HIGH", "cam-02", 3, "Track #3 matched the watched person's reference photos"),
        ev(-240, "TARGET_LOST", "MEDIUM", "cam-02", 3, "Track #3 left the frame heading east"),
        ev(-8, "ZONE_APPROACH", "MEDIUM", "cam-01", 7, "Track #7 is heading toward 'Restricted' and is predicted to enter within about 2 seconds"),
        ev(-5, "ZONE_ENTRY", "CRITICAL", "cam-01", 7, "Track #7 crossed into 'Restricted'", zone_label="Restricted"),
        ev(25, "LOITERING", "MEDIUM", "cam-01", 7, "Track #7 has stayed inside 'Restricted' for 30 seconds"),
    ]


def pick_models(kind: str):
    config = get_config().agent
    if kind == "local":
        return [OllamaModel(config.local_model, base_url=config.local_url, timeout_sec=config.local_timeout_sec)]
    return [m for m in build_chain(config).models if m.kind == kind]


def raw_speed(model: OllamaModel) -> str:
    """Tokens/second for plain generation, from Ollama's own counters."""
    response = model._http.post(f"{model.base_url}/api/generate", json={
        "model": model.name, "stream": False,
        "prompt": "In two sentences, explain why a border post watches its cameras at night.",
    })
    response.raise_for_status()
    data = response.json()
    load = data.get("load_duration", 0) / 1e9
    tokens, secs = data.get("eval_count", 0), data.get("eval_duration", 0) / 1e9
    return f"{tokens / secs:.1f} tokens/s generating ({tokens} tokens), model load {load:.1f}s" if secs else "no timing reported"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kind", choices=["local", "cloud", "claude"], default="local")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    models = pick_models(args.kind)
    if not models:
        print(f"No {args.kind} model is configured; see the AI Analyst section of the README.")
        return 1
    model = models[0]
    print(f"Model: {model.name} ({model.kind})")

    if isinstance(model, OllamaModel):
        try:
            print("Raw speed:", raw_speed(model))
        except Exception as exc:  # report and stop: the brief runs would fail the same way
            print(f"Cannot reach the local model: {exc}")
            return 1

    now = time.time()
    events = sample_events(now)
    incident = None
    grouper = IncidentGrouper()
    for e in (e for e in events if e["camera_id"] == "cam-01"):
        incident = grouper.add(e)
    snapshot = incident.for_agent()
    source = EventSource(store=SampleEvents(events), clock=lambda: now + 30)

    times = []
    for run_no in range(1, args.runs + 1):
        started = time.monotonic()
        try:
            written = write_brief(model, snapshot, source, manager=None, max_steps=get_config().agent.max_steps)
        except ModelUnavailable as exc:
            print(f"Run {run_no}: model unavailable: {exc}")
            return 1
        took = time.monotonic() - started
        times.append(took)
        run = written.run
        tools = [s.get("tool") or "pursuit answer" for s in run.steps]
        print(f"\nRun {run_no}: {took:.1f}s, stopped={run.stopped}, tool calls: {', '.join(tools) or 'none'}, "
              f"model lines used: {'yes' if written.model_contributed else 'NO (rules only)'}")
        print("  " + written.text.replace("\n", "\n  "))
        print("  Proposals for approval: " + ("; ".join(f"{p.action} on {p.camera_id or '-'} ({p.reason})" for p in run.proposals) or "none"))

    print(f"\nSeconds per brief: median {statistics.median(times):.1f}, fastest {min(times):.1f}, slowest {max(times):.1f}")
    print("(The first run includes loading the model into memory.)" if args.kind == "local" else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
