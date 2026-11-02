"""The shift handover report: what the next shift needs to know, on one page.

    GET /api/live/handover?hours=12

A printable page rather than a PDF file: it opens in a browser tab, and the
browser's Print -> Save as PDF produces the file, so no PDF library has to be
installed on the post's machine. Everything on it is read back from the
tamper-evident event log, and the page says whether that log verified.
"""
from __future__ import annotations

import html
import time
from collections import Counter
from datetime import datetime

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from backend.api import agent as agent_api
from backend.api.auth import _token_from, decode_token
from backend.core.events import describe_event
from backend.live.camera_manager import camera_manager

router = APIRouter(prefix="/api/live", tags=["live"])

SEVERITY_ORDER = ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO")


def _t(ts: float | None, with_date: bool = False) -> str:
    if not ts:
        return "—"
    return datetime.fromtimestamp(ts).strftime("%d %b %H:%M" if with_date else "%H:%M:%S")


def _e(value) -> str:
    return html.escape(str(value)) if value is not None else ""


def _label(event_type: str) -> str:
    return event_type.replace("_", " ").capitalize()


def build_report(hours: int, events: list[dict], cameras: list[dict], chain: dict, briefs: list[dict],
                 proposals: list[dict], patterns: list, generated_by: str, now: float) -> str:
    since = now - hours * 3600
    events = sorted((e for e in events if (e.get("wall_time") or 0) >= since), key=lambda e: e.get("wall_time") or 0)
    alarms = [e for e in events if e.get("alarm")]
    by_severity = Counter(e.get("severity") for e in events if not str(e.get("type", "")).startswith("AGENT_"))
    briefs = [b for b in briefs if (b.get("created_at") or 0) >= since]
    pending = [p for p in proposals if p.get("status") == "pending"]
    decided = [p for p in proposals if p.get("status") != "pending" and (p.get("decided_at") or 0) >= since]
    trouble = [c for c in cameras if c.get("state") != "online" or c.get("degraded")]
    # Still tampered: the camera's last tampering alert is newer than its last "view restored".
    last: dict[tuple[str, str], float] = {}
    for e in events:
        if e.get("type") in ("CAMERA_TAMPERED", "CAMERA_VIEW_RESTORED"):
            last[(e.get("camera_name") or "", e["type"])] = e.get("wall_time") or 0
    tampered = {name for (name, kind), ts in last.items()
                if kind == "CAMERA_TAMPERED" and ts > last.get((name, "CAMERA_VIEW_RESTORED"), 0)}

    def rows(items, cells):
        return "".join("<tr>" + "".join(f"<td>{c(i)}</td>" for c in cells) + "</tr>" for i in items)

    open_items = []
    if pending:
        open_items.append(f"{len(pending)} AI proposal{'s' if len(pending) != 1 else ''} still waiting for a decision")
    for c in trouble:
        open_items.append(f"{_e(c.get('name'))} is {_e(c.get('state'))}{' (frames stalled)' if c.get('degraded') else ''}")
    for name in sorted(n for n in tampered if n):
        open_items.append(f"{_e(name)} reported tampering and has not been confirmed back to normal")
    if not chain.get("valid", True):
        open_items.append("The event log failed its integrity check: " + _e(chain.get("summary")))

    severity_cells = "".join(
        f'<div class="stat"><b>{by_severity.get(s, 0)}</b><span>{s.lower()}</span></div>' for s in SEVERITY_ORDER)
    open_html = ("<ul class='open'>" + "".join(f"<li>{i}</li>" for i in open_items) + "</ul>" if open_items
                 else "<p class='ok'>Nothing outstanding: every camera is online and no proposal is waiting.</p>")
    brief_html = "".join(
        f"<div class='brief'><b>{_e(b.get('camera_name'))} · {_t(b.get('created_at'))}</b> "
        f"<span class='muted'>({_e(b.get('model'))})</span><br>{_e(b.get('text'))}</div>"
        for b in briefs[:20]
    ) or ("<p class='muted'>No AI briefs this shift" + ("" if agent_api._service else " (the AI analyst is off)") + ".</p>")
    pattern_html = ("<ul>" + "".join(f"<li>{_e(p.summary)}</li>" for p in patterns[:8]) + "</ul>" if patterns
                    else "<p class='muted'>No repeated plates or same-hour alarms.</p>")

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Drishti shift handover {_t(now, True)}</title>
<style>
  body {{ font: 13px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; color: #15191d; margin: 32px auto; max-width: 900px; padding: 0 20px; }}
  h1 {{ font-size: 22px; margin: 0; }} h2 {{ font-size: 14px; margin: 22px 0 6px; text-transform: uppercase; letter-spacing: .06em; color: #39424c; }}
  .meta {{ color: #5b6570; margin-top: 4px; }}
  .stats {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 12px; }}
  .stat {{ border: 1px solid #d5dbe1; border-radius: 6px; padding: 6px 12px; min-width: 70px; }}
  .stat b {{ display: block; font-size: 18px; }} .stat span {{ color: #5b6570; font-size: 11px; text-transform: uppercase; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  td, th {{ text-align: left; padding: 4px 8px 4px 0; border-bottom: 1px solid #e4e8ec; vertical-align: top; }}
  th {{ font-size: 10.5px; text-transform: uppercase; color: #5b6570; }}
  .open li {{ margin: 3px 0; }} .ok {{ color: #1d7a43; }} .bad {{ color: #b3261e; font-weight: 600; }}
  .brief {{ border-left: 3px solid #3b6fa3; padding: 4px 10px; margin: 8px 0; white-space: pre-wrap; }}
  .muted {{ color: #5b6570; }} .print {{ float: right; }}
  @media print {{ .print {{ display: none; }} body {{ margin: 0 auto; }} }}
</style></head><body>
<button class="print" onclick="window.print()">Print / Save as PDF</button>
<h1>Drishti · Shift handover</h1>
<div class="meta">Last {hours} hours: {_t(since, True)} – {_t(now, True)} · generated by {_e(generated_by)}</div>
<div class="stats"><div class="stat"><b>{len(alarms)}</b><span>siren alarms</span></div>{severity_cells}
<div class="stat"><b>{len(briefs)}</b><span>AI briefs</span></div></div>

<h2>Needs attention now</h2>
{open_html}

<h2>Cameras</h2>
<table><tr><th>Camera</th><th>Location</th><th>State</th><th>Analysis</th></tr>
{rows(cameras, [lambda c: _e(c.get('name')), lambda c: _e(c.get('location') or '—'),
                lambda c: _e(c.get('state')) + (' (stalled)' if c.get('degraded') else ''),
                lambda c: _e(', '.join(c.get('modules') or []) or 'view only')]) or "<tr><td colspan=4 class='muted'>No cameras connected.</td></tr>"}
</table>

<h2>Alarms ({len(alarms)})</h2>
<table><tr><th>Time</th><th>Camera</th><th>Event</th><th>What happened</th></tr>
{rows(alarms[-100:], [lambda e: _t(e.get('wall_time')), lambda e: _e(e.get('camera_name')),
                       lambda e: _e(_label(e.get('type', ''))), lambda e: _e(e.get('reason') or describe_event(e))])
 or "<tr><td colspan=4 class='muted'>No alarms this shift.</td></tr>"}
</table>
{"<p class='muted'>Showing the latest 100 alarms.</p>" if len(alarms) > 100 else ""}

<h2>AI briefs ({len(briefs)})</h2>
{brief_html}

<h2>Decisions on AI proposals</h2>
<table><tr><th>Time</th><th>Proposal</th><th>Camera</th><th>Decision</th></tr>
{rows(decided, [lambda p: _t(p.get('decided_at')), lambda p: _e(p.get('action', '').replace('_', ' ')),
                lambda p: _e(p.get('camera_id') or '—'), lambda p: _e(f"{p.get('status')} by {p.get('decided_by')}")])
 or "<tr><td colspan=4 class='muted'>No decisions this shift.</td></tr>"}
</table>

<h2>Repeats over the last 7 days</h2>
{pattern_html}

<h2>Record integrity</h2>
<p class="{'ok' if chain.get('valid', True) else 'bad'}">{_e(chain.get('summary'))}</p>
<p class="muted">Every line above is read back from Drishti's hash-chained event log.</p>
</body></html>"""


@router.get("/handover", response_class=HTMLResponse)
def handover(request: Request, hours: int = 12) -> HTMLResponse:
    from backend.agent.patterns import recent_patterns

    hours = max(1, min(hours, 72))
    now = time.time()
    store = camera_manager.store
    token = _token_from(request)
    user = decode_token(token) if token else None
    status = store.verify_chain()
    chain = {"valid": status.valid, "summary": (
        f"All {status.total_events} records verified: the log has not been altered." if status.valid
        else f"Integrity check FAILED at record {status.broken_at_id}: {status.detail}")}
    service = agent_api._service
    page = build_report(
        hours=hours,
        events=store.query(limit=1000, since=now - hours * 3600),
        cameras=[s.status_dict() for s in camera_manager.list()],
        chain=chain,
        briefs=service.briefs(limit=200) if service else [],
        proposals=service.proposals() if service else [],
        patterns=recent_patterns(store=store, now=now),
        generated_by=user.username if user else "local operator",
        now=now,
    )
    return HTMLResponse(page)
