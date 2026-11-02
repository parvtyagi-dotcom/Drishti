"""Measure Drishti's accuracy against hand-labelled footage.

"What is your accuracy?" deserves a number measured on footage we labelled
ourselves, not a figure quoted from a model's paper. This script runs the real
analysis pipeline (the same CombinedPipeline Forensic and Live mode use) over
every clip in a labels file and scores what it reports against what a person
wrote down.

    python scripts/evaluate.py                              # demo clips
    python scripts/evaluate.py --labels eval/campus.csv --videos D:/footage
    python scripts/evaluate.py --only anpr,zone

LABELS FILE (CSV, lines starting with # are ignored). One row per fact:

    clip,module,expected,start_sec,end_sec,reference,zone,lowlight,condition[,calibration]

  anpr       expected = the plate text on a vehicle in the clip. One row per
             plate. ANPR runs in automatic mode, so every plate it confirms is
             compared with the labelled ones.
  person_id  expected = present | absent, reference = a photo or a folder of
             photos of the person to look for.
  zone       expected = entry for each real crossing, with start_sec/end_sec
             bracketing the moment the person crosses; or expected = none for
             a clip where nobody crosses (it still counts false alarms).
             zone = path to a zone JSON (as saved under zones/). Start each
             clip with the zone empty: anyone already inside on the first
             frame registers as an entry at 0 s.
  behavior   expected = LOITERING | ABNORMAL_SPEED | SUDDEN_DIRECTION_CHANGE |
             REPEATED_BACK_AND_FORTH for each real occurrence, with start_sec/
             end_sec bracketing it; or expected = none for a clip with nothing
             suspicious (it still counts false alarms). Scored like zone
             crossings: each labelled window matches at most one alert of that
             type, and every unmatched alert is a false alarm.
  lowlight   yes to run low-light enhancement, as an operator would at night.
  condition  any label for splitting results: day, night, far, rain ...
  calibration  optional, behavior only: path to a ground calibration JSON
             (as saved under data/calibrations/) to score behaviour in metres.

Relative paths resolve against the repository; clips resolve against --videos.

Outputs a summary in the terminal plus report.md and report.json in --out.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DEFAULT_LABELS = REPO / "eval" / "demo_labels.csv"
DEFAULT_VIDEOS = REPO / "uploads"
DEFAULT_OUT = REPO / "outputs" / "eval"
MODULES = ("anpr", "person_id", "zone", "behavior")
BEHAVIOR_TYPES = ("LOITERING", "ABNORMAL_SPEED", "SUDDEN_DIRECTION_CHANGE", "REPEATED_BACK_AND_FORTH")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Label:
    clip: str
    module: str
    expected: str
    start_sec: float | None = None
    end_sec: float | None = None
    reference: str = ""
    zone: str = ""
    lowlight: bool = False
    condition: str = ""
    calibration: str = ""
    line: int = 0


@dataclass
class Run:
    """One pipeline pass over one clip, and every label it is scored against."""
    module: str
    clip: str
    reference: str
    zone: str
    lowlight: bool
    calibration: str = ""
    labels: list[Label] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    summary: dict | None = None
    duration_sec: float = 0.0
    processing_fps: float = 0.0
    error: str | None = None

    @property
    def condition(self) -> str:
        conditions = {lab.condition for lab in self.labels if lab.condition}
        return ", ".join(sorted(conditions)) or "unlabelled"


# --- reading labels -----------------------------------------------------------

def _float_or_none(value: str) -> float | None:
    value = (value or "").strip()
    return float(value) if value else None


def read_labels(path: Path) -> list[Label]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = [(n, line) for n, line in enumerate(f, start=1) if line.strip() and not line.lstrip().startswith("#")]
    if not rows:
        raise SystemExit(f"{path} has no labels")
    reader = csv.DictReader([line for _, line in rows])
    labels = []
    for (line_no, _), row in zip(rows[1:], reader):
        row = {k.strip(): (v or "").strip() for k, v in row.items() if k}
        module = row.get("module", "").lower()
        if module not in MODULES:
            raise SystemExit(f"{path}:{line_no}: module must be one of {', '.join(MODULES)}, got {module!r}")
        expected = row.get("expected", "")
        if module == "person_id" and expected.lower() not in ("present", "absent"):
            raise SystemExit(f"{path}:{line_no}: person_id expects 'present' or 'absent'")
        if module == "person_id" and not row.get("reference"):
            raise SystemExit(f"{path}:{line_no}: person_id needs a reference photo or folder")
        if module == "zone":
            if expected.lower() not in ("entry", "none"):
                raise SystemExit(f"{path}:{line_no}: zone expects 'entry' or 'none'")
            if not row.get("zone"):
                raise SystemExit(f"{path}:{line_no}: zone needs a zone JSON path")
            if expected.lower() == "entry" and (not row.get("start_sec") or not row.get("end_sec")):
                raise SystemExit(f"{path}:{line_no}: a zone entry needs start_sec and end_sec")
        if module == "behavior":
            if expected.upper() not in BEHAVIOR_TYPES and expected.lower() != "none":
                raise SystemExit(f"{path}:{line_no}: behavior expects one of {', '.join(BEHAVIOR_TYPES)} or 'none'")
            if expected.lower() != "none" and (not row.get("start_sec") or not row.get("end_sec")):
                raise SystemExit(f"{path}:{line_no}: a behaviour occurrence needs start_sec and end_sec")
        labels.append(Label(
            clip=row.get("clip", ""), module=module, expected=expected,
            start_sec=_float_or_none(row.get("start_sec", "")), end_sec=_float_or_none(row.get("end_sec", "")),
            reference=row.get("reference", ""), zone=row.get("zone", ""),
            lowlight=row.get("lowlight", "").lower() in ("yes", "y", "true", "1"),
            condition=row.get("condition", ""), calibration=row.get("calibration", "") if module == "behavior" else "",
            line=line_no,
        ))
    return labels


def group_runs(labels: list[Label]) -> list[Run]:
    """Labels that need the same pipeline pass share one run."""
    runs: dict[tuple, Run] = {}
    for lab in labels:
        key = (lab.module, lab.clip, lab.reference if lab.module == "person_id" else "",
               lab.zone if lab.module == "zone" else "", lab.lowlight, lab.calibration)
        if key not in runs:
            runs[key] = Run(module=key[0], clip=key[1], reference=key[2], zone=key[3], lowlight=key[4], calibration=key[5])
        runs[key].labels.append(lab)
    return list(runs.values())


def resolve(path_text: str, base: Path) -> Path:
    p = Path(path_text)
    return p if p.is_absolute() else base / p


def reference_photos(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return [path]


# --- running the pipeline ------------------------------------------------------

def execute(run: Run, videos: Path) -> None:
    from backend.modules.zone.geometry import Zone
    from backend.orchestrator import CombinedPipeline, OrchestratorRequest

    clip = resolve(run.clip, videos)
    if not clip.is_file():
        run.error = f"clip not found: {clip}"
        return
    req = OrchestratorRequest(enable_lowlight=run.lowlight)
    if run.module == "anpr":
        req.enable_anpr = True
    elif run.module == "person_id":
        photos = reference_photos(resolve(run.reference, REPO))
        if not photos or not all(p.is_file() for p in photos):
            run.error = f"reference photos not found: {run.reference}"
            return
        req.enable_person_id = True
        req.reference_photo_paths = photos
    elif run.module == "behavior":
        req.enable_behavior = True
        if run.calibration:
            from backend.core.calibration import GroundCalibration

            cal_file = resolve(run.calibration, REPO)
            if not cal_file.is_file():
                run.error = f"calibration file not found: {cal_file}"
                return
            req.ground_calibration = GroundCalibration.from_dict(json.loads(cal_file.read_text(encoding="utf-8")))
    else:
        zone_file = resolve(run.zone, REPO)
        if not zone_file.is_file():
            run.error = f"zone file not found: {zone_file}"
            return
        req.zones = [Zone.from_dict(json.loads(zone_file.read_text(encoding="utf-8")))]

    started = time.monotonic()
    try:
        result = CombinedPipeline(req).run(clip, write_output=False)
    except Exception as exc:  # one bad clip must not sink the whole report
        run.error = f"{type(exc).__name__}: {exc}"
        return
    elapsed = time.monotonic() - started
    run.events = result.events
    run.summary = {"anpr": result.anpr, "person_id": result.person_id, "zone": result.zones,
                   "behavior": result.behavior}[run.module]
    run.duration_sec = result.video_info.duration_sec or 0.0
    frames = result.metrics.get("frames_read") or 0
    run.processing_fps = result.metrics.get("processing_fps") or (frames / elapsed if elapsed else 0.0)


# --- scoring ------------------------------------------------------------------

def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def char_accuracy(truth: str, read: str) -> float:
    if not truth:
        return 0.0
    return max(0.0, 1.0 - edit_distance(truth, read) / len(truth))


def score_anpr(run: Run, normalize) -> dict:
    plates = (run.summary or {}).get("detected_plates", [])
    confirmed = [normalize(p["text"]) for p in plates if p.get("confirmed") and p.get("text")]
    every_read = [normalize(p["text"]) for p in plates if p.get("text")]
    truths = [normalize(lab.expected) for lab in run.labels]
    rows = []
    for lab, truth in zip(run.labels, truths):
        best = max(every_read, key=lambda r: char_accuracy(truth, r), default="")
        rows.append({
            "label": lab, "truth": truth, "exact": truth in confirmed,
            "best_read": best, "char_accuracy": char_accuracy(truth, best) if best else 0.0,
        })
    wrong = [c for c in confirmed if c not in truths]
    return {"plates": rows, "wrong_plates": wrong}


def score_person(run: Run) -> dict:
    confirmed = bool((run.summary or {}).get("confirmed"))
    rows = [{"label": lab, "present": lab.expected.lower() == "present", "confirmed": confirmed} for lab in run.labels]
    return {"rows": rows}


def match_zone_entries(entry_times: list[float], windows: list[tuple[float, float]], tolerance: float) -> tuple[list[float | None], list[float]]:
    """Pair each labelled crossing with at most one detected ZONE_ENTRY.

    Windows are matched in time order to the earliest unused entry inside
    them (widened by the tolerance), so one detection can never satisfy two
    crossings. Entries left over are false alarms.
    """
    unused = sorted(entry_times)
    matched: list[float | None] = []
    for start, end in sorted(windows):
        hit = next((t for t in unused if start - tolerance <= t <= end + tolerance), None)
        if hit is not None:
            unused.remove(hit)
        matched.append(hit)
    return matched, unused


def warning_lead_times(events: list[dict], matched_entries: list[float | None]) -> list[float]:
    """Seconds between the first ZONE_APPROACH warning and the crossing it
    preceded, for the same track. This is the time an operator had to act."""
    leads = []
    entries = [e for e in events if e.get("type") == "ZONE_ENTRY"]
    approaches = [e for e in events if e.get("type") == "ZONE_APPROACH"]
    for t in matched_entries:
        if t is None:
            continue
        entry = next((e for e in entries if e.get("timestamp_sec") == t), None)
        if entry is None:
            continue
        track = entry.get("track_id")
        earlier_entries = [e["timestamp_sec"] for e in entries if e.get("track_id") == track and e["timestamp_sec"] < t]
        since = max(earlier_entries, default=float("-inf"))
        warnings = [a["timestamp_sec"] for a in approaches if a.get("track_id") == track and since < a["timestamp_sec"] <= t]
        if warnings:
            leads.append(t - min(warnings))
    return leads


def score_zone(run: Run, tolerance: float) -> dict:
    entry_times = [e["timestamp_sec"] for e in run.events if e.get("type") == "ZONE_ENTRY"]
    # Same order match_zone_entries walks the windows in, so results line up.
    crossing_labels = sorted((lab for lab in run.labels if lab.expected.lower() == "entry"),
                             key=lambda lab: (lab.start_sec, lab.end_sec))
    windows = [(lab.start_sec, lab.end_sec) for lab in crossing_labels]
    matched, false_alarms = match_zone_entries(entry_times, windows, tolerance)
    return {
        "crossings": [{"label": lab, "detected_at": hit} for lab, hit in zip(crossing_labels, matched)],
        "false_alarms": false_alarms,
        "lead_times": warning_lead_times(run.events, matched),
    }


def score_behavior(run: Run, tolerance: float) -> dict:
    """Per behaviour type, the same one-to-one window matching as zone
    crossings: a labelled occurrence is caught by the earliest unused alert of
    its type inside its window; alerts left over are false alarms."""
    occurrences, false_alarms = [], []
    for btype in BEHAVIOR_TYPES:
        alerts = [e["timestamp_sec"] for e in run.events if e.get("type") == btype]
        labelled = sorted((lab for lab in run.labels if lab.expected.upper() == btype),
                          key=lambda lab: (lab.start_sec, lab.end_sec))
        matched, unused = match_zone_entries(alerts, [(lab.start_sec, lab.end_sec) for lab in labelled], tolerance)
        occurrences += [{"label": lab, "type": btype, "detected_at": hit} for lab, hit in zip(labelled, matched)]
        false_alarms += [(btype, t) for t in unused]
    return {"occurrences": occurrences, "false_alarms": false_alarms}


# --- reporting ----------------------------------------------------------------

def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100.0 * n / d:.0f}%)" if d else "no samples"


def build_report(runs: list[Run], scores: dict[int, dict], labels_path: Path, tolerance: float) -> tuple[str, dict]:
    anpr_rows = [r for i, run in enumerate(runs) if run.module == "anpr" and not run.error for r in scores[i]["plates"]]
    anpr_wrong = [w for i, run in enumerate(runs) if run.module == "anpr" and not run.error for w in scores[i]["wrong_plates"]]
    person_rows = [r for i, run in enumerate(runs) if run.module == "person_id" and not run.error for r in scores[i]["rows"]]
    zone_runs = [(run, scores[i]) for i, run in enumerate(runs) if run.module == "zone" and not run.error]
    crossings = [c for _, s in zone_runs for c in s["crossings"]]
    false_alarms = sum(len(s["false_alarms"]) for _, s in zone_runs)
    zone_hours = sum(run.duration_sec for run, _ in zone_runs) / 3600.0
    leads = [t for _, s in zone_runs for t in s["lead_times"]]
    behavior_runs = [(run, scores[i]) for i, run in enumerate(runs) if run.module == "behavior" and not run.error]
    behaviors = [o for _, s in behavior_runs for o in s["occurrences"]]
    behavior_false = sum(len(s["false_alarms"]) for _, s in behavior_runs)
    behavior_hours = sum(run.duration_sec for run, _ in behavior_runs) / 3600.0

    present = [r for r in person_rows if r["present"]]
    absent = [r for r in person_rows if not r["present"]]
    summary = {
        "anpr_exact": (sum(r["exact"] for r in anpr_rows), len(anpr_rows)),
        "anpr_char_accuracy": statistics.mean(r["char_accuracy"] for r in anpr_rows) if anpr_rows else None,
        "anpr_wrong_plates": len(anpr_wrong),
        "person_correct_match": (sum(r["confirmed"] for r in present), len(present)),
        "person_false_match": (sum(r["confirmed"] for r in absent), len(absent)),
        "zone_caught": (sum(c["detected_at"] is not None for c in crossings), len(crossings)),
        "zone_false_alarms": false_alarms,
        "zone_false_alarms_per_hour": (false_alarms / zone_hours) if zone_hours else None,
        "zone_median_warning_sec": statistics.median(leads) if leads else None,
        "behavior_caught": (sum(o["detected_at"] is not None for o in behaviors), len(behaviors)),
        "behavior_false_alarms": behavior_false,
        "behavior_false_alarms_per_hour": (behavior_false / behavior_hours) if behavior_hours else None,
    }
    speed: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        if not run.error and run.processing_fps:
            speed[run.module].append(run.processing_fps)

    md = [
        "# Drishti accuracy report",
        "",
        f"Labels: `{labels_path.name}` · {len({r.clip for r in runs})} clips · "
        f"{sum(len(r.labels) for r in runs)} labelled facts · run {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "| Module | Measure | Result |",
        "|---|---|---|",
        f"| ANPR | Plates read exactly | {pct(*summary['anpr_exact'])} |",
        f"| ANPR | Character accuracy | {summary['anpr_char_accuracy'] * 100:.0f}% |" if summary["anpr_char_accuracy"] is not None else "| ANPR | Character accuracy | no samples |",
        f"| ANPR | Wrong plates reported | {summary['anpr_wrong_plates']} |",
        f"| Person ID | Target found when present | {pct(*summary['person_correct_match'])} |",
        f"| Person ID | False match when absent | {pct(*summary['person_false_match'])} |",
        f"| Restricted zone | Crossings caught | {pct(*summary['zone_caught'])} |",
        f"| Restricted zone | False alarms | {summary['zone_false_alarms']}"
        + (f" ({summary['zone_false_alarms_per_hour']:.1f} per camera-hour)" if summary["zone_false_alarms_per_hour"] is not None else "") + " |",
        f"| Restricted zone | Median warning before crossing | {summary['zone_median_warning_sec']:.1f} s |" if summary["zone_median_warning_sec"] is not None else "| Restricted zone | Median warning before crossing | no samples |",
    ]
    if behavior_runs:
        md += [
            f"| Behaviour | Occurrences caught | {pct(*summary['behavior_caught'])} |",
            f"| Behaviour | False alarms | {summary['behavior_false_alarms']}"
            + (f" ({summary['behavior_false_alarms_per_hour']:.1f} per camera-hour)"
               if summary["behavior_false_alarms_per_hour"] is not None else "") + " |",
        ]
    for module in MODULES:
        if speed.get(module):
            md.append(f"| Speed | {module} processing on this CPU | {statistics.mean(speed[module]):.1f} fps |")

    by_condition: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in anpr_rows:
        by_condition[r["label"].condition or "unlabelled"]["ANPR exact"].append(r["exact"])
    for r in present:
        by_condition[r["label"].condition or "unlabelled"]["Person found"].append(r["confirmed"])
    for c in crossings:
        by_condition[c["label"].condition or "unlabelled"]["Crossing caught"].append(c["detected_at"] is not None)
    for o in behaviors:
        by_condition[o["label"].condition or "unlabelled"]["Behaviour caught"].append(o["detected_at"] is not None)
    if len(by_condition) > 1:
        cols = ["ANPR exact", "Person found", "Crossing caught", "Behaviour caught"]
        md += ["", "## By condition", "", "| Condition | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for cond in sorted(by_condition):
            cells = [pct(sum(v), len(v)) if (v := by_condition[cond].get(c)) else "–" for c in cols]
            md.append(f"| {cond} | " + " | ".join(cells) + " |")

    failures = []
    for r in anpr_rows:
        if not r["exact"]:
            failures.append(f"ANPR · `{r['label'].clip}` · expected {r['truth']}, best read was {r['best_read'] or 'nothing'}")
    for w in anpr_wrong:
        failures.append(f"ANPR · reported a plate nobody labelled: {w}")
    for r in person_rows:
        if r["present"] and not r["confirmed"]:
            failures.append(f"Person ID · `{r['label'].clip}` · target present but not found (ref `{r['label'].reference}`)")
        if not r["present"] and r["confirmed"]:
            failures.append(f"Person ID · `{r['label'].clip}` · matched a person who is not the target (ref `{r['label'].reference}`)")
    for c in crossings:
        if c["detected_at"] is None:
            failures.append(f"Zone · `{c['label'].clip}` · missed the crossing at {c['label'].start_sec:.1f}–{c['label'].end_sec:.1f} s")
    for run, s in zone_runs:
        for t in s["false_alarms"]:
            failures.append(f"Zone · `{run.clip}` · false alarm at {t:.1f} s")
    for o in behaviors:
        if o["detected_at"] is None:
            failures.append(f"Behaviour · `{o['label'].clip}` · missed {o['type']} at {o['label'].start_sec:.1f}–{o['label'].end_sec:.1f} s")
    for run, s in behavior_runs:
        for btype, t in s["false_alarms"]:
            failures.append(f"Behaviour · `{run.clip}` · false {btype} at {t:.1f} s")
    for run in runs:
        if run.error:
            failures.append(f"Not scored · `{run.clip}` ({run.module}) · {run.error}")
    md += ["", "## Misses and errors", ""] + ([f"- {f}" for f in failures] or ["None."])
    md += ["", f"Zone crossings count as caught within ±{tolerance:g} s of the labelled window."]

    data = {
        "labels_file": str(labels_path), "generated": datetime.now().isoformat(timespec="seconds"),
        "summary": summary, "speed_fps": {k: statistics.mean(v) for k, v in speed.items()},
        "failures": failures,
        "runs": [{"module": r.module, "clip": r.clip, "reference": r.reference, "zone": r.zone, "lowlight": r.lowlight,
                  "condition": r.condition, "processing_fps": r.processing_fps, "duration_sec": r.duration_sec,
                  "error": r.error, "summary": r.summary} for r in runs],
    }
    return "\n".join(md) + "\n", data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--videos", type=Path, default=DEFAULT_VIDEOS, help="folder the clip names resolve against")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only", help="comma-separated modules to run: " + ",".join(MODULES))
    parser.add_argument("--zone-tolerance", type=float, default=1.0,
                        help="seconds of slack around a labelled zone crossing or behaviour occurrence")
    args = parser.parse_args()

    labels = read_labels(args.labels)
    if args.only:
        wanted = {m.strip() for m in args.only.split(",")}
        labels = [lab for lab in labels if lab.module in wanted]
    runs = group_runs(labels)

    from backend.modules.anpr.ocr import normalize_plate

    scores: dict[int, dict] = {}
    for i, run in enumerate(runs, start=1):
        print(f"[{i}/{len(runs)}] {run.module:<9} {run.clip}" + (" (low-light on)" if run.lowlight else ""), flush=True)
        execute(run, args.videos)
        if run.error:
            print(f"    skipped: {run.error}")
    for i, run in enumerate(runs):
        if run.error:
            continue
        if run.module == "anpr":
            scores[i] = score_anpr(run, normalize_plate)
        elif run.module == "person_id":
            scores[i] = score_person(run)
        elif run.module == "behavior":
            scores[i] = score_behavior(run, args.zone_tolerance)
        else:
            scores[i] = score_zone(run, args.zone_tolerance)

    md, data = build_report(runs, scores, args.labels, args.zone_tolerance)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text(md, encoding="utf-8")
    (args.out / "report.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print("\n" + md)
    print(f"Saved {args.out / 'report.md'} and report.json")
    return 1 if any(run.error for run in runs) else 0


if __name__ == "__main__":
    sys.exit(main())
