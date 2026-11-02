"""Check that this machine is ready to run Drishti, before it matters.

Run it the day before and again at the venue, with the network cable out:

    python scripts/preflight.py

Each line is a check with a tick or a cross, and every cross says how to fix
it. Nothing here changes anything; it only looks. Exits non-zero if any
check that would stop the demo fails, so it can also gate a script.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import socket
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

OK, WARN, FAIL = "ok", "warn", "fail"
MARK = {OK: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]"}
results: list[tuple[str, str, str]] = []


def check(name: str, status: str, detail: str = "") -> None:
    results.append((status, name, detail))
    print(f"{MARK[status]} {name}" + (f" - {detail}" if detail else ""), flush=True)


def python_version() -> None:
    ok = sys.version_info >= (3, 11)
    check("Python 3.11 or newer", OK if ok else FAIL, f"found {sys.version.split()[0]}")


def packages() -> None:
    needed = ["torch", "cv2", "ultralytics", "fastapi", "uvicorn", "multipart", "httpx", "numpy",
              "fast_plate_ocr", "open_image_models", "onnxruntime", "lap"]
    missing = [m for m in needed if importlib.util.find_spec(m) is None]
    check("Python packages", FAIL if missing else OK,
          f"missing {', '.join(missing)}: run pip install -r requirements.txt" if missing else f"{len(needed)} found")
    if importlib.util.find_spec("anthropic") is None:
        check("Claude SDK (optional)", WARN, "not installed: cloud AI briefs unavailable; pip install anthropic")


def model_files() -> None:
    from backend.config import OSNET_MODEL_PATH, SFACE_MODEL_PATH, YOLO_MODEL_PATH, YUNET_MODEL_PATH

    missing = [p.name for p in (YOLO_MODEL_PATH, YUNET_MODEL_PATH, SFACE_MODEL_PATH, OSNET_MODEL_PATH) if not p.exists()]
    check("Detection model files", FAIL if missing else OK,
          f"missing in models/: {', '.join(missing)}" if missing else "YOLO, face and re-identification models present")
    home = Path.home()
    plate_cache = [home / ".cache" / "open-image-models", home / ".cache" / "fast-plate-ocr"]
    if not all(p.exists() for p in plate_cache):
        check("Plate reader models", WARN,
              "not downloaded yet; they download on first ANPR use, which needs internet once. "
              "Run scripts/evaluate.py --only anpr while online to fetch them.")
    else:
        check("Plate reader models", OK, "cached for offline use")


def frontend_build() -> None:
    dist = REPO / "frontend" / "dist" / "index.html"
    if not dist.is_file():
        check("Dashboard build", FAIL, "not built: cd frontend && npm install && npm run build")
        return
    newest_src = max((p.stat().st_mtime for p in (REPO / "frontend" / "src").rglob("*") if p.is_file()), default=0)
    stale = newest_src > dist.stat().st_mtime
    check("Dashboard build", WARN if stale else OK,
          "older than the source: rebuild with npm run build in frontend/" if stale else "up to date")


def demo_assets() -> None:
    clips = sorted((REPO / "uploads").glob("smoke_*.mp4"))
    check("Demo clips", OK if clips else WARN,
          f"{len(clips)} in uploads/" if clips else "missing: git checkout -- uploads/")
    cert = REPO / "certs" / "dev-cert.pem"
    check("HTTPS certificate (phone cameras)", OK if cert.is_file() else WARN,
          "present" if cert.is_file() else "missing: python scripts/make_dev_cert.py")


def ports() -> None:
    for port in (8000, 8443):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            in_use = s.connect_ex(("127.0.0.1", port)) == 0
        check(f"Port {port}", OK if not in_use else WARN,
              "free" if not in_use else "in use (Drishti already running? otherwise close whatever holds it)")


def network() -> None:
    from backend.api.network import lan_addresses

    addresses = lan_addresses()
    check("Local network address", OK if addresses else WARN,
          ", ".join(addresses) if addresses else "none: phones, the siren app and IP cameras cannot reach this machine")
    try:
        socket.create_connection(("1.1.1.1", 443), timeout=2).close()
        check("Internet", OK, "reachable (cloud AI can be used)")
    except OSError:
        check("Internet", OK, "not reachable - fine: everything except cloud AI works offline")


def local_ai() -> None:
    import httpx

    from backend.config import get_config

    config = get_config().agent
    try:
        tags = httpx.get(f"{config.local_url}/api/tags", timeout=3).json()
    except Exception:
        check("Local AI (Ollama)", WARN, f"not running at {config.local_url}: briefs fall back to rules. Start the Ollama app.")
        return
    names = {m.get("name") for m in tags.get("models", [])}
    has = config.local_model in names or f"{config.local_model}:latest" in names
    check("Local AI (Ollama)", OK if has else WARN,
          f"running with {config.local_model}" if has else f"running, but {config.local_model} is missing: ollama pull {config.local_model}")
    check("AI analyst switched on", OK if config.enabled else WARN,
          "DRISHTI_AGENTS=1" if config.enabled else "off: set DRISHTI_AGENTS=1 before starting to get AI briefs")


def search_and_sensors() -> None:
    from backend.live.appearance import model_available
    from backend.live.sensors import SensorRegistry

    try:
        import open_clip  # noqa: F401
        installed = True
    except ImportError:
        installed = False
    if not installed:
        check("Appearance search", WARN, "open_clip not installed: search is off (see requirements.txt)")
    else:
        check("Appearance search", OK if model_available() else WARN,
              "model downloaded" if model_available() else "model missing: python scripts/download_search_model.py")
    sensors = SensorRegistry().list()
    check("Sensors", OK if sensors else WARN,
          f"{len(sensors)} configured" if sensors else "none yet: add one on the Sensors page to show sensor fusion")


def accounts_and_log() -> None:
    from backend.config import DATA_DIR, EVENT_DB_PATH

    users_path = DATA_DIR / "users.json"
    try:
        users = json.loads(users_path.read_text(encoding="utf-8"))
        admins = sum(1 for u in users.values() if u.get("role") == "ADMIN")
        check("Accounts", OK if admins else FAIL, f"{len(users)} accounts, {admins} administrator(s)" if admins else "no administrator account")
    except (OSError, ValueError):
        check("Accounts", WARN, "none yet: the first visit to the dashboard creates the administrator")

    if not EVENT_DB_PATH.is_file():
        check("Event log integrity", OK, "no log yet (created on first start)")
        return
    from backend.live.store import EventStore

    status = EventStore(EVENT_DB_PATH).verify_chain()
    check("Event log integrity", OK if status.valid else FAIL,
          f"{status.total_events} records verified" if status.valid else f"broken at record {status.broken_at_id}: {status.detail}")


def disk() -> None:
    free_gb = shutil.disk_usage(REPO).free / 1024 ** 3
    check("Free disk space", OK if free_gb >= 5 else WARN, f"{free_gb:.0f} GB free" + ("" if free_gb >= 5 else ": evidence clips need room"))


def main() -> int:
    print(f"Drishti pre-flight check - {REPO}\n")
    for step in (python_version, packages, model_files, frontend_build, demo_assets, ports, network,
                 local_ai, search_and_sensors, accounts_and_log, disk):
        try:
            step()
        except Exception as exc:  # a broken check must not hide the others
            check(step.__name__.replace("_", " ").capitalize(), FAIL, f"check itself failed: {exc}")
    fails = [r for r in results if r[0] == FAIL]
    warns = [r for r in results if r[0] == WARN]
    print(f"\n{len(results) - len(fails) - len(warns)} ok, {len(warns)} to look at, {len(fails)} blocking.")
    print("Ready to run." if not fails else "Fix the [FAIL] lines before the demo.")
    return 1 if fails else 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.exit(main())
