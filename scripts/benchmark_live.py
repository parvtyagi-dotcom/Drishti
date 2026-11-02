"""Measured performance of Live Mode: per-module cost, resolution, and scaling.

Every number here is produced by running the real pipeline on real footage.
Nothing is extrapolated, and the output is meant to be quotable as-is - a
claimed frame rate that nobody measured is worth less than an honest low one.

Usage:
    python scripts/benchmark_live.py            # full run (~10 minutes)
    python scripts/benchmark_live.py --quick    # shorter samples
"""
from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.core.video import LiveStreamReader  # noqa: E402
from backend.live.camera_manager import live_behavior_config  # noqa: E402
from backend.orchestrator import CombinedPipeline, OrchestratorRequest  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
PERSON_CLIP = REPO / "uploads" / "smoke_person_id.mp4"
VEHICLE_CLIP = REPO / "uploads" / "real_checkpoint_gate.mp4"
REFERENCES = sorted((REPO / "references" / "smoke_target").glob("*.jpg"))


def run_once(source: Path, seconds: float, max_width: int, **request_kwargs) -> dict:
    """Run one configuration and report what it actually achieved."""
    reader = LiveStreamReader(str(source), loop=True, target_fps=12, max_width=max_width)
    reader.connect()
    if request_kwargs.get("enable_behavior"):
        request_kwargs.setdefault("behavior_config", live_behavior_config())
    pipeline = CombinedPipeline(OrchestratorRequest(**request_kwargs))

    events: list[dict] = []
    frame_times: list[float] = []
    started = time.monotonic()
    last = started

    def on_frame(_image, new_events):
        nonlocal last
        now = time.monotonic()
        frame_times.append(now - last)
        last = now
        events.extend(new_events)
        if now - started > seconds:
            reader.stop()

    pipeline.run(reader=reader, write_output=False, frame_callback=on_frame)
    elapsed = time.monotonic() - started
    analysed = reader.status.frames_read

    return {
        "fps": analysed / elapsed if elapsed else 0.0,
        "frames": analysed,
        "dropped": reader.status.frames_dropped,
        "events": len(events),
        "resolution": f"{reader.info.width}x{reader.info.height}",
        # The slowest frames matter more than the average: a feed that stalls
        # for a second at the wrong moment misses the thing it was watching for.
        "p95_frame_ms": (statistics.quantiles(frame_times, n=20)[-1] * 1000)
        if len(frame_times) > 20 else 0.0,
    }


def module_costs(seconds: float) -> None:
    print("\n=== Per-module cost (960px, single camera) ===")
    print(f"{'modules':<34}{'fps':>7}{'p95 ms':>9}{'events':>8}")
    configs = [
        ("detection + tracking only", {"zones": [], "enable_behavior": True}),
        ("+ low-light enhancement", {"enable_behavior": True, "enable_lowlight": True}),
        ("person ID (face + re-ID)", {"enable_person_id": True, "reference_photo_paths": REFERENCES}),
        ("ANPR", {"enable_anpr": True}),
    ]
    for label, kwargs in configs:
        source = VEHICLE_CLIP if "ANPR" in label else PERSON_CLIP
        if not source.is_file():
            print(f"{label:<34}{'skipped - missing footage':>24}")
            continue
        r = run_once(source, seconds, 960, **kwargs)
        print(f"{label:<34}{r['fps']:>7.1f}{r['p95_frame_ms']:>9.0f}{r['events']:>8}")


def resolution_cost(seconds: float) -> None:
    """Cost of analysis resolution.

    Uses genuinely high-resolution footage: the reader never upscales, so
    asking for 1920px from a 960px clip silently measures 960px twice and
    makes resolution look free when it is not.
    """
    if not VEHICLE_CLIP.is_file():
        print("\n=== Analysis resolution === skipped - needs high-resolution footage")
        return
    print("\n=== Analysis resolution (detection + tracking, 4K source) ===")
    print(f"{'width':<34}{'fps':>7}{'p95 ms':>9}{'resolution':>14}")
    for width in (640, 960, 1440, 1920):
        r = run_once(VEHICLE_CLIP, seconds, width, enable_behavior=True)
        print(f"{str(width) + 'px':<34}{r['fps']:>7.1f}{r['p95_frame_ms']:>9.0f}{r['resolution']:>14}")


def concurrency(seconds: float) -> None:
    """How throughput degrades as cameras are added.

    Run in threads, as the server does. Python's GIL is not the limit here -
    OpenCV and ONNX Runtime release it during inference - but CPU cores are,
    so per-camera throughput falls as cameras compete for them.
    """
    print("\n=== Multi-camera scaling (960px, zone + behaviour) ===")
    print(f"{'cameras':<34}{'fps/cam':>9}{'total fps':>11}{'vs 1 cam':>10}")
    baseline = None
    for count in (1, 2, 4):
        results: list[dict] = []
        lock = threading.Lock()

        def worker():
            r = run_once(PERSON_CLIP, seconds, 960, enable_behavior=True)
            with lock:
                results.append(r)

        threads = [threading.Thread(target=worker) for _ in range(count)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        per_cam = statistics.mean(r["fps"] for r in results)
        total = sum(r["fps"] for r in results)
        baseline = baseline or per_cam
        print(f"{str(count) + ' camera(s)':<34}{per_cam:>9.1f}{total:>11.1f}"
              f"{per_cam / baseline * 100:>9.0f}%")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="shorter samples")
    args = parser.parse_args()
    seconds = 20.0 if args.quick else 45.0

    if not PERSON_CLIP.is_file():
        print(f"missing test footage: {PERSON_CLIP}")
        return 1

    print("Drishti live performance - measured, CPU only")
    print(f"sample length: {seconds:.0f}s per configuration")
    print("NOTE: cameras capture at 12fps by design, so any figure at ~12 is")
    print("      hitting that ceiling rather than the CPU. Only results below")
    print("      12 show a genuine compute limit.")
    module_costs(seconds)
    resolution_cost(seconds)
    concurrency(seconds)
    print("\nAll figures measured on this machine; they are not projections.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
