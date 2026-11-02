"""Detector A/B benchmark: YOLOv8n vs YOLO11n, PyTorch vs OpenVINO FP32/INT8.

Runs each model through the real ObjectTracker (YOLO + ByteTrack, the same
classes and thresholds the pipelines use, detection_stride=1 so every frame is
a true detection) over the demo clips, and reports per clip:

  ms/frame   mean detector+tracker time per frame, after a warm-up
  dets/frame mean confirmed tracks per frame (higher = fewer misses, if the
             extra boxes are real - check the annotated output when in doubt)
  mean conf  mean confidence of those tracks
  track ids  distinct track ids seen (lower for the same scene = fewer id
             switches / broken tracks)

The demo clips have no box-level ground truth, so these are proxies, not
precision/recall. Read-only diagnostic: changes no production code.

    python scripts/benchmark_detector.py [--frames 150]

Only yolov8n.pt, yolo11n.pt and the yolo11n INT8 export are committed; a
missing model is reported and skipped. To recreate the others (run from a
scratch directory - ultralytics writes the download into the cwd):

    YOLO("models/yolo11n.pt").export(format="openvino")            # FP32
    YOLO("models/yolov8n.pt").export(format="openvino", int8=True, data=...)

INT8 needs `pip install nncf==2.14.1` and a calibration set: data= is a YOLO
dataset yaml whose val images are frames from the cameras' own footage (the
committed model used 322 frames, two random ones from each clip in uploads/).
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import MODELS_DIR, UPLOADS_DIR, TrackerConfig  # noqa: E402
from backend.core.tracker import ObjectTracker  # noqa: E402

MODELS = {
    "v8n-pt": MODELS_DIR / "yolov8n.pt",
    "v8n-ov": MODELS_DIR / "yolov8n_openvino_model",
    "11n-pt": MODELS_DIR / "yolo11n.pt",
    "11n-ov": MODELS_DIR / "yolo11n_openvino_model",
    "v8n-i8": MODELS_DIR / "yolov8n_int8_openvino_model",
    "11n-i8": MODELS_DIR / "yolo11n_int8_openvino_model",
}
CLIPS = [
    "smoke_person_id.mp4",
    "smoke_person_id_dark.mp4",
    "smoke_person_id_reacquire.mp4",
    "smoke_anpr_test123.mp4",
    "smoke_anpr_test123_dark.mp4",
    "smoke_lowlight_real_dark.mp4",
]
WARMUP = 5


def run(model_path: Path, clip: Path, n_frames: int) -> dict:
    base = TrackerConfig()
    cfg = replace(base, model_path=model_path, detection_stride=1)
    tracker = ObjectTracker(cfg, class_ids=(base.person_class_id, *base.vehicle_class_ids))
    cap = cv2.VideoCapture(str(clip))
    times, counts, confs, ids = [], [], [], set()
    i = 0
    while i < n_frames + WARMUP:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        tracks = tracker.update(frame)
        dt = time.perf_counter() - t0
        if i >= WARMUP:
            times.append(dt)
            counts.append(len(tracks))
            confs.extend(t.confidence for t in tracks)
            ids.update(t.track_id for t in tracks)
        i += 1
    cap.release()
    return {
        "frames": len(times),
        "ms": statistics.mean(times) * 1000 if times else 0.0,
        "dets": statistics.mean(counts) if counts else 0.0,
        "conf": statistics.mean(confs) if confs else 0.0,
        "ids": len(ids),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=150)
    args = ap.parse_args()

    totals: dict[str, list[dict]] = {name: [] for name in MODELS}
    for clip_name in CLIPS:
        clip = UPLOADS_DIR / clip_name
        if not clip.is_file():
            print(f"skip {clip_name}: not found")
            continue
        print(f"\n{clip_name}")
        print(f"  {'model':<8}{'frames':>7}{'ms/frame':>10}{'fps':>7}{'dets/fr':>9}{'mean conf':>11}{'track ids':>11}")
        for name, path in MODELS.items():
            if not path.exists():
                print(f"  {name:<8} missing {path.name}")
                continue
            r = run(path, clip, args.frames)
            totals[name].append(r)
            fps = 1000 / r["ms"] if r["ms"] else 0
            print(f"  {name:<8}{r['frames']:>7}{r['ms']:>10.1f}{fps:>7.1f}{r['dets']:>9.2f}{r['conf']:>11.3f}{r['ids']:>11}")

    print("\nOverall (mean across clips)")
    print(f"  {'model':<8}{'ms/frame':>10}{'fps':>7}{'dets/fr':>9}{'mean conf':>11}")
    for name, rs in totals.items():
        if not rs:
            continue
        ms = statistics.mean(r["ms"] for r in rs)
        print(f"  {name:<8}{ms:>10.1f}{1000 / ms:>7.1f}"
              f"{statistics.mean(r['dets'] for r in rs):>9.2f}{statistics.mean(r['conf'] for r in rs):>11.3f}")


if __name__ == "__main__":
    main()
