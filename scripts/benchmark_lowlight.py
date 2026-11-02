"""Low-light A/B benchmark: no enhancement vs CLAHE+gamma ("fast") vs
Zero-DCE++ ("zero_dce"), on the dark demo clips.

Per clip and method:
  ms/frame    enhancement time alone (detector excluded)
  mean L      mean brightness, LAB L channel 0-255 (the low-light trigger is 65)
  dets/frame  people+vehicles the real detector finds on the enhanced frames
  mean conf   their mean confidence
and end to end, through the real pipelines:
  Person ID   is the target confirmed, and on how many frames is it visible
  ANPR        is TEST123 found

Also writes outputs/lowlight_compare_<clip>.jpg: original | fast | zero_dce.
Like the other benchmarks there is no box-level ground truth for these
clips, so detections and confidence are proxies, not precision/recall.

    python -m scripts.benchmark_lowlight [--frames 60]
"""
from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import replace

import cv2
import numpy as np

from backend.config import OUTPUTS_DIR, REFERENCES_DIR, UPLOADS_DIR, LowLightConfig, TrackerConfig
from backend.core.lowlight import process_frame
from backend.core.tracker import ObjectTracker

CLIPS = ["smoke_person_id_dark.mp4", "smoke_anpr_test123_dark.mp4", "smoke_lowlight_real_dark.mp4"]
METHODS = {
    "none": None,
    "fast": LowLightConfig(backend="fast"),
    "zero_dce": LowLightConfig(backend="zero_dce"),
}


def _mean_l(image: np.ndarray) -> float:
    return float(cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0].mean())


def _read(clip: str, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(UPLOADS_DIR / clip))
    frames = []
    while len(frames) < n:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def frame_level(n_frames: int) -> None:
    base = TrackerConfig()
    for clip in CLIPS:
        frames = _read(clip, n_frames)
        if not frames:
            print(f"skip {clip}: not found")
            continue
        h, w = frames[0].shape[:2]
        print(f"\n{clip}  ({w}x{h}, {len(frames)} frames)")
        print(f"  {'method':<10}{'ms/frame':>10}{'mean L':>8}{'dets/fr':>9}{'mean conf':>11}")
        samples = []
        for name, cfg in METHODS.items():
            if cfg is not None:
                process_frame(frames[0], cfg)  # load/compile outside the timing
            times, out = [], []
            for f in frames:
                t0 = time.perf_counter()
                img = f if cfg is None else process_frame(f, cfg)[0]
                times.append(time.perf_counter() - t0)
                out.append(img)
            tracker = ObjectTracker(replace(base, detection_stride=1),
                                    class_ids=(base.person_class_id, *base.vehicle_class_ids))
            counts, confs = [], []
            for img in out:
                tracks = tracker.update(img)
                counts.append(len(tracks))
                confs.extend(t.confidence for t in tracks)
            ms = statistics.median(times) * 1000 if cfg is not None else 0.0
            print(f"  {name:<10}{ms:>10.1f}{statistics.mean(_mean_l(i) for i in out):>8.1f}"
                  f"{statistics.mean(counts):>9.2f}{(statistics.mean(confs) if confs else 0):>11.3f}")
            samples.append(out[len(out) // 2])
        strip = np.hstack([cv2.resize(s, (480, int(480 * h / w))) for s in samples])
        cv2.imwrite(str(OUTPUTS_DIR / f"lowlight_compare_{clip.removesuffix('.mp4')}.jpg"), strip)


def end_to_end() -> None:
    from backend.modules.anpr.pipeline import AnprPipeline
    from backend.modules.person_id.pipeline import PersonIDPipeline

    ref = REFERENCES_DIR / "smoke_target" / "ref1.jpg"
    print("\nEnd to end on the dark clips")
    print(f"  {'method':<10}{'Person ID confirmed':>21}{'visible frames':>16}{'ANPR TEST123':>14}")
    for name, cfg in METHODS.items():
        out = OUTPUTS_DIR / "benchmark_lowlight.mp4"
        p = PersonIDPipeline().run(UPLOADS_DIR / "smoke_person_id_dark.mp4", [ref], out, lowlight_config=cfg) \
            if ref.is_file() else None
        a = AnprPipeline().run(UPLOADS_DIR / "smoke_anpr_test123_dark.mp4", out, target_plate="TEST123",
                               lowlight_config=cfg)
        out.unlink(missing_ok=True)
        print(f"  {name:<10}{str(p.confirmed) if p else 'no ref':>21}{(p.visible_frame_count if p else 0):>16}"
              f"{str(bool(a.target and a.target.found)):>14}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=60)
    args = ap.parse_args()
    frame_level(args.frames)
    end_to_end()


if __name__ == "__main__":
    main()
