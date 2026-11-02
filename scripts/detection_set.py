"""Box-level test set for the person/vehicle detector: build it, then score models on it.

scripts/evaluate.py answers "did the system raise the right alert?". This
answers the question underneath it: "how many of the people and vehicles in
the picture does the detector actually find, and how many of its boxes are
wrong?" - the number to quote when the detector, the low-light enhancement
or the frame size changes. The benchmarks so far could only compare
confidence and box counts; this measures precision and recall against boxes
a person has checked.

1. SAMPLE frames from your own footage, with boxes pre-drawn by the current
   detector so a person only corrects them rather than drawing every one:

       python -m scripts.detection_set sample --videos D:/footage --every 2 --out eval/detection/campus

   Writes images/*.jpg, labels/*.txt (YOLO format: "class cx cy w h",
   normalized; class 0 = person, 1 = vehicle), classes.txt, and frames.csv
   with a `condition` column to fill in (day, night, rain, far, ...).

2. CORRECT the boxes in any tool that reads YOLO labels - for example
   makesense.ai (free, in the browser, nothing to install: open images and
   labels/*.txt, fix, export "YOLO"), CVAT or Label Studio. Add every
   missed person/vehicle, delete every wrong box, tighten loose ones. An
   UNCHECKED pre-labelled set scores the current detector as perfect; the
   numbers mean something only after a person has been through every frame.
   Mark a frame done by setting checked=yes in frames.csv - only checked
   frames are scored.

3. SCORE one or more models on it:

       python -m scripts.detection_set score --set eval/detection/campus
       python -m scripts.detection_set score --set eval/detection/campus --model models/yolov8n.pt --model models/yolo11n.pt
       python -m scripts.detection_set score --set eval/detection/campus --lowlight fast

   A detection counts if it overlaps a labelled box of the same class with
   IoU >= 0.5 (each labelled box matched at most once). Reports precision,
   recall and F1 per class, by condition, and by size - "far" is a labelled
   box under 60 px tall, the long-range intruder the border cameras most
   need to catch.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CLASSES = ("person", "vehicle")
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
IOU_MATCH = 0.5
FAR_PX = 60  # labelled box height below this counts as "far"


def _detector_classes():
    from backend.config import TrackerConfig

    cfg = TrackerConfig()
    return cfg.person_class_id, set(cfg.vehicle_class_ids)


def detect(model, image: np.ndarray, conf: float) -> list[tuple[int, float, tuple[float, float, float, float]]]:
    """(class 0/1, confidence, xyxy in pixels) for each person/vehicle box."""
    person, vehicles = _detector_classes()
    r = model.predict(image, conf=conf, classes=[person, *sorted(vehicles)], verbose=False)[0]
    if r.boxes is None:
        return []
    out = []
    for cls, c, box in zip(r.boxes.cls.int().tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
        out.append((0 if cls == person else 1, float(c), tuple(box)))
    return out


# --- sample -------------------------------------------------------------------

def sample(args) -> None:
    from ultralytics import YOLO

    from backend.config import YOLO_MODEL_PATH

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    model = YOLO(str(args.model or YOLO_MODEL_PATH))
    clips = sorted(p for p in Path(args.videos).rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES)
    if not clips:
        raise SystemExit(f"no videos under {args.videos}")

    rows = []
    for clip in clips:
        cap = cv2.VideoCapture(str(clip))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, round(fps * args.every))
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % step == 0:
                name = f"{clip.stem}_{index:06d}"
                cv2.imwrite(str(out / "images" / f"{name}.jpg"), frame)
                h, w = frame.shape[:2]
                lines = []
                for cls, _, (x1, y1, x2, y2) in detect(model, frame, args.conf):
                    lines.append(f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
                (out / "labels" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                rows.append({"image": f"{name}.jpg", "clip": clip.name, "time_sec": f"{index / fps:.2f}",
                             "condition": "", "checked": "no"})
            index += 1
        cap.release()
        print(f"{clip.name}: {sum(r['clip'] == clip.name for r in rows)} frames")

    with open(out / "frames.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "clip", "time_sec", "condition", "checked"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} frames in {out}. Pre-labelled boxes are a starting point only - correct every frame,")
    print("then set checked=yes in frames.csv. Only checked frames are scored.")


# --- score --------------------------------------------------------------------

def read_labels(path: Path, w: int, h: int) -> list[tuple[int, tuple[float, float, float, float]]]:
    boxes = []
    if not path.is_file():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
        boxes.append((cls, ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)))
    return boxes


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(truth: list, preds: list) -> tuple[list[bool], int]:
    """Greedy one-to-one matching, most confident prediction first. Returns
    (for each labelled box: was it found, number of unmatched predictions)."""
    found = [False] * len(truth)
    false_pos = 0
    for cls, _, box in sorted(preds, key=lambda p: -p[1]):
        best, best_iou = None, IOU_MATCH
        for i, (tcls, tbox) in enumerate(truth):
            if not found[i] and tcls == cls and (v := iou(box, tbox)) >= best_iou:
                best, best_iou = i, v
        if best is None:
            false_pos += 1
        else:
            found[best] = True
    return found, false_pos


def score(args) -> None:
    from ultralytics import YOLO

    from backend.config import YOLO_MODEL_PATH, LowLightConfig
    from backend.core.lowlight import process_frame

    root = Path(args.set)
    with open(root / "frames.csv", newline="", encoding="utf-8") as f:
        frames = [r for r in csv.DictReader(f) if r.get("checked", "").strip().lower() in ("yes", "y", "1", "true")]
    if not frames:
        raise SystemExit(f"no frames marked checked=yes in {root / 'frames.csv'} - correct the labels first")
    lowlight = LowLightConfig(backend=args.lowlight) if args.lowlight else None
    models = args.model or [str(YOLO_MODEL_PATH)]
    print(f"{len(frames)} checked frames" + (f", low-light '{args.lowlight}' on" if lowlight else ""))

    for model_path in models:
        model = YOLO(str(model_path))
        # tallies[group] = [found, labelled, false positives]
        tallies: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
        for row in frames:
            image = cv2.imread(str(root / "images" / row["image"]))
            if image is None:
                continue
            if lowlight is not None:
                image = process_frame(image, lowlight)[0]
            h, w = image.shape[:2]
            truth = read_labels(root / "labels" / (Path(row["image"]).stem + ".txt"), w, h)
            preds = detect(model, image, args.conf)
            found, _ = match(truth, preds)
            groups_of = lambda cls, box: [
                "all", CLASSES[cls], f"condition: {row.get('condition') or 'unlabelled'}",
                "far (<60px tall)" if box[3] - box[1] < FAR_PX else "near",
            ]
            for (cls, box), hit in zip(truth, found):
                for g in groups_of(cls, box):
                    tallies[g][0] += hit
                    tallies[g][1] += 1
            # False positives, attributed to the class (and condition) they claimed.
            for cls in (0, 1):
                _, fp = match([t for t in truth if t[0] == cls], [p for p in preds if p[0] == cls])
                for g in ("all", CLASSES[cls], f"condition: {row.get('condition') or 'unlabelled'}"):
                    tallies[g][2] += fp

        print(f"\n{Path(model_path).name}")
        print(f"  {'group':<28}{'labelled':>9}{'recall':>8}{'precision':>11}{'F1':>7}")
        for g in sorted(tallies, key=lambda g: (g != "all", g)):
            tp, n, fp = tallies[g]
            recall = tp / n if n else 0.0
            # Size groups have no false-positive count (an extra box has no size label).
            has_fp = not g.startswith(("far", "near"))
            precision = tp / (tp + fp) if has_fp and (tp + fp) else None
            f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
            print(f"  {g:<28}{n:>9}{recall:>8.1%}" + (f"{precision:>11.1%}" if precision is not None else f"{'-':>11}")
                  + (f"{f1:>7.2f}" if f1 is not None else f"{'-':>7}"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="extract and pre-label frames")
    s.add_argument("--videos", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--every", type=float, default=2.0, help="seconds between sampled frames")
    s.add_argument("--model", help="detector used for pre-labelling (default: the configured one)")
    s.add_argument("--conf", type=float, default=0.35)
    c = sub.add_parser("score", help="score models against the checked labels")
    c.add_argument("--set", required=True)
    c.add_argument("--model", action="append", help="repeat to compare several models")
    c.add_argument("--conf", type=float, default=0.35, help="the pipeline's own threshold by default")
    c.add_argument("--lowlight", choices=["fast", "zero_dce"], help="enhance frames before detecting")
    args = ap.parse_args()
    sample(args) if args.cmd == "sample" else score(args)


if __name__ == "__main__":
    main()
