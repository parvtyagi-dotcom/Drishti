# Measuring Drishti on your own footage

Everything measured so far ran on the demo clips. Some of them are daytime
video darkened in software, and the plate is a printed `TEST123`. That is
enough to check the code works, but not to quote an accuracy. Real numbers
need footage that looks like a real deployment, labelled by a person. There
are two tools for this.

| Question | Tool | What you label |
|---|---|---|
| Did the system raise the right **alert**? (plate read, target found, zone crossing, loitering) | `scripts/evaluate.py` | One CSV row per fact, with a time window |
| How many people and vehicles does the **detector** find in each frame? | `scripts/detection_set.py` | Boxes on sampled frames (pre-drawn, you correct them) |

## 1. Record

Aim for 20-30 clips of 30-120 s from the cameras you will demo with, and
cover the conditions that matter:

- **Day and real night.** Not darkened daytime video. The low-light test
  showed that enhancement can help on one and hurt on the other.
- **Near and far.** Include people far from the camera, under ~60 px tall.
- **Something happening, and nothing happening.** Quiet clips are how false
  alarms get counted.
- **Plates:** real Indian plates at the distance and angle the gate camera
  actually sees them.

## 2. Alert labels (`evaluate.py`)

Copy `demo_labels.csv` to `campus.csv` and add one row per fact. Write the
times down while you watch each clip. `start_sec` and `end_sec` should
bracket the moment, a few seconds wide.

```
clip,module,expected,start_sec,end_sec,reference,zone,lowlight,condition,calibration
gate_01.mp4,anpr,DL3CAB1234,,,,,no,day,
gate_01.mp4,person_id,present,,,references/team/asha,,no,day,
fence_03.mp4,zone,entry,12.0,14.5,,zones/<zone-id>.json,no,night,
fence_03.mp4,behavior,LOITERING,30,45,,,no,night,data/calibrations/<camera-id>.json
fence_04.mp4,behavior,none,,,,,no,night,
```

Behaviour types are `LOITERING`, `ABNORMAL_SPEED`, `SUDDEN_DIRECTION_CHANGE`
and `REPEATED_BACK_AND_FORTH`. Use `none` for a clip where nothing
suspicious happens, so false alarms get counted. The `calibration` column is
optional. Point it at a camera's saved ground calibration to score
behaviour in metres.

```
python scripts/evaluate.py --labels eval/campus.csv --videos D:/footage
```

## 3. Detector boxes (`detection_set.py`)

```
python -m scripts.detection_set sample --videos D:/footage --every 2 --out eval/detection/campus
```

1. Open [makesense.ai](https://www.makesense.ai). It is free, runs in the
   browser and needs no account. Load `images/` and choose **Object
   Detection**. Import `labels/*.txt` as YOLO, with labels `person` and
   `vehicle` (the order in `classes.txt`).
2. On every frame:
   - add each missed person or vehicle, including tiny far-away ones;
   - delete wrong boxes;
   - tighten loose boxes.
3. Export as YOLO and replace `labels/` with the export.
4. In `frames.csv`, fill in `condition` (day / night / rain ...) and set
   `checked` to `yes` for every frame you went through.

```
python -m scripts.detection_set score --set eval/detection/campus --model models/yolov8n.pt --model models/yolo11n_int8_openvino_model
python -m scripts.detection_set score --set eval/detection/campus --lowlight fast
```

Only `checked=yes` frames are scored. The pre-drawn boxes come from the
current detector, so an unchecked set just measures the detector against
itself. Against its own pre-labels it already scores ~98%, and that number
means nothing.

## What the reports give you

- **ANPR:** plates read exactly, character accuracy, wrong plates reported.
- **Person ID:** target found when present, false matches when absent.
- **Zones:** crossings caught, false alarms per camera-hour, and seconds of
  warning before the crossing.
- **Behaviour:** occurrences caught, false alarms per camera-hour.
- **Detector:** recall and precision per class, per condition, and for far
  (<60 px tall) against near people.
