# Third-party components and model weights

Drishti builds on open-source models and libraries. This file lists the ones
that ship with, or are downloaded by, this repository, with the licence each
one is distributed under. **Check each upstream licence yourself before any
deployment or redistribution**, especially a government or commercial one.
Items marked *verify* need legal review before they are relied on.

## Model weights in `models/`

| File | Model | Upstream | Licence |
|---|---|---|---|
| `yolo11n.pt`, `yolov8n.pt`, `yolo11n_int8_openvino_model/` | Ultralytics YOLO11n / YOLOv8n person and vehicle detector | ultralytics/ultralytics | **AGPL-3.0**. Distributing or offering a networked service built on it requires releasing the complete corresponding source under AGPL-3.0, or buying an Ultralytics Enterprise licence. Alternatives with permissive licences: RT-DETR (Apache-2.0), YOLOX (Apache-2.0). |
| `face_detection_yunet_2023mar.onnx` | YuNet face detector | opencv/opencv_zoo | MIT |
| `face_recognition_sface_2021dec.onnx` | SFace face recogniser | opencv/opencv_zoo | Apache-2.0 |
| `arcface_w600k_r50.onnx` (optional, downloaded) | InsightFace ArcFace R50 | deepinsight/insightface | *verify*: InsightFace's pretrained models are released for non-commercial research use |
| `osnet_x0_25_msmt17.onnx` | OSNet person re-identification | KaiyangZhou/deep-person-reid | MIT (code); *verify* the MSMT17 training-data terms |
| `zero_dce_pp.pth` | Zero-DCE++ low-light enhancement | Li-Chongyi/Zero-DCE_extension | *verify* upstream licence |
| `drone_audio/model.json` | Drone audio classifier | trained in this repo (`models/drone_audio/train.py`) | Project's own |

## Downloaded at first use

| Package | Purpose | Licence |
|---|---|---|
| `fast-plate-ocr` (`cct-s-v2-global-model`) | Number-plate OCR | MIT |
| `open-image-models` (`yolo-v9-t-256-license-plate-end2end`) | Number-plate detector | MIT |
| `open_clip_torch` + CLIP ViT-B/32 weights | Appearance search | MIT (code); *verify* the weights' dataset terms |

## Not redistributed

* **OpenH264** (`openh264-1.8.0-win64.dll`) is used by OpenCV to write
  browser-playable H.264. Cisco's binary licence covers only binaries
  downloaded from Cisco, so it is no longer tracked in this repository.
  Download `openh264-1.8.0-win64.dll` from
  <https://github.com/cisco/openh264/releases> and place it in `models/`.
  Without it, output videos fall back to MPEG-4 Part 2 (`mp4v`).
* **The Android siren APK** (`dist/Drishti-Siren.apk`) is a build artefact.
  Build it from `siren-app/` (Capacitor).

## Test fixtures

`uploads/smoke_*.mp4` and `references/smoke_target/*.jpg` are the recorded
clips and reference photos the test suite runs against. They show real
people. Keep them only if the people shown consented to this use; otherwise
replace them with synthetic or licensed footage and remove them from the git
history (`git filter-repo`), not just from the latest commit.
