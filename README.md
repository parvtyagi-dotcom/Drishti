# Drishti

Drishti is an AI-assisted video monitoring system for existing CCTV
infrastructure. It combines continuous live monitoring with forensic analysis
of uploaded recordings, and is designed to run on CPU-only hardware.

## Capabilities

- **Live monitoring:** connect RTSP cameras or local video sources, monitor
  their health, and receive alerts.
- **Forensic analysis:** submit a recording for asynchronous processing and
  review its results.
- **Person identification:** use reference images for person detection and
  appearance-based re-identification.
- **Number-plate recognition (ANPR):** detect and validate plates, aggregate
  reads across frames, and compare them with a watchlist.
- **Restricted zones and behaviour analytics:** configure zones and detect
  approaches, crossings, loitering, pacing, and related events.
- **Evidence and operations:** capture event evidence, keep an auditable event
  history, and use the companion Android siren app.

Drishti is a decision-support system. Alerts must be reviewed by an operator;
model output is not proof of identity or wrongdoing.

## Requirements

- Python 3.11 or newer
- Node.js 20.19+ (or a newer version supported by Vite 8)
- Tesseract OCR for the optional ANPR fallback
- A camera/video source and model assets for the features you enable

The Python dependencies include CPU-only PyTorch by default. GPU installations
require selecting compatible PyTorch packages and configuring the device in
`backend/config.py`.

## Run locally

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
npm --prefix frontend ci
npm --prefix frontend run build
python scripts/serve.py
```

On macOS or Linux, activate the environment with
`source .venv/bin/activate`. Open <http://127.0.0.1:8000> in a browser. The
first-run flow asks you to create an administrator account; there is no
default password.

For phone-camera access, browsers require HTTPS. The development certificate
helper is `scripts/make_dev_cert.py`; use certificates only for local testing.
The Android siren client also uses the local network service.

## Model and test assets

Large third-party model weights, reference photos, and recorded demo footage
are intentionally not included in this repository. This avoids publishing
biometric/sample media and redistributing model binaries without their
licenses. Obtain model files from their upstream sources, review the
applicable terms, and place the required files in the `models/` directory.
Use your own properly authorized images and recordings for testing.

The application does not include demo camera footage. Configure a camera or
provide a local video through the application before starting an analysis.

## Development and tests

Run the backend tests:

```powershell
python -m pytest backend/tests -q
```

The full suite includes tests that initialize vision models, so supply the
model assets described above before running all tests.

Run the frontend development server:

```powershell
npm --prefix frontend run dev
```

Build and lint the frontend:

```powershell
npm --prefix frontend run build
npm --prefix frontend run lint
```

## Project layout

```text
backend/       FastAPI service, video pipeline, live camera and event handling
  api/         HTTP endpoints and request/response schemas
  core/        Tracking, video, and event primitives
  live/        Camera management, evidence, and live state
  modules/     ANPR, person identification, zones, and behaviour analysis
  tests/       Backend tests
frontend/      React, TypeScript, and Vite operator interface
siren-app/     Capacitor Android siren client
scripts/       Local server and development utilities
```

## Security and privacy

Do not commit credentials, private keys, event databases, user records,
enrolled reference images, or surveillance footage. The repository ignore
rules exclude runtime state and local assets by default. Use synthetic or
authorized data in development, restrict access to deployments, and rotate any
credential that may have been exposed.

## Limitations

- Recognition quality depends on camera placement, image quality, lighting,
  and the supplied model files. Distant or oblique plates may not be readable.
- CPU throughput varies with hardware, resolution, and the number of active
  cameras.
- This project is not a replacement for human review, site-specific testing,
  or applicable privacy and surveillance requirements.
