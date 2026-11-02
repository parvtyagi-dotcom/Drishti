"""Download the ArcFace face-recognition model used for Target Person ID.

    python scripts/download_face_model.py

Fetches InsightFace's official buffalo_l pack from its GitHub release and
keeps only the recognizer (w600k_r50.onnx, 174 MB) as
models/arcface_w600k_r50.onnx. Without it, person ID falls back to SFace.

InsightFace's pretrained models are licensed for non-commercial research use.
"""
from __future__ import annotations

import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from backend.config import ARCFACE_MODEL_PATH  # noqa: E402

URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
MEMBER = "w600k_r50.onnx"


def main() -> int:
    if ARCFACE_MODEL_PATH.exists():
        print(f"already present: {ARCFACE_MODEL_PATH}")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "buffalo_l.zip"
        print(f"downloading {URL} (about 280 MB)...")
        urllib.request.urlretrieve(URL, archive)
        with zipfile.ZipFile(archive) as zf:
            name = next(n for n in zf.namelist() if n.endswith(MEMBER))
            ARCFACE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            ARCFACE_MODEL_PATH.write_bytes(zf.read(name))
    print(f"saved {ARCFACE_MODEL_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
