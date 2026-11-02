"""Download the appearance-search model once (about 600 MB).

    python scripts/download_search_model.py

Fetches CLIP ViT-B/32 (LAION-2B weights, open licence) from Hugging Face into
models/clip/. After that, search runs fully offline: the server sets
HF_HUB_OFFLINE and never contacts the hub again.
"""
import os
import sys
import time
from pathlib import Path

# This is the one step that is allowed to go online.
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.live.appearance import CLIP_DIR, MODEL_NAME, PRETRAINED, model_available  # noqa: E402


def main() -> int:
    if model_available():
        print(f"Already downloaded: {CLIP_DIR}")
        return 0
    try:
        import open_clip
    except ImportError:
        print("open_clip is not installed. Install it without touching torch:\n"
              "  pip install open_clip_torch -c <a file pinning torch==2.3.1+cpu>")
        return 1
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(f"Downloading {MODEL_NAME} ({PRETRAINED}) to {CLIP_DIR} ...")
    open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED, cache_dir=str(CLIP_DIR))
    print(f"Done in {time.time() - started:.0f} s. Restart Drishti to turn appearance search on.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
