"""Drishti backend.

Set before any submodule imports Ultralytics: by default that library
probes the internet and sends anonymous usage statistics to Google
Analytics whenever YOLO loads. scripts/network_check.py caught it doing so
from the server. Nothing sensitive goes in those statistics, but a border
post's surveillance system should open no connection its operator did not
ask for, so it is switched off here, for every way Drishti is started.
"""
import os

os.environ.setdefault("YOLO_OFFLINE", "True")
# The same for the Hugging Face hub behind appearance search: the model is
# downloaded once by scripts/download_search_model.py, and after that the
# server must never reach out to check for updates.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# pyarrow (pulled in by pandas, which pytesseract imports) ships its own old
# MSVCP140.dll (14.28). Windows keeps whichever copy of a DLL loads first, and
# onnxruntime 1.29 cannot initialise on that one ("DLL initialization routine
# failed") - so a process that touched pandas before onnxruntime lost the
# ReID, plate detector and OCR models. Loading the system runtime first, which
# is newer and backward-compatible, makes import order not matter. Here and
# not in config.py, because some modules (anpr/ocr.py) never import config.
if os.name == "nt":
    try:
        import ctypes

        ctypes.WinDLL(os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "msvcp140.dll"))
    except OSError:
        pass
