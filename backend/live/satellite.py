"""Satellite check: what changed on the ground between two Sentinel-2 passes.

Cameras watch the fence; a 10 m satellite pixel sees the strip behind it. Two
images of the same box a week or a month apart show new tracks, cleared
scrub and new structures that no ground camera covers. Plain numpy + OpenCV:

1. Align the after image to the before image (phase correlation, sub-pixel).
2. Match the after image's colour to the before image so haze and sun angle
   do not count as change.
3. Where red and near-infrared bands exist, compare the vegetation index
   (NDVI); always compare colour as well.
4. Threshold, clean up, and turn each changed blob into a box, an outline,
   an area in square metres and a short reason.

Pairs live in samples/satellite/<id>/ (written by scripts/fetch_sentinel.py):
meta.json, before_rgb.png, after_rgb.png and, optionally, 16-bit
before_red/before_nir/after_red/after_nir PNGs holding reflectance x 10000.
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from backend.config import REPO_ROOT

PAIRS_DIR = REPO_ROOT / "samples" / "satellite"

NDVI_DROP = 0.20        # NDVI fall that counts as vegetation cleared
NDVI_RISE = 0.25        # NDVI rise that counts as new growth
COLOUR_MIN = 22.0       # 8-bit colour distance that is always ignored
COLOUR_SIGMAS = 5.0     # ...and how far above the scene's typical change
MIN_PIXELS = 6          # 600 m² at 10 m pixels
MAX_CHANGES = 30


# --- detection -----------------------------------------------------------------

def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) if img.ndim == 3 else img.astype(np.float32)


def align(before: np.ndarray, after: np.ndarray, *extra: np.ndarray | None):
    """Shift `after` (and any `extra` after-bands) onto `before`. Returns (shift, after, extra...)."""
    a, b = _gray(before), _gray(after)
    win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(a, b, win)
    h, w = a.shape
    if abs(dx) > w / 4 or abs(dy) > h / 4:
        dx = dy = 0.0  # not the same place, or no structure to lock on; trust the georeferencing
    out = [after, *extra]
    if abs(dx) > 0.25 or abs(dy) > 0.25:
        m = np.float32([[1, 0, -dx], [0, 1, -dy]])
        out = [None if x is None else cv2.warpAffine(x, m, (w, h), flags=cv2.INTER_LINEAR,
                                                    borderMode=cv2.BORDER_REPLICATE) for x in out]
    return (round(float(dx), 2), round(float(dy), 2)), *out


def _match_colour(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    s, r = src.astype(np.float32), ref.astype(np.float32)
    axes = (0, 1) if s.ndim == 3 else None
    out = (s - s.mean(axis=axes)) / (s.std(axis=axes) + 1e-6) * r.std(axis=axes) + r.mean(axis=axes)
    return np.clip(out, 0, 255)


def _ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    red, nir = red.astype(np.float32), nir.astype(np.float32)
    return (nir - red) / np.maximum(nir + red, 1.0)


def _about(m2: float) -> str:
    step = 100 if m2 < 10000 else 1000
    return f"{max(step, round(m2 / step) * step):,.0f} m²"


def detect_changes(before: np.ndarray, after: np.ndarray, *,
                   before_red: np.ndarray | None = None, before_nir: np.ndarray | None = None,
                   after_red: np.ndarray | None = None, after_nir: np.ndarray | None = None,
                   pixel_m: float = 10.0, max_changes: int = MAX_CHANGES) -> dict:
    """Find changed regions between two images of the same area (BGR uint8, same size)."""
    if before.shape[:2] != after.shape[:2]:
        after = cv2.resize(after, before.shape[1::-1], interpolation=cv2.INTER_LINEAR)
    use_ndvi = all(x is not None for x in (before_red, before_nir, after_red, after_nir))
    shift, after, after_red, after_nir = align(before, after, after_red if use_ndvi else None,
                                               after_nir if use_ndvi else None)

    b = before.astype(np.float32)
    a = _match_colour(after, before)
    diff = a - b
    dist = np.sqrt((diff ** 2).sum(axis=2)) if diff.ndim == 3 else np.abs(diff)
    med = float(np.median(dist))
    mad = float(np.median(np.abs(dist - med))) * 1.4826
    colour_thr = max(COLOUR_MIN, med + COLOUR_SIGMAS * mad)
    brighter = _gray(a) - _gray(b)

    mask = dist > colour_thr
    d_ndvi = None
    if use_ndvi:
        nb, na = _ndvi(before_red, before_nir), _ndvi(after_red, after_nir)
        d_ndvi = na - nb
        mask |= (d_ndvi < -NDVI_DROP) & (nb > 0.3)
        mask |= d_ndvi > NDVI_RISE

    m = mask.astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)

    px_m2 = pixel_m * pixel_m
    changes = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < MIN_PIXELS:
            continue
        blob = labels == i
        kind, label = _classify(blob, d_ndvi, brighter)
        contours, _ = cv2.findContours(blob.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        poly = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 1.0, True).reshape(-1, 2)
        area_m2 = area * px_m2
        changes.append({
            "kind": kind, "reason": f"{label}, about {_about(area_m2)}",
            "area_m2": round(area_m2), "box": [x, y, w, h],
            "polygon": poly.astype(int).tolist(),
            "strength": round(float(dist[blob].mean() / colour_thr), 2),
        })

    changes.sort(key=lambda c: c["area_m2"], reverse=True)
    total = sum(c["area_m2"] for c in changes)
    for k, c in enumerate(changes, 1):
        c["id"] = k
    return {
        "width": int(before.shape[1]), "height": int(before.shape[0]), "pixel_m": pixel_m,
        "method": "ndvi+colour" if use_ndvi else "colour",
        "shift_px": list(shift), "colour_threshold": round(colour_thr, 1),
        "changed_area_m2": round(total),
        "changed_fraction": round(total / (before.shape[0] * before.shape[1] * px_m2), 4),
        "total_changes": len(changes), "changes": changes[:max_changes],
    }


def _classify(blob: np.ndarray, d_ndvi: np.ndarray | None, brighter: np.ndarray) -> tuple[str, str]:
    if d_ndvi is not None:
        dv = float(d_ndvi[blob].mean())
        if dv < -NDVI_DROP * 0.75:
            return "vegetation_cleared", "Vegetation cleared"
        if dv > NDVI_RISE * 0.75:
            return "vegetation_growth", "New vegetation or crop growth"
    ys, xs = np.nonzero(blob)
    (_, _), (rw, rh), _ = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
    long_, short = max(rw, rh), max(1.0, min(rw, rh))
    if long_ / short >= 4 and blob.sum() / (long_ * short) < 0.8 or long_ / short >= 6:
        return "track", "New path or track"
    if float(brighter[blob].mean()) > 0:
        return "bare_ground", "Ground cleared or new structure (brighter)"
    return "structure", "New structure or dark surface (darker), e.g. roofs, panels, water"


# --- saved pairs ---------------------------------------------------------------

_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_cache: dict[str, dict] = {}
_lock = threading.Lock()


def _pair_dir(pair_id: str) -> Path | None:
    if not _ID.match(pair_id):
        return None
    d = PAIRS_DIR / pair_id
    return d if (d / "meta.json").exists() else None


def list_pairs() -> list[dict]:
    out = []
    if PAIRS_DIR.exists():
        for d in sorted(PAIRS_DIR.iterdir()):
            if (d / "meta.json").exists() and _ID.match(d.name):
                meta = json.loads((d / "meta.json").read_text())
                meta["id"] = d.name
                meta["has_ndvi"] = (d / "before_nir.png").exists() and (d / "after_nir.png").exists()
                meta["has_map"] = all((d / f"{m}.png").exists() for m in MAP_IMAGES)
                meta["checked"] = d.name in _cache
                out.append(meta)
    return sorted(out, key=lambda m: m.get("after_date") or "", reverse=True)  # newest first


# Location maps saved beside a pair by scripts/fetch_sentinel.py: where the box
# sits in the wider region, and the box itself on a street map.
MAP_IMAGES = ("map_region", "map_local")


def image_path(pair_id: str, which: str) -> Path | None:
    d = _pair_dir(pair_id)
    if d is None:
        return None
    if which in ("before", "after"):
        p = d / f"{which}_rgb.png"
    elif which in MAP_IMAGES:
        p = d / f"{which}.png"
    else:
        return None
    return p if p.exists() else None


def run_check(pair_id: str, refresh: bool = False) -> dict | None:
    d = _pair_dir(pair_id)
    if d is None:
        return None
    with _lock:
        if not refresh and pair_id in _cache:
            return _cache[pair_id]
        meta = json.loads((d / "meta.json").read_text())

        def band(name: str) -> np.ndarray | None:
            p = d / f"{name}.png"
            return cv2.imread(str(p), cv2.IMREAD_UNCHANGED) if p.exists() else None

        t0 = time.perf_counter()
        result = detect_changes(
            cv2.imread(str(d / "before_rgb.png")), cv2.imread(str(d / "after_rgb.png")),
            before_red=band("before_red"), before_nir=band("before_nir"),
            after_red=band("after_red"), after_nir=band("after_nir"),
            pixel_m=float(meta.get("pixel_m", 10)),
        )
        result.update(pair_id=pair_id, checked_at=time.time(),
                      seconds=round(time.perf_counter() - t0, 2),
                      before_date=meta.get("before_date"), after_date=meta.get("after_date"))
        _cache[pair_id] = result
        return result
