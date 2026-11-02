"""Fetch a before/after pair of Sentinel-2 images for the Satellite Check page.

Needs internet. No account: it searches Element84's free Earth Search
catalogue and reads only the small window it needs out of each
Cloud-Optimised GeoTIFF on AWS, with HTTP range requests, so a few square
kilometres cost a few MB. Used by scripts/fetch_sentinel.py at sector HQ and by
the page's place search (api/satellite_routes.py) when the machine is online.

fetch_pair() writes samples/satellite/<id>/ with meta.json, before_rgb.png,
after_rgb.png, 16-bit before_red/before_nir/after_red/after_nir PNGs
(reflectance x 10000) and map_region/map_local.png street maps showing where
the box is. Plain numpy + OpenCV + requests: no GDAL or rasterio needed.
"""
from __future__ import annotations

import json
import math
import struct
import zlib
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import requests

from backend.live.satellite import PAIRS_DIR as OUT_ROOT
STAC = "https://earth-search.aws.element84.com/v1/search"
COLLECTIONS = ("sentinel-2-l2a", "sentinel-2-c1-l2a")
ASK_ABOVE_MB = 300
# Scene classification: cloud shadow, medium/high cloud, cirrus, saturated.
CLOUD_CLASSES = (1, 3, 8, 9, 10)


# --- lon/lat -> UTM (Snyder, WGS84) -------------------------------------------

def to_utm(lon: float, lat: float, zone: int, south: bool) -> tuple[float, float]:
    a, f, k0 = 6378137.0, 1 / 298.257223563, 0.9996
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    phi, lam = math.radians(lat), math.radians(lon)
    lam0 = math.radians((zone - 1) * 6 - 180 + 3)
    n = a / math.sqrt(1 - e2 * math.sin(phi) ** 2)
    t, c = math.tan(phi) ** 2, ep2 * math.cos(phi) ** 2
    A = math.cos(phi) * (lam - lam0)
    m = a * ((1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * phi
             - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * math.sin(2 * phi)
             + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * math.sin(4 * phi)
             - (35 * e2**3 / 3072) * math.sin(6 * phi))
    x = k0 * n * (A + (1 - t + c) * A**3 / 6 + (5 - 18 * t + t * t + 72 * c - 58 * ep2) * A**5 / 120) + 500000
    y = k0 * (m + n * math.tan(phi) * (A * A / 2 + (5 - t + 9 * c + 4 * c * c) * A**4 / 24
                                        + (61 - 58 * t + t * t + 600 * c - 330 * ep2) * A**6 / 720))
    return x, (y + 10000000 if south else y)


# --- minimal Cloud-Optimised GeoTIFF window reader -----------------------------

class COG:
    """Reads a window of band 1 at full resolution from a tiled, DEFLATE COG."""

    TYPES = {1: "B", 2: "s", 3: "H", 4: "I", 5: "II", 12: "d", 16: "Q"}

    def __init__(self, url: str):
        self.url = url
        self.session = requests.Session()
        self.bytes_read = 0
        self.head = self._get(0, 65536)
        if self.head[:2] != b"II":
            raise ValueError("only little-endian TIFF is supported")
        self.big = struct.unpack_from("<H", self.head, 2)[0] == 43
        first = struct.unpack_from("<Q" if self.big else "<I", self.head, 8 if self.big else 4)[0]
        self.tags = self._ifd(first)
        t = self.tags
        self.width, self.height = t[256][0], t[257][0]
        self.tile_w, self.tile_h = t[322][0], t[323][0]
        self.offsets, self.counts = t[324], t[325]
        self.compression = t.get(259, [1])[0]
        self.predictor = t.get(317, [1])[0]
        bits = t.get(258, [8])[0]
        self.dtype = np.dtype({8: "u1", 16: "<u2"}[bits])
        self.scale = t[33550][:2]
        self.tie = t[33922][3:5]
        if self.compression not in (1, 8, 32946):
            raise ValueError(f"TIFF compression {self.compression} is not supported (need DEFLATE)")

    def _get(self, start: int, length: int) -> bytes:
        r = self.session.get(self.url, headers={"Range": f"bytes={start}-{start + length - 1}"}, timeout=60)
        r.raise_for_status()
        self.bytes_read += len(r.content)
        return r.content

    def _bytes(self, start: int, length: int) -> bytes:
        if start + length <= len(self.head):
            return self.head[start:start + length]
        return self._get(start, length)

    def _ifd(self, offset: int) -> dict[int, list]:
        cfmt, csize, esize = ("<Q", 8, 20) if self.big else ("<H", 2, 12)
        n = struct.unpack(cfmt, self._bytes(offset, csize))[0]
        raw = self._bytes(offset + csize, n * esize)
        tags: dict[int, list] = {}
        for i in range(n):
            e = raw[i * esize:(i + 1) * esize]
            if self.big:
                tag, typ, count = struct.unpack_from("<HHQ", e)
                inline, vo = e[12:20], 8
            else:
                tag, typ, count = struct.unpack_from("<HHI", e)
                inline, vo = e[8:12], 4
            code = self.TYPES.get(typ)
            if code is None:
                continue
            fmt = "<" + code * count if typ != 2 else f"<{count}s"
            size = struct.calcsize(fmt)
            data = inline[:size] if size <= vo else self._bytes(
                struct.unpack("<Q" if self.big else "<I", inline)[0], size)
            vals = list(struct.unpack(fmt, data))
            if typ == 5:
                vals = [vals[k] / vals[k + 1] for k in range(0, len(vals), 2)]
            tags[tag] = vals
        return tags

    def pixel(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.tie[0]) / self.scale[0], (self.tie[1] - y) / self.scale[1]

    def window(self, x0: int, y0: int, w: int, h: int) -> np.ndarray:
        out = np.zeros((h, w), self.dtype)
        across = math.ceil(self.width / self.tile_w)
        for ty in range(y0 // self.tile_h, (y0 + h - 1) // self.tile_h + 1):
            for tx in range(x0 // self.tile_w, (x0 + w - 1) // self.tile_w + 1):
                if not (0 <= tx < across and 0 <= ty * self.tile_h < self.height):
                    continue
                k = ty * across + tx
                if self.counts[k] == 0:
                    continue
                raw = self._get(self.offsets[k], self.counts[k])
                if self.compression != 1:
                    raw = zlib.decompress(raw)
                tile = np.frombuffer(raw, self.dtype)[: self.tile_w * self.tile_h].reshape(self.tile_h, self.tile_w)
                if self.predictor == 2:
                    tile = np.cumsum(tile, axis=1, dtype=self.dtype)
                ox, oy = tx * self.tile_w, ty * self.tile_h
                ax0, ay0 = max(x0, ox), max(y0, oy)
                ax1, ay1 = min(x0 + w, ox + self.tile_w), min(y0 + h, oy + self.tile_h)
                if ax1 > ax0 and ay1 > ay0:
                    out[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] = tile[ay0 - oy:ay1 - oy, ax0 - ox:ax1 - ox]
        return out


# --- catalogue search ----------------------------------------------------------

def epsg_of(item: dict) -> int:
    p = item["properties"]
    if "proj:epsg" in p:
        return int(p["proj:epsg"])
    return int(str(p.get("proj:code", "")).split(":")[-1])


def search(bbox: list[float], dates: str, max_cloud: float) -> list[dict]:
    for collection in COLLECTIONS:
        r = requests.post(STAC, json={
            "collections": [collection], "bbox": bbox, "datetime": _datetime(dates), "limit": 50,
            "query": {"eo:cloud_cover": {"lt": max_cloud}},
        }, timeout=60)
        r.raise_for_status()
        items = r.json().get("features", [])
        if items:
            return sorted(items, key=lambda i: i["properties"].get("eo:cloud_cover", 100))
    return []


def _datetime(dates: str) -> str:
    start, _, end = dates.partition("/")
    return f"{start}T00:00:00Z/{(end or start)}T23:59:59Z"


def band_scale(asset: dict) -> tuple[float, float]:
    rb = (asset.get("raster:bands") or [{}])[0]
    return float(rb.get("scale", 0.0001)), float(rb.get("offset", 0.0))


def read_scene(item: dict, bbox: list[float], need_epsg: int | None) -> dict | None:
    """Read red, green, blue, nir and the cloud mask for the box, or None if cloudy."""
    epsg = epsg_of(item)
    if need_epsg and epsg != need_epsg:
        return None
    zone, south = epsg % 100, epsg // 100 == 327
    corners = [to_utm(lon, lat, zone, south) for lon in (bbox[0], bbox[2]) for lat in (bbox[1], bbox[3])]
    xs, ys = [c[0] for c in corners], [c[1] for c in corners]
    # Snap to the 10 m grid so both dates share pixels.
    x0, x1 = math.floor(min(xs) / 10) * 10, math.ceil(max(xs) / 10) * 10
    y0, y1 = math.floor(min(ys) / 10) * 10, math.ceil(max(ys) / 10) * 10
    assets = item["assets"]

    scl = COG(assets["scl"]["href"])
    sx, sy = scl.pixel(x0, y1)
    sw, sh = math.ceil((x1 - x0) / scl.scale[0]), math.ceil((y1 - y0) / scl.scale[1])
    if sx < 0 or sy < 0 or sx + sw > scl.width or sy + sh > scl.height:
        return None
    mask = scl.window(int(sx), int(sy), sw, sh)
    if (mask == 0).mean() > 0.01:
        return None  # box crosses the edge of the swath
    cloudy = float(np.isin(mask, CLOUD_CLASSES).mean())
    if cloudy > 0.02:
        print(f"  {item['id']}: {cloudy:.0%} cloud over the box, skipping")
        return None

    bands, read = {}, scl.bytes_read
    for name in ("red", "green", "blue", "nir"):
        cog = COG(assets[name]["href"])
        px, py = cog.pixel(x0, y1)
        w, h = round((x1 - x0) / cog.scale[0]), round((y1 - y0) / cog.scale[1])
        dn = cog.window(int(round(px)), int(round(py)), w, h).astype(np.float32)
        scale, offset = band_scale(assets[name])
        if item["properties"].get("earthsearch:boa_offset_applied"):
            offset = 0.0  # Earth Search already removed the 2022+ BOA offset from the pixels
        bands[name] = np.clip((dn * scale + offset) * 10000, 0, 65535).astype(np.uint16)
        read += cog.bytes_read
    return {"bands": bands, "bytes": read, "utm": [x0, y0, x1, y1], "epsg": epsg, "cloud": cloudy}


def to_rgb(b: dict, gain: float) -> np.ndarray:
    rgb = np.dstack([b["blue"], b["green"], b["red"]]).astype(np.float32) / 10000
    return (np.clip(rgb * gain, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


# --- location maps -------------------------------------------------------------
# Two street-map pictures saved beside the pair, so the page can show where the
# box is without any internet at the post: the wider region with the spot
# marked, and the box itself. OpenStreetMap's tile policy asks for an
# identifying User-Agent and light use; two small maps per pair is that.

TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_HEADERS = {"User-Agent": "Drishti-border-surveillance/1.0 (satellite check location map)"}
MAP_W, MAP_H = 640, 400


def _tile_xy(lon: float, lat: float, z: int) -> tuple[float, float]:
    n = 2**z
    lat_r = math.radians(lat)
    return (lon + 180) / 360 * n, (1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n


def _street_map(bbox: list[float], z: int, session: requests.Session) -> np.ndarray:
    """MAP_W x MAP_H picture at zoom z, centred on the box, with the box drawn."""
    cx, cy = _tile_xy((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, z)
    left, top = cx * 256 - MAP_W / 2, cy * 256 - MAP_H / 2
    canvas = np.zeros((MAP_H, MAP_W, 3), np.uint8)
    for ty in range(int(top // 256), int((top + MAP_H) // 256) + 1):
        for tx in range(int(left // 256), int((left + MAP_W) // 256) + 1):
            resp = session.get(TILE_URL.format(z=z, x=tx % 2**z, y=ty), headers=TILE_HEADERS, timeout=30)
            resp.raise_for_status()
            tile = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
            ox, oy = int(tx * 256 - left), int(ty * 256 - top)
            x0, y0, x1, y1 = max(ox, 0), max(oy, 0), min(ox + 256, MAP_W), min(oy + 256, MAP_H)
            if x1 > x0 and y1 > y0:
                canvas[y0:y1, x0:x1] = tile[y0 - oy:y1 - oy, x0 - ox:x1 - ox]

    def px(lon: float, lat: float) -> tuple[int, int]:
        x, y = _tile_xy(lon, lat, z)
        return int(round(x * 256 - left)), int(round(y * 256 - top))

    (bx0, by0), (bx1, by1) = px(bbox[0], bbox[3]), px(bbox[2], bbox[1])
    red = (48, 59, 255)  # BGR for Apple's system red
    if bx1 - bx0 < 14:
        # The box is a few pixels wide at regional zoom: mark the spot instead.
        c = ((bx0 + bx1) // 2, (by0 + by1) // 2)
        cv2.circle(canvas, c, 16, red, 3, cv2.LINE_AA)
        cv2.circle(canvas, c, 5, red, -1, cv2.LINE_AA)
    else:
        overlay = canvas.copy()
        cv2.rectangle(overlay, (bx0, by0), (bx1, by1), red, -1)
        canvas = cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0)
        cv2.rectangle(canvas, (bx0, by0), (bx1, by1), red, 3, cv2.LINE_AA)
    label = "(c) OpenStreetMap contributors"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    cv2.rectangle(canvas, (MAP_W - tw - 8, MAP_H - th - 8), (MAP_W, MAP_H), (255, 255, 255), -1)
    cv2.putText(canvas, label, (MAP_W - tw - 4, MAP_H - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA)
    return canvas


def save_location_maps(bbox: list[float], out: Path, log: Callable[[str], None] = print) -> None:
    session = requests.Session()
    # Region: a few hundred km across. Local: the box filling about half the picture.
    w_deg = max(bbox[2] - bbox[0], 1e-4)
    local_z = max(8, min(16, int(math.log2(360 * MAP_W / 2 / 256 / w_deg))))
    for name, z in (("map_region", 7), ("map_local", local_z)):
        cv2.imwrite(str(out / f"{name}.png"), _street_map(bbox, z, session))
    log(f"  location maps saved (region zoom 7, local zoom {local_z})")


# --- the whole fetch -------------------------------------------------------------

class FetchError(Exception):
    """A fetch that cannot finish, with a message an operator can act on."""


def box_km(bbox: list[float]) -> tuple[float, float]:
    w = (bbox[2] - bbox[0]) * 111.32 * math.cos(math.radians((bbox[1] + bbox[3]) / 2))
    return w, (bbox[3] - bbox[1]) * 110.57


def estimate_mb(bbox: list[float]) -> float:
    # About four 1024x1024 DEFLATE tiles (~1.5 MB each) per band per 10 km, worst case.
    w_km, h_km = box_km(bbox)
    return 2 * 4 * 1.5 * max(1, math.ceil(w_km / 10.24) + 1) * max(1, math.ceil(h_km / 10.24) + 1)


def bbox_around(lat: float, lon: float, size_km: float) -> list[float]:
    """A size_km square centred on a point, as west,south,east,north."""
    half_lat = size_km / 2 / 110.57
    half_lon = size_km / 2 / (111.32 * math.cos(math.radians(lat)))
    return [round(lon - half_lon, 5), round(lat - half_lat, 5), round(lon + half_lon, 5), round(lat + half_lat, 5)]


def fetch_pair(bbox: list[float], before: str, after: str, name: str, pair_id: str, note: str = "",
               max_cloud: float = 30, log: Callable[[str], None] = print) -> Path:
    """Find a cloud-free scene in each date range, save the pair, return its folder."""
    scenes: dict[str, dict] = {}
    for label, dates in (("before", before), ("after", after)):
        items = search(bbox, dates, max_cloud)
        log(f"{label}: {len(items)} scenes in {dates}")
        need = scenes["before"]["epsg"] if "before" in scenes else None
        for item in items:
            got = read_scene(item, bbox, need)
            if got:
                got["item"] = item
                scenes[label] = got
                log(f"  using {item['id']} ({got['bytes'] / 1e6:.1f} MB read)")
                break
        else:
            raise FetchError(f"No cloud-free picture for the {label} dates ({dates}). Try dates a week or two away.")

    out = OUT_ROOT / pair_id
    out.mkdir(parents=True, exist_ok=True)
    gain = 1 / max(1e-3, float(np.percentile(np.dstack(list(scenes["before"]["bands"].values())[:3]), 99.5)) / 10000)
    for label, s in scenes.items():
        cv2.imwrite(str(out / f"{label}_rgb.png"), to_rgb(s["bands"], gain))
        cv2.imwrite(str(out / f"{label}_red.png"), s["bands"]["red"])
        cv2.imwrite(str(out / f"{label}_nir.png"), s["bands"]["nir"])
    meta = {
        "id": pair_id, "name": name, "note": note, "bbox": bbox, "epsg": scenes["before"]["epsg"],
        "utm": scenes["before"]["utm"], "pixel_m": 10,
        "source": "Copernicus Sentinel-2 L2A via Element84 Earth Search (AWS open data)",
        **{f"{k}_date": s["item"]["properties"]["datetime"][:10] for k, s in scenes.items()},
        **{f"{k}_scene": s["item"]["id"] for k, s in scenes.items()},
        **{f"{k}_cloud": round(s["cloud"], 4) for k, s in scenes.items()},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    try:
        save_location_maps(bbox, out, log)
    except requests.RequestException as exc:
        log(f"  location maps skipped ({exc}); add them later with --map-only {pair_id}")
    size = sum(p.stat().st_size for p in out.iterdir())
    log(f"Saved {out} ({size / 1e6:.2f} MB)")
    return out


# --- place search ------------------------------------------------------------------
# OpenStreetMap's Nominatim, the geocoder behind openstreetmap.org's own search
# box. Its policy asks for an identifying User-Agent and at most one request a
# second; the page only searches when the operator presses Enter.

NOMINATIM = "https://nominatim.openstreetmap.org/search"


def geocode(query: str, limit: int = 6) -> list[dict]:
    r = requests.get(NOMINATIM, params={"q": query, "format": "jsonv2", "limit": limit},
                     headers=TILE_HEADERS, timeout=20)
    r.raise_for_status()
    return [{"name": p["display_name"], "lat": float(p["lat"]), "lon": float(p["lon"]), "kind": p.get("type", "")}
            for p in r.json()]
