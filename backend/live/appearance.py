"""Appearance search: find people and vehicles by what they look like.

"Man in a red jacket", "white pickup truck", "person with a backpack". Each
person or vehicle a live camera tracks leaves one picture (its clearest crop),
which an open image-text model (CLIP ViT-B/32, run on the CPU) turns into a
vector. A text query becomes a vector the same way, and results are ranked by
how close the two are.

This is a search, not a detector: it ranks pictures for a person to look at,
and every result shows the crop it matched so the operator judges it. It never
raises an alarm.

The model is downloaded once with scripts/download_search_model.py. Without
it, search reports itself as unavailable and everything else runs as before.
"""
from __future__ import annotations

import logging
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from backend.config import DATA_DIR, MODELS_DIR

logger = logging.getLogger("backend.live.appearance")

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
CLIP_DIR = MODELS_DIR / "clip"

PERSON_CLASS = 0
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS sightings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT, camera_name TEXT, track_id INTEGER,
    kind TEXT, wall_time REAL, crop BLOB, embedding BLOB
);
CREATE INDEX IF NOT EXISTS sightings_time ON sightings(wall_time);
"""


def model_available() -> bool:
    return any(CLIP_DIR.rglob("*.safetensors")) or any(CLIP_DIR.rglob("open_clip_pytorch_model.bin"))


class ClipModel:
    """Loaded on first use, on its own thread, with few CPU threads so live
    analysis keeps priority."""

    def __init__(self, threads: int = 2):
        self.threads = threads
        self._model = self._preprocess = self._tokenizer = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._model is not None:
            return
        import open_clip
        import torch
        torch.set_num_threads(self.threads)
        model, _, preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED, cache_dir=str(CLIP_DIR))
        model.eval()
        self._model, self._preprocess = model, preprocess
        self._tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    def images(self, crops: list[np.ndarray]) -> np.ndarray:
        import torch
        from PIL import Image
        with self._lock:
            self._load()
            batch = torch.stack([self._preprocess(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in crops])
            with torch.no_grad():
                feats = self._model.encode_image(batch)
        return _normalise(feats.numpy())

    def text(self, phrases: list[str]) -> np.ndarray:
        import torch
        with self._lock:
            self._load()
            with torch.no_grad():
                feats = self._model.encode_text(self._tokenizer(phrases))
        return _normalise(feats.numpy())


def _normalise(x: np.ndarray) -> np.ndarray:
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


@dataclass
class _Pending:
    camera_id: str
    camera_name: str
    track_id: int
    kind: str
    first_seen: float
    last_seen: float
    area: int = 0
    crop: np.ndarray | None = None


class AppearanceIndex:
    # A crop smaller than this (pixels tall) is too blurry to describe.
    MIN_HEIGHT = 48
    # A track is filed once it has been out of view this long, or once it has
    # been watched this long (someone standing still all day still appears).
    IDLE_SEC = 2.0
    MAX_WATCH_SEC = 10.0
    # Pictures this similar to a better match are treated as the same sighting.
    DUPLICATE_SIM = 0.93
    KEEP_DAYS = 7
    MAX_ROWS = 50_000

    def __init__(self, path: Path | None = None, model=None, clock=time.time):
        self.path = path or DATA_DIR / "appearance.db"
        self.model = model or ClipModel()
        self.clock = clock
        self._pending: dict[tuple[str, int], _Pending] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[list[_Pending] | None] = queue.Queue(maxsize=200)
        self._worker: threading.Thread | None = None
        self._init_db()

    # --- storage ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    # --- intake (camera threads) ----------------------------------------------------

    def observe(self, session, image: np.ndarray, tracks) -> None:
        """Keep the clearest crop of each tracked person or vehicle."""
        now = self.clock()
        camera_id, camera_name = session.camera.camera_id, session.camera.name
        h, w = image.shape[:2]
        with self._lock:
            for track_id, cls, (x1, y1, x2, y2) in tracks:
                kind = "person" if cls == PERSON_CLASS else VEHICLE_CLASSES.get(cls)
                if kind is None:
                    continue
                key = (camera_id, track_id)
                item = self._pending.get(key)
                if item is None:
                    item = self._pending[key] = _Pending(camera_id, camera_name, track_id, kind, now, now)
                item.last_seen = now
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                area = (x2 - x1) * (y2 - y1)
                # A body cut off by the frame edge describes badly; prefer a
                # smaller whole view of the same person.
                if x1 <= 2 or y1 <= 2 or x2 >= w - 2 or y2 >= h - 2:
                    area //= 2
                if y2 - y1 >= self.MIN_HEIGHT and area > item.area:
                    item.area = area
                    item.crop = image[y1:y2, x1:x2].copy()
            self._file_finished(now, camera_id)

    def _file_finished(self, now: float, camera_id: str) -> None:
        done = [k for k, p in self._pending.items() if p.camera_id == camera_id and (
            now - p.last_seen >= self.IDLE_SEC or now - p.first_seen >= self.MAX_WATCH_SEC)]
        batch = [self._pending.pop(k) for k in done]
        batch = [p for p in batch if p.crop is not None]
        if not batch:
            return
        self._ensure_worker()
        try:
            self._queue.put_nowait(batch)
        except queue.Full:
            # Falling behind: drop pictures rather than slow the cameras.
            logger.warning("appearance index busy; %d crops skipped", len(batch))

    # --- embedding (worker thread) ------------------------------------------------

    def _ensure_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._work, name="appearance-index", daemon=True)
            self._worker.start()

    def _work(self) -> None:
        while True:
            batch = self._queue.get()
            if batch is None:
                return
            try:
                self.add(batch)
            except Exception:
                logger.exception("appearance indexing failed")

    def add(self, batch: list[_Pending]) -> int:
        vectors = self.model.images([p.crop for p in batch])
        rows = []
        for item, vec in zip(batch, vectors):
            ok, jpeg = cv2.imencode(".jpg", item.crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                rows.append((item.camera_id, item.camera_name, item.track_id, item.kind,
                             item.first_seen, jpeg.tobytes(), vec.astype(np.float16).tobytes()))
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO sightings (camera_id, camera_name, track_id, kind, wall_time, crop, embedding)"
                " VALUES (?,?,?,?,?,?,?)", rows)
            conn.execute("DELETE FROM sightings WHERE wall_time < ?", (self.clock() - self.KEEP_DAYS * 86400,))
            conn.execute("DELETE FROM sightings WHERE id <= (SELECT MAX(id) FROM sightings) - ?", (self.MAX_ROWS,))
        return len(rows)

    def stop(self) -> None:
        if self._worker is not None:
            self._queue.put(None)

    # --- search -------------------------------------------------------------------

    def search(self, query: str, kind: str | None = None, camera: str | None = None,
               since: float | None = None, limit: int = 24) -> list[dict]:
        sql = "SELECT id, camera_id, camera_name, track_id, kind, wall_time, embedding FROM sightings WHERE 1=1"
        args: list = []
        if kind == "person":
            sql += " AND kind = 'person'"
        elif kind == "vehicle":
            sql += " AND kind != 'person'"
        if camera:
            sql += " AND camera_name = ?"
            args.append(camera)
        if since is not None:
            sql += " AND wall_time >= ?"
            args.append(since)
        with self._connect() as conn:
            rows = conn.execute(sql, args).fetchall()
        if not rows:
            return []
        # Two phrasings averaged: CLIP was trained on captions, so "a photo
        # of ..." matches pictures more reliably than a bare phrase.
        q = self.model.text([query, f"a photo of {query}"]).mean(axis=0)
        q /= np.linalg.norm(q) + 1e-8
        matrix = np.stack([np.frombuffer(r["embedding"], dtype=np.float16).astype(np.float32) for r in rows])
        scores = matrix @ q
        # The same person filed twice (a track broken and picked up again)
        # would fill the results with one face, so near-identical pictures
        # are shown once, best-scoring first.
        order: list[int] = []
        for i in np.argsort(-scores):
            if all(float(matrix[i] @ matrix[j]) < self.DUPLICATE_SIM for j in order):
                order.append(int(i))
                if len(order) == limit:
                    break
        return [{
            "id": rows[i]["id"], "camera_id": rows[i]["camera_id"], "camera_name": rows[i]["camera_name"],
            "track_id": rows[i]["track_id"], "kind": rows[i]["kind"], "wall_time": rows[i]["wall_time"],
            "score": round(float(scores[i]), 3),
        } for i in order]

    def crop(self, sighting_id: int) -> bytes | None:
        with self._connect() as conn:
            row = conn.execute("SELECT crop FROM sightings WHERE id = ?", (sighting_id,)).fetchone()
        return row["crop"] if row else None

    def stats(self) -> dict:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n, MIN(wall_time) AS first FROM sightings").fetchone()
        return {"sightings": row["n"], "since": row["first"]}
