"""Evidence capture for live alerts: snapshot, surrounding clip, metadata.

An alert without evidence is an assertion. For a security record to be worth
anything afterwards - in a review, a handover, or an inquiry - someone has to
be able to see what the system saw when it decided.

The hard part of live evidence is that the interesting moment has already
happened by the time the rule fires: a zone intrusion is only knowable once
someone is inside. So frames are kept in a short rolling buffer and the clip
is assembled from *before* the event plus a tail recorded after it, giving the
approach as well as the act.

Frames are buffered as encoded JPEGs, not raw arrays. The session already
encodes every frame for the live view, so buffering costs nothing extra and
uses roughly 10MB per camera instead of the ~190MB the same window of raw
960x540 frames would take.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from backend.core.video import VideoInfo

logger = logging.getLogger("backend.live.evidence")


@dataclass
class EvidenceConfig:
    # Seconds of footage kept before an event. Long enough to show how a
    # subject approached, short enough to stay cheap.
    pre_event_sec: float = 8.0
    # Seconds recorded after the event before the clip is assembled.
    post_event_sec: float = 6.0
    clip_fps: float = 10.0
    jpeg_quality: int = 80
    # Cap on clips held per camera on disk; oldest are removed beyond this so
    # a long deployment cannot silently fill the disk.
    max_clips_per_camera: int = 50


@dataclass
class _PendingClip:
    event: dict
    directory: Path
    assemble_after: float  # time.monotonic() deadline


class EvidenceRecorder:
    """Buffers recent frames and writes evidence bundles for alerts."""

    def __init__(self, camera_id: str, root: Path, config: EvidenceConfig | None = None):
        self.camera_id = camera_id
        self.config = config or EvidenceConfig()
        self.root = Path(root) / camera_id
        self._buffer: deque[tuple[float, bytes]] = deque()
        self._lock = threading.Lock()
        self._pending: queue.Queue[_PendingClip] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()

    # --- frame intake --------------------------------------------------------

    def add_frame(self, jpeg: bytes) -> None:
        """Record one encoded frame. Called on the capture thread, so it does
        no work beyond appending and trimming."""
        now = time.monotonic()
        horizon = now - (self.config.pre_event_sec + self.config.post_event_sec + 2.0)
        with self._lock:
            self._buffer.append((now, jpeg))
            while self._buffer and self._buffer[0][0] < horizon:
                self._buffer.popleft()

    def frames_since(self, since: float) -> list[tuple[float, bytes]]:
        """Buffered frames recorded at or after `since` (time.monotonic()),
        oldest first. Lets a sensor alarm look at what the camera saw just
        before and after it fired."""
        with self._lock:
            return [(t, jpeg) for t, jpeg in self._buffer if t >= since]

    # --- capture -------------------------------------------------------------

    def capture(self, event: dict) -> str | None:
        """Start an evidence bundle for `event`.

        The snapshot is written immediately, because the frame that triggered
        the alert must not be lost if the clip fails. The clip is assembled
        later, once the post-event tail exists.
        """
        event_id = event.get("event_id") or f"{int(time.time() * 1000)}"
        directory = self.root / f"{time.strftime('%Y%m%d-%H%M%S')}_{event.get('type', 'EVENT')}_{event_id}"
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.exception("could not create evidence directory for %s", self.camera_id)
            return None

        with self._lock:
            latest = self._buffer[-1][1] if self._buffer else None
        if latest is not None:
            (directory / "snapshot.jpg").write_bytes(latest)

        (directory / "event.json").write_text(json.dumps(event, indent=2, default=str), encoding="utf-8")

        self._ensure_worker()
        self._pending.put(_PendingClip(
            event=event,
            directory=directory,
            assemble_after=time.monotonic() + self.config.post_event_sec,
        ))
        return directory.name

    # --- clip assembly (worker thread) ---------------------------------------

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(
            target=self._worker_loop, name=f"evidence-{self.camera_id}", daemon=True,
        )
        self._worker.start()

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                pending = self._pending.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                # Wait for the tail to accumulate. Assembly happens here, off
                # the capture thread: decoding a few hundred JPEGs and encoding
                # an MP4 would otherwise stall live analysis exactly when
                # something is happening.
                while not self._stop.is_set() and time.monotonic() < pending.assemble_after:
                    time.sleep(0.2)
                self._write_clip(pending)
            except Exception:
                logger.exception("evidence clip failed for %s", self.camera_id)
            finally:
                self._pending.task_done()

    def _write_clip(self, pending: _PendingClip) -> None:
        window_start = pending.assemble_after - self.config.post_event_sec - self.config.pre_event_sec
        with self._lock:
            frames = [jpeg for ts, jpeg in self._buffer if ts >= window_start]
        if not frames:
            # Nothing buffered for this window. That is the normal case when
            # the alert *is* the camera going down: there is no footage of a
            # feed that had already stopped. Saying so explicitly is what lets
            # the UI report "no footage" instead of leaving the bundle waiting
            # on a clip that can never arrive.
            self._write_manifest(pending, clip=None,
                                 note="no footage buffered - the feed had already stopped")
            self._prune()
            return

        first = cv2.imdecode(np.frombuffer(frames[0], np.uint8), cv2.IMREAD_COLOR)
        if first is None:
            return
        height, width = first.shape[:2]

        # Reuse the shared writer so evidence clips get the same codec
        # fallback - and the same browser-playability warning - as every other
        # video this system produces.
        from backend.core.output import OutputVideoWriter

        info = VideoInfo(
            path=pending.directory / "clip.mp4", fps=self.config.clip_fps,
            width=width, height=height, frame_count=len(frames),
            duration_sec=len(frames) / self.config.clip_fps,
        )
        writer = OutputVideoWriter(pending.directory / "clip.mp4", info)
        try:
            for jpeg in frames:
                image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    continue
                if image.shape[:2] != (height, width):
                    image = cv2.resize(image, (width, height))
                writer.write(image)
        finally:
            writer.close()

        self._write_manifest(pending, clip={
            "frames": writer.frames_written,
            "fps": self.config.clip_fps,
            "pre_event_sec": self.config.pre_event_sec,
            "post_event_sec": self.config.post_event_sec,
            "codec": writer.open_result.codec_used,
            "browser_playable": writer.open_result.browser_playable,
        })
        self._prune()

    def _write_manifest(self, pending: _PendingClip, clip: dict | None,
                        note: str | None = None) -> None:
        """Record the outcome of assembling this bundle's clip.

        Written whether or not a clip was produced: the manifest existing is
        how a consumer tells "assembly finished, there was nothing to record"
        apart from "assembly has not run yet".
        """
        manifest = {"camera_id": self.camera_id, "event": pending.event, "clip": clip}
        if note:
            manifest["note"] = note
        try:
            (pending.directory / "clip.json").write_text(
                json.dumps(manifest, indent=2, default=str), encoding="utf-8",
            )
        except OSError:
            logger.exception("could not write evidence manifest for %s", self.camera_id)

    def _prune(self) -> None:
        """Keep only the most recent bundles for this camera."""
        try:
            bundles = sorted(
                (d for d in self.root.iterdir() if d.is_dir()),
                key=lambda d: d.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            return
        import shutil

        for stale in bundles[self.config.max_clips_per_camera:]:
            shutil.rmtree(stale, ignore_errors=True)

    def stop(self) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=3.0)
            self._worker = None
