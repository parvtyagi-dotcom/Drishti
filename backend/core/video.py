"""Shared video engine: reading, frame sampling, timestamps, metrics.

Every module pipeline (person_id, anpr, zone, behavior, lowlight) reads
frames through VideoReader and writes the annotated result through
core.output.OutputVideoWriter, instead of touching cv2.VideoCapture /
cv2.VideoWriter directly. This keeps codec fallback, FPS/duration handling,
and processing metrics in one place.

Two readers live here, and the difference is finiteness, not file format:

  VideoReader      - a finite uploaded file. Iteration ends at the last
                     frame, frame_count/duration are known up front, and a
                     read failure is fatal. Used by Forensic Mode.
  LiveStreamReader - a continuous camera (RTSP) or a looping local file
                     standing in for one. Iteration never ends on its own,
                     duration is unknown, and a read failure is a
                     *reconnect*, not an error. Used by Live Mode.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    width: int
    height: int
    frame_count: int
    duration_sec: float

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)


@dataclass
class Frame:
    index: int
    timestamp_sec: float
    image: np.ndarray


@dataclass
class ProcessingMetrics:
    frames_read: int = 0
    frames_written: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def elapsed_sec(self) -> float:
        end = self.finished_at or time.monotonic()
        return max(0.0, end - self.started_at)

    @property
    def processing_fps(self) -> float:
        if self.elapsed_sec <= 0:
            return 0.0
        return self.frames_read / self.elapsed_sec

    def as_dict(self) -> dict:
        return {
            "frames_read": self.frames_read,
            "frames_written": self.frames_written,
            "elapsed_sec": round(self.elapsed_sec, 3),
            "processing_fps": round(self.processing_fps, 2),
        }


class VideoOpenError(RuntimeError):
    pass


class VideoReader:
    """Opens an uploaded video and yields every frame with an index/timestamp.

    Full videos are processed by default - there is no frame-count cap.
    Callers that only need periodic work (face recognition, OCR, ...) decide
    their own sampling cadence against Frame.index; VideoReader itself always
    yields every frame so tracking stays smooth and the output video can be
    written frame-for-frame.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise VideoOpenError(f"video file not found: {self.path}")

        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise VideoOpenError(f"OpenCV could not open video: {self.path}")

        fps = self._cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0.1:
            fps = 25.0  # sane fallback for malformed containers
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if frame_count > 0 else 0.0

        self.info = VideoInfo(
            path=self.path,
            fps=fps,
            width=width,
            height=height,
            frame_count=frame_count,
            duration_sec=duration,
        )
        self.metrics = ProcessingMetrics()

    def __iter__(self) -> Iterator[Frame]:
        self.metrics.started_at = time.monotonic()
        index = 0
        while True:
            ok, image = self._cap.read()
            if not ok:
                break
            timestamp = index / self.info.fps
            self.metrics.frames_read += 1
            yield Frame(index=index, timestamp_sec=timestamp, image=image)
            index += 1
        self.metrics.finished_at = time.monotonic()

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# Longest edge fed to analysis, regardless of what the camera sends. Detection
# accuracy is governed by how many pixels an object covers, not by the frame's
# absolute size, so a distant figure gains almost nothing from 4K while the
# per-frame cost rises with every pixel. Modern cameras happily stream 4K;
# analysing it unscaled on CPU is the single easiest way to make a pipeline
# look slow for no accuracy gain.
DEFAULT_MAX_ANALYSIS_WIDTH = 960


ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def rotate_frame(image: np.ndarray, degrees: int) -> np.ndarray:
    code = ROTATIONS.get(degrees)
    return image if code is None else cv2.rotate(image, code)


def downscale_for_analysis(image: np.ndarray, max_width: int | None) -> np.ndarray:
    """Shrink a frame to `max_width` if it is wider. Never upscales - a small
    camera's frames are left exactly as they arrived."""
    if not max_width or max_width <= 0:
        return image
    height, width = image.shape[:2]
    if width <= max_width:
        return image
    scale = max_width / width
    return cv2.resize(
        image, (max_width, max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA,
    )


# DirectShow opens of two webcams at the same moment fail ("raised unknown C++
# exception") - seen when the webcam scan probed while a camera was starting.
# Every DirectShow webcam open takes this lock; each holds it under a second.
WEBCAM_OPEN_LOCK = threading.Lock()

ONLINE = "online"
CONNECTING = "connecting"
RECONNECTING = "reconnecting"
OFFLINE = "offline"
STOPPED = "stopped"


@dataclass
class LiveStreamStatus:
    """Health of one live source. This is what the camera-health board in the
    UI renders, and what makes CAMERA_OFFLINE/CAMERA_DEGRADED expressible as
    real events rather than a crashed background thread."""

    state: str = CONNECTING
    frames_read: int = 0
    frames_dropped: int = 0  # skipped to stay in real time when analysis lags
    reconnect_count: int = 0
    last_frame_at: float | None = None  # time.monotonic() of the last good frame
    measured_fps: float = 0.0
    width: int = 0
    height: int = 0
    source_fps: float = 0.0
    error: str | None = None

    @property
    def frame_age_sec(self) -> float | None:
        """Seconds since the last decoded frame. A live source whose frame age
        keeps growing while it still claims to be online is a frozen camera -
        the failure mode a plain is-the-socket-open check misses entirely."""
        if self.last_frame_at is None:
            return None
        return max(0.0, time.monotonic() - self.last_frame_at)

    # After this long without a frame, the last measured rate is history, not
    # a current reading.
    FPS_STALE_AFTER_SEC = 3.0

    def as_dict(self) -> dict:
        age = self.frame_age_sec
        # A dead camera must not keep reporting the throughput it had while it
        # was alive. Holding the last value steady is the single most
        # misleading thing a health board can do - it makes a failed feed look
        # perfectly healthy on every metric an operator scans first.
        fps = 0.0 if (age is None or age > self.FPS_STALE_AFTER_SEC) else self.measured_fps
        return {
            "state": self.state,
            "frames_read": self.frames_read,
            "frames_dropped": self.frames_dropped,
            "reconnect_count": self.reconnect_count,
            "frame_age_sec": round(age, 2) if age is not None else None,
            "measured_fps": round(fps, 2),
            "width": self.width,
            "height": self.height,
            "source_fps": round(self.source_fps, 2),
            "error": self.error,
        }


class LiveStreamReader:
    """Never-ending frame source for a live camera.

    Accepts an RTSP/HTTP camera URL, or a local file path with `loop=True` so
    a recorded clip can stand in for a physical camera during development and
    demos - the same code path either way, so nothing changes when a real
    border-post camera replaces the stand-in.

    Three things make this different from VideoReader, and all three are
    required for continuous operation rather than being nice-to-haves:

    1. A failed read is not the end of the stream. Cameras drop, networks
       blip, cables get kicked. The iterator marks itself RECONNECTING, backs
       off, reopens, and carries on; only an explicit stop() ends iteration.
       Callers therefore never need try/except around a camera going away.
    2. Timestamps are wall-clock elapsed, not frame_index/fps. Across a
       reconnect or a loop restart there is no continuous frame counter to
       divide, and every downstream rule (zone dwell, loitering windows,
       behaviour cooldowns) reasons in seconds - so those seconds have to be
       real elapsed time, or a reconnect would silently rewind them.
    3. File sources are paced to their own FPS. Without that, a local stand-in
       file is decoded as fast as the CPU allows, which both misrepresents
       live throughput and makes every time-based rule fire at the wrong rate.
       Real RTSP is paced by the network and needs no help.
    """

    # Bounded exponential backoff: fast enough that a brief blip recovers
    # almost immediately, capped so a camera that is genuinely gone doesn't
    # spin the CPU reconnecting.
    INITIAL_RECONNECT_DELAY = 1.0
    MAX_RECONNECT_DELAY = 15.0

    # Never skip more than this many frames at once. A huge skip means the
    # consumer stalled badly (or the machine slept); jumping the scene forward
    # by minutes would be worse than accepting one discontinuity.
    MAX_CATCHUP_SKIP = 120

    def __init__(
        self,
        source: str | Path,
        loop: bool = False,
        pace_to_source_fps: bool | None = None,
        fps_window: int = 30,
        target_fps: float | None = None,
        max_width: int | None = DEFAULT_MAX_ANALYSIS_WIDTH,
        rotate: int = 0,
    ):
        self.source = str(source)
        self.max_width = max_width
        # Clockwise degrees. A phone used as a webcam (DroidCam over USB)
        # sends its picture sideways when held upright, and face detection
        # cannot find a face lying on its side.
        self.rotate = rotate if rotate in ROTATIONS else 0
        # Capture rate to aim for, independent of what the file/camera offers.
        # Analytics cameras are routinely configured well below their maximum
        # (12-15fps is typical), and a lower rate means consecutive analysed
        # frames are closer together in scene time - which is what keeps
        # tracking continuous when inference cannot run at full rate.
        self.target_fps = target_fps
        self.is_network_source = self.source.lower().startswith(
            ("rtsp://", "rtsps://", "http://", "https://")
        )
        # A bare number ("0", "1") is a USB webcam plugged into this machine.
        self.is_webcam = self.source.strip().isdigit()
        self.loop = loop
        # Pace file playback by default (a file has no natural rate); never
        # pace a network source or a webcam, which run at their own rate.
        live = self.is_network_source or self.is_webcam
        self.pace = (not live) if pace_to_source_fps is None else pace_to_source_fps

        self.status = LiveStreamStatus()
        self.metrics = ProcessingMetrics()
        self._cap: cv2.VideoCapture | None = None
        self._info: VideoInfo | None = None
        self._stop = False
        self._fps_window = max(2, fps_window)
        self._recent_frame_times: list[float] = []

    # --- VideoReader-compatible surface --------------------------------------
    #
    # The analysis pipeline is written against VideoReader's `info`. Exposing
    # the same shape here is what lets one pipeline consume a file or a camera
    # without branching on which it got. frame_count/duration are 0 because a
    # live stream genuinely has neither - callers already treat 0 as unknown.

    @property
    def info(self) -> VideoInfo:
        if self._info is None:
            raise VideoOpenError(
                f"connect() must succeed before stream info is available: {self.source}"
            )
        return self._info

    @property
    def is_connected(self) -> bool:
        return self._cap is not None

    def connect(self, timeout_sec: float = 15.0) -> VideoInfo:
        """Open the stream, blocking until it is up or the timeout expires.

        Iteration can open lazily on its own, but a caller that needs frame
        dimensions up front (to scale zone polygons, size an overlay) has to
        know them before the first frame - hence an explicit connect step.
        """
        deadline = time.monotonic() + timeout_sec
        attempt_delay = self.INITIAL_RECONNECT_DELAY
        while not self._stop:
            if self._open():
                self.status.state = ONLINE
                self.status.error = None
                return self.info
            if time.monotonic() >= deadline:
                break
            time.sleep(min(attempt_delay, max(0.0, deadline - time.monotonic())))
            attempt_delay = min(self.MAX_RECONNECT_DELAY, attempt_delay * 2)
        self.status.state = OFFLINE
        self.status.error = f"could not open source: {self.source}"
        raise VideoOpenError(f"could not open live source within {timeout_sec}s: {self.source}")

    # --- connection handling -------------------------------------------------

    def _open(self) -> bool:
        """Open the capture. Returns False instead of raising - the caller is a
        reconnect loop, and 'not open yet' is an expected state for a camera,
        not an exception."""
        if self.is_network_source:
            # OpenCV's FFmpeg backend defaults to UDP for RTSP, which shows up
            # as torn/green macroblocked frames on any lossy link. TCP trades a
            # little latency for frames that actually decode. Only set when the
            # operator hasn't already expressed a preference.
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

        if self.is_webcam:
            # DirectShow opens Windows webcams in about a second; the default
            # backend can take ten. Some virtual cameras (DroidCam's phone
            # webcam among them) register only with Media Foundation, so fall
            # back to it when DirectShow cannot open the index.
            index = int(self.source.strip())
            if os.name == "nt":
                # DirectShow first, every time. Remembering a webcam as
                # Media-Foundation-only after one DirectShow failure stuck
                # two USB webcams on MSMF for good, where they gave black
                # frames at 1fps; the failure itself was only another open
                # of a different webcam at the same moment, which the lock
                # now prevents. MSMF stays the fallback for cameras that
                # register only there (DroidCam's phone webcam).
                with WEBCAM_OPEN_LOCK:
                    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap.release()
                    cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
            else:
                cap = cv2.VideoCapture(index, cv2.CAP_ANY)
            # Webcams open at 640x480 unless asked; faces at a few metres need
            # more pixels than that. A camera without HD keeps its own size.
            # Then ask for 30fps MJPG: uncompressed 720p is all a USB 2.0
            # webcam can push at a crawl (Logitech Brio 100: 5fps raw, 30fps
            # MJPG; measured, two webcams together went 5+10 -> 30+25fps).
            # The order matters - with DirectShow the format request only
            # sticks when it comes after the size and the frame rate. A
            # camera without MJPG ignores it and keeps its raw format.
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        else:
            cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            cap.release()
            return False

        if self.is_network_source:
            # Keep the internal buffer at one frame: on a live feed a backlog
            # is pure latency, since we always want the newest frame, never a
            # queued stale one.
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass  # not supported by every backend; harmless if ignored

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0.1 or fps > 240:
            fps = 25.0  # cameras routinely under-report or lie
        self.status.source_fps = fps
        raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.rotate in (90, 270):
            raw_w, raw_h = raw_h, raw_w
        if self.max_width and raw_w > self.max_width:
            scale = self.max_width / raw_w
            raw_w, raw_h = self.max_width, max(1, int(round(raw_h * scale)))
        self.status.width, self.status.height = raw_w, raw_h
        self._cap = cap
        self._info = VideoInfo(
            path=Path(self.source),
            fps=fps,
            width=self.status.width,
            height=self.status.height,
            frame_count=0,      # unknown / unbounded for a live source
            duration_sec=0.0,
        )
        return True

    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def effective_fps(self) -> float:
        """Rate this reader aims to deliver frames at."""
        if self.target_fps and self.target_fps > 0:
            return min(self.target_fps, self.status.source_fps or self.target_fps)
        return self.status.source_fps

    def _skip_frames(self, count: int) -> int:
        """Discard `count` frames without decoding them.

        grab() advances the stream but skips decode, which is the cheap half of
        the work - important, since the whole point of skipping is that there
        is no time to spare.
        """
        if self._cap is None:
            return 0
        skipped = 0
        for _ in range(count):
            if not self._cap.grab():
                if self.loop and not self.is_network_source:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            skipped += 1
        return skipped

    # --- metrics -------------------------------------------------------------

    def _record_frame_time(self, now: float) -> None:
        self._recent_frame_times.append(now)
        if len(self._recent_frame_times) > self._fps_window:
            self._recent_frame_times.pop(0)
        if len(self._recent_frame_times) >= 2:
            span = self._recent_frame_times[-1] - self._recent_frame_times[0]
            if span > 0:
                self.status.measured_fps = (len(self._recent_frame_times) - 1) / span

    # --- iteration -----------------------------------------------------------

    def __iter__(self) -> Iterator[Frame]:
        index = 0
        started_at = time.monotonic()
        self.metrics.started_at = started_at
        next_due = started_at
        delay = self.INITIAL_RECONNECT_DELAY

        while not self._stop:
            if self._cap is None:
                self.status.state = CONNECTING if index == 0 else RECONNECTING
                if not self._open():
                    self.status.state = OFFLINE
                    self.status.error = f"could not open source: {self.source}"
                    # Sleep in slices so stop() stays responsive during a long
                    # backoff rather than waiting out the full delay.
                    waited = 0.0
                    while waited < delay and not self._stop:
                        time.sleep(min(0.25, delay - waited))
                        waited += 0.25
                    delay = min(self.MAX_RECONNECT_DELAY, delay * 2)
                    continue
                self.status.state = ONLINE
                self.status.error = None
                delay = self.INITIAL_RECONNECT_DELAY
                next_due = time.monotonic()

            ok, image = self._cap.read()

            if not ok:
                if self.loop and not self.is_network_source:
                    # End of the stand-in file: rewind and keep going, so the
                    # "camera" runs indefinitely. Deliberately not counted as a
                    # reconnect - nothing failed.
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, image = self._cap.read()
                    if ok:
                        # fall through and emit this frame normally
                        pass
                if not ok:
                    self._release()
                    self.status.state = RECONNECTING
                    self.status.reconnect_count += 1
                    self.status.error = "stream read failed"
                    continue

            now = time.monotonic()
            self.status.frames_read += 1
            self.metrics.frames_read += 1
            self.status.last_frame_at = now
            self.status.state = ONLINE
            self._record_frame_time(now)

            yield Frame(index=index, timestamp_sec=now - started_at,
                        image=downscale_for_analysis(rotate_frame(image, self.rotate), self.max_width))
            index += 1

            rate = self.effective_fps
            if self.pace and rate > 0:
                interval = 1.0 / rate
                next_due += interval
                sleep_for = next_due - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    # Behind real time: the consumer is slower than the camera.
                    # Discard the frames that elapsed while it was busy instead
                    # of working through a backlog, so scene time keeps
                    # matching wall-clock time. This is what a real camera does
                    # to you anyway - those frames are simply gone - and it is
                    # what keeps "loitered for 10 seconds" meaning ten real
                    # seconds when inference runs below capture rate.
                    behind = -sleep_for
                    skip = min(self.MAX_CATCHUP_SKIP, int(behind * rate))
                    if skip > 0:
                        dropped = self._skip_frames(skip)
                        self.status.frames_dropped += dropped
                        next_due += dropped * interval
                    if next_due < time.monotonic() - 1.0:
                        next_due = time.monotonic()  # too far gone; re-anchor

        self._release()
        self.metrics.finished_at = time.monotonic()
        self.status.state = STOPPED

    def stop(self) -> None:
        """Ask the iterator to finish. Safe to call from another thread - the
        loop checks this flag between frames and during reconnect backoff."""
        self._stop = True

    def close(self) -> None:
        self.stop()
        self._release()

    def __enter__(self) -> "LiveStreamReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


PUSH_SCHEME = "push://"


class PushStreamReader:
    """A live source whose frames are pushed in from outside.

    LiveStreamReader *pulls* from a URL it can open. A phone's browser cannot
    be pulled from - it sits behind the same NAT as everything else and serves
    nothing - so it pushes instead: the page captures its camera and sends
    JPEG frames up a websocket, which land here via push().

    The interface deliberately matches LiveStreamReader, so the analysis
    pipeline consumes a phone exactly like an RTSP camera and none of the
    detection code knows the difference.
    """

    # If the sender goes quiet for this long, the camera is treated as gone
    # rather than silently frozen on its last frame.
    SENDER_TIMEOUT_SEC = 8.0

    def __init__(self, camera_id: str, target_fps: float | None = None, fps_window: int = 30,
                 max_width: int | None = DEFAULT_MAX_ANALYSIS_WIDTH):
        self.camera_id = camera_id
        self.source = f"{PUSH_SCHEME}{camera_id}"
        self.is_network_source = True
        self.loop = False
        self.pace = False  # the sender sets the pace
        self.target_fps = target_fps
        self.max_width = max_width

        self.status = LiveStreamStatus()
        self.metrics = ProcessingMetrics()
        self._info: VideoInfo | None = None
        self._latest: np.ndarray | None = None
        self._seq = 0
        self._condition = threading.Condition()
        self._stop = False
        self._fps_window = max(2, fps_window)
        self._recent_frame_times: list[float] = []

    # --- sender side ---------------------------------------------------------

    def push(self, jpeg: bytes) -> bool:
        """Accept one encoded frame from the sender. Returns False if it could
        not be decoded, so the caller can count bad frames rather than guess."""
        buffer = np.frombuffer(jpeg, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            return False
        image = downscale_for_analysis(image, self.max_width)

        now = time.monotonic()
        with self._condition:
            self._latest = image
            self._seq += 1
            h, w = image.shape[:2]
            if self._info is None or self._info.width != w or self._info.height != h:
                # Dimensions can change when the phone rotates; zone polygons
                # are normalized, so they follow the new frame size correctly.
                self._info = VideoInfo(
                    path=Path(self.source), fps=self.target_fps or 12.0,
                    width=w, height=h, frame_count=0, duration_sec=0.0,
                )
                self.status.width, self.status.height = w, h
                self.status.source_fps = self._info.fps
            self.status.state = ONLINE
            self.status.error = None
            self.status.last_frame_at = now
            self._condition.notify_all()

        self._recent_frame_times.append(now)
        if len(self._recent_frame_times) > self._fps_window:
            self._recent_frame_times.pop(0)
        if len(self._recent_frame_times) >= 2:
            span = self._recent_frame_times[-1] - self._recent_frame_times[0]
            if span > 0:
                self.status.measured_fps = (len(self._recent_frame_times) - 1) / span
        return True

    @property
    def has_sender(self) -> bool:
        age = self.status.frame_age_sec
        return age is not None and age < self.SENDER_TIMEOUT_SEC

    # --- LiveStreamReader-compatible surface ---------------------------------

    @property
    def info(self) -> VideoInfo:
        if self._info is None:
            raise VideoOpenError(f"no frames received yet from {self.source}")
        return self._info

    @property
    def is_connected(self) -> bool:
        return self._info is not None

    def connect(self, timeout_sec: float = 30.0) -> VideoInfo:
        """Wait for the first pushed frame, which is what reveals the sender's
        resolution. Generous default: a person has to physically open the page
        on the phone and grant camera permission."""
        deadline = time.monotonic() + timeout_sec
        with self._condition:
            while self._info is None and not self._stop:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=min(0.5, remaining))
        if self._info is None:
            self.status.state = OFFLINE
            self.status.error = "no frames received from the sending device"
            raise VideoOpenError(f"no frames pushed to {self.source} within {timeout_sec}s")
        self.status.state = ONLINE
        return self._info

    def __iter__(self) -> Iterator[Frame]:
        index = 0
        started_at = time.monotonic()
        self.metrics.started_at = started_at
        last_seq = 0

        while not self._stop:
            with self._condition:
                if self._seq == last_seq:
                    self._condition.wait(timeout=1.0)
                if self._stop:
                    break
                if self._seq == last_seq:
                    # Nothing arrived. Distinguish "sender paused" from "sender
                    # gone" so the health board can show the difference.
                    if not self.has_sender and self.status.state == ONLINE:
                        self.status.state = RECONNECTING
                        self.status.error = "sending device stopped transmitting"
                    continue
                last_seq = self._seq
                image = self._latest

            self.status.frames_read += 1
            self.metrics.frames_read += 1
            yield Frame(index=index, timestamp_sec=time.monotonic() - started_at, image=image)
            index += 1

        self.metrics.finished_at = time.monotonic()
        self.status.state = STOPPED

    def stop(self) -> None:
        self._stop = True
        with self._condition:
            self._condition.notify_all()

    def close(self) -> None:
        self.stop()

    def __enter__(self) -> "PushStreamReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def extract_frame(path: str | Path, fraction: float = 0.5) -> tuple[np.ndarray, VideoInfo]:
    """Grabs one representative frame at `fraction` of the video's duration
    (0.0 = first frame, 0.5 = middle, 1.0 = last) - used to hand the UI a
    frame to draw a zone on. Cheap: opens the file, seeks once, reads once.
    """
    fraction = min(1.0, max(0.0, fraction))
    reader = VideoReader(path)
    try:
        target_index = min(reader.info.frame_count - 1, int(round(fraction * (reader.info.frame_count - 1))))
        target_index = max(0, target_index)
        cap = reader._cap
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_index)
        ok, image = cap.read()
        if not ok:
            # fall back to a linear scan from the start if seeking landed badly
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            image = None
            for _ in range(target_index + 1):
                ok, frame = cap.read()
                if not ok:
                    break
                image = frame
            if image is None:
                raise VideoOpenError(f"could not read any frame from {path}")
        return image, reader.info
    finally:
        reader.close()
