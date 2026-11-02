"""Detection + tracking wrapper around Ultralytics YOLO + ByteTrack.

This is the one place that runs the per-frame detector and turns its output
into stable track ids. Modules (person_id, anpr, zone, behavior) consume
Track objects instead of talking to ultralytics directly.

Performance note: full YOLO+ByteTrack inference is by far the dominant cost
in every pipeline (benchmarked ~75-90% of per-frame time depending on which
other modules are active). TrackerConfig.detection_stride lets a caller run
the real detector only every Nth frame and propagate tracks between samples
via simple linear velocity extrapolation - cheap (pure arithmetic, no model
call) and bounded (never extrapolates more than stride-1 frames before the
next real detection resyncs everything). update(urgent=True) overrides the
stride for one frame, which is how the orchestrator detects every frame
while something is near a zone or a target is locked. Default stride=1 means "detect
every frame", byte-for-byte the original behavior - this is opt-in, not a
default change, so every existing caller/test is unaffected unless it
explicitly sets a larger stride.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ultralytics import YOLO

from backend.config import TrackerConfig


@dataclass
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixel coords
    class_id: int
    confidence: float
    propagated: bool = False  # True if this frame's box was extrapolated, not detected

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class ObjectTracker:
    """Persistent-state YOLO+ByteTrack tracker for a single video's lifetime.

    Create one instance per video processing run (persist=True tracking
    state is tied to the underlying ultralytics predictor and must not be
    shared across unrelated videos).
    """

    def __init__(self, config: TrackerConfig, class_ids: tuple[int, ...] | None = None):
        self.config = config
        self.class_ids = list(class_ids) if class_ids is not None else None
        self._model = YOLO(str(config.model_path))
        self._frame_index = 0
        self._last_detect_index: int | None = None
        self.detection_count = 0  # frames where the model actually ran
        self._last_tracks: dict[int, Track] = {}
        # Real detections only, with the frame each was seen on. Kept apart
        # from _last_tracks because that one is overwritten with extrapolated
        # boxes between detections, and measuring motion from an extrapolated
        # position feeds the tracker's own guesses back into itself.
        self._last_detected: dict[int, tuple[Track, int]] = {}
        self._velocity: dict[int, tuple[float, float, float, float]] = {}

    def update(self, image: np.ndarray, urgent: bool = False) -> list[Track]:
        """urgent=True runs the real detector on this frame whatever the
        stride: the caller has something worth watching closely (a track
        near a zone, a locked target), where a box extrapolated for up to
        stride-1 frames could miss the moment it crosses a line. Otherwise
        the detector runs once detection_stride frames have passed since it
        last ran - counted from the last detection, not the frame number, so
        dropping back from urgent frames does not detect early or late."""
        stride = max(1, self.config.detection_stride)
        do_detect = (
            urgent
            or self._last_detect_index is None
            or self._frame_index - self._last_detect_index >= stride
        )
        if do_detect:
            self._last_detect_index = self._frame_index
        self._frame_index += 1

        if do_detect:
            self.detection_count += 1
            tracks = self._detect(image)
            self._remember(tracks)
            return tracks
        return self._propagate(image.shape[1], image.shape[0])

    def _detect(self, image: np.ndarray) -> list[Track]:
        results = self._model.track(
            image,
            persist=True,
            tracker=self.config.tracker_yaml,
            conf=self.config.confidence,
            iou=self.config.iou,
            classes=self.class_ids,
            device=self.config.device,
            verbose=False,
        )
        tracks: list[Track] = []
        if not results:
            return tracks
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return tracks  # detections with no assigned track id are dropped

        ids = boxes.id.int().cpu().tolist()
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().tolist()
        clss = boxes.cls.int().cpu().tolist()

        for tid, box, conf, cls in zip(ids, xyxy, confs, clss):
            x1, y1, x2, y2 = box
            tracks.append(
                Track(
                    track_id=int(tid),
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    class_id=int(cls),
                    confidence=float(conf),
                )
            )
        return tracks

    def _remember(self, tracks: list[Track]) -> None:
        """Records this real detection as the baseline for propagation, and
        each track's velocity **per frame**.

        Consecutive detections are detection_stride frames apart, so their
        bbox delta is that many frames of movement. Propagation applies the
        velocity once per skipped frame, so the delta has to be divided by the
        frames it actually spans - handing it over undivided made every
        extrapolated box travel a whole stride's worth of motion per frame,
        racing ahead of the vehicle and snapping back at the next detection.
        That overshoot-and-snap, repeating every stride frames, is what a
        viewer sees as a jittering box.

        The elapsed count is measured per track rather than assumed to be the
        stride, because a track missed by one detection and picked up by the
        next has been gone longer than that.
        """
        detected_on = self._frame_index - 1  # update() has already advanced it
        new_velocity: dict[int, tuple[float, float, float, float]] = {}
        for t in tracks:
            previous = self._last_detected.get(t.track_id)
            if previous is None:
                new_velocity[t.track_id] = (0.0, 0.0, 0.0, 0.0)
                continue
            prev_track, prev_frame = previous
            elapsed = max(1, detected_on - prev_frame)
            px1, py1, px2, py2 = prev_track.bbox
            x1, y1, x2, y2 = t.bbox
            new_velocity[t.track_id] = (
                (x1 - px1) / elapsed, (y1 - py1) / elapsed,
                (x2 - px2) / elapsed, (y2 - py2) / elapsed,
            )
        self._last_tracks = {t.track_id: t for t in tracks}
        self._last_detected = {t.track_id: (t, detected_on) for t in tracks}
        self._velocity = new_velocity

    def _propagate(self, width: int, height: int) -> list[Track]:
        """Extrapolates each track's last known box forward by its recorded
        velocity - no model inference. A track that has genuinely left frame
        or stopped existing will drift briefly (at most stride-1 frames)
        until the next real detection corrects or drops it; downstream
        consumers (TargetLock/ZoneMonitor/BehaviorMonitor) already tolerate
        exactly this kind of brief inconsistency by design."""
        propagated: list[Track] = []
        for tid, t in self._last_tracks.items():
            dx1, dy1, dx2, dy2 = self._velocity.get(tid, (0.0, 0.0, 0.0, 0.0))
            x1, y1, x2, y2 = t.bbox
            nx1, ny1, nx2, ny2 = x1 + dx1, y1 + dy1, x2 + dx2, y2 + dy2
            nx1, nx2 = max(0.0, min(width, nx1)), max(0.0, min(width, nx2))
            ny1, ny2 = max(0.0, min(height, ny1)), max(0.0, min(height, ny2))
            if nx2 <= nx1 or ny2 <= ny1:
                continue
            new_track = Track(
                track_id=tid, bbox=(int(nx1), int(ny1), int(nx2), int(ny2)),
                class_id=t.class_id, confidence=t.confidence, propagated=True,
            )
            propagated.append(new_track)
            self._last_tracks[tid] = new_track  # so the NEXT propagated frame extrapolates further, not from the same stale point
        return propagated
