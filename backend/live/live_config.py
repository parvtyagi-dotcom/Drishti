"""Live-mode tuning and the camera / analysis records a session is built from.

Split out of backend.live.camera_manager with LiveSession. The live_*_config
functions retune Forensic Mode's defaults for the few frames a second a CPU
analyses live; Camera and AnalysisSpec describe what a session watches and
which modules it runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from backend.config import AnprConfig, BehaviorConfig, PersonIDConfig, TrackerConfig
from backend.core.video import DEFAULT_MAX_ANALYSIS_WIDTH, PUSH_SCHEME


def live_behavior_config() -> BehaviorConfig:
    """Behaviour thresholds retuned for live analysis rates.

    The defaults in config.BehaviorConfig assume Forensic Mode's conditions:
    every frame of a 30-60fps file, so a track's path is sampled almost
    continuously and an 8px movement floor with a 0.3s checkpoint reliably
    separates real movement from detector jitter.

    Live analysis on CPU runs at roughly 3-6fps, which breaks those
    assumptions in a specific way: consecutive observations are ~0.25s and
    tens of pixels apart, so *every* frame clears an 8px floor and becomes a
    new movement segment, and any tracker wobble between two distant
    observations reads as a direction reversal. Measured on a sparse test
    clip, the file-tuned defaults produced ~120 behaviour events in 30
    seconds - which is noise, not intelligence, and an alert stream nobody can
    act on is worse than none.

    These values restore the intent by spanning a longer baseline per segment
    and demanding more evidence before calling something suspicious.
    """
    return BehaviorConfig(
        # A segment now spans ~1s of real motion rather than a single frame,
        # so direction and speed are measured against a stable baseline.
        sample_interval_sec=1.0,
        min_displacement_px=25.0,
        # Only near-reversals count, not the ordinary course corrections a
        # person makes walking around others. Set this high because the
        # remaining false positives at low frame rates come from tracker ID
        # churn - one identity handed to two people looks like an impossible
        # turn - and only a near-180 degree threshold reliably rejects that.
        direction_change_degrees=120.0,
        reversal_angle_degrees=150.0,
        # Pacing means repeatedly retracing a path, which needs a longer
        # window and more repetitions to distinguish from walking past twice.
        reversal_count=4,
        reversal_window_sec=30.0,
        # One alert per track per 20s: enough to stay informed, not enough to
        # bury the operator.
        behavior_cooldown_seconds=20.0,
        # The same ~1s baseline, in metres, for a calibrated camera.
        min_displacement_m=0.8,
        vehicle_min_displacement_m=2.0,
    )


def live_tracker_config() -> TrackerConfig:
    """Live keeps the person tracker at its normal stride when a target is
    locked - see TrackerConfig.adaptive_person_stride. The zone urgency
    stays on: it only runs the shared detector more often near a zone.

    The person tracker does detect every 2nd analysed frame instead of every
    5th (person_detection_stride; the shared vehicle/zone tracker keeps 5).
    At live's ~3-6 analysed fps a stride of 5 meant one real detection about
    every second, so a walking person's propagated box drifted, BoT-SORT
    gave them a fresh id, and face matches were split across ids before any
    one reached confirmation_observations. Measured on smoke_person_id.mp4
    with the live person-ID config (every frame analysed, CPU): stride 5
    1.87 fps, 3 1.61, 2 2.10 - within run-to-run noise, because face and
    ReID on every 2nd frame dominate the cost, not YOLO (36 vs 90 detector
    calls over 180 frames). Every-frame detection (adaptive_person_stride)
    stays off for the multi-camera CPU load reason above."""
    return TrackerConfig(adaptive_person_stride=False, person_detection_stride=2)


def live_anpr_config() -> AnprConfig:
    """ANPR sampling retuned for live analysis rates.

    ocr_sample_stride exists to keep an expensive step off every frame, and
    5 is right for Forensic Mode: a 30fps file yields ~6 OCR looks a second,
    so a plate is read a dozen times or more while a vehicle crosses the view
    and vote_min_observations agreeing reads is an easy bar to clear.

    Live analysis already drops frames to stay in step with the wall clock,
    running at roughly 3-6fps on CPU. Striding again on top of that decimates
    twice: the same setting yields well under one look per second, so a
    vehicle passing in four seconds is read two or three times - never enough
    agreeing reads to confirm a plate. Measured on real road footage, plates
    were located 46 times and confirmed zero times.

    Sampling every analysed frame restores the intended cadence in time
    rather than in frames. It costs nothing extra per frame, because the
    frames it now samples are ones the pipeline already decoded and detected
    vehicles in.
    """
    return AnprConfig(ocr_sample_stride=1)


def live_person_id_config() -> PersonIDConfig:
    """Face and ReID sampling retuned for live analysis rates.

    Same reasoning as live_anpr_config: face_sample_stride=5 over frames that
    have already been thinned to a few per second leaves only a fraction of a
    face look per second, so a target has to stand still for many seconds
    before confirmation_observations can be met. A person who merely walks
    through the view is read once, or not at all.

    Every second frame, not every frame: at the 12fps live capture rate that
    is still six face looks a second, and checking every frame was what
    capped a camera with people close in view at ~4fps - so its video lagged.
    Measured on the street demo clip: every frame 3.6fps, every 2nd 7.4fps,
    target confirmed both ways.
    """
    # One agreeing reference is enough live: faces there are small and few,
    # and confirmation_observations still demands two separate matching
    # looks. Forensic Mode keeps the stricter default of two references.
    return PersonIDConfig(face_sample_stride=2, reid_sample_stride=2, min_reference_votes=1)


# Capture rate for live analysis. Deliberately far below what the demo clips
# (30-60fps) or a modern camera can deliver: analytics gains nothing from
# 60fps, and a lower rate means consecutive analysed frames sit closer
# together in scene time whenever inference cannot keep up - which on CPU it
# cannot. Matches how analytics cameras are configured in practice.
DEFAULT_CAPTURE_FPS = 12.0


def _source_kind(source: str) -> str:
    """Coarse description of where a camera's frames come from. Returned to
    clients instead of the source itself, which can embed RTSP credentials."""
    lowered = source.lower()
    if lowered.startswith(PUSH_SCHEME):
        return "phone"
    if lowered.startswith(("rtsp://", "rtsps://")):
        return "rtsp"
    if lowered.startswith(("http://", "https://")):
        return "http"
    if lowered.strip().isdigit():
        return "webcam"
    return "file"


@dataclass
class Camera:
    camera_id: str
    name: str
    source: str
    loop: bool = False
    location: str | None = None
    target_fps: float = DEFAULT_CAPTURE_FPS
    # Frames are downscaled to this width before analysis. Raise it for a
    # camera doing ANPR, where legibility depends on plate pixels and the
    # default trades resolution for speed.
    max_width: int = DEFAULT_MAX_ANALYSIS_WIDTH
    # Clockwise degrees applied to every frame (0/90/180/270), for a phone
    # webcam that sends its picture sideways.
    rotate: int = 0
    # Account that added it; None for demo cameras (see backend.api.ownership).
    owner: str | None = None

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            # The source can embed RTSP credentials (rtsp://user:pass@host),
            # so it is never returned to clients - only whether it is a
            # network camera or a local stand-in file.
            "source_kind": _source_kind(self.source),
            "loop": self.loop,
            "target_fps": self.target_fps,
            "max_width": self.max_width,
            "rotate": self.rotate,
            "location": self.location,
        }


@dataclass
class AnalysisSpec:
    """Which analysis modules this camera runs.

    Mirrors the fields of a Forensic Mode job request on purpose: a camera is
    just the same analysis pointed at a stream instead of a file, so the two
    stay configured the same way.
    """

    person_id: bool = False
    reference_photo_paths: list[Path] = field(default_factory=list)
    # Kept so the UI can show which enrolment is active and let the operator
    # change other settings without re-uploading the same photos.
    reference_set_id: str | None = None
    anpr: bool = False
    target_plate: str | None = None
    zones: list = field(default_factory=list)
    behavior: bool = False
    behavior_config: BehaviorConfig | None = None  # defaults to live_behavior_config()
    lowlight: bool = False
    fog: bool = False

    @property
    def any_enabled(self) -> bool:
        return bool(self.person_id or self.anpr or self.zones or self.behavior or self.lowlight or self.fog)

    def module_names(self) -> list[str]:
        names = []
        for enabled, name in (
            (self.person_id, "person_id"), (self.anpr, "anpr"), (bool(self.zones), "zone"),
            (self.behavior, "behavior"), (self.lowlight, "lowlight"), (self.fog, "fog"),
        ):
            if enabled:
                names.append(name)
        return names
