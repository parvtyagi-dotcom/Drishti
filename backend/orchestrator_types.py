"""Request, result and error types for the combined pipeline.

Kept apart from backend.orchestrator so callers that only build a request or
read a result do not need the pipeline code itself; backend.orchestrator
re-exports all three, so existing imports keep working.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from backend.config import (
    AnprConfig, BehaviorConfig, LowLightConfig, PersonIDConfig, TrackerConfig, ZoneConfig,
)
from backend.core.video import VideoInfo
from backend.modules.zone.geometry import Zone
from backend.modules.zone.state import ZoneMonitor


@dataclass
class OrchestratorRequest:
    enable_person_id: bool = False
    reference_photo_paths: list[Path] = field(default_factory=list)
    # Cross-camera appearance sharing. Both optional: a single-camera run
    # behaves exactly as before when they are not supplied.
    share_appearance: Callable[[object], None] | None = None
    match_shared_appearance: Callable[[object], float] | None = None
    person_id_config: PersonIDConfig | None = None
    snapshot_path: Path | None = None

    enable_anpr: bool = False
    target_plate: str | None = None
    anpr_config: AnprConfig | None = None

    zones: list[Zone] = field(default_factory=list)
    zone_config: ZoneConfig | None = None

    enable_behavior: bool = False
    behavior_config: BehaviorConfig | None = None

    enable_lowlight: bool = False
    # Fog mode (core/dehaze.py): clears haze before detection, only on frames
    # that are actually foggy.
    enable_dehaze: bool = False
    lowlight_config: LowLightConfig | None = None

    tracker_config: TrackerConfig | None = None
    evidence_dir: Path | None = None

    # Live only: called each frame with the clean image and what is in view,
    # as (track_id, coco_class_id, bbox) tuples. Group counting and search
    # crops read it; nothing in the analysis depends on it.
    observe_tracks: Callable[[object, list[tuple[int, int, tuple[int, int, int, int]]]], None] | None = None
    # Live only: a core.motion.MotionGate. When set, the detectors skip frames
    # with no movement and nothing tracked. Forensic runs look at every frame.
    motion_gate: object | None = None
    # A core.calibration.GroundCalibration for this camera's view. When set,
    # behaviour analytics works in metres on the ground (separate limits for
    # people and vehicles) instead of pixels.
    ground_calibration: object | None = None


@dataclass
class OrchestratorResult:
    output_video_path: Path | None  # None for a live run, which produces no file
    codec_used: str
    browser_playable: bool
    video_info: VideoInfo
    metrics: dict
    events: list[dict]
    modules_run: list[str]
    person_id: dict | None = None
    anpr: dict | None = None
    zones: list[dict] | None = None
    behavior: dict | None = None
    lowlight: dict | None = None
    evidence_dir: Path | None = None


class OrchestratorError(RuntimeError):
    pass


def enabled_modules(req: OrchestratorRequest) -> list[str]:
    """Names of the modules a request turns on, in a fixed order."""
    flags = (
        ("person_id", req.enable_person_id), ("anpr", req.enable_anpr), ("zone", bool(req.zones)),
        ("behavior", req.enable_behavior), ("lowlight", req.enable_lowlight), ("fog", req.enable_dehaze),
    )
    return [name for name, on in flags if on]


def zone_monitors_for(zones: list[Zone], cfg: ZoneConfig) -> list[ZoneMonitor]:
    """One ZoneMonitor per requested zone, all with the same thresholds."""
    return [
        ZoneMonitor(
            z.zone_id,
            entry_grace_frames=cfg.entry_grace_frames,
            exit_grace_frames=cfg.exit_grace_frames,
            track_absence_grace_frames=cfg.track_absence_grace_frames,
            dwell_threshold_sec=cfg.dwell_threshold_sec,
            approach_prediction_sec=cfg.approach_prediction_sec,
            approach_grace_frames=cfg.approach_grace_frames,
            approach_min_speed_px_per_sec=cfg.approach_min_speed_px_per_sec,
            approach_velocity_window_sec=cfg.approach_velocity_window_sec,
            approach_cooldown_sec=cfg.approach_cooldown_sec,
        )
        for z in zones
    ]
