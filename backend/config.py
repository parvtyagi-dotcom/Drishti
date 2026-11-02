"""Central configuration: paths, device detection, per-module defaults.

All paths are resolved relative to the repo root (parent of backend/), so the
process can be launched from any working directory.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
UPLOADS_DIR = REPO_ROOT / "uploads"
OUTPUTS_DIR = REPO_ROOT / "outputs"
REFERENCES_DIR = REPO_ROOT / "references"
ZONES_DIR = REPO_ROOT / "zones"
JOBS_DIR = REPO_ROOT / "jobs"
EVIDENCE_DIR = REPO_ROOT / "outputs" / "evidence"
DATA_DIR = REPO_ROOT / "data"
EVENT_DB_PATH = DATA_DIR / "events.db"

for _d in (MODELS_DIR, UPLOADS_DIR, OUTPUTS_DIR, REFERENCES_DIR, ZONES_DIR, JOBS_DIR, EVIDENCE_DIR, DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

YUNET_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
# Optional, stronger recognizer (174 MB, not in git): scripts/download_face_model.py
ARCFACE_MODEL_PATH = MODELS_DIR / "arcface_w600k_r50.onnx"
OPENH264_DLL_PATH = MODELS_DIR / "openh264-1.8.0-win64.dll"
# Identity models: bigger replacements were measured and rejected (Oct
# 2026, smoke_person_id clips, pairs labelled by track id: same track >=10-15
# frames apart vs different tracks in one frame).
# - ReID, OSNet x1_0 (Qualcomm AI Hub ONNX, Market-1501, MIT): AUC 0.965 vs
#   0.962. At the code's own thresholds, matched for false-match rate: local
#   reacquire found 0.929 vs 0.926 of same-person pairs, cross-camera 0.879
#   vs 0.826 - but those are same-camera pairs, and it costs ~11 ms per crop
#   against ~2.6, on every tracked person every few frames.
# - Face, AuraFace glintr100 (ResNet100 ArcFace, Apache-2.0, 261 MB): worse
#   on these small CCTV faces - AUC 0.865 vs SFace's 0.910, 0.636 vs 0.705 of
#   same-person pairs at a 1% false-match rate - and 116 vs 39 ms per face.
# Re-test on real labelled footage (two cameras, night) before revisiting.
OSNET_MODEL_PATH = MODELS_DIR / "osnet_x0_25_msmt17.onnx"
ZERO_DCE_MODEL_PATH = MODELS_DIR / "zero_dce_pp.pth"

# OpenCV's FFmpeg plugin encodes browser-playable H.264 through Cisco's
# OpenH264, which it loads by bare file name (openh264-1.8.0-win64.dll) from
# the Windows DLL search path - and models/ was never on it. The load failed,
# FFmpeg logged a misleading "Incorrect library version loaded", and every
# H.264 write fell back to the next codec. The bundled DLL is the right
# version; it only has to be findable. PATH, not os.add_dll_directory: the
# plugin calls plain LoadLibrary, which ignores add_dll_directory.
if os.name == "nt" and OPENH264_DLL_PATH.is_file():
    _path = os.environ.get("PATH", "")
    if str(MODELS_DIR) not in _path.split(os.pathsep):
        os.environ["PATH"] = str(MODELS_DIR) + os.pathsep + _path

LOG_LEVEL = logging.INFO


def detect_device() -> str:
    """Return 'cuda' if a usable GPU is present, else 'cpu'. Never raises."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


DEVICE = detect_device()

# Person/vehicle detector. YOLO11n replaced YOLOv8n; on a CPU it runs as an
# OpenVINO INT8 export calibrated on frames from the demo clips. Measured
# with scripts/benchmark_detector.py on a Ryzen 5 5500U, mean over the six
# smoke clips, through the real tracker: 73.5 ms/frame against 116.1 for
# the old yolov8n.pt (1.6x), the same mean confidence (0.724 vs 0.720), and
# fewer distinct track ids on 5 of 6 clips, i.e. fewer broken tracks. The
# FP32 OpenVINO export was barely faster than PyTorch on this CPU (103 ms),
# so INT8 is where the gain is. The plain .pt is the fallback for a GPU
# (OpenVINO here is CPU-only) or a machine without openvino installed.
YOLO_PT_MODEL_PATH = MODELS_DIR / "yolo11n.pt"
YOLO_OPENVINO_MODEL_PATH = MODELS_DIR / "yolo11n_int8_openvino_model"


def pick_yolo_model(device: str = DEVICE) -> Path:
    """The detector to load: DRISHTI_YOLO_MODEL if set, else the OpenVINO
    INT8 export on a CPU with openvino installed, else yolo11n.pt."""
    override = os.environ.get("DRISHTI_YOLO_MODEL")
    if override:
        path = Path(override)
        return path if path.is_absolute() else MODELS_DIR / path
    if device == "cpu" and YOLO_OPENVINO_MODEL_PATH.is_dir():
        import importlib.util

        if importlib.util.find_spec("openvino") is not None:
            return YOLO_OPENVINO_MODEL_PATH
    return YOLO_PT_MODEL_PATH


# A .pt file, or an export directory (OpenVINO) - check with .exists().
YOLO_MODEL_PATH = pick_yolo_model()


@dataclass
class VideoEngineConfig:
    # Codec fourcc candidates tried in order when opening the output writer.
    output_codec_candidates: tuple[str, ...] = ("avc1", "H264", "mp4v")


@dataclass
class TrackerConfig:
    model_path: Path = YOLO_MODEL_PATH
    device: str = DEVICE
    tracker_yaml: str = "bytetrack.yaml"
    confidence: float = 0.35
    iou: float = 0.5
    # COCO class ids this tracker instance restricts itself to.
    person_class_id: int = 0
    vehicle_class_ids: tuple[int, ...] = (2, 3, 5, 7)  # car, motorcycle, bus, truck

    # Run full YOLO+ByteTrack inference every Nth frame; on the frames in
    # between, track boxes are extrapolated from recent velocity instead of
    # re-running the model (see core.tracker.ObjectTracker). This is the
    # dominant cost in every pipeline (benchmarked ~75-90% of per-frame
    # time), so it's the highest-leverage performance knob in the system.
    # Default is 5. It was 3, chosen when propagation between detections
    # moved a box a whole stride's worth of motion per frame - so every extra
    # frame skipped made the box visibly worse, and 3 was as far as it could
    # be pushed. With velocity measured per frame (see core/tracker.py) the
    # box tracks the object correctly whatever the stride, which makes the
    # setting a pure speed choice again. Measured on the person_id smoke clip:
    # 7.5 fps at stride 3, 11.7 at stride 5 - 56% faster, the same
    # TARGET_CONFIRMED, and box jitter unchanged (0.83px mean vs 0.89px).
    # Beyond 5 the returns flatten while the gaps keep growing. Pass
    # detection_stride=1 explicitly for maximum-accuracy runs where every
    # frame should be truly detected, not propagated.
    detection_stride: int = 5

    # Adaptive stride: detect on every frame, ignoring detection_stride,
    # while anything is near a restricted zone or a target is locked - the
    # moments where an extrapolated box could miss the frame a line is
    # crossed. An empty or quiet scene keeps the cheap stride. "Near" is a
    # footpoint inside the zone, predicted to reach it (ZONE_APPROACH), or
    # within near_zone_frac of the frame diagonal of its edge.
    adaptive_stride: bool = True
    near_zone_frac: float = 0.15
    # Whether a locked target also makes the person tracker (YOLO + BoT-SORT
    # with an OSNet embedding per person) detect every frame. Live turns it
    # off: it already runs face and ReID on every analysed frame, and with
    # two webcams on Person ID the every-frame person detection pushed the
    # server to ~76% of a 12-thread CPU and starved one feed to ~3 fps.
    adaptive_person_stride: bool = True


@dataclass
class PersonIDConfig:
    """Target-person identification & tracking configuration.

    min_reference_votes is a *ceiling* configured for the multi-reference
    case. The pipeline adapts it down (never up) to the number of usable
    reference photos actually available, so a single-reference upload never
    crashes or becomes unsatisfiable. See PersonIDConfig.resolve_votes().
    """

    similarity_threshold: float = 0.42
    min_reference_votes: int = 2
    # Number of separate matching observations (frames) required before a
    # track is promoted to CONFIRMED target.
    confirmation_observations: int = 2
    # Run face detection/recognition on person crops every N frames
    # (periodic sampling - the expensive step).
    face_sample_stride: int = 5
    # How many consecutive frames the target's track id may be absent before
    # we consider it "lost" and eligible for re-acquisition.
    gap_tolerance_frames: int = 45
    # How many consecutive frames of PURE Kalman prediction (no real
    # detection corroborating it) to keep drawing the box for. Deliberately
    # much shorter than gap_tolerance_frames: a linear-velocity prediction is
    # trustworthy for bridging a brief occlusion (someone walks in front of
    # the target for a few frames), but empirically drifts onto an unrelated
    # person's position after the target has genuinely left frame and isn't
    # coming back on that trajectory (measured on real footage: visible drift
    # by ~10 frames after last real detection). Beyond this window the box is
    # hidden (not fabricated) until the SAME track reappears or ReID/face
    # re-associates a new one -- gap_tolerance_frames still governs how long
    # that re-acquisition window stays open.
    max_prediction_frames: int = 12
    # Padding (fraction of bbox size) added around a person box when
    # cropping for face detection, since faces sit near the top of the box.
    crop_pad_frac: float = 0.15

    # --- Person ReID (BoT-SORT + OSNet) -- see backend/modules/person_id/person_reid.py ---
    # Run OSNet embedding on sampled person tracks every N frames -- like
    # face_sample_stride, this is the expensive step, run periodically not
    # every frame. Independent of face_sample_stride: ReID is what maintains
    # identity between face-recognition samples, so it typically wants a
    # shorter interval than face sampling.
    reid_sample_stride: int = 3
    # Minimum cosine similarity against the target's appearance gallery to
    # count as a ReID match. Empirically measured on real footage (this
    # project's own test video): same-person crops ~0.876, different-person
    # ~0.42-0.53 -- 0.65 sits well inside that gap.
    reid_similarity_threshold: float = 0.65
    # Matching a target against how they looked on a *different* camera.
    # Set above reid_similarity_threshold on purpose: a local ReID match is
    # corroborated by continuous tracking on the same view, while a
    # cross-camera match has only clothing and build to go on, under
    # different lighting and angle. Claiming the wrong person crossed a
    # border is far more costly than missing a handover, so this errs
    # towards missing one.
    cross_camera_similarity_threshold: float = 0.78
    # Appearance-only matches (local ReID or cross-camera) needed on the same
    # track to re-acquire a lost target. At reid_similarity_threshold about
    # 1 in 40 different-person comparisons still match (2.7%, measured), and
    # with every person compared several times a second a single match was
    # enough to put the target box on a bystander. A face match still
    # re-acquires immediately.
    reacquire_observations: int = 3
    # Face veto. A face this wide or wider is clear enough to trust a "no":
    # if its best similarity to the reference photos is below
    # face_reject_threshold it is not the target, whatever their clothes look
    # like. SFace on this project's CCTV clips: same person ~0.47 on average,
    # different people ~0.10, so 0.25 is a clear "different person".
    face_reject_threshold: float = 0.25
    face_reject_min_width_px: int = 40
    # How long a vetoed track is barred from appearance matching.
    face_veto_sec: float = 10.0
    # Bounded gallery size -- old embeddings age out rather than growing
    # forever, and a bounded gallery can't be dominated by one bad stretch.
    gallery_max_size: int = 30
    # Only add an observation to the target's appearance gallery when the
    # bbox is at least this many pixels on a side -- a tiny/distant crop's
    # embedding is unreliable and would risk polluting the gallery (identity
    # drift protection, section 12 of the plan).
    min_bbox_side_for_gallery: int = 60
    # Combined target score weights (section 13) -- configurable rather than
    # hardcoded so the balance between signals can be tuned without touching
    # pipeline code. Only face_weight/reid_weight are used for the
    # confirm/reacquire decision today; motion_weight/track_weight are
    # exposed for future use (e.g. penalizing a candidate whose track has
    # very little history) without a config/schema change later.
    face_weight: float = 0.6
    reid_weight: float = 0.4
    motion_weight: float = 0.0
    track_weight: float = 0.0

    @staticmethod
    def resolve_votes(configured_min_votes: int, usable_reference_count: int) -> int:
        """Adapt min_reference_votes to the number of usable references.

        - 0 usable references is a hard error, raised by the caller before
          this is even reached.
        - 1 usable reference -> 1 vote required (can't ask for 2 votes from
          1 reference).
        - >=2 usable references -> min(configured_min_votes, usable_count),
          preserving the intended multi-reference voting behavior.
        """
        if usable_reference_count <= 0:
            raise ValueError("resolve_votes requires at least 1 usable reference")
        return max(1, min(configured_min_votes, usable_reference_count))


@dataclass
class AnprConfig:
    # Skip plate localization/OCR on vehicle crops smaller than this - too
    # small to contain a legible plate no matter how much it's upscaled.
    min_vehicle_width: int = 100
    min_vehicle_height: int = 80

    # Run plate localization + OCR on active vehicle tracks every N frames
    # (the expensive step - periodic sampling, same idea as PersonIDConfig's
    # face_sample_stride).
    ocr_sample_stride: int = 5
    # Cap on stored OCR observations per track (bounds memory/compute; old
    # observations are dropped once the cap is hit).
    max_observations_per_track: int = 20
    # A track's leading (exact-text) OCR reading needs at least this many
    # agreeing observations, AND a mean OCR confidence >= confirm_min_mean_confidence
    # (0-100 scale, Tesseract's convention), before it's promoted from a
    # provisional "read" to a "confirmed" plate.
    vote_min_observations: int = 3
    confirm_min_mean_confidence: float = 55.0
    # OCR reads below this confidence aren't even considered as observations -
    # pure noise floor cut before aggregation ever sees them.
    min_read_confidence: float = 25.0

    # Plate crops that fail these never reach OCR - see modules/anpr/
    # quality.py for the measurements behind each. Width: garbage below
    # ~37px, correct reads from 75px. Sharpness is on a contrast-stretched,
    # 48px-high copy; a 9x9-blurred good plate still scores 130+.
    min_plate_width_px: int = 50
    min_plate_aspect: float = 1.4
    max_plate_aspect: float = 7.0
    min_plate_sharpness: float = 50.0

    # Plate search (Mode 2): normalized-string similarity threshold and max
    # length difference tolerated when comparing an OCR read against the
    # user's requested plate. Deliberately conservative - see
    # modules.anpr.ocr.plates_match for why.
    search_match_threshold: float = 0.80  # 2 misreads on a 10-char plate (HR01AJ8585 read as H01AJ858)
    search_match_max_len_diff: int = 2
    # A single matching OCR read isn't enough to lock the red target box onto
    # a vehicle - require this many independently-matching frame observations
    # (mirrors PersonIDConfig.confirmation_observations).
    search_confirmation_observations: int = 2
    # Tolerate this many consecutive frames without the target vehicle's
    # track before treating it as lost (occlusion/temporary detection gap).
    gap_tolerance_frames: int = 45


@dataclass
class ZoneConfig:
    # Consecutive matching observations required before committing a state
    # transition - absorbs boundary jitter (entry) and brief false-outside
    # blips (exit) without spamming events.
    entry_grace_frames: int = 3
    exit_grace_frames: int = 15
    # Consecutive frames a previously-INSIDE track may go undetected
    # (occlusion/tracker gap) before it's treated as a real exit.
    track_absence_grace_frames: int = 45
    dwell_threshold_sec: float = 15.0
    # "footpoint" (bbox bottom-center, approximates ground contact) or
    # "centroid" (bbox center) - which point is tested against the zone.
    point_mode: str = "footpoint"
    # How long an ENTRY/EXIT/LONG_DWELL banner stays on screen, in seconds.
    banner_display_sec: float = 2.0

    # --- approach prediction (early warning before the boundary is crossed) ---
    #
    # A crossing alarm only ever tells an operator what has already happened.
    # Extrapolating a track's recent motion gives them the seconds before it,
    # which is the difference between responding and recording. All spatial
    # values are in PIXELS of the processing resolution, like BehaviorConfig,
    # so they are resolution-dependent by nature.
    #
    # How far ahead the straight-line prediction looks. Longer sees intent
    # earlier but extrapolates further from real evidence, so it warns on
    # paths that curve away; shorter warns only when entry is nearly certain.
    approach_prediction_sec: float = 2.5
    # Consecutive frames the prediction must agree before warning, the same
    # debounce idea as entry_grace_frames. Low, because a warning is cheap and
    # late warnings are worthless.
    approach_grace_frames: int = 2
    # Motion below this is tracker jitter, not approach. Without it a
    # stationary person's box wobble extrapolates into a phantom approach.
    approach_min_speed_px_per_sec: float = 12.0
    # Baseline for measuring velocity. Spanning several frames rather than
    # differencing consecutive ones keeps detector noise out of the heading -
    # the same reason the tracker measures its own velocity over a span. It
    # must also stay longer than the gap between analysed frames: live
    # analysis on CPU delivers 1-3fps, so a window shorter than a second
    # would never hold two samples to measure between.
    approach_velocity_window_sec: float = 1.5
    # Minimum gap before the same track can raise another approach warning.
    # Someone working along a fence line should not re-warn every second.
    approach_cooldown_sec: float = 20.0


@dataclass
class BehaviorConfig:
    """All spatial thresholds are in PIXELS of the video's own processing
    resolution - they are therefore resolution-dependent by nature (a 60px
    loiter radius means something different on a 320-wide crop than a
    1920-wide frame). Defaults are tuned for the ~960px-wide clips used in
    this project's smoke tests; a deployment processing very different
    resolutions should override these.

    Timing uses SOURCE VIDEO timestamps (frame.timestamp_sec) throughout,
    never wall-clock processing time, so results don't depend on how fast
    the machine happens to process frames.
    """
    loitering_seconds: float = 10.0
    loitering_radius_px: float = 60.0

    direction_change_degrees: float = 70.0  # within the requested 60-90 range
    reversal_angle_degrees: float = 120.0  # a "reversal" is a near-opposite turn, not just any sharp turn

    speed_threshold_px_per_sec: float = 400.0
    acceleration_threshold_px_per_sec: float = 500.0  # |delta speed| between consecutive segments

    reversal_count: int = 3
    reversal_window_sec: float = 20.0
    reversal_radius_px: float = 90.0

    behavior_cooldown_seconds: float = 8.0

    # A movement "segment" (used for direction/speed) is only computed once
    # displacement since the last checkpoint clears this floor - filters
    # per-frame detector/tracker jitter from being read as real movement.
    min_displacement_px: float = 8.0
    # Minimum spacing between segment checkpoints; if displacement stays
    # below min_displacement_px the checkpoint simply doesn't advance yet
    # (so the next check spans a longer, less noisy baseline).
    sample_interval_sec: float = 0.3

    # How long a behavior alert stays visually active on the output video.
    alert_display_sec: float = 2.0

    # Metric thresholds, used instead of the pixel ones on a camera with a
    # ground calibration (core/calibration.py), where positions arrive in
    # metres. Same meaning everywhere in the picture and on every camera.
    # People: brisk walking is ~1.5-2 m/s, jogging starts near 2.5.
    loitering_radius_m: float = 3.0
    speed_threshold_m_per_sec: float = 3.0
    acceleration_threshold_m_per_sec: float = 2.0
    reversal_radius_m: float = 4.0
    min_displacement_m: float = 0.3
    # Vehicles get their own limits - 3 m/s would flag every moving car.
    vehicle_loitering_radius_m: float = 5.0
    vehicle_speed_threshold_m_per_sec: float = 16.7  # 60 km/h
    vehicle_acceleration_threshold_m_per_sec: float = 5.0
    vehicle_reversal_radius_m: float = 8.0
    vehicle_min_displacement_m: float = 1.0
    # "px" or "m": which unit the *_px fields hold. Set by in_metres().
    units: str = "px"

    def in_metres(self, vehicles: bool = False) -> "BehaviorConfig":
        """This config with its distance thresholds swapped for the metric
        ones, for a monitor fed ground positions. The *_px fields then hold
        metres (units says so); every timing and angle setting is kept."""
        from dataclasses import replace

        if vehicles:
            return replace(
                self, units="m", loitering_radius_px=self.vehicle_loitering_radius_m,
                speed_threshold_px_per_sec=self.vehicle_speed_threshold_m_per_sec,
                acceleration_threshold_px_per_sec=self.vehicle_acceleration_threshold_m_per_sec,
                reversal_radius_px=self.vehicle_reversal_radius_m, min_displacement_px=self.vehicle_min_displacement_m,
            )
        return replace(
            self, units="m", loitering_radius_px=self.loitering_radius_m,
            speed_threshold_px_per_sec=self.speed_threshold_m_per_sec,
            acceleration_threshold_px_per_sec=self.acceleration_threshold_m_per_sec,
            reversal_radius_px=self.reversal_radius_m, min_displacement_px=self.min_displacement_m,
        )


@dataclass
class LowLightConfig:
    """Automatic low-light detection + enhancement.

    Detection runs on a downsampled copy of every frame (cheap: a resize
    plus a LAB conversion and two numpy reductions) and only frames
    classified LOW_LIGHT ever reach the enhancement path - bright footage is
    never touched. All thresholds are on OpenCV's 8-bit LAB L channel
    (0-255), so they carry over as-is regardless of processing resolution.
    """
    enabled: bool = True
    # "fast" (CLAHE + adaptive gamma) | "zero_dce" (Zero-DCE++ curve
    # estimation, see core/zero_dce.py). fast stays the default on the
    # evidence of scripts/benchmark_lowlight.py: on the three dark demo clips
    # Zero-DCE++ took 2-3x as long (46-92 vs 14-49 ms/frame), the detector
    # was less confident on its output on all three, and dark-clip Person ID
    # kept the target on 60 frames against fast's 90. Its output is flatter
    # (no local contrast boost) and on monochrome night-vision footage it
    # adds a colour cast - it was trained on colour photos. It does give
    # more natural colour on colour footage, so it stays selectable.
    # zero_dce falls back to fast, with the reason reported, if its weights
    # or torch are unavailable.
    backend: str = "fast"
    zero_dce_model_path: Path = ZERO_DCE_MODEL_PATH
    # Longer side of the copy the Zero-DCE++ network sees; its per-pixel
    # curve map is scaled back up to the full frame.
    zero_dce_work_dim: int = 320

    # A frame is LOW_LIGHT if its mean L is below this OR its 10th-percentile
    # L (catches large dark regions/shadows even when the overall mean looks
    # moderate) is below shadow_p10_threshold.
    mean_threshold: float = 65.0
    shadow_p10_threshold: float = 25.0
    # Brightness analysis is done on a frame resized so its longer side is
    # at most this many pixels - negligible cost regardless of source
    # resolution.
    sample_max_dim: int = 320

    clahe_clip_limit: float = 2.5
    clahe_tile_size: int = 8
    # Adaptive gamma brightens further only if CLAHE alone left the frame
    # below this target L mean; gamma is solved for algebraically, then
    # clamped to gamma_min so it can never over-brighten into a washed-out,
    # noisy result.
    gamma_target_mean: float = 130.0
    gamma_min: float = 0.4


@dataclass
class AgentConfig:
    """AI analyst agents that turn alarms into briefed incidents.

    Off unless DRISHTI_AGENTS=1: briefing calls a language model, and a post
    that has not chosen one must not start sending incident data anywhere.
    The agents only read events and propose actions; detection and the siren
    never depend on them.
    """
    enabled: bool = field(default_factory=lambda: os.environ.get("DRISHTI_AGENTS") == "1")
    # Cloud model, used whenever it is reachable.
    cloud_model: str = field(default_factory=lambda: os.environ.get("DRISHTI_AGENT_MODEL", "claude-opus-5-5"))
    cloud_effort: str = field(default_factory=lambda: os.environ.get("DRISHTI_AGENT_EFFORT", "medium"))
    cloud_timeout_sec: float = 60.0
    # Optional hosted model with a free tier: groq | gemini | openrouter |
    # custom. Its key is read from the provider's own environment variable
    # (GROQ_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY, DRISHTI_HOSTED_API_KEY).
    hosted_provider: str = field(default_factory=lambda: os.environ.get("DRISHTI_HOSTED_PROVIDER", ""))
    hosted_model: str = field(default_factory=lambda: os.environ.get("DRISHTI_HOSTED_MODEL", ""))
    hosted_url: str = field(default_factory=lambda: os.environ.get("DRISHTI_HOSTED_URL", ""))
    # Local model through Ollama, so briefs keep coming with no internet.
    local_model: str = field(default_factory=lambda: os.environ.get("DRISHTI_LOCAL_MODEL", "qwen2.5:3b"))
    local_url: str = field(default_factory=lambda: os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"))
    local_timeout_sec: float = 180.0
    # After a model fails, try the next one first for this long, so an
    # offline post does not wait on the cloud timeout for every incident.
    retry_after_sec: float = 60.0
    # Tool calls one agent may make before it must answer.
    max_steps: int = 8
    # Events on one camera further apart than this start a new incident.
    incident_gap_sec: float = 60.0
    # Wait this long after the first alarm so the brief sees what followed.
    brief_delay_sec: float = 4.0


@dataclass
class AppConfig:
    video: VideoEngineConfig = field(default_factory=VideoEngineConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    person_id: PersonIDConfig = field(default_factory=PersonIDConfig)
    anpr: AnprConfig = field(default_factory=AnprConfig)
    zone: ZoneConfig = field(default_factory=ZoneConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    lowlight: LowLightConfig = field(default_factory=LowLightConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)


def get_config() -> AppConfig:
    return AppConfig()
