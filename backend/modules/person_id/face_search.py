"""Face search in person crops for the live/forensic person ID loop: which
way up to look, whether a face may veto the target, and how a found face is
shown on the output video. Kept out of orchestrator.py so the per-frame loop
stays readable; everything here is cheap pure logic over a FaceDetector."""
from __future__ import annotations

from dataclasses import dataclass

import cv2

from backend.config import PersonIDConfig
from backend.core.output import GREEN, GREY, YELLOW
from backend.core.target_lock import TargetLock
from backend.modules.person_id.recognizer import FACE_ROTATIONS, is_upright, upright_face_with_rotation


# A person crop shorter than this is enlarged before looking for a face: a
# distant person's face is only a few dozen pixels and the detector misses it.
_FACE_UPSCALE_BELOW_PX = 240


@dataclass
class _FaceSighting:
    rel_bbox: tuple[int, int, int, int]  # face box relative to the person box's top-left when found
    similarity: float  # best similarity to the reference photos
    timestamp_sec: float


class _FaceOrientation:
    """Per-run memory of which way up faces are in this camera's crops.

    Searching all four rotations of every person crop (what reference photos
    need, since phones lose orientation metadata) cost four face detections
    per crop on every sample. A camera is either upright or not for a whole
    run, so the per-frame search tries ONE orientation - unrotated, or the
    rotation that last worked once the camera is known to be sideways - and
    only falls back to the others:

      - upright camera (the default): on every PROBE_EVERY_MISSES-th crop
        without an upright face, the other three rotations are tried once.
        That is how a sideways camera is discovered; until then a miss costs
        one detection, not four.
      - sideways camera (a probe found an upright face only after turning):
        the learned rotation is tried first, the rest only on a miss.
    """

    PROBE_EVERY_MISSES = 20

    def __init__(self, exhaustive: bool = False) -> None:
        # exhaustive: try all four rotations on every crop, as Forensic Mode
        # always has. A recording is analysed once, so missing the first faces
        # of a sideways clip while the angle is learned is not worth the time
        # saved; live, where the cost is paid every second, it is.
        self.exhaustive = exhaustive
        self.sideways = False
        self.rotation: int | None = None
        self._misses = 0

    def find(self, detector, image):
        """Returns (image the face was found in, face or None, rotation)."""
        if self.exhaustive:
            return upright_face_with_rotation(image, detector, FACE_ROTATIONS)
        first = self.rotation if self.sideways else None
        turned = image if first is None else cv2.rotate(image, first)
        face = detector.best(turned)
        if face is not None and (first is not None or is_upright(face)):
            return turned, face, first
        if not self.sideways:
            self._misses += 1
            if self._misses % self.PROBE_EVERY_MISSES:
                return turned, face, first
        others = tuple(r for r in FACE_ROTATIONS if r != first)
        turned2, face2, rotation2 = upright_face_with_rotation(image, detector, others)
        if face2 is not None and is_upright(face2):
            self.rotation, self.sideways = rotation2, rotation2 is not None
            return turned2, face2, rotation2
        if face is not None:
            return turned, face, first
        return turned2, face2, rotation2


class _VetoStreaks:
    """Consecutive face vetoes per track; any non-veto face resets it.

    A streak expires window_sec after its last veto (the face-veto period),
    independent of how long a face box stays on screen. Tying it to the 1 s
    display hold meant that at live rates - one face sample a second or less -
    two vetoes in a row could never happen, so a lock that had gone onto the
    wrong person was never released.
    """

    def __init__(self, window_sec: float | None = None) -> None:
        self.window_sec = window_sec
        self._streaks: dict[int, tuple[int, float | None]] = {}

    def _expired(self, last: float | None, now: float | None) -> bool:
        return (self.window_sec is not None and last is not None and now is not None
                and now - last > self.window_sec)

    def record(self, track_id: int, vetoed: bool, now: float | None = None) -> int:
        streak, last = self._streaks.get(track_id, (0, None))
        if self._expired(last, now):
            streak = 0
        streak = streak + 1 if vetoed else 0
        self._streaks[track_id] = (streak, now)
        return streak

    def prune(self, now: float) -> None:
        """Drop expired streaks, so a long live run does not keep one per
        track id it has ever seen."""
        for tid in [t for t, (_, last) in self._streaks.items() if self._expired(last, now)]:
            del self._streaks[tid]

    def forget(self, track_id: int) -> None:
        self._streaks.pop(track_id, None)


def _face_veto_applies(width_px: float, best_sim: float, backend: str, reject_threshold: float,
                       cfg: PersonIDConfig) -> bool:
    """Whether a face is clear enough, and far enough from every reference,
    to say "not the target". ArcFace separates people well enough to judge a
    40px face; SFace (the fallback) put the real target under its 0.25 reject
    line on small CCTV faces, so with SFace only a large face may veto."""
    if best_sim >= reject_threshold:
        return False
    min_width = cfg.face_reject_min_width_px if backend == "arcface" else cfg.face_reject_min_width_px_sface
    return width_px >= min_width


def _face_box_style(track_id: int, similarity: float, lock: TargetLock):
    """(color, label, thickness) for a found face on track_id."""
    if lock.confirmed and track_id == lock.target_track_id:
        return GREEN, f"MATCH {similarity:.2f}", 2
    if lock.candidate_count(track_id) > 0:
        return YELLOW, f"CANDIDATE {similarity:.2f}", 2
    return GREY, None, 1


def _candidate_body_label(count: int, needed: int) -> str:
    return f"CANDIDATE {min(count, needed)}/{needed}"


def _face_bbox_in_crop(bbox, rotation, shape, scale: float) -> tuple[int, int, int, int]:
    """Maps a face box found in a (possibly enlarged, then rotated) crop back
    onto the original crop. shape is (h, w) of the enlarged, UNrotated crop;
    rotation the cv2 code it was turned by (None: not turned)."""
    x1, y1, x2, y2 = bbox
    h, w = shape
    if rotation == cv2.ROTATE_90_CLOCKWISE:
        x1, y1, x2, y2 = y1, h - x2, y2, h - x1
    elif rotation == cv2.ROTATE_180:
        x1, y1, x2, y2 = w - x2, h - y2, w - x1, h - y1
    elif rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
        x1, y1, x2, y2 = w - y2, x1, w - y1, x2
    return tuple(int(round(v / scale)) for v in (x1, y1, x2, y2))


def _locate_face(detector, crop, orientation: _FaceOrientation | None = None):
    """The face in a person crop: (image the face was found in, face, face
    box in the original crop's pixels) or (image, None, None).

    Small crops are enlarged first. With an orientation (the per-frame path)
    only the orientation it expects is searched - see _FaceOrientation; with
    none, all four rotations are (a crop of unknown orientation).
    face.width_px is the face's width in the original crop's pixels.
    """
    scale = 1.0
    if crop.shape[0] < _FACE_UPSCALE_BELOW_PX:
        scale = min(3.0, _FACE_UPSCALE_BELOW_PX / max(1, crop.shape[0]))
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if orientation is None:
        image, face, rotation = upright_face_with_rotation(crop, detector)
    else:
        image, face, rotation = orientation.find(detector, crop)
    if face is None:
        return image, None, None
    face.width_px = (face.bbox[2] - face.bbox[0]) / scale
    return image, face, _face_bbox_in_crop(face.bbox, rotation, crop.shape[:2], scale)


def _find_face(detector, crop):
    """The face in a person crop from a camera mounted at any angle: every
    rotation is tried. Returns (image the face was found in, face)."""
    image, face, _ = _locate_face(detector, crop)
    return image, face
