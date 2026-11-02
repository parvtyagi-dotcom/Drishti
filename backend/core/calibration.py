"""Ground calibration: from pixels in a camera's picture to metres on the ground.

A camera sees in perspective - someone near it crosses many pixels a second,
the same walk far away only a few - so any rule written in pixels ("faster
than 400 px/s", "stayed within 60 px") means something different in every
part of the frame and on every camera. Calibrating once fixes that.

The operator marks the four corners of a rectangle lying flat on the ground
(a parking bay, a stretch of road between two markings, a patch of fence
line), in order around it, and types its real width (first corner to
second) and depth (second corner to third). Those four point pairs define a
homography - a plane-to-plane perspective mapping - that sends any point on
the ground plane in the picture to its position in metres. It is exact for
points on the ground, which is why tracks are mapped by their footpoint
(bottom-centre of the box), not their centre.

Points are stored normalized (0-1 of the frame's width and height), like
zones, so a calibration drawn on a snapshot holds at whatever resolution the
stream is analysed.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from backend.config import DATA_DIR

CALIBRATION_DIR = DATA_DIR / "calibrations"
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class CalibrationError(ValueError):
    pass


@dataclass
class GroundCalibration:
    points: list[tuple[float, float]]  # 4 normalized image points, in order around the rectangle
    width_m: float  # real distance points[0] -> points[1]
    depth_m: float  # real distance points[1] -> points[2]
    _homography: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if len(self.points) != 4:
            raise CalibrationError("mark exactly 4 corners")
        if not all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in self.points):
            raise CalibrationError("corners must lie inside the picture")
        if not (0.1 <= self.width_m <= 10_000 and 0.1 <= self.depth_m <= 10_000):
            raise CalibrationError("width and depth must be between 0.1 m and 10 km")
        src = np.array(self.points, dtype=np.float32)
        if not cv2.isContourConvex(src.reshape(-1, 1, 2)) or cv2.contourArea(src) < 1e-4:
            raise CalibrationError(
                "the corners must go around the rectangle in order, and must not be "
                "squeezed into a line or crossed"
            )
        dst = np.array([(0, 0), (self.width_m, 0), (self.width_m, self.depth_m), (0, self.depth_m)], dtype=np.float32)
        self._homography = cv2.getPerspectiveTransform(src, dst)

    def to_ground(self, point_px: tuple[float, float], frame_w: int, frame_h: int) -> tuple[float, float]:
        """A pixel position in a frame_w x frame_h picture -> metres on the
        ground plane, in the rectangle's own coordinates."""
        p = np.array([[[point_px[0] / frame_w, point_px[1] / frame_h]]], dtype=np.float64)
        x, y = cv2.perspectiveTransform(p, self._homography.astype(np.float64))[0, 0]
        return float(x), float(y)

    def to_dict(self) -> dict:
        return {"points": [list(p) for p in self.points], "width_m": self.width_m, "depth_m": self.depth_m}

    @staticmethod
    def from_dict(d: dict) -> "GroundCalibration":
        return GroundCalibration(
            points=[(float(x), float(y)) for x, y in d["points"]],
            width_m=float(d["width_m"]), depth_m=float(d["depth_m"]),
        )


def _path(camera_id: str) -> Path:
    if not _SAFE_ID.match(camera_id):
        raise CalibrationError("bad camera id")
    return CALIBRATION_DIR / f"{camera_id}.json"


def save_calibration(camera_id: str, calibration: GroundCalibration) -> None:
    path = _path(camera_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(calibration.to_dict()), encoding="utf-8")
    tmp.replace(path)


def load_calibration(camera_id: str) -> GroundCalibration | None:
    """None when the camera is not calibrated, or its file is unreadable."""
    try:
        return GroundCalibration.from_dict(json.loads(_path(camera_id).read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def delete_calibration(camera_id: str) -> bool:
    try:
        _path(camera_id).unlink()
        return True
    except FileNotFoundError:
        return False
