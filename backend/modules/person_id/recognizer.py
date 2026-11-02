"""SFace face recognition (cv2.FaceRecognizerSF) + multi-reference voting.

FaceRecognizerSF.match() returns cosine similarity in the recognizer's own
convention where higher = more similar; OpenCV's documented "same person"
cosine threshold for this model is ~0.363, but this project intentionally
uses a stricter 0.42 default (config.PersonIDConfig.similarity_threshold) to
cut false positives, combined with multi-reference voting.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from backend.modules.person_id.detector import FaceDetection, FaceDetector

logger = logging.getLogger(__name__)


class FaceRecognizer:
    backend = "sface"
    # None: the caller's PersonIDConfig thresholds apply (they were tuned for SFace).
    match_threshold: float | None = None
    reject_threshold: float | None = None

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"SFace model not found: {self.model_path}")
        self._recognizer = cv2.FaceRecognizerSF.create(str(self.model_path), "")

    def embed(self, image: np.ndarray, face: FaceDetection) -> np.ndarray:
        aligned = self._recognizer.alignCrop(image, face.raw)
        return self._recognizer.feature(aligned)

    def similarity(self, feat_a: np.ndarray, feat_b: np.ndarray) -> float:
        return float(self._recognizer.match(feat_a, feat_b, cv2.FaceRecognizerSF_FR_COSINE))


class ArcFaceRecognizer:
    """InsightFace ArcFace (ResNet-50, trained on WebFace600K) - the stronger
    recognizer, used whenever its model has been downloaded
    (scripts/download_face_model.py).

    Measured on real webcam frames against SFace (same YuNet detections): a
    person matched themself in 13/14 frames at a 60 px face against SFace's
    7/14, while the closest different face - a poster portrait behind them -
    scored 0.09 against SFace's 0.28. Its scores sit on a different scale, so
    it carries its own thresholds.
    """

    backend = "arcface"
    match_threshold = 0.30
    # Only a face this far from the target vetoes them; a laughing or turned
    # face of the right person measured as low as 0.02.
    reject_threshold = -0.05
    _TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                          [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)

    def __init__(self, model_path: str | Path):
        import onnxruntime as ort

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ArcFace model not found: {self.model_path}")
        self._session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name

    def embed(self, image: np.ndarray, face: FaceDetection) -> np.ndarray:
        # YuNet's five landmarks come in the same order as ArcFace's template
        # (eyes, nose, mouth corners, image-left first).
        points = face.raw[4:14].reshape(5, 2).astype(np.float32)
        matrix, _ = cv2.estimateAffinePartial2D(points, self._TEMPLATE, method=cv2.LMEDS)
        aligned = cv2.warpAffine(image, matrix, (112, 112), borderValue=0)
        blob = (cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32) - 127.5) / 127.5
        vector = self._session.run(None, {self._input: blob.transpose(2, 0, 1)[None]})[0][0]
        return vector / (np.linalg.norm(vector) + 1e-9)

    def similarity(self, feat_a: np.ndarray, feat_b: np.ndarray) -> float:
        return float(np.dot(feat_a.ravel(), feat_b.ravel()))


def best_face_recognizer(sface_path: str | Path, arcface_path: str | Path):
    """ArcFace when its model is present and loads, SFace otherwise."""
    if Path(arcface_path).exists():
        try:
            return ArcFaceRecognizer(arcface_path)
        except Exception:
            logger.exception("ArcFace failed to load; using SFace")
    return FaceRecognizer(sface_path)


_ROTATIONS = (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE)


REFERENCE_MAX_SIDE = 1280


def upright_face(image: np.ndarray, detector: FaceDetector) -> tuple[np.ndarray, FaceDetection | None]:
    """Finds the face in a reference photo, turning the photo if needed.

    Phone photos often arrive sideways or upside down (orientation metadata
    lost on the way), and the face detector only finds upright faces. Returns
    the image the face was found in, so it can be embedded and kept upright.
    """
    # The detector also finds tilted and even upside-down faces, just with a
    # lower score - so every orientation is tried and the most confident face
    # with level eyes above the mouth wins.
    best, best_key = (image, None), None
    for rotation in _ROTATIONS:
        turned = image if rotation is None else cv2.rotate(image, rotation)
        face = detector.best(turned)
        if face is None:
            continue
        r = face.raw
        upright = abs(r[6] - r[4]) > abs(r[7] - r[5]) and (r[11] + r[13]) / 2 > (r[5] + r[7]) / 2
        key = (upright, float(r[14]))
        if best_key is None or key > best_key:
            best, best_key = (turned, face), key
    return best


@dataclass
class ReferenceEmbedding:
    source_path: Path
    embedding: np.ndarray


class ReferenceSet:
    """Loads reference photos, detects+embeds the face in each, and drops
    unusable photos (no detectable face) rather than crashing.
    """

    def __init__(
        self,
        photo_paths: list[str | Path],
        detector: FaceDetector,
        recognizer: FaceRecognizer,
    ):
        self.detector = detector
        self.recognizer = recognizer
        self.embeddings: list[ReferenceEmbedding] = []
        self.rejected: list[tuple[Path, str]] = []

        for raw_path in photo_paths:
            path = Path(raw_path)
            image = cv2.imread(str(path))
            if image is None:
                self.rejected.append((path, "unreadable image file"))
                continue
            # A full-resolution phone photo (4032x3024) found a face but
            # matched no one in the video, while the same photo at 600px
            # matched at once. Bring it to the scale video faces are seen at.
            scale = REFERENCE_MAX_SIDE / max(image.shape[:2])
            if scale < 1:
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            image, face = upright_face(image, detector)
            if face is None:
                self.rejected.append((path, "no face detected"))
                continue
            embedding = recognizer.embed(image, face)
            self.embeddings.append(ReferenceEmbedding(source_path=path, embedding=embedding))

        if self.rejected:
            logger.warning("ReferenceSet: rejected %d/%d photos: %s",
                            len(self.rejected), len(photo_paths), self.rejected)

    @property
    def usable_count(self) -> int:
        return len(self.embeddings)

    def vote(self, probe_embedding: np.ndarray, threshold: float) -> tuple[int, float]:
        """Compares probe against every usable reference.

        Returns (vote_count, best_similarity) where vote_count is how many
        references matched at/above `threshold`.
        """
        votes = 0
        best = -1.0
        for ref in self.embeddings:
            sim = self.recognizer.similarity(probe_embedding, ref.embedding)
            best = max(best, sim)
            if sim >= threshold:
                votes += 1
        return votes, best
