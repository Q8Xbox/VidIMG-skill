"""Face detection (YuNet) and identity embeddings (SFace), OpenCV zoo models in tools/face-models.

Benchmark (research/bench/RESULTS.md A): SFace cosine separated faithful restorations (0.85-0.89)
from face-destroying ones (0.77-0.81) where face SSIM could not (0.82-0.84 for both).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from . import config

DET = config.FACE_MODELS_DIR / "face_detection_yunet_2023mar.onnx"
REC = config.FACE_MODELS_DIR / "face_recognition_sface_2021dec.onnx"


@dataclass(frozen=True)
class Face:
    box: tuple[float, float, float, float]   # x, y, w, h in image pixels
    score: float
    raw: np.ndarray                          # YuNet row (box + 5 landmarks + score) for alignment

    @property
    def center(self) -> np.ndarray:
        x, y, w, h = self.box
        return np.array([x + w / 2, y + h / 2])

    @property
    def landmarks(self) -> np.ndarray:
        return self.raw[4:14].reshape(5, 2)


@lru_cache(maxsize=1)
def _recognizer():
    return cv2.FaceRecognizerSF.create(str(REC), "")


def available() -> bool:
    return DET.exists() and REC.exists()


def detect(img: np.ndarray, score: float = 0.6, max_side: int = 2048) -> list[Face]:
    """Detect faces; large images are analysed downscaled and boxes mapped back."""
    h, w = img.shape[:2]
    k = min(1.0, max_side / max(h, w))
    small = cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else img
    sh, sw = small.shape[:2]
    det = cv2.FaceDetectorYN.create(str(DET), "", (sw, sh), score, 0.3, 5000)
    _, rows = det.detect(small)
    faces = []
    for r in ([] if rows is None else rows):
        r = r.copy()
        r[:14] /= k
        faces.append(Face((float(r[0]), float(r[1]), float(r[2]), float(r[3])), float(r[14]), r))
    return faces


def embedding(img: np.ndarray, face: Face) -> np.ndarray:
    rec = _recognizer()
    return rec.feature(rec.alignCrop(img, face.raw))


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(_recognizer().match(a, b, cv2.FaceRecognizerSF_FR_COSINE))


def match(ref_faces: list[Face], faces: list[Face]) -> list[Face | None]:
    """For each reference face, the detected face whose centre lies within half its width."""
    out: list[Face | None] = []
    for f in ref_faces:
        best = min(faces, key=lambda g: np.linalg.norm(g.center - f.center), default=None)
        ok = best is not None and np.linalg.norm(best.center - f.center) < f.box[2] / 2
        out.append(best if ok else None)
    return out
