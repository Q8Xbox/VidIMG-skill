"""analyze_image: a measured degradation profile of one image (PLAN.md section 4.1).

Local measurements (fast, no GPU): resolution, noise sigma, sharpness, JPEG quality from the
quantisation tables, exposure clipping, colour cast, faces and their sizes, anime-likeness.
All measurements are local and free (OpenCV, NumPy, Pillow).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from .. import imgfile  # noqa: E402

from .. import faces

_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)


def noise_sigma(gray: np.ndarray) -> float:
    """Immerkaer (1996) fast noise estimate: sigma of additive Gaussian noise, 8-bit units."""
    h, w = gray.shape
    conv = cv2.filter2D(gray.astype(np.float64), -1, _IMMERKAER, borderType=cv2.BORDER_REFLECT)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6 * (w - 2) * (h - 2)))

# IJG standard luminance quantisation table (quality 50), natural order.
_STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=float)


@dataclass
class FaceInfo:
    count: int = 0
    min_height: int = 0
    max_height: int = 0
    small: bool = False        # smallest face under 96 px tall: candidates for face recovery


@dataclass
class ImageProfile:
    path: str
    width: int
    height: int
    channels: int
    bit_depth: int
    noise_sigma: float                    # estimated Gaussian sigma in 8-bit units
    sharpness: float                      # variance of Laplacian on a 1024-px view (higher = sharper)
    jpeg_quality: int | None              # estimated IJG quality, None if not a JPEG
    clipped_dark: float                   # share of pixels at 0..2
    clipped_bright: float                 # share of pixels at 253..255
    color_cast: float                     # chroma distance of the mean colour from neutral (Lab)
    anime_score: float                    # 0..1 flat-colour/strong-edge likeness
    faces: FaceInfo = field(default_factory=FaceInfo)
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def jpeg_quality(path: Path) -> int | None:
    try:
        with Image.open(path) as im:
            if im.format != "JPEG" or not getattr(im, "quantization", None):
                return None
            table = np.array(im.quantization[0], dtype=float)
    except OSError:
        return None
    # Pillow returns the tables in natural order (verified: exact fit to the IJG scaling, while a
    # zigzag remap misfits by 3-32 levels); the sum ratio is robust to the floor/clamp at 1
    natural = table if len(table) == 64 else np.resize(table, 64)
    scale = float(natural.sum() / _STD_LUMA.sum()) * 100
    q = (200 - scale) / 2 if scale <= 100 else 5000 / scale
    return int(max(1, min(100, round(q))))


def _read(path: Path) -> tuple[np.ndarray, int]:
    raw = imgfile.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"cannot read image {path}")
    depth = 16 if raw.dtype == np.uint16 else 8
    if raw.ndim == 2:
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    elif raw.shape[2] == 4:
        raw = raw[:, :, :3]
    img8 = (raw / 257).astype(np.uint8) if depth == 16 else raw
    return img8, depth


def _view(img: np.ndarray, side: int = 1024) -> np.ndarray:
    h, w = img.shape[:2]
    k = min(1.0, side / max(h, w))
    return cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else img


def _grid_tiles(img: np.ndarray, side: int, n: int = 3) -> list[np.ndarray]:
    h, w = img.shape[:2]
    s = min(side, h, w)
    ys = np.linspace(0, h - s, n).astype(int)
    xs = np.linspace(0, w - s, n).astype(int)
    return [img[y:y + s, x:x + s] for y in ys for x in xs]


FLAT_FRAME_STD = 2.0            # 8-bit levels: below this a picture holds nothing to judge


def anime_score(img: np.ndarray) -> float:
    """Flat colour areas + few distinct colours + strong clean edges -> close to 1."""
    v = _view(img, 512)
    if float(np.std(v)) < FLAT_FRAME_STD:
        # a black or single-colour picture (a fade, a title card background) is "flat, few colours" by definition
        # and scored 0.85: it says nothing about drawn or real
        return 0.0
    q = (v // 16).reshape(-1, 3).astype(np.int32)
    distinct = len(np.unique(q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2])) / 4096
    gray = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY)
    flat = float(np.mean(cv2.Laplacian(gray, cv2.CV_32F) == 0))
    edges = float(np.mean(cv2.Canny(gray, 100, 200) > 0))
    s = 0.45 * (1 - min(1.0, distinct / 0.25)) + 0.4 * min(1.0, flat / 0.5) + 0.15 * min(1.0, edges / 0.08)
    return round(float(s), 3)


def analyze(path: Path) -> ImageProfile:
    path = Path(path)
    img, depth = _read(path)
    h, w = img.shape[:2]
    v = _view(img)
    gray = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY)
    # noise and blur at native scale on a 3x3 grid of tiles: median noise (robust to texture),
    # 80th-percentile sharpness (flat sky or wall tiles would otherwise read as "blurred")
    tiles = [cv2.cvtColor(t, cv2.COLOR_BGR2GRAY) for t in _grid_tiles(img, 512)]
    sigma = float(np.median([noise_sigma(t) for t in tiles]))
    sharp = float(np.percentile([cv2.Laplacian(t, cv2.CV_64F).var() for t in tiles], 80))
    lab = cv2.cvtColor(v, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
    cast = float(np.hypot(lab[:, 1].mean() - 128, lab[:, 2].mean() - 128))
    fi = FaceInfo()
    if faces.available():
        found = faces.detect(img)
        if found:
            heights = [int(f.box[3]) for f in found]
            fi = FaceInfo(len(found), min(heights), max(heights), min(heights) < 96)
    prof = ImageProfile(
        path=str(path), width=w, height=h, channels=3, bit_depth=depth,
        noise_sigma=round(sigma, 2), sharpness=round(sharp, 1), jpeg_quality=jpeg_quality(path),
        clipped_dark=round(float(np.mean(gray <= 2)), 4), clipped_bright=round(float(np.mean(gray >= 253)), 4),
        color_cast=round(cast, 2), anime_score=anime_score(img), faces=fi,
    )
    prof.flags = _flags(prof)
    return prof


def _flags(p: ImageProfile) -> list[str]:
    f = []
    if max(p.width, p.height) < 1000:
        f.append("low_resolution")
    # rough pre-filter only (Immerkaer reads low on JPEG-smoothed noise: noisy sample 2.16 vs
    # clean samples <= 1.1); noisy photos get SCUNet denoising before upscaling
    if p.noise_sigma > 2.0:
        f.append("noisy")
    if p.sharpness < 60:
        f.append("soft_or_blurred")
    if p.jpeg_quality is not None and p.jpeg_quality < 70:
        f.append("heavy_jpeg")
    if p.faces.count:
        f.append("faces")
    if p.faces.small:
        f.append("small_faces")
    # flat walls and white clothing also score high (5 real photos in the person's set did), and
    # anime routes to Video2X processors that damage real faces: a detected real face vetoes it
    if p.anime_score >= 0.6 and p.faces.count == 0:
        f.append("anime_like")
    if p.clipped_bright > 0.02 or p.clipped_dark > 0.05:
        f.append("clipped_exposure")
    return f
