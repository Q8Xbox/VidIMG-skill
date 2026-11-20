"""Read and write image files by any path.

OpenCV's own imread / imwrite open files with the Windows code page: a photo in a folder with an Arabic name (or any
name outside that code page) reads as None and is written nowhere, without an error. Python opens such paths fine,
so the bytes go through Python and OpenCV only decodes and encodes them. EXIF orientation, bit depth and the flags
behave exactly as with imread / imwrite."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread(path, flags: int = cv2.IMREAD_COLOR):
    """Like cv2.imread: the image, or None when it cannot be read."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite(path, img, params=None) -> bool:
    """Like cv2.imwrite: the format follows the file extension; True when the file was written."""
    path = Path(path)
    try:
        ok, buf = cv2.imencode(path.suffix or ".png", img, list(params or []))
        if not ok:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        buf.tofile(str(path))
        return True
    except (cv2.error, OSError):
        return False
