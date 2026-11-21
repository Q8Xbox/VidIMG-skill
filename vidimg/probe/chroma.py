"""Chroma shift: colour planes displaced from the luma (line art), common on tape captures and old
cartoons ("ChromaShift(C=-2)" in videohelp thread 323127). Moving the colour back is a correction of
geometry, not a colour change, so it fits the faithful policy.

Measurement, on the planes as stored (4:2:0, MPEG-2/H.264 "left" siting: chroma column j sits on luma
column 2j, chroma row i between luma rows 2i and 2i+1). The luma is brought to the chroma grid with that
siting ([1 2 1]/4 horizontally around column 2j, mean of the two rows), so a correctly aligned clip measures
0 by construction, not by resampling. Then, per colour plane, the displacement that best lines up its edge
strength with the luma's (normalised cross-correlation over +-SEARCH chroma pixels, parabolic sub-pixel
peak). Only frames with enough colour edges count; the result is the median over frames, with its spread.
Shifts are reported in LUMA pixels (positive = colour sits right of / below the lines).

Baseline: clean sources measure about -0.4 px horizontally (testsrc2 -0.40, a still converted by swscale
-0.37/-0.54, a camera file -0.35, true-interlace test clips -0.39/-0.50): many encoders site chroma centred,
half a luma pixel right of where "left" siting puts it. Shifts are judged and corrected relative to that
baseline (BASE_X), so a corrected clip measures like a clean one. Measured on the owner's samples (judged
by colour maps with the line art drawn over them): a 4 px vertical shift (Sample04) and a 1.7 px horizontal
one (Sample01) are real and visibly fixed; 1 px (Sample06) changes nothing visible, hence SHIFT_MIN 1.25.
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .. import media

SEARCH = 6                  # chroma pixels each way (12 luma pixels horizontally)
CHROMA_EDGE_MIN = 1.5       # mean chroma edge strength (8-bit) for a frame to count (colourless frames don't)
BASE_X, BASE_Y = -0.4, 0.0  # what a clean (centre-sited) source measures, luma pixels
SHIFT_MIN = 1.25            # luma pixels beyond the baseline: below this the colour is considered aligned
SPREAD_MAX = 0.5            # luma pixels: frames must agree this well (median absolute deviation) ...
SPREAD_REL = 0.15           # ... or within this fraction of the shift (Sample04: 0.55 on a 4 px shift)
MIN_FRAMES = 4


@dataclass
class ChromaShift:
    u: tuple[float, float] = (0.0, 0.0)      # (dx, dy) of Cb in luma pixels
    v: tuple[float, float] = (0.0, 0.0)      # (dx, dy) of Cr
    spread: float = 0.0                      # largest median absolute deviation over frames (luma pixels)
    spreads: dict = field(default_factory=dict)   # per plane: [dx, dy] median absolute deviation
    frames: int = 0                          # frames with enough colour edges
    peaks: list[float] = field(default_factory=list)   # correlation peak per frame (Cb, Cr averaged)

    def excess(self) -> dict:
        """Shift beyond the clean baseline, luma pixels."""
        return {k: (round(d[0] - BASE_X, 2), round(d[1] - BASE_Y, 2)) for k, d in (("u", self.u), ("v", self.v))}

    @property
    def largest(self) -> float:
        return max(abs(c) for d in self.excess().values() for c in d)

    def consistent(self, plane: str, axis: int) -> bool:
        """Frames agree on this component (each one is judged on its own: on Sample04 the vertical shift was
        steady, MAD 0.18-0.57 px on 4.1-4.3 px, while the horizontal readings scattered, MAD 1.8 px)."""
        mad = (self.spreads.get(plane) or [self.spread, self.spread])[axis]
        return mad <= max(SPREAD_MAX, SPREAD_REL * abs(self.excess()[plane][axis]))

    def axis_ok(self, axis: int) -> bool:
        """Both planes agree frame to frame on this direction. A direction is corrected for both planes or for
        neither: moving one colour plane alone could make new fringes (a tape's chroma delay moves both)."""
        return self.consistent("u", axis) and self.consistent("v", axis)

    @property
    def is_shifted(self) -> bool:
        ex = self.excess()
        return self.frames >= MIN_FRAMES and any(
            self.axis_ok(i) and max(abs(ex["u"][i]), abs(ex["v"][i])) >= SHIFT_MIN for i in (0, 1))

    def correction(self, horizontal_only: bool = False) -> dict:
        """How far to MOVE each colour plane (luma pixels), to a quarter pixel. Only directions both planes agree
        on (axis_ok); small components stay 0. Interlaced sources: horizontal only (a vertical shift measured on woven frames
        mixes the fields)."""
        ex = self.excess()

        def q(k: str, i: int) -> float:
            x = ex[k][i]
            if abs(x) < SHIFT_MIN / 2 or not self.axis_ok(i) or (i == 1 and horizontal_only):
                return 0.0
            return round(-x * 4) / 4 + 0.0
        return {k: [q(k, 0), q(k, 1)] for k in ("u", "v")}

    def to_dict(self) -> dict:
        return asdict(self) | {"excess": self.excess(), "is_shifted": self.is_shifted}


def luma_on_chroma_grid(y: np.ndarray) -> np.ndarray:
    """Luma resampled onto 4:2:0 left-sited chroma positions."""
    y = y.astype(np.float32)
    rows = 0.5 * (y[0::2] + y[1::2])                                   # vertical: between the two rows
    pad = np.pad(rows, ((0, 0), (1, 1)), mode="edge")
    return 0.25 * pad[:, 0:-2:2] + 0.5 * pad[:, 1:-1:2] + 0.25 * pad[:, 2::2]   # horizontal: on column 2j


def _edges(p: np.ndarray) -> np.ndarray:
    p = cv2.GaussianBlur(p.astype(np.float32), (0, 0), 0.7)
    gx, gy = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(gx, gy)


def _peak(scores: np.ndarray, k: int) -> float:
    if 0 < k < len(scores) - 1:
        a, b, c = scores[k - 1], scores[k], scores[k + 1]
        den = a - 2 * b + c
        if den < 0:
            return k + 0.5 * (a - c) / den
    return float(k)


def displacement(ref: np.ndarray, mov: np.ndarray, search: int = SEARCH) -> tuple[float, float, float]:
    """(dx, dy, peak): how far `mov`'s edges sit from `ref`'s (chroma pixels), by normalised correlation."""
    h, w = ref.shape
    m = search + 2
    r = ref[m:h - m, m:w - m]
    r = (r - r.mean()) / (r.std() + 1e-6)
    sc = np.full((2 * search + 1, 2 * search + 1), -1.0, dtype=np.float64)
    for iy, dy in enumerate(range(-search, search + 1)):
        for ix, dx in enumerate(range(-search, search + 1)):
            s = mov[m + dy:h - m + dy, m + dx:w - m + dx]
            s = (s - s.mean()) / (s.std() + 1e-6)
            sc[iy, ix] = float(np.mean(r * s))
    iy, ix = np.unravel_index(int(np.argmax(sc)), sc.shape)
    return _peak(sc[iy], ix) - search, _peak(sc[:, ix], iy) - search, float(sc[iy, ix])


def measure_frame(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> dict | None:
    """Displacement of Cb and Cr against the luma, in luma pixels; None for frames without colour edges."""
    ey = _edges(luma_on_chroma_grid(y))
    eu, ev = _edges(u), _edges(v)
    if max(eu.mean(), ev.mean()) < CHROMA_EDGE_MIN:
        return None
    ux, uy, up = displacement(ey, eu)
    vx, vy, vp = displacement(ey, ev)
    return {"u": (2 * ux, 2 * uy), "v": (2 * vx, 2 * vy), "peak": (up + vp) / 2}


def read_yuv(src: Path, times: list[float]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """One 8-bit 4:2:0 frame (Y, U, V as stored) at each time."""
    info = media.probe(src)
    w, h = info["width"] // 2 * 2, info["height"] // 2 * 2
    out = []
    for t in times:
        b = subprocess.run([media.ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-ss", f"{t:.3f}",
                            "-i", str(src), "-map", "0:v:0", "-frames:v", "1", "-vf", f"crop={w}:{h}:0:0",
                            "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"], capture_output=True, stdin=subprocess.DEVNULL).stdout
        if len(b) < w * h * 3 // 2:
            continue
        a = np.frombuffer(b[: w * h * 3 // 2], np.uint8)
        y = a[: w * h].reshape(h, w)
        u = a[w * h: w * h * 5 // 4].reshape(h // 2, w // 2)
        v = a[w * h * 5 // 4:].reshape(h // 2, w // 2)
        out.append((y, u, v))
    return out


def summarize(results: list[dict]) -> ChromaShift:
    r = [x for x in results if x]
    if not r:
        return ChromaShift()
    arr = {k: np.array([x[k] for x in r]) for k in ("u", "v")}
    med = {k: tuple(round(float(c), 2) for c in np.median(arr[k], axis=0)) for k in arr}
    mads = {k: [round(float(c), 2) for c in np.median(np.abs(arr[k] - np.median(arr[k], axis=0)), axis=0)] for k in arr}
    return ChromaShift(u=med["u"], v=med["v"], spread=max(max(m) for m in mads.values()), spreads=mads,
                       frames=len(r), peaks=[round(x["peak"], 3) for x in r])


def analyze(src: Path, *, count: int = 12, duration: float | None = None) -> ChromaShift:
    """Measure on `count` frames spread over the video. Interlaced sources are measured on whole frames:
    a horizontal shift is the same in both fields; the vertical one is reported but only applied after
    deinterlacing (see pipeline)."""
    src = Path(src)
    dur = duration if duration is not None else media.probe(src, exact=True)["duration"]
    times = [dur * (i + 0.5) / count for i in range(count)]
    return summarize([measure_frame(*f) for f in read_yuv(src, times)])
