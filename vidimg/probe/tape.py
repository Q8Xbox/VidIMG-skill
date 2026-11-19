"""Tape (VHS / analogue capture) defects that software cannot repair, measured so the report can say so
(videohelp thread 380626): head-switching noise in the bottom lines, "flagging" (the top of the picture
bending sideways, a timing error that needs a line time-base corrector during capture) and hard-clipped
highlights (detail above the capture's white level is gone).

All measurements use the luma as stored, on frames spread through the video. Rows are compared with the row
two above (same field, so interlace cannot pass for a defect):
* head switching: a boundary low in the frame below which the PICTURE ITSELF continues displaced sideways (the
  rows below still match the rows above at a shift of SHIFT_JUMP+ px), at the same height in most frames. A
  content edge does not count: a news ticker and a letterbox caption broke the correlation at a fixed height
  in 100 % / 81 % of frames, but nothing below them matched the picture above at any shift;
* flagging: sideways displacement that keeps one direction as it grows towards the top edge and exceeds what
  the same accumulation gives in every one of several bands through the middle of the picture (a scene's
  diagonals accumulate there too);
* clipped highlights: solid patches at the digital maximum in a limited-range source (white is 235; values at
  253+ are "superwhite" that a capture clipped). Only patches that survive a 7x7 erosion count: superwhite
  lettering on a title card (thin strokes, 0.47 % of the frame) is not a blown highlight.
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .. import media

SEARCH = 48                 # px of sideways search between rows
BOTTOM_ZONE = 0.04          # head switching lives in the bottom 4 % (480: 19 rows, 576: 23): VHS switches heads
                            # about 6.5 lines before vertical sync, so 5-10 lines per field show. A news ticker
                            # 29 rows up (6 %) matched itself at its symbols' period and passed every other test
BAND_MIN_ROWS = 4           # VHS shows at least 5 noisy lines per field; a 2-row break sat inside a ticker
FLAT_STD = 3.0              # a row this flat (black border, blank sky) carries no information
SHIFT_JUMP = 3.0            # px: a row displaced at least this much from the row two above is a break ...
PEAK_MATCH = 0.3            # ... while still matching it at that shift (displaced picture, not a content edge)
HS_SHARE = 0.5              # head switching: a break at a consistent height in at least half the frames
HS_ROW_TOL = 2              # rows: "consistent height"
TOP_ZONE = 0.05             # flagging: the top 5 % of the frame
FLAG_PX = 3.0               # px of accumulated displacement at the top edge, beyond every control band's
FLAG_SIGN = 0.7             # share of the row steps that move the same way (the bend grows steadily)
FLAG_CONTROLS = 5           # control bands spread over 20-80 % of the height
FLAG_SHARE = 0.3            # ... in at least this share of frames
CLIP_LEVEL = 253            # 8-bit luma of a clipped superwhite
CLIP_SHARE = 0.002          # share of pixels at it (0.2 %), in at least CLIP_FRAMES of the frames
CLIP_FRAMES = 0.1           # (bright areas come and go with the scenes; every clean clip measured 0 frames)


@dataclass
class TapeDefects:
    frames: int = 0
    head_switching_rows: int = 0            # rows in the bottom band (0 = none)
    head_switching_share: float = 0.0       # share of frames showing the break at that height
    flagging_share: float = 0.0             # share of frames whose top bends sideways
    flagging_px: float = 0.0                # median displacement at the top edge in those frames
    clipped_share: float = 0.0              # share of frames with clipped highlights
    clipped_pixels: float = 0.0             # median share of pixels at the clip level in those frames
    limited_range: bool = True
    bottom_border_rows: int = 0             # flat rows already at the bottom (a capture's own mask)
    rows_per_frame: list[int] = field(default_factory=list)

    @property
    def head_switching(self) -> bool:
        return self.head_switching_rows >= BAND_MIN_ROWS and self.head_switching_share >= HS_SHARE

    @property
    def flagging(self) -> bool:
        return self.flagging_share >= FLAG_SHARE

    @property
    def clipped(self) -> bool:
        return self.limited_range and self.clipped_share >= CLIP_FRAMES

    def to_dict(self) -> dict:
        return asdict(self) | {"head_switching": self.head_switching, "flagging": self.flagging,
                               "clipped": self.clipped}


def _hp(row: np.ndarray) -> np.ndarray:
    r = row.astype(np.float64)
    k = np.ones(15) / 15
    return r - np.convolve(np.pad(r, 7, mode="edge"), k, mode="valid")


def row_shift(a: np.ndarray, b: np.ndarray, search: int = SEARCH) -> tuple[float, float]:
    """(shift, peak): how far row b is displaced sideways from row a (px, sub-pixel), by normalised
    cross-correlation of their high-passed profiles. peak 1 = identical shape."""
    x, y = _hp(a), _hp(b)
    x, y = x - x.mean(), y - y.mean()
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx < 1e-6 or ny < 1e-6:
        return 0.0, 0.0
    n = len(x)
    c = np.fft.irfft(np.fft.rfft(x, 2 * n) * np.conj(np.fft.rfft(y, 2 * n)), 2 * n)
    c = np.concatenate([c[-search:], c[: search + 1]]) / (nx * ny)      # lags -search..+search
    k = int(np.argmax(c))
    s = float(k)
    if 0 < k < len(c) - 1:
        den = c[k - 1] - 2 * c[k] + c[k + 1]
        if den < 0:
            s = k + 0.5 * (c[k - 1] - c[k + 1]) / den
    return -(s - search), float(c[k])           # positive: b sits right of a


def _flat(row: np.ndarray) -> bool:
    return float(np.std(row.astype(np.float32))) < FLAT_STD


def bottom_break(y: np.ndarray) -> tuple[int, int]:
    """(band rows, flat border rows) for one frame: the highest row in the bottom zone that breaks from the row
    two above it; the band runs from there to the bottom (above any flat border)."""
    h = y.shape[0]
    end = h
    while end > h * (1 - BOTTOM_ZONE) and _flat(y[end - 1]):
        end -= 1
    border = h - end
    top = max(2, int(end - h * BOTTOM_ZONE))
    for r in range(top, end):
        if _flat(y[r]) or _flat(y[r - 2]):
            continue
        s, p = row_shift(y[r - 2], y[r])
        if abs(s) >= SHIFT_JUMP and p >= PEAK_MATCH:
            return end - r, border
    return 0, border


def row_steps(y: np.ndarray, start: int, stop: int) -> list[float]:
    """Sideways step of each row against the row two below it, from row `start` up to row `stop` (stop < start)."""
    steps = []
    for r in range(start, stop, -2):
        if r - 2 < 0:
            break
        if _flat(y[r]) or _flat(y[r - 2]):
            continue
        s, p = row_shift(y[r], y[r - 2])
        if p >= PEAK_MATCH:
            steps.append(s)
    return steps


def top_bend(y: np.ndarray) -> float:
    """Displacement at the top edge beyond what bands of the same height through the middle accumulate; 0 when
    the steps do not keep one direction (a bend grows steadily; picture content goes both ways)."""
    h = y.shape[0]
    n = max(8, int(h * TOP_ZONE)) // 2 * 2
    first = 0
    while first < n and _flat(y[first]):
        first += 1
    if n - first < 6:
        return 0.0
    steps = row_steps(y, n + 1, first)
    if len(steps) < 3:
        return 0.0
    top = sum(steps)
    moved = [s for s in steps if abs(s) >= 0.25]          # whole-pixel bends leave many rows unmoved
    if len(moved) < 3 or sum(1 for s in moved if s * top > 0) / len(moved) < FLAG_SIGN:
        return 0.0
    span = n - first
    control = 0.0
    for k in range(FLAG_CONTROLS):
        mid = int(h * (0.2 + 0.6 * k / max(1, FLAG_CONTROLS - 1))) // 2 * 2 + 1
        control = max(control, abs(sum(row_steps(y, mid + span, mid))))
    return max(0.0, abs(top) - control)


def read_luma(src: Path, times: list[float]) -> list[np.ndarray]:
    info = media.probe(src)
    w, h = info["width"], info["height"]
    out = []
    for t in times:
        raw = subprocess.run([media.ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-ss", f"{t:.3f}",
                              "-i", str(src), "-map", "0:v:0", "-frames:v", "1", "-vf",
                              "format=yuv420p16le,extractplanes=y", "-pix_fmt", "gray16le", "-f", "rawvideo", "-"],
                             capture_output=True).stdout
        if len(raw) >= w * h * 2:
            out.append((np.frombuffer(raw[: w * h * 2], "<u2").reshape(h, w) >> 8).astype(np.uint8))
    return out


def summarize(frames: list[np.ndarray], limited_range: bool = True) -> TapeDefects:
    d = TapeDefects(frames=len(frames), limited_range=limited_range)
    if not frames:
        return d
    bands = [bottom_break(f) for f in frames]
    d.rows_per_frame = [b for b, _ in bands]
    d.bottom_border_rows = int(np.median([br for _, br in bands]))
    nz = [b for b in d.rows_per_frame if b]
    if nz:
        mode = int(np.median(nz))
        d.head_switching_share = round(sum(abs(b - mode) <= HS_ROW_TOL for b in d.rows_per_frame) / len(frames), 3)
        d.head_switching_rows = mode
    bends = [top_bend(f) for f in frames]
    flagged = [b for b in bends if b >= FLAG_PX]
    d.flagging_share = round(len(flagged) / len(frames), 3)
    d.flagging_px = round(float(np.median(flagged)), 1) if flagged else 0.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clip = [float(np.mean(cv2.erode((f >= CLIP_LEVEL).astype(np.uint8), kernel))) for f in frames]
    hit = [c for c in clip if c >= CLIP_SHARE]
    d.clipped_share = round(len(hit) / len(frames), 3)
    d.clipped_pixels = round(float(np.median(hit)), 4) if hit else 0.0
    return d


def analyze(src: Path, *, count: int = 16, duration: float | None = None) -> TapeDefects:
    src = Path(src)
    info = media.probe(src)
    dur = duration if duration is not None else media.probe(src, exact=True)["duration"]
    full = str(info.get("pix_fmt") or "").startswith("yuvj")
    frames = read_luma(src, [dur * (i + 0.5) / count for i in range(count)])
    return summarize(frames, limited_range=not full)
