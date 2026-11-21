"""Two things old cartoon transfers carry that enlarging only makes bigger (owner samples, 2026-11-21):

* black borders: bars at the sides, top or bottom that are not picture (Sample01: 8 px left, 22 px right, 4 px
  bottom). A model enlarges them with the picture and sharpens their edge.
* padded repeats: a 25 or 23.976 fps original brought to 29.97 by showing one picture twice at regular
  intervals (Sample01: every 6th frame a copy of the one before). Motion judders, and the copy is one more frame
  to process for nothing.

Both are measured here and only reported: whether to crop or to drop the copies is decided by whoever looks at
the close-ups (survey), because a number cannot tell a black bar from a night scene's edge, nor a padded copy
from a held drawing.

Repeats: a frame is a copy when it differs from the one before by less than REPEAT_LEVEL (mean absolute luma
difference at analysis size) while the frames around it move. Measured on Sample01: copies 0.09-0.13, the
smallest real movement 0.33. Padding shows as copies whose distances are whole multiples of the cycle (6 for
25 -> 29.97, 5 for 23.976 -> 29.97): on Sample01 all 37 distances in a minute were multiples of 6. Animation
drawn on twos or threes repeats at distances 1-3 and is never read as padding.
"""

from __future__ import annotations

import re
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

from .. import media
from . import blend as blend_probe

# ----------------------------------------------------------------------------- borders
BORDER_LEVEL = 24               # luma at or below this is "black" (limited-range black is 16)
BORDER_MIN = 4                  # a side narrower than this is not worth cropping
BORDER_MAX_SHARE = 0.25         # more than this of a dimension is not a border (a dark picture, a letterbox film)
BORDER_FRAMES = 120


@dataclass
class Borders:
    left: int = 0
    right: int = 0
    top: int = 0
    bottom: int = 0
    frames: int = 0

    @property
    def any(self) -> bool:
        return max(self.left, self.right, self.top, self.bottom) >= BORDER_MIN

    def to_dict(self) -> dict:
        return asdict(self) | {"has_borders": self.any}


def even_crop(width: int, height: int, b: dict) -> tuple[int, int, int, int] | None:
    """(w, h, x, y) for ffmpeg's crop, every edge on an even pixel (4:2:0 chroma). An odd border is rounded
    down: one black column stays rather than one picture column going. None when nothing is left to crop."""
    x, y = b.get("left", 0) // 2 * 2, b.get("top", 0) // 2 * 2
    r, bt = b.get("right", 0) // 2 * 2, b.get("bottom", 0) // 2 * 2
    w, h = width - x - r, height - y - bt
    if (x, y, r, bt) == (0, 0, 0, 0) or w < width * (1 - 2 * BORDER_MAX_SHARE) or h < height * (1 - 2 * BORDER_MAX_SHARE):
        return None
    return w, h, x, y


def borders(src: Path, frames: int = BORDER_FRAMES) -> Borders:
    """The black bars common to frames spread over the whole video (ffmpeg cropdetect, accumulated: the result
    is the smallest bar seen on each side, so one bright frame is enough to keep a column)."""
    src = Path(src)
    info = media.probe(src)
    w, h = info["width"], info["height"]
    step = max(1, int((info["frames"] or frames) // frames))
    cp = subprocess.run([media.ffmpeg(), "-hide_banner", "-nostdin", "-i", str(src), "-map", "0:v:0", "-vf",
                         f"select='not(mod(n\\,{step}))',cropdetect=limit={BORDER_LEVEL}:round=2:reset=0",
                         "-an", "-f", "null", "-"], capture_output=True, text=True, encoding="utf-8", errors="replace",
                        stdin=subprocess.DEVNULL, timeout=1800)
    found = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", cp.stderr)
    if not found or not w or not h:
        return Borders()
    cw, ch, cx, cy = map(int, found[-1])
    b = Borders(left=cx, right=max(0, w - cw - cx), top=cy, bottom=max(0, h - ch - cy), frames=len(found))
    if (b.left + b.right) > w * BORDER_MAX_SHARE or (b.top + b.bottom) > h * BORDER_MAX_SHARE:
        return Borders(frames=len(found))          # not bars: mostly dark frames, or a film letterbox to keep
    return b


# ----------------------------------------------------------------------------- padded repeats
REPEAT_LEVEL = 0.12             # mean |luma difference| below this = the same picture again (analysis size)
MOVING_LEVEL = 0.8              # median difference around a frame above this = the scene moves
REPEAT_WIDTH = 160
CYCLES = {6: Fraction(25), 5: Fraction(24000, 1001)}       # copy every Nth frame of 29.97 -> the original rate
PADDED_RATE = Fraction(30000, 1001)
CYCLE_SHARE = 0.8               # share of copy-to-copy distances that are multiples of the cycle
CYCLE_MIN_GAPS = 12


@dataclass
class Repeats:
    judged: int = 0                     # frames inside moving stretches
    repeats: int = 0
    share: float = 0.0
    gaps: dict = field(default_factory=dict)     # distance between neighbouring copies -> how often
    cycle: int | None = None
    original_fps: str | None = None     # exact fraction, when the copies form a padding cycle
    example_s: float | None = None      # time of one copy, for the close-up strip

    @property
    def is_padded(self) -> bool:
        return self.cycle is not None

    def to_dict(self) -> dict:
        return asdict(self) | {"is_padded": self.is_padded}


def frame_steps(frames: list[np.ndarray]) -> np.ndarray:
    f = [x.astype(np.float32) for x in frames]
    return np.array([float(np.mean(np.abs(f[i + 1] - f[i]))) for i in range(len(f) - 1)])


def copies(steps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(is a copy of the frame before, lies in a moving stretch) per frame step."""
    moving = np.array([np.median(steps[max(0, i - 6):i + 7]) > MOVING_LEVEL for i in range(len(steps))], bool)
    return (steps < REPEAT_LEVEL) & moving, moving


def summarize(windows: list[np.ndarray], fps: float, starts: list[float] | None = None) -> Repeats:
    st = Repeats()
    gaps: Counter = Counter()
    for k, steps in enumerate(windows):
        if len(steps) < 12:
            continue
        rep, moving = copies(steps)
        st.judged += int(moving.sum())
        st.repeats += int(rep.sum())
        idx = np.flatnonzero(rep)
        gaps.update(int(g) for g in np.diff(idx))
        if st.example_s is None and len(idx) and starts and fps:
            st.example_s = round(starts[k] + float(idx[len(idx) // 2] + 1) / fps, 3)
    st.share = round(st.repeats / st.judged, 3) if st.judged else 0.0
    st.gaps = {str(g): n for g, n in gaps.most_common(6)}
    total = sum(gaps.values())
    if total >= CYCLE_MIN_GAPS and fps and abs(fps / float(PADDED_RATE) - 1) < 0.005:
        fit = {c: sum(n for g, n in gaps.items() if g % c == 0) / total for c in CYCLES}
        best = max(fit, key=fit.get)
        if fit[best] >= CYCLE_SHARE:
            st.cycle, st.original_fps = best, str(CYCLES[best])
    return st


def repeats(src: Path, *, count: int = 6, seconds: float = 8.0) -> Repeats:
    src = Path(src)
    info = media.probe(src, exact=True)
    spans = blend_probe.windows_for(info["duration"], count, seconds)
    wins = [frame_steps(list(blend_probe.read_gray(src, s, t, width=REPEAT_WIDTH))) for s, t in spans]
    return summarize(wins, info["fps"], [s for s, _ in spans])
