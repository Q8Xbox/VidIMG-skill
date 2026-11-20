"""Blended-field detection: frames (or bobbed fields) that are a mix of their two neighbours.

A field-blending standards converter (NTSC film -> PAL, PAL -> NTSC) writes fields that are weighted
mixes of two source pictures; a "blend deinterlacer" does the same to whole frames. Such material must
be un-blended (sRestore) before anything else: filters and models only sharpen the ghosts (videohelp
thread 323127; Doom9 consensus). Plain deinterlacing or IVTC cannot fix it.

Test per frame c with neighbours a and b, on band-pass images over the pixels that move (band-pass is
linear, so a true blend stays an exact mix, while brightness changes and grain do not count):
    alpha = <c-b, a-b> / <a-b, a-b>,  mix residual = |c - (alpha a + (1-alpha) b)|
    blend when 0.15 <= alpha <= 0.85, the mix residual is below RATIO x the nearest neighbour's distance,
    and below MC_RATIO x the residual of a motion-compensated middle picture (real motion explains c).
A frame is judged only when a -> b moves at least FLOW_MIN pixels (optical flow): with sub-pixel motion a
slow pan is genuinely indistinguishable from a mix (measured: 100 % false "blends" on a slow pan without
this rule, 0 % with it). A cross-fade is a blend too, so the decision uses the share over several windows
spread through the video, never one scene.

Measured (8 s windows, analysis width 360): synthetic field-blend NTSC film -> PAL 26 %, 23.976 -> 25p blend
16 %, PAL -> NTSC field blend 94 %; clean cartoon, true interlace (2 clips), slow pan, real footage 0 %.
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np

from .. import media

WIDTH = 360                 # analysis width (area-downscaled): enough for edges, cheap, less noise
ALPHA_MIN, ALPHA_MAX = 0.15, 0.85
RATIO = 0.6                 # mix residual vs the nearest neighbour's distance
MC_RATIO = 0.7              # mix residual vs the motion-compensated middle picture's residual
FLOW_MIN = 1.5              # px of motion a -> b (75th percentile over moving pixels) needed to judge
MOVE_LEVEL = 4.0            # a pixel moves when a band-pass difference exceeds this (8-bit levels)
MOVING_MIN = 0.003          # share of moving pixels needed to judge a frame at all
SHARE_BLENDED = 0.10        # share of judged frames that are blends -> "blended"
WINDOW_SHARE_MIN = 0.05     # ... and at least two windows above this (one dissolve cannot trigger it)
MIN_JUDGED = 40


@dataclass
class BlendStats:
    judged: int = 0                     # frames with enough motion to judge
    blended: int = 0
    share: float = 0.0
    windows: list[float] = field(default_factory=list)   # share per window (0 when too few judged frames)
    alpha_median: float | None = None
    bobbed: bool = False                # measured on bobbed fields (interlaced input)

    @property
    def is_blended(self) -> bool:
        return (self.judged >= MIN_JUDGED and self.share >= SHARE_BLENDED
                and sum(1 for w in self.windows if w >= WINDOW_SHARE_MIN) >= min(2, len(self.windows)))

    def to_dict(self) -> dict:
        return asdict(self) | {"is_blended": self.is_blended}


def bandpass(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)     # grain and compression noise out
    return g - cv2.blur(g, (7, 7))                                 # brightness and gradients out


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x)))


def judge(a8: np.ndarray, c8: np.ndarray, b8: np.ndarray, dis=None) -> dict | None:
    """Frame c between a and b (8-bit gray, same size). None when there is too little motion to tell."""
    a, c, b = bandpass(a8), bandpass(c8), bandpass(b8)
    moving = (np.abs(a - b) > MOVE_LEVEL) | (np.abs(c - a) > MOVE_LEVEL) | (np.abs(c - b) > MOVE_LEVEL)
    if moving.mean() < MOVING_MIN:
        return None
    dis = dis or cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    f = dis.calc(a8, b8, None)
    flow = float(np.percentile(np.hypot(f[..., 0], f[..., 1])[moving], 75))
    if flow < FLOW_MIN:
        return None
    d, e = (a - b)[moving], (c - b)[moving]
    dd = float(np.dot(d, d))
    if dd <= 0:
        return None
    alpha = float(np.dot(e, d)) / dd
    near = min(_rms((c - a)[moving]), _rms(e))
    mix = _rms(e - alpha * d)
    h, w = a8.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    fa = cv2.remap(a, gx - 0.5 * f[..., 0], gy - 0.5 * f[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    fb = cv2.remap(b, gx + 0.5 * f[..., 0], gy + 0.5 * f[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    motion = _rms((c - 0.5 * (fa + fb))[moving])
    blend = ALPHA_MIN <= alpha <= ALPHA_MAX and mix < RATIO * near and mix < MC_RATIO * motion
    return {"alpha": alpha, "mix": mix, "near": near, "motion": motion, "flow": flow, "blend": blend}


def classify(frames: Iterable[np.ndarray]) -> tuple[list[bool | None], list[float]]:
    """Per frame: True (blend), False (clean motion) or None (not judged). Also the blend weights."""
    fr = list(frames)
    out: list[bool | None] = [None] * len(fr)
    alphas = []
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    for n in range(1, len(fr) - 1):
        r = judge(fr[n - 1], fr[n], fr[n + 1], dis)
        if r is None:
            continue
        out[n] = r["blend"]
        if r["blend"]:
            alphas.append(r["alpha"])
    return out, alphas


def summarize(windows: list[list[np.ndarray]], bobbed: bool = False) -> BlendStats:
    st = BlendStats(bobbed=bobbed)
    alphas: list[float] = []
    for frames in windows:
        res, al = classify(frames)
        j = [r for r in res if r is not None]
        st.judged += len(j)
        st.blended += sum(j)
        st.windows.append(round(sum(j) / len(j), 3) if len(j) >= 10 else 0.0)
        alphas += al
    st.share = round(st.blended / st.judged, 3) if st.judged else 0.0
    st.alpha_median = round(float(np.median(alphas)), 3) if alphas else None
    return st


def read_gray(src: Path, start: float, seconds: float, *, bob: str | None = None,
              width: int = WIDTH) -> Iterator[np.ndarray]:
    """Luma of a span, area-downscaled to 8-bit; bob='tff'|'bff' splits fields first (bwdif, double rate).
    The luma plane is taken as stored (extractplanes): a gray conversion re-scaled untagged 10-bit files by
    up to 4 levels (measured), which made identical frames look different."""
    info = media.probe(src)
    w, h = info["width"], info["height"]        # size only (the duration may be a container's)
    oh = max(2, round(h * width / w / 2) * 2)
    vf = (f"bwdif=mode=send_field:parity={bob}:deint=all," if bob else "") +         f"format=yuv420p16le,extractplanes=y,scale={width}:{oh}:flags=area"
    p = subprocess.Popen([media.ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-ss", f"{start:.3f}",
                          "-t", f"{seconds:g}", "-i", str(src), "-map", "0:v:0", "-vf", vf, "-pix_fmt", "gray16le",
                          "-f", "rawvideo", "-"], stdout=subprocess.PIPE, stdin=subprocess.DEVNULL)
    size = width * oh * 2
    try:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            y = np.frombuffer(buf, "<u2").reshape(oh, width)
            # capped: 65535 + 128 rounds to 256, which wrapped pure white to black on full-range sources
            yield np.minimum((y.astype(np.uint32) + 128) >> 8, 255).astype(np.uint8)
    finally:
        p.stdout.close()
        p.wait()


def windows_for(duration: float, count: int = 5, seconds: float = 6.0) -> list[tuple[float, float]]:
    if duration <= seconds * count:
        return [(0.0, duration)]
    return [(max(0.0, duration * (i + 0.5) / count - seconds / 2), seconds) for i in range(count)]


def analyze(src: Path, *, interlaced: bool = False, field_order: str | None = None, duration: float | None = None,
            count: int = 5, seconds: float = 6.0) -> BlendStats:
    src = Path(src)
    # the video's own length: a container duration stretched by a subtitle track put every window past the end
    dur = duration if duration is not None else media.probe(src, exact=True)["duration"]
    bob = ("bff" if field_order == "bff" else "tff") if interlaced else None
    wins = [list(read_gray(src, s, t, bob=bob)) for s, t in windows_for(dur, count, seconds)]
    return summarize(wins, bobbed=bool(bob))


HELD_STILL = 0.8            # band-pass RMS difference below which two frames show the same picture
HELD_MOVING = 2.0           # ... counted only where the frames around the pair do move


def held_share(frames: list[np.ndarray]) -> float:
    """Share of frame pairs that hold the same picture while the scene around them moves. A rate above the
    original repeats pictures (measured: sRestore at 29.97 on a 25 fps original held 10 points more than
    at 25), which the blend share cannot see. On twos, every other pair is held in any case (share 0.5)."""
    bp = [bandpass(f) for f in frames]
    d = [_rms(bp[i + 1] - bp[i]) for i in range(len(bp) - 1)]
    # context: two pairs on each side (on twos, a moving pair sits between two held ones)
    moving = [i for i in range(len(d))
              if max((d[k] for k in range(max(0, i - 2), min(len(d), i + 3)) if k != i), default=0.0) > HELD_MOVING]
    return round(sum(d[i] < HELD_STILL for i in moving) / len(moving), 3) if moving else 0.0
