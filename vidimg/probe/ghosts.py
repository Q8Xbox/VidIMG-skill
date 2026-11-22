"""Ghost frames: single frames that are a mix of the picture before and the picture after.

A copy made at a low frame rate with frame blending (an owner cartoon at 10 fps, 2026-11-22) shows double images:
one frame in about fourteen of its moving frames is a roughly half-and-half mix of its two neighbours, sometimes of
two different scenes at a cut. Un-blending by selection (sRestore) cannot help: it picks the least blended of
several frames per original picture, and here there is one frame per picture or fewer. A model then redraws the
ghost with crisp double outlines, more visible than in the soft source.

What can be done without inventing anything: a ghost holds no picture of its own. Both halves exist whole in its
neighbours, so the ghost is replaced by the neighbour it holds more of. Every frame of the result is a frame of the
source; the drawing is held one frame longer, as held drawings are in the animation itself.

A frame is a ghost when probe.blend judges it a mix of its two neighbours (weight between 0.15 and 0.85, the mix
explains it far better than either neighbour or their motion-compensated middle, and the neighbours really differ)
AND it stands alone: neither neighbour is judged a mix. Runs of mixes are dissolves and fades, which are pictures
the film has, and stay. Measured on the owner cartoon: 74 ghosts in three minutes, none in a run.

Frames are numbered from the start of the file, read without frame-rate conversion: the count must equal the
file's own frame count, or nothing is reported (a seek, or a reader that fills timestamp gaps, shifts the numbering
by one, which shows the neighbour instead of the ghost).
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .. import media
from . import blend

WIDTH = blend.WIDTH
LOW_FPS = 20.0                  # below this, selection (sRestore) has no spare frames to choose from
SHARE_MIN = 0.03                # share of moving frames that are mixes, from which the profile reports ghosts


@dataclass
class Ghosts:
    frames: int = 0                         # frames read (equals the file's frame count)
    judged: int = 0                         # frames with enough motion around them to judge
    ghosts: list = field(default_factory=list)     # [frame, replace with frame, weight of the earlier neighbour]
    in_runs: int = 0                        # mixes left alone because their neighbours are mixes too
    at_cuts: int = 0                        # of the ghosts: frames between two scenes holding some of both
    error: str | None = None

    @property
    def share(self) -> float:
        return round(len(self.ghosts) / self.judged, 4) if self.judged else 0.0

    def to_dict(self) -> dict:
        return asdict(self) | {"share": self.share, "count": len(self.ghosts)}


def read_gray_exact(src: Path, width: int = WIDTH, limit: int | None = None) -> Iterator[np.ndarray]:
    """Luma of every frame from the start (the first `limit` when given), area-downscaled, one array per stored
    frame (no rate conversion)."""
    info = media.probe(src)
    w, h = info["width"], info["height"]
    oh = max(2, round(h * width / w / 2) * 2)
    errors = tempfile.TemporaryFile()
    p = subprocess.Popen([media.ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(src), "-map", "0:v:0",
                          "-vf", f"format=yuv420p16le,extractplanes=y,scale={width}:{oh}:flags=area", "-fps_mode",
                          "passthrough", *(["-frames:v", str(int(limit))] if limit else []),
                          "-pix_fmt", "gray16le", "-f", "rawvideo", "-"],
                         stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, stderr=errors)
    size = width * oh * 2
    try:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            y = np.frombuffer(buf, "<u2").reshape(oh, width)
            yield np.minimum((y.astype(np.uint32) + 128) >> 8, 255).astype(np.uint8)
    finally:
        p.stdout.close()
        p.wait()
        errors.seek(0)
        media.relay_errors(errors.read().decode("utf-8", "replace"))
        errors.close()


# ----------------------------------------------------------------------------- what one pass measures
# Two more kinds of ghost exist, which are not mixes of their neighbours (measured on the owner cartoon, where the
# neighbour test found 366 and these found about 600 more):
# * at a cut: one frame showing both scenes at once. Its neighbours are different scenes, so the frame before and
#   the frame after stand for the two halves closely enough: it is replaced by the one it holds more of, like any
#   other ghost, and the cut becomes a clean cut. Every frame is still a frame of the source.
# * inside a scene, in fast movement: a mix of two pictures the file holds nowhere else. No frame of the source can
#   stand in for it. It is only reported (find_soft), for the copy that holds created frames (rebuild.py).
# What shows both: two displaced or different pictures at half strength carry about half the edge energy of one
# picture, so the frame's edge energy dips below that of both neighbours. A rhythm test (the blend falls on fixed
# places of a five-frame cycle) threw real ghosts away and was dropped.
CUT_LIKENESS = 0.5              # correlation of the two neighbours' thumbnails below this: not the same picture
CUT_TONES = 0.85                # ... and the share of their brightness histograms that overlaps below this: not the
                                # same scene either. A fast pan fails the first test and passes the second (same
                                # tones, moved); a frame in between is left alone: nothing sure can be said of it
SAME_TONES = 0.9                # brightness histograms overlapping this much: one scene, however far it moved.
                                # Measured on the owner cartoon: one-frame mixes of two scenes 0.41-0.81, ghosts
                                # inside a scene 0.90-0.97
CUT_DIP = 0.90                  # edge energy against the neighbours' mean, for a frame between two scenes
CUT_WEIGHT = (0.1, 0.9)         # the frame must hold some of both (1 = it IS the frame before: a plain cut)
CUT_BOTH = (0.15, 0.95)         # ... and look like both neighbours, like neither completely. At a plain cut from a
                                # detailed scene to a flat one the first flat frame dips against the mean of its
                                # neighbours as well; it is simply a frame of its own scene (likeness near 1)
SOFT_DIP = 0.85                 # ... for a frame inside a scene. Looked at: under 0.80, ghosts in 10 of 10; from
                                # 0.80 to 0.85, in 6 of 8; above, several were not
SOFT_NEIGHBOUR = 0.92           # both neighbours must be at least this (a lone dip, not a soft stretch)
SOFT_MOVE = 2.0                 # mean |luma difference| to each neighbour: the picture is moving


@dataclass
class Soft:
    frames: int = 0
    found: list = field(default_factory=list)       # frame numbers
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self) | {"count": len(self.found)}


def _energy(frame: np.ndarray) -> float:
    lap = cv2.Laplacian(cv2.GaussianBlur(frame.astype(np.float32), (0, 0), 0.8), cv2.CV_32F)
    return float(np.mean(lap * lap))


def _thumb(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(small picture, brightness histogram as shares)."""
    hist = np.bincount((frame.ravel() >> 3).astype(np.int64), minlength=32).astype(np.float32)
    return cv2.resize(frame, (64, 48), interpolation=cv2.INTER_AREA).astype(np.float32).ravel(), hist / max(1.0, hist.sum())


def _likeness(a: tuple, b: tuple) -> tuple[float, float]:
    """(correlation of the small pictures, overlap of the brightness histograms)."""
    x, y = a[0] - a[0].mean(), b[0] - b[0].mean()
    den = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    return (float(np.dot(x, y) / den) if den > 0 else 1.0), float(np.minimum(a[1], b[1]).sum())


def _scene(like: tuple) -> str:
    """The two neighbours of a frame are the 'same' scene, 'different' scenes, or it is 'unsure'."""
    picture, tones = like
    if picture >= CUT_LIKENESS or tones >= SAME_TONES:
        return "same"
    return "different" if tones < CUT_TONES else "unsure"


def _weight(before: np.ndarray, middle: np.ndarray, after: np.ndarray) -> float:
    """How much of `before` the middle frame holds if it is a mix of the two (whole frame, band-passed)."""
    a, c, b = (blend.bandpass(x) for x in (before, middle, after))
    d, e = (a - b).ravel(), (c - b).ravel()
    dd = float(np.dot(d, d))
    return float(np.dot(e, d) / dd) if dd > 0 else 1.0


def _measure(frames: Iterator[np.ndarray], mixes: bool = True) -> dict:
    """One pass over the frames. Per frame: the neighbour-mix verdict (weight, False or None), the edge energy,
    the step from the frame before, how alike its two neighbours are, and, between two scenes, its weight."""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM) if mixes else None
    verdict, alike, weight, both, energy, step = [], [], [], [], [], []
    before = middle = tb = tm = None
    for after in frames:
        ta = _thumb(after)
        energy.append(_energy(after))
        step.append(0.0 if middle is None else float(np.mean(np.abs(after.astype(np.float32) - middle))))
        if before is not None:
            r = blend.judge(before, middle, after, dis) if mixes else None
            verdict.append(None if r is None else (float(r["alpha"]) if r["blend"] else False))
            scene = _scene(_likeness(tb, ta))
            alike.append(scene)
            weight.append(_weight(before, middle, after) if scene == "different" else None)
            both.append((_likeness(tb, tm)[0], _likeness(tm, ta)[0]) if scene == "different" else None)
        before, middle, tb, tm = middle, after, tm, ta
    n = len(energy)

    def pad(x: list, fill) -> list:                 # the first and the last frame have one neighbour only
        return ([fill] + x + [fill]) if n >= 2 else [fill] * n

    e = np.array(energy) if n else np.zeros(0)
    ratio = np.ones(n)
    if n >= 3:
        ratio[1:-1] = e[1:-1] / np.maximum(1e-6, (e[:-2] + e[2:]) / 2)
    return {"n": n, "verdict": pad(verdict, None), "alike": pad(alike, "same"), "weight": pad(weight, None), "both": pad(both, None),
            "ratio": ratio, "step": step}


def _lone_dip(m: dict, i: int, limit: float) -> bool:
    r = m["ratio"]
    return bool(2 <= i < m["n"] - 2 and r[i] < limit and r[i - 1] >= SOFT_NEIGHBOUR and r[i + 1] >= SOFT_NEIGHBOUR)


def find(frames: Iterator[np.ndarray]) -> Ghosts:
    """The ghosts a neighbour can stand in for: lone mixes of their two neighbours, and lone frames between two
    scenes that hold some of both."""
    m = _measure(frames)
    out = Ghosts(frames=m["n"])
    verdicts = m["verdict"]
    out.judged = sum(v is not None for v in verdicts)

    def mix(i: int) -> bool:
        return 0 <= i < len(verdicts) and isinstance(verdicts[i], float)

    for i, v in enumerate(verdicts):
        if isinstance(v, float):
            if mix(i - 1) or mix(i + 1):
                out.in_runs += 1                             # a dissolve or a fade: pictures the film has
                continue
            out.ghosts.append([i, i - 1 if v >= 0.5 else i + 1, round(v, 3)])
            continue
        w, lk = m["weight"][i], m["both"][i]
        if (w is not None and CUT_WEIGHT[0] <= w <= CUT_WEIGHT[1] and _lone_dip(m, i, CUT_DIP)
                and CUT_BOTH[0] <= min(lk) and max(lk) < CUT_BOTH[1] and not mix(i - 1) and not mix(i + 1)):
            out.ghosts.append([i, i - 1 if w >= 0.5 else i + 1, round(w, 3)])
            out.at_cuts += 1
    return out


def find_soft(frames: Iterator[np.ndarray], skip: set | None = None) -> Soft:
    """Ghosts no neighbour can stand in for: inside a scene, in movement, a frame whose edge energy dips alone
    below both neighbours'. skip: frames not to report (those already replaced by a neighbour)."""
    m = _measure(frames, mixes=False)
    out = Soft(frames=m["n"])
    skip = skip or set()
    for i in range(2, m["n"] - 2):
        if (_lone_dip(m, i, SOFT_DIP) and m["alike"][i] == "same" and m["step"][i] > SOFT_MOVE
                and m["step"][i + 1] > SOFT_MOVE and i not in skip):
            out.found.append(i)
    return out


def scan_soft(src: Path, skip: set | None = None) -> Soft:
    src = Path(src)
    want = media.probe(src, exact=True)["frames"]
    got = find_soft(read_gray_exact(src), skip)
    if want and got.frames != want:
        return Soft(frames=got.frames, error=f"read {got.frames} frames of {want}: the frame numbering is not certain")
    return got


EXAMPLE_FRAMES = 1500           # the start of the file searched for a ghost to show on the close-ups


def example(src: Path) -> list | None:
    """One ghost near the start of the file, [frame, replacement, weight], for the survey's close-ups."""
    found = find(read_gray_exact(Path(src), limit=EXAMPLE_FRAMES)).ghosts
    return found[len(found) // 2] if found else None


def scan(src: Path) -> Ghosts:
    src = Path(src)
    want = media.probe(src, exact=True)["frames"]
    got = find(read_gray_exact(src))
    if want and got.frames != want:
        return Ghosts(frames=got.frames, error=f"read {got.frames} frames of {want}: the frame numbering is not certain")
    return got
