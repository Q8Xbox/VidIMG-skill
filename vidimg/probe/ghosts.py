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


def find(frames: Iterator[np.ndarray]) -> Ghosts:
    """Judge every frame against its neighbours; keep the mixes that stand alone."""
    out = Ghosts()
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    judged: list[float | None | bool] = []           # for frames 1 .. n-2: weight (a mix), False (clean), None
    before = middle = None
    for after in frames:
        out.frames += 1
        if before is not None:
            r = blend.judge(before, middle, after, dis)
            judged.append(None if r is None else (float(r["alpha"]) if r["blend"] else False))
        before, middle = middle, after
    # the first and the last frame have one neighbour only and are never judged
    verdicts = ([None] + judged + [None]) if out.frames >= 2 else [None] * out.frames
    out.judged = sum(v is not None for v in verdicts)

    def mix(i: int) -> bool:
        return 0 <= i < len(verdicts) and isinstance(verdicts[i], float)

    for i, v in enumerate(verdicts):
        if not isinstance(v, float):
            continue
        if mix(i - 1) or mix(i + 1):
            out.in_runs += 1
            continue
        out.ghosts.append([i, i - 1 if v >= 0.5 else i + 1, round(v, 3)])
    return out


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
