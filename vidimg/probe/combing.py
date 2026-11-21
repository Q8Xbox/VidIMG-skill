"""Partly interlaced video: a progressive file with stretches of woven fields.

A DVD cartoon is often progressive with a few interlaced stretches (video-edited titles, credits, a scene
taken from another master). The whole-file reading then stays far below the "interlaced" level, nothing is
deinterlaced, and the model enlarges the combing. Measured on an owner DVD episode (2026-11-20): 4.3 % of
frames interlaced, in three stretches (3.7 s, 0.7 s and 22.8 s) plus two 4-frame spots, read as progressive.

Every frame is judged on its own (ffmpeg idet, single-frame detection) over the whole file; interlaced frames
close together form a span. Only the spans are deinterlaced, at the same rate, so every other frame stays
exactly as it is (cleanup.py). Standard-definition files only: interlacing is a property of SD video.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .. import media

MAX_HEIGHT = 576                # judged only up to this height
MERGE_GAP = 12                  # interlaced frames this close belong to one span
MIN_FRAMES = 4                  # interlaced frames a span needs (a single odd frame at a cut is not a stretch)
PAD = 2                         # frames added on each side of a span
PARTLY_MAX_SHARE = 0.3          # above this the whole file is interlaced: the field pre-pass handles it


@dataclass
class Combing:
    frames: int = 0                     # frames judged
    interlaced: int = 0
    share: float = 0.0
    spans: list[list[int]] = field(default_factory=list)     # [first, last] frame, inclusive, padded
    span_frames: int = 0
    tff: bool = True                    # field order of the interlaced frames (majority)

    @property
    def is_partly(self) -> bool:
        return bool(self.spans) and self.share <= PARTLY_MAX_SHARE

    def to_dict(self) -> dict:
        return asdict(self) | {"is_partly_interlaced": self.is_partly}


def spans_of(kinds: list[str]) -> Combing:
    """Group per-frame readings ('tff', 'bff', 'progressive', 'undetermined') into spans."""
    n = len(kinds)
    hits = [i for i, k in enumerate(kinds) if k in ("tff", "bff")]
    out = Combing(frames=n, interlaced=len(hits), share=round(len(hits) / n, 4) if n else 0.0,
                  tff=sum(k == "tff" for k in kinds) >= sum(k == "bff" for k in kinds))
    run: list[int] = []
    for i in hits + [None]:
        if i is not None and (not run or i - run[-1] <= MERGE_GAP):
            run.append(i)
            continue
        if len(run) >= MIN_FRAMES:
            out.spans.append([max(0, run[0] - PAD), min(n - 1, run[-1] + PAD)])
        run = [i] if i is not None else []
    merged: list[list[int]] = []
    for s in out.spans:                                     # padding can make neighbours touch
        if merged and s[0] <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], s[1])
        else:
            merged.append(s)
    out.spans = merged
    out.span_frames = sum(b - a + 1 for a, b in merged)
    return out


def read_kinds(src: Path, timeout: float = 3600) -> list[str]:
    cp = subprocess.run([media.ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(src), "-map", "0:v:0",
                         "-vf", "idet,metadata=print:key=lavfi.idet.single.current_frame:file=-", "-an", "-f", "null",
                         "-"], capture_output=True, text=True, encoding="utf-8", errors="replace",
                        stdin=subprocess.DEVNULL, timeout=timeout)
    return re.findall(r"lavfi\.idet\.single\.current_frame=(\w+)", cp.stdout)


def analyze(src: Path) -> Combing:
    src = Path(src)
    info = media.probe(src)
    if not info.get("height") or info["height"] > MAX_HEIGHT:
        return Combing()
    return spans_of(read_kinds(src))
