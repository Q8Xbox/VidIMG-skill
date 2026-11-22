"""Damaged frames: single frames, or parts of one, that show something neither neighbour shows.

A tape dropout, a torn frame, a burst of broken blocks: the frame before and the frame after agree with each
other, and the frame between them differs from both. That is the opposite of a cut (the frame after agrees with
what follows) and of a ghost (the neighbours differ, and the frame is a mix of them). A model cannot restore such
a frame: it sharpens the damage. What can be done is to make the frame again from its two neighbours, which is a
created frame and is only done on request, in the second file (rebuild.py).

What is measured cannot tell damage from intent. A one-frame white flash, lightning, an impact frame, a blink: all
are single frames that differ from both neighbours, drawn on purpose. So this module only proposes: it lists the
candidates with how much of the frame stands out and in what shape, the survey shows each between its neighbours,
and whoever supervises names the frames to make again.

Measured per frame on a grid of cells (small grey frames, as probe.ghosts reads them): a cell stands out when it
differs from both neighbours by more than CELL_OUT levels while the neighbours differ from each other by less than
CELL_AGREE of that. A frame is a candidate when at least SHARE_MIN of its cells stand out. Candidates next to
another candidate are a run (a flicker, a strobe): reported, but not offered for rebuilding.

Only single frames are found. Two damaged frames in a row are each other's neighbour, so neither has two
neighbours that agree, and neither is a candidate; a rebuild would have nothing sound to work from either.

One reading is a hint for the look, not a verdict: "flash" when the frame itself is nearly one flat tone. A
storm scene of an owner cartoon had 31 of these, white and black, between identical neighbours (2026-11-22):
lightning, drawn on purpose; making them again would remove it. Everything else is "other": torn and dropped-out
tape frames, and also a blink, a mouth shape, a wave that is drawn for one frame. A second reading was tried and
dropped: whether the added picture is made of horizontal lines. Torn frames read 1.3-1.8 on it and a drawn sea
1.9-2.2, the wrong way round.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .. import media
from . import ghosts

GRID = (9, 12)                  # rows, columns of cells
CELL_OUT = 10.0                 # 8-bit levels a cell must differ from BOTH neighbours
CELL_AGREE = 0.4                # ... while the neighbours differ from each other by less than this share of it
SHARE_MIN = 0.05                # share of cells standing out from which a frame is a candidate
FLASH_STD = 10.0                # 8-bit levels: a frame flatter than this is one tone
MAX_LISTED = 200                # candidates kept (the largest first); more than this is not a few damaged frames


@dataclass
class Damage:
    frames: int = 0
    found: list = field(default_factory=list)       # [{frame, share, level, looks, lone}], in frame order
    total: int = 0                                  # candidates before the list was cut to MAX_LISTED
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self) | {"count": len(self.found), "lone": [d["frame"] for d in self.found if d["lone"]]}


def _cells(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(cells of b that stand out from both neighbours, how far each cell is from the nearer neighbour)."""
    size = (GRID[1], GRID[0])
    d = [cv2.resize(np.abs(x - y), size, interpolation=cv2.INTER_AREA) for x, y in ((b, a), (b, c), (a, c))]
    least = np.minimum(d[0], d[1])
    return (least > CELL_OUT) & (d[2] < CELL_AGREE * least), least


def stand_out(before: np.ndarray, frame: np.ndarray, after: np.ndarray) -> float:
    """How much a frame differs from both of its neighbours where they agree with each other: the mean distance
    to the nearer neighbour over the cells where the neighbours agree (8-bit levels). For the check on a frame
    that was made again: it must fall."""
    a, b, c = (x.astype(np.float32) for x in (before, frame, after))
    size = (GRID[1], GRID[0])
    d = [cv2.resize(np.abs(x - y), size, interpolation=cv2.INTER_AREA) for x, y in ((b, a), (b, c), (a, c))]
    least = np.minimum(d[0], d[1])
    return float(np.mean(np.maximum(least - d[2], 0.0)))


class Finder:
    """Fed every frame in order (add), it judges each against its two neighbours."""

    def __init__(self) -> None:
        self.before = self.middle = None
        self.n = 0
        self.rows: list[dict] = []

    def add(self, frame: np.ndarray) -> None:
        after = frame.astype(np.float32)
        if self.before is not None:
            self._judge(self.n - 1, self.before, self.middle, after)
        self.before, self.middle = self.middle, after
        self.n += 1

    def _judge(self, i: int, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> None:
        out, least = _cells(a, b, c)
        share = float(out.mean())
        if share < SHARE_MIN:
            return
        self.rows.append({"frame": i, "share": round(share, 3), "level": round(float(least[out].mean()), 1),
                          "looks": "flash" if float(b.std()) < FLASH_STD else "other"})

    def result(self) -> Damage:
        at = {r["frame"] for r in self.rows}
        rows = [r | {"lone": r["frame"] - 1 not in at and r["frame"] + 1 not in at} for r in self.rows]
        kept = sorted(sorted(rows, key=lambda r: -r["share"])[:MAX_LISTED], key=lambda r: r["frame"])
        return Damage(frames=self.n, found=kept, total=len(rows))


def find(frames) -> Damage:
    f = Finder()
    for frame in frames:
        f.add(frame)
    return f.result()


def scan(src: Path) -> Damage:
    src = Path(src)
    want = media.probe(src, exact=True)["frames"]
    got = find(ghosts.read_gray_exact(src))
    if want and got.frames != want:
        return Damage(frames=got.frames, error=f"read {got.frames} frames of {want}: the frame numbering is not certain")
    return got
