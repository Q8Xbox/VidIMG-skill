"""The scene map: where a video cuts, and what kinds of scene it is made of.

One recipe for a whole video is a compromise: a night scene, a held title card, fast action and a detailed
painted background do not want the same settings. Whoever composes the recipe cannot look at several hundred
scenes, and does not need to: the scenes of one video fall into a handful of kinds. This module splits the video
at its cuts, measures every scene, and sorts the scenes into at most MAX_KINDS kinds, each with frames to look at.

What is measured per scene (one pass over every frame, small and grey, as probe.ghosts reads them):
  * light: the median of the frames' mean brightness (dark / mid / bright);
  * motion: the mean change from one frame to the next (still / moving / fast);
  * detail: the median edge energy (flat / detailed);
  * dips: the share of its frames whose edge energy dips alone below both neighbours' while the picture moves
    (double images; see probe.ghosts). Reported per kind, not used to sort.
The three levels of a scene are its kind. Kinds holding little of the video are joined to the nearest kind that
holds more, so the list stays short. The names say what was measured, not what the scene is: whether a "dark,
still, flat" kind is a night sky or the end credits is for the eye, on the close-ups.

A cut is where two consecutive frames are not the same picture (their small pictures correlate below
CUT_PICTURE) AND not the same tones (their brightness histograms overlap below CUT_TONES). Looked at on three
owner cartoons (2026-11-22): of 28 such places 26 were cuts; movement lowers the first number and keeps the
second (0.88-0.95), a cut lowers both (0.5-0.85). The stricter test probe.ghosts uses for single frames (picture
below 0.5) found 48 scenes in twenty minutes of a copy whose cuts are blended over a frame, where there are
about 350. For such copies the frames two apart are tested as well, from LOW_FPS frames a second up: at 10 fps
two frames apart is a fifth of a second, and that test called ordinary movement a cut in 10 of 14 places.
A wrong cut costs little (two scenes of the same kind); a missed one joins two scenes into one measurement.
A slow fade from one scene into the next is not found: the two scenes then count as one. Stretches shorter
than MIN_SCENE frames (flashes, a one-frame mix at a cut, bursts of movement) join the scene before them.

Frames are numbered from the start of the file, read without frame-rate conversion; when the count does not equal
the file's own frame count nothing is reported (see probe.ghosts).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from .. import media
from . import ghosts

CUT_PICTURE = 0.8               # correlation of two frames' small pictures below this ...
CUT_TONES = 0.85                # ... and overlap of their brightness histograms below this: a cut
MIN_SCENE = 8                   # frames; a shorter stretch joins the scene before it
MAX_KINDS = 6                   # kinds a person or Claude is asked to look at
MIN_SHARE = 0.03                # a kind holding less of the video than this is joined to a larger one
DARK, BRIGHT = 60.0, 140.0      # mean brightness, 8-bit levels
STILL, FAST = 0.6, 6.0          # mean |difference| between consecutive frames, 8-bit levels
FLAT = 60.0                     # median edge energy (probe.ghosts._energy at its analysis width)
LEVELS = {"light": ("dark", "mid", "bright"), "motion": ("still", "moving", "fast"), "detail": ("flat", "detailed")}
EXAMPLES = 2                    # scenes shown per kind


@dataclass
class Scenes:
    frames: int = 0
    scenes: list = field(default_factory=list)      # [first frame, last frame, kind id], in order, covering every frame
    kinds: list = field(default_factory=list)       # see _kinds
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self) | {"count": len(self.scenes)}

    def brief(self) -> dict:
        """Without the scene list (hundreds of entries): what a plan or a summary carries."""
        return {"frames": self.frames, "count": len(self.scenes), "kinds": self.kinds, "error": self.error}


def _level(value: float, low: float, high: float | None = None) -> int:
    return 0 if value < low else (1 if high is None or value <= high else 2)


def _cut(a: tuple, b: tuple) -> bool:
    picture, tones = ghosts._likeness(a, b)
    return picture < CUT_PICTURE and tones < CUT_TONES


def measure(frames: Iterator[np.ndarray], two_apart: bool = True, watch=None) -> dict:
    """One pass: per frame its brightness, edge energy, change from the frame before, and whether it starts a
    new scene. two_apart: also cut where the frames before and after a frame differ (a cut blended over a frame).
    watch: called with every frame (another measurement that needs the same pass, e.g. probe.damage)."""
    light, energy, step, cut = [], [], [], []
    before = tb = tbb = None
    for f in frames:
        if watch is not None:
            watch(f)
        t = ghosts._thumb(f)
        light.append(float(f.mean()))
        energy.append(ghosts._energy(f))
        if before is None:
            step.append(0.0)
            cut.append(True)
        else:
            step.append(float(np.mean(np.abs(f.astype(np.float32) - before))))
            cut.append(_cut(tb, t))
            if two_apart and tbb is not None and _cut(tbb, t):
                cut[-2] = True
        before, tb, tbb = f, t, tb
    return {"n": len(light), "light": np.array(light), "energy": np.array(energy), "step": np.array(step), "cut": cut}


def split(cut: list, min_scene: int = MIN_SCENE) -> list[list[int]]:
    """[first, last] of every scene. A stretch shorter than min_scene joins the scene before it (the first one:
    the scene after it)."""
    n = len(cut)
    starts = [i for i, c in enumerate(cut) if c or i == 0]
    spans = [[a, (starts[k + 1] if k + 1 < len(starts) else n) - 1] for k, a in enumerate(starts)]
    out: list[list[int]] = []
    for a, z in spans:
        if out and (z - a + 1 < min_scene or out[-1][1] - out[-1][0] + 1 < min_scene):
            out[-1][1] = z
        else:
            out.append([a, z])
    return out


def _dips(m: dict, a: int, z: int) -> int:
    e, s = m["energy"], m["step"]
    count = 0
    for i in range(max(a + 1, 2), min(z, m["n"] - 2)):          # both neighbours inside the scene
        mean = (e[i - 1] + e[i + 1]) / 2
        if (mean > 0 and e[i] / mean < ghosts.SOFT_DIP and s[i] > ghosts.SOFT_MOVE and s[i + 1] > ghosts.SOFT_MOVE):
            count += 1
    return count


def _scene_stats(m: dict, a: int, z: int) -> dict:
    steps = m["step"][a + 1:z + 1]
    light = float(np.median(m["light"][a:z + 1]))
    motion = float(steps.mean()) if len(steps) else 0.0
    detail = float(np.median(m["energy"][a:z + 1]))
    return {"first": a, "last": z, "light": light, "motion": motion, "detail": detail, "dips": _dips(m, a, z),
            "key": (_level(light, DARK, BRIGHT), _level(motion, STILL, FAST), _level(detail, FLAT))}


def _name(key: tuple) -> str:
    return ", ".join(LEVELS[k][v] for k, v in zip(("light", "motion", "detail"), key))


def _kinds(stats: list[dict], total: int) -> tuple[list[dict], dict]:
    """The kinds, largest first, and {level key: kind id} for every key seen."""
    frames: dict = {}
    for s in stats:
        frames[s["key"]] = frames.get(s["key"], 0) + s["last"] - s["first"] + 1
    order = sorted(frames, key=lambda k: (-frames[k], k))
    kept = [k for k in order if frames[k] / max(1, total) >= MIN_SHARE][:MAX_KINDS] or order[:1]
    home = {}
    for k in order:                                             # a small kind joins the nearest kept one
        home[k] = k if k in kept else min(kept, key=lambda c: (sum(abs(x - y) for x, y in zip(c, k)), -frames[c]))
    ids = {k: f"k{i + 1}" for i, k in enumerate(kept)}
    kinds = []
    for k in kept:
        mine = [s for s in stats if home[s["key"]] == k]
        own = sorted((s for s in mine if s["key"] == k), key=lambda s: s["first"] - s["last"])
        n = sum(s["last"] - s["first"] + 1 for s in mine)
        shown, picked = [], []
        for s in own:                                           # the longest scenes, not two from the same place
            if all(abs(s["first"] - p["first"]) > total / 10 for p in picked) or len(own) <= EXAMPLES:
                picked.append(s)
                shown.append((s["first"] + s["last"]) // 2)
            if len(shown) == EXAMPLES:
                break
        kinds.append({"id": ids[k], "name": _name(k), "scenes": len(mine), "frames": n,
                      "share": round(n / max(1, total), 4),
                      "light": round(float(np.median([s["light"] for s in mine])), 1),
                      "motion": round(float(np.median([s["motion"] for s in mine])), 2),
                      "detail": round(float(np.median([s["detail"] for s in mine])), 1),
                      "dips": round(sum(s["dips"] for s in mine) / max(1, n), 4),
                      "example_frames": shown})
    return kinds, {k: ids[home[k]] for k in order}


def find(frames: Iterator[np.ndarray], fps: float | None = None, watch=None) -> Scenes:
    m = measure(frames, two_apart=fps is None or fps >= ghosts.LOW_FPS, watch=watch)
    if not m["n"]:
        return Scenes(error="no frame could be read")
    stats = [_scene_stats(m, a, z) for a, z in split(m["cut"])]
    kinds, ids = _kinds(stats, m["n"])
    return Scenes(frames=m["n"], scenes=[[s["first"], s["last"], ids[s["key"]]] for s in stats], kinds=kinds)


def scan(src: Path, watch=None) -> Scenes:
    src = Path(src)
    info = media.probe(src, exact=True)
    want = info["frames"]
    got = find(ghosts.read_gray_exact(src), info.get("fps"), watch)
    if want and got.frames != want:
        return Scenes(frames=got.frames, error=f"read {got.frames} frames of {want}: the frame numbering is not certain")
    return got


def spans_of(scenes: list, kind: str) -> list[list[int]]:
    """[first, last] of every stretch of one kind, neighbouring scenes of that kind joined."""
    out: list[list[int]] = []
    for a, z, k in scenes:
        if k != kind:
            continue
        if out and out[-1][1] == a - 1:
            out[-1][1] = z
        else:
            out.append([a, z])
    return out


def at_count(scenes: list, frames: int, now: int) -> list:
    """The map on a version of the video that has `now` frames over the same duration (after a step that removed
    copies or blends): every scene keeps its share of the running time. With the same count, the map itself."""
    if now == frames or not frames:
        return [list(x) for x in scenes]
    out = []
    for a, z, k in scenes:
        a2, z2 = round(a * now / frames), round((z + 1) * now / frames) - 1
        if z2 >= a2:
            out.append([a2, z2, k])
    if out:
        out[0][0], out[-1][1] = 0, now - 1
    return out


def ranges_by(scenes: list, value: dict, default=None) -> list:
    """[first, last, value of the scene's kind], neighbouring stretches with the same value joined."""
    out: list = []
    for a, z, k in scenes:
        v = value.get(k, default)
        if out and out[-1][2] == v and out[-1][1] == a - 1:
            out[-1][1] = z
        else:
            out.append([a, z, v])
    return out


def kind_at(scenes: list, frame: int) -> str | None:
    return next((k for a, z, k in scenes if a <= frame <= z), None)
