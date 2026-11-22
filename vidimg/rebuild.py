"""Rebuild the ghost frames that selection cannot repair, for the copy that holds created frames.

probe/ghosts.py replaces a ghost by a neighbour when both of its pictures exist whole beside it. In fast movement,
and at many cuts, they do not: the ghost is a mix of two pictures the file holds nowhere else. Such a frame can only
be made again, from the frame before and the frame after, and a frame made that way is a created frame. So it is
never done in the faithful file. It is done, on request, in the second file a recipe may ask for (the frame-rate
copy, or a copy at the original rate), where the record says how many frames were created.

Measured on an owner cartoon at 10 fps (2026-11-22): RIFE between the two neighbours gave a clean single picture
where motion-vector interpolation (mvtools MFlowInter) left a pile of ghosts on large movement; across a cut RIFE
shows one of the two scenes.

How: the frames before and after every ghost are written out as pairs, RIFE doubles that clip (the frame between
each pair is what is wanted, at index 4k+1), and the video is written again with those frames in the ghosts'
places. Every other frame passes through as stored.

The same is done for damaged frames (probe/damage.py), with two additions:
  * runs: up to RUN_MAX frames in a row are made from the sound frame before the run and the sound frame after
    it. RIFE is asked for as many frames between that pair as the run is long (a run of two: the pair tripled,
    the frames at one third and two thirds).
  * regions: a torn frame is usually sound over most of its area. Where the stored frame agrees with the frame
    made for it, the stored frame is kept; only the part that differs is taken from the made frame, with a soft
    edge. Less of the picture is created, and what is kept is the source's own.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

from . import media
from .adapters import video2x

PIX = "yuv420p10le"
RIFE_MODEL = "rife-v4.26"
RUN_MAX = 3                     # frames in a row made from one pair of sound frames (probe.damage finds as many)
REGION_GRID = (9, 12)           # rows, columns of cells in which a stored frame is compared with the made one
REGION_OUT = 10.0               # 8-bit levels: a cell differing more than this is taken from the made frame
REGION_WHOLE = 0.6              # share of cells from which the whole frame is taken (nothing sound is left to keep)


def _stream(path: Path) -> dict:
    cp = subprocess.run([media.ffprobe(), "-v", "error", "-select_streams", "v:0", "-show_entries",
                         "stream=width,height,r_frame_rate", "-of", "csv=p=0", str(path)],
                        capture_output=True, text=True, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL)
    w, h, rate = cp.stdout.strip().split(",")[:3]
    return {"width": int(w), "height": int(h), "rate": rate}


def _frames(path: Path, width: int, height: int) -> Iterator[bytes]:
    """Every frame of `path` as raw 10-bit 4:2:0, in stored order, without rate conversion."""
    size = width * height * 3                       # 1.5 samples per pixel, 2 bytes each
    errors = tempfile.TemporaryFile()
    p = subprocess.Popen([media.ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(path), "-map", "0:v:0",
                          "-vf", f"scale={width}:{height}:flags=bicubic", "-fps_mode", "passthrough", "-pix_fmt", PIX,
                          "-f", "rawvideo", "-"], stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, stderr=errors)
    try:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            yield buf
    finally:
        p.stdout.close()
        p.wait()
        errors.seek(0)
        media.relay_errors(errors.read().decode("utf-8", "replace"))
        errors.close()


def _writer(dst: Path, width: int, height: int, rate: str, audio_from: Path | None = None) -> subprocess.Popen:
    args = [media.ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", PIX,
            "-s", f"{width}x{height}", "-r", rate, "-i", "-"]
    if audio_from is not None:
        args += ["-i", str(audio_from), "-map", "0:v:0", "-map", "1:a?", "-c:a", "copy", "-map_metadata", "1"]
    args += ["-c:v", "ffv1", "-level", "3", "-pix_fmt", PIX, str(dst)]
    return subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _close(p: subprocess.Popen) -> bool:
    p.stdin.close()
    return p.wait() == 0


def runs_of(frames: list[int]) -> list[list[int]]:
    """[first, last] of every stretch of consecutive frame numbers."""
    out: list[list[int]] = []
    for i in sorted({int(f) for f in frames}):
        if out and out[-1][1] == i - 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


def write_pairs(feed: Path, dst: Path, runs: list[list[int]]) -> int:
    """The frame before and the frame after each run, in order, as one clip. Returns how many pairs were written.
    A list of frame numbers is read as runs of one."""
    runs = [[r, r] if isinstance(r, int) else list(r) for r in runs]
    s = _stream(feed)
    first_of = {a: z for a, z in runs}
    out = _writer(dst, s["width"], s["height"], s["rate"])
    waiting: dict[int, bytes] = {}                          # frame number wanted as the "after" -> its "before"
    n = pairs = 0
    try:
        for frame in _frames(feed, s["width"], s["height"]):
            if n in waiting:                                # the first sound frame after a run
                out.stdin.write(waiting.pop(n))
                out.stdin.write(frame)
                pairs += 1
            if n + 1 in first_of and (n + 1 not in waiting):
                waiting[first_of[n + 1] + 1] = frame        # a run starts with the next frame: this is its "before"
            n += 1
    finally:
        ok = _close(out)
    return pairs if ok else 0


def _planes(buf: bytes, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = width * height
    c = (width // 2) * (height // 2)
    a = np.frombuffer(buf, "<u2")
    return (a[:y].reshape(height, width), a[y:y + c].reshape(height // 2, width // 2),
            a[y + c:y + 2 * c].reshape(height // 2, width // 2))


def blend_region(stored: bytes, made: bytes, width: int, height: int) -> tuple[bytes, float]:
    """The stored frame with only the part that differs from the made frame taken from it. Returns the frame and
    the share of the picture that was taken (1.0 = the whole made frame)."""
    ys, us, vs = _planes(stored, width, height)
    ym, um, vm = _planes(made, width, height)
    size = (REGION_GRID[1], REGION_GRID[0])
    apart = cv2.resize(np.abs(ys.astype(np.float32) - ym.astype(np.float32)) / 4.0, size, interpolation=cv2.INTER_AREA)
    out = apart > REGION_OUT
    if float(out.mean()) >= REGION_WHOLE:
        return made, 1.0
    if not out.any():
        return stored, 0.0
    wide = cv2.dilate(out.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)   # one cell around
    mask = np.clip(cv2.resize(wide, (width, height), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)
    half = cv2.resize(mask, (width // 2, height // 2), interpolation=cv2.INTER_AREA)

    def mix(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> bytes:
        return np.clip(a.astype(np.float32) * (1 - m) + b.astype(np.float32) * m + 0.5, 0, 1023).astype("<u2").tobytes()

    return mix(ys, ym, mask) + mix(us, um, half) + mix(vs, vm, half), float(mask.mean())


def compose(feed: Path, mids: dict | Path, dst: Path, runs: list, *, regions: bool = False) -> tuple[int, int, float]:
    """`feed` with the frames of each run replaced by the frames RIFE made between the run's two neighbours.
    mids: {run length: clip}; in the clip of length L the pair of run k was multiplied by L + 1, so its frames
    sit at 2k(L+1) + 1 ... + L. A single clip is read as that of runs of one. regions: see the module text.
    Returns (frames written, frames replaced, picture taken from made frames, in frames' worth)."""
    runs = [[r, r] if isinstance(r, int) else list(r) for r in runs]
    if not isinstance(mids, dict):
        mids = {1: mids}
    s = _stream(feed)
    w, h = s["width"], s["height"]
    place: dict[int, tuple[int, int]] = {}                  # frame -> (run length, index in that length's clip)
    seen: dict[int, int] = {}
    for a, z in sorted(runs):
        length = z - a + 1
        k = seen.get(length, 0)
        seen[length] = k + 1
        for j in range(length):
            place[a + j] = (length, 2 * k * (length + 1) + 1 + j)
    readers = {length: _frames(Path(clip), w, h) for length, clip in mids.items()}
    at = {length: -1 for length in readers}                 # index of the last frame taken from each clip
    out = _writer(dst, w, h, s["rate"], audio_from=feed)
    n = replaced = 0
    taken = 0.0
    try:
        for frame in _frames(feed, w, h):
            target = place.get(n)
            if target is not None and target[0] in readers:
                length, index = target
                new = None
                while at[length] < index:
                    new = next(readers[length], None)
                    at[length] += 1
                    if new is None:
                        break
                if new is not None:
                    if regions:
                        frame, share = blend_region(frame, new, w, h)
                    else:
                        frame, share = new, 1.0
                    replaced += 1
                    taken += share
            out.stdin.write(frame)
            n += 1
    finally:
        for r in readers.values():
            r.close()
        ok = _close(out)
    return (n, replaced, taken) if ok else (0, 0, 0.0)


def rebuild(feed: Path, dst: Path, ghosts: list[int], work: Path, *, drawn: bool = True,
            log: Callable[[str], None] = print, what: str = "ghost frames in movement", regions: bool = False) -> dict:
    """Write `dst`: `feed` (lossless) with the listed frames made again from their neighbours. Frames listed side
    by side are a run, made from the frames either side of it (up to RUN_MAX; a longer run is an error: the
    caller leaves those out). work: a folder of its own (each stage writes into a sub-folder that does not hold
    its input). what: how the log names the frames. regions: keep the stored frame where it agrees with the
    made one (see the module text)."""
    feed, dst, work = Path(feed), Path(dst), Path(work)
    runs = runs_of(ghosts)
    count = sum(z - a + 1 for a, z in runs)
    info: dict = {"asked": count, "runs": sum(1 for a, z in runs if z > a)}
    if any(z - a + 1 > RUN_MAX for a, z in runs):
        return info | {"ok": False, "error": f"a run of more than {RUN_MAX} frames cannot be made from its neighbours"}
    for sub in ("pairs", "mids"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    mids: dict[int, Path] = {}
    made_files = []
    for length in sorted({z - a + 1 for a, z in runs}):
        mine = [r for r in runs if r[1] - r[0] + 1 == length]
        pairs = work / "pairs" / ("pairs.mkv" if length == 1 else f"pairs_{length}.mkv")
        wrote = write_pairs(feed, pairs, mine)
        if wrote != len(mine):
            return info | {"ok": False, "error": f"{wrote} of {len(mine)} neighbour pairs could be written"}
        clip = work / "mids" / ("mids.mkv" if length == 1 else f"mids_{length}.mkv")
        r = video2x.run(pairs, clip, "rife", content="anime" if drawn else "real", fps_multiplier=length + 1,
                        model=RIFE_MODEL, codec="ffv1", extra={})
        if not r.ok:
            return info | {"ok": False, "error": f"RIFE failed: {r.error}"}
        got = media.probe(clip, exact=True)["frames"]
        # 2 frames per pair, multiplied: the last pair's frames end at 2(k-1)(L+1) + L; one fewer is tolerated
        if got < 2 * (len(mine) - 1) * (length + 1) + length + 1 - 1:
            return info | {"ok": False, "error": f"RIFE returned {got} frames for {len(mine)} pairs (runs of {length})"}
        mids[length] = clip
        made_files += [pairs, clip]
    frames, replaced, taken = compose(feed, mids, dst, runs, regions=regions)
    if frames:
        # the frames were written anew from a pipe, which starts the video at zero: a feed whose sound starts
        # later or earlier than its picture lost that offset here, and the copy's check refused it (a test cut
        # with a 2.6 s offset, 2026-11-22)
        media.fix_av_offset(feed, dst)
    log(f"   {what}: {replaced} of {frames} frames made again from their neighbours"
        + (f" ({taken:.1f} frames' worth of picture taken from the made frames)" if regions else ""))
    for f in made_files:
        f.unlink(missing_ok=True)
    return info | {"ok": bool(frames) and replaced == count, "frames": frames, "replaced": replaced,
                   **({"picture_created": round(taken, 2)} if regions else {}),
                   "error": None if replaced == count else f"{replaced} of {count} frames were replaced"}
