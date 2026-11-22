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
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Iterator

from . import media
from .adapters import video2x

PIX = "yuv420p10le"
RIFE_MODEL = "rife-v4.26"


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


def write_pairs(feed: Path, dst: Path, ghosts: list[int]) -> int:
    """The frame before and the frame after each ghost, in order, as one clip. Returns how many pairs were written."""
    s = _stream(feed)
    want = set(ghosts)
    out = _writer(dst, s["width"], s["height"], s["rate"])
    before = middle = None
    n = pairs = 0
    try:
        for after in _frames(feed, s["width"], s["height"]):
            if n - 1 in want and before is not None:        # `middle` is frame n-1: the ghost; its neighbours are at hand
                out.stdin.write(before)
                out.stdin.write(after)
                pairs += 1
            before, middle = middle, after
            n += 1
    finally:
        ok = _close(out)
    return pairs if ok else 0


def compose(feed: Path, mids: Path, dst: Path, ghosts: list[int]) -> tuple[int, int]:
    """`feed` with each ghost replaced by the frame RIFE made between its neighbours (index 4k+1 of `mids`).
    Returns (frames written, frames replaced)."""
    s = _stream(feed)
    order = sorted(ghosts)
    place = {g: 4 * k + 1 for k, g in enumerate(order)}
    made = _frames(mids, s["width"], s["height"])
    at = -1                                             # index of the last frame taken from `mids`
    out = _writer(dst, s["width"], s["height"], s["rate"], audio_from=feed)
    n = replaced = 0
    try:
        for frame in _frames(feed, s["width"], s["height"]):
            target = place.get(n)
            if target is not None:
                new = None
                while at < target:
                    new = next(made, None)
                    at += 1
                    if new is None:
                        break
                if new is not None:
                    frame = new
                    replaced += 1
            out.stdin.write(frame)
            n += 1
    finally:
        made.close()
        ok = _close(out)
    return (n, replaced) if ok else (0, 0)


def rebuild(feed: Path, dst: Path, ghosts: list[int], work: Path, *, drawn: bool = True,
            log: Callable[[str], None] = print, what: str = "ghost frames in movement") -> dict:
    """Write `dst`: `feed` (lossless) with the listed frames made again from their neighbours.
    work: a folder of its own (each stage writes into a sub-folder that does not hold its input).
    what: how the log names the frames (ghosts, or damaged frames: the same is done to both)."""
    feed, dst, work = Path(feed), Path(dst), Path(work)
    ghosts = sorted({int(g) for g in ghosts})
    info: dict = {"asked": len(ghosts)}
    for sub in ("pairs", "mids"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pairs = work / "pairs" / "pairs.mkv"
    wrote = write_pairs(feed, pairs, ghosts)
    if wrote != len(ghosts):
        return info | {"ok": False, "error": f"{wrote} of {len(ghosts)} neighbour pairs could be written"}
    mids = work / "mids" / "mids.mkv"
    r = video2x.run(pairs, mids, "rife", content="anime" if drawn else "real", fps_multiplier=2, model=RIFE_MODEL,
                    codec="ffv1", extra={})
    if not r.ok:
        return info | {"ok": False, "error": f"RIFE failed: {r.error}"}
    got = media.probe(mids, exact=True)["frames"]
    if got < 4 * len(ghosts) - 2:                       # 2 frames per pair, doubled: 4k - 1 (one fewer is tolerated)
        return info | {"ok": False, "error": f"RIFE returned {got} frames for {len(ghosts)} pairs"}
    frames, replaced = compose(feed, mids, dst, ghosts)
    if frames:
        # the frames were written anew from a pipe, which starts the video at zero: a feed whose sound starts
        # later or earlier than its picture lost that offset here, and the copy's check refused it (a test cut
        # with a 2.6 s offset, 2026-11-22)
        media.fix_av_offset(feed, dst)
    log(f"   {what}: {replaced} of {frames} frames made again from their neighbours")
    for f in (pairs, mids):
        f.unlink(missing_ok=True)
    return info | {"ok": bool(frames) and replaced == len(ghosts), "frames": frames, "replaced": replaced,
                   "error": None if replaced == len(ghosts) else f"{replaced} of {len(ghosts)} frames were replaced"}
