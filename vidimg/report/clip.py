"""Comparison clips: the source and the result playing side by side, for the person who judges the result.

A still sheet shows lines, colour and texture. It cannot show what only exists in movement: flicker, ghost
frames, a torn frame, a frame that was made again. Those were the things added to the tool last, and the only way
to judge them was to open two video files and try to keep them in step (owner run, 2026-11-22). So every delivered
video gets a short clip of a few moments, both pictures at the same height and in step, and a second file with
frames made again gets one that plays the places where frames were made, slowed down.

The clip is for a person: it is plain H.264 so that a browser plays it in the report. It is made from the
delivered files and changes nothing; when it cannot be made the report simply has none.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .. import imgfile, media

HEIGHT = 540                    # each picture's height in the clip (the result's own when that is smaller)
MOMENTS = (0.2, 0.5, 0.8)       # where in the video the moments of the ordinary clip are taken
SECONDS = 4.0                   # length of each moment
PLACES_MAX = 6                  # places where frames were made again that the second clip plays
PLACE_SECONDS = 1.2             # of the video per place ...
PLACE_LEAD = 0.5                # ... starting this long before the made frame
SLOW = 4.0                      # ... played this many times slower: a single frame is then visible
FPS_MAX = 60.0


def moments(duration: float, seconds: float = SECONDS) -> list[tuple[float, float]]:
    """(start, length) of the moments of an ordinary comparison: spread over the video, never past its end."""
    if not duration or duration <= seconds * 1.5:
        return [(0.0, max(0.5, float(duration or seconds)))]
    return [(round(max(0.0, min(duration - seconds, duration * k)), 2), seconds) for k in MOMENTS]


def places(frames: list[int], fps: float, duration: float) -> list[tuple[float, float]]:
    """(start, length) around frames that were made again: up to PLACES_MAX of them, spread over the list, each
    from a little before the frame. Frames close together share a place."""
    if not frames or not fps:
        return []
    starts: list[float] = []
    for f in sorted({int(x) for x in frames}):
        t = max(0.0, f / fps - PLACE_LEAD)
        if not starts or t >= starts[-1] + PLACE_SECONDS:
            starts.append(round(t, 3))
    step = max(1, -(-len(starts) // PLACES_MAX))
    return [(t, min(PLACE_SECONDS, max(0.2, duration - t))) for t in starts[::step][:PLACES_MAX] if t < duration]


def _matrix(path: Path, height: int) -> str:
    from .. import texture
    tag = (media.colour_hint(path) or {}).get("color_space") or texture._probe_stream(path).get("color_space")
    return texture._matrix(tag, height)


def _header(labels: list[str], widths: list[int], dst: Path) -> Path:
    cells = []
    for text, w in zip(labels, widths):
        bar = np.full((28, w, 3), 30, np.uint8)
        cv2.putText(bar, text.encode("ascii", "replace").decode()[:max(8, w // 10)], (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(bar)
    imgfile.imwrite(str(dst), np.hstack(cells))
    return dst


def command(panels: list[tuple[str, Path]], infos: list[dict], header: Path, start: float, seconds: float, dst: Path,
            *, slow: float = 1.0, height: int = HEIGHT) -> tuple[list[str], list[int]]:
    """The ffmpeg call for one moment: every picture from the same time, at the same height, in a row under its
    label. Returns the arguments and the pictures' widths."""
    h = min(height, infos[-1]["height"]) // 2 * 2
    widths = [max(2, round(h * (media.display_aspect(i) or i["width"] / i["height"]) / 2) * 2) for i in infos]
    fps = min(FPS_MAX, infos[-1]["fps"] or 25.0)
    args = [media.ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    for _, path in panels:
        args += ["-ss", f"{max(0.0, start):.3f}", "-t", f"{seconds:.3f}", "-i", str(path)]
    args += ["-loop", "1", "-t", f"{seconds * slow:.3f}", "-i", str(header)]
    parts = []
    for k, ((_, path), info, w) in enumerate(zip(panels, infos, widths)):
        parts.append(f"[{k}:v]scale={w}:{h}:flags=bicubic:in_color_matrix={_matrix(path, info['height'])}:"
                     f"out_color_matrix=bt709,setsar=1,setpts={slow:g}*(PTS-STARTPTS),fps={fps:g},format=yuv420p[p{k}]")
    row = "".join(f"[p{k}]" for k in range(len(panels)))
    parts.append(f"{row}hstack=inputs={len(panels)}[row]" if len(panels) > 1 else f"{row}null[row]")
    parts.append(f"[{len(panels)}:v]fps={fps:g},format=yuv420p[head]")
    parts.append("[head][row]vstack=shortest=1[v]")
    args += ["-filter_complex", ";".join(parts), "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "20", "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-movflags", "+faststart", str(dst)]
    return args, widths


def make(panels: list[tuple[str, Path]], dst: Path, spans: list[tuple[float, float]], *, slow: float = 1.0,
         height: int = HEIGHT) -> Path | None:
    """Write `dst`: the pictures of `panels` ([(label, file), ...]; the last one is the result and sets the size
    and the rate) side by side for each (start, length) of `spans`, one after the other. slow: played this many
    times slower. None when nothing could be written."""
    panels = [(label, Path(p)) for label, p in panels]
    dst = Path(dst)
    if not spans or not all(p.exists() for _, p in panels):
        return None
    try:
        infos = [media.probe(p) for _, p in panels]
        dst.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="vidimg_clip_") as tmp:
            tmp = Path(tmp)
            parts: list[Path] = []
            header = None
            for i, (start, seconds) in enumerate(spans):
                part = tmp / f"part{i}.mp4"
                if header is None:
                    _, widths = command(panels, infos, tmp / "h.png", start, seconds, part, slow=slow, height=height)
                    header = _header([label for label, _ in panels], widths, tmp / "h.png")
                args, _ = command(panels, infos, header, start, seconds, part, slow=slow, height=height)
                cp = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                    stdin=subprocess.DEVNULL, timeout=1800)
                if cp.returncode == 0 and part.exists() and part.stat().st_size > 0:
                    parts.append(part)
            if not parts:
                return None
            listing = tmp / "parts.txt"
            listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
            cp = subprocess.run([media.ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "concat",
                                 "-safe", "0", "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(dst)],
                                capture_output=True, text=True, encoding="utf-8", errors="replace",
                                stdin=subprocess.DEVNULL, timeout=600)
            return dst if cp.returncode == 0 and dst.exists() else None
    except Exception:  # noqa: BLE001  (an aid for the person; a run never fails on it)
        return None


def before_after(source: Path, result: Path, dst: Path) -> dict | None:
    """The ordinary clip of a delivered video: a few moments, the source beside the result."""
    try:
        spans = moments(media.probe(result).get("duration") or 0.0)
    except Exception:  # noqa: BLE001  (see make)
        return None
    made = make([("source", source), ("restored", result)], dst, spans)
    return {"path": str(made), "what": "the source beside the restored video, a few moments at normal speed"} if made else None


def made_again(faithful: Path, second: Path, dst: Path, frames: list[int]) -> dict | None:
    """The clip of a second file: the places where frames were made again, slowed down, the faithful file beside
    the second one."""
    try:
        info = media.probe(second)
        spans = places(frames, info.get("fps") or 0.0, info.get("duration") or 0.0)
    except Exception:  # noqa: BLE001  (see make)
        return None
    made = make([("faithful file", faithful), ("frames made again", second)], dst, spans, slow=SLOW)
    return {"path": str(made), "what": f"{len(spans)} of the places where frames were made again, {SLOW:g} times "
                                       "slower: the faithful file beside the second file"} if made else None
