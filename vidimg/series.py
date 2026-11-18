"""Series batches: restore the numbered episodes of one folder into ONE output folder, then write one
comparison report for the whole series.

* episodes are found by the number in the file name (`S01E05`, `Ep 05`, `Episode 5`, `Title - 05 [..]`);
* each episode runs the normal guarded video pipeline (`pipeline_video.restore_video`): previews,
  approved model order, full run, final guard. A result the guard fails is not delivered (the original
  stays the deliverable) and the batch moves on to the next episode;
* after delivery: a frame-alignment check (the result, shrunk back, must match the original frame at
  the same index, not a neighbour) and before/after frames + measurements for the report;
* resumable: `batch_status.json` in the output folder records every episode. A rerun skips finished
  episodes, and an episode stopped mid-run continues from the pipeline's own checkpoints in `work/`.

Layout of the output folder: `<stem>_restored.<ext>` per episode, `records/`, `reports/` (one guard page
per episode), `comparison/EpNN/`, `sheets/`, `batch_status.json`, and
`Series Comparison Report - Episodes AA-BB.html`.
"""

from __future__ import annotations

import ctypes
import json
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np

VIDEO_EXT = {".mkv", ".mp4", ".avi", ".mov", ".m4v", ".ts", ".m2ts", ".wmv", ".webm", ".mpg", ".mpeg", ".vob"}

_PATTERNS = (
    r"[Ss]\d{1,2}\s*[Ee](\d{1,3})(?!\d)",                              # S01E05
    r"(?:^|[\s._\-\[(])(?:[Ee][Pp]|Episode)\s*\.?\s*(\d{1,3})(?!\d)",  # Ep05, EP 5, Episode 5
    r"\s-\s(\d{1,3})(?=[\s.\[(v]|$)",                                  # Title - 05 [tags]
)


def episode_number(name: str) -> int | None:
    """The episode number in a file name, or None when it is not clear."""
    stem = Path(name).stem
    for pattern in _PATTERNS:
        m = re.search(pattern, stem)
        if m:
            return int(m.group(1))
    # fallback: exactly one free-standing 1-3 digit number outside [tags] and (tags)
    clean = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", stem)
    nums = re.findall(r"(?<![\dA-Za-z])(\d{1,3})(?![\dA-Za-z])", clean)
    return int(nums[0]) if len(nums) == 1 else None


def parse_episodes(spec: str | None) -> set[int] | None:
    """'1-26' or '1,3,5-7' -> a set of episode numbers; None/'' -> every episode."""
    if not spec:
        return None
    wanted: set[int] = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        a, b = int(lo), int(hi or lo)
        if a > b:
            raise ValueError(f"bad episode range {part!r}")
        wanted.update(range(a, b + 1))
    return wanted


def find_episodes(src_dir: Path, wanted: set[int] | None = None) -> tuple[list[tuple[int, Path]], list[str]]:
    """Numbered episodes in src_dir (not recursive), sorted by number; plus notes on skipped files."""
    found: dict[int, Path] = {}
    notes: list[str] = []
    for f in sorted(Path(src_dir).iterdir()):
        if not f.is_file() or f.suffix.lower() not in VIDEO_EXT:
            continue
        n = episode_number(f.name)
        if n is None:
            notes.append(f"no episode number found, skipped: {f.name}")
        elif wanted is None or n in wanted:
            if n in found:
                notes.append(f"two files for episode {n:02d}; using {found[n].name}, skipped {f.name}")
            else:
                found[n] = f
    if wanted:
        missing = sorted(wanted - set(found))
        if missing:
            notes.append(f"episodes not found: {', '.join(f'{n:02d}' for n in missing)}")
    return sorted(found.items()), notes


def default_out_dir(src_dir: Path, episodes: list[tuple[int, Path]], scale: float | None) -> Path:
    """A new folder inside the series folder, e.g. 'VidIMG x2 - Episodes 01-26'."""
    tag = f"x{scale:g}" if scale else "restored"
    first, last = episodes[0][0], episodes[-1][0]
    span = f"Episode {first:02d}" if first == last else f"Episodes {first:02d}-{last:02d}"
    return Path(src_dir) / f"VidIMG {tag} - {span}"


# ---- checks after delivery ----------------------------------------------------------------------------

def _probe(path: Path) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,nb_frames,r_frame_rate:format=duration", "-of", "json", str(path)],
                       capture_output=True, text=True)
    j = json.loads(r.stdout or "{}")
    s, f = (j.get("streams") or [{}])[0], j.get("format") or {}
    num, _, den = (s.get("r_frame_rate") or "0/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    dur = float(f.get("duration") or 0)
    frames = int(s.get("nb_frames") or 0) or int(round(dur * fps))
    return {"w": int(s.get("width") or 0), "h": int(s.get("height") or 0), "fps": fps, "duration": dur,
            "frames": frames}


def _frames(path: Path, start: float, count: int, w: int, h: int) -> list[np.ndarray]:
    """`count` decoded frames from `start` seconds (accurate seek), at the stream's storage size."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.6f}", "-i", str(path), "-frames:v", str(count),
                          "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         capture_output=True).stdout
    size = w * h * 3
    return [np.frombuffer(raw[k * size:(k + 1) * size], np.uint8).reshape(h, w, 3) for k in range(len(raw) // size)]


def decisive_offset(a: list[np.ndarray], b: list[np.ndarray], reach: int = 3, motion_db: float = 40.0,
                    margin_db: float = 0.5) -> int | None:
    """Offset of the result sequence b against the original sequence a (same start, same size), judged on
    the first original frame that differs from its neighbours (a still scene cannot tell frames apart)
    and only when the best match beats the runner-up clearly. None = no decisive frame in this window."""
    import cv2
    for m in range(reach, min(len(a), len(b)) - reach):
        if max(cv2.PSNR(a[m], a[m - 1]), cv2.PSNR(a[m], a[m + 1])) > motion_db:
            continue
        row = [cv2.PSNR(a[m], x) for x in b[m - reach:m + reach + 1]]
        best = int(np.argmax(row))
        if row[best] - sorted(row)[-2] >= margin_db:
            return best - reach
    return None


def alignment_offsets(original: Path, result: Path, centres: list[int] | None = None, reach: int = 3,
                      window: int = 45, tries: int = 5) -> list[int | None]:
    """Near each centre frame of the original: does the result frame with the same index show the same
    picture? 0 everywhere = no dropped, added or shifted frames.

    Both files are decoded from the same moment (half a frame early, never on a frame boundary, so files
    with millisecond and exact timestamps start on the same frame); the result is shrunk to the original
    size and compared by PSNR with its neighbours -reach..+reach. Still scenes are skipped (moving on by
    3 s, up to `tries` windows). None = no moving, decisive frame found near that centre."""
    import cv2
    po, pr = _probe(original), _probe(result)
    if centres is None:
        centres = [int(po["frames"] * 0.2), int(po["frames"] * 0.6)]
    fps = po["fps"] or 25.0
    offsets: list[int | None] = []
    for c in centres:
        found = None
        for k in range(tries):
            start = max(0.0, (c + k * 3 * fps - 0.5) / fps)
            a = _frames(original, start, window, po["w"], po["h"])
            b = [cv2.resize(x, (po["w"], po["h"]), interpolation=cv2.INTER_AREA)
                 for x in _frames(result, start, window, pr["w"], pr["h"])]
            found = decisive_offset(a, b, reach)
            if found is not None:
                break
        offsets.append(found)
    return offsets


# ---- the batch ----------------------------------------------------------------------------------------

def _keep_awake() -> None:
    """Ask Windows not to idle-sleep while this process runs (like a video player; released on exit)."""
    if sys.platform == "win32":
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        except Exception:
            pass


def restore_series(src_dir: Path, out: Path | None = None, wanted: set[int] | None = None, *,
                   scale: float | None = 2, content: str = "auto", model: str | None = None,
                   encoder: str = "hevc10", log: Callable[[str], None] = print) -> dict:
    """Restore every wanted episode into one folder and write the series report. Returns the status dict."""
    from . import pipeline_video
    from .report import compare, series_report

    src_dir = Path(src_dir)
    episodes, notes = find_episodes(src_dir, wanted)
    for n in notes:
        log(f"note: {n}")
    if not episodes:
        raise ValueError(f"no numbered episodes found in {src_dir}")
    out = Path(out) if out else default_out_dir(src_dir, episodes, scale)
    for sub in ("records", "reports", "comparison"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    status_file = out / "batch_status.json"
    status = json.loads(status_file.read_text(encoding="utf-8")) if status_file.exists() else {}
    status.setdefault("_options", {})
    status["_options"].update({"source": str(src_dir), "scale": scale, "content": content, "model": model,
                               "episodes": [n for n, _ in episodes]})

    def save() -> None:
        status_file.write_text(json.dumps(status, indent=1, default=str), encoding="utf-8")

    _keep_awake()
    log(f"{len(episodes)} episodes -> {out}")
    for n, src in episodes:
        key = f"{n:02d}"
        st = status.get(key, {})
        if st.get("delivered") and Path(st["delivered"]).exists() and st.get("compared"):
            log(f"ep {key}: already done, skipped")
            continue
        if st.get("source") and st["source"] != str(src):
            st = {}                                       # a different file now carries this number
        if st.get("source") and "delivered" in st and not st["delivered"] and not st.get("error"):
            log(f"ep {key}: not delivered last time (the guard kept the original); "
                f"remove its entry from batch_status.json to try again")
            continue
        t0 = time.time()
        try:
            if not (st.get("delivered") and Path(st["delivered"]).exists()):
                log(f"ep {key}: restoring {src.name}")
                rec = pipeline_video.restore_video(src, out, scale=scale, content=content, model=model,
                                                   encoder=encoder, log=lambda m: log(f"ep {key}: {m.strip()}"))
                d = rec.to_dict()
                (out / "records" / f"{src.stem}.json").write_text(
                    json.dumps(d, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
                compare.video_page(d, out / "reports" / f"{src.stem}.html")
                st = {"source": str(src), "delivered": d.get("delivered"), "passed": d.get("passed"),
                      "chosen": d.get("chosen"), "notes": d.get("notes"),
                      "minutes": round((time.time() - t0) / 60, 1),
                      "gates": {k: [g["status"], g["value"]] for k, g in (d.get("final_gates") or {}).items()}}
                status[key] = st
                save()
                log(f"ep {key}: {'DELIVERED' if st['delivered'] else 'NOT delivered'} with {st['chosen']} "
                    f"in {st['minutes']} min")
            if st.get("delivered"):
                dst = Path(st["delivered"])
                st["alignment_offsets"] = alignment_offsets(src, dst)
                series_report.episode_frames(src, dst, out / "comparison" / f"Ep{key}")
                st["compared"] = True
                st.pop("error", None)
                status[key] = st
                save()
                log(f"ep {key}: compared; alignment offsets {st['alignment_offsets']}")
        except Exception as e:  # one bad episode must not stop the series
            status[key] = {**st, "source": str(src), "error": f"{type(e).__name__}: {e}"}
            save()
            log(f"ep {key}: ERROR {e}\n{traceback.format_exc()}")

    report = series_report.write(out)
    done = sorted(k for k, v in status.items() if not k.startswith("_") and v.get("delivered"))
    log(f"SERIES FINISHED: {len(done)} delivered of {len(episodes)}; report: {report}")
    return status
