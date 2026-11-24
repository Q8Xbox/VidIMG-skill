"""Engine-neutral media helpers (free tools only): stream probing with the system ffprobe, encoder
settings, and colour-tag sidecars for intermediates whose container cannot store tags (NUT).

Moved out of the former Topaz Video AI adapter when the tool became free-only (2026-11-16).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise FileNotFoundError("ffmpeg not on PATH (winget install Gyan.FFmpeg)")
    return exe


def run_quiet(argv: list, **kw) -> subprocess.CompletedProcess:
    """Run a tool whose output is not wanted WITHOUT handing it this process's own output streams.
    Inside the MCP server those streams belong to the host application, and a child that inherits them can
    fail to start at all ([WinError 50] on every video analysis, 2026-11-20). What the tool says on its error
    stream is relayed through Python instead, so it still reaches the job log."""
    r = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                       text=True, encoding="utf-8", errors="replace", **kw)
    relay_errors(r.stderr)
    return r


def relay_errors(text: str | None) -> None:
    import sys
    if text and text.strip() and sys.stderr is not None:
        try:
            print(text.rstrip()[-4000:], file=sys.stderr, flush=True)
        except (OSError, ValueError):
            pass                                # a closed error stream must not fail the measurement


def ffprobe() -> str:
    exe = shutil.which("ffprobe")
    if not exe:
        raise FileNotFoundError("ffprobe not on PATH (winget install Gyan.FFmpeg)")
    return exe


def count_frames(path: Path) -> int | None:
    """Exact video frame count by counting packets (demux only, no decode: seconds even for long files)."""
    r = subprocess.run([ffprobe(), "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                        "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600, stdin=subprocess.DEVNULL)
    s = r.stdout.strip().split(",")[0]
    return int(s) if s.isdigit() else None


# The media the tool accepts: one list for the survey, the MCP tools, the command line and series folders (they each
# had their own and disagreed: .vob, .3gp and .webm were videos in one place and ignored in another). No .heic:
# neither OpenCV nor Pillow here can open it, so listing it only moved the failure to the middle of a run.
IMAGE_EXT = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"})
VIDEO_EXT = frozenset({".mp4", ".mkv", ".mov", ".avi", ".m4v", ".mpg", ".mpeg", ".wmv", ".ts", ".mts", ".m2ts", ".3gp",
                       ".webm", ".vob"})


def concat_line(path: Path) -> str:
    """One `file` line for ffmpeg's concat demuxer. Inside its single quotes an apostrophe ends the string, so it
    is written as '\\'' (a folder named "Dad's birthday" made every join fail after all chunks were rendered)."""
    return "file '" + Path(path).as_posix().replace("'", "'\\''") + "'\n"


def _rate(text: str | None) -> float:
    """"30000/1001" -> 29.97; missing, malformed or 0/0 (ffprobe's "unknown") -> 0.0."""
    num, _, den = (text or "").partition("/")
    try:
        n, d = float(num), float(den or 1)
    except ValueError:
        return 0.0
    return n / d if d else 0.0


def probe(path: Path, exact: bool = False) -> dict:
    """Stream facts used by the planners and by output verification.
    exact=True: when the container does not store the frame count (Matroska, many MP4/AVI from editors),
    count it instead of estimating duration x fps, and repair a duration that includes a non-zero start
    time (a clip cut from the middle with its timestamps kept reported 25120 frames for 120)."""
    out = subprocess.run([ffprobe(), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                         capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, stdin=subprocess.DEVNULL)
    data = json.loads(out.stdout or "{}")
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), {})
    # avg_frame_rate is "0/0" for some streams (single-picture or damaged headers): the nominal rate then
    fps = _rate(v.get("avg_frame_rate")) or _rate(v.get("r_frame_rate"))
    fmt = data.get("format", {})
    duration = float(v.get("duration") or fmt.get("duration") or 0)
    stored = str(v.get("nb_frames", "")).isdigit()
    frames = int(v["nb_frames"]) if stored else round(duration * fps)
    if exact and not stored:
        counted = count_frames(path)
        if counted:
            frames = counted
            start = float(fmt.get("start_time") or 0)
            if fps and start > 1 and abs(counted / fps - duration) > 1 and abs(counted / fps - (duration - start)) < 1:
                duration -= start
            elif fps and abs(counted / fps - duration) > max(1.0, 0.02 * duration):
                duration = counted / fps          # container duration stretched by another stream (subtitles)
            elif fps and not v.get("duration"):
                # The video stream stores no duration of its own (NUT, Matroska): the container's then also
                # covers an audio track that runs a little longer than the picture, and measured against a
                # source whose video stream does store one (MP4: 1613.12 s video, 1613.42 s audio) it failed
                # the duration gate on an identical frame count. Measure the picture itself.
                duration = counted / fps
    return {
        "width": v.get("width"), "height": v.get("height"), "fps": round(fps, 4), "frames": frames,
        "duration": duration, "codec": v.get("codec_name"), "pix_fmt": v.get("pix_fmt"),
        "field_order": v.get("field_order"), "color_primaries": v.get("color_primaries"),
        "color_transfer": v.get("color_transfer"),
        "audio_streams": sum(1 for s in streams if s.get("codec_type") == "audio"),
        "audio_codecs": [s.get("codec_name") for s in streams if s.get("codec_type") == "audio"],
        "subtitle_codecs": [s.get("codec_name") for s in streams if s.get("codec_type") == "subtitle"],
        "attachments": sum(1 for s in streams if s.get("codec_type") == "attachment"),
        "video_start": _start(v),
        "audio_start": next((_start(s) for s in streams if s.get("codec_type") == "audio"), None),
        "sar": _sar(v),
    }


def _sar(stream: dict) -> str:
    """Sample (pixel) aspect ratio as 'num:den'; missing, N/A or 0:1 means square pixels."""
    n, _, d = str(stream.get("sample_aspect_ratio") or "1:1").partition(":")
    return f"{n}:{d}" if n.isdigit() and d.isdigit() and int(n) > 0 and int(d) > 0 else "1:1"


def display_aspect(info: dict) -> float | None:
    """Width / height as shown on screen (storage size x pixel aspect)."""
    if not info.get("width") or not info.get("height"):
        return None
    n, d = (int(x) for x in info.get("sar", "1:1").split(":"))
    return info["width"] * n / (info["height"] * d)


def fix_aspect(src: Path, dst: Path) -> str | None:
    """Give `dst` the source's display aspect (remux, no re-encode: container tag plus the H.264/HEVC VUI).
    Needed because a raw-frame pipe carries no pixel aspect: the free engine turned a PAL 720x576 at 16:15
    (4:3 on screen) into 1440x1152 at 1:1, shown 5:4 - every face 6.7 % narrower (measured 2026-11-17)."""
    from fractions import Fraction
    a, b = probe(src), probe(dst)
    da, db = display_aspect(a), display_aspect(b)
    if not da or not db or abs(db / da - 1) < 0.002:
        return None
    sar = (Fraction(a["width"], a["height"]) * Fraction(*map(int, a["sar"].split(":")))
           * Fraction(b["height"], b["width"])).limit_denominator(65535)
    dar = (Fraction(b["width"], b["height"]) * sar).limit_denominator(65535)
    tmp = dst.with_name(dst.stem + ".aspect" + dst.suffix)
    args = [ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(dst), "-map", "0", "-c", "copy",
            "-map_metadata", "0", "-aspect", f"{dar.numerator}:{dar.denominator}"]
    bsf = {"h264": "h264_metadata", "hevc": "hevc_metadata"}.get(b["codec"])
    if bsf:
        args += ["-bsf:v", f"{bsf}=sample_aspect_ratio={sar.numerator}/{sar.denominator}"]
    if dst.suffix.lower() in (".mp4", ".mov", ".m4v"):
        args += ["-movflags", "+write_colr+faststart"]
    cp = subprocess.run([*args, str(tmp)], capture_output=True, text=True, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL)
    if cp.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        return "display aspect could not be restored: " + cp.stderr.strip()[-200:]
    tmp.replace(dst)
    return f"display aspect restored (pixel aspect {b['sar']} -> {sar.numerator}:{sar.denominator})"


def _start(stream: dict) -> float | None:
    try:
        return float(stream["start_time"])
    except (KeyError, TypeError, ValueError):
        return None


def av_offset(info: dict) -> float | None:
    """Video start minus audio start (seconds): the lip-sync relation a restoration must keep."""
    if info.get("video_start") is None or info.get("audio_start") is None:
        return None
    return info["video_start"] - info["audio_start"]


def fix_av_offset(src: Path, dst: Path) -> str | None:
    """Give `dst` the source's video/audio start relation (remux, no re-encode). Needed because both engines
    restart the video clock at 0 while the audio keeps its own: the owner's Sample04 starts its video 0.200 s
    after the audio, and Video2X 6.4.0 also re-times its output (0.081 s and 0.913 s shifts measured)."""
    a, b = probe(src), probe(dst)
    oa, ob = av_offset(a), av_offset(b)
    if oa is None or ob is None or abs(oa - ob) < 0.005:
        return None
    shift = oa - ob
    tmp = dst.with_name(dst.stem + ".sync" + dst.suffix)
    args = [ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-itsoffset", f"{shift:.6f}", "-i", str(dst),
            "-i", str(dst), "-map", "0:v", "-map", "1:a?", "-map", "1:s?", "-map", "1:t?", "-c", "copy", "-map_metadata", "1"]
    if dst.suffix.lower() in (".mp4", ".mov", ".m4v"):
        args += ["-movflags", "+write_colr+faststart"]
    cp = subprocess.run([*args, str(tmp)], capture_output=True, text=True, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL)
    if cp.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        return f"audio/video offset could not be restored ({shift:+.3f} s): " + cp.stderr.strip()[-200:]
    tmp.replace(dst)
    return f"audio/video offset restored ({shift:+.3f} s shift)"


# NVENC needs an NVIDIA GPU; nvenc() falls back to the free x265/x264 encoders elsewhere
ENCODERS: dict[str, list[str]] = {
    "hevc10": ["-c:v", "hevc_nvenc", "-profile:v", "main10", "-pix_fmt", "p010le", "-preset", "p7",
               "-tune", "hq", "-rc", "constqp", "-qp", "16"],
    "h264": ["-c:v", "h264_nvenc", "-profile:v", "high", "-pix_fmt", "yuv420p", "-preset", "p7",
             "-tune", "hq", "-rc", "constqp", "-qp", "16"],
    "prores": ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le", "-vendor", "apl0"],
    "ffv1": ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "yuv444p10le"],      # lossless intermediate
    # lossless 10-bit intermediate that MP4 chunks can hold (the texture pass reads it; deleted afterwards)
    "lossless": ["-c:v", "hevc_nvenc", "-preset", "p4", "-tune", "lossless", "-pix_fmt", "p010le"],
    # final encode after texture return. NVENC at QP 16 flattened low-contrast texture by itself (flat area between
    # the lines 0.21 -> 0.34 on a lossless input, QP 12 / adaptive quantisation barely better); x265 medium with the
    # grain tune kept it (0.19, source 0.18) at 27 fps for 720x576 x2, about 1.7x the file size
    "hevc10-grain": ["-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-crf", "14", "-preset", "medium", "-tune", "grain",
                     "-x265-params", "log-level=error"],
}
CPU_FALLBACK = {
    "hevc10": ["-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-crf", "14", "-preset", "slow"],
    "h264": ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "14", "-preset", "slow"],
    "lossless": ["-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-pix_fmt", "yuv420p10le"],
}
_NVENC: bool | None = None


def nvenc() -> bool:
    global _NVENC
    if _NVENC is None:
        r = subprocess.run([ffmpeg(), "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=s=256x256:d=0.1",
                            "-c:v", "hevc_nvenc", "-f", "null", "-"], capture_output=True, stdin=subprocess.DEVNULL)
        _NVENC = r.returncode == 0
    return _NVENC


_FILTERS: str | None = None


def has_filter(name: str) -> bool:
    """Does this ffmpeg build have the filter? (asked once per process; False without ffmpeg)."""
    global _FILTERS
    if _FILTERS is None:
        ff = shutil.which("ffmpeg")
        _FILTERS = subprocess.run([ff, "-hide_banner", "-filters"], capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", stdin=subprocess.DEVNULL).stdout if ff else ""
    return f" {name} " in _FILTERS


def encoder_args(name: str) -> list[str]:
    if name not in ENCODERS:
        raise ValueError(f"unknown encoder {name!r}; known: {sorted(ENCODERS)}")
    if name in CPU_FALLBACK and not nvenc():
        return list(CPU_FALLBACK[name])
    return list(ENCODERS[name])


# ----------------------------------------------------------------------------- colour tags
COLOUR_KEYS = {"color_range": ("range", "-color_range"), "color_primaries": ("color_primaries", "-color_primaries"),
               "color_transfer": ("color_trc", "-color_trc"), "color_space": ("colorspace", "-colorspace")}


def colour_sidecar(path: Path) -> Path:
    return Path(path).with_name(Path(path).name + ".colour.json")


def write_colour_sidecar(path: Path, colour: dict) -> dict:
    """Record the source's colour tags next to an intermediate whose container cannot store them (NUT).
    Without them an untagged SD intermediate was read with the wrong matrix (measured: 1.3 VMAF / 1 dB
    on 12/12 segments, 2026-11-15)."""
    known = {k: v for k, v in colour.items() if k in COLOUR_KEYS and v and v not in ("unknown", "reserved")}
    colour_sidecar(path).write_text(json.dumps(known), encoding="utf-8")
    return known


def colour_hint(path: Path) -> dict:
    p = colour_sidecar(path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            return {}
    return {}


def colour_args(colour: dict) -> tuple[str, list[str]]:
    """(setparams filter prefix, output tag options) for known colour tags."""
    if not colour:
        return "", []
    sp = ":".join(f"{COLOUR_KEYS[k][0]}={v}" for k, v in colour.items() if k in COLOUR_KEYS)
    opts = [x for k, v in colour.items() if k in COLOUR_KEYS for x in (COLOUR_KEYS[k][1], v)]
    return (f"setparams={sp}," if sp else ""), opts
