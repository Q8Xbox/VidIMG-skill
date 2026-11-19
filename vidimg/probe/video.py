"""analyze_video: measured profile of a video (PLAN.md sections 4.1 and 5.2).

Uses the system ffmpeg's analysis filters (idet, blockdetect, blurdetect, scdet, mpdecimate) on a
bounded sample, plus ffprobe facts and face/noise measurements on a few extracted frames.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .. import faces
from .. import media
from . import blend as blend_probe
from .image import anime_score, noise_sigma


@dataclass
class VideoProfile:
    path: str
    width: int
    height: int
    fps: float
    frames: int
    duration: float
    codec: str | None
    bitrate_kbps: float | None
    bits_per_pixel: float | None          # bitrate / (w*h*fps): main compression signal
    interlaced_ratio: float               # share of sampled frames idet calls TFF/BFF
    field_order: str | None               # tff | bff | progressive (from idet majority)
    blockiness: float | None              # blockdetect mean (compression artefacts)
    blur: float | None                    # blurdetect mean
    scene_cuts: list[float] = field(default_factory=list)   # seconds, in the sampled span
    duplicate_ratio: float = 0.0          # mpdecimate drop share (repeated frames)
    repeated_field_ratio: float = 0.0     # idet repeated top+bottom fields: ~0.4 for hard 3:2 telecine
    blend_share: float = 0.0              # share of moving frames (bobbed fields when interlaced) that are blends
    blend: dict = field(default_factory=dict)   # probe.blend statistics (windows, weights, decision)
    noise_sigma: float = 0.0              # median over sampled frames
    faces_max: int = 0                    # most faces seen in one sampled frame
    face_min_height: int = 0
    anime_score: float = 0.0
    audio_streams: int = 0
    sample_seconds: float = 0.0
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _system_ffmpeg() -> str:
    """The system ffmpeg (full build) has the analysis filters (libvmaf, idet, scdet)."""
    return media.ffmpeg()


def _filter_log(src: Path, vf: str, seconds: float) -> str:
    out = subprocess.run(
        [_system_ffmpeg(), "-hide_banner", "-nostats", "-t", f"{seconds:g}", "-i", str(src), "-vf", vf,
         "-an", "-f", "null", "-"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=900, errors="replace",
    )
    return out.stderr


def _idet(src: Path, seconds: float) -> tuple[float, str | None, float]:
    log = _filter_log(src, "idet", seconds)
    rep = re.findall(r"Repeated Fields: Neither:\s*(\d+)\s*Top:\s*(\d+)\s*Bottom:\s*(\d+)", log)
    repeated = 0.0
    if rep:
        nei, top, bot = map(int, rep[-1])
        repeated = round((top + bot) / max(1, nei + top + bot), 3)
    # ffmpeg prints a summary for an initial empty graph first: use the last one
    found = re.findall(r"Multi frame detection: TFF:\s*(\d+)\s*BFF:\s*(\d+)\s*Progressive:\s*(\d+)", log)
    if not found:
        return 0.0, None, repeated
    tff, bff, prog = map(int, found[-1])
    if tff + bff + prog == 0:
        return 0.0, None, repeated
    total = tff + bff + prog
    order = "tff" if tff >= max(bff, prog) else "bff" if bff >= prog else "progressive"
    return round((tff + bff) / total, 3), order, repeated


def _mean_metric(src: Path, vf: str, key: str, seconds: float) -> float | None:
    log = _filter_log(src, vf, seconds)
    found = re.findall(rf"{key}:\s*([0-9.]+)", log)
    return float(found[-1]) if found else None


def _scenes(src: Path, seconds: float) -> list[float]:
    log = _filter_log(src, "scdet=threshold=10", seconds)
    return [round(float(t), 3) for t in re.findall(r"lavfi\.scd\.time:\s*([0-9.]+)", log)]


def _duplicates(src: Path, seconds: float, frames_in_span: int) -> float:
    log = _filter_log(src, "mpdecimate,showinfo", seconds)
    kept = len(re.findall(r"\bn:\s*\d+", log))
    return round(max(0.0, 1 - kept / max(1, frames_in_span)), 3) if kept else 0.0


def _sample_frames(src: Path, info: dict, n: int = 6) -> list[np.ndarray]:
    frames = []
    with tempfile.TemporaryDirectory(prefix="vidimg_vp_") as tmp:
        for i in range(n):
            t = info["duration"] * (i + 0.5) / n
            p = Path(tmp) / f"f{i}.png"
            subprocess.run([_system_ffmpeg(), "-hide_banner", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(src),
                            "-frames:v", "1", "-y", str(p)], stdin=subprocess.DEVNULL, timeout=120)
            img = cv2.imread(str(p))
            if img is not None:
                frames.append(img)
    return frames


def analyze(path: Path, *, sample_seconds: float = 60.0) -> VideoProfile:
    path = Path(path)
    info = media.probe(path)
    span = min(sample_seconds, info["duration"] or sample_seconds)
    fmt = subprocess.run([media.ffprobe(), "-v", "error", "-show_entries", "format=bit_rate", "-of",
                          "default=nw=1:nk=1", str(path)], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    try:
        bitrate = round(int(fmt.stdout.strip()) / 1000, 1)
    except ValueError:
        bitrate = None
    inter, order, repeated = _idet(path, span)
    try:       # blended fields from a standards conversion: only sRestore fixes them (probe/blend.py)
        bl = blend_probe.analyze(path, interlaced=inter >= 0.3, field_order=order).to_dict()
    except Exception as e:  # noqa: BLE001  (an analysis aid; never fail the whole profile on it)
        bl = {"error": f"{type(e).__name__}: {e}"}
    frames = _sample_frames(path, info)
    face_counts, face_heights, sig = [], [], []
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        sig.append(noise_sigma(g))
        if faces.available():
            found = faces.detect(f)
            face_counts.append(len(found))
            face_heights += [int(x.box[3]) for x in found]
    p = VideoProfile(
        path=str(path), width=info["width"], height=info["height"], fps=info["fps"], frames=info["frames"],
        duration=round(info["duration"], 3), codec=info["codec"], bitrate_kbps=bitrate,
        bits_per_pixel=(round(bitrate * 1000 / (info["width"] * info["height"] * info["fps"]), 4)
                        if bitrate and info["width"] and info["height"] and info["fps"] else None),
        interlaced_ratio=inter, field_order=order, repeated_field_ratio=repeated,
        blend_share=float(bl.get("share") or 0.0), blend=bl,
        blockiness=_mean_metric(path, "blockdetect", r"block mean", span),
        blur=_mean_metric(path, "blurdetect", r"blur mean", span),
        scene_cuts=_scenes(path, span), duplicate_ratio=_duplicates(path, span, round(span * (info["fps"] or 0))),
        noise_sigma=round(float(np.median(sig)), 2) if sig else 0.0,
        faces_max=max(face_counts, default=0), face_min_height=min(face_heights, default=0),
        anime_score=round(float(np.median([anime_score(f) for f in frames])), 3) if frames else 0.0,
        audio_streams=info["audio_streams"], sample_seconds=span,
    )
    p.flags = _flags(p)
    return p


def _flags(p: VideoProfile) -> list[str]:
    f = []
    # blended fields/frames (standards conversion): un-blend first. A blend-converted interlaced source can
    # show telecine-like repeated fields (33 % on a synthetic NTSC film -> PAL clip), but IVTC cannot fix
    # blends, so it is handled as interlaced (double-rate QTGMC) followed by sRestore
    if p.blend.get("is_blended"):
        f.append("blended")
        if p.interlaced_ratio >= 0.3:
            f.append("interlaced")
    # hard 3:2 telecine: idet's multi-frame vote calls it TFF/BFF (119/120 frames on a synthetic
    # 24p->29.97 clip) but 42-46% of fields are repeats; true interlace and progressive show 0-3%.
    # Telecine must be field-matched (IVTC), never deinterlaced (Doom9, Didee; measured 21 dB vs 46 dB)
    if "blended" in f:
        pass
    elif p.interlaced_ratio >= 0.3 and p.repeated_field_ratio >= 0.2:
        f.append("telecine")
    elif p.interlaced_ratio >= 0.5:
        f.append("interlaced")
    if (p.height or 0) <= 480:
        f.append("sd")
    # H.264-class codecs: < 0.04 bits/pixel shows visible compression (600 kbps 720p24 = 0.024);
    # blockdetect alone barely moves on real footage (1.04 at 600k vs 1.18 at 80k); both to calibrate
    if (p.bits_per_pixel is not None and p.bits_per_pixel < 0.04) or (p.blockiness or 0) > 1.15:
        f.append("compressed")
    if p.noise_sigma > 2.0:
        f.append("noisy")
    if p.duplicate_ratio > 0.1:
        f.append("duplicate_frames")
    if p.faces_max:
        f.append("faces")
    if p.anime_score >= 0.6 and p.faces_max == 0:   # a real face vetoes anime routing
        f.append("anime_like")
    return f
