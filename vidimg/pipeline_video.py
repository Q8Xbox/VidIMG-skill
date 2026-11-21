"""Video restoration pipeline (free tools only): analyse -> classical field handling -> plan -> guarded
previews -> full run (chunked and resumable when long) -> final guard -> report.

Engines: open restoration models on PyTorch CUDA (vidimg/adapters/free_sr.py) and Video2X for anime.
Field handling is classical and happens first (QTGMC / TFM+TDecimate via AviSynth+ or VapourSynth,
ffmpeg bwdif/fieldmatch as the last resort): the models only ever see progressive frames. Measured on the
person's footage: QTGMC 42.3 dB vs BWDIF 40.4 dB against ground truth; TFM+TDecimate bit-exact on clean
telecine. Candidates are tried on short preview segments first; the first one that passes the guard on
every preview is used for the full video, which is verified again at the end. If none passes, nothing is
delivered and the record explains why.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable

from . import cleanup, config
from .adapters import avisynth as avs
from .adapters import free_sr, video2x
from .adapters import vapoursynth as vsb
from . import media
from .adapters import AdapterResult, mark_output, safe_output_dir
from . import texture as texture_pass
from .guard.video import (_ffmpeg, _frames_by_index, verify_chroma, verify_cleanup, verify_crop, verify_deblend,
                          verify_derepeat, verify_mask, verify_spans, verify_video)
from .probe import blend as blend_probe
from .probe import chroma as chroma_probe
from .probe import framing as framing_probe
from .probe import video as probe

PREVIEW_SECONDS = 4.0
# long videos (phase 5): processed in chunks so a stop or crash costs one chunk, not hours
LONG_VIDEO_SECONDS = 300.0
CHUNK_SECONDS = 120.0
CHUNK_CONTEXT_SECONDS = 1.0
FREE_NONGAN_PIXELS_PER_S = 433_000   # measured: RealESRNet 0.47 fps on 1280x720 input (RTX 5060 Ti, tiled)
FREE_NONGAN_MAX_HOURS = 3.0          # longer estimated runs use the compact model only     # real neighbouring frames given to temporal models at each boundary, trimmed after
PREVIEW_COUNT = 3


@dataclass
class VideoAttempt:
    label: str
    passed_previews: int
    previews: int
    failed: list[list[str]] = field(default_factory=list)
    vmaf: list[float | None] = field(default_factory=list)


@dataclass
class VideoRecord:
    source: str
    profile: dict
    plan: str
    chosen: str | None = None
    delivered: str | None = None
    passed: bool = False
    final_gates: dict = field(default_factory=dict)
    attempts: list[VideoAttempt] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    sheets: list[str] = field(default_factory=list)
    prepass: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


CONTENT_CHOICES = ("auto", "anime", "real")


def target_size(p: probe.VideoProfile, scale: float | None = None) -> tuple[int | None, str]:
    """Output height: an explicit scale (e.g. 2 -> exactly twice the height); otherwise SD x2, 720p-class up
    to 1080p, larger left at size (enhance only)."""
    h = p.height or 0
    if scale:
        out = int(round(h * scale / 2)) * 2
        return out, f"x{scale:g} ({p.width}x{h} -> {out}p, requested)"
    if h <= 540:
        return h * 2, f"x2 ({p.width}x{h} -> {h * 2}p)"
    if h <= 800:
        return 1080, f"to 1080p ({p.width}x{h})"
    return None, f"enhance only ({p.width}x{h})"


def run_engine(src: Path, dst: Path, stages: list, encoder: str = "hevc10", **kw):
    """Free engines only: the open models (PyTorch) or Video2X's anime models, chosen by the stages."""
    anime = next((s for s in stages if isinstance(s, video2x.AnimeSR)), None)
    if anime is not None:
        r = video2x.run_stage(src, dst, anime, encoder)
    else:
        r = free_sr.run(src, dst, stages, encoder, **kw)
    if getattr(r, "ok", False) and kw.get("start") is None:   # chunks are re-timed by the join, not here
        note = media.fix_av_offset(src, dst)
        if note:
            r.details = {**(r.details or {}), "av_sync": note}
        note = media.fix_aspect(src, dst)
        if note:
            r.details = {**(r.details or {}), "aspect": note}
    return r


def is_anime(p: probe.VideoProfile, content: str = "auto") -> bool:
    """content: 'anime' / 'real' as decided by the person or by Claude looking at the frames sheet
    (vidimg analyze writes one); 'auto' trusts the probe, which is deliberately conservative: old,
    compressed cartoons score like real footage and YuNet finds 'faces' in drawings (owner's samples,
    2026-11-16), so auto keeps the real path unless the probe is sure."""
    if content not in CONTENT_CHOICES:
        raise ValueError(f"content must be one of {CONTENT_CHOICES}")
    return content == "anime" or (content == "auto" and "anime_like" in p.flags)


# Models a person can ask for by name (--model). Without a choice the pipeline keeps its approved order and
# takes the first candidate that passes the guard. A named model is the only candidate: if the guard rejects
# it, the original stays the deliverable (no silent fallback to another look).
MODEL_CHOICES = config.VIDEO_MODEL_CHOICES


def plan_candidates(p: probe.VideoProfile, scale: float | None = None, content: str = "auto",
                    model: str | None = None, strict: bool = True) -> tuple[str, list[tuple[str, Callable[[], list]]]]:
    """strict: the named model is the only candidate (a person insisted on it). Not strict: it goes first and the
    approved order follows if the guard rejects it (Claude's pick from the analysis, model_guide)."""
    why, cands = _plan_candidates(p, scale, content, model is not None)
    if model is None:
        return why, cands
    if model not in MODEL_CHOICES:
        raise ValueError(f"model must be one of {MODEL_CHOICES}")
    picked = [c for c in cands if c[0].split(" ", 1)[1] == model]
    if not picked:
        if strict:
            return f"{why}; requested model {model} does not apply to this content", []
        return f"{why}; chosen model {model} does not apply to this content: approved order instead", \
            [c for c in cands if c[0] != "anime cugan-pro"]
    if strict:
        return f"{why}; only {model} (requested)", picked
    rest = [c for c in cands if c is not picked[0] and c[0] != "anime cugan-pro"]
    return f"{why}; {model} first (chosen from the analysis), then the approved order", picked + rest


# Measured facts per model, for model_guide (Claude chooses from these and the frames sheet). Numbers: cartoon
# ground truth (PLAN.md 3.7), the person's 12 real segments (identity / VMAF), the owner's cartoon samples.
MODEL_FACTS = {
    "esrgan-v3": {"for": "anime", "speed": "fast (Video2X, GPU)",
                  "good": "cleanest line art on cartoons; all owner cartoon samples delivered with it (VMAF 94-98)",
                  "weak": "drawn realistic faces drift (identity 0.687 on a 1991 cartoon); smooths painted texture "
                          "(texture return puts it back)"},
    "cugan": {"for": "anime", "speed": "fast (Video2X, GPU)",
              "good": "Real-CUGAN without denoise: similar to anime-v3, a little softer lines, keeps more of the source's "
                      "look",
              "weak": "cartoon ground truth VMAF 78.2 vs 79.3 for anime-v3"},
    "cugan-pro": {"for": "anime", "speed": "fast (Video2X, GPU)",
                  "good": "line weight closest to the source; the only anime model that kept realistic drawn faces "
                          "(identity 0.799 vs 0.687)",
                  "weak": "less clean than anime-v3 on rough sources"},
    "general-v3": {"for": "both", "speed": "about 7.8 fps at 720p",
                   "good": "fast; highest VMAF on cartoon ground truth (82.4) and real segments (89.4); fine without "
                           "faces and for long videos",
                   "weak": "sharper but reshapes real faces more (identity 0.861)"},
    "realesrnet": {"for": "real", "speed": "slow (about 433k input pixels/s)",
                   "good": "most faithful on real faces (identity 0.880, VMAF 84.5); no invented texture",
                   "weak": "softer look; slow on long or large videos"},
    "bsrnet": {"for": "real", "speed": "slow (about 433k input pixels/s)",
               "good": "as faithful as RealESRNet on faces (0.879); the most faithful photo model on the person's "
                       "ground truth",
               "weak": "slightly lower VMAF (82.0); slow"},
}


def _suggest(p: probe.VideoProfile, anime: bool, slow_h: float, fast_h: float) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if anime:
        pick = "esrgan-v3"
        reasons.append("cartoon/anime: anime-v3 gave the cleanest line art on every owner sample")
        if p.faces_max:
            reasons.append("faces were detected: if the frames sheet shows REALISTIC drawn faces (not stylised), "
                           "choose cugan-pro instead (it kept them: identity 0.799 vs 0.687)")
    elif p.faces_max and slow_h <= FREE_NONGAN_MAX_HOURS:
        pick = "realesrnet"
        reasons.append(f"real faces and the faithful model fits the time budget (about {slow_h:.1f} h): RealESRNet "
                       "keeps identity best")
        if p.face_min_height and p.face_min_height < 24:
            reasons.append("faces are tiny (< 24 px): no model can restore them reliably; the guard keeps the original "
                           "look there")
    elif slow_h > FREE_NONGAN_MAX_HOURS:
        pick = "general-v3"
        reasons.append(f"the faithful models would need about {slow_h:.0f} h; general-v3 takes about {fast_h:.1f} h")
        if p.faces_max:
            reasons.append("it reshapes faces a little more (identity 0.861); choose realesrnet only if the person "
                           "accepts the long run")
    else:
        pick = "realesrnet"
        reasons.append("real footage without faces, short enough for the faithful model; general-v3 is the sharper, "
                       "faster alternative when the person prefers that look")
    if p.noise_sigma > 2:
        reasons.append(f"noisy source (sigma {p.noise_sigma:.1f}): SCUNet is not used on video; the models clean it, and "
                       "texture return puts the source's texture back")
    return pick, reasons


def model_guide(p: probe.VideoProfile, scale: float | None = None, content: str = "auto") -> dict:
    """What Claude needs to choose the model for THIS video: the signals that matter, each model's measured strengths
    and weaknesses with its estimated run time here, and a suggested pick with the reasons. Claude confirms the
    content from the frames sheet (drawn vs real, realistic drawn faces) and may choose differently."""
    anime = is_anime(p, content)
    frames = (p.frames or round(p.duration * (p.fps or 30))) * (2 if "interlaced" in p.flags else 1)
    slow_h = frames * p.width * p.height / FREE_NONGAN_PIXELS_PER_S / 3600
    fast_h = frames * p.width * p.height / (FREE_NONGAN_PIXELS_PER_S * 18) / 3600     # general-v3 ~18x faster
    hours = {"realesrnet": slow_h, "bsrnet": slow_h, "general-v3": fast_h}
    reading = "anime/cartoon" if anime else "real footage"
    # the probe often reads old cartoons as real footage (three owner cartoons in a row): the frames sheet decides
    signals = {"content": reading if content != "auto" else f"{reading} (probe only; confirm on the frames sheet)",
               "size": f"{p.width}x{p.height} @ {p.fps:g} fps, {p.duration / 60:.1f} min",
               "faces": (f"up to {p.faces_max} per frame, smallest {p.face_min_height} px" if p.faces_max else "none seen"),
               "noise_sigma": p.noise_sigma, "compressed": "compressed" in p.flags,
               "flags": p.flags}
    models = []
    for name, f in MODEL_FACTS.items():
        applies = f["for"] == "both" or (f["for"] == "anime") == anime
        m = {"model": name, "applies": applies, **{k: f[k] for k in ("speed", "good", "weak")}}
        if name in hours:
            m["est_hours"] = round(hours[name], 2)
        models.append(m)
    pick, reasons = _suggest(p, anime, slow_h, fast_h)
    other = {}
    if content == "auto":       # the probe often misreads old cartoons: give both, Claude decides from the sheet
        other = {"suggested_if_anime": _suggest(p, True, slow_h, fast_h)[0],
                 "suggested_if_real": _suggest(p, False, slow_h, fast_h)[0]}
    return {"signals": signals, "models": models, "suggested": pick, "why": reasons, **other,
            "how": "Choose the model yourself from these facts and the frames sheet, tell the person which and why, "
                   "and pass it as restore_video(model=...). It goes first; if the guard rejects it on the previews the "
                   "approved order follows. model_strict=True only when the person insists on one model."}


def _plan_candidates(p: probe.VideoProfile, scale: float | None, content: str,
                     all_models: bool = False) -> tuple[str, list[tuple[str, Callable[[], list]]]]:
    out_h, size_note = target_size(p, scale)
    if is_anime(p, content):
        # anime / cartoon: Video2X anime models (free, Vulkan). Measured on the owner's cartoon samples
        # (PLAN.md 3.7): Real-ESRGAN anime-v3 first, Real-CUGAN as the alternative.
        k = int(round(scale)) if scale else 2
        if not scale and p.height > 720:
            return f"anime/cartoon at {p.height}p: already large for a x2 anime model (ask for --scale 2 to force)", []
        if k not in (2, 3, 4):
            return f"anime/cartoon: Video2X anime models support x2/x3/x4, not x{scale:g}", []
        why = (f"{size_note}; anime/cartoon ({'requested' if content == 'anime' else 'detected'}): "
               f"Video2X Real-ESRGAN anime-v3 x{k} first, Real-CUGAN second")
        h_out = out_h or p.height * k
        cands = [
            ("anime esrgan-v3", lambda: [video2x.AnimeSR("realesrgan", "realesr-animevideov3", k)]),
            ("anime cugan", lambda: [video2x.AnimeSR("realcugan", "models-se", k)]),
            ("free general-v3", lambda: [free_sr.FreeSR(model="general_x4v3", height=h_out)])]
        if all_models:
            # Real-CUGAN pro: closer line weight and the only anime model that passed the face check on the
            # clips of a 1991 cartoon with realistic faces (identity 0.799 vs 0.687 for anime-v3); on request only
            cands.append(("anime cugan-pro", lambda: [video2x.AnimeSR("realcugan", "models-pro", k)]))
        return why + ", general-v3 third", cands
    h = out_h or p.height
    # measured on this GPU (RTX 5060 Ti): the faithful non-GAN models process ~433k input pixels/s
    # (RealESRNet 0.47 fps at 720p), the compact general-v3 ~7.8 fps at 720p. Ground truth on the
    # person's 12 video segments: RealESRNet VMAF 84.5 / identity 0.880, BSRNet 82.0 / 0.879,
    # general-v3 89.4 / 0.861 (sharper, reshapes faces more).
    frames = (p.frames or round(p.duration * (p.fps or 30))) * (2 if "interlaced" in p.flags else 1)
    est_h = frames * p.width * p.height / FREE_NONGAN_PIXELS_PER_S / 3600
    general = ("free general-v3", lambda: [free_sr.FreeSR(model="general_x4v3", height=h)])
    if est_h <= FREE_NONGAN_MAX_HOURS:
        cands = [("free realesrnet", lambda: [free_sr.FreeSR(model="realesrnet_x4", height=h)]),
                 ("free bsrnet", lambda: [free_sr.FreeSR(model="bsrnet_x4", height=h)]), general]
        speed = f"faithful non-GAN models first (est. {est_h:.1f} h)"
    else:
        cands = [general]
        speed = f"general-v3 only: the faithful models would need about {est_h:.0f} h"
    why = (f"{size_note}; free engine (open models), {speed}"
           + ("; IVTC first (classical)" if "telecine" in p.flags else "")
           + ("; deinterlace first (classical, double rate)" if "interlaced" in p.flags else "")
           + (f"; {p.faces_max} faces in samples" if p.faces_max else ""))
    return why, cands



# ----------------------------------------------------------------------------- long videos: chunks
def chunk_plan(duration: float, chunk: float | None = None, cuts: list[float] | None = None) -> list[tuple[float, float]]:
    """(start, length) per chunk; the last chunk absorbs a short remainder instead of being tiny.
    With scene cuts, each boundary moves to the nearest cut within +-25% of the chunk length. The open models
    work frame by frame with fixed weights, so joins are exact; a cut still hides any residual change from
    the context frames of temporal steps."""
    chunk = chunk or CHUNK_SECONDS
    n = max(1, int(duration // chunk))
    if duration - n * chunk > chunk * 0.25:
        n += 1
    edges = [min(duration, i * chunk) for i in range(n)] + [duration]
    if cuts:
        window = chunk * 0.25
        for i in range(1, n):
            near = [c for c in cuts if abs(c - edges[i]) <= window and edges[i - 1] + chunk * 0.5 < c < duration - chunk * 0.25]
            if near:
                edges[i] = min(near, key=lambda c: abs(c - edges[i]))
    return [(edges[i], edges[i + 1] - edges[i]) for i in range(n)]


def fingerprint(*parts) -> str:
    """What made a checkpoint, as a short hash: a checkpoint is reused only when this still matches."""
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def source_id(src: Path) -> dict:
    st = Path(src).stat()
    return {"path": str(Path(src).resolve()), "size": st.st_size, "mtime": int(st.st_mtime)}


def wipe_checkpoints(work: Path, keep: tuple[str, ...] = ()) -> None:
    """Remove everything a previous run left in the work folder except `keep` (names directly in it)."""
    for f in Path(work).iterdir() if Path(work).is_dir() else []:
        if f.name in keep:
            continue
        shutil.rmtree(f, ignore_errors=True) if f.is_dir() else f.unlink(missing_ok=True)


def work_key(out_dir: Path, src: Path) -> str:
    """Name of the video's work folder and deliverable: its stem, unless another file with the same stem (ep01.avi
    next to ep01.mkv) already owns that work folder; then the stem plus a short hash of the path."""
    try:
        owner = json.loads((Path(out_dir) / "work" / src.stem / "profile.json").read_text(encoding="utf-8"))["source"]["path"]
    except (OSError, ValueError, KeyError, TypeError):
        return src.stem
    mine = str(Path(src).resolve())
    return src.stem if owner.lower() == mine.lower() else f"{src.stem}-{hashlib.sha1(mine.lower().encode()).hexdigest()[:8]}"


def scene_cuts(video: Path, work: Path, log=print) -> list[float]:
    """Scene-cut times over the whole video (once; cached in the work folder for resume)."""
    cache = work / "cuts.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    log("   scanning scene cuts for chunk boundaries")
    cp = subprocess.run([_ffmpeg(), "-hide_banner", "-nostats", "-i", str(video), "-an", "-vf",
                         "scale=480:-2,scdet=threshold=10", "-f", "null", "-"], capture_output=True, text=True, encoding="utf-8",
                        stdin=subprocess.DEVNULL, errors="replace")
    import re as _re
    cuts = sorted({round(float(t), 3) for t in _re.findall(r"lavfi\.scd\.time:\s*([0-9.]+)", cp.stderr)})
    work.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(cuts), encoding="utf-8")
    return cuts


def run_chunked(base: Path, dst: Path, stages_fn: Callable[[], list], encoder: str, work: Path, *,
                on_progress: Callable[[int, int], None] | None = None, log=print) -> AdapterResult:
    """Render `base` in chunks with context frames on both sides (trimmed exactly), skip chunks already
    done (resume), join them losslessly and mux audio + metadata from `base` in one step."""
    # exact: frames counted and the rate as a fraction. The container's duration can be longer than the video
    # (audio or subtitles that run on), which planned a last chunk past the end: it failed after hours
    info = media.probe(base, exact=True)
    rate = avs._rate(base)[0]
    fps = float(rate) if rate else info["fps"]
    n_frames = info["frames"]
    dur = n_frames / fps if n_frames and fps else info["duration"]
    # where the video's first frame sits on the file's clock (it may start after the audio): seeks are on that clock
    v0 = max(0.0, (info.get("video_start") or 0.0) - min(info.get("video_start") or 0.0, info.get("audio_start") or 0.0))
    cdir = work / "chunks"
    cdir.mkdir(parents=True, exist_ok=True)
    # chunks are reused only when the same input, model, scale and encoder made them: a rerun with another model
    # used to join old and new chunks into one file. Chunks from before this check carry no fingerprint and are
    # trusted as they are
    made_by = fingerprint(base.name, base.stat().st_size, [getattr(s, "describe", lambda s=s: repr(s))() for s in stages_fn()],
                          encoder, CHUNK_SECONDS, CHUNK_CONTEXT_SECONDS)
    fp_file = cdir / "fingerprint.json"
    stored = json.loads(fp_file.read_text(encoding="utf-8")) if fp_file.exists() else None
    if stored is not None and stored != made_by:
        log("   chunks from a run with other settings or another input: starting them again")
        shutil.rmtree(cdir, ignore_errors=True)
        cdir.mkdir(parents=True, exist_ok=True)
        (work / "cuts.json").unlink(missing_ok=True)
    fp_file.write_text(json.dumps(made_by), encoding="utf-8")
    cuts = scene_cuts(base, work, log)
    plan_file = cdir / "plan.json"
    if plan_file.exists():                       # resume: never re-plan under finished chunks
        plan = [tuple(x) for x in json.loads(plan_file.read_text(encoding="utf-8"))]
    else:
        plan = chunk_plan(dur, cuts=cuts)
        plan_file.write_text(json.dumps(plan), encoding="utf-8")
    on_cut = sum(1 for st, _ in plan[1:] if any(abs(st - c) < 1e-3 for c in cuts))
    log(f"   {len(plan)} chunks; {on_cut} of {max(0, len(plan) - 1)} boundaries on scene cuts")
    total = n_frames or round(dur * fps)

    def frame_at(t: float) -> int:
        return max(0, min(total, round(t * fps)))

    done_frames = 0
    parts = []
    for i, (start, length) in enumerate(plan):
        part, marker = cdir / f"c{i:04d}.mp4", cdir / f"c{i:04d}.done"
        # everything in frame numbers. first..last are this chunk's frames, lead is where decoding starts (context)
        first = frame_at(start)
        last = total if i == len(plan) - 1 else frame_at(start + length)
        frames_want = last - first
        parts.append(part)
        if marker.exists() and part.exists():
            done_frames += frames_want
            continue
        if frames_want <= 0:
            parts.pop()                                    # a plan made for a longer duration: nothing left here
            continue
        pre = min(CHUNK_CONTEXT_SECONDS, start)
        post = max(0.0, min(CHUNK_CONTEXT_SECONDS, dur - (start + length)))
        lead = frame_at(start - pre)
        skip = first - lead
        # seek to half a frame BEFORE frame `lead`: the decoder starts at the first frame at or after the seek time,
        # so a seek exactly on a frame time landed on that frame or the next one depending on rounding
        # (29.97 fps: 119 s is frame 3566.43, decoding began at 3567 while the trim assumed 3566; one frame was
        # lost at the join and one repeated)
        seek = v0 + (lead - 0.5) / fps if lead > 0 else 0.0
        span = (last - lead + 1) / fps + post
        trim = f"trim=start_frame={skip}:end_frame={skip + frames_want},setpts=PTS-STARTPTS"

        def prog(n: int, _t: int, base_n=done_frames) -> None:
            if on_progress:
                on_progress(base_n + max(0, n - skip), total)

        r = run_engine(base, part, stages_fn(), encoder, start=seek, duration=span,
                         audio=False, post_vf=trim, expected_frames=frames_want, on_progress=prog)
        if not r.ok:
            return AdapterResult(False, error=f"chunk {i + 1}/{len(plan)}: {r.error}", run=r.run)
        marker.write_text(json.dumps({"start": start, "length": length, "frames": frames_want}), encoding="utf-8")
        done_frames += frames_want
        log(f"   chunk {i + 1}/{len(plan)} done ({start:.0f}-{start + length:.0f}s)")
    listing = cdir / "concat.txt"
    listing.write_text("".join(media.concat_line(pp) for pp in parts), encoding="utf-8")
    args = [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "concat", "-safe", "0",
            "-i", str(listing), "-i", str(base), "-map", "0:v:0", "-map", "1:a?", "-c", "copy",
            "-map_metadata", "1", *media.colour_args(media.colour_hint(base))[1]]
    if dst.suffix.lower() in (".mp4", ".mov", ".m4v"):
        args += ["-movflags", "+write_colr+faststart"]
    cp = subprocess.run([*args, str(dst)], capture_output=True, text=True, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL)
    if cp.returncode != 0 or not dst.exists():
        return AdapterResult(False, error="join failed: " + cp.stderr.strip()[-300:])
    sync = media.fix_av_offset(base, dst)
    aspect = media.fix_aspect(base, dst)
    out = media.probe(dst)
    details = {"chunks": len(plan), "boundaries_on_cuts": on_cut, "expected_frames": total, "output": out,
               "av_sync": sync, "aspect": aspect}
    if abs(out["frames"] - total) > max(2, total * 0.002):
        return AdapterResult(False, [dst], error=f"joined frame count {out['frames']} != {total}", details=details)
    if info["audio_streams"] and not out["audio_streams"]:
        return AdapterResult(False, [dst], error="audio stream was lost in the join", details=details)
    return AdapterResult(True, [dst], details=details)


# ----------------------------------------------------------------------------- classical pre-pass
def chroma_stage(p: probe.VideoProfile) -> avs.ChromaShift | None:
    """The chroma-shift correction the profile asks for (None when the colour is aligned). Interlaced and
    telecined sources get the horizontal part only (measured on woven frames, a vertical shift mixes fields)."""
    if "chroma_shift" not in p.flags or not p.chroma:
        return None
    st = chroma_probe.ChromaShift(u=tuple(p.chroma["u"]), v=tuple(p.chroma["v"]), spread=p.chroma.get("spread", 0.0),
                                  spreads=p.chroma.get("spreads") or {}, frames=p.chroma.get("frames", 0))
    c = st.correction(horizontal_only=bool({"interlaced", "telecine"} & set(p.flags)))
    if not any(c["u"] + c["v"]):
        return None
    return avs.ChromaShift(u_dx=c["u"][0], u_dy=c["u"][1], v_dx=c["v"][0], v_dy=c["v"][1])


def _describe_chroma(cs: avs.ChromaShift) -> str:
    def d(dx: float, dy: float) -> str:
        parts = ([f"{abs(dx):g} px {'right' if dx > 0 else 'left'}"] if dx else []) + \
                ([f"{abs(dy):g} px {'down' if dy > 0 else 'up'}"] if dy else [])
        return ", ".join(parts) or "unchanged"
    return f"chroma shift: colour moved back onto the lines (Cb {d(cs.u_dx, cs.u_dy)} / Cr {d(cs.v_dx, cs.v_dy)})"


MASK_MARGIN = 2                # rows added above the measured head-switching band


def mask_stage(p: probe.VideoProfile) -> avs.MaskBottom | None:
    """The bottom mask for measured head-switching noise (asked for with mask_tape_noise): the band plus a small
    margin, rounded up to even rows (4:2:0), to a multiple of 4 for interlaced sources."""
    if "head_switching" not in p.flags:
        return None
    rows = int(p.tape.get("head_switching_rows") or 0) + int(p.tape.get("bottom_border_rows") or 0) + MASK_MARGIN
    step = 4 if {"interlaced", "telecine"} & set(p.flags) else 2
    return avs.MaskBottom(rows=-(-rows // step) * step)


def tape_notes(p: probe.VideoProfile, masked: bool = False) -> list[str]:
    """Plain-language notes on tape defects (videohelp thread 380626): what was found and what can be done."""
    t, notes = p.tape or {}, []
    if "head_switching" in p.flags:
        rows = t.get("head_switching_rows")
        notes.append(f"VHS head-switching noise in the bottom {rows} lines ({t.get('head_switching_share', 0):.0%} "
                     "of sampled scenes): " + ("masked black, as asked" if masked else
                     "left as it is (ask for mask_tape_noise / --mask-tape-noise to black it out)"))
    if "top_flagging" in p.flags:
        notes.append(f"the top of the picture bends sideways (tape \"flagging\", about {t.get('flagging_px', 0):g} px "
                     f"in {t.get('flagging_share', 0):.0%} of sampled frames): a timing error that software cannot fix; "
                     "capture the tape again through a VCR or DVD recorder with a line time-base corrector (TBC)")
    if "clipped_highlights" in p.flags:
        notes.append(f"bright areas are clipped at the top of the range ({t.get('clipped_share', 0):.0%} of sampled "
                     "frames): the detail there is gone and cannot be restored; for a tape, capture again with "
                     "lower brightness/contrast (and losslessly, e.g. HuffYUV or Lagarith with PCM audio)")
    return notes


def plan_prepass(p: probe.VideoProfile, mask_tape_noise: bool = False) -> tuple[list, str]:
    """AviSynth+ / VapourSynth stages that must run before any model (empty when none is needed).
    mask_tape_noise: also black out measured head-switching noise (only when the person asks)."""
    tff = p.field_order != "bff"
    stages: list = []
    why: list[str] = []
    if "blended" in p.flags:
        share = f"{p.blend_share:.0%} of moving {'fields' if 'interlaced' in p.flags else 'frames'} are blends"
        if "interlaced" in p.flags:
            stages.append(avs.Deinterlace(tff=tff, tr2=2 if "noisy" in p.flags else 1))
            why.append(f"blended fields ({share}): QTGMC double rate, then sRestore back to the original rate")
        else:
            why.append(f"blended frames ({share}): sRestore back to the original rate")
    elif "telecine" in p.flags:
        stages.append(avs.IVTC(tff=tff))
        why.append(f"telecine ({'TFF' if tff else 'BFF'}): IVTC with TFM+TDecimate")
    elif "interlaced" in p.flags:
        stages.append(avs.Deinterlace(tff=tff, tr2=2 if "noisy" in p.flags else 1))
        why.append(f"interlaced ({'TFF' if tff else 'BFF'}): QTGMC SourceMatch=3 Lossless=2, double rate")
    cs = chroma_stage(p)
    if cs:
        stages.append(cs)
        why.append(_describe_chroma(cs))
    ms = mask_stage(p) if mask_tape_noise else None
    if ms:
        stages.append(ms)
        why.append(f"head-switching noise: bottom {ms.rows} rows masked black (asked for)")
    if "blended" in p.flags:
        stages.append(avs.SRestore())
    return stages, "; ".join(why)


def _has_field_stage(stages: list) -> bool:
    return any(isinstance(s, (avs.Deinterlace, avs.IVTC)) for s in stages)


def _fallback_chroma_vf(stages: list) -> str | None:
    """ffmpeg's chromashift (whole chroma pixels only) for the no-engine fallback."""
    cs = next((s for s in stages if isinstance(s, avs.ChromaShift)), None)
    if cs is None:
        return None
    v = {k: round(x / 2) for k, x in (("cbh", cs.u_dx), ("cbv", cs.u_dy), ("crh", cs.v_dx), ("crv", cs.v_dy))}
    return "chromashift=" + ":".join(f"{k}={x}" for k, x in v.items()) if any(v.values()) else None


def _fallback_vf(p: probe.VideoProfile) -> tuple[str, float]:
    """ffmpeg field handling (bwdif / fieldmatch+decimate), used when no classical engine is available;
    also the conventional reference the pre-pass is checked against. Returns (filter, rate factor)."""
    order = "bff" if p.field_order == "bff" else "tff"
    if "telecine" in p.flags:
        return f"fieldmatch=order={order},bwdif=mode=send_frame:deint=interlaced,decimate", 0.8
    return f"bwdif=mode=send_field:parity={order}:deint=all", 2.0


# ----------------------------------------------------------------------------- un-blending (sRestore)
DEBLEND_RATES = (Fraction(24000, 1001), Fraction(25), Fraction(30000, 1001))
DEBLEND_TIE = 0.03            # trial blend shares this close count as equal
DEBLEND_HELD_TIE = 0.02       # a higher rate is kept only if it holds pictures at most this much more often:
                              # one rate step above the original adds (r - r0) / r held pictures, 4.1 % for
                              # 25 vs 23.976 (measured +3.0 points), 16.6 % for 29.97 vs 25 (measured +10.1)
DEBLEND_TRIAL_SECONDS = 10.0
DEBLEND_KEEP = 0.85           # the trial must cut the blend share at least this much (the guard then checks
                              # the whole video: b001's trial spans happened to hold few blends, 12 % -> 9.5 %,
                              # while the whole clip went 22 % -> 8 %)


def deblend_rates(in_fps: Fraction) -> list[Fraction]:
    """Original rates worth trying: below the input rate, and not below a fifth of it (sRestore's range)."""
    return [r for r in DEBLEND_RATES if r < in_fps * Fraction(98, 100) and r * 5 >= in_fps]


def pick_rate(shares: dict, held: dict | None = None) -> Fraction | None:
    """Lowest blend share. Among rates within DEBLEND_TIE of it: the highest one that does not repeat pictures
    more often than the others (a rate above the original holds pictures; one below drops real pictures,
    which no blend measure sees, so of the rates that hold nothing extra the highest is the safe choice)."""
    if not shares:
        return None
    best = min(shares.values())
    tied = [r for r, v in shares.items() if v <= best + DEBLEND_TIE]
    held = held or {}
    floor = min((held.get(r, 0.0) for r in tied), default=0.0)
    return max(r for r in tied if held.get(r, 0.0) <= floor + DEBLEND_HELD_TIE)


def _trial_spans(duration: float, fps: Fraction) -> list[tuple[int, int]]:
    count = round(min(DEBLEND_TRIAL_SECONDS, duration / 2) * fps)
    starts = [0.0] if duration <= DEBLEND_TRIAL_SECONDS * 2.5 else [duration * 0.3, duration * 0.7]
    return [(max(0, round(t * fps)), count) for t in starts]


def choose_deblend_rate(base: Path, work: Path, log=print) -> dict:
    """Run sRestore at each candidate rate on two short spans and measure what is left blended."""
    in_fps = avs._rate(base)[0]
    info = media.probe(base, exact=True)
    rates = deblend_rates(in_fps)
    out: dict = {"in_fps": str(in_fps), "rates": [str(r) for r in rates], "trials": {}}
    if not rates:
        out["error"] = f"no original rate below {float(in_fps):.3f} fps to restore"
        return out
    spans = _trial_spans(info["duration"], in_fps)
    tdir = work / "deblend_trials"
    tdir.mkdir(parents=True, exist_ok=True)
    before = [list(blend_probe.read_gray(base, float(f / in_fps), float(n / in_fps))) for f, n in spans]
    out["trial_input_share"] = blend_probe.summarize(before).share
    shares: dict = {}
    held: dict = {}
    for r in rates:
        wins = []
        for k, (first, count) in enumerate(spans):
            dst = tdir / f"r{float(r):.3f}_{k}.mkv"
            res = vsb.render_span(base, dst, [avs.SRestore(frate=str(r), in_fps=str(in_fps))], first, count,
                                  work=tdir)
            if not res.ok:
                out["trials"][str(r)] = {"error": res.error}
                break
            wins.append(list(blend_probe.read_gray(dst, 0, float(count / in_fps) + 1)))
        else:
            shares[r] = blend_probe.summarize(wins).share
            held[r] = round(sum(blend_probe.held_share(w) * len(w) for w in wins) / max(1, sum(map(len, wins))), 3)
            out["trials"][str(r)] = {"share": shares[r], "held": held[r]}
            log(f"   sRestore trial at {float(r):.3f} fps: {shares[r]:.1%} blended, {held[r]:.1%} held "
                f"(input {out['trial_input_share']:.1%} blended)")
    best = pick_rate(shares, held)
    out["chosen"] = str(best) if best else None
    return out


def run_deblend(base: Path, work: Path, log=print) -> tuple[Path | None, dict]:
    """sRestore on a progressive input (the source, or the double-rate pre-pass). None when it was not done or
    did not pass its guard: the caller keeps the blended input (still restorable, just not un-blended)."""
    ddir = work / "deblend"                 # its own folder: an output folder may not hold the pass's input
    dst, done = ddir / "deblend.nut", ddir / "deblend.done"
    made_by = fingerprint(base.name, base.stat().st_size)
    if dst.exists() and done.exists():
        info = json.loads(done.read_text(encoding="utf-8"))
        if info.get("fingerprint") in (None, made_by):          # None: made before fingerprints existed
            log(f"   sRestore: reusing the finished one ({info.get('fps_out')} fps)")
            return dst, info | {"reused": True}
        log("   sRestore: the finished one was made from another input: doing it again")
        for d in (ddir, work / "deblend_trials", work / "chunks"):
            shutil.rmtree(d, ignore_errors=True)
    ok, why = vsb.available()
    if not ok:
        return None, {"skipped": f"VapourSynth not available ({why}); blended frames stay"}
    trial = choose_deblend_rate(base, work, log)
    info: dict = {"trial": trial}
    rate = Fraction(trial["chosen"]) if trial.get("chosen") else None
    in_share = trial.get("trial_input_share") or 0.0
    if rate is None:
        info["skipped"] = trial.get("error") or "no sRestore trial finished"
        return None, info
    left = trial["trials"][str(rate)]["share"]
    if left > max(0.05, DEBLEND_KEEP * in_share):
        info["skipped"] = f"sRestore left {left:.1%} blended (input {in_share:.1%}): not worth applying"
        return None, info
    in_fps = Fraction(trial["in_fps"])
    stage = avs.SRestore(frate=str(rate), in_fps=str(in_fps))
    ddir.mkdir(parents=True, exist_ok=True)
    r = vsb.render(base, dst, [stage], work=ddir)
    info.update(engine="vapoursynth", fps_in=str(in_fps), fps_out=str(rate), ok=r.ok, error=r.error,
                frames=r.details.get("frames"))
    if not r.ok:
        info["skipped"] = f"sRestore failed: {r.error}"
        return None, info
    colour = media.colour_hint(base) or media.colour_hint(dst)
    info["colour"] = media.write_colour_sidecar(dst, colour) if colour else {}
    info["av_sync"] = media.fix_av_offset(base, dst)
    info["aspect"] = media.fix_aspect(base, dst)
    in_full = blend_probe.analyze(base).share
    rep_ = verify_deblend(base, dst, in_fps=in_fps, out_fps=rate, in_share=in_full)
    info["guard"] = {g.name: {"status": g.status, "value": g.value, "detail": g.detail} for g in rep_.gates}
    info["guard_passed"] = rep_.passed
    info["blend_share_before"] = in_full
    info["blend_share_after"] = info["guard"].get("blends", {}).get("value")
    log(f"   sRestore {float(in_fps):.3f} -> {float(rate):.3f} fps: guard "
        f"{'PASS' if rep_.passed else 'FAIL ' + ','.join(rep_.failed())}")
    if not rep_.passed:
        info["skipped"] = f"sRestore output failed its guard ({', '.join(rep_.failed())}): blended frames stay"
        return None, info
    info["fingerprint"] = made_by
    done.write_text(json.dumps(info, default=str), encoding="utf-8")
    return dst, info


# ----------------------------------------------------------------------------- composed clean-up (cleanup.py)
def run_spans(base: Path, work: Path, p: probe.VideoProfile, log=print) -> tuple[Path | None, dict]:
    """Deinterlace only the measured interlaced stretches (probe/combing.py). None when it was not done or did
    not pass its guard: the caller keeps the input."""
    sdir = work / "spans"
    dst, done = sdir / "spans.nut", sdir / "spans.done"
    c = p.combing or {}
    spans = c.get("spans") or []
    if not spans:
        return None, {"skipped": "no interlaced stretch was measured"}
    frames = media.probe(base, exact=True)["frames"]
    if frames != c.get("frames"):
        return None, {"skipped": f"an earlier step changed the frame numbering ({c.get('frames')} -> {frames} frames)"}
    made_by = fingerprint(base.name, base.stat().st_size, spans)
    if dst.exists() and done.exists():
        info = json.loads(done.read_text(encoding="utf-8"))
        if info.get("fingerprint") == made_by:
            log("   interlaced stretches: reusing the finished pass")
            return dst, info | {"reused": True}
        shutil.rmtree(sdir, ignore_errors=True)
        shutil.rmtree(work / "chunks", ignore_errors=True)
    ok, why = avs.available()
    if not ok:
        return None, {"skipped": f"AviSynth+ not available ({why}); the interlaced stretches stay"}
    sdir.mkdir(parents=True, exist_ok=True)
    r = avs.render(base, dst, [avs.DeinterlaceSpans(spans=spans, frames=frames, tff=bool(c.get("tff", True)))], work=sdir)
    info: dict = {"engine": "avisynth", "spans": spans, "span_frames": c.get("span_frames"), "ok": r.ok,
                  "error": r.error, "frames": r.details.get("frames")}
    if not r.ok:
        info["skipped"] = f"the pass failed: {r.error}"
        return None, info
    colour = media.colour_hint(base) or media.colour_hint(dst)
    info["colour"] = media.write_colour_sidecar(dst, colour) if colour else {}
    info["av_sync"] = media.fix_av_offset(base, dst)
    info["aspect"] = media.fix_aspect(base, dst)
    rep_ = verify_spans(base, dst, spans=spans)
    info["guard"] = {g.name: {"status": g.status, "value": g.value, "detail": g.detail} for g in rep_.gates}
    info["guard_passed"] = rep_.passed
    log(f"   interlaced stretches deinterlaced ({len(spans)} spans, {c.get('span_frames')} frames): guard "
        f"{'PASS' if rep_.passed else 'FAIL ' + ','.join(rep_.failed())}")
    if not rep_.passed:
        info["skipped"] = f"the pass failed its guard ({', '.join(rep_.failed())}): the stretches stay as they were"
        return None, info
    info["fingerprint"] = made_by
    done.write_text(json.dumps(info, default=str), encoding="utf-8")
    return dst, info


def run_cleanup(base: Path, work: Path, filters: list[dict], log=print, drawn: bool = False) -> tuple[Path | None, dict]:
    """The composed picture filters (deblock, dehalo) as one pass. None when it was not done or did not pass its
    guard: the caller keeps the input."""
    cdir = work / "cleanup"
    dst, done = cdir / "cleanup.nut", cdir / "cleanup.done"
    stages = cleanup.picture_stages(filters)
    if not stages:
        return None, {"skipped": "no picture filter in the recipe"}
    said = cleanup.describe([f for f in filters if f["name"] != "deinterlace_scenes"])
    made_by = fingerprint(base.name, base.stat().st_size, [repr(s) for s in stages])
    if dst.exists() and done.exists():
        info = json.loads(done.read_text(encoding="utf-8"))
        if info.get("fingerprint") == made_by:
            log(f"   clean-up: reusing the finished pass ({said})")
            return dst, info | {"reused": True}
        shutil.rmtree(cdir, ignore_errors=True)
        shutil.rmtree(work / "chunks", ignore_errors=True)
    ok, why = avs.available()
    if not ok:
        return None, {"skipped": f"AviSynth+ not available ({why}); no clean-up"}
    cdir.mkdir(parents=True, exist_ok=True)
    r = avs.render(base, dst, stages, work=cdir)
    info: dict = {"engine": "avisynth", "filters": said, "ok": r.ok, "error": r.error,
                  "script": r.details.get("script"), "frames": r.details.get("frames")}
    if not r.ok:
        info["skipped"] = f"the pass failed: {r.error}"
        return None, info
    colour = media.colour_hint(base) or media.colour_hint(dst)
    info["colour"] = media.write_colour_sidecar(dst, colour) if colour else {}
    info["av_sync"] = media.fix_av_offset(base, dst)
    info["aspect"] = media.fix_aspect(base, dst)
    rep_ = verify_cleanup(base, dst, deblocked=any(isinstance(s, avs.Deblock) for s in stages), drawn_faces=drawn)
    info["guard"] = {g.name: {"status": g.status, "value": g.value} for g in rep_.gates}
    info["guard_passed"] = rep_.passed
    log(f"   clean-up ({said}): guard {'PASS' if rep_.passed else 'FAIL ' + ','.join(rep_.failed())}")
    if not rep_.passed:
        info["skipped"] = f"the pass failed its guard ({', '.join(rep_.failed())}): no clean-up"
        return None, info
    info["fingerprint"] = made_by
    done.write_text(json.dumps({k: v for k, v in info.items() if k != "script"}, default=str), encoding="utf-8")
    return dst, info


# ----------------------------------------------------------------------------- padded repeats, black borders
def run_derepeat(base: Path, work: Path, p: probe.VideoProfile, log=print) -> tuple[Path | None, dict]:
    """Drop the copied frames of a padded original (probe/framing.py) back to its own rate. None when it was not
    done or did not pass its guard: the caller keeps the padded input."""
    ddir = work / "derepeat"                # its own folder: an output folder may not hold the pass's input
    dst, done = ddir / "derepeat.nut", ddir / "derepeat.done"
    rate = (p.repeats or {}).get("original_fps")
    if not rate:
        return None, {"skipped": "no padding cycle was measured"}
    made_by = fingerprint(base.name, base.stat().st_size, rate)
    if dst.exists() and done.exists():
        info = json.loads(done.read_text(encoding="utf-8"))
        if info.get("fingerprint") == made_by:
            log(f"   repeats: reusing the finished pass ({info.get('fps_out')} fps)")
            return dst, info | {"reused": True}
        shutil.rmtree(ddir, ignore_errors=True)
        shutil.rmtree(work / "chunks", ignore_errors=True)
    ok, why = avs.available()
    if not ok:
        return None, {"skipped": f"AviSynth+ not available ({why}); the copied frames stay"}
    in_fps = avs._rate(base)[0]
    if not in_fps or Fraction(rate) >= in_fps:
        return None, {"skipped": f"the input runs at {float(in_fps or 0):.3f} fps: nothing to drop for {rate}"}
    ddir.mkdir(parents=True, exist_ok=True)
    r = avs.render(base, dst, [avs.DropRepeats(frate=rate, in_fps=str(in_fps))], work=ddir)
    info: dict = {"engine": "avisynth", "fps_in": str(in_fps), "fps_out": rate, "ok": r.ok, "error": r.error,
                  "frames": r.details.get("frames"), "cycle": p.repeats.get("cycle"),
                  "share_before": p.repeats.get("share")}
    if not r.ok:
        info["skipped"] = f"the pass failed: {r.error}"
        return None, info
    colour = media.colour_hint(base) or media.colour_hint(dst)
    info["colour"] = media.write_colour_sidecar(dst, colour) if colour else {}
    info["av_sync"] = media.fix_av_offset(base, dst)
    info["aspect"] = media.fix_aspect(base, dst)
    rep_ = verify_derepeat(base, dst, in_fps=in_fps, out_fps=Fraction(rate))
    info["guard"] = {g.name: {"status": g.status, "value": g.value, "detail": g.detail} for g in rep_.gates}
    info["guard_passed"] = rep_.passed
    info["share_after"] = info["guard"].get("repeats", {}).get("value")
    log(f"   repeats removed {float(in_fps):.3f} -> {float(Fraction(rate)):.3f} fps: guard "
        f"{'PASS' if rep_.passed else 'FAIL ' + ','.join(rep_.failed())}")
    if not rep_.passed:
        info["skipped"] = f"the pass failed its guard ({', '.join(rep_.failed())}): the copied frames stay"
        return None, info
    info["fingerprint"] = made_by
    done.write_text(json.dumps(info, default=str), encoding="utf-8")
    return dst, info


def run_crop(base: Path, work: Path, p: probe.VideoProfile, log=print) -> tuple[Path | None, dict]:
    """Cut the measured black bars away (lossless, even edges, pixel shape kept). None when it was not done or
    did not pass its guard: the caller keeps the bars."""
    cdir = work / "crop"
    dst, done = cdir / "crop.nut", cdir / "crop.done"
    bi = media.probe(base)
    crop = framing_probe.even_crop(bi["width"], bi["height"], p.borders or {})
    if crop is None or (bi["width"], bi["height"]) != (p.width, p.height):
        return None, {"skipped": "no bar wide enough to cut on even pixels"}
    made_by = fingerprint(base.name, base.stat().st_size, list(crop))
    if dst.exists() and done.exists():
        info = json.loads(done.read_text(encoding="utf-8"))
        if info.get("fingerprint") == made_by:
            log("   borders: reusing the finished crop")
            return dst, info | {"reused": True}
        shutil.rmtree(cdir, ignore_errors=True)
        shutil.rmtree(work / "chunks", ignore_errors=True)
    colour = media.colour_hint(base) or avs._rate(base)[2]
    if avs.is_full_range(colour):
        return None, {"skipped": "full-range source: the lossless crop would have to change its levels"}
    cdir.mkdir(parents=True, exist_ok=True)
    w, h, x, y = crop
    sar = (bi.get("sar") or "1:1").replace(":", "/")
    cp = subprocess.run([_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(base),
                         "-map", "0:v:0", "-map", "0:a?", "-c:a", "copy", "-map_metadata", "0",
                         "-vf", f"crop={w}:{h}:{x}:{y},setsar={sar if not sar.startswith('0') else '1'}",
                         "-fps_mode", "passthrough", *avs.INTERMEDIATE, str(dst)],
                        capture_output=True, text=True, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL)
    info: dict = {"crop": {"width": w, "height": h, "x": x, "y": y}, "borders": {k: p.borders.get(k) for k in
                  ("left", "right", "top", "bottom")}, "ok": cp.returncode == 0 and dst.exists(),
                  "error": cp.stderr.strip()[-300:] or None}
    if not info["ok"]:
        info["skipped"] = f"the crop failed: {info['error']}"
        return None, info
    info["colour"] = media.write_colour_sidecar(dst, colour) if colour else {}
    info["av_sync"] = media.fix_av_offset(base, dst)
    rep_ = verify_crop(base, dst, crop=crop)
    info["guard"] = {g.name: {"status": g.status, "value": g.value, "detail": g.detail} for g in rep_.gates}
    info["guard_passed"] = rep_.passed
    log(f"   borders cropped {bi['width']}x{bi['height']} -> {w}x{h}: guard "
        f"{'PASS' if rep_.passed else 'FAIL ' + ','.join(rep_.failed())}")
    if not rep_.passed:
        info["skipped"] = f"the crop failed its guard ({', '.join(rep_.failed())}): the bars stay"
        return None, info
    info["fingerprint"] = made_by
    done.write_text(json.dumps(info, default=str), encoding="utf-8")
    return dst, info


def run_prepass(src: Path, work: Path, p: probe.VideoProfile, log=print,
                mask_tape_noise: bool = False) -> tuple[Path | None, dict]:
    """Classical pre-pass: field handling (when planned), then un-blending (when planned). Returns the new base,
    or None when field handling failed (the original then stays the deliverable)."""
    stages, why = plan_prepass(p, mask_tape_noise)
    field_stages = [s for s in stages if not isinstance(s, avs.SRestore)]
    # of the PLAN, not of what ran: a step dropped after its check failed must not look like a new plan on resume
    made_by = fingerprint(source_id(src), [repr(s) for s in field_stages])
    if field_stages:
        base, info = _run_field_prepass(src, work, p, field_stages, why, log, made_by)
        notes = {}
        for kind, key in ((avs.ChromaShift, "chroma_check"), (avs.MaskBottom, "mask_check")):
            if base is not None and any(isinstance(s, kind) for s in field_stages) \
                    and not info.get(key, {}).get("passed", False):
                # a correction did not come out as planned: the pass again without it (or the source as it is)
                field_stages = [s for s in field_stages if not isinstance(s, kind)]
                log(f"   {key.replace('_', ' ')} failed: continuing without that step")
                for f in ("prepass.nut", "prepass.done"):
                    (work / f).unlink(missing_ok=True)
                notes[key] = info.get(key)
                base, info = (_run_field_prepass(src, work, p, field_stages, why, log, made_by) if field_stages
                              else (src, {"plan": why}))
        info.update(notes)
    else:
        base, info = src, {"plan": why}
    if base is None:
        return None, info
    if any(isinstance(s, avs.SRestore) for s in stages):
        un, info["deblend"] = run_deblend(base, work, log)
        if un is not None:
            base = un
        else:
            log(f"   sRestore not applied: {info['deblend'].get('skipped')}")
    return base, info


def _run_field_prepass(src: Path, work: Path, p: probe.VideoProfile, stages: list, why: str,
                       log=print, made_by: str | None = None) -> tuple[Path | None, dict]:
    info: dict = {"plan": why}
    if not stages:
        return None, info
    dst = work / "prepass.nut"
    done = work / "prepass.done"
    if dst.exists() and done.exists() and media.colour_sidecar(dst).exists():
        stored = json.loads(done.read_text(encoding="utf-8"))
        if made_by is None or stored.get("fingerprint") in (None, made_by):     # None: made before fingerprints
            info.update(stored, reused=True)
            log(f"   pre-pass: reusing the finished one ({info.get('engine')})")
            return dst, info
        # another plan (a mask asked for now, a different source under this name): everything built on the old
        # pre-pass goes with it
        log("   pre-pass: the finished one was made with another plan: doing it again")
        wipe_checkpoints(work, keep=("profile.json",))
        (work / "previews").mkdir(parents=True, exist_ok=True)
    engines ={"avisynth": avs, "vapoursynth": vsb}
    tried: list[str] = []
    for name in config.classical_engines():
        eng = engines[name]
        ok, reason = eng.available()
        if ok and name == "vapoursynth" and not vsb.supports(stages):
            ok, reason = False, "stage not implemented in the VapourSynth backend"
        if not ok:
            tried.append(f"{name}: {reason}")
            continue
        r = eng.render(src, dst, stages, work=work)
        info.update(engine=name, ok=r.ok, error=r.error, script=r.details.get("script") or r.details.get("spec"),
                    fps_out=r.details.get("fps_out"), frames=r.details.get("frames"))
        if r.ok:
            break
        tried.append(f"{name}: {r.error}")
    info["skipped"] = tried
    has_field = _has_field_stage(stages)
    if not info.get("ok"):
        reason = "; ".join(tried)
        ms = next((s for s in stages if isinstance(s, avs.MaskBottom)), None)
        vf = ",".join(x for x in ((_fallback_vf(p)[0] if has_field else None), _fallback_chroma_vf(stages),
                                  f"drawbox=x=0:y=ih-{ms.rows}:w=iw:h={ms.rows}:color=black:t=fill" if ms else None)
                      if x) or "null"
        colour = avs._rate(src)[2]
        if avs.is_full_range(colour):
            vf += ",scale=in_range=full:out_range=limited"
            colour = {**colour, "color_range": "tv"}
        info.update(engine="ffmpeg-fallback", reason=reason, filter=vf)
        cp = subprocess.run([_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(src),
                             "-map", "0:v:0", "-map", "0:a?", "-c:a", "copy", "-map_metadata", "0", "-vf", vf,
                             *avs.INTERMEDIATE, str(dst)], capture_output=True, text=True, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL)
        info.update(ok=cp.returncode == 0 and dst.exists(), error=cp.stderr.strip()[-300:] or None)
        if info["ok"]:
            info["colour"] = media.write_colour_sidecar(dst, colour)
    log(f"   pre-pass: {why} [{info['engine']}] {'ok' if info['ok'] else 'FAILED ' + str(info['error'])}")
    if not info["ok"]:
        return None, info
    info["av_sync"] = media.fix_av_offset(src, dst)
    info["aspect"] = media.fix_aspect(src, dst)
    vf, rate = _fallback_vf(p) if has_field else (None, 1.0)
    rep_ = verify_video(src, dst, src_vf=vf, src_rate=rate)
    info["guard"] = {g.name: {"status": g.status, "value": g.value} for g in rep_.gates}
    info["guard_passed"] = rep_.passed
    log(f"   pre-pass guard: {'PASS' if rep_.passed else 'FAIL ' + ','.join(rep_.failed())}")
    if rep_.passed and any(isinstance(s, avs.ChromaShift) for s in stages):
        cs = next(s for s in stages if isinstance(s, avs.ChromaShift))
        masked = any(isinstance(s, avs.MaskBottom) for s in stages)
        ch = verify_chroma(src, dst, moved={"u": [cs.u_dx, cs.u_dy], "v": [cs.v_dx, cs.v_dy]},
                           before=p.chroma.get("excess") or {"u": [0, 0], "v": [0, 0]},
                           luma_must_match=not has_field and not masked)
        info["chroma_check"] = {"passed": ch.passed, "gates": {g.name: {"status": g.status, "value": g.value,
                                                                         "detail": g.detail} for g in ch.gates}}
        log(f"   chroma check: {'PASS' if ch.passed else 'FAIL ' + ','.join(ch.failed())}")
    if rep_.passed and any(isinstance(s, avs.MaskBottom) for s in stages):
        ms = next(s for s in stages if isinstance(s, avs.MaskBottom))
        mk = verify_mask(src, dst, rows=ms.rows, luma_must_match=not has_field)
        info["mask_check"] = {"passed": mk.passed, "rows": ms.rows, "gates": {
            g.name: {"status": g.status, "value": g.value, "detail": g.detail} for g in mk.gates}}
        log(f"   mask check: {'PASS' if mk.passed else 'FAIL ' + ','.join(mk.failed())}")
    if rep_.passed:
        info["fingerprint"] = made_by
        done.write_text(json.dumps({k: v for k, v in info.items() if k != "script"}, default=str), encoding="utf-8")
    return (dst if rep_.passed else None), info


def _preview_starts(p: probe.VideoProfile) -> list[float]:
    """Spread previews over the video, nudged off scene cuts in the sampled span."""
    d = p.duration
    if d <= PREVIEW_SECONDS * 1.5:
        return [0.0]
    starts = [max(0.0, min(d - PREVIEW_SECONDS, d * f)) for f in (0.2, 0.5, 0.8)][:PREVIEW_COUNT]
    cuts = p.scene_cuts or []
    fixed = []
    for s in starts:
        for c in cuts:
            if s <= c <= s + PREVIEW_SECONDS:
                s = min(d - PREVIEW_SECONDS, c + 0.2)
        fixed.append(round(s, 2))
    return fixed


def _cut(src: Path, start: float, seconds: float, dst: Path) -> None:
    # a cut from the NUT pre-pass gets the source's colour tags written into the .mkv container
    tags = media.colour_args(media.colour_hint(src))[1]
    media.run_quiet([_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.2f}", "-t", f"{seconds:g}",
                     "-i", str(src), "-map", "0:v:0", "-map", "0:a:0?", "-sn", "-dn",   # subtitles inflated the clip's
                     "-c:v", "ffv1", *tags, "-c:a", "flac", str(dst)], check=True)      # duration (Sample05: 1118 s)


def restore_video(src: Path, out_dir: Path, *, preview_only: bool = False, encoder: str = "hevc10",
                  on_progress: Callable[[int, int], None] | None = None, log=print,
                  scale: float | None = None, content: str = "auto", model: str | None = None,
                  mask_tape_noise: bool = False, model_strict: bool = True, remove_repeats: bool = False,
                  crop_borders: bool = False, filters=None) -> VideoRecord:
    filters = cleanup.parse(filters)
    src, out_dir = Path(src), Path(out_dir)
    safe_output_dir(out_dir, src)
    mark_output(out_dir)
    work = out_dir / "work" / work_key(out_dir, src)
    (work / "previews").mkdir(parents=True, exist_ok=True)
    # resume checkpoints: the measured profile and the guarded choice of model (a resumed long job goes
    # straight to its unfinished chunks instead of repeating minutes of analysis and previews)
    prof_file, choice_file = work / "profile.json", work / "choice.json"
    src_id = source_id(src)
    cached = json.loads(prof_file.read_text(encoding="utf-8")) if prof_file.exists() else {}
    if cached.get("source") == src_id:
        prof = probe.VideoProfile(**cached["profile"])
        log("   profile: reused from checkpoint")
    else:
        if cached:
            # the file under this name changed: its pre-pass, previews, choice and chunks belong to the old one
            log("   the source changed since the last run: its checkpoints are discarded")
            wipe_checkpoints(work)
            (work / "previews").mkdir(parents=True, exist_ok=True)
        prof = probe.analyze(src)
        prof_file.write_text(json.dumps({"source": src_id, "profile": prof.to_dict()}, default=str), encoding="utf-8")
        if choice_file.exists():
            choice_file.unlink()               # a different file under the same name: choose again
    rec = VideoRecord(str(src), prof.to_dict(), "")
    rec.plan, cands = plan_candidates(prof, scale, content, model, strict=model_strict)
    options = ({"scale": scale, "content": content} | ({"model": model} if model else {})
               | ({"model_first": True} if model and not model_strict else {})
               | ({"engine": config.VIDEO_ENGINE} if config.VIDEO_ENGINE != "auto" else {})
               | ({"texture": config.TEXTURE_STRENGTH} if config.TEXTURE_STRENGTH != 1.0 else {})
               | ({"mask_tape_noise": True} if mask_tape_noise else {})
               | ({"remove_repeats": True} if remove_repeats else {})
               | ({"crop_borders": True} if crop_borders else {})
               | ({"filters": cleanup.describe(filters)} if filters else {}))
    drawn = is_anime(prof, content)
    log(f"   plan: {rec.plan}")
    for n in tape_notes(prof, masked=mask_tape_noise and "head_switching" in prof.flags):
        rec.notes.append(n)
        log(f"   note: {n}")
    if not cands:
        rec.notes.append(f"nothing to do: {rec.plan}; the original stays the deliverable")
        (work / "record.json").write_text(json.dumps(rec.to_dict(), indent=1, default=str), encoding="utf-8")
        return rec
    base = src
    if plan_prepass(prof, mask_tape_noise)[0]:
        pre, rec.prepass = run_prepass(src, work, prof, log, mask_tape_noise=mask_tape_noise)
        if pre is None:
            rec.notes.append("classical pre-pass failed or did not pass the guard: the original stays the deliverable")
            (work / "record.json").write_text(json.dumps(rec.to_dict(), indent=1, default=str), encoding="utf-8")
            return rec
        base = pre                     # progressive; audio and metadata carried along
        cc = rec.prepass.get("chroma_check") or {}
        if cc.get("passed"):
            rec.notes.append(next((w for w in str(rec.prepass.get("plan", "")).split("; ")
                                   if w.startswith("chroma shift")), "chroma shift corrected"))
        elif cc:
            rec.notes.append("chroma shift found, but the correction did not pass its check: colour left as it was")
        d = rec.prepass.get("deblend") or {}
        if d.get("guard_passed"):
            rec.notes.append(f"blended frames removed with sRestore: {float(Fraction(d['fps_in'])):.3f} -> "
                             f"{float(Fraction(d['fps_out'])):.3f} fps (blends {d['blend_share_before']:.0%} -> "
                             f"{d['blend_share_after']:.0%}); fewer frames, same duration and audio")
        elif d.get("skipped"):
            rec.notes.append(f"blended frames found but not removed: {d['skipped']}")
    # framing, each its own guarded pass on progressive frames: a pass that fails leaves the video as it was
    shape_ref = src
    if cleanup.wants_spans(filters):
        if "partly_interlaced" not in prof.flags:
            rec.notes.append("deinterlace_scenes asked for, but no interlaced stretch was measured: nothing done")
        else:
            un, rec.prepass["spans"] = run_spans(base, work, prof, log)
            d = rec.prepass["spans"]
            if un is not None:
                base = un
                rec.notes.append(f"interlaced stretches deinterlaced: {len(d['spans'])} spans, {d.get('span_frames')} "
                                 f"of {prof.combing.get('frames')} frames; every other frame is untouched")
            else:
                rec.notes.append(f"interlaced stretches found but not deinterlaced: {d.get('skipped')}")
    if remove_repeats:
        if "padded_repeats" not in prof.flags:
            rec.notes.append("repeat removal asked for, but no padding cycle was measured: nothing dropped")
        else:
            un, rec.prepass["derepeat"] = run_derepeat(base, work, prof, log)
            d = rec.prepass["derepeat"]
            if un is not None:
                base = un
                rec.notes.append(f"copied frames removed: {float(Fraction(d['fps_in'])):.3f} -> "
                                 f"{float(Fraction(d['fps_out'])):.3f} fps, the original's rate (every "
                                 f"{d.get('cycle')}th frame was a copy); fewer frames, same duration and audio")
            else:
                rec.notes.append(f"copied frames found but not removed: {d.get('skipped')}")
    if cleanup.picture_stages(filters):
        cl, rec.prepass["cleanup"] = run_cleanup(base, work, filters, log, drawn=drawn)
        d = rec.prepass["cleanup"]
        if cl is not None:
            base = cl
            rec.notes.append(f"clean-up before the model: {d['filters']}")
        else:
            rec.notes.append(f"clean-up asked for but not applied: {d.get('skipped')}")
    if crop_borders:
        if "black_borders" not in prof.flags:
            rec.notes.append("border crop asked for, but no black bar was measured: nothing cut")
        else:
            cut, rec.prepass["crop"] = run_crop(base, work, prof, log)
            d = rec.prepass["crop"]
            if cut is not None:
                base = shape_ref = cut
                b = d["borders"]
                rec.notes.append(f"black borders cut away (left {b['left']}, right {b['right']}, top {b['top']}, "
                                 f"bottom {b['bottom']} px): {prof.width}x{prof.height} -> "
                                 f"{d['crop']['width']}x{d['crop']['height']} before enlarging; the picture itself "
                                 "is untouched")
            else:
                rec.notes.append(f"black borders found but not cut: {d.get('skipped')}")
    saved = json.loads(choice_file.read_text(encoding="utf-8")) if choice_file.exists() else {}
    labels = [label for label, _ in cands]
    if saved.get("chosen") in labels and saved.get("options", options) == options and not preview_only:
        rec.chosen = saved["chosen"]
        rec.attempts = [VideoAttempt(**a) for a in saved.get("attempts", [])]
        rec.notes.append("resumed: model choice and previews reused from the checkpoint")
        log(f"   choice: {rec.chosen} (reused from checkpoint)")
        return _full_run(src, out_dir, work, base, prof, rec, dict(cands)[rec.chosen], encoder, on_progress, log,
                         drawn=drawn, shape_ref=shape_ref)
    starts = _preview_starts(prof)
    clips = []
    for i, s in enumerate(starts):
        c = work / "previews" / f"src_{i}.mkv"
        _cut(base, s, PREVIEW_SECONDS, c)
        clips.append(c)

    ref_vmaf: dict[int, float | None] = {}
    ref_vf = None                      # the AI stage always sees progressive frames
    chosen = None
    reference = cands[0][0]            # the calibrated first candidate is the VMAF reference for the others
    for label, make in cands:
        att = VideoAttempt(label, 0, len(clips))
        for i, c in enumerate(clips):
            o = work / "previews" / "out" / f"{label.replace(' ', '_')}_{i}.mkv"
            o.parent.mkdir(exist_ok=True)
            r = run_engine(c, o, make(), "ffv1")
            if not r.ok:
                att.failed.append([f"engine: {r.error}"])
                att.vmaf.append(None)
                continue
            rep = verify_video(c, o, reference_vmaf=None if label == reference else ref_vmaf.get(i), src_vf=ref_vf,
                               drawn_faces=drawn)
            v = next((g.value for g in rep.gates if g.name == "vmaf"), None)
            if label == reference:
                ref_vmaf[i] = v
            att.vmaf.append(v)
            att.failed.append(rep.failed())
            att.passed_previews += rep.passed
        rec.attempts.append(att)
        log(f"   {label:<14} previews passed {att.passed_previews}/{att.previews}  "
            + "; ".join(",".join(f) or "ok" for f in att.failed))
        if att.passed_previews == att.previews:
            chosen = (label, make)
            break
    if not chosen:
        rec.notes.append("no candidate passed the guard on every preview: the original stays the deliverable")
        return rec
    rec.chosen = chosen[0]
    choice_file.write_text(json.dumps({"chosen": rec.chosen, "options": options,
                                       "attempts": [asdict(a) for a in rec.attempts]}), encoding="utf-8")
    if preview_only:
        rec.notes.append("preview only: full run not started")
        return rec
    return _full_run(src, out_dir, work, base, prof, rec, chosen[1], encoder, on_progress, log, drawn=drawn,
                     shape_ref=shape_ref)


def _full_run(src: Path, out_dir: Path, work: Path, base: Path, prof: probe.VideoProfile, rec: VideoRecord,
              make: Callable[[], list], encoder: str, on_progress, log, drawn: bool = False,
              shape_ref: Path | None = None) -> VideoRecord:
    chosen = (rec.chosen, make)
    ref_vf = None
    dst = out_dir / f"{work.name}_restored{output_suffix(src, encoder)}"   # work.name: the stem (see work_key)
    strength = config.TEXTURE_STRENGTH
    # texture return: the model writes a lossless intermediate (an NVENC QP 16 one lost a third of what the pass
    # gives back: flat area 0.28 vs 0.19 lossless, source 0.18), the pass writes the deliverable
    model_out, model_enc = dst, encoder
    if strength > 0:
        model_enc = "lossless" if _room_for_lossless(work, prof, base, make()) else encoder
        # always Matroska: a full episode's lossless stream (32 GB) broke Video2X's MP4 writer at the very end
        # (error -22 while closing the file, header unreadable, 52 minutes lost); short clips never showed it
        # in its own folder: a pre-pass intermediate sits directly in `work`, and writing next to an input is
        # refused (PolicyError), so every pre-pass video without blend removal failed here
        model_out = work / "model" / "model_out.mkv"
        model_out.parent.mkdir(parents=True, exist_ok=True)
        (work / "model_out.mkv").unlink(missing_ok=True)      # left by an interrupted run before this layout
        if model_enc != "lossless":
            rec.notes.append("not enough free disk space for a lossless intermediate: texture returned from the "
                             "normal encode (a little less texture kept)")
    log(f"   full run with {rec.chosen} -> {dst.name}" + (f" (then source texture returned, strength {strength:g})"
                                                         if strength > 0 else ""))
    if prof.duration > LONG_VIDEO_SECONDS and not any(isinstance(s, video2x.AnimeSR) for s in make()):
        log(f"   long video ({prof.duration / 60:.0f} min): {len(chunk_plan(prof.duration))} resumable chunks")
        r = run_chunked(base, model_out, chosen[1], model_enc, work, on_progress=on_progress, log=log)
    else:
        r = run_engine(base, model_out, chosen[1](), model_enc, on_progress=on_progress)
    if not r.ok:
        rec.notes.append(f"full run failed: {r.error}")
        return rec
    if strength > 0:
        final_enc = "hevc10-grain" if encoder == "hevc10" else encoder
        log(f"   returning source texture (strength {strength:g}, final encode {final_enc})")
        t = texture_pass.apply(base, model_out, dst, strength, final_enc)
        if t.ok:
            rec.notes.append(f"source texture returned between the lines (strength {strength:g}); final encode "
                             f"{final_enc}")
            media.fix_av_offset(base, dst)
            model_out.unlink(missing_ok=True)
        else:
            rec.notes.append(f"texture return failed ({t.error}): delivered without it")
            cp = subprocess.run([_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(model_out),
                                 "-map", "0", "-c:a", "copy", *media.encoder_args(encoder), "-map_metadata", "0",
                                 str(dst)], capture_output=True, text=True, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL)
            if cp.returncode != 0 or not dst.exists():
                rec.notes.append("full run failed: could not write the deliverable from the model output")
                return rec
            model_out.unlink(missing_ok=True)
    side = carry_side_streams(src, dst)
    if side:
        rec.notes.append(side)
    # against the original: a pre-pass intermediate may have lost it. After a border crop the cropped pass is
    # the reference (same pixel shape as the original, checked by its guard; the frame itself is narrower)
    aspect = media.fix_aspect(shape_ref or src, dst)
    if aspect:
        rec.notes.append(aspect)
    final = verify_video(base, dst, reference_vmaf=None, src_vf=ref_vf, drawn_faces=drawn)
    rec.final_gates = {g.name: {"status": g.status, "value": g.value, "threshold": g.threshold} for g in final.gates}
    rec.passed = final.passed
    shown = dst
    if final.passed:
        rec.delivered = str(dst)
    else:
        # out of the results folder: under the deliverable's name it looked like a result to a person, and to the
        # next scan of the folder
        shown = work / "rejected" / dst.name
        shown.parent.mkdir(parents=True, exist_ok=True)
        shown.unlink(missing_ok=True)
        try:
            shutil.move(str(dst), str(shown))
        except OSError:
            shown = dst
        rec.notes.append(f"full output failed the guard ({final.failed()}): not delivered; kept for inspection "
                         f"in {shown}")
    # with a pre-pass the "before" frames are the deinterlaced / IVTC'd ones (same frame indices)
    rec.sheets = [str(p) for p in review_frames(base, shown, out_dir / "sheets")]
    if rec.passed and not config.KEEP_WORK:
        freed = clean_work(work)
        if freed:
            log(f"   intermediates removed ({freed / 1e9:.1f} GB); VIDIMG_KEEP_WORK=1 keeps them")
    (work / "record.json").write_text(json.dumps(rec.to_dict(), indent=1, default=str), encoding="utf-8")
    return rec


BULKY = ("chunks", "model", "previews", "deblend", "deblend_trials", "derepeat", "crop", "spans", "cleanup", "rejected", "prepass.nut", "prepass.nut.colour.json", "prepass.lwi",
         "cuts.json")


def clean_work(work: Path) -> int:
    """After a delivered result: remove the large intermediates of this video (only names this pipeline creates;
    the small records stay: profile, choice, record). Returns the bytes freed. A result the guard rejected keeps
    everything for inspection, and an interrupted run keeps everything for the resume."""
    freed = 0
    for name in BULKY:
        p = Path(work) / name
        try:
            if p.is_dir():
                freed += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                shutil.rmtree(p, ignore_errors=True)
            elif p.exists():
                freed += p.stat().st_size
                p.unlink()
        except OSError:
            pass
    return freed


def _out_pixels(stages: list, w: int, h: int) -> int:
    """Pixels per output frame for these stages (x2 when they do not say)."""
    for s in stages:
        if hasattr(s, "out_size"):
            ow, oh = s.out_size(w, h)
            return ow * oh
        if getattr(s, "scale", None):
            return round(w * h * float(s.scale) ** 2)
    return w * h * 4


def _room_for_lossless(work: Path, prof: probe.VideoProfile, base: Path, stages: list | None = None) -> bool:
    """A lossless intermediate needs about 0.3 bytes per output pixel (measured 0.23 at 1440x1152); ask for twice
    that, plus room for the deliverable. The output size comes from the stages: assuming x2 said "enough room" for
    a x4 run that needs four times as much."""
    info = media.probe(base)
    frames = info["frames"] or round(prof.duration * (prof.fps or 25))
    out_px = _out_pixels(stages or [], info["width"] or prof.width, info["height"] or prof.height)
    need = frames * out_px * 0.3 * 2
    try:
        return shutil.disk_usage(work).free > need
    except OSError:
        return False


def frames_sheet(src: Path, dst: Path, n: int = 6) -> Path | None:
    """A 3x2 sheet of frames spread over the video, for a person (or Claude) to see what it is: cartoon or
    real footage, letterbox, logos. Pixel statistics cannot reliably tell old compressed cartoons from real
    video (PLAN.md 3.7); a look can."""
    info = media.probe(src)
    if not info["duration"]:
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    inputs = []
    for k in range(n):
        inputs += ["-ss", f"{info['duration'] * (k + 0.5) / n:.2f}", "-i", str(src)]
    cells = "".join(f"[{i}:v]scale=480:-2,setsar=1[c{i}];" for i in range(n))
    rows = [f"[c{3 * r}][c{3 * r + 1}][c{3 * r + 2}]hstack=3[r{r}];" for r in range(n // 3)]
    graph = cells + "".join(rows) + "".join(f"[r{r}]" for r in range(n // 3)) + f"vstack={n // 3}"
    cp = subprocess.run([_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *inputs, "-filter_complex", graph,
                         "-frames:v", "1", str(dst)], capture_output=True, stdin=subprocess.DEVNULL)
    return dst if cp.returncode == 0 and dst.exists() else None


def output_suffix(src: Path, encoder: str = "hevc10") -> str:
    """MP4 unless the source carries streams MP4 cannot hold or plays badly: subtitles (ASS/SSA break the
    MP4 muxer outright: Sample05), font attachments, PCM audio (Sample03.avi). Those go to Matroska.
    The codec decides too: MP4 cannot hold FFV1 (Matroska) and ProRes belongs in QuickTime (.mov); both were
    offered as encoders and then failed at the very end of the run."""
    info = media.probe(src)
    if info["subtitle_codecs"] or info["attachments"] or any(str(c).startswith("pcm_") for c in info["audio_codecs"]):
        return ".mkv"
    return {"ffv1": ".mkv", "prores": ".mov"}.get(encoder, ".mp4")


def carry_side_streams(src: Path, dst: Path) -> str | None:
    """Subtitles and attachments come straight from the original (a pre-pass intermediate keeps only video and
    audio); video and audio stay as rendered. Returns a note, or None when there was nothing to carry."""
    info = media.probe(src)
    if not (info["subtitle_codecs"] or info["attachments"]) or dst.suffix.lower() != ".mkv":
        return None
    tmp = dst.with_name(dst.stem + ".side.mkv")
    cp = subprocess.run([_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(dst), "-i", str(src),
                         "-map", "0:v", "-map", "0:a?", "-map", "1:s?", "-map", "1:t?", "-c", "copy",
                         "-map_metadata", "0", str(tmp)], capture_output=True, text=True, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL)
    if cp.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        return "subtitles could not be carried over: " + cp.stderr.strip()[-200:]
    tmp.replace(dst)
    return f"kept {len(info['subtitle_codecs'])} subtitle stream(s)" + (
        f" and {info['attachments']} attachment(s)" if info["attachments"] else "") + " from the original"


def review_frames(src: Path, dst: Path, folder: Path, n: int = 3) -> list[Path]:
    """Before/after sheets for n frames spread over the video (same frame index in both)."""
    from .report import compare

    a, b = media.probe(src), media.probe(dst)
    total = min(a["frames"], b["frames"])
    idx = [int(total * (i + 1) / (n + 1)) for i in range(n)]
    out = []
    with tempfile.TemporaryDirectory(prefix="vidimg_rv_") as tmp:
        sm = _frames_by_index(src, idx, Path(tmp), "s")
        dm = _frames_by_index(dst, idx, Path(tmp), "d")
        for k, i in enumerate(idx):
            if i in sm and i in dm:
                # named after the result: after a pre-pass `src` is always "prepass", so every video in one
                # output folder wrote the same three files
                out.append(compare.sheet(sm[i], dm[i], folder / f"{dst.stem}_frame{k + 1}.jpg"))
    return out
