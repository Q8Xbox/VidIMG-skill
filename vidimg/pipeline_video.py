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

import json
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable

from . import config
from .adapters import avisynth as avs
from .adapters import free_sr, video2x
from .adapters import vapoursynth as vsb
from . import media
from .adapters import AdapterResult, safe_output_dir
from .guard.video import _ffmpeg, _frames_by_index, verify_deblend, verify_video
from .probe import blend as blend_probe
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
                    model: str | None = None) -> tuple[str, list[tuple[str, Callable[[], list]]]]:
    why, cands = _plan_candidates(p, scale, content, model is not None)
    if model is None:
        return why, cands
    if model not in MODEL_CHOICES:
        raise ValueError(f"model must be one of {MODEL_CHOICES}")
    picked = [c for c in cands if c[0].split(" ", 1)[1] == model]
    if not picked:
        return f"{why}; requested model {model} does not apply to this content", []
    return f"{why}; only {model} (requested)", picked


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


def scene_cuts(video: Path, work: Path, log=print) -> list[float]:
    """Scene-cut times over the whole video (once; cached in the work folder for resume)."""
    cache = work / "cuts.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    log("   scanning scene cuts for chunk boundaries")
    cp = subprocess.run([_ffmpeg(), "-hide_banner", "-nostats", "-i", str(video), "-an", "-vf",
                         "scale=480:-2,scdet=threshold=10", "-f", "null", "-"], capture_output=True, text=True,
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
    info = media.probe(base)
    fps, dur = info["fps"], info["duration"]
    cdir = work / "chunks"
    cdir.mkdir(parents=True, exist_ok=True)
    cuts = scene_cuts(base, work, log)
    plan_file = cdir / "plan.json"
    if plan_file.exists():                       # resume: never re-plan under finished chunks
        plan = [tuple(x) for x in json.loads(plan_file.read_text(encoding="utf-8"))]
    else:
        plan = chunk_plan(dur, cuts=cuts)
        plan_file.write_text(json.dumps(plan), encoding="utf-8")
    on_cut = sum(1 for st, _ in plan[1:] if any(abs(st - c) < 1e-3 for c in cuts))
    log(f"   {len(plan)} chunks; {on_cut} of {max(0, len(plan) - 1)} boundaries on scene cuts")
    total = info["frames"] or round(dur * fps)
    done_frames = 0
    parts = []
    for i, (start, length) in enumerate(plan):
        part, marker = cdir / f"c{i:04d}.mp4", cdir / f"c{i:04d}.done"
        frames_want = round((start + length) * fps) - round(start * fps)
        parts.append(part)
        if marker.exists() and part.exists():
            done_frames += frames_want
            continue
        pre = min(CHUNK_CONTEXT_SECONDS, start)
        post = min(CHUNK_CONTEXT_SECONDS, dur - (start + length))
        skip = round(start * fps) - round((start - pre) * fps)
        trim = f"trim=start_frame={skip}:end_frame={skip + frames_want},setpts=PTS-STARTPTS"

        def prog(n: int, _t: int, base_n=done_frames) -> None:
            if on_progress:
                on_progress(base_n + max(0, n - skip), total)

        r = run_engine(base, part, stages_fn(), encoder, start=start - pre, duration=length + pre + post,
                         audio=False, post_vf=trim, expected_frames=frames_want, on_progress=prog)
        if not r.ok:
            return AdapterResult(False, error=f"chunk {i + 1}/{len(plan)}: {r.error}", run=r.run)
        marker.write_text(json.dumps({"start": start, "length": length, "frames": frames_want}), encoding="utf-8")
        done_frames += frames_want
        log(f"   chunk {i + 1}/{len(plan)} done ({start:.0f}-{start + length:.0f}s)")
    listing = cdir / "concat.txt"
    listing.write_text("".join(f"file '{pp.as_posix()}'\n" for pp in parts), encoding="utf-8")
    args = [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "concat", "-safe", "0",
            "-i", str(listing), "-i", str(base), "-map", "0:v:0", "-map", "1:a?", "-c", "copy",
            "-map_metadata", "1", *media.colour_args(media.colour_hint(base))[1]]
    if dst.suffix.lower() in (".mp4", ".mov", ".m4v"):
        args += ["-movflags", "+write_colr+faststart"]
    cp = subprocess.run([*args, str(dst)], capture_output=True, text=True, stdin=subprocess.DEVNULL)
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
def plan_prepass(p: probe.VideoProfile) -> tuple[list, str]:
    """AviSynth+ / VapourSynth stages that must run before any model (empty when none is needed)."""
    tff = p.field_order != "bff"
    if "blended" in p.flags:
        share = f"{p.blend_share:.0%} of moving {'fields' if 'interlaced' in p.flags else 'frames'} are blends"
        if "interlaced" in p.flags:
            return ([avs.Deinterlace(tff=tff, tr2=2 if "noisy" in p.flags else 1), avs.SRestore()],
                    f"blended fields ({share}): QTGMC double rate, then sRestore back to the original rate")
        return [avs.SRestore()], f"blended frames ({share}): sRestore back to the original rate"
    if "telecine" in p.flags:
        return [avs.IVTC(tff=tff)], f"telecine ({'TFF' if tff else 'BFF'}): IVTC with TFM+TDecimate"
    if "interlaced" in p.flags:
        noisy = "noisy" in p.flags
        return ([avs.Deinterlace(tff=tff, tr2=2 if noisy else 1)],
                f"interlaced ({'TFF' if tff else 'BFF'}): QTGMC SourceMatch=3 Lossless=2, double rate")
    return [], ""


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
    if dst.exists() and done.exists():
        info = json.loads(done.read_text(encoding="utf-8"))
        log(f"   sRestore: reusing the finished one ({info.get('fps_out')} fps)")
        return dst, info | {"reused": True}
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
    done.write_text(json.dumps(info, default=str), encoding="utf-8")
    return dst, info


def run_prepass(src: Path, work: Path, p: probe.VideoProfile, log=print) -> tuple[Path | None, dict]:
    """Classical pre-pass: field handling (when planned), then un-blending (when planned). Returns the new base,
    or None when field handling failed (the original then stays the deliverable)."""
    stages, why = plan_prepass(p)
    field_stages = [s for s in stages if not isinstance(s, avs.SRestore)]
    if field_stages:
        base, info = _run_field_prepass(src, work, p, field_stages, why, log)
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
                       log=print) -> tuple[Path | None, dict]:
    info: dict = {"plan": why}
    if not stages:
        return None, info
    dst = work / "prepass.nut"
    done = work / "prepass.done"
    if dst.exists() and done.exists() and media.colour_sidecar(dst).exists():
        info.update(json.loads(done.read_text(encoding="utf-8")), reused=True)
        log(f"   pre-pass: reusing the finished one ({info.get('engine')})")
        return dst, info
    engines = {"avisynth": avs, "vapoursynth": vsb}
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
    if not info.get("ok"):
        reason = "; ".join(tried)
        vf, _ = _fallback_vf(p)
        colour = avs._rate(src)[2]
        if avs.is_full_range(colour):
            vf += ",scale=in_range=full:out_range=limited"
            colour = {**colour, "color_range": "tv"}
        info.update(engine="ffmpeg-fallback", reason=reason, filter=vf)
        cp = subprocess.run([_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(src),
                             "-map", "0:v:0", "-map", "0:a?", "-c:a", "copy", "-map_metadata", "0", "-vf", vf,
                             *avs.INTERMEDIATE, str(dst)], capture_output=True, text=True, stdin=subprocess.DEVNULL)
        info.update(ok=cp.returncode == 0 and dst.exists(), error=cp.stderr.strip()[-300:] or None)
        if info["ok"]:
            info["colour"] = media.write_colour_sidecar(dst, colour)
    log(f"   pre-pass: {why} [{info['engine']}] {'ok' if info['ok'] else 'FAILED ' + str(info['error'])}")
    if not info["ok"]:
        return None, info
    info["av_sync"] = media.fix_av_offset(src, dst)
    info["aspect"] = media.fix_aspect(src, dst)
    vf, rate = _fallback_vf(p)
    rep_ = verify_video(src, dst, src_vf=vf, src_rate=rate)
    info["guard"] = {g.name: {"status": g.status, "value": g.value} for g in rep_.gates}
    info["guard_passed"] = rep_.passed
    log(f"   pre-pass guard: {'PASS' if rep_.passed else 'FAIL ' + ','.join(rep_.failed())}")
    if rep_.passed:
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
    subprocess.run([_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.2f}", "-t", f"{seconds:g}",
                    "-i", str(src), "-map", "0:v:0", "-map", "0:a:0?", "-sn", "-dn",   # subtitles inflated the clip's
                    "-c:v", "ffv1", *tags, "-c:a", "flac", str(dst)], check=True,       # duration (Sample05: 1118 s)
                   stdin=subprocess.DEVNULL)


def restore_video(src: Path, out_dir: Path, *, preview_only: bool = False, encoder: str = "hevc10",
                  on_progress: Callable[[int, int], None] | None = None, log=print,
                  scale: float | None = None, content: str = "auto", model: str | None = None) -> VideoRecord:
    src, out_dir = Path(src), Path(out_dir)
    safe_output_dir(out_dir, src)
    work = out_dir / "work" / src.stem
    (work / "previews").mkdir(parents=True, exist_ok=True)
    # resume checkpoints: the measured profile and the guarded choice of model (a resumed long job goes
    # straight to its unfinished chunks instead of repeating minutes of analysis and previews)
    prof_file, choice_file = work / "profile.json", work / "choice.json"
    src_id = {"path": str(src.resolve()), "size": src.stat().st_size, "mtime": int(src.stat().st_mtime)}
    cached = json.loads(prof_file.read_text(encoding="utf-8")) if prof_file.exists() else {}
    if cached.get("source") == src_id:
        prof = probe.VideoProfile(**cached["profile"])
        log("   profile: reused from checkpoint")
    else:
        prof = probe.analyze(src)
        prof_file.write_text(json.dumps({"source": src_id, "profile": prof.to_dict()}, default=str), encoding="utf-8")
        if choice_file.exists():
            choice_file.unlink()               # a different file under the same name: choose again
    rec = VideoRecord(str(src), prof.to_dict(), "")
    rec.plan, cands = plan_candidates(prof, scale, content, model)
    options = ({"scale": scale, "content": content} | ({"model": model} if model else {})
               | ({"engine": config.VIDEO_ENGINE} if config.VIDEO_ENGINE != "auto" else {}))
    drawn = is_anime(prof, content)
    log(f"   plan: {rec.plan}")
    if not cands:
        rec.notes.append(f"nothing to do: {rec.plan}; the original stays the deliverable")
        (work / "record.json").write_text(json.dumps(rec.to_dict(), indent=1, default=str), encoding="utf-8")
        return rec
    base = src
    if plan_prepass(prof)[0]:
        pre, rec.prepass = run_prepass(src, work, prof, log)
        if pre is None:
            rec.notes.append("classical pre-pass failed or did not pass the guard: the original stays the deliverable")
            (work / "record.json").write_text(json.dumps(rec.to_dict(), indent=1, default=str), encoding="utf-8")
            return rec
        base = pre                     # progressive; audio and metadata carried along
        d = rec.prepass.get("deblend") or {}
        if d.get("guard_passed"):
            rec.notes.append(f"blended frames removed with sRestore: {float(Fraction(d['fps_in'])):.3f} -> "
                             f"{float(Fraction(d['fps_out'])):.3f} fps (blends {d['blend_share_before']:.0%} -> "
                             f"{d['blend_share_after']:.0%}); fewer frames, same duration and audio")
        elif d.get("skipped"):
            rec.notes.append(f"blended frames found but not removed: {d['skipped']}")
    saved = json.loads(choice_file.read_text(encoding="utf-8")) if choice_file.exists() else {}
    labels = [label for label, _ in cands]
    if saved.get("chosen") in labels and saved.get("options", options) == options and not preview_only:
        rec.chosen = saved["chosen"]
        rec.attempts = [VideoAttempt(**a) for a in saved.get("attempts", [])]
        rec.notes.append("resumed: model choice and previews reused from the checkpoint")
        log(f"   choice: {rec.chosen} (reused from checkpoint)")
        return _full_run(src, out_dir, work, base, prof, rec, dict(cands)[rec.chosen], encoder, on_progress, log,
                         drawn=drawn)
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
    return _full_run(src, out_dir, work, base, prof, rec, chosen[1], encoder, on_progress, log, drawn=drawn)


def _full_run(src: Path, out_dir: Path, work: Path, base: Path, prof: probe.VideoProfile, rec: VideoRecord,
              make: Callable[[], list], encoder: str, on_progress, log, drawn: bool = False) -> VideoRecord:
    chosen = (rec.chosen, make)
    ref_vf = None
    dst = out_dir / f"{src.stem}_restored{output_suffix(src)}"
    log(f"   full run with {rec.chosen} -> {dst.name}")
    if prof.duration > LONG_VIDEO_SECONDS and not any(isinstance(s, video2x.AnimeSR) for s in make()):
        log(f"   long video ({prof.duration / 60:.0f} min): {len(chunk_plan(prof.duration))} resumable chunks")
        r = run_chunked(base, dst, chosen[1], encoder, work, on_progress=on_progress, log=log)
    else:
        r = run_engine(base, dst, chosen[1](), encoder, on_progress=on_progress)
    if not r.ok:
        rec.notes.append(f"full run failed: {r.error}")
        return rec
    side = carry_side_streams(src, dst)
    if side:
        rec.notes.append(side)
    aspect = media.fix_aspect(src, dst)        # against the original: a pre-pass intermediate may have lost it
    if aspect:
        rec.notes.append(aspect)
    final = verify_video(base, dst, reference_vmaf=None, src_vf=ref_vf, drawn_faces=drawn)
    rec.final_gates = {g.name: {"status": g.status, "value": g.value, "threshold": g.threshold} for g in final.gates}
    rec.passed = final.passed
    if final.passed:
        rec.delivered = str(dst)
    else:
        rec.notes.append(f"full output failed the guard ({final.failed()}): kept for inspection, not delivered")
    # with a pre-pass the "before" frames are the deinterlaced / IVTC'd ones (same frame indices)
    rec.sheets = [str(p) for p in review_frames(base, dst, out_dir / "sheets")]
    (work / "record.json").write_text(json.dumps(rec.to_dict(), indent=1, default=str), encoding="utf-8")
    return rec


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


def output_suffix(src: Path) -> str:
    """MP4 unless the source carries streams MP4 cannot hold or plays badly: subtitles (ASS/SSA break the
    MP4 muxer outright: Sample05), font attachments, PCM audio (Sample03.avi). Those go to Matroska."""
    info = media.probe(src)
    if info["subtitle_codecs"] or info["attachments"] or any(str(c).startswith("pcm_") for c in info["audio_codecs"]):
        return ".mkv"
    return ".mp4"


def carry_side_streams(src: Path, dst: Path) -> str | None:
    """Subtitles and attachments come straight from the original (a pre-pass intermediate keeps only video and
    audio); video and audio stay as rendered. Returns a note, or None when there was nothing to carry."""
    info = media.probe(src)
    if not (info["subtitle_codecs"] or info["attachments"]) or dst.suffix.lower() != ".mkv":
        return None
    tmp = dst.with_name(dst.stem + ".side.mkv")
    cp = subprocess.run([_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(dst), "-i", str(src),
                         "-map", "0:v", "-map", "0:a?", "-map", "1:s?", "-map", "1:t?", "-c", "copy",
                         "-map_metadata", "0", str(tmp)], capture_output=True, text=True, stdin=subprocess.DEVNULL)
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
                out.append(compare.sheet(sm[i], dm[i], folder / f"{src.stem}_frame{k + 1}.jpg"))
    return out
