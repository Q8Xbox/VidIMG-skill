"""VidIMG MCP server (stdio) for Claude Code: faithful restoration of personal photos and videos.

Tools (PLAN.md section 4.1):
  doctor, analyze_media, restore_photos, restore_video, job_status, list_jobs, cancel_job, resume_job,
  verify_restoration, compare_frames
Restorations run as background jobs (vidimg/jobs.py): the tool returns a job id at once and
job_status reports progress, the guard's verdict and the report page.

Policy enforced here, not left to the caller:
  * originals are never written: outputs go to the workspace, and an output folder that holds an
    input is refused;
  * free tools only: open restoration models on PyTorch CUDA, Video2X for anime, AviSynth+ /
    VapourSynth for field handling; no generative models (face restorers, diffusion) are exposed;
  * every delivered result has passed the fidelity guard; otherwise the original stays the deliverable.

Run: python vidimg/mcp_server.py   (registered in .mcp.json)
"""

from __future__ import annotations

import contextlib
import sys
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):                       # launched as a script by the MCP client
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from vidimg import config, jobs  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp", ".heic"}
VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".m4v", ".mpg", ".mpeg", ".wmv", ".ts", ".mts", ".m2ts", ".3gp", ".webm"}

mcp = FastMCP("vidimg", instructions=(
    "Faithful restoration of personal photos and videos with free tools only: open non-GAN models "
    "(BSRNet, RealESRNet, SwinIR, Real-ESRGAN general) on PyTorch CUDA, SCUNet denoising, Video2X for "
    "anime, and a classical AviSynth+/VapourSynth pre-pass, every result checked by a calibrated "
    "fidelity guard (face identity, structure, colour, temporal). Typical flow: analyze_media -> "
    "restore_photos / restore_video (background job) -> job_status until done -> show the report. "
    "Never claim a result is faithful unless the guard passed it. Report to the person in Arabic."))


@contextlib.contextmanager
def _quiet():
    """Engines and pipelines print progress; stdout belongs to the MCP protocol."""
    with contextlib.redirect_stdout(sys.stderr):
        yield


def _existing(path: str) -> Path:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"not found: {p}")
    return p.resolve()


def _out_dir(out_dir: str | None, inputs: list[Path], kind: str) -> Path:
    out = Path(out_dir).expanduser().resolve() if out_dir else \
        config.OUTPUT_ROOT / kind / datetime.now().strftime("%Y%m%d-%H%M%S")
    for i in inputs:
        folder = i if i.is_dir() else i.parent
        if out == folder:
            raise ValueError(f"output folder {out} is the folder of the original {i}; "
                             "nothing is ever written next to the originals")
    return out


@mcp.tool()
def doctor() -> dict:
    """Check the installation (free tools only): the open-model GPU runtime, FFmpeg features, the
    classical engines (AviSynth+ / VapourSynth), the GPU, Video2X and the face models. Run this first
    when anything fails."""
    from vidimg import env
    from vidimg.adapters import avisynth, vapoursynth
    with _quiet():
        rep = env.doctor()
    rep["classical_engines"] = {"order": list(config.classical_engines()),
                                "avisynth": avisynth.available(), "vapoursynth": vapoursynth.available()}
    return rep


@mcp.tool()
def analyze_media(path: str) -> dict:
    """Measure a photo or video and return its degradation profile plus the restoration plan the
    pipeline would follow (nothing is written). Photos: noise, softness, JPEG quality, faces and their
    sizes, anime detection. Videos: interlace/telecine, compression, blur, noise, scene cuts, duplicates,
    faces; the classical pre-pass and the model candidates."""
    p = _existing(path)
    with _quiet():
        if p.suffix.lower() in VIDEO_EXT:
            from vidimg import pipeline_video
            from vidimg.probe import video as pv
            prof = pv.analyze(p)
            why, cands = pipeline_video.plan_candidates(prof)
            pre, pre_why = pipeline_video.plan_prepass(prof)
            sheet = pipeline_video.frames_sheet(p, config.OUTPUT_ROOT / "analysis" / f"{p.stem}_frames.jpg")
            return {"type": "video", "profile": prof.to_dict(),
                    "frames_sheet": str(sheet) if sheet else None,
                    "content_hint": ("LOOK at frames_sheet: if it is a cartoon/anime, call restore_video with "
                                     "content='anime' (the automatic probe often misses old compressed cartoons and "
                                     "mistakes drawn faces for real ones); real footage: content='real'"),
                    "plan": {"summary": why, "prepass": pre_why or None,
                             "candidates_in_order": [label for label, _ in cands]}}
        if p.suffix.lower() in IMAGE_EXT:
            from vidimg import pipeline
            from vidimg.probe import image as pi
            prof = pi.analyze(p)
            why, cands = pipeline.plan_candidates(prof)
            return {"type": "image", "profile": prof.to_dict(),
                    "plan": {"summary": why, "candidates_in_order": [label for label, _ in cands]}}
    raise ValueError(f"unsupported file type: {p.suffix}")


@mcp.tool()
def restore_photos(paths: list[str], out_dir: str | None = None, deblur: bool = False) -> dict:
    """Start a background job restoring photos (files and/or folders). For each photo: analyse, denoise
    with SCUNet if noisy, try the planned open models (BSRNet, RealESRNet, SwinIR; Video2X anime models
    for cartoons), keep the first that passes the fidelity guard, otherwise keep the original. Produces
    16-bit outputs, before/after sheets with face zooms and an Arabic review page.
    deblur=True (opt-in, only when the person asks): photos the analysis flags as soft/blurred are first
    deblurred with NAFNet-GoPro and judged by the guard's deblur mode; sharp photos are never deblurred.
    Returns the job; poll job_status(job_id)."""
    files = [_existing(x) for x in paths]
    if not files:
        raise ValueError("no input")
    out = _out_dir(out_dir, files, "review")
    label = ", ".join(f.name for f in files[:3]) + (f" (+{len(files) - 3})" if len(files) > 3 else "")
    return jobs.start("restore-photos", [str(f) for f in files] + (["--deblur"] if deblur else []), out, label)


@mcp.tool()
def restore_video(path: str, out_dir: str | None = None, preview_only: bool = False,
                  scale: float | None = None, content: str = "auto") -> dict:
    """Start a background job restoring one video: analyse; classical field handling first when needed
    (QTGMC for interlace, TFM+TDecimate for telecine); open-model candidates on three guarded preview
    segments (the faithful non-GAN models when their estimated run fits, otherwise Real-ESRGAN general;
    Video2X for anime); full run (chunked and resumable when long) with the first candidate that passes
    every preview; final guard; HEVC 10-bit with audio and metadata kept, Arabic report.
    preview_only=True stops after choosing the model.
    scale: exact output scale (2 = twice the height); default automatic (SD x2, 720p-class to 1080p).
    content: 'anime' for cartoons/anime (Video2X anime models), 'real' for camera footage, 'auto' to trust the
    probe. Decide it by looking at analyze_media's frames_sheet first. Returns the job; poll job_status(job_id)."""
    src = _existing(path)
    if src.suffix.lower() not in VIDEO_EXT:
        raise ValueError(f"not a video: {src.name}")
    out = _out_dir(out_dir, [src], "review_video")
    if content not in ("auto", "anime", "real"):
        raise ValueError("content must be auto, anime or real")
    args = [str(src)] + (["--preview-only"] if preview_only else []) + (["--scale", f"{scale:g}"] if scale else [])         + (["--content", content] if content != "auto" else [])
    return jobs.start("restore-video", args, out, src.name)


@mcp.tool()
def job_status(job_id: str) -> dict:
    """State of a restoration job (starting / queued behind another GPU job / running / done / failed /
    cancelled / crashed), the last log lines, and when finished the result: what was delivered, the
    guard's gates and the path of the report page."""
    return jobs.status(job_id)


@mcp.tool()
def list_jobs(limit: int = 20) -> dict:
    """Recent restoration jobs, newest first."""
    return {"jobs": jobs.list_jobs(limit)}


@mcp.tool()
def cancel_job(job_id: str) -> dict:
    """Stop a queued or running job (its partial outputs stay in the workspace, nothing is delivered)."""
    return jobs.cancel(job_id)


@mcp.tool()
def resume_job(job_id: str) -> dict:
    """Restart a cancelled, crashed or failed job where it stopped (same output folder): finished photos
    and finished video chunks are kept and skipped."""
    return jobs.resume(job_id)


@mcp.tool()
def verify_restoration(original: str, result: str, interpolated: bool = False) -> dict:
    """Run the calibrated fidelity guard on any original/result pair (photo or video), e.g. a result
    made outside this tool. Returns every gate with value, threshold and status, plus suggestions."""
    a, b = _existing(original), _existing(result)
    with _quiet():
        if a.suffix.lower() in VIDEO_EXT:
            from vidimg.guard.video import verify_video
            rep = verify_video(a, b, interpolated=interpolated)
        else:
            from vidimg import guard
            rep = guard.verify_image(a, b)
    return rep.to_dict()


@mcp.tool()
def compare_frames(original: str, result: str, out_path: str | None = None) -> dict:
    """Make a before/after sheet (whole image plus face zooms) for two images, saved in the workspace.
    For videos use the report of the restore job, which already holds frame sheets."""
    a, b = _existing(original), _existing(result)
    from vidimg.report import compare
    out = Path(out_path).resolve() if out_path else config.OUTPUT_ROOT / "compare" / f"{a.stem}_vs_{b.stem}.jpg"
    if out.parent == a.parent:
        raise ValueError("the sheet would be written next to the original; choose another folder")
    out.parent.mkdir(parents=True, exist_ok=True)
    with _quiet():
        return {"sheet": str(compare.sheet(a, b, out))}


def _isolate_protocol_streams() -> None:
    """stdin/stdout ARE the MCP protocol pipe. Child processes (ffmpeg, ffprobe, the model workers) inherit
    the process's standard handles: one that reads stdin swallows protocol bytes or blocks, one that
    writes stdout corrupts the stream. Keep the protocol on private duplicates and point the inherited
    handles at NUL (stdin) and stderr (stdout). The CRT's dup2 on fds 0-2 also updates the Win32 standard
    handles that subprocess hands to children."""
    import io
    import os
    proto_in, proto_out = os.dup(0), os.dup(1)
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.dup2(2, 1)
    os.close(null)
    sys.stdin = io.TextIOWrapper(os.fdopen(proto_in, "rb", buffering=0), encoding="utf-8")
    sys.stdout = io.TextIOWrapper(os.fdopen(proto_out, "wb", buffering=0), encoding="utf-8", write_through=True)


if __name__ == "__main__":
    _isolate_protocol_streams()
    mcp.run()
