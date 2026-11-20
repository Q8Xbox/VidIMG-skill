"""VidIMG MCP server (stdio) for Claude Code: faithful restoration of personal photos and videos.

Tools (PLAN.md section 4.1):
  survey_media, decide_plan, run_plan (the automatic flow: any mix of photos, videos and series folders),
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
import uuid
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
    "fidelity guard (face identity, structure, colour, temporal). Default flow for anything the person "
    "gives you (one file, many files, folders, a series): survey_media -> job_status until done -> LOOK at "
    "the contact sheets and correct the plan with decide_plan -> run_plan -> job_status until done -> show "
    "the combined report. You make the decisions (content, model) from the analysis and what you see, and "
    "say which and why; ask the person only about opt-in steps they did not mention. The single-file tools "
    "(analyze_media, restore_photos, restore_video) remain for one-off or special requests. "
    "Never claim a result is faithful unless the guard passed it. Report to the person in English."))


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
    # the default folder is unique per call: a timestamp to the second gave two calls made together one folder,
    # and each overwrote the other's record
    out = Path(out_dir).expanduser().resolve() if out_dir else \
        config.OUTPUT_ROOT / kind / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4])
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
def survey_media(paths: list[str], out_dir: str | None = None) -> dict:
    """START HERE for any restoration request. Background job: finds every photo, single video and numbered series
    (3+ numbered episodes in one folder) under the given files/folders, analyses them, and writes a plan with a
    decision already filled in per item (content, model from the model guide, scale, texture, reasons), plus
    contact sheets: all videos' frame sheets on a few labelled images. Poll job_status(job_id); its result holds
    the plan summary: items, contact_sheets and what to do next. Nothing is restored yet and nothing is written
    next to the originals."""
    files = [_existing(x) for x in paths]
    if not files:
        raise ValueError("no input")
    out = _out_dir(out_dir, files, "auto")
    label = "survey: " + ", ".join(f.name for f in files[:3]) + (f" (+{len(files) - 3})" if len(files) > 3 else "")
    return jobs.start("survey", [str(f) for f in files], out, label)


@mcp.tool()
def decide_plan(plan: str, decisions: dict[str, dict]) -> dict:
    """Correct a surveyed plan after LOOKING at its contact sheets (open each image): the analysis often reads
    old cartoons as real footage, and you may know better which model fits. plan: the survey's output folder (or its
    plan.json). decisions: {item_id: {...}} with any of
      content 'anime' | 'real' | 'auto'   (changing it re-picks the model from the guide unless you give one)
      model   esrgan-v3 | cugan | cugan-pro | general-v3 | realesrnet | bsrnet  (goes first; the approved order
              follows if the guard rejects it), model_strict true only when the person insists on that model
      scale (e.g. 2), texture 0..1 (source texture put back; default 1), mask_tape_noise (only if the person asks),
      preview_only (single videos), deblur (photos item; only if the person asks for blur removal),
      skip true, reason "why you decided this" (shown in the report).
    Confirm every video item, even when the analysis was right ({"v1": {"content": "anime"}}). Returns the summary."""
    from vidimg import auto
    with _quiet():
        return auto.summary(auto.decide(Path(plan).expanduser(), decisions))


@mcp.tool()
def run_plan(plan: str) -> dict:
    """Background job: run (or continue) a surveyed plan: the photo batch, every video and every series, each
    through its guarded pipeline with the plan's decisions, then ONE combined report (index.html) linking the
    detailed ones. Resumable: finished items are skipped. plan: the survey's output folder. Poll job_status; while
    it runs the result already shows which items are done. When finished, give the person the report path and a
    short account of what was delivered, what kept its original and why."""
    from vidimg import auto
    p = auto.plan_path(Path(plan).expanduser())
    out = Path(auto.load(p)["out"])
    return jobs.start("run-plan", [], out, f"run plan: {out.name}")


@mcp.tool()
def analyze_media(path: str) -> dict:
    """Measure a photo or video and return its degradation profile plus the restoration plan the
    pipeline would follow (nothing is written). Photos: noise, softness, JPEG quality, faces and their
    sizes, anime detection. Videos: interlace/telecine, blended fields, chroma shift, tape defects (tape_advice: head
    switching, flagging, clipped highlights), compression, blur, noise, scene cuts, duplicates,
    faces; the classical pre-pass and the model candidates; model_guide: what you need to choose the model yourself
    (signals, measured strengths/weaknesses and estimated hours per model, a suggested pick with reasons)."""
    p = _existing(path)
    with _quiet():
        if p.suffix.lower() in VIDEO_EXT:
            from vidimg import pipeline_video
            from vidimg.probe import video as pv
            prof = pv.analyze(p)
            why, cands = pipeline_video.plan_candidates(prof)
            pre, pre_why = pipeline_video.plan_prepass(prof)
            sheet = pipeline_video.frames_sheet(p, config.OUTPUT_ROOT / "analysis" / f"{p.stem}_frames.jpg")
            advice = pipeline_video.tape_notes(prof)
            guide = pipeline_video.model_guide(prof)
            return {"type": "video", "profile": prof.to_dict(), "tape_advice": advice or None, "model_guide": guide,
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
    16-bit outputs, before/after sheets with face zooms and an English review page.
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
                  scale: float | None = None, content: str = "auto", model: str | None = None,
                  engine: str | None = None, mask_tape_noise: bool = False, texture: float | None = None,
                  model_strict: bool = False) -> dict:
    """Start a background job restoring one video: analyse; classical field handling first when needed
    (QTGMC for interlace, TFM+TDecimate for telecine, chroma-shift correction, sRestore for blended fields
    from a standards conversion); open-model candidates on three guarded preview
    segments (the faithful non-GAN models when their estimated run fits, otherwise Real-ESRGAN general;
    Video2X for anime); full run (chunked and resumable when long) with the first candidate that passes
    every preview; final guard; HEVC 10-bit with audio and metadata kept, English report.
    preview_only=True stops after choosing the model.
    scale: exact output scale (2 = twice the height); default automatic (SD x2, 720p-class to 1080p).
    content: 'anime' for cartoons/anime (Video2X anime models), 'real' for camera footage, 'auto' to trust the
    probe. Decide it by looking at analyze_media's frames_sheet first.
    model: CHOOSE IT YOURSELF from analyze_media's model_guide (signals, each model's measured strengths and
    weaknesses, estimated hours, a suggested pick) and what you see on the frames sheet (drawn or real, realistic
    drawn faces, painted backgrounds), and tell the person which model and why. One of esrgan-v3, cugan, cugan-pro
    (anime/cartoon), general-v3, realesrnet, bsrnet. It goes first; if the guard rejects it on the previews the
    approved order follows, so a wrong guess costs time, never fidelity. The person may name a model instead.
    model_strict: True only when the person insists on that one model: it is then the only candidate, and if the
    guard rejects it the original stays.
    engine: leave unset (auto: PyTorch where CUDA works, vs-mlrt / ONNX Runtime DirectML otherwise); 'torch' or
    'mlrt' only when the person asks.
    mask_tape_noise: only when the person asks to hide VHS head-switching noise (analyze_media's tape_advice
    says when it was found): the bottom lines it covers are blacked out; the frame keeps its size.
    texture: source texture put back after the model, 0..1 (default 1). 0 only when the person wants the model's
    clean, smooth look; lower values when returned compression noise bothers them. Returns the job; poll
    job_status(job_id)."""
    src = _existing(path)
    if src.suffix.lower() not in VIDEO_EXT:
        raise ValueError(f"not a video: {src.name}")
    out = _out_dir(out_dir, [src], "review_video")
    if content not in ("auto", "anime", "real"):
        raise ValueError("content must be auto, anime or real")
    if model is not None and model not in config.VIDEO_MODEL_CHOICES:
        raise ValueError(f"model must be one of {config.VIDEO_MODEL_CHOICES}")
    if engine is not None and engine not in config.VIDEO_ENGINE_CHOICES:
        raise ValueError(f"engine must be one of {config.VIDEO_ENGINE_CHOICES}")
    if texture is not None and not 0 <= texture <= 1:
        raise ValueError("texture must be between 0 and 1")
    args = ([str(src)] + (["--preview-only"] if preview_only else []) + (["--scale", f"{scale:g}"] if scale else [])
            + (["--content", content] if content != "auto" else [])
            + ([("--model" if model_strict else "--prefer"), model] if model else [])
            + (["--engine", engine] if engine else []) + (["--mask-tape-noise"] if mask_tape_noise else [])
            + (["--texture", f"{texture:g}"] if texture is not None else []))
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
