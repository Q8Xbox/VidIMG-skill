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
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):                       # launched as a script by the MCP client
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from vidimg import config, jobs  # noqa: E402

from vidimg.media import IMAGE_EXT, VIDEO_EXT  # noqa: E402  (one list for every entry point)

mcp = FastMCP("vidimg", instructions=(
    "Faithful restoration of personal photos and videos with free tools only: open non-GAN models "
    "(BSRNet, RealESRNet, SwinIR, Real-ESRGAN general) on PyTorch CUDA, SCUNet denoising, Video2X for "
    "anime, and a classical AviSynth+/VapourSynth pre-pass, every result checked by a calibrated "
    "fidelity guard (face identity, structure, colour, temporal). Default flow for anything the person "
    "gives you (one file, many files, folders, a series): survey_media -> job_status until done -> LOOK at "
    "the contact sheets and each video's closeups, and correct the plan with decide_plan (try_recipes first "
    "when a look must settle the choice) -> run_plan -> job_status until done -> LOOK at the delivered "
    "videos' before/after sheets and record review_results -> show the combined report. There are no fixed "
    "paths: you compose a recipe for each video from the block library (which steps, in which order, which "
    "models, at which size, drawn or real faces), drawing on the close-ups and on the similar cases the survey "
    "returns; when no block fits what you see, add one with add_block. The guard alone decides whether a "
    "result is faithful enough to deliver, and nothing you compose can change it. Ask the "
    "person only about opt-in steps they did not mention. The single-file tools "
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
      model   esrgan-v3 | cugan | cugan-pro | janai-sd | janai-hd | general-v3 | realesrnet | bsrnet  (goes first; the approved order
              follows if the guard rejects it), model_strict true only when the person insists on that model
      scale (e.g. 2), texture 0..1 (source texture put back; default 1), mask_tape_noise (only if the person asks),
      preview_only (single videos), deblur (photos item; only if the person asks for blur removal),
      crop_borders / remove_repeats (single videos): the item's `findings` say what was measured (black bars; a
              slower original padded with copied frames). Set one true only after you LOOKED at the item's
              `closeups` image and the finding is real there: the red outline leaves only black outside it; the
              frames marked COPY repeat a drawing in the middle of a movement. Each runs as its own checked pass
              and is dropped, with a note, if its check fails,
      filters (videos and series): the clean-up YOU compose for this source, run before the model, from the
              catalogue in each pending item's `filter_catalogue`: "deinterlace_scenes" (a progressive video with
              interlaced stretches: only those are deinterlaced), {"name": "deblock", "strength": "light" | "medium" |
              "strong"} (compression block edges), {"name": "dehalo", "strength": ...} (light halos beside dark
              lines). Choose a filter only when the close-ups show its defect, the mildest strength that removes it,
              and say what you saw in the reason. The tool fixes the order, checks the script, and judges each pass
              with its guard; a pass that fails is dropped with a note. [] for none,
      recipe (videos and series): the WHOLE treatment, composed by you; it replaces the keys above (they
              remain as shorthand). {"faces": "drawn" | "real", "steps": [...], "models": [...], "model_strict":
              false, "scale": 1 | 2 | 3 | 4 | null, "texture": 0..1 | null}. faces says how the guard judges
              faces (advisory on drawings, strict on people) and is independent of the model. steps are blocks
              from the survey's `blocks` list, in the order you want them run, each a name, "name:preset" or
              {"block", "preset", "settings"}. models are tried in your order on the previews, then the usual
              order unless model_strict. scale 1 keeps the size (an anime model then runs at x2 and is scaled
              back). "interpolate": 2 | 3 | 4 ONLY when the person asks for smoother motion or for the frames a
              low-frame-rate copy is missing: a SECOND file at that many times the frame rate is written beside
              the faithful one, with frames created between the originals; tell the person how many are created.
              "rebuild_ghosts": true ONLY when the person asks to have the double images removed that deghost
              cannot replace (inside a scene, in fast movement): they are made again from their neighbours in
              that second file (alone it writes a copy at the original rate); the faithful file keeps them.
              "rebuild_damaged": [frame numbers] ONLY when the person wants damaged frames made again: the
              frames of the source (from the item's damaged.frames, or others you saw) that you judged on
              damaged.sheet to be real damage (tape tearing, dropouts, broken blocks), never a flash or a frame
              drawn that way. They are made again from their two neighbours in that second file; the faithful
              file keeps them. Tell the person how many frames are created. Frames named side by side
              (up to 3, or up to 6 where the scene holds still: see damaged.runs) are made as a run from the sound frames around it, so name a damaged
              neighbour together with its frame. Only the part of a frame that differs from the made one is
              replaced.
              "per_kind": {"k3": {"texture": 0.5, "steps": {"deblock": "strong", "dehalo": false}}} when a kind
              of scene (the item's scenes.kinds, shown on scenes.sheet) wants something else than the rest:
              its own texture strength, another preset or settings for a step of the recipe, or that step left
              out. A step may carry "kinds": ["k2"] to run in those kinds only. Single videos only; one model
              for the whole video; only script filters and texture can differ.
              Start from the item's proposed_recipe and its similar_cases / history, and change what the
              close-ups call for. Refused: a step that needs the source's frame numbers after one that changes
              them, and an anime model with faces "real". Advice about order comes back in the reasons,
      skip true, reason "why you decided this" (shown in the report).
    When two or more recipes are plausible and the choice is about how the result looks, compare them with
    try_recipes first.
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
def try_recipes(plan: str, item_id: str, recipes: list[dict]) -> dict:
    """Background job: compare 2 to 4 recipes for ONE video of a surveyed plan before deciding. Each recipe runs
    the video's three guarded previews; the result is one image, a column per recipe beside the source, plus
    per recipe whether the guard passed it. Use it when the analysis leaves a real choice that only a look can
    settle: which model keeps the line art and the painted backgrounds best, how much source texture to return
    (texture 1 / 0.5 / 0), with or without a clean-up filter and at which strength, with or without a framing
    step. recipes: [{name, recipe: {faces, steps, models, scale, texture}}, ...] (or the shorthand keys
    content, model, scale, texture, filters, remove_repeats, crop_borders); anything left out comes from the item's current decision. Poll
    job_status; then LOOK at result items[...].trials.sheet, choose only among recipes the guard passed, tell the
    person which and why, and set it with decide_plan. Nothing is delivered by a trial. A trial costs a few
    minutes per recipe (the framing passes run on the whole video), so do not use it for choices the analysis
    already settles."""
    from vidimg import auto
    p = auto.plan_path(Path(plan).expanduser())
    loaded = auto.load(p)
    out = Path(loaded["out"])
    if not any(it["id"] == item_id and it["type"] in ("video", "series") for it in loaded["items"]):
        raise ValueError(f"no video item {item_id!r} in the plan")
    if not auto.TRIAL_RECIPES[0] <= len(recipes) <= auto.TRIAL_RECIPES[1]:
        raise ValueError(f"give {auto.TRIAL_RECIPES[0]} to {auto.TRIAL_RECIPES[1]} recipes to compare")
    spec = out / "trials" / item_id / "recipes.json"
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(json.dumps(recipes, indent=1, ensure_ascii=False), encoding="utf-8")
    return jobs.start("try-recipes", [item_id, str(spec)], out, f"try recipes: {item_id} ({len(recipes)})")


@mcp.tool()
def block_library() -> dict:
    """The block library: every step a recipe can use, built-in and local, with what each does, when to use it,
    its presets and settings, what it changes and what it costs, whether it is proven or still a candidate, and
    its `record` on earlier videos (asked for, kept or dropped by the guard, how the results were reviewed).
    `defects` lists what a restorer looks for, what each looks like, what measures it, and the blocks that address
    it: an empty `blocks` list there is a gap, and the next video that shows that defect is the occasion to write
    the block. The survey returns the same lists; call this outside a survey or after add_block / note_block."""
    from vidimg import cases, library
    blocks, problems = library.load()
    return {"blocks": library.guide(blocks, cases.track_record()), "defects": library.coverage(blocks),
            "skipped": problems, "local_folder": str(library.local_dir())}


@mcp.tool()
def add_block(definition: dict) -> dict:
    """Add a new step to the local block library when a video shows a defect no existing block addresses. YOU
    grow the list: from your own knowledge of how such a defect is restored (which filter, which settings, what it
    costs), prompted by what the analysis measured (an item's `gaps`) or what you see on the close-ups. The library
    then has it for this and every later video; nothing in the tool's code changes. A new block is a CANDIDATE
    until a comparison shows it helps. definition:
      {"name": "lower_case_name", "kind": "filter", "engine": "avisynth" | "ffmpeg",
       "addresses": ["the defect(s) it is for: names from `defects`, or a new name"],
       "what": "what it does", "when": "what you must SEE on the close-ups to use it",
       "script": ["one or more lines; {setting} placeholders"],
       "params": {"setting": {"type": "int" | "float" | "choice", "min": .., "max": .., "default": ..}},
       "presets": {"light": {...}, "medium": {...}, "strong": {...}}, "default_preset": "light",
       "softens": "what it costs", "tried_on": ["where it helped"]}
    avisynth scripts use the pinned plugin set only (DeHalo_alpha, Deblock_QED, SMDegrain, SpotLess, KNLMeansCL,
    vinverse, RgTools, masktools2, mvtools2, neo-fft3d, neo-dfttest, DePan, QTGMC helpers); ffmpeg scripts are a
    plain chain of allow-listed clean-up filters. A block is a picture filter: it keeps every frame, its size and
    its rate, loads nothing, writes nothing, and does not sharpen (sharpening before the model compounds halos).
    You do not choose its check: every scripted block is judged by the full video guard. After adding it, TRY it
    with try_recipes against the recipe without it and look at the sheet before using it in a real run; then
    record what you saw with note_block (and whether it is proven). Returns the stored block, or the reason it was
    refused."""
    from vidimg import library
    path = library.save_local(definition)
    blocks, _ = library.load()
    return {"saved": str(path), "block": next(g for g in library.guide(blocks) if g["block"] == definition["name"])}


@mcp.tool()
def note_block(name: str, note: str, proven: bool | None = None) -> dict:
    """Record what a trial or a run showed about a block, built-in or local: on what kind of source, at which
    preset, what changed and what it cost ("VHS cartoon, light: the rim beside lines is gone, lines keep their
    weight; strong ate thin lines"). proven=true when a side-by-side comparison showed it helps (a candidate becomes
    proven), proven=false when it did not (it stays or goes back to candidate), omit to add the note only. The
    note appears in the block's tried_on for every later decision. This is how the library learns from the videos."""
    from vidimg import library
    b = library.note(name, note, proven)
    return {"block": b.name, "status": b.status, "tried_on": b.tried_on}


@mcp.tool()
def review_results(plan: str, reviews: dict[str, dict]) -> dict:
    """After run_plan: record how each delivered video LOOKS. Open the before/after sheets of every delivered
    item (result.sheets) and judge what the guard cannot: do the lines stay clean, did painted backgrounds turn
    to smooth blobs, are halos or blocks sharper than before, does anything look worse than the source.
    reviews: {item_id: {verdict: 'good' | 'acceptable' | 'redo', notes: "what you saw", person_verdict: "the
    person's own words about this result, when they gave any"}}. The review joins the video's case in the case
    library, where the next survey finds it for similar sources. 'acceptable' names what
    is not ideal; 'redo' sets the item back to pending: change its decision with decide_plan (another model, less
    or more texture, a framing step) and run_plan again. The verdict and notes go into the combined report, and
    you tell the person the same. Returns the summary."""
    from vidimg import auto
    with _quiet():
        return auto.summary(auto.review(Path(plan).expanduser(), reviews))


@mcp.tool()
def analyze_media(path: str) -> dict:
    """Measure a photo or video and return its degradation profile plus the restoration plan the
    pipeline would follow (nothing is written). Photos: noise, softness, JPEG quality, faces and their
    sizes, anime detection. Videos: interlace/telecine, blended fields, chroma shift, tape defects (tape_advice: head
    switching, flagging, clipped highlights), compression, blur, noise, scene cuts, duplicates,
    faces, black borders and padded repeat frames (profile.borders / profile.repeats; corrected only when asked,
    see restore_video); the classical pre-pass and the model candidates; model_guide: what you need to choose the model yourself
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
                  model_strict: bool = False, remove_repeats: bool = False, crop_borders: bool = False,
                  filters: list | None = None, recipe: dict | None = None) -> dict:
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
    drawn faces, painted backgrounds), and tell the person which model and why. One of esrgan-v3, cugan, cugan-pro, janai-sd, janai-hd
    (anime/cartoon), general-v3, realesrnet, bsrnet. It goes first; if the guard rejects it on the previews the
    approved order follows, so a wrong guess costs time, never fidelity. The person may name a model instead.
    model_strict: True only when the person insists on that one model: it is then the only candidate, and if the
    guard rejects it the original stays.
    engine: leave unset (auto: PyTorch where CUDA works, vs-mlrt / ONNX Runtime DirectML otherwise); 'torch' or
    'mlrt' only when the person asks.
    mask_tape_noise: only when the person asks to hide VHS head-switching noise (analyze_media's tape_advice
    says when it was found): the bottom lines it covers are blacked out; the frame keeps its size.
    texture: source texture put back after the model, 0..1 (default 1). 0 only when the person wants the model's
    clean, smooth look; lower values when returned compression noise bothers them.
    remove_repeats: when analyze_media's profile.repeats shows a padding cycle (a 25 or 23.976 fps original
    padded to 29.97 with copied frames) and you agree after a look: the copies are dropped, back to the original
    rate (fewer frames, same duration and audio). crop_borders: when profile.borders shows black bars and they
    are bars, not picture: they are cut away before enlarging (a narrower frame, never stretched). Each is its own
    checked pass; one that fails its check is skipped with a note.
    filters: clean-up you compose for this source, run before the model: "deinterlace_scenes" (analyze_media's
    profile.combing shows interlaced stretches), {"name": "deblock", "strength": "light|medium|strong"},
    {"name": "dehalo", "strength": ...}; only what the frames show is needed, mildest strength first.
    recipe: the whole treatment composed by you from block_library(), as in decide_plan: {"faces", "steps",
    "models", "model_strict", "scale", "texture"}. It replaces content / model / scale / filters / remove_repeats /
    crop_borders / texture. Returns the job; poll job_status(job_id)."""
    from vidimg import cleanup
    from vidimg import recipe as recipe_mod
    chosen = cleanup.parse(filters)
    if recipe is not None:
        recipe = recipe_mod.validate(recipe)[0]
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
            + (["--texture", f"{texture:g}"] if texture is not None else [])
            + (["--remove-repeats"] if remove_repeats else []) + (["--crop-borders"] if crop_borders else [])
            + (["--filters", ",".join(f["name"] + (":" + f["strength"] if f.get("strength") else "") for f in chosen)]
               if chosen else []))
    if recipe is not None:
        spec = out / "recipe.json"
        spec.parent.mkdir(parents=True, exist_ok=True)
        spec.write_text(json.dumps(recipe, indent=1, ensure_ascii=False), encoding="utf-8")
        args += ["--recipe", str(spec)]
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
