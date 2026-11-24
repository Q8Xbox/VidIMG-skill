# VidIMG Restore

[![tests](https://github.com/Q8Xbox/VidIMG-skill/actions/workflows/tests.yml/badge.svg)](https://github.com/Q8Xbox/VidIMG-skill/actions/workflows/tests.yml)

Faithful photo and video restoration for Claude Code, with **free tools only**: open restoration models
on PyTorch CUDA, Video2X for anime, AviSynth+ / VapourSynth for field handling, and FFmpeg. No paid
software is used or needed; the tool refuses to start any program from a paid software installation.
Every result goes through a fidelity guard calibrated on the owner's own photos and videos, and a result
that fails it is never delivered.

**Install (Claude Code plugin):**

```
/plugin marketplace add Q8Xbox/VidIMG-skill
/plugin install vidimg@vidimg
```

Restart Claude Code, then run `/vidimg:setup-vidimg D:/VidIMG`: it installs and verifies everything (about 5 GB of
free, SHA-256-pinned downloads into `%LOCALAPPDATA%\VidIMG`). From a clone: `python vidimg_run.py setup --output D:/VidIMG`.
Two things go outside the VidIMG folder, each only after you agree: FFmpeg (through winget) and the Python packages
of `requirements.txt`, which go into the Python that runs VidIMG; when they are already there, that Python is not touched.
Requirements: Windows 10/11 64-bit, Python 3.12+ on PATH, an NVIDIA GPU (AMD and Intel GPUs run the faithful video models through vs-mlrt / ONNX Runtime DirectML). Install and usage guide: [GUIDE.md](GUIDE.md).

Design and evidence: [PLAN.md](PLAN.md) · every requirement and the install steps:
[REQUIREMENTS.md](REQUIREMENTS.md).

**Status:** version 0.2.0; phases 0–6 complete. Phase 6 acceptance: the owner reviewed 19 photos restored with the
open models and the WhatsApp video restored with RealESRNet: "all faces look right" (2026-11-16). A full code
review followed (2026-11-20): the guard no longer passes without evidence, long runs resume safely, and ordinary
input (odd file names, rotated photos, non-Latin paths) no longer stops a run. What is still open is listed under
[Known limits](#known-limits).

## How it decides

- **Photos:** SCUNet denoise when noisy, then BSRNet → RealESRNet → SwinIR (non-GAN, most faithful first;
  on the owner's ground truth BSRNet scored true identity 0.811, +1.58 dB over bicubic). Anime/cartoons
  (no real face detected) go to Video2X's anime models.
- **Video:** classical field handling first (QTGMC for interlace, TFM+TDecimate for telecine). Blended
  fields or frames from a standards conversion (NTSC↔PAL) are detected and un-blended with sRestore back to
  the original frame rate; it only picks existing frames, and the guard checks that every output frame is
  a copy of an input frame. Colour displaced from the line art (chroma shift, common on tape copies) is
  measured and moved back by sub-pixel resampling of the colour planes only; a check re-measures it and
  requires the luma to stay bit-identical. Tape defects software cannot repair are reported in plain words:
  VHS head-switching noise in the bottom lines (blacked out only on request, `--mask-tape-noise`), the top
  of the picture bending sideways ("flagging", which needs a line time-base corrector during capture) and
  clipped highlights. Then the
  faithful RealESRNet / BSRNet when their estimated run fits in about 3 hours, otherwise Real-ESRGAN
  general-v3 (7.8 fps at 720p → 1080p). Long videos run in 2-minute chunks with context frames and
  boundaries on scene cuts, audio muxed once, resumable. Chunks are cut on exact frame numbers (checked frame by
  frame against one whole run at 29.97 fps) and planned from the counted frames, not the container's duration.
- **Texture kept:** every model also smooths away fine texture between the lines (painted backgrounds, grain,
  cloth). After the model, the source's own texture is put back between the lines (`--texture 0..1`, default 1;
  nothing invented, the lines stay as sharp as the model made them) and the result is encoded with x265's grain
  tune, because the hardware HEVC encoder flattened low-contrast texture by itself. The guard and the reports
  show the flat area between the lines against a plain upscale.
- **Colour and brightness kept through the open models:** a source without a colour tag is read the way players
  show it (BT.709 from 720 lines up, BT.601 below; its own tag always wins), where FFmpeg's default read HD as
  BT.601 and turned skin towards pink. The conversions to and from the model's RGB are rounded: FFmpeg's default
  truncates and left every frame 1.1 to 1.5 levels of 255 darker. The way back uses zscale when the installed
  FFmpeg has it (0.01 from exact); without it the result is about 0.3 levels brighter than the source.
- **Automatic flow:** `survey_media` → `decide_plan` → `run_plan` (CLI `auto`, `run-plan`): give it any mix of
  photos, videos and series folders. It sorts and analyses everything, fills in a decision per item (content,
  model, scale, texture) with reasons, and shows all videos on a few contact sheets so Claude confirms "drawn or
  real" for every one by looking once. One resumable job then runs the whole plan and writes one combined report.
- **Claude supervises, the guard vetoes:** the survey also gives Claude a close-up image per video (line art
  enlarged without smoothing, the measured picture area outlined, a strip of consecutive frames) and a list of
  findings that are corrected only when Claude confirms them by eye. `try_recipes` runs two to four recipes on a
  video's guarded previews and lays them side by side, so the choice between models or texture strengths is made
  by looking; only recipes the guard passed can be chosen. After the run Claude looks at the before/after sheets
  and records a review (`review_results`: good, acceptable, redo) that appears in the report.
- **No fixed paths: a recipe per video, from a library that grows.** The pipeline does not route a video down
  an "anime" or a "real footage" path. Every step is a block in a library (`vidimg/library.py`): field handling,
  un-blending, deinterlacing stretches, removing copied frames, cropping bars, the picture filters, every model,
  the texture return. Claude composes a recipe for each video (`vidimg/recipe.py`): whether the faces are drawn
  or real (that alone sets how strictly the guard judges faces), which steps in which order, which models, at
  which size (any model at any size: an anime model at the source's own size runs at x2 and is scaled back), how
  much texture. Only what cannot work is refused; advice about order comes back as notes. The older settings
  (`content`, `model`, `scale`, `filters`, ...) still work and are read as a recipe.
  - **Ghost frames of a low-rate copy** (`deghost`): a copy made at 10 or 12 fps with blending has single frames
    that are mixes of the picture before and the picture after. Each is replaced by the neighbour it holds more
    of, so every frame of the result is a frame of the source; mixes in runs are dissolves and stay. Its check
    confirms the replaced frames are those neighbours, the others are untouched and the ghosts are gone. A
    one-frame mix of two scenes at a cut is replaced the same way, by the scene it holds more of. Ghosts inside
    a scene in fast movement are mixes of two pictures the file holds nowhere else: no neighbour can stand in
    for them, so the faithful file keeps them. On request (`rebuild_ghosts`) they are made again from their two
    neighbours by RIFE, in the second file only, with the count in the record.
  - **AnimeJaNai models.** Two more anime models Claude can name in a recipe: `janai-sd` for old SD cartoons
    (sharp lines, painted backgrounds kept) and `janai-hd` for HD sources. They run on the vs-mlrt engine and are
    downloaded on first use from their project (3 MB, checksum pinned). Their licence is CC BY-NC-SA 4.0:
    personal, non-commercial use.
  - **Comparison clips.** Every delivered video gets a short clip of a few moments with the source and the
    result playing side by side and in step, and a second file with frames made again gets one that plays those
    places four times slower. Stills cannot show flicker, ghosts or a torn frame. The clips are in the report.
  - **Damaged frames.** The survey lists single frames that show something neither neighbour shows, and a sheet
    shows the largest between their neighbours. Tape tearing and dropouts look like that; so do a flash, lightning
    or a blink, drawn on purpose, and no measurement tells them apart, so whoever supervises looks and names the
    damaged ones. Candidates that read like tape damage (far more outlier pixels than the frames around them) are
    marked and shown first; it is a hint for where to look, with thresholds from three videos, and nothing is made
    again because of it. No model restores a torn frame. On request (`rebuild_damaged`) the named frames are made again
    from their two neighbours by RIFE, in the second file only; its check confirms every other frame is unchanged
    and that the made frames stand out less than the ones they replace. Two or three damaged frames in a row (up
    to six where the scene holds still) are found as a run and made from the sound frames either side of it; and only the part of a frame that differs
    from the made one is replaced, so the sound part of a torn frame stays the source's own.
  - **Settings by kind of scene.** The survey splits each video at its cuts and sorts the scenes into up to six
    kinds by light, motion and detail, with a sheet showing each. A recipe's `per_kind` gives a kind its own
    texture strength, another strength of a filter, or leaves a filter out there; a step's `kinds` limits it to
    some kinds. One model serves the whole video. Every variant of a filter is computed on the whole clip and
    each stretch is taken from its own; the pass's check confirms that stretches a filter leaves out are
    untouched, and the final check judges a frame of every kind.
  - **Created frames only on request.** A recipe's `interpolate` (2, 3 or 4) writes a second file at that many
    times the frame rate, with frames created between the originals (RIFE), beside the faithful file, which is
    still the delivered one. The guard confirms the original frames are unchanged in the copy; nothing can vouch
    for the created ones, and the record says how many they are.
  - **Blocks are data.** A block is one JSON file (`vidimg/blocks/`, and `<output root>/library/blocks/` for the
    ones added on this machine): what it does, when to use it, its script with settings and presets, what it
    costs. Adding a tool is adding a file (`add_block`); no code changes. A scripted block is bounded: a picture
    filter that keeps every frame, its size and its rate, from the private plugin set or an allow-list of ffmpeg
    filters, loading and writing nothing, never sharpening before the model. It cannot choose its own check:
    every scripted block is judged by the full video guard. The one exception is made in code, not in a block:
    a step that moves the picture (`stabilise`, for a trembling picture) is judged by a check written for it,
    which lines each frame up with its source before comparing them, limits how far a frame may move, and
    requires the picture to come out steadier; only a block shipped with the tool may name such a check.
  - **The library grows from knowledge and from the videos.** `vidimg/defects.json` lists what a restorer looks
    for (halos, rainbows, dot crawl, banding, spots, jitter, ...), what each looks like and what measures it, if
    anything; each block names the defects it addresses. A defect with no block is a gap: the survey reports the
    gaps a video was measured to have, and Claude writes the missing block from what it knows about restoring
    that defect. A new block is a candidate until a side-by-side comparison shows it helps (`note_block` records
    what was seen and marks it proven); a candidate in a recipe draws a note to compare first. Every block also
    carries its record from the case library: how often it was asked for, kept or dropped by the guard, and how
    the results were reviewed.
  - **Every video leaves a case.** After a run its case is written to `<output root>/library/cases/`: the
    measurements, the recipe, what the guard said, the review, the person's verdict. The next survey shows the
    most similar earlier cases beside each new video, and the file's own earlier runs.
  - **Time estimates from this machine's own runs** (`vidimg/estimate.py`). Every run records how long each stage
    took (steps, previews, model, texture return and final encode, check, second file) in its case, and the survey
    gives each pending item an estimate for its recipe as it stands: minutes, a range, the stages, and whether it
    rests on timed runs or on built-in rates. The earlier per-model figures counted the model only and were about
    half the truth on whole videos; against five timed runs the new estimates land within about 30 %.
  - **The guard is code and stays code.** Nothing in a recipe or a block can change what counts as faithful.
- **Clean-up composed per video** (`filters`, on Claude's decision, before the model; `vidimg/cleanup.py`): a
  restorer writes a script for each source, so Claude picks from a small catalogue what the close-ups call for
  and the tool builds, checks and guards the script. `deinterlace_scenes` deinterlaces only the interlaced
  stretches of an otherwise progressive video (a DVD cartoon read as progressive with 4.6 % of its frames combed;
  every frame outside the stretches is passed through untouched, and the check confirms both). `deblock` removes
  compression block edges a model would sharpen into a grid, and `dehalo` pulls back the light rim old edge
  enhancement left beside lines, each at light, medium or strong. The order is fixed, nothing sharpens, and a
  pass that fails its guard is dropped with a note.
- **Framing fixes for old cartoon transfers** (single videos, on Claude's decision, each its own checked pass):
  black bars are cut away before enlarging (`crop_borders`; lossless, even edges, pixel shape kept, and the check
  confirms that what was cut was black and the kept picture is untouched), and a slower original padded to 29.97
  with copied frames goes back to its own rate (`remove_repeats`; TDecimate chooses frames, and the check
  confirms every output frame is an input frame, no drawing was dropped and the copies are gone).
- **Batches:** photos are processed in rounds; photos with identical settings share one model call
  (the model loads once) and every photo is checkpointed in `<out>/items/`. A photo that cannot be read is
  recorded and retried on resume; the others continue. Two photos with the same file name keep their own results.
- **Guard:** face identity (SFace), face and overall structure, landmarks, eye colour, colour shift,
  invented detail; for video also frames, duration, audio, lip sync, on-screen shape, flicker and VMAF.
  - It needs evidence: when too few frames could be compared, the result fails (`coverage`) instead of passing
    on frame count and audio alone.
  - It looks at the whole video: one frame pair per minute (6 to 40) and VMAF on windows spread over its length;
    one bad stretch fails the result even when the average looks fine.
  - It reports moved frame times (`timing`, a warning) and flicker that only shows in some samples.
  - Interpolated results are compared on the source frames they kept.
- **Long runs:** every checkpoint remembers what made it. A rerun with another model, scale, encoder or source
  renders again instead of mixing old and new pieces; two files with the same name get separate work folders.
  When an encoder dies or goes silent the run ends with an error instead of hanging and holding the GPU. After a
  delivered result the large intermediates are deleted (`VIDIMG_KEEP_WORK=1` keeps them); a result the guard
  rejects is moved to `work/<name>/rejected/`, out of the results folder.
- **Files as they come:** any file name works, including Arabic and other non-Latin names and folders, apostrophes
  and `#`. Phone photos are read in their EXIF orientation, and a photo's colour profile is carried into the
  result. Photos: jpg, jpeg, png, tif, tiff, webp, bmp. Videos: mp4, mkv, mov, avi, m4v, mpg, mpeg, wmv, ts, mts,
  m2ts, 3gp, webm, vob. The container follows the codec and the source: MP4 by default, Matroska when the source
  has subtitles, attachments or PCM audio (or for FFV1), QuickTime for ProRes.

## In Claude Code

Open this folder in Claude Code and approve the project MCP server `vidimg` when asked (`.mcp.json`).

| Command | What it does |
|---|---|
| `/restore <files and/or folders>` | The automatic flow: survey everything, decide per item, run, one combined report |
| `/restore-photo <files or folders>` | Analyse, restore with guarded candidates, review page |
| `/restore-video <video> [preview]` | Classical field handling if needed, guarded previews, full run, report |
| `/setup-vidimg [output folder]` | Install / repair everything on this machine and connect Claude Code |
| `/analyze-media <file>` | Profile and plan only; writes nothing |
| `/restore-jobs [job id]` | Status and results of background jobs |

Or ask in plain words ("restore the photos in this folder"): the `restore-media` skill
(`.claude/skills/restore-media/SKILL.md`) holds the workflow and the approved policy.

MCP tools (`vidimg/mcp_server.py`): `doctor`, `survey_media`, `decide_plan`, `try_recipes`, `run_plan`,
`review_results` (the automatic flow), `block_library`, `add_block`, `note_block` (the library recipes are
composed from, and how it grows),
`analyze_media`, `restore_photos`, `restore_video`, `job_status`, `list_jobs`, `cancel_job`, `resume_job`,
`verify_restoration`, `compare_frames`.
Restorations run as detached background jobs (`vidimg/jobs.py`, files under `<output root>/jobs/`), one GPU
job at a time, first come first served; `resume_job` continues a stopped job where it stopped and `cancel_job`
works at any moment, also before the job has started. Decisions sent with `decide_plan` while a plan is running
are kept: items not started yet use them, and the item running at that moment is set back to pending.

Safety hook (`.claude/settings.json` → `vidimg/hooks/pretool.py`), before every shell or file-write call:
- **denied:** ffmpeg writing onto its own input; Video2X Real-ESRGAN / Real-CUGAN on content not declared anime;
- **asks the person first:** deleting, moving or writing photos/videos outside `workspace/`.

## Commands (shell)

```bash
python -m vidimg doctor                                            # environment report
python -m vidimg auto FILES_AND_FOLDERS [--out DIR]                # survey anything into one plan with decisions filled in
python -m vidimg run-plan --out DIR                                # run that plan (resumable) + one combined report
python -m vidimg restore PHOTOS_OR_FOLDERS [--out DIR] [--deblur]  # photo pipeline + review page (--deblur: opt-in, soft photos only)
python -m vidimg restore-video IN.mp4 [--out DIR] [--preview-only] [--scale 2] [--content anime|real|auto]
                               [--model NAME | --prefer NAME] [--engine auto|torch|mlrt] [--texture 0..1]
                               [--mask-tape-noise] [--encoder hevc10|h264|prores|ffv1]
                                                                   # video pipeline + report.html
python -m vidimg restore-series FOLDER [--episodes 1-26] [--out DIR] [--scale 2] [--content anime|real|auto]
                               [--model NAME | --prefer NAME] [--texture 0..1] [--mask-tape-noise] [--report-only]
                                                                   # numbered episodes into ONE folder (resumable) + one series report
python -m vidimg analyze FILE                                      # degradation profile and plan
python -m vidimg verify ORIGINAL RESULT [--interpolated]           # fidelity guard on any pair
python -m vidimg upscale IN.jpg OUT/ --model bsrnet_x4 --scale 2   # one open model on one photo
python -m vidimg denoise IN.jpg OUT/                               # SCUNet denoise
python -m vidimg video2x IN.mp4 OUT/out.mkv --processor rife       # RIFE interpolation (on request)
python -m vidimg calibrate MY_PHOTOS/ --apply                      # tune the guard on your photos (>= 20)
python vidimg_run.py setup [--output DIR] [--yes] [--check]          # first run: install + verify everything
python -m vidimg.free_install                                      # open-model runtime + models
python -m vidimg.vs_install [--relock]                             # portable VapourSynth engine (--relock: accept its files after a deliberate update)
python -m vidimg.avs_install                                       # optional AviSynth+ plugin set
```

`--model` uses only that model; `--prefer` tries it first and falls back to the approved order. Add `--json`
before the command for machine-readable output. Every external call is logged as a JSON evidence record in
`<output root>/logs/`. Outputs, jobs and logs go to the output root: the `VIDIMG_OUTPUT` environment variable when
set, otherwise the folder chosen at setup (stored in `vidimg.local.json`), otherwise `workspace/`. Calibration
state and research evidence always stay in `workspace/`. Each results folder carries a small `.vidimg-output`
marker, so a later survey of a parent folder does not take its work files for your media.

## Safety rules built into the code

- Outputs never go into the folder of an input, and inputs are never overwritten.
- No generative models (face restorers, diffusion upscalers); only non-GAN models for photos.
- Video2X Real-ESRGAN / Real-CUGAN / libplacebo require `content="anime"` (they distorted real faces).
- Video field handling is classical; the models only see progressive frames.
- No program from a paid software installation is ever started (`vidimg/proc.py`).
- A result the guard could not check is not delivered: no compared frames means a failure, not a pass.
- Downloads are checked, and the check is enforced: a model whose SHA-256 differs from the recorded one is set
  aside and not used, and the VapourSynth engine refuses to run while one of its files differs from the first install.
- `calibrate --apply` only writes thresholds inside a plausible range, so a bad fit cannot switch a gate off.

## Tests

```bash
python -m pytest            # about 330 tests: under a minute without FFmpeg, about four minutes with it
```

No GPU, models, FFmpeg or media files are needed: the engines are replaced by stubs. With FFmpeg on the path a
few more tests run real conversions on generated frames. The same suite runs on GitHub for every pull request
and every push to `main` (Windows, Python 3.14, no FFmpeg). Tests that need FFmpeg or a private benchmark photo
skip without them.

## Known limits

- **Variable frame rate** (many phone clips): the result is written at a constant rate. The guard warns when
  frames moved in time (`timing`), but the drift itself is not corrected.
- **HD footage that needs a pre-pass** (interlaced or telecined HD): by the rule above the open models
  now read it with the HD colour matrix, but no such footage has been run since. The Video2X anime models go
  through another path, which has not been checked on it either.
- **Several newer guard limits are chosen values, not calibrated ones:** the sampling density, the flicker and
  timing limits, and the calibration ranges. They behaved correctly on the owner's samples.
- **`.heic` photos** are not supported; convert them first.
- **Windows only.**

## License

[Apache License 2.0](LICENSE): free to use, modify and redistribute, including commercially, with an explicit
patent grant; keep the copyright and [NOTICE](NOTICE). Third-party programs and models are downloaded by setup
from their official releases under their own licenses (listed in NOTICE). Contributions: [CONTRIBUTING.md](CONTRIBUTING.md).

