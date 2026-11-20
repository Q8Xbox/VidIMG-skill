# VidIMG Restore

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
Requirements: Windows 10/11 64-bit, Python 3.12+ on PATH, an NVIDIA GPU (AMD and Intel GPUs run the faithful video models through vs-mlrt / ONNX Runtime DirectML). Install and usage guide: [GUIDE.md](GUIDE.md).

Design and evidence: [PLAN.md](PLAN.md) · every requirement and the install steps:
[REQUIREMENTS.md](REQUIREMENTS.md).

**Status:** phases 0–6 complete. Phase 6 acceptance: the owner reviewed 19 photos restored with the
open models and the WhatsApp video restored with RealESRNet: "all faces look right" (2026-11-16).

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
  boundaries on scene cuts, audio muxed once, resumable.
- **Texture kept:** every model also smooths away fine texture between the lines (painted backgrounds, grain,
  cloth). After the model, the source's own texture is put back between the lines (`--texture 0..1`, default 1;
  nothing invented, the lines stay as sharp as the model made them) and the result is encoded with x265's grain
  tune, because the hardware HEVC encoder flattened low-contrast texture by itself. The guard and the reports
  show the flat area between the lines against a plain upscale.
- **Automatic flow:** `survey_media` → `decide_plan` → `run_plan` (CLI `auto`, `run-plan`): give it any mix of
  photos, videos and series folders. It sorts and analyses everything, fills in a decision per item (content,
  model, scale, texture) with reasons, and shows all videos on a few contact sheets so Claude confirms "drawn or
  real" for every one by looking once. One resumable job then runs the whole plan and writes one combined report.
- **Batches:** photos are processed in rounds; photos with identical settings share one model call
  (the model loads once) and every photo is checkpointed in `<out>/items/`.
- **Guard:** face identity (SFace), face and overall structure, landmarks, eye colour, colour shift,
  invented detail; for video also frames, duration, audio, flicker and VMAF.

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

MCP tools (`vidimg/mcp_server.py`): `doctor`, `survey_media`, `decide_plan`, `run_plan` (the automatic flow),
`analyze_media`, `restore_photos`, `restore_video`, `job_status`, `list_jobs`, `cancel_job`, `resume_job`,
`verify_restoration`, `compare_frames`.
Restorations run as detached background jobs (`vidimg/jobs.py`, files under `workspace/jobs/`), one GPU
job at a time; `resume_job` continues a stopped job where it stopped.

Safety hook (`.claude/settings.json` → `vidimg/hooks/pretool.py`), before every shell or file-write call:
- **denied:** ffmpeg writing onto its own input; Video2X Real-ESRGAN / Real-CUGAN on content not declared anime;
- **asks the person first:** deleting, moving or writing photos/videos outside `workspace/`.

## Commands (shell)

```bash
python -m vidimg doctor                                            # environment report
python -m vidimg auto FILES_AND_FOLDERS [--out DIR]                # survey anything into one plan with decisions filled in
python -m vidimg run-plan --out DIR                                # run that plan (resumable) + one combined report
python -m vidimg restore PHOTOS_OR_FOLDERS [--out DIR] [--deblur]  # photo pipeline + review page (--deblur: opt-in, soft photos only)
python -m vidimg restore-video IN.mp4 [--out DIR] [--preview-only] [--scale 2] [--content anime|real|auto] [--model NAME] [--engine auto|torch|mlrt]
                                                                   # video pipeline + report.html
python -m vidimg restore-series FOLDER [--episodes 1-26] [--out DIR] [--scale 2] [--content anime|real|auto] [--model NAME]
                                                                   # numbered episodes into ONE folder (resumable) + one series report
python -m vidimg analyze FILE                                      # degradation profile and plan
python -m vidimg verify ORIGINAL RESULT                            # fidelity guard on any pair
python -m vidimg upscale IN.jpg OUT/ --model bsrnet_x4 --scale 2   # one open model on one photo
python -m vidimg denoise IN.jpg OUT/                               # SCUNet denoise
python -m vidimg video2x IN.mp4 OUT/out.mkv --processor rife       # RIFE interpolation (on request)
python -m vidimg calibrate MY_PHOTOS/ --apply                      # tune the guard on your photos (>= 20)
python vidimg_run.py setup [--output DIR] [--yes] [--check]          # first run: install + verify everything
python -m vidimg.free_install                                      # open-model runtime + models
python -m vidimg.vs_install                                        # portable VapourSynth engine
python -m vidimg.avs_install                                       # optional AviSynth+ plugin set
```

Add `--json` before the command for machine-readable output. Every external call is logged as a JSON
evidence record in `<output root>/logs/`. Outputs, jobs and logs go to `VIDIMG_OUTPUT` when set (the owner uses
`D:/VidIMG`, set in `.mcp.json` and `.claude/settings.json`), otherwise to `workspace/`; calibration state and
research evidence always stay in `workspace/`.

## Safety rules built into the code

- Outputs never go into the folder of an input, and inputs are never overwritten.
- No generative models (face restorers, diffusion upscalers); only non-GAN models for photos.
- Video2X Real-ESRGAN / Real-CUGAN / libplacebo require `content="anime"` (they distorted real faces).
- Video field handling is classical; the models only see progressive frames.
- No program from a paid software installation is ever started (`vidimg/proc.py`).

## Tests

```bash
python -m pytest            # unit tests (no programs run)
```

## License

[Apache License 2.0](LICENSE): free to use, modify and redistribute, including commercially, with an explicit
patent grant; keep the copyright and [NOTICE](NOTICE). Third-party programs and models are downloaded by setup
from their official releases under their own licenses (listed in NOTICE). Contributions: [CONTRIBUTING.md](CONTRIBUTING.md).

