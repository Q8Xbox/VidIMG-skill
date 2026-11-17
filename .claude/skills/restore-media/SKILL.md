---
name: restore-media
description: Faithfully restore personal photos and videos (upscale, denoise, deinterlace, recover detail) with the VidIMG tool — free tools only (open non-GAN models on PyTorch CUDA, SCUNet, Video2X for anime, AviSynth+/VapourSynth field handling), every result checked by a calibrated fidelity guard. Use when the person asks to restore, enhance, upscale, clean up, deinterlace or improve the quality of their photos or videos, or to check whether an enhanced file still looks like the original.
---

# Restore media faithfully (VidIMG, free tools only)

The goal is **more quality without changing who or what is in the picture**: recover missing pixels,
never invent a different face. A result that fails the guard is not delivered; the original stays the
deliverable. Report to the person in **English**, even when they write in another language; keep file paths and technical
names as they are. **Only free tools are used**; never suggest or run paid software.

## First use on a machine

If the `vidimg` MCP tools are missing, or `doctor()` reports a failure, run `python "${CLAUDE_PLUGIN_ROOT}/vidimg_run.py" setup --check`
in the shell. When it is not READY, follow `/setup-vidimg` (explain the ~5 GB of free downloads, ask for the
output folder, then `python "${CLAUDE_PLUGIN_ROOT}/vidimg_run.py" setup --yes --output <folder>`; FFmpeg via winget only with a clear yes).
After setup the person restarts Claude Code (a cloned project also approves the `vidimg` MCP server).
(`${CLAUDE_PLUGIN_ROOT}` is the VidIMG folder. When VidIMG is opened as a project instead of installed as a plugin, that text stays unfilled: use the project folder, e.g. `python vidimg_run.py setup`.)

## Tools (MCP server `vidimg`)

| Step | Tool | Notes |
|---|---|---|
| Check the setup | `doctor()` | First, if anything fails or on a new machine. |
| Understand the file | `analyze_media(path)` | Profile + the plan the pipeline will follow. For videos it also returns `frames_sheet`: **look at it** to decide cartoon/anime vs real footage. |
| Restore photos | `restore_photos(paths, out_dir?, deblur?)` | Background job → `job_status`. `deblur=true` only when the person asks for blur removal. |
| Restore a video | `restore_video(path, out_dir?, preview_only?, scale?, content?, model?)` | Background job → `job_status`. `content='anime'` for cartoons/anime, `'real'` for camera footage (decide from the frames sheet); `scale=2` when the person asks for exactly 2x. Use `preview_only=true` first for long videos. `model` only when the person picks one (e.g. `cugan-pro` after comparing): it becomes the only candidate, and a guard rejection keeps the original. |
| Follow a job | `job_status(job_id)`, `list_jobs()`, `cancel_job(job_id)`, `resume_job(job_id)` | Poll about every 30–60 s for videos, 15–30 s for photos. Jobs queue on one GPU lock; a stopped job resumes where it stopped. |
| Judge any pair | `verify_restoration(original, result)` | For files made elsewhere. |
| Before/after | `compare_frames(original, result)` | Image sheet with face zooms. |

If the MCP tools are not available, the same pipeline runs from the shell:
`python "${CLAUDE_PLUGIN_ROOT}/vidimg_run.py" restore <files|folders> [--out DIR]` and
`python "${CLAUDE_PLUGIN_ROOT}/vidimg_run.py" restore-video <video> [--out DIR] [--preview-only]`.

## Workflow

1. `analyze_media` on each file (or a sample of a folder). For a video, open its `frames_sheet` image and decide
   whether it is a cartoon/anime or real footage: the automatic probe misses old compressed cartoons and takes drawn
   faces for real ones (measured on the owner's samples). Tell the person, briefly, what you found (content type,
   noise, compression, interlace/telecine, face sizes) and what the plan is.
2. Start the job. Default outputs go to `<VIDIMG_OUTPUT or workspace>/review*/<time>/`; never choose an output folder that
   holds originals (the tool refuses it anyway).
3. Poll `job_status` until `done`/`failed`. While it runs, relay progress sparingly.
4. Report from the job result, in English:
   - delivered or not, per file; for videos the chosen model, the classical pre-pass (if any) and the
     final gates (VMAF, identity, structure, colour, flicker);
   - the report page path (`report` field) so the person can look at the before/after frames;
   - for anything not delivered, the reason the guard gave, and that the original is untouched.
5. Ask the person to look at the faces in the report. Their visual verdict outranks the numbers.

## What the tool uses (approved defaults — do not change on your own)

- **Photos:** SCUNet denoise when noisy; then BSRNet → RealESRNet → SwinIR, non-GAN models only (on the
  person's ground truth BSRNet was the most faithful: true identity 0.811). The person reviewed 19
  restored photos and said "all faces look right".
- **Video:** classical field handling first (QTGMC for interlace, TFM+TDecimate for telecine); then the
  faithful RealESRNet / BSRNet when their estimated run fits in about 3 hours, otherwise the fast
  Real-ESRGAN general-v3. Long videos run in resumable chunks.
- **Anime / cartoons** (`content='anime'`, or detected with no real face): Video2X Real-ESRGAN anime-v3 first,
  Real-CUGAN second, general-v3 third (cartoon ground truth, PLAN.md 3.7). Face gates are advisory on drawn faces.
  Real-ESRGAN / Real-CUGAN anime models are never used on real faces (they distorted them in testing).
- **Outputs** go to `VIDIMG_OUTPUT` (the owner's D:/VidIMG). Videos with subtitles, font attachments or PCM audio
  are written as MKV and keep those streams; lip sync (video/audio start offset) is restored and checked.
- **No generative models** (face restorers such as GFPGAN/CodeFormer, diffusion upscalers): they change
  identity. **No colour or lighting changes.** **Frame interpolation only on request** (RIFE via Video2X).
- **Output:** photos 16-bit PNG; videos HEVC 10-bit with audio and metadata copied.
- **Never send the person's photos or videos to external services.** Everything runs locally.

## What the guard means (for explaining results)

| Gate | Meaning |
|---|---|
| identity | SFace face-embedding similarity per face (two tiers: below 0.79 fails; 0.79–0.84 must also keep face structure) |
| face_backprojection / backprojection | Does the result, scaled back down, still match the original (structure kept, nothing invented)? The whole-image one compares structure only (brightness is the colour gate's job, so a slightly deeper black on a dark frame does not fail it) and only fails below 0.93 (removing JPEG damage is allowed) |
| landmarks, eye_color | Facial geometry and eye colour unchanged |
| color | ΔE colour shift ≤ 4 |
| hallucination | Warning when much more fine texture appears than the original could hold |
| frames, duration, audio | Video structure identical |
| av_sync | Video/audio start offset unchanged (lip sync) |
| aspect | On-screen shape unchanged (pixel aspect kept: a PAL 720x576 at 16:15 must stay 4:3, not 5:4) |
| flicker | Added temporal flicker ≤ 0.02 |
| vmaf | Perceived quality relative to the first (reference) candidate on the same clip |

A warning (`warn`) is delivered with a note; a failure (`fail`) is never delivered.

## Limits to state honestly

- Non-GAN models are faithful but look **softer** than tools that invent texture.
- Faces under about 20–24 px in the source cannot be restored reliably; the tool keeps the original there.
- Very low-bitrate damage cannot be fully undone; the result is cleaner, not a new recording.
- Deblurring is opt-in (`deblur=true`) and runs only on photos the analysis flags as soft (NAFNet-GoPro, checked by the guard's deblur mode). Strong on camera-shake blur, small gain on out-of-focus blur; photos only. A deblur the guard rejects is skipped and the photo continues without it.

Background and evidence: `PLAN.md`, `REQUIREMENTS.md`, `research/avisynth/05_integration_design.md`.
