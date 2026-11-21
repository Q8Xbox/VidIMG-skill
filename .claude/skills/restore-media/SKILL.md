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
| **Automatic flow (default)** | `survey_media(paths)` → `decide_plan(plan, decisions)` → `run_plan(plan)` → `review_results(plan, reviews)` | For any request: one file, many files, folders, a series, or a mix. See "Automatic flow" below. |
| Compare recipes by eye | `try_recipes(plan, item_id, recipes)` | Background job → `job_status`. 2–4 recipes on one video's guarded previews, side by side on one image. Before `decide_plan`, when a look must settle the choice. |
| Check the setup | `doctor()` | First, if anything fails or on a new machine. |
| Understand the file | `analyze_media(path)` | Profile + the plan the pipeline will follow. For videos it also returns `frames_sheet`: **look at it** to decide cartoon/anime vs real footage. |
| Restore photos | `restore_photos(paths, out_dir?, deblur?)` | Background job → `job_status`. `deblur=true` only when the person asks for blur removal. |
| Restore a video | `restore_video(path, out_dir?, preview_only?, scale?, content?, model?)` | Background job → `job_status`. `content='anime'` for cartoons/anime, `'real'` for camera footage (decide from the frames sheet); `scale=2` when the person asks for exactly 2x. Use `preview_only=true` first for long videos. `model`: **you choose it** from `model_guide` and the frames sheet (see Workflow step 2); it goes first and the approved order follows if the guard rejects it. `model_strict=true` only when the person insists on one model (then a guard rejection keeps the original). |
| Follow a job | `job_status(job_id)`, `list_jobs()`, `cancel_job(job_id)`, `resume_job(job_id)` | Poll about every 30–60 s for videos, 15–30 s for photos. Jobs queue on one GPU lock; a stopped job resumes where it stopped. |
| Judge any pair | `verify_restoration(original, result)` | For files made elsewhere. |
| Before/after | `compare_frames(original, result)` | Image sheet with face zooms. |

If the MCP tools are not available, the same pipeline runs from the shell:
`python "${CLAUDE_PLUGIN_ROOT}/vidimg_run.py" restore <files|folders> [--out DIR]` and
`python "${CLAUDE_PLUGIN_ROOT}/vidimg_run.py" restore-video <video> [--out DIR] [--preview-only]`.
For several numbered episodes of one series ("episodes 1 to 26"), use the shell command
`python "${CLAUDE_PLUGIN_ROOT}/vidimg_run.py" restore-series <folder> --episodes 1-26 [--content anime] [--scale 2]`:
one output folder inside the series folder, resumable, and one English series comparison report at the end.
The series command runs for hours (about 30 min per 24-min SD episode at x2): start it in the background and read
`batch_status.json` in the output folder for progress.
Both video commands take `--engine auto|torch|mlrt` (MCP: `engine`): leave it on auto (PyTorch on NVIDIA, vs-mlrt / ONNX Runtime
DirectML on AMD or Intel GPUs) unless the person asks.

## Automatic flow (use this by default)

1. `survey_media(paths)` (background job → `job_status`). It finds every photo, single video and numbered series
   (3+ numbered episodes in one folder) under the paths, analyses them, and returns a plan: one item per unit of
   work (`photos`, `v1`, `v2`, …, `s1`, …), each with a decision already filled in (content, model from the model
   guide, scale, texture) and its reasons, plus `contact_sheets`: all videos' frame sheets on a few labelled images.
   Each video also has `closeups` (one image: line art enlarged without smoothing, the frame with the measured
   picture area outlined, a strip of consecutive frames) and `findings` (defects measured but corrected only on
   your decision).
2. **Look at every contact sheet and every video's close-ups**, then send your corrections in ONE
   `decide_plan(plan, {...})` call. You are the supervisor here, not a relay for the numbers: they are wrong often
   enough that only a look settles it (a halo score read white eyes beside dark pupils as ringing; a black bar and
   a night scene's edge measure the same). On the close-ups judge, per video:
   - `black_borders` finding → `crop_borders: true` only when everything outside the red outline is black;
   - `padded_repeats` finding → `remove_repeats: true` only when the frames marked COPY repeat a drawing in the
     middle of a movement (padding), not a drawing the animator held;
   - `partly_interlaced` finding → `filters: ["deinterlace_scenes"]` only when the comb close-ups show
     horizontal teeth on moving edges (not thin drawn stripes);
   - the line close-ups: halos, block edges, stair-steps or colour off the lines that enlarging will make worse.
     Say what you see in the `reason`; choose the model and `texture` with it in mind.
   **Compose the clean-up for this source** as a restorer would write a script for it, with `filters`, from the
   catalogue each pending item carries (`filter_catalogue`): `deinterlace_scenes`, `deblock` and `dehalo`, the
   last two at `light`, `medium` or `strong`. Use a filter only when you can see its defect, and the mildest
   strength that removes it: every filter also softens something (deblock: fine texture; dehalo: the area beside
   lines). Dehalo has no measurement at all; only your look at the line close-ups can call for it. The tool fixes
   the order, checks the script and judges each pass with its guard, so a wrong choice costs time, not fidelity;
   when unsure whether a filter helps, or which strength, put the variants through `try_recipes`.
   When two or more recipes are plausible and the choice is about how the result looks (which model keeps the
   lines and the painted backgrounds, how much texture to return), run `try_recipes` and look at its sheet before
   deciding; choose only among recipes the guard passed. The analysis
   often reads old cartoons as real footage, so confirm `content` for every video and series item, and change the
   model when what you see calls for it (the item's `guide` gives the pick for either reading; realistic drawn
   faces: `cugan-pro`). Add a short `reason`; it appears in the report. You decide; do not ask the person to choose.
   Ask them only about opt-in steps they did not mention and the survey shows are relevant: `deblur` when photos
   are soft, `mask_tape_noise` when `tape_advice` reports head-switching noise.
3. Tell the person in a few lines what was found and what you decided and why, then `run_plan(plan)` (background
   job → `job_status`; the result shows finished items while it runs). It runs the photo batch, every video and
   every series through the guarded pipelines, and writes ONE combined report (`index.html`) linking the detailed
   ones. A stopped run continues with `resume_job` or another `run_plan`: finished items are skipped.
4. **Look at the result.** For every delivered video open its before/after sheets (`result.sheets`) and record
   what you see with `review_results(plan, {id: {verdict, notes}})`: `good`, `acceptable` (name what is not
   ideal) or `redo` (the item goes back to pending: change its decision and run the plan again). The guard has
   already decided the result is faithful; your review is about how it looks (clean lines, backgrounds not turned
   to smooth blobs, no defect made sharper than in the source). It appears in the report.
5. Report per item: delivered or original kept and why, guard warnings, your review, the combined report's path.
   Ask the person to look at the faces. Their visual verdict outranks the numbers and yours.

Shell equivalent: `python "${CLAUDE_PLUGIN_ROOT}/vidimg_run.py" auto <paths> [--out DIR]` (survey; add `--run` to run at
once with the analysis' own decisions) and `python "${CLAUDE_PLUGIN_ROOT}/vidimg_run.py" run-plan --out DIR`.

## Workflow with the single-file tools (one-off or special requests)

1. `analyze_media` on each file (or a sample of a folder). For a video, open its `frames_sheet` image and decide
   whether it is a cartoon/anime or real footage: the automatic probe misses old compressed cartoons and takes drawn
   faces for real ones (measured on the owner's samples). Tell the person, briefly, what you found (content type,
   noise, compression, interlace/telecine, blended fields, chroma shift, tape defects, face sizes) and what the plan is.
   When `tape_advice` is present, pass it on: head-switching noise can be blacked out only if the person asks
   (`mask_tape_noise=True`); flagging and clipped highlights need a new capture (line TBC, lossless, lower brightness).
2. **Choose the model yourself.** `analyze_media` returns `model_guide`: the signals that matter for this video
   (content, faces and their size, noise, compression, flags), every model's measured strengths and weaknesses
   with its estimated hours here, and a suggested pick with reasons. Check the suggestion against what you see on
   the frames sheet, for example realistic drawn faces in a cartoon (cugan-pro), a long video with no faces
   (general-v3), or real faces when the faithful model fits the time (realesrnet). Then tell the person in one or two
   sentences which model you chose and why, and pass it as `model=`. Do not ask them to choose; they can still name
   a model, which you then pass with `model_strict=true` if they insist on it.
3. Start the job. Default outputs go to `<VIDIMG_OUTPUT or workspace>/review*/<time>/`; never choose an output folder that
   holds originals (the tool refuses it anyway).
4. Poll `job_status` until `done`/`failed`. While it runs, relay progress sparingly.
5. Report from the job result, in English:
   - delivered or not, per file; for videos the chosen model, the classical pre-pass (if any) and the
     final gates (VMAF, identity, structure, colour, flicker);
   - the report page path (`report` field) so the person can look at the before/after frames;
   - for anything not delivered, the reason the guard gave, and that the original is untouched.
6. Ask the person to look at the faces in the report. Their visual verdict outranks the numbers.

## What the tool uses (approved defaults — do not change on your own; the video model is your choice per video, from `model_guide`)

- **Photos:** SCUNet denoise when noisy; then BSRNet → RealESRNet → SwinIR, non-GAN models only (on the
  person's ground truth BSRNet was the most faithful: true identity 0.811). The person reviewed 19
  restored photos and said "all faces look right".
- **Video:** classical field handling first (QTGMC for interlace, TFM+TDecimate for telecine). Blended
  fields/frames from an NTSC↔PAL conversion (ghosted double images; profile flag `blended`) are un-blended
  with sRestore back to the original rate (VapourSynth; rate chosen by trial; fewer frames, same duration and
  audio; its guard requires every output frame to be a copy of an input frame). Colour sitting beside the
  line art (profile flag `chroma_shift`, >= 1.25 px agreed by the sampled frames) is moved back by resampling
  the colour planes only; its check re-measures and requires bit-identical luma. Tape defects (flags
  `head_switching`, `top_flagging`, `clipped_highlights`) are reported, not repaired; only head-switching noise
  is masked, and only on request (`mask_tape_noise`). Then the
  faithful RealESRNet / BSRNet when their estimated run fits in about 3 hours, otherwise the fast
  Real-ESRGAN general-v3. Long videos run in resumable chunks.
- **Texture return** (default strength 1): every model smooths fine texture between the lines; afterwards the
  source's own texture is put back there (lines untouched, nothing invented) and the file is encoded with x265
  grain tune. `texture=0` only when the person wants the model's clean look; lower it if returned compression
  noise bothers them.
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
| texture | Warning when much more of the area between the lines turned flat than in a plain upscale (over-smoothing, "denoised to death") |
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
