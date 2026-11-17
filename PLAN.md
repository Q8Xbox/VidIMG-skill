# "VidIMG Restore" plan (version 4): a Claude Code tool that restores photos and video faithfully to the original, with free tools only

> **Date:** 2026-11-16 · **Machine:** Windows 11 Pro · i9-12900K · 64GB RAM · NVIDIA RTX 5060 Ti 16GB
> **Principle:** free tools only (open models, Video2X, AviSynth+/VapourSynth, FFmpeg), and every decision rests on **measurement against the true original** of your photos and clips.
> **Full requirements and installation steps:** [`REQUIREMENTS.md`](REQUIREMENTS.md) · **Usage guide:** [`README.md`](README.md)

### What's new in version 4
- 🧹 **Every plan or feature that depended on paid software has been removed** at your request (2026-11-16): no Topaz (Gigapixel, Photo AI, Video AI and Sharpen AI), no engine adapter, no sharpening step, and no stage 7. The tool **refuses to run any program from a paid program's folder**.
- 📦 The removed code is kept for the record only in [`archive/topaz/`](archive/topaz/) (no part of the tool imports it), together with the previous version of this plan.
- ✅ The tool is complete for free: unit tests pass, `doctor` is clean, and photo and video restoration work end to end.

---

## 0. Summary

| Aspect | Decision | Evidence (against the true original) |
|---|---|---|
| Photo upscaling | BSRNet → RealESRNet → SwinIR (models that do not invent texture: non-GAN) | BSRNet: true identity **0.811** and +1.58 dB over simple upscaling; RealESRNet 0.798; SwinIR 0.782 |
| Photo noise | SCUNet before upscaling | A model trained on real photos, with no invented texture |
| Video | RealESRNet/BSRNet if the estimated run is ≤ 3 hours, otherwise Real-ESRGAN general-v3 | RealESRNet: VMAF 90.1 and identity 0.889; general-v3: 93.5 and 0.868 but faster (7.8 fps at 720p) |
| Interlacing and telecine | QTGMC, and TFM+TDecimate, with classical processing first | QTGMC 42.3 dB vs bwdif 40.4; TFM+TDecimate bit-exact on clean telecine |
| Anime | Video2X (Real-CUGAN and Real-ESRGAN anime) | Never used on real faces |
| Guard | Face identity, structure, facial features, eye color, color, invented detail, flicker, and VMAF | Accepts 61/65 faithful and rejects 27/35 harmful across 100 labeled outputs |
| Acceptance | **Your verdict: "all faces intact"** on 19 photos and the WhatsApp video (free tools only) | `workspace/acceptance/human_verdict.json` |

---

## 1. Policy (changes only by your decision)

1. **Free tools only.** No paid software, no license, no subscription.
2. **Fidelity before beauty:** no generative models (face restorers such as GFPGAN/CodeFormer, or diffusion upscaling), and no GAN models for photos.
3. **The original is never touched:** nothing is written next to or over the originals; outputs go in `workspace/`.
4. **The guard is mandatory:** what it rejects is not delivered and the original stays; a warning is delivered with a note.
5. **No change to colors or lighting**, and **interpolation only on request** (RIFE via Video2X).
6. **Output:** 16-bit PNG photos, and HEVC 10-bit video with audio and metadata copied.
7. **Everything is local:** your photos and clips are never sent to any external service.
8. **Report language:** English.

---

## 2. Environment

Full details in [`REQUIREMENTS.md`](REQUIREMENTS.md). In short:
- **Windows 10/11, and an NVIDIA card with at least 6GB of memory** (the tool runs on the CPU without one, but very slowly).
- **Software:**
  - Python 3.14
  - FFmpeg 8 (full build)
  - The open-model environment `tools/sr-venv` (PyTorch CUDA with spandrel), with 7 models pinned by SHA-256 hash
  - Portable VapourSynth or AviSynth+
  - Video2X
  - YuNet/SFace face models
- **Installation from scratch** in five commands, and `python -m vidimg doctor` checks everything.

---

## 3. Evidence and measurements (all against the true original)

### 3.1 Photos (12 of your photos, controlled degradation: downscaling plus JPEG)
| Model | Identity match with the original | Gain over simple upscaling |
|---|---|---|
| **BSRNet** (default) | **0.811** | **+1.58 dB** |
| RealESRNet | 0.798 | +0.62 dB |
| Classical upscaling (Lanczos) | 0.785 | 0.00 |
| SwinIR | 0.782 | +0.24 dB |
| Real-ESRGAN (GAN) | 0.742–0.759 | −1.0 to −2.7 dB |

On 13 photos with small faces, 12 were delivered. In the final acceptance, 19 of 20 upscalable photos were delivered, in 4 minutes.

### 3.2 Video (12 clips from your 720p videos, low-quality clip vs the original)
| Model | VMAF | Identity | PSNR | Speed (720p → 1080p) |
|---|---|---|---|---|
| RealESRNet | 90.1 | 0.889 | 36.89 | 0.47 fps |
| BSRNet | 87.8 | 0.889 | 38.69 | similar |
| general-v3 | 93.5 | 0.868 | 33.60 | **7.8 fps** |

> This measurement was rerun after fixing F14 (a duplicated frame that shifted the outputs in 6 of 12 clips, so the earlier figures came out lower than they really were: 84.5/82.0/89.4). The old one is kept in `workspace/bench_free/video/summary_before_fpsfix.json`.

The rule: the faithful models are used if their run is estimated at three hours or less (frames × pixels, with interlaced material counted double); otherwise general-v3 is used and the guard protects the faces.

### 3.3 Interlacing and telecine ([`research/avisynth/05_integration_design.md`](research/avisynth/05_integration_design.md))
| Test | Best | Comparison |
|---|---|---|
| Deinterlacing (PSNR) | **QTGMC SourceMatch=3: 42.29 dB** (worst frame 39.79) | bwdif: 40.42 (35.06) |
| Clean telecine | **TFM+TDecimate: bit-exact** | fieldmatch+bwdif: 89.3 dB (worst frame 25.7) |
| Compressed telecine | **46.2 dB** | 43.3 dB, while wrong deinterlacing gives 21 dB |
| Cleanup before the model (Deblock/SMDegrain) | **No gain** (VMAF −0.3 to −0.4 on 12/12) | so it stays off |

AviSynth+ is the primary engine, VapourSynth a second engine with proven parity (±0.1 dB), and ffmpeg a last fallback.

### 3.4 The guard
- **Calibration:** on your photos and clips, with labels from the true original (identity in two tiers 0.79/0.84, face structure 0.945, color ΔE ≤ 4, flicker ≤ 0.02).
- **The global structure gate for photos** is now a warning above 0.93: across 100 labeled outputs it did not catch a single harmful output on its own, and the decisions are identical before and after the change.
- **Important lesson:** "back-projection" raises the guard's scores but **moves the output away from the true original** (0.811 → 0.774) because it copies the input's defects, so it was rejected. Model decisions are therefore based on the true original, not on the guard alone.
- **WhatsApp video with the free models (2026-11-16):** **delivered with RealESRNet** after fixing F14 and enabling "structure only" (`workspace/acceptance/video_whatsapp_free3`): 3743 matching frames, identity 0.811, face structure 0.992, global structure 0.996, ΔE 1.8, flicker 0.0002, and VMAF 84.0. Before that it was not delivered in two attempts:
  - The first cause was a **bug** in the free video engine, not in the faces: a duplicated frame at the start shifted every frame (F14), so the guard saw "changed" faces in the moving-camera shot. Fixed.
  - After the fix, each model passed two of three previews: BSRNet and general-v3 fail on a scene with two small pictures (18–20 pixels) hanging on the wall that the model smoothed slightly, and RealESRNet fails another preview on the global structure gate (0.854). A new full run with the fixed engine (`workspace/acceptance/video_whatsapp_free2`) confirmed the same result.
  - RealESRNet's failure was caused by a near-black title frame with faint text: the model cleaned compression noise out of the black, and on a nearly flat frame the structure metric drops sharply (0.79) even though the two images match visually and there are no faces. This is a weakness of the global structure gate on dark frames; addressing it first requires measuring on the labeled outputs.
- **Measurement of "structure gate on structure only" (2026-11-16):** the option `guard.BACKPROJ_STRUCTURE_ONLY` (**enabled by your decision** after the measurement) drops the luminance term from SSIM and leaves luminance to the color gate (ΔE). The luminance term in SSIM is relative, so near black a darkening of three levels reads as a structural collapse (0.79 vs 0.997 for structure only).
  - 385 labeled photo outputs and 72 video outputs: **no decision changes** (change in the value: median 0.0003).
  - Stress test on the valid set (161 outputs): an entirely dark scene and a 20% lighting change: decisions identical. Crushing the blacks (everything below 24 becomes black): **one output in 161 slips through** that used to be rejected by structure alone.
  - An attempt to close that gap with a "black-level shift" gate failed: faithful outputs shift the shadows by up to ±8–12 levels, and crushing starts at −5, so there is no separating threshold.
  - With the option: RealESRNet passes all three WhatsApp previews.
  - General note: when the whole image is dark, the entire guard weakens (it rejected only 7 of 35 harmful outputs, with or without the option).
- **Measurement of "tolerance for small faces in the facial-feature and face-structure gates" (under 24 pixels in the source):** the option exists in the code (`guard.STRUCTURE_MIN_SOURCE_FACE`) but is **off**.
  - The valid calibration set (faces ≥ 40 pixels, 77 outputs) and 72 video outputs: **no decision changes**.
  - Small-face photos (the first calibration, 175 outputs): 61 decisions changed, including **24 harmful outputs that had been rejected and became accepted** (among them generative face restorers), versus 20 additional faithful outputs. In other words, this gate is the only protection for very small faces, so enabling the option is not recommended.
- **Recalibration of the video gates on the free models (2026-11-16)** (`workspace/calibration_video/20261116-153325`, 12 clips from your videos and 60 outputs):
  - The relative VMAF gate **stayed at 4.0**: harmful outputs deviate from the reference candidate by only 0.02–2.3, so VMAF does not separate them. The arithmetically "best" threshold (0) follows a labeling that penalizes softness (BSRNet and RealESRNet with identity 0.90–0.93) and would have rejected faithful outputs.
  - With identity labeling (the tool's policy): the whole guard accepts **31 of 33** faithful and rejects **9 of 10** harmful (temporal boiling 9 of 9 via the flicker and face gates). The only output that slipped through was general-v3 on a single clip from MBZ, whose identity is low for all candidates (0.40–0.55).

### 3.5 Batches and long videos
- **Grouping photos by settings** into a single call (the model is loaded once), with a resume point for each photo.
- **Long videos in chunks** of two minutes, with one second of context on each side that is trimmed precisely, boundaries on scene cuts, audio muxed once, and resumption from the next chunk.
- **Proof of resumption:** a 56-minute video was interrupted twice, then resumed and delivered without intervention, with frames, audio and duration matching the original (the chunk mechanism is independent of the model used).

### 3.6 Free deblurring (measured 2026-11-16, **optional for photos**)
Open models (MIT) supported by spandrel: NAFNet-GoPro (motion), NAFNet-REDS (blur with JPEG), and FFTformer-RealBlur (real camera blur). On 25 of your photos with controlled degradation, against the original (`workspace/bench_deblur`):

| Case | Best | Gain | Identity with the original | Current guard |
|---|---|---|---|---|
| Motion blur | **NAFNet-GoPro** | **+4.6 dB** | 0.52 → **0.64** | Accepts only 3 of 25 |
| Lens blur (the most common) | No useful model; classical sharpening +1.4 and NAFNet +0.9 | Small | 0.61 → 0.66 | 22/25 |
| Non-blurred photo | **Every model harms it** | NAFNet −6.4 dB | 0.92 → 0.86 | — |

- The current blur detection separates well on this data (no false alarm on 25 sharp photos; 24/25 lens, 22/25 motion), but it counts 12-megapixel phone photos as "soft" (5 of your originals), and on one of them the improvement was slight and NAFNet slightly changed the shape of the eye.
- **Why it was not adopted:** (1) the guard compares against the blurry input, so it rejects correct deblurring (identity 16 and face structure 22 of 25), and it needs a special mode calibrated on these labeled outputs; (2) there is no detector that tells motion blur from lens blur; (3) there is no useful model for lens blur (KBNet for lens blur is available only through a OneDrive folder that requires signing in).
- The trial weights, their sources and their hashes are in `tools/free-models-trial/SOURCES.json` (not used by the tool).
- **What was built afterwards (option `--deblur` / `deblur=True`, off by default):**
  - **The detector:** I tried two detectors to tell motion from lens blur (gradient direction, and the drop in autocorrelation at the blur length), **and neither separated them** on your photos. It turned out not to be needed: NAFNet-GoPro did no harm on lens blur (+0.9 dB and identity 0.61 → 0.66). So the step runs on every photo the analysis flags as "soft" (`soft_or_blurred`, which made no mistake on 25 sharp photos), and never on sharp ones.
  - **Guard mode for deblurring:** it searches for the blur (Gaussian, or a motion line with an angle and length) that maps the output back to the input, then applies the same gates to the re-blurred output, just as upscaling is judged by downscaling it. On the realistic cases: motion **17 of 23** correct outputs accepted (the normal guard: 2), and lens **21 of 21**. The six rejected are group photos with small faces (a conservative error: they continue without deblurring). This mode does not distinguish "extra sharpening of a sharp photo", so the "soft" condition remains the main protection.
  - **Full run:** 6 photos (3 motion, 2 lens, 1 sharp): the sharp one was not touched, 3 were deblurred, 2 had their deblurring rejected by the guard and so continued with upscaling alone, and all six were delivered. Example: a motion photo 31.1 → **35.4 dB** against the original.
  - The model is installed in `tools/free-models/manifest.json` as an optional model (`nafnet_gopro`, MIT, SHA-256 hash in `lock.json`).

### 3.7 Cartoon samples study (`Desktop/Samples`, 2026-11-16)
You asked for Sample01–05 to be upscaled to exactly 2x, and for a study of b001 (the original) vs a001 (your Video2X output). The outputs are in `D:/VidIMG/samples_2x`.

**What was found:**
- **All the samples are cartoons,** but the automatic analysis did not recognize them (anime score 0.43–0.56), and the face detector saw "faces" in the drawings. The hand-crafted metrics (flatness, colors, edge sharpness) **do not separate** old compressed cartoons from your real videos (the real MBZ video scored 0.62).
  → Added the option `--content anime|real|auto`; the analysis now outputs a **frame sheet** that Claude looks at to decide the content type, and the automatic mode stays conservative.
- **a001 = Real-ESRGAN anime-v3 ×2** (a 41.8 dB match vs ≤ 34.8 for the other models). But it **dropped one of every five real frames** (29.97 → 23.976); b001's frames are unique, not blended, so the motion in a001 is less smooth.
- **Ground truth for cartoons** (12 clips, ×½ downscale with compression, then ×2 restoration, `workspace/samples_study/anime_gt`):

| Model | VMAF | PSNR | Visual note |
|---|---|---|---|
| general-v3 | **82.4** | 32.1 | Sharp lines and more background texture |
| Real-CUGAN pro conservative | 80.4 | 32.0 | Blotches in the background |
| RealESRNet | 79.5 | 33.7 | Softer |
| Real-ESRGAN anime-v3 | 79.3 | 32.7 | **Clean lines** (a001's model) |
| Real-CUGAN se | 78.2 | **34.4** | Clean lines |

  → Anime path: anime-v3, then CUGAN se, then general-v3. The face gates are **advisory** on drawn faces, because the identity model is trained on real faces.

**Results (exactly 2x upscaling, `--content anime`, outputs in `D:/VidIMG/samples_2x`):**

| Sample | Original → Output | Frames | Model | VMAF | Notes |
|---|---|---|---|---|---|
| Sample01 | 640×480 → 1280×960 | 38720 matching | anime-v3 | 94.4 | |
| Sample02 | 1920×1080 → 3840×2160 | 33322 matching | anime-v3 | 96.4 | 4K; about two hours and 15 minutes |
| Sample03 | 640×480 → 1280×960 (MKV) | 1946 matching | anime-v3 | 98.0 | AVI and PCM; FFV1 intermediate for the timestamps |
| Sample04 | 512×384 → 1024×768 | 12120 matching | anime-v3 | 98.4 | Video-to-audio offset (0.2 s) preserved |
| Sample05 | 720×480 → 1440×960 (MKV) | 34468 matching | anime-v3 | 97.7 | Subtitles preserved |
| b001 | 720×576 → 1440×1152 | **43259** (a001: 34607) | anime-v3 | 96.6 | Same model as in a001, but with all the frames |
| Sample06 | 720×544 → 1440×1088 (MKV) | 30571 matching | anime-v3 | 97.1 | VHS, non-square pixels 11696:11565 restored (see 3.8); 26 minutes |
| Sample07 | 768×576 → 1536×1152 | 33191 matching | anime-v3 | 96.3 | Video starts 0.08 s after the audio, preserved; 25 minutes |

Note on a001: it also lost the 16:15 pixel aspect ratio (SAR unset), so it displays as 5:4 instead of 4:3, i.e. the picture is squeezed horizontally by 6.7%.

**Faults fixed by this study:**
1. **Video2X 6.4 drops the last two frames on every run** → they are rebuilt by duplicating the last frame and concatenating without re-encoding.
2. **Video2X shifts the video timing relative to the audio** (0.08 and 0.91 seconds), **and the free engine lost Sample04's offset** (0.2 seconds) → the function `media.fix_av_offset` after every engine, mux and pre-pass, and a new guard gate `av_sync`.
3. **Video2X wrote every timestamp as zero** with MPEG-4 Part 2 (Sample03.avi), producing a file that does not play correctly, and a misleading VMAF of "25" → encoding without B-frames, restamping the timestamps, a `timestamps` gate, and VMAF bounded by frame count rather than by time.
4. **The estimated frame count** from the duration was wrong for a clip starting in the middle of the file (25120 instead of 120) → exact packet counting (`probe(exact=True)`).
5. **ASS subtitles break the MP4 container** and were being lost → an MKV container automatically when there are subtitles, attachments (fonts) or PCM audio, with the subtitles copied from the original.
6. Anime4K (libplacebo) needs the output dimensions; Video2X now encodes with hevc_nvenc on the GPU; `--scale 2` was added for exactly 2x; and the anime path is allowed above 720p when requested.
7. **The flicker metric penalized grain removal:** a grainy night-sky shot in Sample05 "failed" at 0.031, even though the output changes less between frames than the source. The output is now normalized by the larger detail of the two (output or source), so the value dropped to −0.008. Re-checking all the labeled outputs (132 real videos and 108 cartoons) changed no decision, and all 36 boiling outputs are still caught.
8. **Trimming the previews copied the subtitles,** so the clip duration became 1118 seconds instead of 4 (and general-v3 was failed unfairly). → Trimming takes only the video and audio, and the duration is computed from the frame count if another stream inflated it.
9. **The job queue was not in order:** Sample02 (4K) jumped ahead of the b001 job. → The queue is now oldest first.
10. **Video2X timestamps and B-frames:** restamping a clip that has B-frames in packet order **scrambles the presentation order**, which is what hit the first Sample03 output (distinct timestamps, but not ascending in presentation, and VMAF did not notice). Disabling B-frames is not possible: `--max-b-frames 0` with NVENC in Video2X 6.4 fails (invalid param). → Sources with unreliable timestamps (MPEG-4 Part 2, MS-MPEG4, H.263, WMV and AVI) are first converted to a lossless FFV1 intermediate, so Video2X's timestamps come out correct. Restamping happens only without B-frames, and **the `timestamps` gate now checks the presentation order.**
11. **The tail in MP4:** the two-frame suffix was encoded with a 1/30000 timebase and the file with 1/16000, so concatenation multiplied its timestamps by 1.875 (b001's tail at 2706 seconds). → The file's own timebase, a duration correction in the concat list, the frame rate taken from the source, and a check of the last frame's position before acceptance. The guard had already caught it (duration gate), so the file was not delivered.
12. **The protection hook** blocked merely listing the Video2X models folder → it now only blocks actually running Video2X with an anime processor without `anime`.

### 3.8 Samples 06 and 07, pixel aspect ratio, and first-time installation (2026-11-17)

**Result:** both samples passed every gate with the current settings, with no change to the anime path. But the study uncovered a geometric-fidelity flaw that no gate was checking:

13. **The pixel aspect ratio (SAR) gets lost.** The free engine reads frames from a raw pipe that carries no pixel aspect ratio, so it turned PAL 720×576 at 16:15 (displayed as 4:3) into 1440×1152 at 1:1 (displayed as 5:4): **every face 6.7% narrower**, and the guard did not notice because it compares the frames after scaling them. Video2X 6.4 also dropped Sample06's unusual ratio 11696:11565 (a −1.1% difference), although it preserved 16:15 and 8:9. → Three fixes:
    - `setsar` in the free engine's encoder (the root fix).
    - `media.fix_aspect`: a remux without re-encoding that sets the container tag and the VUI in H.264/HEVC, after every engine, mux and pre-pass, and at the end against the original.
    - **A new `aspect` gate:** rejects any change in the display aspect ratio greater than 0.5%.

    All of this was tested on b001 (16:15) and Sample06, and the delivered Sample06 output was fixed by remuxing and then re-checked: it passed.
14. **The protection hook blocked an analysis command** (`ffmpeg -i x.avi -f null -`), mistaking the input for the output. → It now excludes everything following `-i` from the list of outputs.
15. **Automatic anime detection failed again:** Sample07 scored 0.23, and a real "face" (a drawn one) was found in it. → The decision stays with the frame sheet, or with the user.

**First-time installation:** a new user would have faced three obstacles:
- There is no installer for Video2X or for the face models, and both are outside the distribution (`tools/` is in .gitignore).
- `.mcp.json` points to a fixed path on the owner's machine.
- The results folder `D:/VidIMG` may not exist on another machine.

→ Added `python -m vidimg.setup` (and `/setup-vidimg` in Claude Code), a single command that checks and installs everything and can be rerun:
- **Core requirements:** the operating system, FFmpeg (from winget, after asking), and Python packages.
- **Downloaded with pinned SHA-256 hashes:** the face models, PyTorch with the models, Video2X, and VapourSynth.
- **Connecting Claude Code:** this machine's paths and the results folder, while keeping the hook and the permissions.

A results folder on a drive that does not exist falls back to `workspace/`, and `doctor` shows the results folder and its free space.

**Measured on a clean copy** of the project (662 KB without `tools/`):
- The full installation took **6 minutes** and ended with READY: 8 models, PyTorch 2.14.1+cu130 seeing the card, Video2X, and VapourSynth.
- Then a photo restoration and a PAL video restoration through the real path both succeeded on it (RealESRNet, 1350×1080 at 16:15, every gate including `aspect`).

User guide: [GUIDE.md](GUIDE.md).

---

## 4. Architecture

```
┌──────────────────────────── Claude Code ────────────────────────────────────────────┐
│ Skill: restore-media · Commands: /restore-photo /restore-video                      │
│        /analyze-media /restore-jobs · Protection hook (PreToolUse)                  │
└────────────────┬────────────────────────────────────────────────────────────────────┘
                 │ MCP (stdio) · background jobs with a GPU lock and resume
┌────────────────▼────────────── vidimg (Python 3.14) ────────────────────────────────┐
│ probe → analyze → plan → (classical pre-pass) → previews → run → guard              │
│                                                                                     │
│  Photos: SCUNet (noise) → BSRNet/RealESRNet/SwinIR   [tools/sr-venv]                │
│  Video:  QTGMC/TIVTC [AviSynth+ | VapourSynth | ffmpeg]                             │
│          → RealESRNet/BSRNet or general-v3 (resumable chunks)                       │
│  Anime:  Video2X (Real-CUGAN / Real-ESRGAN anime)                                   │
│  Guard:  YuNet+SFace · structure · color · hallucination · flicker · VMAF (ffmpeg)  │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

| Module | Role |
|---|---|
| `vidimg/adapters/free_sr.py` + `sr_runner.py` | Open models for photos and video (tiles, batches, exact fractional frame rate) |
| `vidimg/adapters/video2x.py` | Anime and interpolation |
| `vidimg/adapters/avisynth.py` / `vapoursynth.py` | Classical field processing |
| `vidimg/media.py` | ffmpeg/ffprobe, encoders, and color tags |
| `vidimg/pipeline.py` / `pipeline_video.py` / `batch.py` | Decision engine, batches and chunks |
| `vidimg/guard/` | The guard |
| `vidimg/proc.py` | Running programs with an evidence log, and refusing any paid program |

---

## 5. Decision engine

### 5.1 Photos
```
1) analyze → profile (noise, sharpness, JPEG quality, faces and their sizes, anime detection)
2) Anime, no real face     → Video2X: Real-CUGAN, then Real-ESRGAN anime
3) Noise                   → SCUNet first
4) Already large (> 2000)  → no upscaling
5) Upscaling (×4 if ≤ 900, otherwise ×2): BSRNet, then RealESRNet, then SwinIR, with exact resizing to the requested size
6) Guard: deliver the first clean output, otherwise the first output with a warning, otherwise keep the original
```

### 5.2 Video
```
1) analyze (idet with repeated fields, compression, noise, scene cuts, faces)
2) Telecine  → TFM+TDecimate · interlaced → QTGMC at double rate · guard for the pre-pass
3) Anime     → Video2X ×2
4) Candidates: RealESRNet → BSRNet → general-v3 (or general-v3 alone if the estimate exceeds 3 hours)
5) Previews with the guard on 3 clips, then a full run (chunks if over 5 minutes), then a final guard
6) Output HEVC 10-bit (NVENC, or x265 without NVIDIA) with audio and metadata
```

---

## 6. Claude Code integration

- **MCP server** (`vidimg/mcp_server.py`, ten tools):
  - Analysis and running: `doctor` · `analyze_media` · `restore_photos` · `restore_video`
  - Job management: `job_status` · `list_jobs` · `cancel_job` · `resume_job`
  - Verification and comparison: `verify_restoration` · `compare_frames`
- **`restore-media` skill** and four commands.
- **Protection hook:**
  - **Blocks** ffmpeg from writing over its own input.
  - **Blocks** anime models on non-anime content.
  - **Asks** before deleting, moving, or writing a photo or video outside the workspace.

---

## 7. Roadmap

| Stage | Content | Status |
|---|---|---|
| 0. Setup | `doctor` and pinned requirements (`free_install`, `vs_install`, `avs_install`) | ✅ |
| 1. Adapters | `free_sr` (photos and video), `video2x`, `avisynth` / `vapoursynth`, and `media` | ✅ |
| 2. Analysis and guard | `probe/`, `guard/`, and calibration on your photos and clips | ✅ |
| 3. Decision engine | `pipeline.py`, `pipeline_video.py`, classical field processing, and anime routing | ✅ |
| 4. Claude Code | MCP, skill, commands, hook, and background jobs | ✅ |
| 5. Batches and performance | Photo grouping, resumable video chunks, and a GPU queue | ✅ |
| 6. Acceptance | Your verdict on 19 photos: "all faces intact" · WhatsApp video with RealESRNet: "faces intact" · a cartoon clip delivered via Video2X | ✅ |
| Video recalibration | Video gates on the free models: VMAF stays at 4.0, and the guard 31/33 faithful and 9/10 harmful (see 3.4) | ✅ |
| Deblurring (optional) | NAFNet-GoPro on "soft" photos only, with a special guard mode (see 3.6) | ✅ |
| Proposed | Linux support (only the background jobs are Windows-specific) | ⏳ |

---

## 8. Hidden behaviors and their fixes (free tools)

| # | Tool | Behavior | Fix |
|---|---|---|---|
| F1 | AviSynth+ | `AddAutoloadDir` in 3.7.5 without `utf8=` · QTGMC outputs a 19001/317 rate · L-SMASH writes `.lwi` next to the source | Explicit `AssumeFPS` · `cachefile=` inside the workspace |
| F2 | ffmpeg | Matroska with millisecond timestamps breaks exact frame rates, comparing frames by timestamp goes wrong, and 10-bit to 8-bit adds dither | FFV1 intermediate in NUT · metrics by order and at 10-bit depth |
| F3 | idet | Labels telecine as "TFF" | Telecine = repeated-field ratio ≥ 0.2 |
| F4 | NUT | Does not keep color tags | `.colour.json` sidecar file applied with `setparams` |
| F5 | Full range | A `yuvj` source compresses the levels in the output | Full-to-limited conversion in the pre-pass (lossless at 10-bit) |
| F6 | Guard | VMAF pairs frames by timestamp | Pairing by order |
| F7 | VapourSynth | R73+ rejects API 3 plugins, and vsrepo needs R80+ | Install R82, then pin the core to R72 in an isolated environment |
| F8 | ffmpeg 8 | Color tags on the output with a Y4M input insert a conversion | No tags in intermediate commands |
| F9 | MCP stdio | Subprocesses inherit the protocol channel and hang | Handle isolation at server start |
| F10 | ffmpeg | Trimming with stream copy leaves an edit list (665 vs 601 frames) | Trimming by decoding (`-ss`/`-t`) |
| F11 | Heavy models | A full HD frame fills the memory, so Windows falls back to system memory | 384-pixel tiles and one frame at a time |
| F12 | Frame rate | A rounded decimal number turned 60000/1001 into 599401/10000 | Exact fractional frame rate |
| F13 | Guard | Comparing against the damaged input rewards copying its defects | Decisions on the true original (`bench_free.py`) |
| F14 | ffmpeg (decoding for the model) | Raw output at a constant frame rate (the default) duplicates the first frame if there is a gap in the source timestamps (121 → 122), so every following frame shifts by one and the guard sees "changed faces" in moving scenes | `-fps_mode passthrough` in decoding (`free_sr.run`) |
| F15 | Video2X 6.4.0 | Drops the last two frames, shifts the video timing relative to the audio, and writes zero timestamps with MPEG-4 Part 2 | `video2x._restore_tail`, `_fix_timestamps` and `media.fix_av_offset` |
| F16 | Source | Video that starts after the audio (Sample04: 0.2 s) loses the offset if encoding starts from zero | `fix_av_offset` and the `av_sync` gate |
| F17 | MP4 | Does not accept ASS subtitles (the whole write fails) | MKV automatically and `carry_side_streams` |
| F18 | AVI/MPEG-4 Part 2 | No presentation timestamps, and the frame count in the header is one more than what decodes | Packet counting, restamping, and VMAF bounded by frame count |
| F19 | Analysis | Old compressed cartoons measure like real video, and YuNet sees faces in drawings | `--content` and the frame sheet |
| F20 | Video2X 6.4.0 | Its NVENC does not work without B-frames (`--max-b-frames 0` fails, and `-e bf=0` is ignored); and the tail's different timebase in MP4 inflates its timestamps on concatenation | FFV1 intermediate for sources without timestamps, `-video_track_timescale`, and a presentation-order check |

---

## 9. Evidence log

| # | Evidence | Location |
|---|---|---|
| D1 | Photos against the true original | `vidimg/bench_free.py` · `workspace/calibration/20261114-235009/summary_free.json` |
| D2 | Video against the true original, and speed | `vidimg/bench_free_video.py` · `workspace/bench_free/` |
| D3 | AviSynth+ study and the engine decision | `research/avisynth/01–05` · `research/avisynth/vs_parity.py` |
| D4 | Guard calibration | `workspace/state/guard_thresholds.json` · `workspace/calibration/` · `workspace/calibration_video/` (free: `20261116-153325`) |
| D5 | Human acceptance | `workspace/acceptance/index.html` · `workspace/acceptance/human_verdict.json` |
| D6 | Resumption and long video | `workspace/review_video/mbz_full/` · `workspace/jobs/` |
| D7 | Pinned requirements | `tools/free-models/lock.json` · `tools/vapoursynth/lock.json` · `tools/avisynth/lock.json` |
| D9 | Deblurring measurement | `workspace/bench_deblur/summary.json` · `tools/free-models-trial/SOURCES.json` |
| D8 | Removed Topaz history | `archive/topaz/` (the code and the previous plan) |
