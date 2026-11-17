# VidIMG: requirements (free tools only)

VidIMG uses **only free software and open model weights**. No paid program, licence or subscription is
used or needed, and the tool refuses to start any program from a paid software installation.
Verified on the reference machine on 2026-11-16.

| Job | Free tool |
|---|---|
| Photo upscaling | BSRNet → RealESRNet → SwinIR (non-GAN open models) on PyTorch CUDA |
| Photo denoising | SCUNet (real-photo model) |
| Video enhancement | RealESRNet / BSRNet when the run fits (~3 h estimate), otherwise Real-ESRGAN general-v3 |
| Anime / cartoon | Video2X (Real-CUGAN, Real-ESRGAN anime-v3) |
| Deinterlace / IVTC | QTGMC, TFM+TDecimate (AviSynth+ or VapourSynth); ffmpeg bwdif / fieldmatch as fallback |
| Frame interpolation (on request) | RIFE via Video2X |
| Fidelity guard | OpenCV YuNet + SFace face models, FFmpeg libvmaf, scikit-image |

---

## 1. Hardware

| Item | Minimum | Recommended (reference machine) | Why |
|---|---|---|---|
| OS | Windows 10/11 x64 | Windows 11 Pro 24H2+ | Background jobs use Windows APIs today |
| GPU | NVIDIA, 6 GB VRAM, CUDA 13 driver | RTX 5060 Ti 16 GB (driver 621.24) | PyTorch CUDA models; NVENC HEVC 10-bit encoding. Without NVIDIA: CPU fallback works but is very slow; encoding falls back to x265 |
| CPU | 4 cores | i9-12900K (24 threads) | ffmpeg decode/encode, QTGMC, guard |
| RAM | 16 GB | 64 GB | Video pipelines and 16-bit photos |
| Disk (software) | ~5 GB | — | sr-venv 3.3 GB, models 0.4 GB, Video2X 0.5 GB, vs-venv 0.2 GB, AviSynth plugins 0.1 GB |
| Disk (working space) | 20 GB free | 50 GB+ free (owner: D:/VidIMG via `VIDIMG_OUTPUT`) | An hour of 1080p HEVC is ~3 GB of chunks plus ~3 GB joined; 16-bit PNG photos ~40 MB each per candidate |

## 2. Software (all free)

| Component | Version verified | Licence | Install | Used for |
|---|---|---|---|---|
| Python | 3.14.5 (64-bit) | PSF | python.org | the tool |
| Python packages | `requirements.txt` | BSD / MIT / Apache | `pip install -r requirements.txt` | main environment |
| FFmpeg (full build) | 8.0.1 (gyan.dev) | GPL | `winget install Gyan.FFmpeg` | decode/encode, analysis filters, **libvmaf**, **AviSynth demuxer**, NVENC |
| NVIDIA driver | 621.24 (CUDA 13 capable) | proprietary, free of charge | nvidia.com | GPU for PyTorch and NVENC |
| Open-model runtime (tools/sr-venv) | PyTorch 2.14.1+cu130, spandrel 0.4.2 | BSD / MIT | `python -m vidimg.free_install` | runs the open models |
| Open models (tools/free-models) | 7 files, 394 MB (+ optional NAFNet 259 MB), SHA-256 locked | BSD-3 / Apache-2.0 | `python -m vidimg.free_install` | see section 3 |
| Face models (tools/face-models) | YuNet 2023mar, SFace 2021dec | Apache-2.0 (OpenCV Zoo) | `python -m vidimg.setup` (SHA-256 pinned) | fidelity guard: face identity |
| Classical engine: one of | | | | deinterlace (QTGMC), IVTC (TFM/TDecimate) |
| — VapourSynth (portable) | R72 pinned in tools/vs-venv | LGPL | `python -m vidimg.vs_install` | nothing installed system-wide |
| — AviSynth+ (optional) | 3.7.5 + pinned plugin set | GPL | GitHub installer, then `python -m vidimg.avs_install` (needs 7-Zip) | primary when present |
| — neither | | | | ffmpeg bwdif / fieldmatch fallback |
| Video2X | 6.4.0 | AGPL-3.0 | `python -m vidimg.setup` (GitHub release, SHA-256 pinned) | anime content, RIFE |
| 7-Zip | 24+ | LGPL | `winget install 7zip.7zip` | only for the AviSynth plugin archives |

Optional: Claude Code (needs a Claude plan) for `/restore-photo`, `/restore-video`, the MCP server and the
safety hook. Without it, every feature runs from the shell (`python -m vidimg ...`).

## 3. Open models (official author releases only; SHA-256 in tools/free-models/lock.json)

| Key | Model | Use | Training | Licence |
|---|---|---|---|---|
| bsrnet_x4 | BSRNet (BSRGAN project) | photos, first choice | non-GAN (PSNR) | Apache-2.0 |
| realesrnet_x4 | RealESRNet x4plus | photos second; video first when the run fits | non-GAN (PSNR) | BSD-3-Clause |
| swinir_real_x4 | SwinIR-M real-world x4 PSNR | photos, third | non-GAN (PSNR) | Apache-2.0 |
| general_x4v3 | Real-ESRGAN general v3 (compact) | video, fast (long videos) | GAN, compact | BSD-3-Clause |
| scunet_real | SCUNet real-photo | denoise before upscaling | non-GAN (PSNR) | Apache-2.0 |
| nafnet_gopro (optional) | NAFNet GoPro width64 | opt-in photo deblur (`--deblur`) | non-GAN (PSNR) | MIT (weights from the authors' README Google Drive link) |
| realesrgan_x4 / _x2 | Real-ESRGAN x4plus / x2plus | calibration reference only | GAN | BSD-3-Clause |

Not used, by policy: generative face restorers (GFPGAN, CodeFormer: they change identity; CodeFormer is
non-commercial) and diffusion upscalers.

**Measured on the owner's material** (against the real originals):
- Photos (12): BSRNet true identity 0.811 and +1.58 dB over bicubic; RealESRNet 0.798; SwinIR 0.782.
  The owner reviewed 19 delivered photos: "all faces look right" (2026-11-16).
- Video (12 segments, re-measured after the frame-offset fix): RealESRNet VMAF 90.1 / identity 0.889; BSRNet 87.8 / 0.889; general-v3 93.5 / 0.868.
- Speed at 720p → 1080p (RTX 5060 Ti): general-v3 7.8 fps (an hour of 720p in ~3.8 h); RealESRNet 0.47 fps.

## 4. Install from nothing

Prerequisites: 64-bit Windows 10/11, Python 3.12+ on PATH, an NVIDIA driver. Then, in the project folder:

```bash
python vidimg_run.py setup --output D:/VidIMG
```

It installs or verifies, in order: the main Python packages, FFmpeg (winget `Gyan.FFmpeg`, only after asking),
the OpenCV face models, PyTorch CUDA + the open models (tools/sr-venv, tools/free-models), Video2X 6.4.0, the
portable VapourSynth engine, and finally points `.mcp.json` / `.claude/settings.json` at this machine's Python,
project folder and output folder (other settings kept). Every download is an official release checked against a
pinned SHA-256. `--check` verifies without changing anything; `--yes` asks nothing. Re-running repairs.
Measured on a fresh copy of the project (no tools/): see PLAN.md 3.8. In Claude Code, `/setup-vidimg` does the same
with explanations. Arabic step-by-step guide: [GUIDE.ar.md](GUIDE.ar.md).

Only `tools/*/manifest.json` and `tools/*/lock.json` are distributed; everything else in `tools/` is downloaded.

Optional AviSynth+ engine: install AviSynth+ from its GitHub release and 7-Zip
(`winget install 7zip.7zip`), then run `python -m vidimg.avs_install`.

## 5. Known limits

- Non-GAN models are faithful but look **softer** than tools that invent texture.
- Video with small faces: the fast general-v3 model sometimes reshapes them; the guard then keeps the
  original. The faithful models run only when their estimated run is ≤ 3 hours (frames × pixels;
  interlaced counts double).
- Deblurring is opt-in (`restore --deblur`, MCP `deblur=True`) and runs only on photos the analysis flags as soft: NAFNet-GoPro, judged by the guard's deblur mode (PLAN.md 3.6). It cannot tell motion from defocus blur; on defocus the gain is small. Photos only, not video.
- Windows-only today: background jobs (`vidimg/jobs.py`) use Windows APIs.
