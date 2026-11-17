# VidIMG User Guide: First-Time Installation and Correct Use

VidIMG is a tool that improves your personal photos and videos **faithfully**. It raises resolution and removes noise, compression artifacts and interlacing,
but it does not change faces or colors. It uses **free tools only** and runs only on your own computer, so your photos are never sent to any outside service.
A calibrated "guard" checks every result before it is delivered. Anything the guard rejects is not delivered, and the original stays as it is.

---

## 1. Requirements

### What you need before you start

| Item | Minimum | Recommended | Note |
|---|---|---|---|
| Operating system | Windows 10/11, 64-bit | Windows 11 | Background jobs currently rely on Windows interfaces |
| Graphics card | NVIDIA with 6 GB of memory and a driver that supports CUDA 13 | RTX with 12–16 GB of memory | Without one, the tool runs on the processor, but very slowly |
| Memory (RAM) | 16 GB | 32–64 GB | |
| Space for programs | 10 GB | — | About 5 GB of downloads, installed inside the tool's own folder |
| Space for results | 20 GB | 100 GB or more on a large drive | One hour of 1080p video needs about 6 GB while it is being processed |
| Python | 3.12 or newer (64-bit) | 3.14 | From python.org, with the **Add python.exe to PATH** option turned on |
| NVIDIA driver | Recent | — | From nvidia.com |
| Claude Code | Desktop app or command line | — | Needs a Claude subscription. Without it, the full tool still works from the command line |
| Internet | For the first installation only | — | After installation, the tool works offline |

### What the setup command installs for you

Everything that is downloaded is saved in **the VidIMG folder**. This is `%LOCALAPPDATA%\VidIMG` when you install VidIMG as a plugin, or the
project folder itself when you clone it. So nothing is lost when the plugin is updated, and deleting this folder removes everything.

Every component here is free. Each one is downloaded from its official source and checked against a SHA-256 fingerprint fixed in the code. Nothing is installed
system-wide except FFmpeg, and only after you agree.

| Component | Size | Source | Location | Purpose |
|---|---|---|---|---|
| Core Python packages | About 300 MB | PyPI (`requirements.txt`) | Your Python environment | Run the tool and the guard |
| PyTorch CUDA and spandrel | About 3.3 GB | download.pytorch.org | `tools/sr-venv` (isolated environment) | Run the open models on the graphics card |
| Open models (BSRNet, RealESRNet, SwinIR, SCUNet, general-v3, NAFNet) | About 650 MB | Their authors' official releases | `tools/free-models` | Enhancement and noise removal |
| YuNet and SFace face models | 38 MB | OpenCV Zoo | `tools/face-models` | Used by the guard to check that faces keep their identity |
| Video2X 6.4.0 | 188 MB | GitHub (k4yt3x) | `tools/video2x-6.4.0` | Cartoons and animation, and frame generation on request |
| Portable VapourSynth R72 with its plugins | About 220 MB | PyPI and vsrepo | `tools/vs-venv` | Deinterlacing (QTGMC) and inverse telecine (IVTC) |
| FFmpeg (full build) | About 200 MB | `winget install Gyan.FFmpeg` | The system, **after asking you** | Encoding, analysis and VMAF |
| Your local settings | — | — | `vidimg.local.json` in the VidIMG folder (and `.mcp.json` for the cloned copy) | Saves the results folder you chose and the Python path on your computer |

**What is not installed automatically (optional):**
- **AviSynth+:** A second classic engine. The tool works without it, because VapourSynth is enough. If you want it: install AviSynth+ from
  its releases page on GitHub, and 7-Zip (`winget install 7zip.7zip`), then run `python -m vidimg.avs_install`.
- **Microsoft Visual C++ Runtime:** Video2X needs it. Most computers already have it. If yours does not, the command tells you and gives you the install line:
  `winget install Microsoft.VCRedist.2015+.x64`.

---

## 2. First-time installation, step by step

### Step 1: Install Python if you do not have it

Download it from python.org and turn on the **Add python.exe to PATH** option, or install it with this command:

```bash
winget install Python.Python.3.14
```

Then open a new terminal window and check the version (3.12 or newer):

```bash
python --version
```

### Step 2: Add VidIMG to Claude Code as a plugin

In Claude Code, in any folder, type these two commands:

```
/plugin marketplace add Q8Xbox/VidIMG-skill
```

```
/plugin install vidimg@vidimg
```

The first command adds the GitHub repository as a plugin source. The second installs the plugin. Then restart Claude Code
(`/exit`, then `claude`). You now have:
- **The skill** `restore-media`, which Claude uses automatically when you ask it to restore a photo or video.
- **The commands** `/vidimg:setup-vidimg`, `/vidimg:restore-video`, `/vidimg:restore-photo`, `/vidimg:analyze-media`
  and `/vidimg:restore-jobs`.
- **An MCP server** named `vidimg`, which holds the analysis, restoration and guard tools.
- **The safety hook**, which stops anything from writing over your originals and asks you before any photo or video outside the work folder is deleted.

To update the plugin later: `/plugin marketplace update vidimg`. To remove it: `/plugin uninstall vidimg@vidimg`,
then delete `%LOCALAPPDATA%\VidIMG` if you also want to delete the downloaded models.

> **Alternative for developers:** Clone the repository (`git clone https://github.com/Q8Xbox/VidIMG-skill.git`) and open its folder in
> Claude Code. The same skill and commands work without the `vidimg:` prefix, and everything is saved inside the project folder.
> When you open it, approve the `vidimg` MCP server and the project settings.

### Step 3: Run the setup command

**Option 1: From inside Claude Code (easiest)**

```
/vidimg:setup-vidimg D:/VidIMG
```

Claude will check your computer, explain what will be downloaded (about 5 GB), and ask you for a results folder. Then it installs everything in the background
and tells you the result of each step. You can also just ask in your own words: "Install the VidIMG tool on my computer".

**Option 2: Directly from the terminal**

Inside the cloned project folder:

```bash
python vidimg_run.py setup --output D:/VidIMG
```

(For the installed plugin, the `vidimg_run.py` file is in the plugin's folder inside `~/.claude/plugins/cache`. Option 1 is
easier.)

- `--output`: The folder where results, jobs and logs are saved. Choose a drive with plenty of space. If you do not set it,
  the command asks you for it. If you leave it empty, it uses `workspace` inside the VidIMG folder.
- The command asks you before it installs FFmpeg through winget. Add `--yes` if you want it to install without asking anything.
- Installation takes about 6 minutes on a fast connection (measured on a clean copy of the project), and can take up to 30 minutes on a slow connection.
- You can run it again at any time: it only checks the parts that are fine and completes the parts that are missing. So it also repairs the installation.
- To only check, without installing or changing anything: add `--check`.

At the end, it prints the status of each step, followed by one of these two words:
- **`READY`**: Everything is ready.
- **`NOT READY`**: Fix the line marked `FAIL` as it suggests, then run the command again.

### Step 4: Restart Claude Code

After `READY`, restart Claude Code (`/exit`, then `claude`) so the `vidimg` tools load with the packages that were installed.
If the tools do not appear, type `/mcp` and make sure the `vidimg` server is connected.

### Step 5: Check that everything works

Type in Claude Code: "Check whether the VidIMG tool is ready". The check runs `doctor()`. Or run it from the terminal:

```bash
python vidimg_run.py doctor
```

You should see `healthy: True`, with your graphics card's name next to `ai_engine` and `video2x`.

### Step 6 (optional): Calibrate the guard on your own photos

The guard comes already calibrated on real photos, with carefully tested values. If you want to tune it to your own family's faces, collect at least 20 photos
with clear faces and run:

```bash
python -m vidimg calibrate "D:/MyPhotos/calibration" --apply
```

---

## 3. Correct use in detail

### 3.1 The main way: ask in your own words inside Claude Code

You do not need to memorize any commands. Just type what you want in plain words, for example:

- "Improve the photos in this folder: D:\Family Photos\1995"
- "Make this video twice as big: D:\Videos\wedding.avi"
- "This video is an old cartoon, improve it 2x: D:\Cartoons\episode1.mp4"
- "Analyze this video and tell me what its problems are before you start"
- "Does this enhanced copy still look like the original? The original is this and the result is that"

Claude follows the approved workflow in the `restore-media` skill and reports back to you in English.

### 3.2 Short commands

When VidIMG is installed as a plugin, the commands start with the `vidimg:` prefix, for example `/vidimg:restore-video`. In the cloned copy, you type them without the prefix.

| Command | What it does |
|---|---|
| `/setup-vidimg [results folder]` | Installs or repairs the tool |
| `/analyze-media <file>` | Analyzes the file and shows the plan without writing anything |
| `/restore-photo <files or folders>` | Improves the photos and makes a review page |
| `/restore-video <video> [preview]` | Improves the video: checked previews, then full processing, then a report |
| `/restore-jobs [job number]` | Shows the status of running jobs and their results |

### 3.3 The correct workflow for video

1. **Analysis first.** `analyze_media` measures noise, compression, interlacing, and faces and their sizes. It also makes a **frame sheet**,
   which is one image showing 6 shots from the video.
2. **Decide the content type from the frame sheet.** Is it **animation/cartoon** or **real footage**? This is the most important decision in
   the whole process:
   - **Cartoon** (`content='anime'`): Video2X is used with the Real-ESRGAN anime-v3 model first, then Real-CUGAN, then general-v3.
   - **Real footage** (`content='real'`): The faithful models RealESRNet and BSRNet are used, or the fast general-v3 for long videos.
   - **Why not leave this decision to automatic detection?** Because automatic detection makes mistakes on old, compressed cartoons and takes drawn faces
     for real faces. And anime models distort real faces, so the safety hook blocks them on any content that has not been declared a cartoon.
3. **Upscale size.** If you do not set a size, videos up to 540p are doubled, and videos above that, up to 800p, are raised to 1080p.
   For a specific size, ask for it clearly, for example "2x" or `--scale 2`.
4. **Previews.** Three short clips spread across the video are processed with each candidate, and the guard checks them. Full processing
   starts only with the first candidate whose previews all pass. For a long video, you can ask for the preview alone first (`preview_only` or `--preview-only`).
5. **Full processing.** It runs in the background, and you can close the conversation while it runs. Jobs wait their turn on the graphics card in the order
   they arrived, and a job that stopped resumes from where it stopped.
6. **Final check.** The whole result is checked: frame count, duration, audio, timestamps, sync, displayed
   dimensions, colors, flicker, and VMAF.
7. **Report.** `report.html` shows before and after frames. **Look at the faces yourself**: your judgment comes before the numbers.

**What the tool keeps automatically:**
- The audio, unchanged.
- Attached subtitles and fonts, for which it chooses an MKV container.
- PCM audio, also in MKV.
- The start offset between audio and picture, so lip sync is not thrown off.
- The pixel aspect ratio, so a 4:3 PAL video does not turn into 5:4.
- All frames, so none are dropped.

### 3.4 The correct workflow for photos

1. Give it files or a whole folder. Folders are processed in batches.
2. For each photo: noise is removed with SCUNet if the photo is noisy, then it is upscaled with the first model the guard accepts, in this order:
   BSRNet, then RealESRNet, then SwinIR.
3. Deblurring is **optional**: ask for it clearly by saying "remove the blur", or use `--deblur`. It only works on photos
   that the analysis classifies as blurry.
4. The results are 16-bit PNG images, along with a review page that compares each photo with its original.

### 3.5 Following your jobs

- "What is the job's status?" is answered by `job_status`, and "Show all jobs" is answered by `list_jobs`.
- "Stop the job" is done by `cancel_job`, and "Continue the stopped job" is done by `resume_job`.
- Results are in the results folder you chose during installation, for example `D:\VidIMG`, and logs are in `D:\VidIMG\logs`.

### 3.6 Reading the guard's result

| Gate | What it means |
|---|---|
| identity | Is the face the same person? (face fingerprint similarity using the SFace model) |
| backprojection | If we shrink the result to the original's size, does it match? (no new content was invented) |
| landmarks, eye_color | Facial features and eye color did not change |
| color | Color change does not exceed ΔE = 4 |
| hallucination | A warning when finer detail appears than the original can support |
| frames, duration, audio | Frame count, duration and audio match the original |
| timestamps, av_sync | Frame order is correct, and sync between audio and picture is kept |
| aspect | The picture's shape on screen did not change |
| flicker | No flicker was added between frames |
| vmaf | Perceived video quality compared with the original |

- **`warn` (warning):** The result is delivered with a note.
- **`fail` (failure):** The result is never delivered, and the original stays as it is.

### 3.7 Golden rules

1. **Do not put results in the originals folder.** The tool refuses to do that anyway, and it never writes over any original.
2. **Set the content type correctly:** cartoon or real footage. Never use anime models on real faces.
3. **Generate extra frames (RIFE) only on request.** It changes the frame count, and it is not an "improvement".
4. **Do not expect invented details.** Faithful models give a softer image than "generative AI" tools, but they do not change
   people's features. This is intentional.
5. **Always keep your originals.** The result is an improved copy; it does not replace the original.
6. **Start with a preview for long videos** before hours of processing.

### 3.8 Using it without Claude Code (command line)

From inside the cloned project folder (or replace `python -m vidimg` with `python <plugin folder>/vidimg_run.py`):

```bash
python -m vidimg analyze "D:/Videos/episode.mp4"
```

```bash
python -m vidimg restore-video "D:/Videos/episode.mp4" --content anime --scale 2
```

```bash
python -m vidimg restore-video "D:/Videos/wedding.avi" --content real --preview-only
```

```bash
python -m vidimg restore "D:/Photos/1995" --out "D:/VidIMG/Photos1995"
```

```bash
python -m vidimg verify "D:/original.jpg" "D:/result.png"
```

### 3.9 Real measured times (on an RTX 5060 Ti with 16 GB of memory, 2x upscale, cartoon)

| Video | Length | Full processing time |
|---|---|---|
| 640×480 | 21 minutes | About 30 minutes |
| 720×480 | 24 minutes | About 28 minutes |
| 720×576 | 24 minutes | About 42 minutes |
| 1920×1080 to 4K | 22 minutes | About 2 hours 20 minutes |

Real footage is slower: the fast general-v3 model processes about 7.8 frames per second from 720p to 1080p.

---

## 4. Troubleshooting

| Problem | Solution |
|---|---|
| `vidimg` tools do not appear in Claude Code | Run `/vidimg:setup-vidimg`. If it shows READY, restart Claude Code and check the server with `/mcp` |
| `ffmpeg not on PATH` after installing it | Open a new terminal window, or run the setup command again, because it finds FFmpeg automatically |
| `SHA-256 ... does not match` | The downloaded file is not the official release, or it was damaged during download. Delete the file named in the message and run again. **Do not skip this check** |
| `PyTorch sees no CUDA GPU` | Update the NVIDIA driver, then restart. Without the graphics card the tool works, but very slowly |
| `Video2X did not start` | Install `winget install Microsoft.VCRedist.2015+.x64` and update the graphics card driver |
| The result was "not delivered" | The guard rejected it for a reason given in the report, and the original is safe. Try a different content type if the one chosen was wrong |
| Disk space | Change the results folder: `/vidimg:setup-vidimg E:/VidIMG` |

---

## 5. For developers

- Design and evidence: [PLAN.md](PLAN.md). Detailed requirements: [REQUIREMENTS.md](REQUIREMENTS.md).
- Tests: `python -m pytest`. They do not run the models.
- The skill: `.claude/skills/restore-media/SKILL.md`. The commands: `.claude/commands/`. The server: `vidimg/mcp_server.py`.
- The `tools/` folder is not distributed; everything in it is downloaded by the install command. Only its `manifest.json` and `lock.json` files are distributed,
  and they contain the official release links and their fingerprints.

---

## License

The project is licensed under the [Apache 2.0](LICENSE) license: anyone may use, modify and redistribute it, even commercially, as long as they keep
the copyright notice and the [NOTICE](NOTICE) file. External programs and models are downloaded from their official sources under their own licenses.
To contribute to development, see [CONTRIBUTING.md](CONTRIBUTING.md).
