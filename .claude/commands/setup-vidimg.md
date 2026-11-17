---
description: Install or repair everything VidIMG needs on this machine (free tools only) and connect it to Claude Code
argument-hint: [output folder, e.g. D:/VidIMG]
---
First run `python "${CLAUDE_PLUGIN_ROOT}/vidimg_run.py" setup --check` and show the person, in Arabic, which parts are missing.

Then explain what the install will do before running it: about 5 GB of downloads (PyTorch CUDA, the open
models, Video2X 6.4.0, the OpenCV face models, portable VapourSynth), all free and official releases checked
against pinned SHA-256 hashes, installed in VidIMG's own folder (`%LOCALAPPDATA%\VidIMG` for a plugin install, the project folder for a clone), never system-wide; FFmpeg comes from winget only
if it is missing. Ask which folder should hold restored files, jobs and logs (a drive with space; default
`workspace/`) unless it was given: $ARGUMENTS

After the person agrees, run `python "${CLAUDE_PLUGIN_ROOT}/vidimg_run.py" setup --yes --output "<folder>"` in the background (about 6-30 minutes,
depending on the connection) and report each step's result. Winget installing FFmpeg is a system-wide install: name it
and get a clear yes before a run that may do it. When it ends with READY, tell the person to restart Claude
Code (`/exit`, then `claude`); for a cloned project also approve the `vidimg` MCP server. Then run `doctor()`. If a step fails, quote its
note and the fix it suggests; never work around a failed SHA-256 check.

(`${CLAUDE_PLUGIN_ROOT}` is the VidIMG folder. When VidIMG is opened as a project instead of installed as a plugin, that text stays unfilled: use the project folder, e.g. `python vidimg_run.py setup`.)
