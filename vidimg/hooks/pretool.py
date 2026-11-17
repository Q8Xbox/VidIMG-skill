"""PreToolUse hook: media-safety rules for Claude Code in this project (PLAN.md section 4).

Reads the hook JSON on stdin; prints a decision only when a rule applies (otherwise the normal
permission flow continues). Standalone (stdlib only) so it stays fast and cannot break on imports.

deny  - ffmpeg writing its output onto one of its inputs
      - Video2X Real-ESRGAN / Real-CUGAN on content not declared anime (they destroy real faces)
ask   - deleting, moving or overwriting photos/videos outside the workspace (shell or Write/Edit)
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
from pathlib import Path

PROJECT = Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parents[2])
_CODE = Path(__file__).resolve().parents[2]
# VidIMG's home (same rule as vidimg.config: a plugin install keeps its data in %LOCALAPPDATA%/VidIMG)
if os.environ.get("VIDIMG_HOME"):
    HOME = Path(os.environ["VIDIMG_HOME"]).expanduser()
elif "/.claude/plugins/" in _CODE.as_posix().lower():
    HOME = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".local" / "share") / "VidIMG"
else:
    HOME = _CODE


def _setting_output() -> str | None:
    try:
        return json.loads((HOME / "vidimg.local.json").read_text(encoding="utf-8")).get("output")
    except (OSError, ValueError):
        return None


WORKSPACES = {(PROJECT / "workspace").resolve(), (HOME / "workspace").resolve()}
# the output folder chosen at setup (or VIDIMG_OUTPUT, e.g. D:/VidIMG) counts as workspace too
_out = os.environ.get("VIDIMG_OUTPUT") or _setting_output()
OUTPUT = Path(_out).resolve() if _out else None
MEDIA = r"\.(jpe?g|png|tiff?|webp|bmp|heic|dng|cr2|nef|arw|mp4|mkv|mov|avi|m4v|mpe?g|wmv|ts|mts|m2ts|3gp|webm)"
MEDIA_RE = re.compile(MEDIA + r"\b", re.I)


def decide(decision: str, reason: str) -> None:
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": decision,
                                             "permissionDecisionReason": "VidIMG media safety: " + reason}},
                     ensure_ascii=False))
    sys.exit(0)


def _inside_workspace(p: str) -> bool:
    try:
        path = Path(p.strip("\"'"))
        path = (PROJECT / path) if not path.is_absolute() else path
        r = path.resolve()
        return any(root == r or root in r.parents for root in (*WORKSPACES, OUTPUT) if root is not None)
    except (OSError, ValueError):
        return False


def _tokens(cmd: str) -> list[str]:
    try:
        return shlex.split(cmd, posix=True)
    except ValueError:
        return cmd.split()


def check_shell(cmd: str) -> None:
    low = cmd.lower()
    # only an actual Video2X run with an anime processor (listing the models folder used to be denied too)
    if re.search(r"(video2x(\.exe)?|vidimg\s+video2x)\b[^|;&]*?(-p|--processor)[\s=]+(realesrgan|realcugan)", low) \
            and "anime" not in low:
        decide("deny", "Real-ESRGAN / Real-CUGAN destroy real faces (measured); use them only for anime "
                       "(--content anime); real footage uses the faithful open models.")
    # ffmpeg output == input
    if re.search(r"\bffmpeg(\.exe)?\b", low):
        for seg in re.split(r"&&|\|\||;|\|", cmd):
            t = _tokens(seg)
            if not any(re.search(r"ffmpeg(\.exe)?$", x, re.I) for x in t):
                continue
            inputs = {Path(t[i + 1]).as_posix().lower() for i, x in enumerate(t[:-1]) if x == "-i"}
            # outputs never follow -i (ffmpeg ... -i a.avi -f null - has no media output at all)
            outs = [x for i, x in enumerate(t[1:], 1) if MEDIA_RE.search(x) and not x.startswith("-") and t[i - 1] != "-i"]
            if outs and Path(outs[-1]).as_posix().lower() in inputs:
                decide("deny", f"ffmpeg would write its output onto its input ({outs[-1]}).")
    # deleting / moving media outside the workspace
    if re.search(r"(^|[\s;&|(])(rm|del|erase|rmdir|remove-item|mv|move|move-item|ren|rename-item)\b", low):
        for x in _tokens(cmd):
            if MEDIA_RE.search(x) and not _inside_workspace(x):
                decide("ask", f"this command deletes or moves a photo/video outside the workspace ({x}); "
                              "originals are never touched without the person's approval.")


def check_write(path: str) -> None:
    if path and MEDIA_RE.search(path) and not _inside_workspace(path):
        decide("ask", f"writing a photo/video file outside the workspace ({path}) could replace an original.")


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except ValueError:
        return
    tool, inp = data.get("tool_name", ""), data.get("tool_input") or {}
    if tool in ("Bash", "PowerShell"):
        check_shell(str(inp.get("command", "")))
    elif tool in ("Write", "Edit", "NotebookEdit"):
        check_write(str(inp.get("file_path") or inp.get("notebook_path") or ""))


if __name__ == "__main__":
    main()
