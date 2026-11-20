"""First-run setup on a new Windows machine: one command installs and verifies everything VidIMG needs and
connects it to Claude Code. Every step is idempotent (run it again to repair or verify) and every download is
an official release checked against a pinned SHA-256. Nothing paid. Two things go outside the VidIMG folder, each
only on approval: FFmpeg through winget, and the Python packages of requirements.txt into the Python that runs
VidIMG (skipped when they are already there).

Usage:
    python -m vidimg.setup                     # interactive: asks before winget and for the output folder
    python -m vidimg.setup --output D:/VidIMG  # outputs (restorations, jobs, logs) on a big drive
    python -m vidimg.setup --yes               # no questions (winget allowed, default output folder)
    python -m vidimg.setup --check             # verify only, install nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from . import config

FACE_MODELS = {   # OpenCV Zoo (Apache-2.0); hashes verified against the official files 2026-11-17
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    "face_recognition_sface_2021dec.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
}
VIDEO2X_ZIP = ("https://github.com/k4yt3x/video2x/releases/download/6.4.0/video2x-windows-amd64.zip",
               "0337b6dcae2bad2fd13e43fbdb73ef728d2f6289f37cc3617e87ac3a1a5cf793")     # AGPL-3.0, 188 MB
WINGET_FFMPEG = ["winget", "install", "--id", "Gyan.FFmpeg", "-e", "--source", "winget"]


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _fetch(url: str, sha: str, dst: Path, log) -> None:
    from .free_install import _download
    if dst.exists() and _sha(dst) == sha:
        return
    log(f"  downloading {dst.name}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    _download(url, dst)
    got = _sha(dst)
    if got != sha:
        dst.unlink(missing_ok=True)
        raise RuntimeError(f"{dst.name}: SHA-256 {got[:12]}... does not match the pinned release; refusing")


def _ask(question: str, yes: bool) -> bool:
    if yes:
        return True
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


# ----------------------------------------------------------------------------- steps
def step_system(log, **_) -> dict:
    ok = sys.platform == "win32" and platform.architecture()[0] == "64bit" and sys.version_info >= (3, 12)
    vc = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32" / "vcruntime140_1.dll"
    smi = shutil.which("nvidia-smi")
    gpu = subprocess.run([smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL).stdout.strip() if smi else ""
    config.HOME.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(config.HOME.anchor).free // 2**30
    notes = []
    if not ok:
        notes.append("needs 64-bit Windows 10/11 and Python 3.12+")
    if not vc.exists():
        notes.append("Microsoft Visual C++ 2015-2022 x64 runtime missing (Video2X needs it): "
                     "winget install Microsoft.VCRedist.2015+.x64")
    if not gpu:
        notes.append("no NVIDIA GPU found: the models run on the CPU (very slow) and encoding uses x265")
    if disk < 10:
        notes.append(f"only {disk} GB free on the project drive; the software needs about 5 GB")
    return {"ok": ok, "python": sys.version.split()[0], "gpu": gpu or None, "free_GB": disk, "notes": notes}


def _winget_ffmpeg_dir() -> Path | None:
    links = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links"
    return links if (links / "ffmpeg.exe").exists() else None


def step_ffmpeg(log, yes=False, check=False, **_) -> dict:
    if not shutil.which("ffmpeg") and (d := _winget_ffmpeg_dir()):
        os.environ["PATH"] = f"{d}{os.pathsep}{os.environ['PATH']}"      # installed earlier, shell not reopened
    if not shutil.which("ffmpeg") and not check:
        if shutil.which("winget") and _ask("FFmpeg is missing. Install the free full build with winget (Gyan.FFmpeg)?", yes):
            subprocess.run(WINGET_FFMPEG + (["--accept-source-agreements", "--accept-package-agreements"] if yes else []))
            if d := _winget_ffmpeg_dir():
                os.environ["PATH"] = f"{d}{os.pathsep}{os.environ['PATH']}"
    ff = shutil.which("ffmpeg")
    if not ff:
        return {"ok": False, "notes": ["FFmpeg not found: winget install Gyan.FFmpeg (then reopen the terminal)"]}
    filters = subprocess.run([ff, "-hide_banner", "-filters"], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    missing = [f for f in ("libvmaf", "idet", "scdet", "bwdif") if f" {f} " not in filters]
    return {"ok": not missing and bool(shutil.which("ffprobe")), "path": ff,
            "notes": [f"this FFmpeg build lacks {missing}: use the gyan.dev full build"] if missing else []}


def step_python_packages(log, check=False, yes=False, **_) -> dict:
    req = config.PROJECT_ROOT / "requirements.txt"
    have = subprocess.run([sys.executable, "-c", "import numpy, cv2, skimage, scipy, PIL, mcp"], capture_output=True)
    if have.returncode == 0:
        return {"ok": True, "notes": []}                  # already there: this Python is not touched
    if not check:
        # these go into the Python that runs VidIMG. Outside a virtual environment that is the person's own
        # Python, so it is their decision
        if sys.prefix == sys.base_prefix and not _ask(
                f"Install VidIMG's Python packages (requirements.txt) into {sys.executable}?", yes):
            return {"ok": False, "notes": [f"Python packages missing; install them with: "
                                           f"\"{sys.executable}\" -m pip install -r \"{req}\""]}
        log("  pip install -r requirements.txt")
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)], capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode:
            return {"ok": False, "notes": [r.stderr.strip()[-400:]]}
    probe = subprocess.run([sys.executable, "-c", "import numpy, cv2, skimage, scipy, PIL, mcp"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return {"ok": probe.returncode == 0, "notes": [probe.stderr.strip()[-300:]] if probe.returncode else []}


def step_face_models(log, check=False, **_) -> dict:
    notes = []
    for name, (url, sha) in FACE_MODELS.items():
        dst = config.FACE_MODELS_DIR / name
        try:
            if check:
                if not dst.exists() or _sha(dst) != sha:
                    notes.append(f"{name} missing or changed")
            else:
                _fetch(url, sha, dst, log)
        except Exception as e:                                          # noqa: BLE001
            notes.append(str(e))
    return {"ok": not notes, "notes": notes}


def step_video2x(log, check=False, **_) -> dict:
    exe = config.VIDEO2X_EXE
    folder = exe.parent
    if not exe.exists() and not check:
        z = folder / "video2x-windows-amd64.zip"
        try:
            _fetch(*VIDEO2X_ZIP, z, log)
        except Exception as e:                                          # noqa: BLE001
            return {"ok": False, "notes": [str(e)]}
        with zipfile.ZipFile(z) as zf:
            for m in zf.namelist():                                     # no path traversal out of the folder
                target = (folder / m).resolve()
                if folder.resolve() not in target.parents and target != folder.resolve():
                    return {"ok": False, "notes": [f"unexpected path in the Video2X archive: {m}"]}
            zf.extractall(folder)
        z.unlink(missing_ok=True)                                       # 188 MB; the extracted files are what runs
    if not exe.exists():
        return {"ok": False, "notes": ["Video2X 6.4.0 missing (needed for cartoons/anime and RIFE only)"]}
    out = subprocess.run([str(exe), "-l"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60, stdin=subprocess.DEVNULL)
    dev = (out.stdout.strip().splitlines() or ["no Vulkan device listed"])[0]
    return {"ok": out.returncode == 0, "device": dev,
            "notes": [] if out.returncode == 0 else ["Video2X did not start: install the VC++ x64 runtime and the GPU driver"]}


def step_models(log, check=False, **_) -> dict:
    from . import free_install
    if check:
        from .adapters import free_sr
        ok, why = free_sr.available()
        return {"ok": ok, "notes": [] if ok else [why]}
    r = free_install.install(log=log)
    notes = list(r["errors"]) + ([] if r.get("gpu") else ["PyTorch sees no CUDA GPU: CPU fallback (very slow)"])
    return {"ok": not r["errors"], "runtime": r.get("runtime"), "notes": notes}


def step_vapoursynth(log, check=False, **_) -> dict:
    from .adapters import vapoursynth
    if not check:
        from . import vs_install
        try:
            vs_install.install(log=log)
        except SystemExit as e:
            return {"ok": False, "notes": [f"VapourSynth install failed (deinterlacing falls back to ffmpeg): {str(e)[-300:]}"]}
    ok, why = vapoursynth.available()
    return {"ok": ok, "notes": [] if ok else [f"{why} (interlaced/telecine fall back to ffmpeg bwdif/fieldmatch)"]}


def step_vs_mlrt(log, check=False, **_) -> dict:
    """Optional second engine: vs-mlrt (ONNX Runtime DirectML) for the faithful models on any DirectX 12 GPU."""
    from . import mlrt_install
    if not check:
        r = mlrt_install.install(log=log)
        if r["errors"]:
            return {"ok": False, "notes": r["errors"] + ["optional: without it, non-NVIDIA GPUs use the slow CPU path"]}
    ok, why = mlrt_install.available("realesrnet_x4")
    return {"ok": ok, "notes": [] if ok else [f"{why} (optional: only needed on AMD/Intel GPUs)"]}


def _is_plugin() -> bool:
    """Installed through the Claude Code plugin marketplace (the plugin itself provides the MCP server and hook)."""
    return config.HOME != config.PROJECT_ROOT and "/.claude/plugins/" in config.PROJECT_ROOT.as_posix().lower()


def step_claude_code(log, output: str | None = None, yes=False, check=False, **_) -> dict:
    """Remember the output folder (vidimg.local.json in VidIMG's home) and, for a cloned project, point its
    .mcp.json at THIS machine's Python. A plugin install needs nothing more: the plugin declares the MCP server
    and the safety hook itself. The committed .claude/settings.json is never rewritten."""
    current = config.local_settings().get("output")
    if output is None:
        output = current if current and Path(current).anchor and Path(Path(current).anchor).exists() else None
    if output is None and not yes and not check:
        try:
            output = input(f"Folder for restored files, jobs and logs [{config.WORKSPACE}]: ").strip() or None
        except EOFError:
            output = None
    out = Path(output).expanduser().resolve() if output else config.WORKSPACE
    plugin = _is_plugin()
    mcp_path = config.PROJECT_ROOT / ".mcp.json"
    mcp = json.loads(mcp_path.read_text(encoding="utf-8")) if mcp_path.exists() and not plugin else {}
    if check:
        server = mcp.get("mcpServers", {}).get("vidimg", {})
        ok = plugin or (bool(server) and Path(server.get("args", [""])[0]).exists()
                        and bool(shutil.which(server.get("command", ""))))
        return {"ok": ok, "output": str(out), "notes": [] if ok else ["Claude Code is not connected: run setup"]}
    out.mkdir(parents=True, exist_ok=True)
    config.SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.SETTINGS_FILE.write_text(json.dumps({**config.local_settings(), "output": out.as_posix()}, indent=2),
                                    encoding="utf-8")
    if not plugin:
        mcp.setdefault("mcpServers", {})["vidimg"] = {
            "command": Path(sys.executable).as_posix(),
            "args": [(config.PROJECT_ROOT / "vidimg" / "mcp_server.py").as_posix()],
            "env": {"PYTHONIOENCODING": "utf-8"}}
        mcp_path.write_text(json.dumps(mcp, indent=2), encoding="utf-8")
    return {"ok": True, "output": str(out), "notes": [f"VidIMG home (tools, workspace): {config.HOME}"]}


STEPS = [("system", step_system), ("ffmpeg", step_ffmpeg), ("python packages", step_python_packages),
         ("face models", step_face_models), ("open models + PyTorch", step_models), ("Video2X", step_video2x),
         ("VapourSynth", step_vapoursynth), ("vs-mlrt", step_vs_mlrt), ("Claude Code", step_claude_code)]
REQUIRED = {"system", "ffmpeg", "python packages", "face models", "open models + PyTorch", "Claude Code"}


def run(output: str | None = None, yes: bool = False, check: bool = False, log=print) -> dict:
    report = {}
    for name, fn in STEPS:
        log(f"[{name}]")
        try:
            r = fn(log, output=output, yes=yes, check=check)
        except Exception as e:                                          # noqa: BLE001
            r = {"ok": False, "notes": [f"{type(e).__name__}: {e}"]}
        report[name] = r
        log(f"  {'OK' if r['ok'] else ('FAIL' if name in REQUIRED else 'WARN')}"
            + "".join(f"\n  - {n}" for n in r.get("notes", [])))
        if name == "python packages" and not r["ok"]:
            break                                                       # nothing below imports without them
    report["ready"] = all(report.get(n, {}).get("ok") for n in REQUIRED)
    log(("\nREADY: restart Claude Code (/exit, then claude) so the vidimg tools load." if _is_plugin() else
         "\nREADY: open this folder in Claude Code and approve the 'vidimg' MCP server when asked.")
        if report["ready"] else "\nNOT READY: fix the FAIL lines above and run the setup again.")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m vidimg.setup", description=__doc__.split("\n\n")[0])
    ap.add_argument("--output", help="folder for restored files, jobs and logs (default: workspace/)")
    ap.add_argument("--yes", action="store_true", help="no questions: allow winget for FFmpeg, default folders")
    ap.add_argument("--check", action="store_true", help="verify only; install and change nothing")
    ap.add_argument("--json", action="store_true", help="print the report as JSON at the end")
    a = ap.parse_args(argv)
    rep = run(a.output, a.yes, a.check)
    if a.json:
        print(json.dumps(rep, indent=1, default=str))
    return 0 if rep["ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
