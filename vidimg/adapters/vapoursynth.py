"""VapourSynth backend for the classical pre-pass: the SECOND engine behind the same stage interface as
vidimg/adapters/avisynth.py (decision record: research/avisynth/05_integration_design.md, section 9).

* Parity measured on the ground-truth clips: QTGMC within +-0.1 dB of AviSynth+ on 4/4 clips, TFM+TDecimate
  identical. Quality is not the reason to prefer either engine.
* Runs in an isolated environment (tools/vs-venv, built by `python -m vidimg.vs_install`): VapourSynth R72
  pinned, because R73+ no longer loads API 3 plugins and MVTools/TIVTC have no API 4 Windows builds yet.
  Nothing is installed system-wide, so this engine also works on machines without AviSynth+.
* Used automatically when AviSynth+ is unavailable (before the plain BWDIF fallback). Promotion to primary:
  when API 4 Windows builds of MVTools + TIVTC exist and parity is re-measured on the current core.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from .. import config, proc
from . import AdapterResult, avisynth as avs, safe_output_dir
from .. import media

VENV = config.TOOLS_DIR / "vs-venv"
PYTHON = VENV / "Scripts" / "python.exe"
PLUGINS = VENV / "Lib" / "site-packages" / "vapoursynth" / "plugins" / "vsrepo"
RUNNER = Path(__file__).resolve().parents[1] / "vs_runner.py"
SUPPORTED = (avs.Deinterlace, avs.IVTC, avs.ChromaShift, avs.MaskBottom, avs.SRestore)
_KIND = {avs.Deinterlace: "deinterlace", avs.IVTC: "ivtc", avs.ChromaShift: "chroma", avs.MaskBottom: "mask",
         avs.SRestore: "srestore"}


def available() -> tuple[bool, str]:
    if not PYTHON.exists():
        return False, "VapourSynth environment missing: run python -m vidimg.vs_install"
    if not (PLUGINS / "libmvtools.dll").exists():
        return False, "VapourSynth plugins missing: run python -m vidimg.vs_install"
    from .. import vs_install
    if vs_install.VIOLATION.exists():
        return False, ("VapourSynth files changed since the first install (tools/vapoursynth/lock_violation.json): "
                       "run python -m vidimg.vs_install to check again, or --relock after a deliberate update")
    if not avs.system_ffmpeg():
        return False, "no ffmpeg on PATH"
    return True, "ok"


def supports(stages: list) -> bool:
    return all(isinstance(s, SUPPORTED) for s in stages)


def spec(src: Path, stages: list, cache: Path, colour: dict, fps, trim: tuple[int, int] | None = None) -> dict:
    sp = {"src": str(src), "cache": str(cache), "fps": [fps.numerator, fps.denominator],
          "full_range": avs.is_full_range(colour), "plugins": str(PLUGINS),
          "stages": [{"kind": _KIND[type(s)], **asdict(s)} for s in avs.order(stages)]}
    if trim:
        sp["trim"] = list(trim)
    return sp


def render_span(src: Path, dst: Path, stages: list, first: int, count: int, *, work: Path,
                timeout: float = 900) -> AdapterResult:
    """A short span through the stages, video only (FFV1 .mkv): trials such as choosing sRestore's rate."""
    ok, why = available()
    if not ok:
        return AdapterResult(False, error=why)
    src, dst = Path(src).resolve(), Path(dst).resolve()
    safe_output_dir(dst.parent, src)
    work.mkdir(parents=True, exist_ok=True)
    fps, _, colour = avs._rate(src)
    sp = spec(src, stages, work / (src.stem + ".vs.lwi"), colour, fps, trim=(first, count))
    sp_path = work / (dst.stem + ".vs.json")
    sp_path.write_text(json.dumps(sp, indent=1), encoding="utf-8")
    log_path = work / (dst.stem + ".vs.log")
    with open(log_path, "wb") as log_f:
        worker = subprocess.Popen([str(PYTHON), "-I", str(RUNNER), str(sp_path)], stdout=subprocess.PIPE,
                                  stderr=log_f, stdin=subprocess.DEVNULL)
        r = proc.run("vapoursynth", [avs.system_ffmpeg(), "-hide_banner", "-nostdin", "-y", "-f", "yuv4mpegpipe",
                                     "-i", "pipe:0", *avs.INTERMEDIATE, str(dst)],
                     stdin_pipe=worker.stdout, timeout=timeout)
        worker.stdout.close()
        worker.wait()
    if worker.returncode != 0 or r.returncode != 0 or not dst.exists():
        werr = log_path.read_text(encoding="utf-8", errors="replace")
        msg = next((ln for ln in reversed(werr.splitlines()) if ln.strip() and "autoloading failed" not in ln), None)
        return AdapterResult(False, error=msg or f"ffmpeg exit {r.returncode}", run=r)
    return AdapterResult(True, [dst], run=r)


def render(src: Path, dst: Path, stages: list, *, work: Path | None = None, timeout: float | None = None,
           on_progress: Callable[[int, int], None] | None = None) -> AdapterResult:
    ok, why = available()
    if not ok:
        return AdapterResult(False, error=why)
    if not supports(stages):
        return AdapterResult(False, error="the VapourSynth backend implements field handling, chroma shift and sRestore only")
    src, dst = Path(src).resolve(), Path(dst).resolve()
    if dst.suffix.lower() != ".nut":
        raise ValueError("the classical intermediate is FFV1 in .nut")
    safe_output_dir(dst.parent, src)
    work = Path(work or dst.parent).resolve()
    work.mkdir(parents=True, exist_ok=True)
    fps, packets, colour = avs._rate(src)
    if not fps:
        return AdapterResult(False, error="could not read the source frame rate")
    sp = spec(src, stages, work / (dst.stem + ".vs.lwi"), colour, fps)
    sp_path = work / (dst.stem + ".vs.json")
    sp_path.write_text(json.dumps(sp, indent=1), encoding="utf-8")
    out_fps = fps
    for s in stages:
        out_fps *= s.rate()
    if avs.is_full_range(colour):
        colour = {**colour, "color_range": "tv"}
    # no colour options on the ffmpeg command: NUT cannot store them (the sidecar does), and with a Y4M
    # input ffmpeg 8 then inserted a conversion that cost 7 dB on the telecine test (46.16 -> 39.06 dB)

    # worker (isolated Python, -I: no user site, no cwd on sys.path) -> Y4M pipe -> system ffmpeg
    log_path = work / (dst.stem + ".vs.log")
    log_f = open(log_path, "wb")             # a file, not a pipe: index progress could fill a pipe and deadlock
    worker = subprocess.Popen([str(PYTHON), "-I", str(RUNNER), str(sp_path)], stdout=subprocess.PIPE,
                              stderr=log_f, stdin=subprocess.DEVNULL)
    args = [avs.system_ffmpeg(), "-hide_banner", "-nostdin", "-y", "-f", "yuv4mpegpipe", "-i", "pipe:0",
            "-i", str(src), "-map", "0:v:0", "-map", "1:a?", "-c:a", "copy", "-map_metadata", "1",
            *avs.INTERMEDIATE, "-metadata", "comment=VidIMG VapourSynth pre-pass: " +
            json.dumps(sp["stages"], separators=(",", ":")), str(dst)]
    r = proc.run("vapoursynth", args, stdin_pipe=worker.stdout, timeout=timeout)
    worker.stdout.close()
    worker.wait()
    log_f.close()
    werr = log_path.read_text(encoding="utf-8", errors="replace")
    details = {"spec": sp, "fps_in": str(fps), "fps_out": str(out_fps), "worker_exit": worker.returncode}
    if worker.returncode != 0 or r.returncode != 0 or not dst.exists():
        msg = next((ln for ln in reversed(werr.splitlines()) if ln.strip() and "autoloading failed" not in ln), None)
        return AdapterResult(False, error=msg or f"ffmpeg exit {r.returncode}", run=r, details=details)
    details["colour"] = media.write_colour_sidecar(dst, colour)
    expected = round(packets * float(out_fps / fps)) if packets else 0
    got = avs._rate(dst)[1]
    details.update(expected_frames=expected, frames=got)
    if expected and abs(got - expected) > max(2, expected * 0.002):
        return AdapterResult(False, [dst], error=f"frame count {got} != expected {expected}", run=r, details=details)
    return AdapterResult(True, [dst], run=r, details=details)
