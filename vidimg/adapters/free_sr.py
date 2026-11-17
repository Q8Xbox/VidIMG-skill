"""Free engine (no paid software): open restoration models run by vidimg/sr_runner.py in the isolated
GPU environment tools/sr-venv (PyTorch CUDA + spandrel). One contract for images and video so the
pipelines, the guard and the batches work unchanged:

* images: grouped calls (one model load for many photos), exact output size, 16-bit PNG;
* video: `run()` (start/duration, audio, post_vf, expected_frames, progress):
  system ffmpeg decode -> GPU worker -> system ffmpeg 10-bit encode, audio and metadata copied.

Models and licences: tools/free-models/manifest.json (BSD-3 / Apache-2.0, official releases).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .. import config, proc
from . import AdapterResult, safe_output_dir
from .. import media

ROOT = config.TOOLS_DIR / "free-models"
PYTHON = config.TOOLS_DIR / "sr-venv" / "Scripts" / "python.exe"
RUNNER = Path(__file__).resolve().parents[1] / "sr_runner.py"


def manifest() -> dict:
    return json.loads((config.MANIFEST_DIR / "free-models" / "manifest.json").read_text(encoding="utf-8"))["models"]


def model_file(key: str) -> Path:
    m = manifest()[key]
    return ROOT / (m.get("file") or m["url"].rsplit("/", 1)[-1])


def model_available(key: str) -> bool:
    return PYTHON.exists() and key in manifest() and model_file(key).exists()


def available() -> tuple[bool, str]:
    if not PYTHON.exists():
        return False, "free GPU runtime missing: run python -m vidimg.free_install"
    missing = [k for k, m in manifest().items() if not m.get("optional") and not model_file(k).exists()]
    if missing:
        return False, f"free models missing ({', '.join(missing)}): run python -m vidimg.free_install"
    if not shutil.which("ffmpeg"):
        return False, "no ffmpeg on PATH"
    return True, "ok"


# ----------------------------------------------------------------------------- images
def upscale_images(inputs: list[Path], out_dir: Path, *, model: str, scale: float, suffix: str,
                   tile: int = 512, timeout: float | None = None) -> dict[str, dict]:
    """One worker call for every image: {input path: {"ok", "output", "error", "seconds"}}."""
    ok, why = available()
    if not ok:
        return {str(i): {"ok": False, "error": why} for i in inputs}
    out_dir = Path(out_dir)
    for i in inputs:
        safe_output_dir(out_dir, Path(i))
    spec = {"model": str(model_file(model)), "inputs": [str(Path(i).resolve()) for i in inputs],
            "out_dir": str(out_dir.resolve()), "suffix": suffix, "scale": float(scale), "tile": tile}
    sp = out_dir / f".spec{suffix}.json"
    sp.write_text(json.dumps(spec), encoding="utf-8")
    r = proc.run("free_sr_images", [PYTHON, "-I", RUNNER, "image", sp], timeout=timeout or 600 + 120 * len(inputs),
                 keep_lines=5000)
    results = {}
    for ln in r.stdout_tail:
        if ln.startswith("{") and '"input"' in ln:
            try:
                d = json.loads(ln)
            except ValueError:
                continue
            results[d["input"]] = d
    err = next((ln for ln in reversed(r.stdout_tail) if "Error" in ln), None) or f"worker exit {r.returncode}"
    out = {}
    for i in inputs:
        d = results.get(str(Path(i).resolve()))
        out[str(i)] = d if d and d.get("ok") and Path(d["output"]).exists() else {"ok": False, "error": (d or {}).get("error") or err}
    return out


def upscale(image: Path, out_dir: Path, *, model: str, scale: float, suffix: str | None = None, **_) -> AdapterResult:
    suffix = suffix or f"_free_{model}"
    t0 = time.monotonic()
    d = upscale_images([Path(image)], Path(out_dir), model=model, scale=scale, suffix=suffix)[str(image)]
    if d.get("ok"):
        return AdapterResult(True, [Path(d["output"])], details={"seconds": d.get("seconds"), "wall": time.monotonic() - t0})
    return AdapterResult(False, error=d.get("error"))


def denoise(image: Path, out_dir: Path) -> AdapterResult:
    """SCUNet real-photo denoise at the original size."""
    return upscale(image, out_dir, model="scunet_real", scale=1.0, suffix="_denoised")


# ----------------------------------------------------------------------------- video
@dataclass
class FreeSR:
    """Free video enhancement stage: an open model at its native scale, then exact resize."""

    model: str = "general_x4v3"
    height: int | None = None
    width: int | None = None
    scale: float | None = None
    label: str = ""

    def out_size(self, w: int, h: int) -> tuple[int, int]:
        if self.height and not self.width:
            oh = self.height
            ow = round(w * oh / h / 2) * 2
        elif self.width and not self.height:
            ow = self.width
            oh = round(h * ow / w / 2) * 2
        elif self.width and self.height:
            ow, oh = self.width, self.height
        else:
            s = self.scale or 2
            ow, oh = round(w * s / 2) * 2, round(h * s / 2) * 2
        return ow, oh

    def describe(self) -> str:
        return f"free:{self.model}" + (f":h={self.height}" if self.height else "") + (f":x{self.scale:g}" if self.scale else "")


def run(src: Path, dst: Path, stages: list, encoder: str = "hevc10", *,
        on_progress: Callable[[int, int], None] | None = None, timeout: float | None = None,
        start: float | None = None, duration: float | None = None, expected_frames: int | None = None,
        audio: bool = True, post_vf: str | None = None, colour: dict | None = None, **_) -> AdapterResult:
    ok, why = available()
    if not ok:
        return AdapterResult(False, error=why)
    st = next(s for s in stages if isinstance(s, FreeSR))
    src, dst = Path(src), Path(dst)
    safe_output_dir(dst.parent, src)
    info = media.probe(src, exact=start is None and duration is None)
    w, h, fps = info["width"], info["height"], info["fps"]
    ow, oh = st.out_size(w, h)
    expected = expected_frames if expected_frames is not None else (
        round(min(duration, info["duration"] - (start or 0)) * fps) if duration else info["frames"])
    seek = (["-ss", f"{start:g}"] if start is not None else []) + (["-t", f"{duration:g}"] if duration is not None else [])
    colour = colour if colour is not None else (media.colour_hint(src) or
                                                {k: info.get(k) for k in ("color_primaries", "color_transfer") if info.get(k)})
    full = colour.get("color_range") in ("pc", "jpeg") or str(info.get("pix_fmt", "")).startswith("yuvj")
    spec = {"model": str(model_file(st.model)), "width": w, "height": h, "out_width": ow, "out_height": oh}
    if not manifest()[st.model]["training"].endswith("compact") and w * h > 300_000:
        # heavy (RRDB/Swin) models on whole HD frames overflow VRAM and Windows silently falls back to
        # system memory (observed: 16 min stuck on 150 frames at 720p): tile them, one frame at a time
        spec.update(tile=384, batch=1)
    sp = dst.parent / f".{dst.stem}.spec.json"
    sp.write_text(json.dumps(spec), encoding="utf-8")
    ff = shutil.which("ffmpeg")
    # exact rational frame rate (a rounded decimal turned 60000/1001 into 599401/10000)
    from . import avisynth as _avs
    exact = _avs._rate(src)[0]
    rate = f"{exact.numerator}/{exact.denominator}" if exact else (f"{fps:g}" if fps else "30")
    dec = subprocess.Popen([ff, "-hide_banner", "-loglevel", "error", "-nostdin", *seek, "-i", str(src), "-map", "0:v:0",
                            "-vf", f"scale=in_range={'full' if full else 'tv'}:out_range=full,format=rgb24",
                            # passthrough: every decoded frame exactly once. The default (cfr) for raw
                            # output duplicated the first frame of clips whose timestamps leave a gap
                            # (WhatsApp preview: 121 -> 122 frames), shifting every later frame by one
                            "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    worker_log = dst.parent / f".{dst.stem}.worker.log"
    wlog = open(worker_log, "wb")
    worker = subprocess.Popen([str(PYTHON), "-I", str(RUNNER), "video", str(sp)], stdin=dec.stdout,
                              stdout=subprocess.PIPE, stderr=wlog)
    dec.stdout.close()
    # HD output gets BT.709 (SD->HD matrix conversion, Doom9: convert 601->709 when SD becomes HD)
    hd = oh >= 720
    matrix = "bt709" if hd else ("bt470bg" if colour.get("color_space") in (None, "bt470bg", "smpte170m") else colour["color_space"])
    vf = f"scale=in_range=full:out_range=tv:out_color_matrix={'bt709' if hd else 'bt601'}" + (f",{post_vf}" if post_vf else "")
    # the raw pipe has no pixel aspect: keep the source's display aspect (PAL 16:15 came out 1:1, shown 5:4)
    from fractions import Fraction
    sn, sd = (int(x) for x in info.get("sar", "1:1").split(":"))
    sar = (Fraction(w * sn, h * sd) * Fraction(oh, ow)).limit_denominator(65535)
    vf += f",setsar={sar.numerator}/{sar.denominator}"
    tags = ["-colorspace", matrix, "-color_range", "tv"] + (["-color_primaries", "bt709", "-color_trc", "bt709"] if hd else [])
    enc_args = [ff, "-hide_banner", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{ow}x{oh}",
                "-r", rate, "-i", "pipe:0", *seek, "-i", str(src), "-map", "0:v:0",
                *(["-map", "1:a?", "-c:a", "copy"] if audio else ["-an"]), "-vf", vf,
                *media.encoder_args(encoder), *tags, "-map_metadata", "1",
                "-metadata", f"comment=Restored with VidIMG (free engine): {st.describe()}"]
    if dst.suffix.lower() in (".mp4", ".mov", ".m4v"):
        enc_args += ["-movflags", "+write_colr+faststart"]
    frame_re = re.compile(r"frame=\s*(\d+)")

    def line(t: str) -> None:
        m = frame_re.search(t)
        if m and on_progress:
            on_progress(int(m.group(1)), expected)

    r = proc.run("free_sr_video", [*enc_args, str(dst)], stdin_pipe=worker.stdout, on_line=line, timeout=timeout)
    worker.stdout.close()
    worker.wait()
    dec.wait()
    wlog.close()
    if worker.returncode != 0 or r.returncode != 0 or not dst.exists():
        werr = worker_log.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        return AdapterResult(False, error=(werr[-1] if werr else None) or f"encoder exit {r.returncode}", run=r)
    out = media.probe(dst, exact=True)
    details = {"input": info, "output": out, "expected_frames": expected, "stage": st.describe()}
    if expected and abs(out["frames"] - expected) > max(2, expected * 0.002):
        return AdapterResult(False, [dst], error=f"frame count {out['frames']} != expected {expected}", run=r, details=details)
    if audio and info["audio_streams"] and not out["audio_streams"]:
        return AdapterResult(False, [dst], error="audio stream was lost", run=r, details=details)
    return AdapterResult(True, [dst], run=r, details=details)
