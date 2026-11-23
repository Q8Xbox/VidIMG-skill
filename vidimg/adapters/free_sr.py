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
MLRT_RUNNER = Path(__file__).resolve().parents[1] / "mlrt_runner.py"
ENGINE_CHOICES = ("auto", "torch", "mlrt")


def _torch_cuda() -> bool:
    """Does the PyTorch model environment see a CUDA GPU? (asked once per process)"""
    global _CUDA
    if _CUDA is None:
        try:
            r = subprocess.run([str(PYTHON), "-I", "-c", "import torch; print(torch.cuda.is_available())"],
                               capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, stdin=subprocess.DEVNULL)
            _CUDA = r.returncode == 0 and r.stdout.strip().endswith("True")
        except (OSError, subprocess.TimeoutExpired):
            _CUDA = False
    return _CUDA


_CUDA: bool | None = None


def video_engine(model: str, preference: str | None = None) -> tuple[str | None, str]:
    """Which engine runs this video model: ('torch' | 'mlrt' | None, reason).

    torch = PyTorch CUDA (tools/sr-venv); mlrt = vs-mlrt, ONNX Runtime DirectML inside VidIMG's VapourSynth
    (any DirectX 12 GPU). 'auto' keeps PyTorch where CUDA works and uses vs-mlrt otherwise (AMD / Intel
    GPUs, or no CUDA build). Only models with an ONNX conversion can use vs-mlrt."""
    from .. import mlrt_install
    pref = (preference or config.VIDEO_ENGINE or "auto").lower()
    if pref not in ENGINE_CHOICES:
        return None, f"unknown engine {pref!r}; choose one of {', '.join(ENGINE_CHOICES)}"
    m_ok, m_why = mlrt_install.available(model)
    if model in mlrt_install.downloads():                   # exists as ONNX only: no PyTorch weights to fall back to
        return ("mlrt", "an ONNX-only model") if m_ok else (None, m_why)
    if pref == "mlrt":
        return ("mlrt", "requested") if m_ok else (None, m_why)
    t_ok = PYTHON.exists() and model_file(model).exists()
    if pref == "torch":
        return ("torch", "requested") if t_ok else (None, "PyTorch model environment or weights missing: run python -m vidimg.free_install")
    if t_ok and _torch_cuda():
        return "torch", "auto: PyTorch CUDA"
    if m_ok:
        return "mlrt", "auto: no CUDA GPU for PyTorch, ONNX Runtime DirectML instead"
    return ("torch", "auto: PyTorch (no CUDA GPU and no vs-mlrt: slow CPU)") if t_ok else (None, m_why)


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
    engine: str = ""          # "" = config.VIDEO_ENGINE (auto / torch / mlrt)

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


STALL_SECONDS = 1800.0          # encoder silent this long = the worker upstream hangs (no frame takes 30 min)
CHILD_EXIT_SECONDS = 60.0       # decoder and worker get this long to end by themselves once the encoder is gone


def run(src: Path, dst: Path, stages: list, encoder: str = "hevc10", *,
        on_progress: Callable[[int, int], None] | None = None, timeout: float | None = None,
        start: float | None = None, duration: float | None = None, expected_frames: int | None = None,
        audio: bool = True, post_vf: str | None = None, colour: dict | None = None, **_) -> AdapterResult:
    st = next(s for s in stages if isinstance(s, FreeSR))
    engine, why = video_engine(st.model, st.engine or None)
    if engine is None:
        return AdapterResult(False, error=why)
    if engine == "torch":
        ok, why = available()
        if not ok:
            return AdapterResult(False, error=why)
    src, dst = Path(src), Path(dst)
    safe_output_dir(dst.parent, src)
    info = media.probe(src, exact=start is None and duration is None)
    w, h, fps = info["width"], info["height"], info["fps"]
    ow, oh = st.out_size(w, h)
    expected = expected_frames if expected_frames is not None else (
        round(min(duration, info["duration"] - (start or 0)) * fps) if duration else info["frames"])
    # microseconds: "%g" keeps six digits in all, so past 1000 s a seek lost its fraction (1234.5678 -> 1234.57)
    seek = (["-ss", f"{start:.6f}"] if start is not None else []) + (["-t", f"{duration:.6f}"] if duration is not None else [])
    colour = colour if colour is not None else (media.colour_hint(src) or
                                                {k: info.get(k) for k in ("color_primaries", "color_transfer") if info.get(k)})
    full = colour.get("color_range") in ("pc", "jpeg") or str(info.get("pix_fmt", "")).startswith("yuvj")
    from .. import mlrt_install
    fetched = mlrt_install.downloads().get(st.model)
    heavy = not fetched and not manifest()[st.model]["training"].endswith("compact")
    if engine == "mlrt":
        from . import vapoursynth as _vs
        if fetched:
            try:
                mlrt_install.ensure(st.model, log=lambda m: None)
            except Exception as e:  # noqa: BLE001
                return AdapterResult(False, error=f"{st.model} could not be downloaded: {e}")
        spec = {"onnx": str(mlrt_install.onnx_path(st.model)), "width": w, "height": h, "out_width": ow,
                "out_height": oh, "frames": expected or info["frames"], "plugins": str(_vs.PLUGINS),
                "vsmlrt_dir": str(mlrt_install.ROOT), "backend": "dml", "fp16": True, "streams": 1,
                "half": bool(fetched and fetched.get("half")),
                # whole SD frames fit (tested 720x576 x4); larger heavy-model frames run in tiles
                "tile": 512 if heavy and w * h > 450_000 else 0}
        worker_cmd = [str(mlrt_install.VS_PY), "-I", str(MLRT_RUNNER)]
    else:
        spec = {"model": str(model_file(st.model)), "width": w, "height": h, "out_width": ow, "out_height": oh}
        if heavy:
            # heavy (RRDB/Swin) models: four batched 720p frames overflowed VRAM and Windows silently fell back
            # to system memory (16 min stuck on 150 frames). The worker sizes it from the card's memory now:
            # whole frames when they fit (no tile seams), otherwise 512-px tiles one frame at a time
            spec.update(tile="auto", batch="auto", heavy=True)
        worker_cmd = [str(PYTHON), "-I", str(RUNNER), "video"]
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
    worker = subprocess.Popen([*worker_cmd, str(sp)], stdin=dec.stdout, stdout=subprocess.PIPE, stderr=wlog)
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
                "-metadata", f"comment=Restored with VidIMG (free engine, {engine}): {st.describe()}"]
    if dst.suffix.lower() in (".mp4", ".mov", ".m4v"):
        enc_args += ["-movflags", "+write_colr+faststart"]
    frame_re = re.compile(r"frame=\s*(\d+)")

    def line(t: str) -> None:
        m = frame_re.search(t)
        if m and on_progress:
            on_progress(int(m.group(1)), expected)

    try:
        r = proc.run("free_sr_video", [*enc_args, str(dst)], stdin_pipe=worker.stdout, on_line=line, timeout=timeout,
                     stall_timeout=STALL_SECONDS)
    finally:
        # the encoder is gone (finished, died on a full disk, or killed as stalled): nothing may outlive it. A
        # worker left writing into a closed pipe used to block for ever and keep the GPU
        worker.stdout.close()
        for child in (worker, dec):
            try:
                child.wait(timeout=CHILD_EXIT_SECONDS)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        wlog.close()
    if worker.returncode != 0 or r.returncode != 0 or not dst.exists():
        werr = worker_log.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        return AdapterResult(False, error=(werr[-1] if werr else None) or f"encoder exit {r.returncode}", run=r)
    out = media.probe(dst, exact=True)
    details = {"input": info, "output": out, "expected_frames": expected, "stage": st.describe(), "engine": engine,
               "engine_reason": why}
    if expected and abs(out["frames"] - expected) > max(2, expected * 0.002):
        return AdapterResult(False, [dst], error=f"frame count {out['frames']} != expected {expected}", run=r, details=details)
    if audio and info["audio_streams"] and not out["audio_streams"]:
        return AdapterResult(False, [dst], error="audio stream was lost", run=r, details=details)
    return AdapterResult(True, [dst], run=r, details=details)
