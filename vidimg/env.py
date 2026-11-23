"""Environment doctor (free tools only): the open-model GPU runtime, FFmpeg features, the classical
engines (AviSynth+ / VapourSynth), the GPU, Video2X and the face models. No paid software is checked or
required.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field

from . import config


@dataclass
class Check:
    name: str
    status: str  # ok | warn | fail | info | n/a
    message: str
    data: dict = field(default_factory=dict)


def _gpu() -> Check:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return Check("gpu", "warn", "nvidia-smi not found: the open models fall back to the CPU (very slow)")
    out = subprocess.run([smi, "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv,noheader"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30, stdin=subprocess.DEVNULL).stdout.strip()
    return Check("gpu", "ok", out, {"nvidia_smi": out})


def _engine_checks() -> list[Check]:
    from . import mlrt_install
    from .adapters import avisynth, free_sr, vapoursynth
    out: list[Check] = []
    m_ok, m_why = mlrt_install.available()
    models = [k for k in [*mlrt_install.manifest()['models'], *mlrt_install.downloads()] if mlrt_install.onnx_path(k).exists()]
    out.append(Check("vs_mlrt", "ok" if m_ok and models else "info",
                     f"vs-mlrt (ONNX Runtime DirectML, any DirectX 12 GPU): {', '.join(models) or 'no ONNX models'}"
                     if m_ok else f"optional second engine for AMD/Intel GPUs: {m_why}"))
    ok, why = free_sr.available()
    if ok:
        r = subprocess.run([str(free_sr.PYTHON), "-I", "-c", "import torch;print(torch.__version__, torch.cuda.is_available(), "
                            "torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL, timeout=120)
        cuda = "True" in r.stdout
        out.append(Check("ai_engine", "ok" if cuda else "warn",
                         f"PyTorch {r.stdout.strip()}; {len(free_sr.manifest())} open models"
                         + ("" if cuda else " (no CUDA GPU: CPU fallback, very slow)")))
    else:
        out.append(Check("ai_engine", "fail", why))
    ff = shutil.which("ffmpeg")
    if ff:
        filt = subprocess.run([ff, "-hide_banner", "-filters"], capture_output=True, text=True, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL).stdout
        dmx = subprocess.run([ff, "-hide_banner", "-demuxers"], capture_output=True, text=True, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL).stdout
        miss = [n for n, ok_ in (("libvmaf", " libvmaf " in filt), ("idet", " idet " in filt), ("scdet", " scdet " in filt))
                if not ok_]
        out.append(Check("ffmpeg", "fail" if miss else "ok", f"{ff}" + (f"; missing filters {miss}" if miss else "")
                         + ("" if " avisynth " in dmx else "; no AviSynth demuxer (AviSynth+ engine unavailable)")))
    else:
        out.append(Check("ffmpeg", "fail", "ffmpeg not on PATH (winget install Gyan.FFmpeg)"))
    if not shutil.which("ffprobe"):
        out.append(Check("ffprobe", "fail", "ffprobe not on PATH (installed with FFmpeg)"))
    a, v = avisynth.available(), vapoursynth.available()
    out.append(Check("classical_engine", "ok" if a[0] or v[0] else "warn",
                     f"AviSynth+: {'ok' if a[0] else a[1]}; VapourSynth: {'ok' if v[0] else v[1]}"
                     + ("" if a[0] or v[0] else "; interlaced/telecine fall back to ffmpeg bwdif/fieldmatch")))
    return out


def doctor(**_) -> dict:
    """Run every check; returns a JSON-serialisable report with an overall `healthy` flag."""
    config.ensure_workspace()
    checks = [Check("mode", "ok", "free tools only (no paid software)"), *_engine_checks(), _gpu()]
    if config.VIDEO2X_EXE.exists():
        out = subprocess.run([str(config.VIDEO2X_EXE), "-l"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
                             stdin=subprocess.DEVNULL)
        dev = out.stdout.strip().splitlines()[0] if out.stdout.strip() else "no Vulkan device listed"
        checks.append(Check("video2x", "ok" if "0." in dev else "warn", f"{dev} (anime content, RIFE interpolation)"))
    else:
        checks.append(Check("video2x", "warn", f"Video2X not found at {config.VIDEO2X_EXE} (only needed for anime / RIFE)"))
    faces_ok = all((config.FACE_MODELS_DIR / n).exists() for n in
                   ("face_detection_yunet_2023mar.onnx", "face_recognition_sface_2021dec.onnx"))
    checks.append(Check("face_models", "ok" if faces_ok else "fail",
                        "YuNet + SFace present" if faces_ok else "face identity models missing (guard disabled)"))
    try:
        free = shutil.disk_usage(config.OUTPUT_ROOT if config.OUTPUT_ROOT.exists() else config.OUTPUT_ROOT.anchor).free / 2**30
        checks.append(Check("output", "ok" if free >= 20 else "warn",
                            f"{config.OUTPUT_ROOT} ({free:.0f} GB free)" + ("" if free >= 20 else
                            "; videos need tens of GB: set another folder with python -m vidimg.setup --output")))
    except OSError as e:
        checks.append(Check("output", "warn", f"{config.OUTPUT_ROOT}: {e}"))
    healthy = not any(c.status == "fail" for c in checks)
    return {"healthy": healthy, "checks": [asdict(c) for c in checks]}


def print_report(report: dict, stream=sys.stdout) -> None:
    icons = {"ok": "OK  ", "warn": "WARN", "fail": "FAIL", "info": "INFO"}
    for c in report["checks"]:
        stream.write(f"[{icons.get(c['status'], c['status'])}] {c['name']:<32} {c['message']}\n")
    stream.write(f"\nhealthy: {report['healthy']}\n")
