"""Project paths and the approved default policy (PLAN.md section 5).

Free tools only (decided by the owner 2026-11-16): open restoration models on PyTorch CUDA, Video2X,
AviSynth+ / VapourSynth, FFmpeg and the OpenCV face models. No paid software is used or required."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent      # the code (read-only when installed as a plugin)
MANIFEST_DIR = PROJECT_ROOT / "tools"                       # pinned manifests and SHA-256 locks shipped with the code


def _home() -> Path:
    """Where downloads (tools/), the workspace and local settings live. A cloned project keeps them in its own
    folder; a Claude Code plugin is installed in a cache folder that is replaced on every update, so it keeps them
    in %LOCALAPPDATA%/VidIMG instead. VIDIMG_HOME overrides both."""
    if os.environ.get("VIDIMG_HOME"):
        return Path(os.environ["VIDIMG_HOME"]).expanduser()
    if "/.claude/plugins/" in PROJECT_ROOT.as_posix().lower():
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".local" / "share") / "VidIMG"
    return PROJECT_ROOT


HOME = _home()
TOOLS_DIR = HOME / "tools"
WORKSPACE = HOME / "workspace"
SETTINGS_FILE = HOME / "vidimg.local.json"                  # written by python -m vidimg.setup ({"output": ...})


def local_settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


# Outputs (restorations, jobs, run logs, analysis sheets) can live on a big drive: VIDIMG_OUTPUT, or the folder
# chosen at setup (vidimg.local.json). Calibration state always stays in the workspace. An output folder on a
# drive this machine does not have (a copied project) falls back to the workspace.
_out = os.environ.get("VIDIMG_OUTPUT") or local_settings().get("output")
OUTPUT_ROOT = Path(_out) if _out and Path(Path(_out).anchor or ".").exists() else WORKSPACE
LOG_DIR = OUTPUT_ROOT / "logs"
STATE_DIR = WORKSPACE / "state"
OUTPUT_DIR = WORKSPACE / "output"
PREVIEW_DIR = WORKSPACE / "previews"

VIDEO2X_EXE = TOOLS_DIR / "video2x-6.4.0" / "video2x.exe"
FACE_MODELS_DIR = TOOLS_DIR / "face-models"

@dataclass(frozen=True)
class Policy:
    """Approved defaults: the most conservative choice that keeps the original look."""

    allow_generative: bool = False          # no generative models (face restorers, diffusion) unless asked
    lighting_and_color: bool = False        # never change the look unless asked
    frame_interpolation: bool = False       # only on request (RIFE via Video2X)
    image_bit_depth: int = 16               # PNG 16-bit outputs
    video_codec: str = "hevc_10bit"         # HEVC 10-bit (NVENC, or x265 without an NVIDIA GPU); audio/metadata copied
    identity_threshold: float = 0.79        # SFace cosine vs bicubic input; calibrated 2026-11-15 (guard)
    max_refine_rounds: int = 3
    deblur: bool = False                    # opt-in: NAFNet-GoPro on photos flagged soft/blurred, judged by
                                            # the guard's deblur mode (PLAN.md 3.6)


DEFAULT_POLICY = Policy()

# Classical pre-pass engines in order of preference (decision record: research/avisynth/05, section 9).
# AviSynth+ is primary (maintained Windows plugin stack); VapourSynth is the validated second engine
# (parity +-0.1 dB) and runs when AviSynth+ is missing; ffmpeg's bwdif / fieldmatch is the last resort.
# Override for tests or a future promotion: VIDIMG_CLASSICAL_ENGINES=vapoursynth,avisynth
CLASSICAL_ENGINES_DEFAULT = ("avisynth", "vapoursynth")

def classical_engines() -> tuple[str, ...]:
    import os
    raw = os.environ.get("VIDIMG_CLASSICAL_ENGINES")
    names = tuple(n.strip() for n in raw.split(",") if n.strip()) if raw else CLASSICAL_ENGINES_DEFAULT
    unknown = [n for n in names if n not in CLASSICAL_ENGINES_DEFAULT]
    if unknown:
        raise ValueError(f"unknown classical engine(s) {unknown}; known: {CLASSICAL_ENGINES_DEFAULT}")
    return names


# Video models a person can ask for by name (restore-video --model / restore_video(model=...)); the default is
# the approved order, first candidate that passes the guard
VIDEO_MODEL_CHOICES = ("esrgan-v3", "cugan", "cugan-pro", "janai-sd", "janai-hd", "general-v3", "realesrnet", "bsrnet")
# engine for the open video models (RealESRNet, BSRNet, general-v3): auto = PyTorch where CUDA works, otherwise
# vs-mlrt (ONNX Runtime DirectML: AMD / Intel GPUs); torch / mlrt force one. CLI --engine sets it per run.
VIDEO_ENGINE_CHOICES = ("auto", "torch", "mlrt")
VIDEO_ENGINE = os.environ.get("VIDIMG_VIDEO_ENGINE", "auto")
# Texture return after the model (vidimg/texture.py): how much of the source's own texture between the lines is put
# back (0 = off, 1 = all the model removed). Every model flattens painted backgrounds and grain ("denoised to death",
# Doom9); at 1 the flat area came back to the source's level with the lines kept sharp.
TEXTURE_STRENGTH = float(os.environ.get("VIDIMG_TEXTURE", "1.0"))
# Intermediates of a delivered video (pre-pass, chunks, previews) are deleted once the guard has passed the result:
# they are several times the size of the deliverable, per episode. VIDIMG_KEEP_WORK=1 keeps them.
KEEP_WORK = os.environ.get("VIDIMG_KEEP_WORK", "").strip().lower() in ("1", "true", "yes")

# Video2X processors that must never touch real human faces (benchmark: identity 0.77-0.81).
VIDEO2X_ANIME_ONLY = frozenset({"realesrgan", "realcugan", "libplacebo"})


def ensure_workspace() -> None:
    for d in (LOG_DIR, STATE_DIR, OUTPUT_DIR, PREVIEW_DIR):
        d.mkdir(parents=True, exist_ok=True)
