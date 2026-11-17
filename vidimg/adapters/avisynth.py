"""AviSynth+ classical pre-processing (field handling, deblocking, temporal denoising) in front of
the AI stage (open models). Research: research/avisynth/01-05.

Design (05_integration_design.md):
* every generated script loads ONLY the pinned private plugin set (tools/avisynth/plugins, installed by
  vidimg.avs_install): ClearAutoloadDirs()+AddAutoloadDir() come first, so the system plugin folders
  are never used (verified: a system-only function fails to resolve);
* L-SMASH's index goes to the workspace (cachefile=), never next to the user's original;
* processing in 16-bit, output 10-bit (ConvertBits(10, dither=1)) to a lossless FFV1 intermediate that
  (NUT container) also carries the source's audio and metadata, so the AI stage that follows copies both;
* the frame rate is always set explicitly at the end (QTGMC's own arithmetic yields 19001/317);
* no sharpening is emitted before the AI stage (Doom9: sharpening before AI enhancement compounds halos),
  and one owner for denoising (measured: classical degraining before the AI model gave no gain).

Rendering uses the system ffmpeg (`-f avisynth`).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable

from .. import config, proc
from . import AdapterResult, safe_output_dir

PLUGINS = config.TOOLS_DIR / "avisynth" / "plugins"
INTERMEDIATE = ["-c:v", "ffv1", "-level", "3", "-slices", "16", "-slicecrc", "1", "-pix_fmt", "yuv420p10le"]


# ----------------------------------------------------------------------------- environment
def system_ffmpeg() -> str | None:
    """An ffmpeg that can open .avs (the system one; AviSynth.dll comes from System32)."""
    exe = shutil.which("ffmpeg")
    if not exe:
        return None
    r = subprocess.run([exe, "-hide_banner", "-demuxers"], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    return exe if re.search(r"^\s*D\s+avisynth\b", r.stdout, re.M) else None


def available() -> tuple[bool, str]:
    if not system_ffmpeg():
        return False, "no ffmpeg with the AviSynth demuxer on PATH"
    if not (PLUGINS / "QTGMC.avsi").exists():
        return False, "private plugin set missing: run python -m vidimg.avs_install"
    if not Path(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "AviSynth.dll").exists():
        return False, "AviSynth+ is not installed (System32\\AviSynth.dll)"
    return True, "ok"


def avs_str(value: str | Path) -> str:
    """An AviSynth string literal. Plain strings cannot hold '"', triple-quoted ones can."""
    s = str(value)
    if '"""' in s:
        raise ValueError(f"cannot quote {s!r} for AviSynth")
    return f'"""{s}"""' if '"' in s else f'"{s}"'


def _rate(path: Path) -> tuple[Fraction, int, dict]:
    ff = shutil.which("ffprobe") or "ffprobe"
    r = subprocess.run([ff, "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                        "stream=r_frame_rate,avg_frame_rate,nb_read_packets,field_order,pix_fmt,color_range,color_space,"
                        "color_transfer,color_primaries", "-of", "json", str(path)],
                       capture_output=True, text=True, stdin=subprocess.DEVNULL)
    s = json.loads(r.stdout or "{}").get("streams", [{}])[0]
    fps = Fraction(s.get("r_frame_rate") or "0/1")
    avg = Fraction(s.get("avg_frame_rate") or "0/1")
    # phone clips are often VFR with an r_frame_rate far from the real cadence: trust avg then
    if avg and (not fps or abs(float(fps) - float(avg)) > 0.5):
        fps = avg.limit_denominator(1001)
    if str(s.get("pix_fmt", "")).startswith("yuvj"):
        s["color_range"] = "pc"               # JPEG-style full range (phone/webcam/MJPEG clips)
    return fps, int(s.get("nb_read_packets") or 0), s


def is_full_range(colour: dict) -> bool:
    return colour.get("color_range") in ("pc", "jpeg", "full")


# ----------------------------------------------------------------------------- stages
@dataclass
class Deinterlace:
    """True interlace -> double rate progressive. QTGMC (Doom9 reference) or BWDIF (fast)."""

    tff: bool = True
    method: str = "qtgmc"                  # qtgmc | bwdif
    preset: str = "Slower"
    source_match: int = 3                  # 3 + lossless 2 = the community's maximum-fidelity setting
    lossless: int = 2
    sharpness: float = 0.2                 # QTGMC default 1.0 oversharpens motion (Vit)
    tr2: int = 1                           # 1 clean, 2-3 noisy/shimmering sources
    ez_denoise: float = 0.0                # QTGMC's own noise stage (0 = off, keep grain)
    single_rate: bool = False              # FPSDivisor=2

    def rate(self) -> Fraction:
        return Fraction(1) if self.single_rate else Fraction(2)

    def lines(self) -> list[str]:
        order = "AssumeTFF()" if self.tff else "AssumeBFF()"
        if self.method == "bwdif":
            return [order, f"BWDIF(field={-2 if not self.single_rate else -1})"]
        if self.method != "qtgmc":
            raise ValueError(f"unknown deinterlacer {self.method!r}")
        args = [f'Preset="{self.preset}"', f"TR2={self.tr2}", f"Sharpness={self.sharpness:g}"]
        if self.source_match:
            args += [f"SourceMatch={self.source_match}", f"Lossless={self.lossless}"]
        if self.ez_denoise:
            args += [f"EZDenoise={self.ez_denoise:g}", 'NoisePreset="Slow"']
        if self.single_rate:
            args.append("FPSDivisor=2")
        return [order, f"QTGMC({', '.join(args)})"]


@dataclass
class IVTC:
    """Hard telecine: field matching (lossless) + decimation, with QTGMC for leftover combed frames."""

    tff: bool = True
    cycle: int = 5                         # 3:2 NTSC film; 8mm at 16/18 fps uses cycle 3 / cycleR 2 of 5
    cycle_r: int = 1
    mode: int = 1
    pp: int = 6

    def rate(self) -> Fraction:
        return Fraction(self.cycle - self.cycle_r, self.cycle)

    def lines(self) -> list[str]:
        order = "AssumeTFF()" if self.tff else "AssumeBFF()"
        return [order,
                "__deint = QTGMC(Preset=\"Slower\", SourceMatch=3, Lossless=2, Sharpness=0.2, FPSDivisor=2)",
                f"TFM(order={1 if self.tff else 0}, mode={self.mode}, PP={self.pp}, clip2=__deint)",
                f"TDecimate(mode=1, cycleR={self.cycle_r}, cycle={self.cycle})"]


@dataclass
class Deblock:
    """MPEG block removal on the untouched 8x8 grid (before any crop/resize/denoise)."""

    quant1: int = 24
    quant2: int = 26

    def rate(self) -> Fraction:
        return Fraction(1)

    def lines(self) -> list[str]:
        return [f"Deblock_QED(quant1={self.quant1}, quant2={self.quant2})"]


@dataclass
class Degrain:
    """Motion-compensated temporal denoise (SMDegrain). Never sharpens (contrasharp=false)."""

    tr: int = 2
    thsad: int = 250
    prefilter: int = 1                     # MinBlur
    refine: bool = True
    chroma: bool = True

    def rate(self) -> Fraction:
        return Fraction(1)

    def lines(self) -> list[str]:
        return [f"SMDegrain(tr={self.tr}, thSAD={self.thsad}, RefineMotion={str(self.refine).lower()}, "
                f"prefilter={self.prefilter}, contrasharp=false, chroma={str(self.chroma).lower()}"
                f"{'' if self.chroma else ', plane=0'})"]


@dataclass
class ChromaDenoise:
    """Spatial chroma clean-up on the GPU (VHS/phone chroma noise); luma untouched."""

    h: float = 1.0

    def rate(self) -> Fraction:
        return Fraction(1)

    def lines(self) -> list[str]:
        return [f'KNLMeansCL(d=1, a=2, h={self.h:g}, channels="UV", device_type="gpu")']


@dataclass
class Repair:
    """QTGMC progressive repair (InputType=1/2): shimmer from bad deinterlacing or AI flicker."""

    input_type: int = 1

    def rate(self) -> Fraction:
        return Fraction(1)

    def lines(self) -> list[str]:
        return [f'QTGMC(Preset="Slower", InputType={self.input_type}, Sharpness=0.0, TR2=1)']


AvsStage = Deinterlace | IVTC | Deblock | Degrain | ChromaDenoise | Repair
# processing order from the Doom9 consensus (04_doom9_practice.md 1.1). Deblock runs on the original
# grid; for interlaced input, field handling comes first (Deblock_QED needs progressive frames).
_ORDER = {Deblock: 1, Deinterlace: 2, IVTC: 2, ChromaDenoise: 3, Degrain: 4, Repair: 5}


def order(stages: list[AvsStage]) -> list[AvsStage]:
    field_stage = any(isinstance(s, (Deinterlace, IVTC)) for s in stages)
    key = dict(_ORDER)
    if field_stage:
        key[Deblock] = 2.5
    return sorted(stages, key=lambda s: key[type(s)])


# ----------------------------------------------------------------------------- script
@dataclass
class Script:
    src: Path
    stages: list[AvsStage]
    cache: Path                             # .lwi index location (workspace)
    fps: Fraction
    threads: int = field(default_factory=lambda: max(2, min(16, (os.cpu_count() or 8) // 2)))
    full_range: bool = False                # normalise to limited range (lossless at 10-bit: 256 -> 876 levels)

    def out_fps(self) -> Fraction:
        f = self.fps
        for s in self.stages:
            f *= s.rate()
        return f

    def text(self) -> str:
        num, den = self.fps.numerator, self.fps.denominator
        out = self.out_fps()
        body = [
            "# generated by VidIMG (vidimg/adapters/avisynth.py); private pinned plugin set only",
            "ClearAutoloadDirs()",
            f"AddAutoloadDir({avs_str(PLUGINS)})",
            f"LWLibavVideoSource({avs_str(self.src)}, cachefile={avs_str(self.cache)}, fpsnum={num}, fpsden={den})",
            "ConvertBits(16)",
        ]
        if self.full_range:
            # untagged NUT + full-range data had the AI stage's HEVC path squeeze the levels (luma 0-237 became
            # 3-220; preview 41.4 dB vs full run 29.1 dB): everything downstream works in limited range
            body.append('ColorYUV(levels="PC->TV")')
        for s in order(self.stages):
            body += s.lines()
        body += ["ConvertBits(10, dither=1)", f"AssumeFPS({out.numerator}, {out.denominator})",
                 f"Prefetch({self.threads})"]
        return "\n".join(body) + "\n"


_SHARPENERS = re.compile(r"\b(LSFmod|LSFplus|CAS|Sharpen|UnsharpMask|aWarpSharp2?|ex_ContraSharpening)\s*\(", re.I)


def lint(text: str) -> list[str]:
    """Rules from 02_language_reference.md section 'lint' that matter for generated scripts."""
    problems = []
    code = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if len(code) < 2 or code[0] != "ClearAutoloadDirs()" or not code[1].startswith("AddAutoloadDir("):
        problems.append("the first statements must be ClearAutoloadDirs() then AddAutoloadDir(<private plugins>)")
    if any(re.match(r"(LoadPlugin|Import)\s*\(", ln) for ln in code):
        problems.append("no LoadPlugin/Import: plugins come only from the private autoload folder")
    if any("DirectShowSource" in ln for ln in code):
        problems.append("DirectShowSource is not allowed (non-deterministic decoder)")
    pf = [i for i, ln in enumerate(code) if ln.startswith("Prefetch(")]
    if len(pf) != 1 or pf[0] != len(code) - 1:
        problems.append("exactly one Prefetch(), as the last statement")
    if any(re.match(r"return\b", ln) for ln in code):
        problems.append("no top-level return (code after it, including Prefetch, never runs)")
    if any(_SHARPENERS.search(ln) for ln in code) or re.search(r"contrasharp\s*=\s*(true|[1-9])", text, re.I):
        problems.append("no sharpening before the AI stage (halos compound; plastic look)")
    m = re.search(r"Sharpness\s*=\s*([0-9.]+)", text)
    if m and float(m.group(1)) > 0.5:
        problems.append("QTGMC Sharpness above 0.5 oversharpens motion")
    if not any(ln.startswith("AssumeFPS(") for ln in code):
        problems.append("the output frame rate must be set explicitly")
    if "cachefile=" not in text:
        problems.append("L-SMASH must index into the workspace (cachefile=), not next to the original")
    if code and not any(ln.startswith("ConvertBits(10") for ln in code):
        problems.append("output must be dithered to 10-bit")
    return problems


# ----------------------------------------------------------------------------- render
def render(src: Path, dst: Path, stages: list[AvsStage], *, work: Path | None = None,
           start: float | None = None, duration: float | None = None, timeout: float | None = None,
           on_progress: Callable[[int, int], None] | None = None) -> AdapterResult:
    """Run the stages and write a lossless 10-bit FFV1 .nut with the source's audio and metadata."""
    ok, why = available()
    if not ok:
        return AdapterResult(False, error=why)
    src, dst = Path(src).resolve(), Path(dst).resolve()
    if dst.suffix.lower() != ".nut":
        # NUT keeps the exact rational frame rate (Matroska's 1 ms grid turns 60000/1001 into 19001/317)
        raise ValueError("the AviSynth intermediate is FFV1 in .nut")
    safe_output_dir(dst.parent, src)
    work = Path(work or dst.parent).resolve()
    work.mkdir(parents=True, exist_ok=True)
    fps, packets, colour = _rate(src)
    if not fps:
        return AdapterResult(False, error="could not read the source frame rate")
    script = Script(src, stages, work / (dst.stem + ".lwi"), fps, full_range=is_full_range(colour))
    if script.full_range:
        colour = {**colour, "color_range": "tv"}
    text = script.text()
    problems = lint(text)
    if problems:
        return AdapterResult(False, error="script lint: " + "; ".join(problems), details={"script": text})
    avs = work / (dst.stem + ".avs")
    avs.write_text(text, encoding="utf-8")

    out_fps = script.out_fps()
    seek = []
    if start is not None:
        seek += ["-ss", f"{start:g}"]
    if duration is not None:
        seek += ["-t", f"{duration:g}"]
    # colour tags travel in the sidecar (NUT cannot store them); passing them to ffmpeg as output options
    # only risks an inserted conversion (measured on the Y4M path of the VapourSynth backend)
    args = [system_ffmpeg(), "-hide_banner", "-nostdin", "-y", *seek, "-f", "avisynth", "-i", str(avs),
            *seek, "-i", str(src), "-map", "0:v:0", "-map", "1:a?", "-c:a", "copy", "-map_metadata", "1",
            *INTERMEDIATE, "-metadata", "comment=VidIMG AviSynth+ pre-pass: " +
            " | ".join(ln for s in order(stages) for ln in s.lines() if not ln.startswith("__")), str(dst)]
    expected = round(packets * float(out_fps / fps)) if packets else 0
    if duration is not None:
        expected = round(duration * float(out_fps))

    def line(t: str) -> None:
        m = re.search(r"frame=\s*(\d+)", t)
        if m and on_progress:
            on_progress(int(m.group(1)), expected)

    r = proc.run("avisynth", args, on_line=line, timeout=timeout)
    details = {"script": text, "avs": str(avs), "fps_in": str(fps), "fps_out": str(out_fps), "expected_frames": expected}
    if r.returncode != 0 or not dst.exists():
        err = next((ln for ln in reversed(r.stdout_tail) if "rror" in ln or "avisynth" in ln.lower()), None)
        return AdapterResult(False, error=err or f"ffmpeg exit {r.returncode}", run=r, details=details)
    from .. import media
    details["colour"] = media.write_colour_sidecar(dst, colour)
    got = _rate(dst)[1]
    details["frames"] = got
    if expected and abs(got - expected) > max(2, expected * 0.002):
        return AdapterResult(False, [dst], error=f"frame count {got} != expected {expected}", run=r, details=details)
    return AdapterResult(True, [dst], run=r, details=details)
