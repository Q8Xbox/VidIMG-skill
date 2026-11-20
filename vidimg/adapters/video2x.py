"""Video2X 6.4.0 adapter (PLAN.md section 3.5). Run as a separate process (AGPL-3.0).

Role: RIFE interpolation (fast, original frames bit-exact, 43.6 dB in the benchmark), anime and
cartoon content (Real-CUGAN, Real-ESRGAN anime) and RIFE frame interpolation.
Real-ESRGAN / Real-CUGAN destroyed real faces in the benchmark (identity 0.77-0.81), so those
processors are refused unless the caller declares the content as anime/cartoon.
"""

from __future__ import annotations

from pathlib import Path

from .. import config, proc
from . import AdapterResult, PolicyError, safe_output_dir

REALESRGAN_MODELS = ("realesr-animevideov3", "realesrgan-plus-anime", "realesrgan-plus")  # generalv3 rejected by 6.4.0
REALCUGAN_MODELS = ("models-se", "models-pro", "models-nose")
RIFE_MODELS = ("rife-v4.26", "rife-v4.25", "rife-v4.25-lite", "rife-v4.6", "rife-anime", "rife-UHD")


def build_args(src: Path, dst: Path, processor: str, *, content: str, scale: int | None = None,
               noise: int | None = None, model: str | None = None, fps_multiplier: int | None = None,
               scene_threshold: float | None = None, codec: str = "libx265", extra: dict[str, str] | None = None,
               device: int = 0, pix_fmt: str | None = None, out_width: int | None = None,
               out_height: int | None = None, max_b_frames: int | None = None) -> list[str]:
    if processor not in ("realesrgan", "realcugan", "libplacebo", "rife"):
        raise ValueError(f"unknown processor {processor!r}")
    if processor in config.VIDEO2X_ANIME_ONLY and content != "anime":
        raise PolicyError(f"{processor} is only allowed for anime/cartoon content (it distorted real faces in testing)")
    args = ["-i", str(src), "-o", str(dst), "-p", processor, "-d", str(device), "-c", codec, "--no-progress"]
    if pix_fmt:
        args += ["--pix-fmt", pix_fmt]
    if max_b_frames is not None:          # Video2X's own option; "-e bf=0" is overridden by its default (-1)
        args += ["--max-b-frames", str(max_b_frames)]
    if processor == "rife":
        args += ["-m", str(fps_multiplier or 2), "--rife-model", model or "rife-v4.26"]
        if scene_threshold is not None:
            args += ["-t", f"{scene_threshold:g}"]
    else:
        args += ["-s", str(scale or 2)]
        if processor == "realesrgan":
            m = model or "realesr-animevideov3"
            if m not in REALESRGAN_MODELS:
                raise ValueError(f"Real-ESRGAN model {m!r} not accepted by Video2X 6.4.0; use one of {REALESRGAN_MODELS}")
            args += ["--realesrgan-model", m]
        elif processor == "realcugan":
            # 6.4.0 rejects -1 in any form (observed: "-n -1" and "--noise-level=-1" both invalid);
            # omitting the flag selects the no-denoise default, so it is passed only for 0..3
            args += ["--realcugan-model", model or "models-se"]
            if noise is not None and noise >= 0:
                args += [f"--noise-level={noise}"]
        elif processor == "libplacebo":
            # Video2X 6.4.0 requires the output size for libplacebo ("Output width and height must be set")
            if not (out_width and out_height):
                raise ValueError("libplacebo needs out_width/out_height")
            args += ["--libplacebo-shader", model or "anime4k-v4-a", "-w", str(out_width), "-h", str(out_height)]
    if extra is None:  # libx265/libx264 quality default; lossless codecs pass extra={}
        extra = {"crf": "16"} if codec in ("libx265", "libx264") else {}
    for k, v in extra.items():
        args += ["-e", f"{k}={v}"]
    return args


def run(src: Path, dst: Path, processor: str, *, content: str, timeout: float | None = None, **kw) -> AdapterResult:
    src, dst = Path(src), Path(dst)
    safe_output_dir(dst.parent, src)
    if processor == "libplacebo" and not (kw.get("out_width") and kw.get("out_height")):
        from .. import media
        info = media.probe(src)
        s = kw.get("scale") or 2
        kw = {**kw, "out_width": int(info["width"] * s) // 2 * 2, "out_height": int(info["height"] * s) // 2 * 2}
    try:
        args = build_args(src, dst, processor, content=content, **kw)
    except (ValueError, PolicyError) as e:
        return AdapterResult(False, error=str(e))
    if not config.VIDEO2X_EXE.exists():
        return AdapterResult(False, error=f"Video2X not found at {config.VIDEO2X_EXE}")
    if dst.exists():
        dst.unlink()
    r = proc.run("video2x", [config.VIDEO2X_EXE, *args], timeout=timeout)
    # Video2X 6.4.0 can crash on exit (0xC0000005) after writing a complete file: judge by its
    # own success line plus the file.
    finished = "Video processed successfully" in r.text()
    if dst.exists() and dst.stat().st_size > 0 and (r.returncode == 0 or finished):
        return AdapterResult(True, [dst], run=r, details={"exit_code": r.returncode,
                                                         "crash_after_save": r.returncode != 0})
    err = next((ln for ln in reversed(r.stdout_tail) if "error" in ln.lower() or "critical" in ln.lower()), None)
    return AdapterResult(False, error=err or f"video2x exit {r.returncode}", run=r)


# ----------------------------------------------------------------------------- anime routing (pipelines)
from dataclasses import dataclass as _dataclass  # noqa: E402


@_dataclass
class AnimeSR:
    """Anime / cartoon enhancement stage for the video pipeline (free: Video2X, Vulkan). Never used on
    real faces: the probe only flags anime when no real face is detected, and the adapter refuses these
    processors unless content="anime"."""

    processor: str = "realesrgan"
    model: str = "realesr-animevideov3"
    scale: int = 2

    def describe(self) -> str:
        return f"video2x:{self.processor}:{self.model}:x{self.scale}"


ENCODER_TO_CODEC = {"ffv1": ("ffv1", None, {}),
                    "lossless": ("libx264", "yuv420p10le", {"qp": "0", "preset": "ultrafast"}),
                    "hevc10": ("libx265", "yuv420p10le", {"crf": "16", "preset": "slow", "x265-params": "bframes=0"}),
                    "h264": ("libx264", "yuv420p", {"crf": "16", "bf": "0"}), "prores": ("prores_ks", "yuv422p10le", {})}


# Video2X 6.4.0's own FFmpeg has NVENC: 10-bit HEVC on the GPU (measured 7.0 s vs 9.6 s with libx265 slow on a
# 120-frame 576p clip, and the CPU stays free for a 4K run); libx265/libx264 without an NVIDIA GPU
# bf=0: no B-frames, so packet order = display order and broken timestamps can be re-stamped without re-encoding
NVENC_CODEC = {"hevc10": ("hevc_nvenc", "p010le", {"preset": "p7", "tune": "hq", "rc": "constqp", "qp": "16"}),
               "lossless": ("hevc_nvenc", "p010le", {"preset": "p4", "tune": "lossless"}),
               "h264": ("h264_nvenc", "yuv420p", {"preset": "p7", "tune": "hq", "rc": "constqp", "qp": "16"})}


def run_stage(src: Path, dst: Path, st: AnimeSR, encoder: str = "hevc10", **_) -> AdapterResult:
    from .. import media
    codec, pix, extra = (NVENC_CODEC[encoder] if encoder in NVENC_CODEC and media.nvenc()
                         else ENCODER_TO_CODEC.get(encoder, ENCODER_TO_CODEC["hevc10"]))
    feed, tmp = _timestamp_safe_input(src, dst)
    r = run(feed, dst, st.processor, content="anime", scale=st.scale, model=st.model, codec=codec, pix_fmt=pix,
            extra=dict(extra))
    if tmp is not None:
        tmp.unlink(missing_ok=True)
    if r.ok:
        notes = [n for n in (_fix_timestamps(src, dst), _restore_tail(src, dst, codec, pix, extra)) if n]
        if notes:
            r.details = {**(r.details or {}), "repairs": notes}
        # a file that ends early is not a result: Video2X can stop short and still leave a playable file, which
        # then went through the texture pass and the final encode before the guard's frame count refused it
        want, have = media.probe(src, exact=True)["frames"], media.probe(dst, exact=True)["frames"]
        if want and want - have > TAIL_MAX:
            return AdapterResult(False, error=f"Video2X output is incomplete: {have} of {want} frames", run=r.run,
                                 details=r.details)
    return r


TAIL_MAX = 4        # frames Video2X is known to drop at the very end (2 measured); more than this is a cut-off file


# decoders that hand Video2X 6.4.0 frames without usable timestamps: its output then had every timestamp at 0
# (owner's Sample03, MPEG-4 Part 2 in AVI; the same stream remuxed to MKV too). NVENC in Video2X cannot run without
# B-frames (--max-b-frames 0 -> "EncodePicture failed: invalid param"), so re-stamping its output is not an option.
NO_PTS_CODECS = {"mpeg4", "msmpeg4v1", "msmpeg4v2", "msmpeg4v3", "h263", "h263p", "wmv1", "wmv2", "wmv3", "flv1"}


def _timestamp_safe_input(src: Path, dst: Path) -> tuple[Path, Path | None]:
    """For such sources, a lossless FFV1 intermediate (video + audio copy) next to the output; Video2X reads that and
    keeps correct timestamps (measured). Returns (input for Video2X, temporary file to delete or None)."""
    import subprocess
    from .. import media
    info = media.probe(src)
    if info["codec"] not in NO_PTS_CODECS and src.suffix.lower() != ".avi":
        return src, None
    tmp = dst.parent / ".video2x_input" / (dst.stem + ".ffv1in.mkv")   # not in the output folder itself (safety rule)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    cp = subprocess.run([media.ffmpeg(), "-v", "error", "-y", "-i", str(src), "-map", "0:v:0", "-map", "0:a?",
                         "-c:v", "ffv1", "-level", "3", "-c:a", "copy", str(tmp)],
                        capture_output=True, stdin=subprocess.DEVNULL)
    if cp.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        return src, None
    return tmp, tmp


def _fix_timestamps(src: Path, dst: Path) -> str | None:
    """Video2X 6.4.0 writes every video timestamp as 0 for some inputs (owner's Sample03: MPEG-4 Part 2 in AVI, and
    the same stream remuxed to MKV), so players and every time-based tool mis-handle the file. Video2X processes
    frame by frame at the source rate, so the true time of frame N is N / rate: re-stamp without re-encoding
    (our encodes use no B-frames, so packet order is display order)."""
    import subprocess
    from fractions import Fraction
    from .. import media
    from . import avisynth as _avs
    pts = subprocess.run([media.ffprobe(), "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
                          "-read_intervals", "%+#40", "-of", "csv=p=0", str(dst)],
                         capture_output=True, text=True, stdin=subprocess.DEVNULL).stdout.split()
    vals = [float(x) for x in pts if x not in ("N/A", "")]
    if len(vals) >= 3 and len(set(vals)) == len(vals) and all(b > a for a, b in zip(sorted(vals), sorted(vals)[1:])):
        return None                                          # sane: distinct, increasing
    bfr = subprocess.run([media.ffprobe(), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=has_b_frames",
                          "-of", "csv=p=0", str(dst)], capture_output=True, text=True, stdin=subprocess.DEVNULL).stdout.strip()
    if bfr not in ("", "0"):
        # packet order is not display order: N/rate stamps would shuffle frames (happened to Sample03 before
        # --max-b-frames 0); leave it broken so the guard's timestamps gate rejects it
        return "video timestamps are broken and the stream has B-frames: not re-stamped (guard will reject)"
    rate = _avs._rate(src)[0] or Fraction(round(media.probe(src)["fps"] * 1000), 1000)
    num, den = rate.numerator, rate.denominator
    tmp = dst.with_name(dst.stem + ".ts" + dst.suffix)
    expr = f"N*{den}/({num}*TB)"
    cp = subprocess.run([media.ffmpeg(), "-v", "error", "-y", "-i", str(dst), "-map", "0", "-c", "copy",
                         "-bsf:v", f"setts=pts={expr}:dts={expr}", str(tmp)],
                        capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if cp.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        return "video timestamps are broken and could not be re-stamped: " + cp.stderr.strip()[-200:]
    tmp.replace(dst)
    return f"video timestamps re-stamped at {num}/{den} fps (Video2X wrote them as 0)"


def _restore_tail(src: Path, dst: Path, codec: str, pix: str | None, extra: dict) -> str | None:
    """Video2X 6.4.0 drops the LAST 2 frames of every run (398->396, 600->598, 40->38 measured; the others stay
    aligned one-to-one). Append clones of the last output frame, encoded with the same settings, by stream-copy
    concat: frame count, duration and the audio alignment at the end are kept without re-encoding the output."""
    import subprocess
    from .. import media
    want, have = media.probe(src, exact=True)["frames"], media.probe(dst, exact=True)
    missing = want - have["frames"]
    if not 0 < missing <= TAIL_MAX:
        return None
    # the SOURCE's exact rate: Video2X's own output rate fields can be wrong (b001: avg 15.98 fps for 29.97 content,
    # which placed the rebuilt tail at 2706 s in a 1443 s video)
    from fractions import Fraction
    from . import avisynth as _avs
    fr = _avs._rate(src)[0] or Fraction(round(media.probe(src)["fps"] * 1000), 1000)
    rate = f"{fr.numerator}/{fr.denominator}"
    work = dst.with_name(dst.stem + ".tail")
    work.mkdir(exist_ok=True)
    tail, vonly, joined = work / f"tail{dst.suffix}", work / f"video{dst.suffix}", work / f"joined{dst.suffix}"
    enc = ["-c:v", codec] + (["-pix_fmt", pix] if pix else []) + [x for k, v in extra.items() for x in (f"-{k}", v)]
    if codec == "hevc_nvenc":
        enc += ["-profile:v", "main10"]
    enc += ["-bf", "0"]
    tb = subprocess.run([media.ffprobe(), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=time_base",
                         "-of", "csv=p=0", str(dst)], capture_output=True, text=True, stdin=subprocess.DEVNULL).stdout.strip()
    if dst.suffix.lower() in (".mp4", ".mov", ".m4v") and "/" in tb:
        # the tail must share the main file's time base: concat copies timestamps without rescaling them
        enc += ["-video_track_timescale", tb.split("/")[1]]
    ff = media.ffmpeg()
    steps = [
        [ff, "-v", "error", "-y", "-sseof", "-1", "-i", str(dst), "-map", "0:v",
         "-vf", f"reverse,select='eq(n\\,0)',loop={missing - 1}:1:0,setpts=N/({rate})/TB", "-frames:v", str(missing), "-r", rate,
         *enc, str(tail)],
        [ff, "-v", "error", "-y", "-i", str(dst), "-map", "0:v", "-c", "copy", str(vonly)],
    ]
    for args in steps:
        if subprocess.run(args, capture_output=True, stdin=subprocess.DEVNULL).returncode != 0:
            return f"{missing} frame(s) missing at the end; tail could not be rebuilt"
    lst = work / "concat.txt"
    # explicit duration for the first part: the concat demuxer otherwise offsets the tail by the container's
    # duration, which Video2X writes wrong
    first = have["frames"] * fr.denominator / fr.numerator
    lst.write_text(f"file '{vonly.as_posix()}'\nduration {first:.6f}\nfile '{tail.as_posix()}'\n", encoding="utf-8")
    cp = subprocess.run([ff, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-i", str(dst),
                         "-map", "0:v", "-map", "1:a?", "-map", "1:s?", "-map", "1:t?", "-c", "copy", str(joined)],
                        capture_output=True, stdin=subprocess.DEVNULL)
    if cp.returncode != 0 or media.probe(joined, exact=True)["frames"] != want:
        return f"{missing} frame(s) missing at the end; tail join failed"
    # no B-frames in our encodes: even timestamps N/rate are exact, whatever the parts carried
    bfr = subprocess.run([media.ffprobe(), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=has_b_frames",
                          "-of", "csv=p=0", str(joined)], capture_output=True, text=True, stdin=subprocess.DEVNULL).stdout.strip()
    if bfr == "0":
        expr = f"N*{fr.denominator}/({fr.numerator}*TB)"
        even = work / f"even{dst.suffix}"
        cp2 = subprocess.run([ff, "-v", "error", "-y", "-i", str(joined), "-map", "0", "-c", "copy",
                              "-bsf:v", f"setts=pts={expr}:dts={expr}", str(even)], capture_output=True, stdin=subprocess.DEVNULL)
        if cp2.returncode == 0 and even.exists():
            even.replace(joined)
    vend = subprocess.run([media.ffprobe(), "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
                           "-read_intervals", "99999999%", "-of", "csv=p=0", str(joined)],
                          capture_output=True, text=True, stdin=subprocess.DEVNULL).stdout.split()
    last = max((float(x) for x in vend if x not in ("N/A", "")), default=0.0)
    expect = (want - 1) * fr.denominator / fr.numerator
    if last and abs(last - expect) > 1.0:
        return f"{missing} frame(s) missing at the end; tail join placed the last frame at {last:.1f} s (expected {expect:.1f})"
    joined.replace(dst)
    for f in work.iterdir():
        f.unlink()
    work.rmdir()
    return f"restored {missing} trailing frame(s) Video2X dropped"
