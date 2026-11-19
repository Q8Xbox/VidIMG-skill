"""VapourSynth worker: runs INSIDE the isolated environment (tools/vs-venv, core R72 pinned) and writes
Y4M to stdout. Standalone on purpose (stdlib + vapoursynth only): the main tool's Python never imports
VapourSynth, so the two runtimes cannot conflict.

    tools/vs-venv/Scripts/python -I vidimg/vs_runner.py spec.json  > out.y4m

spec: {"src", "cache", "fps": [num, den], "full_range", "plugins", "stages": [{"kind": ...}, ...],
       "trim": [first, count] (optional: a span of the source, for short trials)}
Same stage parameters as vidimg/adapters/avisynth.py; parity measured in research/avisynth/05 section 9.

sRestore must see every frame, in order: its decisions carry state from one frame to the next. havsfunc
forced that with Cache(make_linear=True), which core R72 no longer has (it passes the clip through with a
warning), so its own final decimation skipped frames and reset the state constantly (measured: 15 % of the
frames still blended, 13 % below 30 dB against the clean original; evaluated in order: 0 % and 0 %). Here
the full-rate result is evaluated frame by frame and the decimation is done while writing.
"""

from __future__ import annotations

import json
import math
import sys
from fractions import Fraction

try:
    import vapoursynth as vs
    core = vs.core
except ImportError:          # imported by the main tool's tests for selection() only
    vs = core = None


def _srestore_full_rate(havsfunc, c: vs.VideoNode, frate: Fraction) -> vs.VideoNode:
    """havsfunc.srestore without its final ChangeFPS: one output frame per input frame (the chosen picture)."""
    real = havsfunc.ChangeFPS
    havsfunc.ChangeFPS = lambda clip, num, den=1: clip
    try:
        return havsfunc.srestore(c, frate=float(frate), omode=6, mode=2)
    finally:
        havsfunc.ChangeFPS = real


def build(spec: dict) -> tuple[vs.VideoNode, Fraction | None]:
    """The processed clip, and the target rate when it still has to be decimated linearly (sRestore)."""
    core.std.LoadAllPlugins(spec["plugins"])
    import havsfunc  # needs the plugins loaded first

    num, den = spec["fps"]
    c = core.lsmas.LWLibavSource(spec["src"], cachefile=spec["cache"], fpsnum=num, fpsden=den)
    if spec.get("trim"):
        first, count = spec["trim"]
        c = c[first: first + count]
    rate_num, rate_den = num, den
    if spec.get("full_range"):
        c = core.resize.Point(c, format=vs.YUV420P16, range_in_s="full", range_s="limited")
    else:
        c = core.resize.Point(c, format=vs.YUV420P16)
    linear = None
    for st in spec["stages"]:
        k = st["kind"]
        if linear is not None:
            raise SystemExit("sRestore must be the last stage")
        if k == "deinterlace":
            if st["method"] == "bwdif":
                c = core.bwdif.Bwdif(c, field=3 if st["tff"] else 2) if not st["single_rate"] else \
                    core.bwdif.Bwdif(c, field=1 if st["tff"] else 0)
            else:
                kw = dict(Preset=st["preset"], TR2=st["tr2"], Sharpness=st["sharpness"], TFF=st["tff"])
                if st["source_match"]:
                    kw.update(SourceMatch=st["source_match"], Lossless=st["lossless"])
                if st["ez_denoise"]:
                    kw.update(EZDenoise=st["ez_denoise"], NoisePreset="Slow")
                if st["single_rate"]:
                    kw.update(FPSDivisor=2)
                c = havsfunc.QTGMC(c, **kw)
            if not st["single_rate"]:
                rate_num *= 2
        elif k == "ivtc":
            c8 = core.resize.Point(c, format=vs.YUV420P8)       # the TIVTC port is 8-bit
            deint = havsfunc.QTGMC(c8, Preset="Slower", SourceMatch=3, Lossless=2, Sharpness=0.2,
                                   TFF=st["tff"], FPSDivisor=2)
            c8 = core.tivtc.TFM(c8, order=1 if st["tff"] else 0, mode=st["mode"], PP=st["pp"], clip2=deint)
            c = core.resize.Point(core.tivtc.TDecimate(c8, mode=1, cycleR=st["cycle_r"], cycle=st["cycle"]),
                                  format=vs.YUV420P16)
            rate_num *= st["cycle"] - st["cycle_r"]
            rate_den *= st["cycle"]
        elif k == "chroma":
            # move each colour plane by (dx, dy) luma px: sample from -d / subsampling (Spline36, same size)
            sw, sh = 1 << c.format.subsampling_w, 1 << c.format.subsampling_h
            planes = [core.std.ShufflePlanes(c, i, vs.GRAY) for i in range(3)]
            for i, (dx, dy) in ((1, (st["u_dx"], st["u_dy"])), (2, (st["v_dx"], st["v_dy"]))):
                if dx or dy:
                    planes[i] = core.resize.Spline36(planes[i], src_left=-dx / sw, src_top=-dy / sh)
            c = core.std.ShufflePlanes(planes, [0, 0, 0], vs.YUV)
        elif k == "srestore":
            c = core.std.AssumeFPS(c, fpsnum=rate_num, fpsden=rate_den)
            linear = Fraction(st["frate"])
            c = _srestore_full_rate(havsfunc, c, linear)
        else:
            raise SystemExit(f"stage {k!r} is not implemented in the VapourSynth backend")
    c = core.resize.Point(c, format=vs.YUV420P10, dither_type="error_diffusion")
    return core.std.AssumeFPS(c, fpsnum=rate_num, fpsden=rate_den), linear


def selection(n_in: int, in_fps: Fraction, out_fps: Fraction) -> dict[int, int]:
    """Input frame -> how many times it is written (ChangeFPS's mapping: output k shows floor(k / factor))."""
    factor = out_fps / in_fps
    picks: dict[int, int] = {}
    for k in range(math.floor(n_in * factor)):
        n = math.floor(k / factor)
        picks[n] = picks.get(n, 0) + 1
    return picks


def write_linear(clip: vs.VideoNode, out_fps: Fraction, out) -> int:
    """Evaluate every frame in order (sRestore's state machine) and write the selected ones as Y4M 10-bit."""
    import numpy as np

    picks = selection(clip.num_frames, Fraction(clip.fps_num, clip.fps_den), out_fps)
    out.write(f"YUV4MPEG2 W{clip.width} H{clip.height} F{out_fps.numerator}:{out_fps.denominator} Ip A1:1 "
              f"C420p10 XYSCSS=420P10\n".encode())
    written = 0
    for n in range(clip.num_frames):
        f = clip.get_frame(n)
        reps = picks.get(n, 0)
        if reps:
            data = b"".join(np.asarray(f[p]).astype("<u2").tobytes() for p in range(3))
            for _ in range(reps):
                out.write(b"FRAME\n" + data)
            written += reps
    out.flush()
    return written


if __name__ == "__main__":
    spec = json.load(open(sys.argv[1], encoding="utf-8"))
    clip, linear = build(spec)
    if linear is None:
        clip.output(sys.stdout.buffer, y4m=True)
    else:
        n = write_linear(clip, linear, sys.stdout.buffer)
        sys.stderr.write(json.dumps({"srestore": {"in": clip.num_frames, "out": n, "fps": str(linear)}}) + "\n")
