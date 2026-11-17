"""VapourSynth worker: runs INSIDE the isolated environment (tools/vs-venv, core R72 pinned) and writes
Y4M to stdout. Standalone on purpose (stdlib + vapoursynth only): the main tool's Python never imports
VapourSynth, so the two runtimes cannot conflict.

    tools/vs-venv/Scripts/python -I vidimg/vs_runner.py spec.json  > out.y4m

spec: {"src", "cache", "fps": [num, den], "full_range", "plugins", "stages": [{"kind": ...}, ...]}
Same stage parameters as vidimg/adapters/avisynth.py; parity measured in research/avisynth/05 section 9.
"""

import json
import sys

import vapoursynth as vs

core = vs.core


def build(spec: dict) -> vs.VideoNode:
    core.std.LoadAllPlugins(spec["plugins"])
    import havsfunc  # needs the plugins loaded first

    num, den = spec["fps"]
    c = core.lsmas.LWLibavSource(spec["src"], cachefile=spec["cache"], fpsnum=num, fpsden=den)
    rate_num, rate_den = num, den
    if spec.get("full_range"):
        c = core.resize.Point(c, format=vs.YUV420P16, range_in_s="full", range_s="limited")
    else:
        c = core.resize.Point(c, format=vs.YUV420P16)
    for st in spec["stages"]:
        k = st["kind"]
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
        else:
            raise SystemExit(f"stage {k!r} is not implemented in the VapourSynth backend")
    c = core.resize.Point(c, format=vs.YUV420P10, dither_type="error_diffusion")
    return core.std.AssumeFPS(c, fpsnum=rate_num, fpsden=rate_den)


if __name__ == "__main__":
    spec = json.load(open(sys.argv[1], encoding="utf-8"))
    clip = build(spec)
    clip.output(sys.stdout.buffer, y4m=True)
