"""Ground-truth scoring helpers for the benchmarks (free tools only): VMAF / PSNR / SSIM with frames paired
by index (timestamp pairing mis-paired MKV vs NUT frames) at 10-bit, and SFace face identity."""

from __future__ import annotations

from . import imgfile

import re
import subprocess
from pathlib import Path

from . import config, media

CAL = config.WORKSPACE / "calibration_video" / "20261115-150331"
_ffmpeg = media.ffmpeg


def _run(args: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", stdin=subprocess.DEVNULL, timeout=timeout, errors="replace")


def _metrics(gt: Path, out: Path, gt_vf: str = "") -> dict:
    """VMAF / PSNR / SSIM with frames paired BY INDEX. Pairing by timestamp is wrong here: Matroska
    stores ms timestamps while NUT keeps exact ones, so framesync paired neighbouring frames
    (observed: QTGMC 27.1 dB by timestamp vs 36.3 dB by index on the same files)."""
    g = media.probe(gt)
    # both sides in 10-bit: 8-bit ground truth converts exactly, while squeezing a 10-bit result
    # into 8-bit makes swscale dither and costs it ~2 dB (TIVTC measured 38 dB instead of bit-exact)
    idx = "settb=1/30,setpts=N"
    graph = (f"[1:v]scale={g['width']}:{g['height']}:flags=bicubic,setsar=1,format=yuv420p10le,{idx},split=3[d1][d2][d3];"
             f"[0:v]{gt_vf}setsar=1,format=yuv420p10le,{idx},split=3[r1][r2][r3];"
             f"[d1][r1]libvmaf=n_threads=8;[d2][r2]psnr;[d3][r3]ssim")
    r = _run([_ffmpeg(), "-hide_banner", "-nostats", "-i", str(gt), "-i", str(out), "-lavfi", graph,
              "-shortest", "-f", "null", "-"])
    v = re.findall(r"VMAF score:\s*([0-9.]+)", r.stderr)
    p = re.findall(r"PSNR .*? average:([0-9.inf]+)", r.stderr)
    s = re.findall(r"SSIM .*? All:([0-9.]+)", r.stderr)
    return {"vmaf": float(v[-1]) if v else None, "psnr": float(p[-1]) if p and p[-1] != "inf" else None,
            "ssim": float(s[-1]) if s else None}


def _identity(gt: Path, out: Path, gt_vf: str | None = None, samples: int = 6) -> float | None:
    """Lowest SFace similarity of the original's faces, frames paired by index."""
    import tempfile
    import cv2
    from . import faces, guard
    from .guard.video import _frames_by_index
    n = min(media.probe(out)["frames"], media.probe(gt)["frames"] // (2 if gt_vf else 1))
    idx = [int(n * (i + 0.5) / samples) for i in range(samples)]
    sims = []
    with tempfile.TemporaryDirectory(prefix="vidimg_bid_") as tmp:
        gm = _frames_by_index(gt, idx, Path(tmp), "g", pre_vf=gt_vf.rstrip(",") if gt_vf else None)
        om = _frames_by_index(out, idx, Path(tmp), "o")
        for i in idx:
            if i not in gm or i not in om:
                continue
            g, o = imgfile.imread(str(gm[i])), imgfile.imread(str(om[i]))
            o = cv2.resize(o, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_AREA)
            fs = faces.detect(g, score=guard.FACE_SCORE)
            got = faces.match(fs, faces.detect(o, score=0.4))
            sims += [faces.similarity(faces.embedding(g, f), faces.embedding(o, x or f)) for f, x in zip(fs, got)]
    return min(sims) if sims else None


def _score(gt: Path, out: Path, gt_vf: str = "") -> dict:
    m = _metrics(gt, out, gt_vf)
    i = _identity(gt, out, gt_vf or None)
    return {"vmaf": m["vmaf"] and round(m["vmaf"], 2), "identity": i and round(i, 4),
            "psnr": m["psnr"] and round(m["psnr"], 2), "ssim": m["ssim"] and round(m["ssim"], 4)}


EVEN = r"select=not(mod(n\,2)),"   # even frames = top-field moments
