"""Texture between the lines: measuring over-smoothing, and returning the source's own texture.

Why (Doom9 reply to the 26-episode report: "basically denoised to death"): every free model, anime-v3, Real-CUGAN
without denoise, general-v3 and the faithful RealESRNet alike, wipes the fine texture between the lines (painted
backgrounds, grain, cloth) while sharpening the lines themselves. Band energy and VMAF did not show it (the
sharper lines outweigh the lost texture); the share of FLAT area between the source's lines does.

Measure (output size, luma): flat = local standard deviation (7x7) below 1 level, outside the dilated edges of a
bicubic upscale of the source (the reference that adds and removes nothing). Measured, flat area bicubic -> model:
report zooms 7-61 % -> 27-90 %; owner samples +12..+22 points; all five candidates +10..+24 points.

Return: out + strength * mask * (band(src) - band(out)), with band = G(0.8) - G(4) at output size (texture at the
SOURCE's scale; the bicubic upscale has almost nothing at 1 output pixel), only between the lines (mask, from the
source's edges) and only where the source has more texture than the output (the model's own detail is never
subtracted), limited to +-12 levels. Nothing is invented: it is the source's texture, at its own place. Measured
at strength 1: flat area b001 0.36 -> 0.22 (bicubic 0.20), Sample07 0.53 -> 0.42 (0.38), Sample06 0.87 -> 0.80
(0.77); line sharpness unchanged (1.00 of the model's); a 60 % plain blend got less texture back and lost
sharpness (0.94). It also returns some of the source's compression noise: hence a strength setting.
"""

from __future__ import annotations

from collections import deque

import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import media
from .adapters import AdapterResult

FLAT_STD = 1.0              # 8-bit levels: a 7x7 neighbourhood flatter than this has no texture
EDGE_BLUR, EDGE_LO, EDGE_HI, EDGE_DILATE = 1.5, 30, 80, 9
GIVE_MAX = 12.0             # 8-bit levels of texture returned at most per pixel
FLAT_INCREASE_WARN = 0.08   # guard: flat area grew by more than this (share of the area between the lines)


def edges(ref8: np.ndarray) -> np.ndarray:
    """The source's lines (dilated), on an 8-bit luma at output size."""
    e = cv2.Canny(cv2.GaussianBlur(ref8, (0, 0), EDGE_BLUR), EDGE_LO, EDGE_HI)
    return cv2.dilate(e, np.ones((EDGE_DILATE, EDGE_DILATE), np.uint8)) > 0


def _local_std(g: np.ndarray) -> np.ndarray:
    g = g.astype(np.float32)
    m = cv2.blur(g, (7, 7))
    return np.sqrt(np.maximum(cv2.blur(g * g, (7, 7)) - m * m, 0))


def flat_shares(src8: np.ndarray, out8: np.ndarray) -> tuple[float, float]:
    """(flat share of the bicubic reference, flat share of the output) between the lines, at output size."""
    h, w = out8.shape[:2]
    ref = cv2.resize(src8, (w, h), interpolation=cv2.INTER_CUBIC)
    between = ~edges(ref)
    if not between.any():
        return 0.0, 0.0
    return float(np.mean(_local_std(ref)[between] < FLAT_STD)), float(np.mean(_local_std(out8)[between] < FLAT_STD))


def band(g: np.ndarray) -> np.ndarray:
    g = g.astype(np.float32)
    return cv2.GaussianBlur(g, (0, 0), 0.8) - cv2.GaussianBlur(g, (0, 0), 4.0)


def return_texture(ref: np.ndarray, out: np.ndarray, strength: float, scale: float = 1.0) -> np.ndarray:
    """ref: the source upscaled to output size, out: the model output (same luma scale; `scale` = levels per
    8-bit level, e.g. 257 for 16-bit). Returns the output with the source's texture put back (float32)."""
    if strength <= 0:
        return out.astype(np.float32)
    ref8 = np.clip(ref / scale + 0.5, 0, 255).astype(np.uint8)
    mask = cv2.GaussianBlur((~edges(ref8)).astype(np.float32), (0, 0), 2.0)
    br, bo = band(ref), band(out)
    give = np.clip(br - bo, -GIVE_MAX * scale, GIVE_MAX * scale)
    give = np.where(np.abs(br) > np.abs(bo), give, 0.0)
    return out.astype(np.float32) + strength * mask * give


# ----------------------------------------------------------------------------- video pass
def _probe_stream(path: Path) -> dict:
    r = subprocess.run([media.ffprobe(), "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,color_space,color_range,color_primaries,color_transfer,"
                        "sample_aspect_ratio", "-of", "json", str(path)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return (json.loads(r.stdout or "{}").get("streams") or [{}])[0]


def _matrix(name: str | None, height: int) -> str:
    if name in ("bt709",):
        return "bt709"
    if name in ("bt470bg", "smpte170m", "bt601"):
        return "bt601"
    return "bt709" if height >= 720 else "bt601"


def apply(base: Path, restored: Path, dst: Path, strength: float, encoder: str = "hevc10", *, workers: int = 6,
          on_progress: Callable[[int, int], None] | None = None) -> AdapterResult:
    """Write `dst`: `restored` with the texture of `base` (the model's input) returned in its luma. Chroma, audio,
    metadata, colour tags and pixel aspect come from `restored`; same encoder settings as the model pass."""
    base, restored, dst = Path(base), Path(restored), Path(dst)
    bi, ri = media.probe(base, exact=True), media.probe(restored, exact=True)
    so = _probe_stream(restored)
    w, h, n = ri["width"], ri["height"], ri["frames"]
    if bi["frames"] and abs(bi["frames"] - n) > 2:
        return AdapterResult(False, error=f"frame counts differ ({bi['frames']} vs {n})")
    in_m = _matrix((media.colour_hint(base) or {}).get("color_space") or _probe_stream(base).get("color_space"),
                   bi["height"])
    out_m = _matrix(so.get("color_space"), h)
    ff = media.ffmpeg()
    # the source at output size, in the output's matrix, so its luma is comparable (bicubic: adds nothing)
    dec_b = subprocess.Popen([ff, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(base), "-map", "0:v:0",
                              "-fps_mode", "passthrough", "-vf",
                              f"scale={w}:{h}:flags=bicubic:in_color_matrix={in_m}:out_color_matrix={out_m}:"
                              f"in_range=tv:out_range=tv,format=yuv420p16le,extractplanes=y",
                              "-pix_fmt", "gray16le", "-f", "rawvideo", "-"],
                             stdout=subprocess.PIPE, stdin=subprocess.DEVNULL)
    dec_r = subprocess.Popen([ff, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(restored), "-map", "0:v:0",
                              "-fps_mode", "passthrough", "-pix_fmt", "yuv420p16le", "-f", "rawvideo", "-"],
                             stdout=subprocess.PIPE, stdin=subprocess.DEVNULL)
    exact = subprocess.run([media.ffprobe(), "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=r_frame_rate", "-of", "csv=p=0", str(restored)], capture_output=True,
                           text=True, encoding="utf-8", errors="replace").stdout.strip() or f"{ri['fps']:g}"
    tags = [x for k, opt in (("color_space", "-colorspace"), ("color_range", "-color_range"),
                             ("color_primaries", "-color_primaries"), ("color_transfer", "-color_trc"))
            if so.get(k) and so[k] not in ("unknown", "reserved") for x in (opt, so[k])]
    sar = so.get("sample_aspect_ratio") or "1:1"
    enc = [ff, "-hide_banner", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p16le", "-s", f"{w}x{h}",
           "-r", exact, "-i", "pipe:0", "-i", str(restored), "-map", "0:v:0", "-map", "1:a?", "-c:a", "copy",
           "-vf", f"setsar={sar.replace(':', '/')}", *media.encoder_args(encoder), *tags, "-map_metadata", "1",
           "-metadata", f"comment=Restored with VidIMG; source texture returned (strength {strength:g})"]
    if dst.suffix.lower() in (".mp4", ".mov", ".m4v"):
        enc += ["-movflags", "+write_colr+faststart"]
    ysz, csz = w * h * 2, (w // 2) * (h // 2) * 2

    def frames():
        while True:
            yb = dec_b.stdout.read(ysz)
            fr = dec_r.stdout.read(ysz + 2 * csz)
            if len(yb) < ysz or len(fr) < ysz + 2 * csz:
                return
            yield yb, fr

    def work(item):
        yb, fr = item
        ref = np.frombuffer(yb, "<u2").reshape(h, w).astype(np.float32)
        out = np.frombuffer(fr[:ysz], "<u2").reshape(h, w).astype(np.float32)
        y = np.clip(return_texture(ref, out, strength, scale=257.0) + 0.5, 0, 65535).astype("<u2")
        return y.tobytes() + fr[ysz:]

    count = [0]
    err: list[str] = []

    def feed(pipe) -> None:
        try:
            # a bounded window: Executor.map would read the whole video into memory before the first frame is
            # written (55 GB on a 25-minute episode, measured: the page file filled the system drive)
            def put(buf: bytes) -> None:
                pipe.write(buf)
                count[0] += 1
                if on_progress and count[0] % 50 == 0:
                    on_progress(count[0], n)

            with ThreadPoolExecutor(max_workers=workers) as ex:
                pending: deque = deque()
                for item in frames():
                    pending.append(ex.submit(work, item))
                    if len(pending) >= workers * 2:
                        put(pending.popleft().result())
                while pending:
                    put(pending.popleft().result())
        except Exception as e:  # noqa: BLE001
            err.append(f"{type(e).__name__}: {e}")
        finally:
            pipe.close()

    p = subprocess.Popen([*enc, str(dst)], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    t = threading.Thread(target=feed, args=(p.stdin,), daemon=True)
    t.start()
    stderr = p.stderr.read().decode("utf-8", "replace")
    p.wait()
    t.join()
    for d in (dec_b, dec_r):
        d.stdout.close()
        d.wait()
    if err or p.returncode != 0 or not dst.exists():
        return AdapterResult(False, error=(err[0] if err else stderr.strip()[-300:] or f"ffmpeg exit {p.returncode}"))
    out = media.probe(dst, exact=True)
    if abs(out["frames"] - n) > max(2, n * 0.002):
        return AdapterResult(False, [dst], error=f"frame count {out['frames']} != {n}")
    return AdapterResult(True, [dst], details={"strength": strength, "frames": count[0], "matrix": [in_m, out_m]})
