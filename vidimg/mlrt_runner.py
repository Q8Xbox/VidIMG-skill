"""vs-mlrt worker: runs INSIDE tools/vs-venv (VapourSynth R72 + vs-mlrt), never imported by the main tool.

Same pipe contract as the PyTorch worker (vidimg/sr_runner.py video mode), so decoding, encoding,
audio, metadata and the guard are shared: rgb24 frames (full range) of width x height on stdin, rgb48le
frames of out_width x out_height on stdout, one out frame per in frame, in order.

The frames come from the pipe into a VapourSynth clip through std.ModifyFrame (a reader thread keeps
them in order; frames past the end of the pipe are never written), go through vsmlrt.inference
(ONNX Runtime: DirectML on any DirectX 12 GPU, or CPU), and are resized to the exact output size:
box (area) average for an integer shrink like the PyTorch path, Spline36 otherwise.

    tools/vs-venv/Scripts/python -I vidimg/mlrt_runner.py spec.json
spec: {"onnx", "width", "height", "out_width", "out_height", "frames", "plugins", "vsmlrt_dir",
       "backend": "dml"|"cpu", "fp16": bool, "tile": int|0, "streams": int,
       "half": bool}   half: the ONNX file itself takes and returns half-float frames (it is not converted)
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import numpy as np


def area_shrink(a: np.ndarray, ow: int, oh: int) -> np.ndarray | None:
    """HxWxC -> ohxowxC by box averaging when both factors are the same integer; None otherwise."""
    h, w = a.shape[:2]
    if h % oh or w % ow or h // oh != w // ow:
        return None
    k = h // oh
    return a if k == 1 else a.reshape(oh, k, ow, k, a.shape[2]).mean(axis=(1, 3))


CLOSED = "output pipe closed before the last frame: the encoder stopped"


def write_frame(stream, data: bytes) -> bool:
    """Write one frame (or, with no data, flush). False when the reader has closed the pipe."""
    try:
        if data:
            stream.write(data)
        else:
            stream.flush()
        return True
    except OSError:                 # BrokenPipeError, or EINVAL from a closed pipe on Windows
        return False


class PipeFrames:
    """Sequential stdin reader shared by VapourSynth's frame requests (which may arrive out of order)."""

    def __init__(self, stream, w: int, h: int):
        self.stream, self.size, self.shape = stream, w * h * 3, (h, w, 3)
        self.frames: dict[int, np.ndarray] = {}
        self.read = 0              # frames read so far
        self.eof = False
        self.lock = threading.Lock()
        self.black = np.zeros(self.shape, np.uint8)

    def get(self, n: int) -> np.ndarray:
        with self.lock:
            while self.read <= n and not self.eof:
                buf = self.stream.read(self.size)
                if len(buf) < self.size:
                    self.eof = True
                    break
                self.frames[self.read] = np.frombuffer(buf, np.uint8).reshape(self.shape)
                self.read += 1
            return self.frames.pop(n, self.black)

    def real(self, n: int) -> bool:
        """Was frame n a real frame from the pipe? (valid once frame n has been requested)."""
        with self.lock:
            return n < self.read


def main(spec: dict) -> None:
    import vapoursynth as vs
    core = vs.core
    core.std.LoadAllPlugins(spec["plugins"])
    vsmlrt_dir = Path(spec["vsmlrt_dir"])
    core.std.LoadPlugin(str(vsmlrt_dir / "vsort.dll"))
    sys.path.insert(0, str(vsmlrt_dir))
    import vsmlrt  # needs core.ort loaded first

    w, h, ow, oh = spec["width"], spec["height"], spec["out_width"], spec["out_height"]
    # room for a few more frames than expected: the decoder decides the real count, the pipe end stops us
    length = int(spec["frames"] * 1.02) + 16
    pipe = PipeFrames(sys.stdin.buffer, w, h)
    blank = core.std.BlankClip(width=w, height=h, format=vs.RGB24, length=length, keep=False)

    def fill(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        out = f.copy()
        a = pipe.get(n)
        for p in range(3):
            np.copyto(np.asarray(out[p]), a[:, :, p])
        return out

    src = core.std.ModifyFrame(blank, blank, fill)
    half = bool(spec.get("half"))
    rgb = core.resize.Point(src, format=vs.RGBH if half else vs.RGBS)
    if spec.get("backend", "dml") == "cpu":
        be = vsmlrt.Backend.ORT_CPU(num_streams=int(spec.get("streams", 1)))
    else:
        be = vsmlrt.Backend.ORT_DML(fp16=bool(spec.get("fp16", True)) and not half, num_streams=int(spec.get("streams", 1)))
    tile = int(spec.get("tile") or 0)
    kw = {"tilesize": (tile, tile), "overlap": (16, 16)} if tile and (w > tile or h > tile) else {}
    up = vsmlrt.inference(rgb, spec["onnx"], backend=be, **kw)
    k = up.width // w
    box = up.width == ow * (up.width // ow) and up.height == oh * (up.height // oh) and up.width // ow == up.height // oh
    if not box:
        up = core.resize.Spline36(up, ow, oh)
    wr = sys.stdout.buffer
    n_out = 0
    for n, f in enumerate(up.frames(prefetch=4, backlog=8)):
        if not pipe.real(n):
            break
        # float32 before the 16-bit write: a half-float cannot hold 65535 (it overflows to infinity)
        a = np.stack([np.asarray(f[p]) for p in range(3)], axis=-1).astype(np.float32, copy=False)
        if box:
            a = area_shrink(a, ow, oh)
        if not write_frame(wr, (np.clip(a, 0, 1) * 65535 + 0.5).astype("<u2").tobytes()):
            # not an error by itself: a chunk's encoder stops reading once its frames are in. Whether the run
            # worked is judged by the encoder's exit code and the frame count, in the caller (as in sr_runner)
            sys.stderr.write(CLOSED + "\n")
            return
        n_out += 1
    if not write_frame(wr, b""):
        sys.stderr.write(CLOSED + "\n")
        return
    sys.stderr.write(json.dumps({"frames": n_out, "native_scale": k, "resize": "area" if box else "spline36"}) + "\n")


if __name__ == "__main__":
    main(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
