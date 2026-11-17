"""Free-engine GPU worker: runs INSIDE tools/sr-venv (PyTorch CUDA + spandrel), never imported by the
main tool. Models are the pinned open ones in tools/free-models (manifest.json).

    image:  python -I sr_runner.py image spec.json
            spec = {"model", "inputs": [...], "out_dir", "suffix", "scale": float, "tile": int}
            -> <out_dir>/<stem><suffix>.png (16-bit), prints one JSON line per image
    video:  python -I sr_runner.py video spec.json   (raw rgb24 frames on stdin, raw rgb48le on stdout)
            spec = {"model", "width", "height", "out_width", "out_height", "batch"}

Output size is always the requested one: the model's native scale is followed by a Lanczos resize
(down) or nothing; a model never decides the output size by itself.
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from spandrel import ImageModelDescriptor, ModelLoader

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(path: str):
    d = ModelLoader().load_from_file(path)
    if not isinstance(d, ImageModelDescriptor):
        raise SystemExit(f"{path}: not an image model")
    d.model.eval().to(DEV)
    half = DEV.type == "cuda" and d.supports_half
    if half:
        d.model.half()
    return d, half


def _pad_to(t: torch.Tensor, mult: int) -> tuple[torch.Tensor, int, int]:
    h, w = t.shape[-2:]
    ph, pw = (-h) % mult, (-w) % mult
    if ph or pw:
        t = torch.nn.functional.pad(t, (0, pw, 0, ph), mode="reflect")
    return t, h, w


@torch.inference_mode()
def run_tensor(d, half: bool, x: torch.Tensor, tile: int = 0, overlap: int = 16) -> torch.Tensor:
    """x: NCHW float in [0,1] on DEV -> model output, tiled when tile > 0 (overlaps cropped)."""
    mult = max(1, getattr(d.size_requirements, "multiple_of", 1) or 1)
    s = d.scale
    x = x.half() if half else x.float()
    n, c, h, w = x.shape
    if not tile or (h <= tile and w <= tile):
        xp, h0, w0 = _pad_to(x, mult)
        return d.model(xp)[..., : h0 * s, : w0 * s].float().clamp(0, 1)
    out = torch.zeros((n, c, h * s, w * s), device=x.device, dtype=torch.float32)
    step = tile - 2 * overlap
    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            ya, xa = max(0, y0 - overlap), max(0, x0 - overlap)
            yb, xb = min(h, y0 + step + overlap), min(w, x0 + step + overlap)
            patch, ph, pw = _pad_to(x[..., ya:yb, xa:xb], mult)
            res = d.model(patch)[..., : ph * s, : pw * s].float().clamp(0, 1)
            cy, cx = (y0 - ya) * s, (x0 - xa) * s
            hh, ww = (min(h, y0 + step) - y0) * s, (min(w, x0 + step) - x0) * s
            out[..., y0 * s: y0 * s + hh, x0 * s: x0 * s + ww] = res[..., cy: cy + hh, cx: cx + ww]
    return out


def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    if img.shape[1] == w and img.shape[0] == h:
        return img
    interp = cv2.INTER_AREA if w < img.shape[1] else cv2.INTER_LANCZOS4
    return cv2.resize(img, (w, h), interpolation=interp)


def image_mode(spec: dict) -> None:
    d, half = load(spec["model"])
    out_dir = Path(spec["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for inp in spec["inputs"]:
        t0 = time.monotonic()
        src = cv2.imread(inp, cv2.IMREAD_UNCHANGED)
        if src is None:
            print(json.dumps({"input": inp, "ok": False, "error": "unreadable"}), flush=True)
            continue
        if src.ndim == 2:
            src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        if src.shape[2] == 4:
            src = src[..., :3]
        peak = 65535.0 if src.dtype == np.uint16 else 255.0
        rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB).astype(np.float32) / peak
        x = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(DEV)
        try:
            y = run_tensor(d, half, x, spec.get("tile", 512))
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            y = run_tensor(d, half, x, 256)
        out = y[0].permute(1, 2, 0).cpu().numpy()
        tw, th = round(src.shape[1] * spec["scale"]), round(src.shape[0] * spec["scale"])
        out = _resize(out, tw, th)
        dst = out_dir / f"{Path(inp).stem}{spec['suffix']}.png"
        cv2.imwrite(str(dst), cv2.cvtColor((np.clip(out, 0, 1) * 65535 + 0.5).astype(np.uint16), cv2.COLOR_RGB2BGR))
        print(json.dumps({"input": inp, "ok": True, "output": str(dst), "seconds": round(time.monotonic() - t0, 2)}),
              flush=True)


def video_mode(spec: dict) -> None:
    """Pipelined: a reader thread, batched GPU inference, a writer thread (decode, model and encode overlap)."""
    import queue
    import threading

    torch.backends.cudnn.benchmark = True
    d, half = load(spec["model"])
    w, h, ow, oh = spec["width"], spec["height"], spec["out_width"], spec["out_height"]
    batch = max(1, int(spec.get("batch", 4)))
    fsize = w * h * 3
    rd, wr = sys.stdin.buffer, sys.stdout.buffer
    q_in: "queue.Queue[bytes | None]" = queue.Queue(maxsize=batch * 4)
    q_out: "queue.Queue[np.ndarray | None]" = queue.Queue(maxsize=batch * 4)

    def reader() -> None:
        while True:
            buf = rd.read(fsize)
            if len(buf) < fsize:
                q_in.put(None)
                return
            q_in.put(buf)

    def writer() -> None:
        while (a := q_out.get()) is not None:
            wr.write(a.tobytes())
        wr.flush()

    tr, tw = threading.Thread(target=reader, daemon=True), threading.Thread(target=writer)
    tr.start()
    tw.start()
    n, done = 0, False
    with torch.inference_mode():
        while not done:
            bufs = []
            while len(bufs) < batch:
                b = q_in.get()
                if b is None:
                    done = True
                    break
                bufs.append(b)
            if not bufs:
                break
            x = torch.from_numpy(np.frombuffer(b"".join(bufs), np.uint8).reshape(len(bufs), h, w, 3)).to(DEV, non_blocking=True)
            x = x.permute(0, 3, 1, 2).float().div_(255.0)
            y = run_tensor(d, half, x, spec.get("tile", 0))
            if (y.shape[-1], y.shape[-2]) != (ow, oh):
                mode = "area" if ow < y.shape[-1] else "bicubic"
                y = torch.nn.functional.interpolate(y, size=(oh, ow), mode=mode,
                                                    **({} if mode == "area" else {"antialias": True}))
            u16 = (y.clamp_(0, 1).mul_(65535).add_(0.5)).to(torch.int32).permute(0, 2, 3, 1).cpu().numpy().astype("<u2")
            for a in u16:
                q_out.put(a)
            n += len(bufs)
    q_out.put(None)
    tw.join()
    sys.stderr.write(json.dumps({"frames": n}) + "\n")


if __name__ == "__main__":
    mode, spec_path = sys.argv[1], sys.argv[2]
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    {"image": image_mode, "video": video_mode}[mode](spec)
