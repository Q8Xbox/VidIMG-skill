"""ONNX export worker: runs INSIDE tools/sr-venv (PyTorch + spandrel + onnx + onnxruntime), never imported
by the main tool. Converts official .pth weights to ONNX for vs-mlrt and checks every file before use.

Graph contract (vs-mlrt's convention): input "input" float32 NCHW RGB in [0, 1], dynamic N/H/W; output
"output" the model's native scale. fp32 weights (vs-mlrt can run fp16 itself). The doc string records the
source checkpoint and its SHA-256.

Check: the ONNX file on ONNX Runtime (CPU) and the checkpoint on PyTorch fp32 must agree on a random
image (max abs difference below 1e-3 of full scale), otherwise the file is deleted.

    tools/sr-venv/Scripts/python -I vidimg/onnx_export.py spec.json
spec: {"models": [{"key", "pth", "onnx"}, ...]}  -> one JSON line per model on stdout
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

MAX_DIFF = 1e-3


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def export(key: str, pth: Path, dst: Path) -> dict:
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from spandrel import ModelLoader

    t0 = time.monotonic()
    d = ModelLoader().load_from_file(str(pth))
    m = d.model.eval().float().cpu()
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.onnx")
    with torch.inference_mode():
        torch.onnx.export(m, (torch.rand(1, 3, 64, 64),), str(tmp), input_names=["input"], output_names=["output"],
                          opset_version=17, dynamo=False,
                          dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                                        "output": {0: "batch", 2: "out_height", 3: "out_width"}})
    mo = onnx.load(str(tmp))
    mo.doc_string = f"VidIMG {key}: {d.architecture.name} x{d.scale} from {pth.name} (sha256 {_sha(pth)})"
    onnx.checker.check_model(mo)
    onnx.save(mo, str(tmp))
    x = np.random.default_rng(0).random((1, 3, 48, 64), dtype=np.float32)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    got = ort.InferenceSession(str(tmp), so, providers=["CPUExecutionProvider"]).run(None, {"input": x})[0]
    with torch.inference_mode():
        ref = m(torch.from_numpy(x)).numpy()
    diff = float(np.abs(got - ref).max())
    if got.shape != ref.shape or diff > MAX_DIFF:
        tmp.unlink(missing_ok=True)
        return {"key": key, "ok": False, "error": f"ONNX output differs from PyTorch (max {diff:.2e}, shape {got.shape})"}
    tmp.replace(dst)
    return {"key": key, "ok": True, "onnx": str(dst), "scale": d.scale, "max_diff": diff, "sha256": _sha(dst),
            "MB": round(dst.stat().st_size / 2 ** 20, 1), "seconds": round(time.monotonic() - t0, 1)}


if __name__ == "__main__":
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    for item in spec["models"]:
        try:
            r = export(item["key"], Path(item["pth"]), Path(item["onnx"]))
        except Exception as e:  # noqa: BLE001
            r = {"key": item["key"], "ok": False, "error": f"{type(e).__name__}: {e}"}
        print(json.dumps(r), flush=True)
