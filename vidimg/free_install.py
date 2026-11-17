"""Install / verify the free engine: models from tools/free-models/manifest.json (official releases,
SHA-256 trust-on-first-use in lock.json) and the isolated GPU runtime tools/sr-venv
(PyTorch CUDA build + spandrel). Nothing paid, nothing system-wide.

Usage: python -m vidimg.free_install            (install or verify)
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from . import config

ROOT = config.TOOLS_DIR / "free-models"
MANIFEST, LOCK = config.MANIFEST_DIR / "free-models" / "manifest.json", config.MANIFEST_DIR / "free-models" / "lock.json"
VENV = config.TOOLS_DIR / "sr-venv"
PY = VENV / "Scripts" / "python.exe"
TORCH = "torch==2.14.1"
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"          # CUDA 13: Blackwell (sm_120) and older
PIP_PINS = ["spandrel==0.4.2", "numpy", "opencv-python-headless", "pillow"]


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def model_path(key: str) -> Path:
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))["models"][key]
    return ROOT / (m.get("file") or m["url"].rsplit("/", 1)[-1])


def _download(url: str, dst: Path, attempts: int = 4) -> None:
    part = dst.with_name(dst.name + ".part")
    last = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "vidimg-free-install"})
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as f:
                if "text/html" in (r.headers.get("Content-Type") or ""):
                    raise IOError("got a web page instead of the model file (link changed or quota hit)")
                want = int(r.headers.get("Content-Length") or 0)
                while b := r.read(1 << 20):
                    f.write(b)
            if want and part.stat().st_size != want:
                raise IOError(f"short read {part.stat().st_size}/{want}")
            part.replace(dst)
            return
        except Exception as e:                                  # noqa: BLE001
            last = e
            part.unlink(missing_ok=True)
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"{url}: {last}")


def install(log=print) -> dict:
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))["models"]
    lock = json.loads(LOCK.read_text(encoding="utf-8")) if LOCK.exists() else {}
    report = {"models": {}, "errors": []}
    for key, m in man.items():
        dst = model_path(key)
        try:
            if not dst.exists():
                log(f"  downloading {key}")
                _download(m["url"], dst)
            d = _sha(dst)
            if m["url"] in lock and lock[m["url"]] != d:
                report["errors"].append(f"{key}: SHA-256 changed since first download; refusing")
                continue
            lock[m["url"]] = d
            report["models"][key] = {"file": dst.name, "MB": round(dst.stat().st_size / 2**20, 1), "license": m["license"]}
        except Exception as e:                                  # noqa: BLE001
            report["errors"].append(f"{key}: {e}")
    LOCK.write_text(json.dumps(lock, indent=1, sort_keys=True), encoding="utf-8")
    if not PY.exists():
        log("  creating tools/sr-venv")
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
    have = subprocess.run([str(PY), "-c", "import torch, spandrel; print(torch.__version__)"], capture_output=True, text=True)
    if have.returncode != 0 or "+cu" not in have.stdout:
        log("  installing PyTorch (CUDA) + spandrel: about 3 GB")
        subprocess.run([str(PY), "-m", "pip", "install", "-q", TORCH, "--index-url", TORCH_INDEX], check=True)
        subprocess.run([str(PY), "-m", "pip", "install", "-q", *PIP_PINS], check=True)
    probe = subprocess.run([str(PY), "-c", "import torch; print(torch.__version__, torch.cuda.is_available(), "
                            "torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"],
                           capture_output=True, text=True)
    report["runtime"] = probe.stdout.strip()
    report["gpu"] = "True" in probe.stdout
    report["ok"] = not report["errors"] and report["gpu"]
    return report


if __name__ == "__main__":
    print(json.dumps(install(), indent=1))
