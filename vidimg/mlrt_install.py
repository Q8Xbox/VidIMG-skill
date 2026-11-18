"""Install / verify the vs-mlrt engine (ONNX Runtime + DirectML inside VidIMG's VapourSynth).

* plugin: the official AmusementClub/vs-mlrt release assets in tools/vs-mlrt/manifest.json, SHA-256
  pinned; unpacked with Windows' own tar (bsdtar reads .7z), into tools/vs-mlrt;
* models: NOT downloaded. The official weights already pinned in tools/free-models/lock.json are
  converted to ONNX by vidimg/onnx_export.py inside tools/sr-venv, and each file is checked against
  PyTorch before it is kept (tools/vs-mlrt/models/<key>.onnx);
* needs tools/vs-venv (python -m vidimg.vs_install) and tools/sr-venv with the weights
  (python -m vidimg.free_install). Nothing is installed system-wide.

Usage: python -m vidimg.mlrt_install            (install or verify)
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from . import config

ROOT = config.TOOLS_DIR / "vs-mlrt"
MODELS = ROOT / "models"
MANIFEST = config.MANIFEST_DIR / "vs-mlrt" / "manifest.json"
PLUGIN = ROOT / "vsort.dll"
SCRIPT = ROOT / "vsmlrt.py"
SR_PY = config.TOOLS_DIR / "sr-venv" / "Scripts" / "python.exe"
VS_PY = config.TOOLS_DIR / "vs-venv" / "Scripts" / "python.exe"
VS_PLUGINS = config.TOOLS_DIR / "vs-venv" / "Lib" / "site-packages" / "vapoursynth" / "plugins" / "vsrepo"
EXPORTER = Path(__file__).resolve().parent / "onnx_export.py"


def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def onnx_path(key: str) -> Path:
    return MODELS / f"{key}.onnx"


def available(model: str | None = None) -> tuple[bool, str]:
    """Can the vs-mlrt engine run (this model)? (ok, reason)."""
    if not VS_PY.exists():
        return False, "VapourSynth environment missing: run python -m vidimg.vs_install"
    if not (PLUGIN.exists() and SCRIPT.exists() and (ROOT / "vsort" / "onnxruntime.dll").exists()):
        return False, "vs-mlrt missing: run python -m vidimg.mlrt_install"
    if model is not None and not onnx_path(model).exists():
        return False, f"no ONNX file for {model}: run python -m vidimg.mlrt_install"
    if not shutil.which("ffmpeg"):
        return False, "no ffmpeg on PATH"
    return True, "ok"


def _unpack(archive: Path, dest: Path) -> None:
    tar = Path(r"C:\Windows\System32\tar.exe") if sys.platform == "win32" else Path(shutil.which("tar") or "tar")
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(tar), "-xf", str(archive), "-C", str(dest)], check=True, capture_output=True)


def install(log=print) -> dict:
    from .free_install import model_path
    from .setup import _fetch
    man = manifest()
    report: dict = {"release": man["release"], "models": {}, "errors": []}
    if not VS_PY.exists():
        report["errors"].append("VapourSynth environment missing: run python -m vidimg.vs_install first")
        report["ok"] = False
        return report
    dl = ROOT / "_downloads"
    for name, a in man["assets"].items():
        try:
            _fetch(a["url"], a["sha256"], dl / name, log)
        except Exception as e:  # noqa: BLE001
            report["errors"].append(f"{name}: {e}")
    if report["errors"]:
        report["ok"] = False
        return report
    if not available()[0]:
        log("  unpacking vs-mlrt")
        for name in man["assets"]:
            _unpack(dl / name, ROOT)
    # ONNX conversion needs onnx + onnxruntime in the model environment
    if not SR_PY.exists():
        report["errors"].append("model environment missing: run python -m vidimg.free_install first")
    else:
        have = subprocess.run([str(SR_PY), "-c", "import onnx, onnxruntime, onnxscript"], capture_output=True)
        if have.returncode != 0:
            log("  installing onnx + onnxruntime into tools/sr-venv (about 35 MB)")
            subprocess.run([str(SR_PY), "-m", "pip", "install", "-q", *man["pip_pins_sr_venv"]], check=True)
        todo = []
        for key in man["models"]:
            pth = model_path(key)
            if onnx_path(key).exists():
                report["models"][key] = {"onnx": onnx_path(key).name, "status": "present"}
            elif pth.exists():
                todo.append({"key": key, "pth": str(pth), "onnx": str(onnx_path(key))})
            else:
                report["errors"].append(f"{key}: official weights missing ({pth.name}): run python -m vidimg.free_install")
        if todo:
            log(f"  converting {', '.join(t['key'] for t in todo)} to ONNX (checked against PyTorch)")
            MODELS.mkdir(parents=True, exist_ok=True)
            spec = MODELS / ".export_spec.json"
            spec.write_text(json.dumps({"models": todo}), encoding="utf-8")
            r = subprocess.run([str(SR_PY), "-I", str(EXPORTER), str(spec)], capture_output=True, text=True)
            spec.unlink(missing_ok=True)
            for ln in r.stdout.splitlines():
                if ln.startswith("{"):
                    d = json.loads(ln)
                    if d.get("ok"):
                        report["models"][d["key"]] = {k: d[k] for k in ("MB", "max_diff", "seconds")} | {"status": "converted"}
                    else:
                        report["errors"].append(f"{d['key']}: {d.get('error')}")
            if r.returncode != 0:
                report["errors"].append(f"ONNX export exited {r.returncode}: {r.stderr.strip()[-300:]}")
    report["probe"] = probe()
    report["ok"] = not report["errors"] and bool(report["probe"].get("ok"))
    return report


def probe() -> dict:
    """Load the plugin in VidIMG's VapourSynth and list what ONNX Runtime offers (no model is run)."""
    code = ("import json, vapoursynth as vs; c = vs.core; "
            f"c.std.LoadPlugin({str(PLUGIN)!r}); v = c.ort.Version(); "
            "print(json.dumps({k: (x.decode() if isinstance(x, bytes) else str(x)) for k, x in v.items()}))")
    r = subprocess.run([str(VS_PY), "-I", "-c", code], capture_output=True, text=True)
    try:
        return {"ok": r.returncode == 0, **json.loads(r.stdout.strip().splitlines()[-1])}
    except (ValueError, IndexError):
        return {"ok": False, "error": (r.stderr or r.stdout).strip()[-300:]}


if __name__ == "__main__":
    print(json.dumps(install(), indent=1))
