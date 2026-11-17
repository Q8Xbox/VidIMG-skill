"""Build / verify the isolated VapourSynth environment (the second classical engine).

* tools/vs-venv: a private virtual environment; nothing is installed system-wide and the main
  tool's Python never imports VapourSynth (vidimg/vs_runner.py runs inside this environment).
* VapourSynth core pinned to R72: R73+ refuses API 3 plugins, and MVTools / TIVTC have no API 4
  Windows builds yet (vsrepo win64: MVTools v24, TIVTC v2; GitHub v26+/v4 ship no Windows binaries).
* Plugin versions are pinned in PINNED and every installed file's SHA-256 is recorded in
  tools/vapoursynth/lock.json on first install (trust on first use); later runs refuse changes.

Usage: python -m vidimg.vs_install            (install or verify)
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from . import config

VENV = config.TOOLS_DIR / "vs-venv"
PY = VENV / "Scripts" / "python.exe"
VSREPO = VENV / "Scripts" / "vsrepo.exe"
LOCK = config.MANIFEST_DIR / "vapoursynth" / "lock.json"
PIP_PINS = ["vapoursynth==72", "vsrepo==2.0.0", "vsutil==0.8.0"]
# vsrepo namespaces to install, and the versions their pip metadata must report afterwards
VSREPO_PACKAGES = ["lsmas", "mv", "znedi3", "nnedi3", "fmtc", "tivtc", "vivtc", "bwdif", "havsfunc", "zsmooth"]
PINNED = {"LSMASHSource": "1296.0.0.0", "MVTools": "24", "ZNEDI3": "3.3", "NNEDI3": "12", "fmtconv": "30",
          "TIVTC": "2", "VIVTC": "1", "Bwdif": "4.1", "havsfunc": "0+5baf9c2", "Zsmooth": "0.20.0",
          "VapourSynth": "72", "vsutil": "0.8.0"}


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _run(args: list, **kw) -> subprocess.CompletedProcess:
    return subprocess.run([str(a) for a in args], capture_output=True, text=True, stdin=subprocess.DEVNULL,
                          errors="replace", **kw)


def _versions() -> dict[str, str]:
    if not PY.exists():
        return {}
    r = _run([PY, "-m", "pip", "list", "--format", "json"])
    try:
        return {d["name"]: d["version"] for d in json.loads(r.stdout or "[]")}
    except ValueError:
        return {}


def files() -> list[Path]:
    site = VENV / "Lib" / "site-packages"
    plug = site / "vapoursynth" / "plugins" / "vsrepo"
    return sorted([*plug.glob("*.dll"), *[site / f for f in ("havsfunc.py", "mvsfunc", "vsutil") if (site / f).exists()]])


def _digest(p: Path) -> str:
    if p.is_dir():
        h = hashlib.sha256()
        for f in sorted(p.rglob("*.py")):
            h.update(f.name.encode() + bytes.fromhex(_sha(f)))
        return h.hexdigest()
    return _sha(p)


def install(log=print) -> dict:
    if not PY.exists():
        log("creating tools/vs-venv")
        _run([sys.executable, "-m", "venv", VENV], check=True)
    # vsrepo 2.0.0 only runs on core R80+ (vapoursynth.get_plugin_dir) while the plugins need R72 (API 3):
    # install the plugins under R82, then pin the runtime core to R72 without dependency resolution
    # (vsstubs, pulled in by vsrepo, asks for R80+ but only generates editor stubs)
    have = _versions()
    if any(have.get(k) != v for k, v in PINNED.items() if k not in ("VapourSynth",)):
        for args in (["vapoursynth==82", "vsrepo==2.0.0", "vsutil==0.8.0"],):
            r = _run([PY, "-m", "pip", "install", "-q", *args])
            if r.returncode:
                raise SystemExit(r.stdout + r.stderr)
        _run([VSREPO, "update"])
        _run([VSREPO, "install", *VSREPO_PACKAGES])
    r = _run([PY, "-m", "pip", "install", "-q", "--no-deps", "vapoursynth==72"])
    if r.returncode:
        raise SystemExit(r.stdout + r.stderr)
    installed = _versions()
    drift = {k: (installed.get(k), v) for k, v in PINNED.items() if installed.get(k) != v}
    lock = json.loads(LOCK.read_text(encoding="utf-8")) if LOCK.exists() else {}
    changed = []
    for f in files():
        key = str(f.relative_to(VENV)).replace("\\", "/")
        d = _digest(f)
        if key in lock and lock[key] != d:
            changed.append(key)
        lock.setdefault(key, d)
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps(lock, indent=1, sort_keys=True), encoding="utf-8")
    # functional check: the plugins the backend needs load and havsfunc imports
    probe = _run([PY, "-I", "-c", "import vapoursynth as vs; c=vs.core; "
                  f"c.std.LoadAllPlugins(r'{VENV / 'Lib/site-packages/vapoursynth/plugins/vsrepo'}'); "
                  "import havsfunc; print(','.join(n for n in ('lsmas','mv','znedi3','tivtc','bwdif') if hasattr(c, n)))"])
    loaded = probe.stdout.strip().split(",") if probe.returncode == 0 else []
    report = {"pip": PIP_PINS, "versions": installed, "drift": drift, "hash_changes": changed,
              "plugins_loaded": loaded, "ok": not drift and not changed and len(loaded) == 5}
    return report


if __name__ == "__main__":
    print(json.dumps(install(), indent=1))
