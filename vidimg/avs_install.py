"""Install the pinned private AviSynth+ plugin set (tools/avisynth/manifest.json).

* downloads only the URLs in the manifest (official GitHub releases, fftw.org, pinned commits);
* trust-on-first-use: SHA-256 of every download is written to tools/avisynth/lock.json and every
  later install must match it exactly;
* archives are extracted with 7-Zip / zipfile (nothing is executed); only the x64 DLLs named in the
  manifest are copied into tools/avisynth/plugins;
* the system AviSynth+ folders are never modified.

Usage: python -m vidimg.avs_install            (install/verify)
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from . import config

ROOT = config.TOOLS_DIR / "avisynth"
MANIFEST = config.MANIFEST_DIR / "avisynth" / "manifest.json"
LOCK = config.MANIFEST_DIR / "avisynth" / "lock.json"
PLUGINS = ROOT / "plugins"
DOWNLOADS = ROOT / "downloads"
_BAD = re.compile(r"(^|[\\/_\-. ])(x86|win32|xp|arm64|arm|aarch64|avx512|i386)([\\/_\-. ]|$)", re.I)


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _download(url: str, dst: Path, attempts: int = 4) -> Path:
    """Atomic, retried download: data goes to <dst>.part and is renamed only when the byte count
    matches Content-Length, so an interrupted transfer can never be mistaken for a finished file
    (and never poisons lock.json). GitHub release CDNs reset long transfers now and then (WinError 10054)."""
    import time
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst
    part = dst.with_name(dst.name + ".part")
    last: Exception | None = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "vidimg-avs-install"})
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as f:
                want = int(r.headers.get("Content-Length") or 0)
                shutil.copyfileobj(r, f, 1 << 20)
            got = part.stat().st_size
            if want and got != want:
                raise IOError(f"short read {got}/{want} bytes")
            part.replace(dst)
            return dst
        except Exception as e:  # noqa: BLE001 - network errors of every kind are retried
            last = e
            part.unlink(missing_ok=True)
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"download failed after {attempts} attempts: {url}: {last}")


SEVEN_ZIP = Path(r"C:\Program Files\7-Zip\7z.exe")


def _extract(archive: Path, into: Path) -> None:
    """Extract without running anything from the archive. Most plugin archives use the BCJ2 filter,
    which py7zr cannot decode (observed), so 7-Zip (winget 7zip.7zip) is used when present."""
    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(into)
        return
    if SEVEN_ZIP.exists():
        import subprocess
        r = subprocess.run([str(SEVEN_ZIP), "x", "-y", f"-o{into}", str(archive)], capture_output=True, text=True,
                           stdin=subprocess.DEVNULL)
        if r.returncode != 0:
            raise RuntimeError(f"7-Zip failed on {archive.name}: {r.stderr.strip() or r.stdout[-300:]}")
        return
    import py7zr
    with py7zr.SevenZipFile(archive, "r") as z:
        z.extractall(into)


def _pick_x64(root: Path, name: str) -> Path | None:
    """The x64 build of a DLL inside an extracted archive (archives also ship x86/XP/ARM64 builds)."""
    cands = [p for p in root.rglob("*") if p.is_file() and p.name.lower() == name.lower()]
    good = [p for p in cands if not _BAD.search(str(p.relative_to(root)))]
    pref = [p for p in good if re.search(r"x64|amd64|win64|64", str(p.relative_to(root)), re.I)]
    for group in (pref, good):
        if group:
            # prefer plain builds over CPU-specific ones (e.g. "AVX2"/"clang" variants) when both exist
            return sorted(group, key=lambda p: (bool(re.search(r"avx2|clang|icl", str(p), re.I)), len(str(p))))[0]
    return None


def install(log=print) -> dict:
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8")) if LOCK.exists() else {}
    PLUGINS.mkdir(parents=True, exist_ok=True)
    report = {"installed": [], "errors": []}
    for item in man["archives"]:
        arc = _download(item["url"], DOWNLOADS / item["url"].rsplit("/", 1)[-1])
        digest = _sha(arc)
        key = item["url"]
        if key in lock and lock[key] != digest:
            report["errors"].append(f"{item['name']}: SHA-256 changed since first download ({arc.name}); refusing")
            continue
        lock[key] = digest
        with tempfile.TemporaryDirectory(prefix="vidimg_avs_") as tmp:
            _extract(arc, Path(tmp))
            for dll in item["dlls"]:
                p = _pick_x64(Path(tmp), dll)
                if not p:
                    report["errors"].append(f"{item['name']}: {dll} (x64) not found in {arc.name}")
                    continue
                shutil.copyfile(p, PLUGINS / dll)
                report["installed"].append({"file": dll, "from": f"{arc.name}:{p.relative_to(tmp)}", "version": item["version"]})
        log(f"  {item['name']:<14} {item['version']:<10} ok")
    for item in man.get("local_dlls", []):
        src = Path(item["from"])
        if src.exists():
            shutil.copyfile(src, PLUGINS / src.name)
            lock[f"local:{src}"] = _sha(src)
            report["installed"].append({"file": src.name, "from": str(src), "version": "local"})
    for repo, spec in man["scripts"].items():
        for f in spec["files"]:
            url = f"https://raw.githubusercontent.com/{repo}/{spec['commit']}/{urllib.request.quote(f)}"
            dst = _download(url, DOWNLOADS / "scripts" / repo.replace("/", "_") / Path(f).name)
            digest = _sha(dst)
            if url in lock and lock[url] != digest:
                report["errors"].append(f"{f}: SHA-256 changed; refusing")
                continue
            lock[url] = digest
            shutil.copyfile(dst, PLUGINS / Path(f).name)
            report["installed"].append({"file": Path(f).name, "from": f"{repo}@{spec['commit'][:8]}", "version": "script"})
        log(f"  {repo:<30} {len(spec['files'])} scripts")
    LOCK.write_text(json.dumps(lock, indent=1, sort_keys=True), encoding="utf-8")
    (ROOT / "installed.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


if __name__ == "__main__":
    r = install()
    print(f"\n{len(r['installed'])} files installed, {len(r['errors'])} errors")
    for e in r["errors"]:
        print("  ERROR", e)
