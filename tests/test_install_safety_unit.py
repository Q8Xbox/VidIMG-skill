"""Installers and calibration refuse what they report as wrong; versions and docs agree with the code.
Nothing is downloaded or installed."""

import json
import re
import sys
import tomllib
from pathlib import Path

import vidimg
from vidimg import calibrate, free_install, guard, setup as vsetup, vs_install
from vidimg.adapters import vapoursynth as vsb

ROOT = Path(__file__).parent.parent


def test_a_model_with_a_wrong_hash_is_set_aside(tmp_path, monkeypatch):
    manifest, lock = tmp_path / "manifest.json", tmp_path / "lock.json"
    url = "https://example.invalid/models/net_x4.pth"
    manifest.write_text(json.dumps({"models": {"net": {"url": url, "license": "MIT"}}}), encoding="utf-8")
    model = tmp_path / "net_x4.pth"
    model.write_bytes(b"tampered")
    lock.write_text(json.dumps({url: "0" * 64}), encoding="utf-8")
    for name, value in (("MANIFEST", manifest), ("LOCK", lock), ("ROOT", tmp_path), ("PY", Path(sys.executable))):
        monkeypatch.setattr(free_install, name, value)
    monkeypatch.setattr(free_install.subprocess, "run",
                        lambda *a, **k: type("R", (), {"returncode": 0, "stdout": "2.14.1+cu130 True gpu"})())
    report = free_install.install(log=lambda *_: None)
    assert not report["ok"] and "set aside" in report["errors"][0]
    assert not model.exists() and (tmp_path / "net_x4.pth.rejected").read_bytes() == b"tampered"
    assert json.loads(lock.read_text(encoding="utf-8"))[url] == "0" * 64          # the reference is not overwritten


def test_changed_vapoursynth_files_disable_the_engine(tmp_path, monkeypatch):
    python, plugins = tmp_path / "python.exe", tmp_path / "plugins"
    plugins.mkdir()
    python.write_bytes(b"")
    (plugins / "libmvtools.dll").write_bytes(b"")
    monkeypatch.setattr(vsb, "PYTHON", python)
    monkeypatch.setattr(vsb, "PLUGINS", plugins)
    monkeypatch.setattr(vsb.avs, "system_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(vs_install, "VIOLATION", tmp_path / "lock_violation.json")
    assert vsb.available() == (True, "ok")
    vs_install.VIOLATION.write_text('{"changed": ["x.dll"]}', encoding="utf-8")
    ok, why = vsb.available()
    assert not ok and "--relock" in why


def test_calibration_cannot_apply_a_degenerate_threshold():
    defaults = {k: v for k, v in guard.DEFAULT_THRESHOLDS.items() if isinstance(v, float)}
    assert set(defaults) == set(calibrate.APPLY_BOUNDS)                 # every threshold has a range
    for k, v in defaults.items():
        lo, hi = calibrate.APPLY_BOUNDS[k]
        assert lo < v < hi, k                                           # and the default sits inside it
    lo, hi = calibrate.APPLY_BOUNDS["identity_min"]
    assert not lo <= 0.0 <= hi and not lo <= 1.0 <= hi                  # "always pass" / "always fail" are out


def test_setup_leaves_a_complete_python_alone_and_asks_before_installing(monkeypatch):
    calls = []

    def run(args, **kw):
        calls.append(args)
        return type("R", (), {"returncode": 0 if state["have"] else 1, "stdout": "", "stderr": ""})()

    state = {"have": True}
    monkeypatch.setattr(vsetup.subprocess, "run", run)
    assert vsetup.step_python_packages(lambda *_: None)["ok"] and len(calls) == 1       # only the import check
    state["have"] = False
    calls.clear()
    monkeypatch.setattr(vsetup.sys, "prefix", "C:/Python")                 # not a virtual environment
    monkeypatch.setattr(vsetup.sys, "base_prefix", "C:/Python")
    monkeypatch.setattr(vsetup, "_ask", lambda question, yes: False)       # the person says no
    r = vsetup.step_python_packages(lambda *_: None)
    assert not r["ok"] and "pip install -r" in r["notes"][0]
    assert not any("pip" in map(str, c) for c in calls)                    # nothing was installed


def test_versions_and_docs_agree_with_the_code():
    plugin = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))["version"]
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert vidimg.__version__ == plugin == project["project"]["version"]
    assert project["build-system"]["build-backend"] == "setuptools.build_meta"
    wanted = {re.split(r"[=<>]", line.split("#")[0].strip())[0].lower()
              for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
              if line.strip() and not line.startswith("#")} - {"pytest"}
    declared = {re.split(r"[=<>]", d)[0].lower() for d in project["project"]["dependencies"]}
    assert wanted <= declared, wanted - declared                         # what setup installs, the package declares
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    tools = re.findall(r"@mcp\.tool\(\)\s+def (\w+)", (ROOT / "vidimg" / "mcp_server.py").read_text(encoding="utf-8"))
    assert tools and all(f"`{t}`" in readme for t in tools), [t for t in tools if f"`{t}`" not in readme]
    for cmd in (ROOT / ".claude" / "commands").glob("*.md"):
        assert f"`/{cmd.stem}" in readme, cmd.stem
