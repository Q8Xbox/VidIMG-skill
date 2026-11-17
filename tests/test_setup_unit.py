"""First-run setup and install layouts: a cloned project vs a Claude Code plugin install."""

import json
import subprocess
import sys
from pathlib import Path


def _layout(monkeypatch, code: Path, home: Path):
    from vidimg import config
    monkeypatch.setattr(config, "PROJECT_ROOT", code)
    monkeypatch.setattr(config, "HOME", home)
    monkeypatch.setattr(config, "WORKSPACE", home / "workspace")
    monkeypatch.setattr(config, "SETTINGS_FILE", home / "vidimg.local.json")


def test_clone_points_mcp_here_and_never_rewrites_settings(tmp_path, monkeypatch):
    from vidimg import setup
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".mcp.json").write_text(json.dumps({"mcpServers": {"vidimg": {"command": "python", "args": ["C:/old/x.py"]},
                                                                   "other": {"command": "x"}}}))
    settings = json.dumps({"hooks": {"PreToolUse": []}, "permissions": {"allow": ["Read"]}})
    (tmp_path / ".claude" / "settings.json").write_text(settings)
    _layout(monkeypatch, tmp_path, tmp_path)
    r = setup.step_claude_code(print, output=str(tmp_path / "out"), yes=True)
    assert r["ok"] and (tmp_path / "out").is_dir()
    mcp = json.loads((tmp_path / ".mcp.json").read_text())
    assert mcp["mcpServers"]["other"] == {"command": "x"}                       # other servers untouched
    v = mcp["mcpServers"]["vidimg"]
    assert v["command"] == Path(sys.executable).as_posix()
    assert v["args"] == [(tmp_path / "vidimg" / "mcp_server.py").as_posix()]
    assert (tmp_path / ".claude" / "settings.json").read_text() == settings      # committed file untouched
    assert json.loads((tmp_path / "vidimg.local.json").read_text())["output"] == (tmp_path / "out").resolve().as_posix()


def test_plugin_install_keeps_data_outside_the_plugin_cache(tmp_path, monkeypatch):
    from vidimg import setup
    code = tmp_path / ".claude" / "plugins" / "cache" / "vidimg" / "vidimg" / "0.2.0"
    code.mkdir(parents=True)
    home = tmp_path / "LocalAppData" / "VidIMG"
    _layout(monkeypatch, code, home)
    assert setup._is_plugin()
    r = setup.step_claude_code(print, output=str(tmp_path / "out"), yes=True)
    assert r["ok"] and not (code / ".mcp.json").exists()                        # the plugin declares the server
    assert json.loads((home / "vidimg.local.json").read_text())["output"] == (tmp_path / "out").resolve().as_posix()
    assert setup.step_claude_code(print, check=True)["ok"]


def test_home_rule(monkeypatch):
    import importlib
    from vidimg import config
    monkeypatch.delenv("VIDIMG_HOME", raising=False)
    assert importlib.reload(config).HOME == config.PROJECT_ROOT                # a clone keeps everything in place
    monkeypatch.setenv("VIDIMG_HOME", "Q:/vidimg-home")
    try:
        c = importlib.reload(config)
        assert c.HOME == Path("Q:/vidimg-home") and c.TOOLS_DIR == c.HOME / "tools"
        assert c.MANIFEST_DIR == c.PROJECT_ROOT / "tools"                     # pinned manifests ship with the code
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_output_on_a_missing_drive_falls_back(monkeypatch):
    import importlib
    from vidimg import config
    monkeypatch.setenv("VIDIMG_OUTPUT", "Q:/surely/not/here")
    try:
        assert importlib.reload(config).OUTPUT_ROOT == config.WORKSPACE
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_hook_trusts_the_output_folder_chosen_at_setup(tmp_path):
    hook = Path(__file__).resolve().parents[1] / "vidimg" / "hooks" / "pretool.py"
    (tmp_path / "vidimg.local.json").write_text(json.dumps({"output": (tmp_path / "out").as_posix()}))
    env = {k: v for k, v in __import__("os").environ.items() if k != "VIDIMG_OUTPUT"}
    env["VIDIMG_HOME"] = str(tmp_path)

    def decide(path):
        r = subprocess.run([sys.executable, str(hook)], input=json.dumps({"tool_name": "Bash", "tool_input": {
            "command": f"rm '{path}'"}}), capture_output=True, text=True, env=env, timeout=30)
        return json.loads(r.stdout)["hookSpecificOutput"]["permissionDecision"] if r.stdout.strip() else "allow"
    assert decide((tmp_path / "out" / "a.mp4").as_posix()) == "allow"
    assert decide((tmp_path / "workspace" / "a.mp4").as_posix()) == "allow"
    assert decide("C:/Users/x/Videos/a.mp4") == "ask"
