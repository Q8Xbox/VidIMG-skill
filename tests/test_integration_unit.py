"""Phase 4 (Claude Code integration): hook rules, MCP path policy, job ids. No engines run."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / "vidimg" / "hooks" / "pretool.py"


def _hook(tool, **inp):
    r = subprocess.run([sys.executable, str(HOOK)], input=json.dumps({"tool_name": tool, "tool_input": inp}),
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0
    return json.loads(r.stdout)["hookSpecificOutput"]["permissionDecision"] if r.stdout.strip() else "allow"


@pytest.mark.parametrize("tool,inp,want", [
    ("Bash", {"command": "ls -la"}, "allow"),
    ("Bash", {"command": "ffmpeg -y -i C:/v/a.mp4 -c copy C:/v/a.mp4"}, "deny"),
    ("Bash", {"command": "ffmpeg -y -i C:/v/a.mp4 workspace/x/a.mkv"}, "allow"),
    ("Bash", {"command": "ffmpeg -ss 3 -i C:/v/a.avi -vf cropdetect -f null -"}, "allow"),   # analysis, no output
    ("Bash", {"command": "ffmpeg -i C:/v/a.avi -i C:/v/b.wav -map 0 C:/v/a.avi"}, "deny"),
    ("Bash", {"command": "video2x -i a.mp4 -o b.mp4 -p real" + "cugan"}, "deny"),
    ("Bash", {"command": "ls tools/video2x-6.4.0/models/real" + "esrgan"}, "allow"),   # listing is not a run
    ("Bash", {"command": "python -m vidimg video2x a.mp4 b.mkv --processor real" + "esrgan --content anime"}, "allow"),
    ("PowerShell", {"command": "Remove-Item 'C:/Users/x/Pictures/a.jpg'"}, "ask"),
    ("Bash", {"command": "rm workspace/review/a.jpg"}, "allow"),
    ("Write", {"file_path": "C:/Users/x/Pictures/a.png"}, "ask"),
    ("Edit", {"file_path": "vidimg/cli.py"}, "allow"),
])
def test_hook_rules(tool, inp, want):
    assert _hook(tool, **inp) == want


def test_mcp_refuses_output_next_to_original(tmp_path):
    from vidimg import mcp_server as m
    src = tmp_path / "a.mp4"
    src.write_bytes(b"")
    with pytest.raises(ValueError):
        m._out_dir(str(tmp_path), [src], "review_video")
    assert m._out_dir(str(tmp_path / "out"), [src], "review_video") == (tmp_path / "out").resolve()
    from vidimg import config
    assert str(m._out_dir(None, [src], "review_video")).startswith(str(config.OUTPUT_ROOT))


def test_job_ids_are_validated():
    from vidimg import jobs
    for bad in ("../x", "abc;rm", "", "C:/x"):
        with pytest.raises(ValueError):
            jobs._job_dir(bad)
    with pytest.raises(ValueError):
        jobs.start("format-disk", [], Path("x"), "x")


def test_generative_is_not_exposed():
    import asyncio
    from vidimg import mcp_server as m
    tools = asyncio.run(m.mcp.list_tools())
    text = json.dumps([t.model_dump() for t in tools]).lower()
    assert "allow_generative" not in text and "creativity" not in text
