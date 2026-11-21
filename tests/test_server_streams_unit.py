"""Tools started from code that can run inside the MCP server must not inherit the server's own output
streams: those belong to the host application, and a child handed them failed to start
([WinError 50] on every video analysis over MCP, 2026-11-20)."""

import ast
import sys
from pathlib import Path

from vidimg import media

ROOT = Path(__file__).resolve().parents[1] / "vidimg"

# These run only from a terminal or inside a job process, whose streams are a console or the job's log file.
TERMINAL_OR_JOB_ONLY = {"free_install.py", "mlrt_install.py", "setup.py", "texture.py"}


def _inheriting_calls():
    found = []
    for f in sorted(ROOT.rglob("*.py")):
        if f.name in TERMINAL_OR_JOB_ONLY:
            continue
        for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and isinstance(n.func.value, ast.Name) and n.func.value.id == "subprocess"
                    and n.func.attr in ("run", "Popen", "call", "check_call", "check_output")):
                continue
            given = {k.arg for k in n.keywords}
            captured = "capture_output" in given or n.func.attr == "check_output"
            if not (captured or "stdout" in given) or not (captured or "stderr" in given):
                found.append(f"{f.relative_to(ROOT)}:{n.lineno}")
            # the input stream too: an inherited one failed the same way in Windows' handle duplication
            # ([WinError 50] / [WinError 6], intermittently, seen under the test runner 2026-11-21)
            elif f.name != "texture.py" and not {"stdin", "input"} & given:
                found.append(f"{f.relative_to(ROOT)}:{n.lineno} (stdin)")
    return found


def test_no_tool_inherits_the_servers_streams():
    assert _inheriting_calls() == []


def test_run_quiet_relays_what_the_tool_says(capfd):
    r = media.run_quiet([sys.executable, "-S", "-E", "-c",
                         "import sys; print('to stdout'); print('the reason', file=sys.stderr); sys.exit(3)"])
    out, err = capfd.readouterr()
    assert r.returncode == 3
    assert "to stdout" not in out                 # the child's output goes nowhere
    assert "the reason" in err                    # its errors are passed on by Python, not by inheritance


def test_run_quiet_survives_a_missing_error_stream(monkeypatch):
    monkeypatch.setattr(sys, "stderr", None)      # a windowless host gives Python no error stream at all
    r = media.run_quiet([sys.executable, "-S", "-E", "-c", "import sys; print('x', file=sys.stderr)"])
    assert r.returncode == 0
