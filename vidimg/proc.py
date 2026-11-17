"""Subprocess runner: every external call is logged as a JSON evidence record."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

from . import config


@dataclass
class RunResult:
    tool: str
    argv: list[str]
    returncode: int
    seconds: float
    stdout_tail: list[str] = field(default_factory=list)
    log_path: str | None = None

    def text(self) -> str:
        return "\n".join(self.stdout_tail)


class PaidSoftwareRefused(RuntimeError):
    """Something tried to start a program from a paid software installation."""


PAID_INSTALL_MARKERS = ("topaz labs llc",)


def _refuse_paid_software(exe: str) -> None:
    """Hard guarantee (free tools only): no program inside a paid software installation is ever started,
    so a hidden dependency fails loudly instead of working only on machines that happen to have it."""
    if any(m in str(exe).replace("/", "\\").lower() for m in PAID_INSTALL_MARKERS):
        raise PaidSoftwareRefused(f"refused to run paid software: {exe}")


def run(
    tool: str,
    argv: Iterable[str | os.PathLike],
    *,
    env: dict[str, str] | None = None,
    cwd: str | os.PathLike | None = None,
    timeout: float | None = None,
    on_line: Callable[[str], None] | None = None,
    stdin_text: str | None = None,
    stdin_pipe=None,
    keep_lines: int = 400,
) -> RunResult:
    """Run a program, stream merged stdout/stderr line by line, keep the tail, log the call."""
    args = [str(a) for a in argv]
    _refuse_paid_software(args[0])
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    tail: deque[str] = deque(maxlen=keep_lines)
    start = time.monotonic()
    proc = subprocess.Popen(
        args,
        cwd=cwd,
        env=full_env,
        stdin=stdin_pipe if stdin_pipe is not None else (subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if stdin_text is not None and proc.stdin:
        proc.stdin.write(stdin_text)
        proc.stdin.close()

    def pump() -> None:
        assert proc.stdout is not None
        # ffmpeg writes progress with '\r'; split on both so callers see every update
        buf = ""
        while True:
            chunk = proc.stdout.read(1)
            if not chunk:
                break
            if chunk in "\r\n":
                if buf:
                    tail.append(buf)
                    if on_line:
                        on_line(buf)
                buf = ""
            else:
                buf += chunk
        if buf:
            tail.append(buf)
            if on_line:
                on_line(buf)

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    try:
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        rc = proc.wait()
        tail.append(f"[vidimg] killed after timeout of {timeout}s")
    reader.join(timeout=5)
    result = RunResult(tool, args, rc, round(time.monotonic() - start, 2), list(tail))
    result.log_path = str(_log(result, env))
    return result


def _log(result: RunResult, env: dict[str, str] | None) -> Path:
    config.ensure_workspace()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = config.LOG_DIR / f"{stamp}_{result.tool}.json"
    record = asdict(result) | {"env_overrides": env or {}, "time": stamp}
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return path
