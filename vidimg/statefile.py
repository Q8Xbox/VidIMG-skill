"""Small JSON state files shared between processes (job state, the plan): written atomically, and changed under a
lock so that two processes never lose each other's update.

Before this module each writer replaced the whole file with its own picture of it: a job's worker overwrote a cancel
request that arrived in its first second, the plan runner overwrote decisions made while it ran, and two writers
using the same temporary name could crash each other."""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator


def read_json(path: Path, default=None):
    """The file's content, or `default` ({} when not given) when it is missing or not readable yet."""
    for attempt in range(3):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError:
            break
        except (OSError, ValueError):
            time.sleep(0.05 * (attempt + 1))        # another process is replacing it right now
    return {} if default is None else default


def write_json(path: Path, data) -> None:
    """Write through a temporary file with a unique name, then replace. The replace is retried: on Windows it is
    refused for a moment while another process holds the target open."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:6]}.tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    for attempt in range(40):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            time.sleep(0.05)
    tmp.unlink(missing_ok=True)
    raise PermissionError(f"could not replace {path}: it stays open in another program")


@contextmanager
def locked(path: Path, timeout: float = 30.0) -> Iterator[None]:
    """Exclusive lock for read-change-write of `path`, across processes (a lock file next to it)."""
    lock = Path(path).with_name(Path(path).name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    f = open(lock, "a+b")
    try:
        try:
            import msvcrt
        except ImportError:                          # not Windows: the tool does not run there, tests may
            msvcrt = None
        if msvcrt is not None:
            end = time.monotonic() + timeout
            while True:
                try:
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() > end:
                        raise TimeoutError(f"{lock} is held by another process") from None
                    time.sleep(0.05)
        try:
            yield
        finally:
            if msvcrt is not None:
                try:
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
    finally:
        f.close()


def update(path: Path, change: Callable[[dict], dict | None] | None = None, **fields) -> dict:
    """Read, change and write `path` in one locked step. `fields` are merged in; `change(data)` may edit the dict in
    place or return a new one. Returns what was written."""
    with locked(path):
        data = read_json(path)
        if change is not None:
            data = change(data) or data
        data.update(fields)
        write_json(path, data)
        return data
