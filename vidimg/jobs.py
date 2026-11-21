"""Background jobs for long restorations (phase 4: Claude Code / MCP integration).

A restore takes minutes, longer than a tool call should block, so the MCP server starts a detached worker
(`python -m vidimg.jobs run <id>`) and returns at once. Everything a job knows lives in files under
workspace/jobs/<id>/ (meta.json, state.json, log.txt), so status survives an MCP server restart.
Only one GPU job runs at a time: workers take an exclusive lock (workspace/jobs/gpu.lock) and report
"queued" while they wait.
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from . import config, statefile

ROOT = config.OUTPUT_ROOT / "jobs"
LOCK = ROOT / "gpu.lock"
KINDS = {"restore-photos": "restore", "restore-video": "restore-video", "survey": "auto", "run-plan": "run-plan",
         "try-recipes": "try-recipes"}


def _job_dir(job_id: str) -> Path:
    if not job_id or any(c not in "0123456789abcdef-" for c in job_id):
        raise ValueError(f"invalid job id {job_id!r}")
    return ROOT / job_id


def _write(path: Path, data: dict) -> None:
    statefile.write_json(path, data)


def _read(path: Path) -> dict:
    return statefile.read_json(path)


def _set(d: Path, **fields) -> dict:
    """Merge fields into the job's state. Every writer merges: a writer that replaced the whole state dropped a
    cancel request another process had just recorded."""
    return statefile.update(d / "state.json", **fields)


def _proc(pid: int | None) -> tuple[bool, int | None]:
    """(still running, creation time) of a process."""
    if not pid:
        return False, None
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, int(pid))          # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False, None
    try:
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        t = [ctypes.c_ulonglong() for _ in range(4)]
        made = t[0].value if k32.GetProcessTimes(h, *(ctypes.byref(x) for x in t)) else None
        return bool(ok) and code.value == 259, made        # STILL_ACTIVE
    finally:
        k32.CloseHandle(h)


def _born(pid: int | None) -> int | None:
    return _proc(pid)[1]


def _alive(pid: int | None, born: int | None = None) -> bool:
    """The process recorded for a job is still running. `born` is its creation time as recorded with the pid:
    Windows reuses process ids, so after a reboot a finished job's id can belong to any other program, which kept
    the job "running" for ever and could hold the queue. Without `born` (state from an older version) the id alone
    decides."""
    alive, made = _proc(pid)
    return alive and (born is None or made is None or made == born)


def start(kind: str, args: list[str], out_dir: Path, label: str) -> dict:
    """Create a job and launch its detached worker. `args` are the CLI arguments after the subcommand."""
    if kind not in KINDS:
        raise ValueError(f"unknown job kind {kind!r}")
    job_id = datetime.now().strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:6]
    d = _job_dir(job_id)
    d.mkdir(parents=True)
    meta = {"id": job_id, "kind": kind, "label": label, "out": str(out_dir), "created": datetime.now().isoformat(timespec="seconds"),
            "created_ns": time.time_ns(),
            "argv": [sys.executable, "-m", "vidimg", KINDS[kind], *args, "--out", str(out_dir)]}
    _write(d / "meta.json", meta)
    _write(d / "state.json", {"state": "starting"})
    flags = 0x00000008 | 0x00000200 | 0x08000000           # DETACHED_PROCESS | NEW_PROCESS_GROUP | NO_WINDOW
    with open(d / "worker.txt", "wb") as wlog:
        p = subprocess.Popen([sys.executable, "-m", "vidimg.jobs", "run", job_id], cwd=str(config.PROJECT_ROOT),
                             stdin=subprocess.DEVNULL, stdout=wlog, stderr=subprocess.STDOUT, creationflags=flags)
    # only the worker's identity is added: the worker may already have written "queued" or "running" itself
    _set(d, worker_pid=p.pid, worker_born=_born(p.pid))
    return status(job_id)


def _queue_key(d: Path) -> tuple:
    m = _read(d / "meta.json")
    return (m.get("created_ns") or 0, d.name)


def _older_waiting(d: Path) -> bool:
    """First come, first served: another live job queued earlier is still waiting for the GPU (the bare file lock
    let whichever waiter polled first win: the owner's 4K Sample02 overtook b001, 2026-11-16)."""
    mine = _queue_key(d)
    for other in ROOT.iterdir():
        if other == d or not (other / "meta.json").exists():
            continue
        st = _read(other / "state.json")
        if st.get("state") in ("starting", "queued") and not st.get("cancel") and _alive(st.get("worker_pid"), st.get("worker_born"))                 and _queue_key(other) < mine:
            return True
    return False


def _acquire_lock(d: Path):
    import msvcrt
    ROOT.mkdir(parents=True, exist_ok=True)
    f = open(LOCK, "a+b")
    while True:
        try:
            if _older_waiting(d):
                raise OSError("an older job is waiting")
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            return f
        except OSError:
            st = _read(d / "state.json")
            if st.get("cancel"):
                f.close()
                return None
            _set(d, state="queued", worker_pid=os.getpid(), worker_born=_born(os.getpid()))
            time.sleep(3)


def run(job_id: str) -> int:
    """Worker body: wait for the GPU lock, run the CLI, record the outcome."""
    d = _job_dir(job_id)
    meta = _read(d / "meta.json")
    _set(d, worker_pid=os.getpid(), worker_born=_born(os.getpid()))
    lock = _acquire_lock(d)
    now = lambda: datetime.now().isoformat(timespec="seconds")  # noqa: E731
    if lock is None or _read(d / "state.json").get("cancel"):   # cancelled while waiting, or just before the start
        if lock is not None:
            lock.close()
        _set(d, state="cancelled", finished=now())
        return 1
    try:
        with open(d / "log.txt", "w", encoding="utf-8", errors="replace") as log:
            p = subprocess.Popen(meta["argv"], cwd=str(config.PROJECT_ROOT), stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, env={**os.environ, "PYTHONIOENCODING": "utf-8",
                                                                "PYTHONUNBUFFERED": "1"},
                                 creationflags=0x08000000)
            st = _set(d, state="running", pid=p.pid, pid_born=_born(p.pid), started=now())
            if st.get("cancel"):
                # the request arrived between the check above and the start: cancel() saw no process to stop
                subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True, stdin=subprocess.DEVNULL)
            code = p.wait()
        st = _read(d / "state.json")
        final = "cancelled" if st.get("cancel") else ("done" if code == 0 else "failed")
        _set(d, state=final, returncode=code, finished=now())
        return code
    finally:
        lock.close()


def _tail(path: Path, n: int = 12) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    lines = [ln for ln in text.replace("\r", "\n").splitlines() if ln.strip() and "WARN:" not in ln]
    return lines[-n:]


def _result(meta: dict) -> dict:
    out = Path(meta["out"])
    if meta["kind"] in ("survey", "run-plan", "try-recipes"):
        from . import auto
        try:
            return auto.summary(auto.load(out))
        except (OSError, ValueError):
            return {}
    if meta["kind"] == "restore-video":
        rec = _read(out / "record.json")
        if not rec:
            return {}
        return {"passed": rec.get("passed"), "delivered": rec.get("delivered"), "chosen": rec.get("chosen"),
                "plan": rec.get("plan"), "prepass": {k: rec.get("prepass", {}).get(k) for k in ("plan", "engine", "guard_passed")},
                "final_gates": {k: (g.get("status"), g.get("value")) for k, g in (rec.get("final_gates") or {}).items()},
                "notes": rec.get("notes"), "report": str(out / "report.html"), "sheets": rec.get("sheets")}
    recs = _read(out / "records.json")
    if not recs:
        return {}
    items = [{"source": r["source"], "passed": r["passed"], "delivered": r["delivered"], "plan": r["plan"],
              "notes": r.get("notes")} for r in recs]
    return {"passed": sum(r["passed"] for r in recs), "total": len(recs), "items": items,
            "report": str(out / "sheets" / "index.html")}


def status(job_id: str) -> dict:
    d = _job_dir(job_id)
    meta, st = _read(d / "meta.json"), _read(d / "state.json")
    if not meta:
        raise FileNotFoundError(f"no job {job_id}")
    state = st.get("state", "unknown")
    if state in ("starting", "queued", "running") and not _alive(st.get("worker_pid"), st.get("worker_born")) \
            and not _alive(st.get("pid"), st.get("pid_born")):
        state = "crashed"
    res = {"id": job_id, "kind": meta["kind"], "label": meta["label"], "state": state, "out": meta["out"],
           "created": meta["created"], "started": st.get("started"), "finished": st.get("finished"),
           "returncode": st.get("returncode"), "resumed": st.get("resumed", 0), "log_tail": _tail(d / "log.txt")}
    if state in ("done", "failed") or (meta["kind"] == "run-plan" and state in ("running", "cancelled", "crashed")):
        res["result"] = _result(meta)                      # a running plan shows which items are finished
    return res


def list_jobs(limit: int = 20) -> list[dict]:
    if not ROOT.exists():
        return []
    ids = sorted((p.name for p in ROOT.iterdir() if p.is_dir()), reverse=True)[:limit]
    out = []
    for i in ids:
        try:
            s = status(i)
            out.append({k: s[k] for k in ("id", "kind", "label", "state", "created", "finished")})
        except (FileNotFoundError, ValueError):
            continue
    return out


def _launch(job_id: str) -> int:
    d = _job_dir(job_id)
    flags = 0x00000008 | 0x00000200 | 0x08000000           # DETACHED_PROCESS | NEW_PROCESS_GROUP | NO_WINDOW
    with open(d / "worker.txt", "ab") as wlog:
        p = subprocess.Popen([sys.executable, "-m", "vidimg.jobs", "run", job_id], cwd=str(config.PROJECT_ROOT),
                             stdin=subprocess.DEVNULL, stdout=wlog, stderr=subprocess.STDOUT, creationflags=flags)
    return p.pid


def resume(job_id: str) -> dict:
    """Restart a stopped job (cancelled, crashed or failed) into the same output folder: photo batches
    skip finished photos (per-photo checkpoints) and long videos skip finished chunks."""
    d = _job_dir(job_id)
    st = status(job_id)
    if st["state"] in ("starting", "queued", "running"):
        raise ValueError(f"job {job_id} is still {st['state']}")
    if st["state"] == "done":
        raise ValueError(f"job {job_id} already finished")
    old = _read(d / "state.json")
    log = d / "log.txt"
    if log.exists():                                        # keep the previous attempt's log
        log.replace(d / f"log.before-resume-{datetime.now().strftime('%H%M%S')}.txt")
    _write(d / "state.json", {"state": "starting", "resumed": (old.get("resumed") or 0) + 1})   # a clean slate
    pid = _launch(job_id)
    _set(d, worker_pid=pid, worker_born=_born(pid))
    return status(job_id)


def cancel(job_id: str) -> dict:
    d = _job_dir(job_id)
    st = _set(d, cancel=True)                              # merged: the worker's own writes keep it
    if _alive(st.get("pid"), st.get("pid_born")):
        subprocess.run(["taskkill", "/PID", str(st["pid"]), "/T", "/F"], capture_output=True, stdin=subprocess.DEVNULL)
    elif st.get("state") in ("starting", "queued", "running") and not _alive(st.get("worker_pid"), st.get("worker_born")):
        # nothing is running any more (a reboot, a crash): the job would otherwise stay "crashed" with a pending cancel
        _set(d, state="cancelled", finished=datetime.now().isoformat(timespec="seconds"))
    return status(job_id)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "run":
        raise SystemExit(run(sys.argv[2]))
    print(json.dumps(list_jobs(), indent=1, ensure_ascii=False))
