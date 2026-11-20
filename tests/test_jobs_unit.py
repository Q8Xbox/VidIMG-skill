"""Background jobs and shared state files: no update is lost between processes, a job is judged alive by more than
its process id, cancelling works at any moment, and the plan keeps decisions made while it runs.
Real (tiny) Python processes are started; no engines, GPU or FFmpeg."""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from vidimg import auto, jobs, statefile


@pytest.fixture()
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "ROOT", tmp_path / "jobs")
    monkeypatch.setattr(jobs, "LOCK", tmp_path / "jobs" / "gpu.lock")
    (tmp_path / "jobs").mkdir()
    return tmp_path


def _job(root: Path, job_id: str, code: str, **state) -> Path:
    d = root / "jobs" / job_id
    d.mkdir()
    statefile.write_json(d / "meta.json", {"id": job_id, "kind": "restore-video", "label": "t", "out": str(root / "o"),
                                           "created": "now", "created_ns": time.time_ns(),
                                           "argv": [sys.executable, "-c", code]})
    statefile.write_json(d / "state.json", {"state": "starting", **state})
    return d


def test_updates_from_many_writers_are_all_kept(tmp_path):
    path = tmp_path / "state.json"
    statefile.write_json(path, {"n": 0})

    def bump():
        for _ in range(25):
            statefile.update(path, lambda d: d.update(n=d["n"] + 1))

    code = ("import sys; sys.path.insert(0, sys.argv[1])\n"
            "from pathlib import Path\nfrom vidimg import statefile\n"
            "for _ in range(25): statefile.update(Path(sys.argv[2]), lambda d: d.update(n=d['n'] + 1))\n")
    others = [subprocess.Popen([sys.executable, "-c", code, str(Path(__file__).parent.parent), str(path)])
              for _ in range(2)]
    threads = [threading.Thread(target=bump) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(p.wait(timeout=120) == 0 for p in others)
    assert statefile.read_json(path)["n"] == 100            # 2 threads + 2 processes, 25 each: none lost
    assert not list(tmp_path.glob("*.tmp"))


def test_a_reused_process_id_is_not_a_live_job(root):
    me, born = os.getpid(), jobs._born(os.getpid())
    assert born and jobs._alive(me, born) and jobs._alive(me)         # with and without a recorded creation time
    assert not jobs._alive(me, born + 1)                              # same id, another process (after a reboot)
    _job(root, "20260101000000-aaaaaa", "pass", state="running", worker_pid=me, worker_born=born + 1, pid=me, pid_born=born + 1)
    assert jobs.status("20260101000000-aaaaaa")["state"] == "crashed"
    _job(root, "20260101000001-aaaaaa", "pass", state="queued", worker_pid=me, worker_born=born + 1)
    mine = _job(root, "20260101000002-aaaaaa", "pass")
    assert not jobs._older_waiting(mine)                              # the stale one does not hold the queue
    assert jobs.cancel("20260101000001-aaaaaa")["state"] == "cancelled"


def test_cancel_is_kept_whatever_the_worker_writes(root):
    d = _job(root, "20260101000000-bbbbbb", "pass")
    jobs._set(d, cancel=True)
    jobs._set(d, state="queued", worker_pid=1)                        # the worker's own write, a moment later
    assert jobs._read(d / "state.json")["cancel"] is True


def test_a_job_cancelled_before_its_start_never_runs(root):
    marker = root / "ran.txt"
    d = _job(root, "20260101000000-cccccc", f"open(r'{marker}', 'w').close()", cancel=True)
    assert jobs.run(d.name) == 1
    assert jobs._read(d / "state.json")["state"] == "cancelled" and not marker.exists()


def test_a_job_runs_and_can_be_cancelled_while_running(root):
    done = _job(root, "20260101000000-dddddd", "print('hello')")
    assert jobs.run(done.name) == 0
    st = jobs.status(done.name)
    assert st["state"] == "done" and st["log_tail"] == ["hello"]
    long = _job(root, "20260101000001-dddddd", "import time; print('up', flush=True); time.sleep(120)")
    t = threading.Thread(target=jobs.run, args=(long.name,))
    t0 = time.monotonic()
    t.start()
    while jobs._read(long / "state.json").get("state") != "running" and time.monotonic() - t0 < 30:
        time.sleep(0.1)
    jobs.cancel(long.name)
    t.join(timeout=30)
    assert not t.is_alive() and jobs.status(long.name)["state"] == "cancelled"


def _plan(out: Path) -> None:
    video = {"content": "auto", "model": "realesrnet", "model_strict": False, "scale": None, "texture": None,
             "mask_tape_noise": False, "preview_only": False, "skip": False}
    items = [{"id": f"v{k}", "type": "video", "source": f"C:/in/clip{k}.mp4", "decision": dict(video),
              "decided_by": "analysis", "reasons": [], "state": "pending"} for k in (1, 2, 3)]
    out.mkdir(parents=True, exist_ok=True)
    auto.save({"version": 1, "created": "now", "out": str(out), "inputs": [], "ignored_files": 0, "contact_sheets": [],
               "items": items})


def test_decisions_made_during_a_run_are_kept(tmp_path, monkeypatch):
    _plan(tmp_path)
    ran = []

    def video(it, out, log):
        ran.append((it["id"], it["decision"]["content"], it["decision"]["skip"]))
        if it["id"] == "v1":                                # while v1 runs, someone corrects the plan
            auto.decide(tmp_path, {"v1": {"content": "anime", "reason": "it is a cartoon"},
                                   "v2": {"content": "real"}, "v3": {"skip": True}})
        it.update(state="delivered", delivered="x_restored.mp4", result={"chosen": "m", "notes": []})

    monkeypatch.setattr(auto, "_run_video", video)
    plan = auto.run(tmp_path, log=lambda m: None)
    items = {i["id"]: i for i in plan["items"]}
    assert ran == [("v1", "auto", False), ("v2", "real", False)]          # v2 with the new decision, v3 skipped
    assert items["v3"]["state"] == "skipped" and items["v2"]["state"] == "delivered"
    assert items["v1"]["decision"]["content"] == "anime" and items["v1"]["state"] == "pending"   # to run again
    assert "it is a cartoon" in items["v1"]["reasons"] and "redo" not in items["v1"]
    on_disk = {i["id"]: i for i in json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))["items"]}
    assert on_disk["v1"]["decision"]["content"] == "anime" and on_disk["v2"]["decision"]["content"] == "real"
    plan = auto.run(tmp_path, log=lambda m: None)                         # the next run does v1 with the new one
    assert ran[-1] == ("v1", "anime", False) and all(i["state"] in auto.DONE for i in plan["items"])


def test_default_output_folders_do_not_collide(tmp_path, monkeypatch):
    from vidimg import mcp_server

    monkeypatch.setattr(mcp_server.config, "OUTPUT_ROOT", tmp_path)
    src = tmp_path / "in" / "a.jpg"
    made = {mcp_server._out_dir(None, [src], "review") for _ in range(20)}
    assert len(made) == 20 and all(p.parent == tmp_path / "review" for p in made)
