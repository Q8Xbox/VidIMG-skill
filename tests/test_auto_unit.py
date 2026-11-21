"""The automatic flow (survey -> decide -> run -> one report): finding the work, corrections, resumable running
and the combined report. No analysis, GPU or video files: the pipelines underneath are replaced by stubs."""

import json
from pathlib import Path

import pytest

from vidimg import auto, jobs


def _touch(root: Path, *names: str) -> None:
    for n in names:
        p = root / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")


def test_scan_sorts_photos_videos_and_series(tmp_path):
    _touch(tmp_path, "a.jpg", "sub/b.PNG", "clip.mp4", "notes.txt",
           "Show/Show - 01.mkv", "Show/Show - 02.mkv", "Show/Show - 03.mkv", "Show/extras.mkv",
           "Two/Ep 1.mp4", "Two/Ep 2.mp4",                                   # two episodes: not a series
           "old_restored.mp4", "Show/VidIMG x2 - Episodes 01-03/Show - 01_restored.mkv",
           "run/work/clip/profile.json", "run/work/clip/chunks/c0.mkv",      # a video run's work folder
           "new/.vidimg-output", "new/survey/contact_1.jpg", "new/previews/p.mp4",
           "Work/holiday.mp4", "Previews/sheets/scan.jpg")                   # the person's own folders
    found = auto.scan([tmp_path])
    assert sorted(p.name for p in found["photos"]) == ["a.jpg", "b.PNG", "scan.jpg"]
    assert sorted(p.name for p in found["videos"]) == ["Ep 1.mp4", "Ep 2.mp4", "clip.mp4", "extras.mkv", "holiday.mp4"]
    assert len(found["series"]) == 1 and [n for n, _ in found["series"][0]["episodes"]] == [1, 2, 3]
    assert found["ignored"] == 8                 # notes.txt, a marker, a checkpoint and VidIMG's own outputs
    with pytest.raises(FileNotFoundError):
        auto.scan([tmp_path / "missing"])


def _plan(out: Path) -> dict:
    guide = {"suggested": "realesrnet", "why": ["probe says real"], "suggested_if_anime": "esrgan-v3",
             "suggested_if_real": "realesrnet"}
    video = {"content": "auto", "model": "realesrnet", "model_strict": False, "scale": None, "texture": None,
             "mask_tape_noise": False, "preview_only": False, "skip": False}
    plan = {"version": 1, "created": "now", "out": str(out), "inputs": [], "ignored_files": 0, "contact_sheets": [],
            "items": [
                {"id": "photos", "type": "photos", "sources": ["p1.jpg", "p2.jpg"], "count": 2,
                 "counts": {"noisy": 0, "soft_or_blurred": 1, "anime_like": 0, "faces": 2, "large": 0},
                 "decision": {"deblur": False, "skip": False}, "decided_by": "analysis", "reasons": [], "state": "pending"},
                {"id": "v1", "type": "video", "source": "C:/in/clip.mp4", "guide": guide, "decision": dict(video),
                 "decided_by": "analysis", "reasons": ["probe says real"], "needs_look": "confirm", "state": "pending"},
                {"id": "s1", "type": "series", "source": "C:/in/Show", "folder": "C:/in/Show", "episodes": [1, 2, 3],
                 "guide": guide, "decision": {k: v for k, v in video.items() if k != "preview_only"} | {"scale": 2},
                 "decided_by": "analysis", "reasons": [], "needs_look": "confirm", "state": "pending"}]}
    out.mkdir(parents=True, exist_ok=True)
    auto.save(plan)
    return plan


def test_decide_corrects_the_plan_and_rejects_bad_input(tmp_path):
    _plan(tmp_path)
    plan = auto.decide(tmp_path, {"v1": {"content": "anime", "reason": "it is a cartoon"},
                                  "s1": {"content": "anime", "model": "cugan-pro", "texture": 0.5},
                                  "photos": {"deblur": True}})
    v1, s1, ph = (next(i for i in plan["items"] if i["id"] == k) for k in ("v1", "s1", "photos"))
    assert v1["decision"]["content"] == "anime" and v1["decision"]["model"] == "esrgan-v3"   # re-picked from the guide
    assert v1["decided_by"] == "claude" and "needs_look" not in v1 and "it is a cartoon" in v1["reasons"]
    assert s1["decision"]["model"] == "cugan-pro" and s1["decision"]["texture"] == 0.5
    assert ph["decision"]["deblur"] is True
    for bad in ({"nope": {"content": "anime"}}, {"photos": {"content": "anime"}}, {"v1": {"content": "cartoon"}},
                {"v1": {"model": "gfpgan"}}, {"v1": {"texture": 2}}, {"s1": {"preview_only": True}},
                {"v1": {"scale": 9}}):
        with pytest.raises(ValueError):
            auto.decide(tmp_path, bad)
    assert auto.summary(auto.load(tmp_path))["next"]                         # still pending: tells Claude what to do


def test_run_executes_in_order_resumes_and_survives_a_bad_item(tmp_path, monkeypatch):
    _plan(tmp_path)
    auto.decide(tmp_path, {"s1": {"skip": True}, "v1": {"content": "anime", "texture": 0.4}})
    calls, seen_texture = [], []

    def photos(it, out, log):
        calls.append("photos")
        raise RuntimeError("disk full")

    def video(it, out, log):
        calls.append("v1")
        seen_texture.append(auto.config.TEXTURE_STRENGTH)
        it.update(state="delivered", delivered="x_restored.mp4", result={"chosen": "anime esrgan-v3", "notes": ["ok"]})

    monkeypatch.setattr(auto, "_run_photos", photos)
    monkeypatch.setattr(auto, "_run_video", video)
    before = auto.config.TEXTURE_STRENGTH
    plan = auto.run(tmp_path, log=lambda m: None)
    states = {i["id"]: i["state"] for i in plan["items"]}
    assert states == {"photos": "error", "v1": "delivered", "s1": "skipped"} and calls == ["photos", "v1"]
    assert seen_texture == [0.4] and auto.config.TEXTURE_STRENGTH == before  # per-item setting, then restored
    page = Path(plan["report"]).read_text(encoding="utf-8")
    assert "disk full" in page and "Delivered" in page and "anime esrgan-v3" in page
    # second run: only the failed item is tried again
    monkeypatch.setattr(auto, "_run_photos", lambda it, out, log: (calls.append("photos again"),
                                                                   it.update(state="delivered", result={"delivered": 2, "total": 2})))
    plan = auto.run(tmp_path, log=lambda m: None)
    assert calls == ["photos", "v1", "photos again"] and all(i["state"] in auto.DONE for i in plan["items"])
    assert "review_results" in auto.summary(plan)["next"]              # delivered: now look at the result


def test_jobs_know_the_new_kinds_and_report_the_plan(tmp_path):
    assert jobs.KINDS["survey"] == "auto" and jobs.KINDS["run-plan"] == "run-plan"
    _plan(tmp_path)
    res = jobs._result({"kind": "run-plan", "out": str(tmp_path)})
    assert [i["id"] for i in res["items"]] == ["photos", "v1", "s1"] and res["plan"].endswith("plan.json")
    assert jobs._result({"kind": "survey", "out": str(tmp_path / "none")}) == {}


def test_cli_run_plan_needs_a_plan(tmp_path, capsys):
    from vidimg import cli
    assert cli.main(["run-plan", "--out", str(tmp_path)]) == 2
    assert "run the survey first" in capsys.readouterr().out
