"""Checkpoints are reused only when the same thing made them; a program that goes silent is stopped; a rejected
result does not stay under the deliverable's name. No engines, GPU or FFmpeg."""

import json
import sys
import time
from pathlib import Path

from vidimg import pipeline_video as pv, proc
from vidimg.adapters import AdapterResult, free_sr
from vidimg.guard import Gate, GuardReport
from vidimg.probe.video import VideoProfile


class _Stage:
    def __init__(self, model: str):
        self.model = model

    def describe(self) -> str:
        return f"free:{self.model}:x2"


def _chunk_env(monkeypatch, tmp_path, rendered: list):
    base = tmp_path / "in" / "base.mkv"
    base.parent.mkdir(parents=True, exist_ok=True)
    base.write_bytes(b"x" * 10)
    info = {"fps": 25.0, "duration": 300.0, "frames": 7500, "audio_streams": 0}
    monkeypatch.setattr(pv.media, "probe", lambda p, exact=False: dict(info))
    monkeypatch.setattr(pv.media, "fix_av_offset", lambda a, b: None)
    monkeypatch.setattr(pv.media, "fix_aspect", lambda a, b: None)
    monkeypatch.setattr(pv.media, "colour_hint", lambda p: {})
    monkeypatch.setattr(pv, "scene_cuts", lambda b, w, log=print: [])
    monkeypatch.setattr(pv, "_ffmpeg", lambda: "ffmpeg")

    def engine(src, dst, stages, encoder, **kw):
        rendered.append((dst.name, stages[0].model))
        dst.write_bytes(stages[0].model.encode())
        return AdapterResult(True, [dst])

    def join(args, **kw):                                   # the concat step: write the output file
        Path(args[-1]).write_bytes(b"joined")
        return type("R", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(pv, "run_engine", engine)
    monkeypatch.setattr(pv.subprocess, "run", join)
    return base


def test_chunks_are_not_reused_across_models_or_encoders(tmp_path, monkeypatch):
    rendered: list = []
    base = _chunk_env(monkeypatch, tmp_path, rendered)
    work, dst = tmp_path / "work", tmp_path / "out" / "o.mkv"
    dst.parent.mkdir()

    def run(model, encoder="hevc10"):
        rendered.clear()
        assert pv.run_chunked(base, dst, lambda: [_Stage(model)], encoder, work, log=lambda *_: None).ok
        return len(rendered)

    n = run("realesrnet")
    assert n >= 2
    assert run("realesrnet") == 0                           # same settings: every chunk reused
    assert run("bsrnet") == n                               # another model: none of the old chunks
    assert {m for _, m in rendered} == {"bsrnet"}
    assert (work / "chunks" / "c0000.mp4").read_bytes() == b"bsrnet"
    assert run("bsrnet", "h264") == n                       # another encoder too
    (work / "chunks" / "fingerprint.json").unlink()         # chunks from before fingerprints: trusted as they are
    assert run("realesrnet") == 0


def test_two_files_with_the_same_stem_get_their_own_work_folder(tmp_path):
    out = tmp_path / "out"
    avi, mkv = tmp_path / "in" / "ep01.avi", tmp_path / "in" / "ep01.mkv"
    avi.parent.mkdir()
    avi.write_bytes(b"a")
    mkv.write_bytes(b"b")
    assert pv.work_key(out, avi) == "ep01"                  # nobody owns it yet
    owner = out / "work" / "ep01"
    owner.mkdir(parents=True)
    (owner / "profile.json").write_text(json.dumps({"source": pv.source_id(avi), "profile": {}}), encoding="utf-8")
    assert pv.work_key(out, avi) == "ep01"                  # its own folder, also on resume
    other = pv.work_key(out, mkv)
    assert other.startswith("ep01-") and other == pv.work_key(out, mkv)


def test_prepass_from_another_plan_is_not_reused(tmp_path, monkeypatch):
    work = tmp_path / "work"
    (work / "chunks").mkdir(parents=True)
    (work / "chunks" / "c0000.mp4").write_bytes(b"old")
    (work / "profile.json").write_text("{}", encoding="utf-8")
    pre, done = work / "prepass.nut", work / "prepass.done"
    pre.write_bytes(b"x")
    pv.media.colour_sidecar(pre).write_text("{}", encoding="utf-8")
    done.write_text(json.dumps({"engine": "avisynth", "fingerprint": "aaaa"}), encoding="utf-8")
    prof = VideoProfile("s.mpg", 720, 576, 25.0, 250, 10.0, "mpeg2video", 4000, 0.4, 0.9, "tff", None, None)
    stage = pv.avs.MaskBottom(rows=8)
    base, info = pv._run_field_prepass(tmp_path / "s.mpg", work, prof, [stage], "mask", lambda *_: None, "aaaa")
    assert base == pre and info["reused"]                   # same plan: reused
    monkeypatch.setattr(pv.config, "classical_engines", lambda: [])
    monkeypatch.setattr(pv, "_ffmpeg", lambda: "ffmpeg")              # the fallback's command is stubbed below
    monkeypatch.setattr(pv.avs, "_rate", lambda p: (None, 0, {}))
    monkeypatch.setattr(pv.subprocess, "run", lambda *a, **k: type("R", (), {"returncode": 1, "stderr": "no"})())
    base, info = pv._run_field_prepass(tmp_path / "s.mpg", work, prof, [stage], "mask", lambda *_: None, "bbbb")
    assert base is None and not info.get("reused")          # another plan: done again (here: no engine, so it fails)
    assert not (work / "chunks").exists() and (work / "profile.json").exists()   # what was built on it is gone
    done.write_text(json.dumps({"engine": "avisynth"}), encoding="utf-8")        # from before fingerprints
    pre.write_bytes(b"x")
    pv.media.colour_sidecar(pre).write_text("{}", encoding="utf-8")
    assert pv._run_field_prepass(tmp_path / "s.mpg", work, prof, [stage], "mask", lambda *_: None, "bbbb")[1]["reused"]


def test_rejected_output_leaves_the_results_folder(tmp_path, monkeypatch):
    out, work = tmp_path / "out", tmp_path / "out" / "work" / "clip"
    work.mkdir(parents=True)
    base = tmp_path / "in" / "clip.mp4"
    base.parent.mkdir()
    base.write_bytes(b"x")

    def engine(src, dst, stages, encoder, **kw):
        dst.write_bytes(b"video")
        return AdapterResult(True, [dst])

    monkeypatch.setattr(pv, "run_engine", engine)
    monkeypatch.setattr(pv, "output_suffix", lambda src, enc=None: ".mp4")
    monkeypatch.setattr(pv.config, "TEXTURE_STRENGTH", 0.0)
    monkeypatch.setattr(pv, "carry_side_streams", lambda s, d: None)
    monkeypatch.setattr(pv.media, "fix_aspect", lambda s, d: None)
    monkeypatch.setattr(pv, "review_frames", lambda b, d, folder: [])
    monkeypatch.setattr(pv, "verify_video", lambda *a, **k: GuardReport(False, [Gate("vmaf", "fail", 10.0, 60.0)]))
    prof = VideoProfile(str(base), 720, 576, 25.0, 250, 10.0, "h264", 4000, 0.4, 0.0, "progressive", None, None)
    rec = pv._full_run(base, out, work, base, prof, pv.VideoRecord(str(base), {}, "", chosen="m"), lambda: [],
                       "hevc10", None, lambda *_: None)
    assert not rec.passed and rec.delivered is None
    assert not (out / "clip_restored.mp4").exists()
    assert (work / "rejected" / "clip_restored.mp4").read_bytes() == b"video"
    assert "kept for inspection" in rec.notes[-1]


def test_a_silent_program_is_stopped(monkeypatch, tmp_path):
    monkeypatch.setattr(proc, "_log", lambda result, env: tmp_path / "log.json")
    t = time.monotonic()
    r = proc.run("sleeper", [sys.executable, "-c", "import time; print('start', flush=True); time.sleep(60)"],
                 stall_timeout=1.0)
    assert time.monotonic() - t < 30 and r.returncode != 0
    assert "stalled" in r.stdout_tail[-1] and r.stdout_tail[0] == "start"
    r = proc.run("talker", [sys.executable, "-c",
                            "import time\nfor i in range(6):\n print(i, flush=True); time.sleep(0.4)"], stall_timeout=1.5)
    assert r.returncode == 0 and r.stdout_tail[-1] == "5"   # slow but talking: left alone


def test_free_engine_has_a_stall_limit():
    assert 600 <= free_sr.STALL_SECONDS <= 3600
