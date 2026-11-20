"""Delivery safety: cases where a wrong picture could be delivered, the guard passed without looking, or ordinary
input stopped a run. Engines, GPU and video files are replaced by stubs."""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from vidimg import batch, guard, pipeline_video as pv
from vidimg.adapters import AdapterResult, safe_output_dir
from vidimg.guard import video as gv
from vidimg.probe.video import VideoProfile


def _photo(path: Path, value: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((8, 8, 3), value, np.uint8))
    return path


def test_same_name_photos_keep_their_own_model_output(tmp_path, monkeypatch):
    out = tmp_path / "out"
    a, b = _photo(tmp_path / "A" / "IMG_1.png", 40), _photo(tmp_path / "B" / "IMG_1.png", 200)

    def fake_upscale(inputs, folder, *, model, scale, suffix):
        for i in inputs:                                    # the worker names outputs by input stem
            Path(folder).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path(folder) / f"{Path(i).stem}{suffix}.png"), cv2.imread(str(i)))

    judged = {}
    monkeypatch.setattr(batch.free_sr, "upscale_images", fake_upscale)
    monkeypatch.setattr(batch, "_judge", lambda it, label, params, output, *r: judged.update({it.src: output}) or True)
    for src in (a, b):                                      # same stem: one call each, same round and settings
        it = batch.Item(src, out)
        it.state.update(status="planned", stage_in=str(src), candidates=[["bsrnet", {"model": "m", "scale": 2}]])
        batch._run_group([it], "bsrnet", {"model": "m", "scale": 2}, 0, out, lambda *_: None)
    assert judged[a] != judged[b]
    assert int(cv2.imread(str(judged[a]))[0, 0, 0]) == 40 and int(cv2.imread(str(judged[b]))[0, 0, 0]) == 200


def test_one_unreadable_photo_does_not_stop_the_batch(tmp_path, monkeypatch):
    good, bad = _photo(tmp_path / "in" / "good.png", 90), tmp_path / "in" / "bad.jpg"
    bad.write_bytes(b"not a picture")
    calls = []

    def prepare(it, policy, log):
        calls.append(it.src.name)
        if it.src == bad:
            raise ValueError("cannot read")
        it.state.update(status="skipped", profile={})

    monkeypatch.setattr(batch, "_prepare", prepare)
    recs = batch.restore_images([bad, good], tmp_path / "out", log=lambda *_: None)
    assert len(recs) == 2 and not recs[0]["delivered"] and "cannot read" in recs[0]["notes"][0]
    batch.restore_images([bad, good], tmp_path / "out", log=lambda *_: None)
    assert calls == ["bad.jpg", "good.png", "bad.jpg"]      # the broken one is tried again, the finished one is not


def _stub_video(monkeypatch, frames_for):
    info = {"frames": 600, "duration": 24.0, "audio_streams": 1, "fps": 25.0, "width": 96, "height": 72, "sar": "1:1"}
    monkeypatch.setattr(gv.media, "probe", lambda p, exact=False: dict(info))
    monkeypatch.setattr(gv.media, "av_offset", lambda i: None)
    monkeypatch.setattr(gv.media, "display_aspect", lambda i: None)
    monkeypatch.setattr(gv.media, "ffprobe", lambda: "ffprobe")
    monkeypatch.setattr(gv.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="", stderr="", returncode=0))
    monkeypatch.setattr(gv, "_vmaf", lambda *a, **k: 95.0)
    monkeypatch.setattr(gv, "_frames_by_index", frames_for)


def test_video_guard_fails_when_no_frames_were_compared(tmp_path, monkeypatch):
    _stub_video(monkeypatch, lambda path, indices, folder, tag, pre_vf=None: {})
    rep = gv.verify_video(tmp_path / "a.mkv", tmp_path / "b.mkv")
    assert not rep.passed and rep.failed() == ["coverage"]


def test_video_guard_counts_the_pairs_it_compared(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    img = cv2.GaussianBlur(rng.integers(0, 255, (72, 96, 3), dtype=np.uint8), (0, 0), 2)

    def frames(path, indices, folder, tag, pre_vf=None):
        out = {}
        for i in indices:
            out[i] = Path(folder) / f"{tag}_{i}.png"
            cv2.imwrite(str(out[i]), img)
        return out

    _stub_video(monkeypatch, frames)
    rep = gv.verify_video(tmp_path / "a.mkv", tmp_path / "b.mkv")
    cov = next(g for g in rep.gates if g.name == "coverage")
    assert cov.status == "pass" and cov.value == 6


def test_black_frame_does_not_trip_the_hallucination_gate(tmp_path):
    rng = np.random.default_rng(2)
    src = np.full((90, 120, 3), 16, np.uint8)                                  # a fade to black
    out = np.clip(16 + rng.normal(0, 0.6, (180, 240, 3)), 0, 255).round().astype(np.uint8)   # faint encoder noise
    cv2.imwrite(str(tmp_path / "s.png"), src)
    cv2.imwrite(str(tmp_path / "o.png"), out)
    rep = guard.verify_image(tmp_path / "s.png", tmp_path / "o.png")
    assert next(g for g in rep.gates if g.name == "hallucination").status == "pass"
    loud = np.clip(16 + rng.normal(0, 12, (180, 240, 3)), 0, 255).round().astype(np.uint8)   # invented texture
    cv2.imwrite(str(tmp_path / "l.png"), loud)
    rep = guard.verify_image(tmp_path / "s.png", tmp_path / "l.png")
    assert next(g for g in rep.gates if g.name == "hallucination").status == "fail"


def test_model_output_never_sits_next_to_the_prepass(tmp_path, monkeypatch):
    work = tmp_path / "out" / "work" / "clip"
    work.mkdir(parents=True)
    base = work / "prepass.nut"                             # pre-pass without blend removal: directly in `work`
    base.write_bytes(b"x")
    seen = {}

    def engine(src, dst, stages, encoder, **kw):
        safe_output_dir(dst.parent, src)                    # what every engine adapter does first
        seen["dst"] = dst
        return AdapterResult(False, error="stub")

    monkeypatch.setattr(pv, "run_engine", engine)
    monkeypatch.setattr(pv, "_room_for_lossless", lambda *a: True)
    prof = VideoProfile(str(base), 720, 576, 25.0, 250, 10.0, "mpeg2video", 4000, 0.4, 0.9, "tff", None, None)
    rec = pv.VideoRecord(str(base), {}, "", chosen="realesrnet")
    rec = pv._full_run(tmp_path / "clip.mpg", tmp_path / "out", work, base, prof, rec, lambda: [], "hevc10",
                       None, lambda *_: None)
    assert seen["dst"].parent != base.parent and rec.notes == ["full run failed: stub"]
